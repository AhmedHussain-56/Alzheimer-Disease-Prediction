from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from app.models import db, User, UserPrediction, TestLog, TrainedModel
from app.routes.auth import login_required
import os
import cv2
import numpy as np
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

user_bp = Blueprint('user', __name__, url_prefix='/user')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

def allowed_file(filename):
    """Check if file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@user_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    predictions_count = UserPrediction.query.filter_by(user_id=user_id).count()
    recent_predictions = UserPrediction.query.filter_by(user_id=user_id).order_by(
        UserPrediction.created_at.desc()
    ).limit(5).all()
    
    # Get the active model for dashboard display
    active_model = TrainedModel.query.filter_by(is_active=True).first()
    if not active_model:
        # Fallback: get the most recent model
        active_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
    
    return render_template('user/dashboard.html', 
                         user=user,
                         predictions_count=predictions_count,
                         recent_predictions=recent_predictions,
                         active_model=active_model)

@user_bp.route('/profile')
@login_required
def profile():
    """View user profile"""
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    return render_template('user/profile.html', user=user)

@user_bp.route('/edit-profile', methods=['POST'])
@login_required
def edit_profile():
    """Edit user profile"""
    user_id = session.get('user_id')
    data = request.form
    
    try:
        user = User.query.get(user_id)
        user.first_name = data.get('first_name', user.first_name)
        user.last_name = data.get('last_name', user.last_name)
        user.phone = data.get('phone', user.phone)
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating profile: {str(e)}', 'danger')
    
    return redirect(url_for('user.profile'))

@user_bp.route('/test-data', methods=['GET', 'POST'])
@login_required
def test_data():
    """Test data upload and prediction"""
    if request.method == 'POST':
        user_id = session.get('user_id')
        
        if 'file' not in request.files:
            logging.warning('No file provided in request')
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            logging.warning('No file selected')
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logging.warning(f'File type not allowed: {file.filename}')
            return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400
        
        try:
            logging.debug('Starting file upload process')
            start_time = time.time()
            
            # Read image
            file_data = file.read()
            logging.debug('File read successfully')
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            logging.debug('Image decoded successfully')

            if img is None:
                logging.error('Invalid image file')
                return jsonify({'status': 'error', 'message': 'Invalid image file'}), 400

            # Preprocess
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            logging.debug('Image preprocessing completed')

            # Load latest model
            from app.models import TrainedModel
            latest_model = TrainedModel.query.filter_by(is_active=True).order_by(
                TrainedModel.created_at.desc()
            ).first()
            logging.debug('Latest model loaded')

            if not latest_model:
                logging.error('No trained model available')
                return jsonify({'status': 'error', 'message': 'No trained model available'}), 404

            # Make prediction
            # Parse filename supporting both Windows (\) and Linux (/) separators dynamically
            model_filename = latest_model.model_path.replace('\\', '/').split('/')[-1]
            # Use TFLite version of the model to save memory
            if model_filename.endswith('.h5'):
                model_filename = model_filename[:-3] + '.tflite'
            
            resolved_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'saved_models', model_filename)
            logging.debug(f'Loading TFLite model from: {resolved_model_path}')
            
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                from tensorflow import lite as tflite

            interpreter = tflite.Interpreter(model_path=resolved_model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Make sure img is float32
            img_input = img.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            logging.debug('TFLite prediction completed')
            pred_label = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            predicted_class = classes[pred_label]
            
            processing_time = time.time() - start_time
            
            # Save prediction
            prediction = UserPrediction(
                user_id=user_id,
                image_path=os.path.join('uploads', secure_filename(file.filename)),
                image_filename=secure_filename(file.filename),
                predicted_class=predicted_class,
                confidence=confidence,
                model_used=latest_model.model_name,
                processing_time=processing_time
            )
            prediction.set_all_predictions({
                'NonDemented': float(predictions[0]),
                'VeryMildDemented': float(predictions[1]),
                'MildDemented': float(predictions[2]),
                'ModerateDemented': float(predictions[3])
            })
            
            # Save test log
            test_log = TestLog(
                user_id=user_id,
                image_filename=secure_filename(file.filename),
                result_class=predicted_class,
                confidence=confidence,
                processing_time=processing_time,
                status='success'
            )
            
            db.session.add(prediction)
            db.session.add(test_log)
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'predicted_class': predicted_class,
                'confidence': f'{confidence:.4f}',
                'all_predictions': {
                    'NonDemented': f'{predictions[0]:.4f}',
                    'VeryMildDemented': f'{predictions[1]:.4f}',
                    'MildDemented': f'{predictions[2]:.4f}',
                    'ModerateDemented': f'{predictions[3]:.4f}'
                },
                'processing_time': f'{processing_time:.2f}s',
                'model_used': latest_model.model_name
            })
        
        except Exception as e:
            logging.error(f'Error during file upload process: {str(e)}')
            db.session.rollback()
            test_log = TestLog(
                user_id=user_id,
                image_filename=secure_filename(file.filename),
                result_class='Error',  # Fallback to satisfy NOT NULL db constraint
                confidence=0.0,
                processing_time=0.0,
                status='error',
                error_message=str(e)
            )
            db.session.add(test_log)
            db.session.commit()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return render_template('user/test_data.html')

@user_bp.route('/results')
@login_required
def results():
    """View test results"""
    user_id = session.get('user_id')
    predictions = UserPrediction.query.filter_by(user_id=user_id).order_by(
        UserPrediction.created_at.desc()
    ).all()
    
    return render_template('user/results.html', predictions=predictions)

@user_bp.route('/results-charts')
@login_required
def results_charts():
    """View results with charts"""
    user_id = session.get('user_id')
    predictions = UserPrediction.query.filter_by(user_id=user_id).all()
    
    # Prepare data for charts
    class_counts = {}
    confidence_scores = []
    
    for pred in predictions:
        class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1
        confidence_scores.append(pred.confidence)
    
    return render_template('user/results_charts.html', 
                         predictions=predictions,
                         class_counts=class_counts,
                         confidence_scores=confidence_scores)

@user_bp.route('/test-logs')
@login_required
def test_logs():
    """View test logs"""
    user_id = session.get('user_id')
    logs = TestLog.query.filter_by(user_id=user_id).order_by(
        TestLog.created_at.desc()
    ).all()
    
    return render_template('user/test_logs.html', logs=logs)

@user_bp.route('/api/chart-data')
@login_required
def api_chart_data():
    """API endpoint for chart data"""
    user_id = session.get('user_id')
    predictions = UserPrediction.query.filter_by(user_id=user_id).all()
    
    class_counts = {}
    for pred in predictions:
        class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1
    
    return jsonify({
        'classes': list(class_counts.keys()),
        'counts': list(class_counts.values())
    })

@user_bp.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))
