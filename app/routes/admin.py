from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from app.models import db, TrainedModel, UserPrediction, TestLog, AdminPrediction, TrainingLog, PerformanceReport
from app.routes.auth import admin_login_required
from models.deep_learning import DataLoader, ResNet101Model, UNetModel, SimpleDLModel
import os
import json
import io
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
@admin_login_required
def dashboard():
    """Admin dashboard"""
    admin_id = session.get('admin_id')
    models_count = TrainedModel.query.filter_by(admin_id=admin_id).count()
    total_predictions = AdminPrediction.query.join(TrainedModel).filter(
        TrainedModel.admin_id == admin_id
    ).count()
    
    # Get best accuracy from all trained models
    from sqlalchemy import func
    best_model = TrainedModel.query.filter_by(admin_id=admin_id).order_by(
        TrainedModel.accuracy.desc()
    ).first()
    
    # Get the active model
    active_model = TrainedModel.query.filter_by(admin_id=admin_id, is_active=True).first()
    if not active_model and best_model:
        active_model = best_model
    
    return render_template('admin/dashboard.html', 
                         models_count=models_count,
                         total_predictions=total_predictions,
                         best_model=best_model,
                         active_model=active_model)

@admin_bp.route('/dataset-details')
@admin_login_required
def dataset_details():
    """View dataset details"""
    dataset_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset')
    
    dataset_info = {
        'train': {},
        'test': {}
    }
    
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_folder, split)
        if os.path.exists(split_path):
            classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            for class_name in classes:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    image_count = len([f for f in os.listdir(class_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    dataset_info[split][class_name] = image_count
    
    return render_template('admin/dataset_details.html', dataset_info=dataset_info)

@admin_bp.route('/train-model', methods=['GET', 'POST'])
@admin_login_required
def train_model():
    """Train a new model"""
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        algorithm = data.get('algorithm')
        epochs = int(data.get('epochs', 20))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        
        try:
            admin_id = session.get('admin_id')
            dataset_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset')
            
            # Load data
            data_loader = DataLoader(dataset_folder)
            X_train, y_train, X_test, y_test = data_loader.load_train_test_data()
            
            # Create and train model
            model_name = f"{algorithm}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saved_models')
            os.makedirs(saved_models_dir, exist_ok=True)
            model_path = os.path.join(saved_models_dir, f"{model_name}.h5")
            
            if algorithm == 'resnet101':
                model = ResNet101Model()
            elif algorithm == 'unet':
                model = UNetModel()
            else:
                model = SimpleDLModel()
            
            model.build_model()
            history = model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
            model.save_model(model_path)
            
            # Evaluate
            eval_results = model.evaluate(X_test, y_test)
            
            # Save to database
            trained_model = TrainedModel(
                admin_id=admin_id,
                model_name=model_name,
                algorithm=algorithm,
                model_path=model_path,
                accuracy=eval_results['accuracy'],
                precision=eval_results['classification_report']['weighted avg']['precision'],
                recall=eval_results['classification_report']['weighted avg']['recall'],
                f1_score=eval_results['classification_report']['weighted avg']['f1-score'],
                training_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                is_active=True
            )
            trained_model.set_confusion_matrix({
                'matrix': eval_results['confusion_matrix'],
                'classes': ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
            })
            trained_model.set_training_history({
                'loss': history.history.get('loss', []),
                'accuracy': history.history.get('accuracy', []),
                'val_loss': history.history.get('val_loss', []),
                'val_accuracy': history.history.get('val_accuracy', [])
            })
            
            db.session.add(trained_model)
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'message': f'Model trained successfully with {eval_results["accuracy"]:.4f} accuracy',
                'model_id': trained_model.id
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return render_template('admin/train_model.html')

@admin_bp.route('/models')
@admin_login_required
def models_list():
    """List all trained models"""
    admin_id = session.get('admin_id')
    models = TrainedModel.query.filter_by(admin_id=admin_id).all()
    return render_template('admin/models_list.html', models=models)

@admin_bp.route('/model/<int:model_id>')
@admin_login_required
def model_details(model_id):
    """View model details"""
    model = TrainedModel.query.get_or_404(model_id)
    
    if model.admin_id != session.get('admin_id'):
        flash('Unauthorized', 'danger')
        return redirect(url_for('admin.models_list'))
    
    return render_template('admin/model_details.html', model=model)

@admin_bp.route('/toggle-model/<int:model_id>', methods=['POST'])
@admin_login_required
def toggle_model(model_id):
    """Activate or deactivate a model. Only one model can be active at a time."""
    model = TrainedModel.query.get_or_404(model_id)
    
    if model.admin_id != session.get('admin_id'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    try:
        if model.is_active:
            # Deactivate this model
            model.is_active = False
            db.session.commit()
            return jsonify({
                'status': 'success',
                'message': f'Model "{model.model_name}" has been deactivated.',
                'is_active': False
            })
        else:
            # Deactivate all other models first, then activate this one
            TrainedModel.query.filter_by(admin_id=session.get('admin_id')).update({'is_active': False})
            model.is_active = True
            db.session.commit()
            return jsonify({
                'status': 'success',
                'message': f'Model "{model.model_name}" is now the active model for predictions.',
                'is_active': True
            })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@admin_bp.route('/re-evaluate/<int:model_id>', methods=['POST'])
@admin_login_required
def re_evaluate_model(model_id):
    """Re-evaluate a saved model on the test dataset without retraining."""
    model_obj = TrainedModel.query.get_or_404(model_id)
    
    if model_obj.admin_id != session.get('admin_id'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    try:
        # Check model file exists
        model_filename = os.path.basename(model_obj.model_path)
        resolved_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'saved_models', model_filename)
        if not os.path.exists(resolved_model_path):
            return jsonify({'status': 'error', 'message': f'Model file not found: {resolved_model_path}'}), 404
        
        # Load saved model
        from tensorflow import keras
        keras_model = keras.models.load_model(resolved_model_path)
        
        # Load test data
        dataset_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset')
        data_loader = DataLoader(dataset_folder)
        _, _, X_test, y_test = data_loader.load_train_test_data()
        
        # Make predictions
        predictions = keras_model.predict(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score as sklearn_f1
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        accuracy = accuracy_score(y_test, pred_labels)
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2, 3])
        report = classification_report(
            y_test, pred_labels,
            labels=[0, 1, 2, 3],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Update model record in database
        model_obj.accuracy = float(accuracy)
        model_obj.precision = report['weighted avg']['precision']
        model_obj.recall = report['weighted avg']['recall']
        model_obj.f1_score = report['weighted avg']['f1-score']
        model_obj.set_confusion_matrix({
            'matrix': cm.tolist(),
            'classes': class_names
        })
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': f'Model re-evaluated. Accuracy: {accuracy:.4f}, F1: {report["weighted avg"]["f1-score"]:.4f}',
            'accuracy': float(accuracy),
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@admin_bp.route('/run-inference', methods=['GET', 'POST'])
@admin_login_required
def run_inference():
    """Run inference on test dataset"""
    if request.method == 'POST':
        data = request.get_json()
        model_id = data.get('model_id')
        
        try:
            model_obj = TrainedModel.query.get_or_404(model_id)
            
            if model_obj.admin_id != session.get('admin_id'):
                return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
            
            # Load model
            from tensorflow import keras
            model_filename = os.path.basename(model_obj.model_path)
            resolved_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'saved_models', model_filename)
            keras_model = keras.models.load_model(resolved_model_path)
            
            # Load test data
            dataset_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset')
            data_loader = DataLoader(dataset_folder)
            _, _, X_test, y_test = data_loader.load_train_test_data()
            
            # Make predictions
            predictions = keras_model.predict(X_test)
            pred_labels = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(pred_labels == y_test)
            cm = confusion_matrix(y_test, pred_labels)
            
            # Save predictions
            for i in range(len(pred_labels)):
                pred = AdminPrediction(
                    model_id=model_id,
                    image_path=f"test_data/image_{i}.jpg",
                    image_filename=f"test_image_{i}.jpg",
                    predicted_class=['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'][pred_labels[i]],
                    confidence=float(np.max(predictions[i])),
                    actual_class=['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'][y_test[i]],
                    is_correct=pred_labels[i] == y_test[i]
                )
                pred.set_all_predictions({
                    'NonDemented': float(predictions[i][0]),
                    'VeryMildDemented': float(predictions[i][1]),
                    'MildDemented': float(predictions[i][2]),
                    'ModerateDemented': float(predictions[i][3])
                })
                db.session.add(pred)
            
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist()
            })
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    admin_id = session.get('admin_id')
    models = TrainedModel.query.filter_by(admin_id=admin_id, is_active=True).all()
    return render_template('admin/run_inference.html', models=models)

@admin_bp.route('/performance-reports')
@admin_login_required
def performance_reports():
    """View performance reports"""
    admin_id = session.get('admin_id')
    models = TrainedModel.query.filter_by(admin_id=admin_id).all()
    return render_template('admin/performance_reports.html', models=models)

@admin_bp.route('/confusion-matrix/<int:model_id>')
@admin_login_required
def get_confusion_matrix(model_id):
    """Get confusion matrix for model"""
    model = TrainedModel.query.get_or_404(model_id)
    
    if model.admin_id != session.get('admin_id'):
        return jsonify({'status': 'error'}), 403
    
    cm_data = model.get_confusion_matrix()
    return jsonify(cm_data)

@admin_bp.route('/training-logs')
@admin_login_required
def training_logs():
    """View training logs"""
    admin_id = session.get('admin_id')
    logs = TrainingLog.query.filter_by(admin_id=admin_id).order_by(
        TrainingLog.created_at.desc()
    ).all()
    return render_template('admin/training_logs.html', logs=logs)

@admin_bp.route('/api/training-logs')
@admin_login_required
def api_training_logs():
    """API endpoint for training logs"""
    admin_id = session.get('admin_id')
    logs = TrainingLog.query.filter_by(admin_id=admin_id).order_by(
        TrainingLog.created_at.desc()
    ).limit(100).all()
    
    return jsonify([{
        'epoch': log.epoch,
        'batch': log.batch,
        'loss': log.loss,
        'accuracy': log.accuracy,
        'val_loss': log.val_loss,
        'val_accuracy': log.val_accuracy,
        'created_at': log.created_at.isoformat()
    } for log in logs])
