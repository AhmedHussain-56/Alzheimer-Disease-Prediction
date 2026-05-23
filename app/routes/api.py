from flask import Blueprint, jsonify
from app.models import db, TrainedModel, UserPrediction, AdminPrediction

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Alzheimer Prediction System is running'})

@api_bp.route('/models/stats')
def models_stats():
    """Get models statistics"""
    total_models = TrainedModel.query.count()
    active_models = TrainedModel.query.filter_by(is_active=True).count()
    avg_accuracy = db.session.query(db.func.avg(TrainedModel.accuracy)).scalar() or 0
    
    return jsonify({
        'total_models': total_models,
        'active_models': active_models,
        'average_accuracy': float(avg_accuracy)
    })

@api_bp.route('/predictions/stats')
def predictions_stats():
    """Get predictions statistics"""
    total_predictions = UserPrediction.query.count()
    admin_predictions = AdminPrediction.query.count()
    
    return jsonify({
        'user_predictions': total_predictions,
        'admin_predictions': admin_predictions,
        'total': total_predictions + admin_predictions
    })
