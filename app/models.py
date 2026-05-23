from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    """User model for registration and login"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    predictions = db.relationship('UserPrediction', backref='user', lazy=True, cascade='all, delete-orphan')
    test_logs = db.relationship('TestLog', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }

class Admin(db.Model):
    """Admin model"""
    __tablename__ = 'admins'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    trained_models = db.relationship('TrainedModel', backref='admin', lazy=True)
    training_logs = db.relationship('TrainingLog', backref='admin', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class TrainedModel(db.Model):
    """Trained model information"""
    __tablename__ = 'trained_models'
    
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('admins.id'), nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)  # resnet101, unet, transfer_learning, deep_learning
    model_path = db.Column(db.String(255), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    training_epochs = db.Column(db.Integer, nullable=True)
    batch_size = db.Column(db.Integer, nullable=True)
    learning_rate = db.Column(db.Float, nullable=True)
    confusion_matrix = db.Column(db.Text, nullable=True)  # JSON string
    training_history = db.Column(db.Text, nullable=True)  # JSON string
    parameters_json = db.Column(db.Text, nullable=True)  # JSON string for all params
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    predictions = db.relationship('AdminPrediction', backref='model', lazy=True)
    
    def set_confusion_matrix(self, cm_dict):
        self.confusion_matrix = json.dumps(cm_dict)
    
    def get_confusion_matrix(self):
        if self.confusion_matrix:
            return json.loads(self.confusion_matrix)
        return None
    
    def set_training_history(self, history_dict):
        self.training_history = json.dumps(history_dict)
    
    def get_training_history(self):
        if self.training_history:
            return json.loads(self.training_history)
        return None

class UserPrediction(db.Model):
    """User predictions/test results"""
    __tablename__ = 'user_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    image_filename = db.Column(db.String(120), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)  # NonDemented, VeryMildDemented, MildDemented, ModerateDemented
    confidence = db.Column(db.Float, nullable=False)
    all_predictions = db.Column(db.Text, nullable=True)  # JSON - all class probabilities
    model_used = db.Column(db.String(120), nullable=True)
    processing_time = db.Column(db.Float, nullable=True)  # in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_all_predictions(self, predictions_dict):
        self.all_predictions = json.dumps(predictions_dict)
    
    def get_all_predictions(self):
        if self.all_predictions:
            return json.loads(self.all_predictions)
        return None

class AdminPrediction(db.Model):
    """Admin predictions for batch testing"""
    __tablename__ = 'admin_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('trained_models.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    image_filename = db.Column(db.String(120), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    actual_class = db.Column(db.String(50), nullable=True)
    is_correct = db.Column(db.Boolean, nullable=True)
    all_predictions = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_all_predictions(self, predictions_dict):
        self.all_predictions = json.dumps(predictions_dict)
    
    def get_all_predictions(self):
        if self.all_predictions:
            return json.loads(self.all_predictions)
        return None

class TrainingLog(db.Model):
    """Training logs for distributed training"""
    __tablename__ = 'training_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('admins.id'), nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    algorithm = db.Column(db.String(50), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    batch = db.Column(db.Integer, nullable=False)
    loss = db.Column(db.Float, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    val_loss = db.Column(db.Float, nullable=True)
    val_accuracy = db.Column(db.Float, nullable=True)
    log_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TestLog(db.Model):
    """User test logs"""
    __tablename__ = 'test_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(120), nullable=False)
    result_class = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), default='success')  # success, error
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PerformanceReport(db.Model):
    """Performance reports for models"""
    __tablename__ = 'performance_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('trained_models.id'), nullable=False)
    total_predictions = db.Column(db.Integer, default=0)
    correct_predictions = db.Column(db.Integer, default=0)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    confusion_matrix = db.Column(db.Text, nullable=True)
    report_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_confusion_matrix(self, cm_dict):
        self.confusion_matrix = json.dumps(cm_dict)
    
    def get_confusion_matrix(self):
        if self.confusion_matrix:
            return json.loads(self.confusion_matrix)
        return None
