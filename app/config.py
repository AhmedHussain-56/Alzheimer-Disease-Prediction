import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///alzheimer_db.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max file size
    
    # Session settings
    TEMPLATES_AUTO_RELOAD = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Model settings
    MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
    
    # Ensure folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test_db.db'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
