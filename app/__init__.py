from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
from functools import wraps
from datetime import timedelta
import os
from app.config import config
from app.models import db, User, Admin, TrainedModel, UserPrediction, AdminPrediction, TrainingLog, TestLog, PerformanceReport
import json

def create_app(config_name='development'):
    """Application factory"""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize database
    db.init_app(app)
    
    # Session configuration
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.user import user_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(api_bp)
    
    @app.route('/')
    def index():
        """Home page"""
        if 'user_id' in session:
            return redirect(url_for('user.dashboard'))
        elif 'admin_id' in session:
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('auth.login'))
    
    @app.before_request
    def set_session_timeout():
        """Set session timeout"""
        session.permanent = True
        app.permanent_session_lifetime = timedelta(hours=24)
    
    return app
