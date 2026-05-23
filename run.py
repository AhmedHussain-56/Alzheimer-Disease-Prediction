#!/usr/bin/env python
"""
Alzheimer Disease Prediction System - IoMT & Deep Learning
Main application entry point
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User, Admin, TrainedModel, UserPrediction, AdminPrediction, TrainingLog, TestLog, PerformanceReport

# Create Flask app
app = create_app(os.getenv('FLASK_ENV', 'development'))

@app.shell_context_processor
def make_shell_context():
    """Make context available in shell"""
    return {
        'db': db,
        'User': User,
        'Admin': Admin,
        'TrainedModel': TrainedModel,
        'UserPrediction': UserPrediction,
        'AdminPrediction': AdminPrediction,
        'TrainingLog': TrainingLog,
        'TestLog': TestLog,
        'PerformanceReport': PerformanceReport
    }

@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print('Database initialized!')

@app.cli.command()
def create_admin():
    """Create default admin user"""
    with app.app_context():
        # Check if admin already exists
        admin = Admin.query.filter_by(username='admin').first()
        if admin:
            print('Admin user already exists!')
            return
        
        # Create admin
        admin = Admin(
            username='admin',
            email='admin@alzheimer.local'
        )
        admin.set_password('admin123')
        
        db.session.add(admin)
        db.session.commit()
        
        print('Admin user created successfully!')
        print('Username: admin')
        print('Password: admin123')
        print('IMPORTANT: Change the password in production!')

@app.cli.command()
def create_demo_user():
    """Create demo user"""
    with app.app_context():
        # Check if user already exists
        user = User.query.filter_by(username='demo').first()
        if user:
            print('Demo user already exists!')
            return
        
        # Create user
        user = User(
            username='demo',
            email='demo@alzheimer.local',
            first_name='Demo',
            last_name='User',
            phone='1234567890'
        )
        user.set_password('demo123')
        
        db.session.add(user)
        db.session.commit()
        
        print('Demo user created successfully!')
        print('Username: demo')
        print('Password: demo123')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
