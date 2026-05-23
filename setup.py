#!/usr/bin/env python
"""
Setup script - Initialize database and create admin user
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import Admin, User

def setup():
    """Initialize database and create default users"""
    
    # Create Flask app
    app = create_app('development')
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created!")
        
        # Check and create admin
        admin = Admin.query.filter_by(username='admin').first()
        if not admin:
            admin = Admin(
                username='admin',
                email='admin@alzheimer.local'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created!")
            print("   Username: admin")
            print("   Password: admin123")
        else:
            print("✅ Admin user already exists")
        
        # Check and create demo user
        user = User.query.filter_by(username='demo').first()
        if not user:
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
            print("✅ Demo user created!")
            print("   Username: demo")
            print("   Password: demo123")
        else:
            print("✅ Demo user already exists")
        
        print("\n✅ Setup completed successfully!")

if __name__ == '__main__':
    setup()
