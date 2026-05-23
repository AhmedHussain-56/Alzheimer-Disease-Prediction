from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app.models import db, User, Admin
from functools import wraps

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

def login_required(f):
    """Check if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_login_required(f):
    """Check if admin is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin login required', 'warning')
            return redirect(url_for('auth.admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        data = request.form
        
        # Validate required fields
        if not all([data.get('username'), data.get('email'), data.get('first_name'), 
                   data.get('last_name'), data.get('password'), data.get('phone')]):
            flash('All fields are required', 'danger')
            return redirect(url_for('auth.register'))
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(email=data['email']).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('auth.register'))
        
        try:
            # Create new user
            user = User(
                username=data['username'],
                email=data['email'],
                first_name=data['first_name'],
                last_name=data['last_name'],
                phone=data['phone']
            )
            user.set_password(data['password'])
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error during registration: {str(e)}', 'danger')
            return redirect(url_for('auth.register'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        data = request.form
        
        user = User.query.filter_by(username=data.get('username')).first()
        
        if user and user.check_password(data.get('password', '')):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_name'] = f"{user.first_name} {user.last_name}"
            flash(f'Welcome {user.first_name}!', 'success')
            return redirect(url_for('user.dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """Admin login"""
    if request.method == 'POST':
        data = request.form
        
        admin = Admin.query.filter_by(username=data.get('username')).first()
        
        if admin and admin.check_password(data.get('password', '')):
            session['admin_id'] = admin.id
            session['admin_username'] = admin.username
            flash(f'Welcome Admin {admin.username}!', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid admin credentials', 'danger')
    
    return render_template('admin_login.html')

@auth_bp.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/admin-logout')
def admin_logout():
    """Admin logout"""
    session.clear()
    flash('Admin logged out', 'info')
    return redirect(url_for('auth.admin_login'))
