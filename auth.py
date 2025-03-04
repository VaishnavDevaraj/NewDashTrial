from flask import Blueprint, request, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import re
from functools import wraps
from database import Database

auth_bp = Blueprint('auth', __name__)
db = Database()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        
        # Basic validation
        if not email or not password:
            flash('Email and password are required', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            flash('Invalid email format', 'danger')
            return render_template('register.html')
        
        # Password strength validation
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return render_template('register.html')
        
        # Check if user already exists
        existing_user = db.get_user_by_email(email)
        if existing_user:
            flash('Email already registered', 'danger')
            return render_template('register.html')
        
        # Create new user
        password_hash = generate_password_hash(password)
        user_id = db.add_user(email, password_hash, name)
        
        if user_id:
            # Log the registration in history
            db.add_history(user_id, 'registration', {'email': email})
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Registration failed', 'danger')
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Email and password are required', 'danger')
            return render_template('login.html')
        
        user = db.get_user_by_email(email)
        
        if user and check_password_hash(user['password_hash'], password):
            # Store user info in session
            session['user_id'] = user['id']
            session['email'] = user['email']
            session['name'] = user['name']
            
            # Log the login in history
            db.add_history(user['id'], 'login', {'email': email})
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard.index'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    if 'user_id' in session:
        # Log the logout in history
        db.add_history(session['user_id'], 'logout', {'email': session.get('email')})
        
    # Clear session
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = db.get_user_by_email(session['email'])
    
    if request.method == 'POST':
        name = request.form.get('name')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # Update name
        if name and name != user['name']:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET name = ? WHERE id = ?", (name, user['id']))
            conn.commit()
            conn.close()
            session['name'] = name
            flash('Profile updated successfully', 'success')
        
        # Update password
        if current_password and new_password:
            if not check_password_hash(user['password_hash'], current_password):
                flash('Current password is incorrect', 'danger')
            elif new_password != confirm_password:
                flash('New passwords do not match', 'danger')
            elif len(new_password) < 8:
                flash('Password must be at least 8 characters long', 'danger')
            else:
                password_hash = generate_password_hash(new_password)
                conn = db.get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user['id']))
                conn.commit()
                conn.close()
                
                # Log password change in history
                db.add_history(user['id'], 'password_change', {'email': user['email']})
                
                flash('Password updated successfully', 'success')
    
    return render_template('profile.html', user=user)
