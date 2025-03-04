from flask import Flask, render_template, redirect, url_for
import os
from auth import auth_bp
from dashboard import dashboard_bp
from history import history_bp
from config import SECRET_KEY, DEBUG, UPLOAD_FOLDER
from database import Database

# Initialize the application
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['DEBUG'] = DEBUG
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
db = Database()

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
app.register_blueprint(history_bp, url_prefix='/history')

@app.route('/')
def index():
    return redirect(url_for('dashboard.index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
