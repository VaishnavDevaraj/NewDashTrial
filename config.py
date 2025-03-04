# Configuration settings for the application

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'smart_dashboard',
    'user': 'dashboard_user',
    'password': 'your_password_here'  # In production, use environment variables
}

# Flask configuration
SECRET_KEY = 'your_secret_key_here'  # Change this in production
DEBUG = True

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'sqlite', 'db'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Visualization settings
DEFAULT_CHART_COLORS = [
    '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
    '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
]
MAX_CATEGORIES_IN_CHART = 10
