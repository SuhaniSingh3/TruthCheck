"""
TruthCheck Flask Extensions
Initialized here to avoid circular imports. Each extension is created without
an app instance and later bound via init_app() in the application factory.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# --- Database ORM ---
db = SQLAlchemy()

# --- Authentication Manager ---
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'
