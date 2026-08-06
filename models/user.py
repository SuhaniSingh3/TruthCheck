"""
User Model — TruthCheck Authentication
Stores user credentials, preferences, and profile data.
"""
from extensions import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


class User(UserMixin, db.Model):
    """Registered user with authentication and personalization support."""

    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)

    # --- Preferences ---
    theme_preference = db.Column(db.String(30), default='theme-cosmic')
    language_preference = db.Column(db.String(10), default='en')
    dark_mode = db.Column(db.Boolean, default=True)

    # --- Timestamps ---
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # --- Relationships ---
    reports = db.relationship('Report', backref='user', lazy='dynamic',
                              cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and store the user's password securely."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    @property
    def total_analyses(self):
        """Count of all analyses performed by this user."""
        return self.reports.count()

    @property
    def fake_detected(self):
        """Count of analyses that returned FAKE."""
        return self.reports.filter(
            Report.prediction.ilike('%fake%')
        ).count()

    @property
    def real_detected(self):
        """Count of analyses that returned REAL."""
        return self.reports.filter(
            Report.prediction.ilike('%real%')
        ).count()

    def __repr__(self):
        return f'<User {self.username}>'


# Required by Flask-Login: callback to reload user from session
from extensions import login_manager

@login_manager.user_loader
def load_user(user_id):
    """Load a user by their primary key for Flask-Login session management."""
    return User.query.get(int(user_id))
