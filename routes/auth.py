"""
TruthCheck — Authentication Routes
Login, signup, logout, password reset, and profile management.
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime

from extensions import db
from models.user import User
from config import Config

auth_bp = Blueprint('auth', __name__)


# ──────────────────────────────────────────────
# Login
# ──────────────────────────────────────────────
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login.

    GET  — render the login form.
    POST — validate credentials, log the user in, and redirect to the dashboard.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            remember = bool(request.form.get('remember'))

            if not email or not password:
                flash('Please enter both email and password.', 'error')
                return render_template('auth/login.html')

            user = User.query.filter_by(email=email).first()

            if user is None or not user.check_password(password):
                flash('Invalid email or password.', 'error')
                return render_template('auth/login.html')

            # Update last login timestamp
            user.last_login = datetime.utcnow()
            db.session.commit()

            login_user(user, remember=remember)
            flash('Welcome back!', 'success')

            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.dashboard'))
        except Exception as e:
            db.session.rollback()
            flash(f'Login error: {str(e)}', 'error')
            return render_template('auth/login.html')

    return render_template('auth/login.html')


# ──────────────────────────────────────────────
# Signup
# ──────────────────────────────────────────────
@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle new user registration.

    GET  — render the signup form.
    POST — validate inputs, create the user, log them in, and redirect.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')

            # --- Validation ---
            errors = []
            if not username or len(username) < 3:
                errors.append('Username must be at least 3 characters.')
            if not email or '@' not in email:
                errors.append('Please enter a valid email address.')
            if not password or len(password) < 6:
                errors.append('Password must be at least 6 characters.')
            if password != confirm_password:
                errors.append('Passwords do not match.')

            if User.query.filter_by(username=username).first():
                errors.append('Username already taken.')
            if User.query.filter_by(email=email).first():
                errors.append('Email already registered.')

            if errors:
                for err in errors:
                    flash(err, 'error')
                return render_template('auth/signup.html')

            # --- Create user ---
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            login_user(user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('main.dashboard'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration error: {str(e)}', 'error')
            return render_template('auth/signup.html')

    return render_template('auth/signup.html')


# ──────────────────────────────────────────────
# Logout
# ──────────────────────────────────────────────
@auth_bp.route('/logout')
def logout():
    """Log the current user out and redirect to the landing page."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.landing'))


# ──────────────────────────────────────────────
# Forgot Password
# ──────────────────────────────────────────────
@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle password-reset requests.

    GET  — render the forgot-password form.
    POST — display a success message (no email backend yet).
    """
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if email:
            flash(
                'If an account with that email exists, a password reset link '
                'has been sent. Please check your inbox.',
                'success',
            )
        else:
            flash('Please enter your email address.', 'error')
        return render_template('auth/forgot_password.html')

    return render_template('auth/forgot_password.html')


# ──────────────────────────────────────────────
# Profile
# ──────────────────────────────────────────────
@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View or update user profile preferences.

    GET  — render the profile page.
    POST — update username, email, theme, and language preferences.
    """
    if request.method == 'POST':
        try:
            new_username = request.form.get('username', '').strip()
            new_email = request.form.get('email', '').strip().lower()
            new_theme = request.form.get('theme_preference', current_user.theme_preference)
            new_language = request.form.get('language_preference', current_user.language_preference)
            new_full_name = request.form.get('full_name', '').strip()

            # --- Validate uniqueness ---
            if new_username and new_username != current_user.username:
                if User.query.filter_by(username=new_username).first():
                    flash('Username already taken.', 'error')
                    return render_template(
                        'auth/profile.html',
                        supported_languages=Config.SUPPORTED_LANGUAGES,
                    )
                current_user.username = new_username

            if new_email and new_email != current_user.email:
                if User.query.filter_by(email=new_email).first():
                    flash('Email already registered.', 'error')
                    return render_template(
                        'auth/profile.html',
                        supported_languages=Config.SUPPORTED_LANGUAGES,
                    )
                current_user.email = new_email

            current_user.theme_preference = new_theme
            current_user.language_preference = new_language
            if new_full_name:
                current_user.full_name = new_full_name

            db.session.commit()
            flash('Profile updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'error')

    return render_template(
        'auth/profile.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )


# ──────────────────────────────────────────────
# Change Password
# ──────────────────────────────────────────────
@auth_bp.route('/profile/password', methods=['POST'])
@login_required
def change_password():
    """Change the authenticated user's password.

    Expects form fields: current_password, new_password, confirm_new_password.
    """
    try:
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_new = request.form.get('confirm_new_password', '')

        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('auth.profile'))

        if len(new_password) < 6:
            flash('New password must be at least 6 characters.', 'error')
            return redirect(url_for('auth.profile'))

        if new_password != confirm_new:
            flash('New passwords do not match.', 'error')
            return redirect(url_for('auth.profile'))

        current_user.set_password(new_password)
        db.session.commit()
        flash('Password changed successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error changing password: {str(e)}', 'error')

    return redirect(url_for('auth.profile'))
