"""
TruthCheck Route Blueprints
Central registration point for all Flask blueprints.
"""


def register_blueprints(app):
    """Import and register all route blueprints with the Flask application.

    This function acts as the single entry point for blueprint registration,
    keeping ``app.py`` clean. Each blueprint is imported inside the function
    to avoid circular-import issues.

    Args:
        app: The Flask application instance.
    """
    from routes.main import main_bp
    from routes.auth import auth_bp
    from routes.analysis import analysis_bp
    from routes.youtube import youtube_bp
    from routes.url_verify import url_bp
    from routes.image_detect import image_bp
    from routes.video_detect import video_bp
    from routes.chat import chat_bp
    from routes.history import history_bp
    from routes.reports import reports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(youtube_bp)
    app.register_blueprint(url_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(reports_bp)
