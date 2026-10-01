"""
TruthCheck Route Blueprints
===========================
Central registration point for active Flask blueprints.
"""


def register_blueprints(app):
    """Import and register active route blueprints with the Flask application.

    Args:
        app: The Flask application instance.
    """
    from routes.main import main_bp
    from routes.auth import auth_bp
    from routes.analysis import analysis_bp
    from routes.image_detect import image_bp
    from routes.chat import chat_bp
    from routes.history import history_bp
    from routes.reports import reports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(reports_bp)
