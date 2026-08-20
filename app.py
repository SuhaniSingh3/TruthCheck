"""
TruthCheck — AI-Powered Misinformation Detection Platform
==========================================================
Main application entry point. Creates the Flask app, registers blueprints,
initializes extensions, and starts the development server.

Vercel Compatibility:
  - No writes to /var/task (read-only on Vercel)
  - /tmp used only when absolutely required by library
  - Database init is lazy (skipped on Vercel cold start when no DB URL)
  - All optional services fail gracefully with clear log messages
  - No duplicate route registrations (blueprints own /predict, /result, /landing)
"""
import os
import json
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─── Vercel Detection ──────────────────────────────────────────────────────────
from config import IS_VERCEL

# ─── Groq Client (module-level, for /predict preserved endpoint) ──────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
try:
    if GROQ_API_KEY:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized.")
    else:
        logger.warning("GROQ_API_KEY not set — Groq prediction unavailable.")
except Exception as _groq_err:
    logger.warning("Groq client init failed (non-critical): %s", _groq_err)


# ─── Application Factory ───────────────────────────────────────────────────────

def create_app():
    """
    Application factory that creates and configures the Flask app.
    Designed to start cleanly on Vercel even when DATABASE_URL is missing.
    """
    logger.info("Starting TruthCheck Flask application")
    logger.info("Vercel environment: %s", IS_VERCEL)

    from flask import Flask, request, jsonify, render_template
    app = Flask(__name__)

    # ── Load Configuration ──────────────────────────────────────────────────
    logger.info("Loading configuration")
    from config import Config
    app.config.from_object(Config)

    # ── Directory Setup (LOCAL ONLY — never write to /var/task on Vercel) ──
    if not IS_VERCEL:
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        try:
            os.makedirs(upload_folder, exist_ok=True)
            logger.info("Upload folder ready: %s", upload_folder)
        except OSError as e:
            logger.warning("Could not create upload folder %s: %s", upload_folder, e)
        try:
            # instance_path is the Flask instance folder (for SQLite locally)
            instance_dir = app.instance_path
            os.makedirs(instance_dir, exist_ok=True)
            logger.info("Instance folder ready: %s", instance_dir)
        except OSError as e:
            logger.warning("Could not create instance folder: %s", e)
    else:
        logger.info("Vercel mode: skipping local directory creation")

    # ── Initialize Extensions ───────────────────────────────────────────────
    logger.info("Initializing database extension")
    try:
        from extensions import db, login_manager
        db.init_app(app)
        login_manager.init_app(app)
        logger.info("Flask-SQLAlchemy and Flask-Login initialized")
    except Exception as ext_err:
        logger.exception("Extension init failed: %s", ext_err)
        raise  # This is critical — can't continue without db/login_manager

    # ── Database Table Creation ─────────────────────────────────────────────
    # On Vercel: only run if DATABASE_URL is configured (PostgreSQL).
    # Never attempt SQLite writes to read-only /var/task filesystem.
    # Locally: always attempt table creation.
    logger.info("Initializing database tables")
    db_url = app.config.get('SQLALCHEMY_DATABASE_URI', '')
    should_create_tables = False

    if IS_VERCEL:
        # Only create tables on Vercel if we have a real external DATABASE_URL
        has_external_db = bool(os.getenv('DATABASE_URL'))
        if has_external_db:
            should_create_tables = True
            logger.info("Vercel + DATABASE_URL detected — will attempt table creation")
        else:
            logger.warning(
                "Vercel mode with no DATABASE_URL — skipping db.create_all(). "
                "Database-dependent features will be unavailable."
            )
    else:
        should_create_tables = True
        logger.info("Local mode — will create tables (SQLite)")

    if should_create_tables:
        with app.app_context():
            try:
                from models.user import User
                from models.report import Report
                db.create_all()
                logger.info("Database tables verified/created successfully")
            except Exception as db_init_err:
                logger.warning(
                    "Database table creation failed (non-critical): %s", db_init_err
                )

    # ── Register Blueprints ─────────────────────────────────────────────────
    # Blueprints own: /predict, /result, /landing, /image-detect, etc.
    # DO NOT register duplicate routes here.
    logger.info("Registering blueprints")
    _bp_error = None
    try:
        from routes import register_blueprints
        register_blueprints(app)
        logger.info("All blueprints registered successfully")
    except Exception as bp_err:
        logger.exception("Blueprint registration failed: %s", bp_err)
        _bp_error = str(bp_err)
        # Non-fatal on Vercel: register a fallback route so the function
        # responds with a diagnostic JSON rather than a bare 500 crash page.
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def blueprint_error_fallback(path):
            from flask import jsonify
            return jsonify({
                'error': 'Application startup error',
                'detail': _bp_error,
                'status': 'blueprint_registration_failed',
            }), 503

    # ── Inject Global Template Variables ────────────────────────────────────
    @app.context_processor
    def inject_globals():
        """Make config and utility data available to all templates."""
        return {
            'supported_languages': Config.SUPPORTED_LANGUAGES,
            'app_version': '2.0.0',
            'current_year': datetime.now().year,
        }

    # ── Health Endpoint ─────────────────────────────────────────────────────
    # Registered directly (not in blueprint) — always available.
    @app.route('/health', methods=['GET'])
    def health():
        """
        Health check endpoint.
        Returns HTTP 200 even if optional services are unavailable.
        Never exposes secrets.
        """
        groq_configured = bool(GROQ_API_KEY)
        db_configured = bool(os.getenv('DATABASE_URL') or not IS_VERCEL)

        # Quick DB connectivity check (non-fatal)
        db_connected = False
        try:
            from extensions import db as _db
            with app.app_context():
                _db.session.execute(_db.text('SELECT 1'))
                db_connected = True
        except Exception:
            pass

        return jsonify({
            'status': 'healthy',
            'vercel': IS_VERCEL,
            'groq_configured': groq_configured,
            'database_configured': db_configured,
            'database_connected': db_connected,
            'image_verification': True,
            'timestamp': datetime.now().isoformat(),
        }), 200

    # ── Global Error Handlers ───────────────────────────────────────────────
    # Return JSON for all HTTP errors so Vercel never shows a raw exception page.

    @app.errorhandler(404)
    def not_found(e):
        from flask import jsonify, request as req
        return jsonify({'error': 'Not found', 'path': req.path}), 404

    @app.errorhandler(413)
    def payload_too_large(e):
        from flask import jsonify
        return jsonify({'error': 'File too large. Maximum allowed size is 20 MB.'}), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        from flask import jsonify
        logger.error("Unhandled 500 error: %s", e)
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        from flask import jsonify
        logger.exception("Unhandled exception: %s", e)
        return jsonify({'error': 'Unexpected server error', 'detail': str(e)}), 500

    logger.info("TruthCheck application initialized successfully")
    return app


# ─── PRESERVED: Original Groq Prediction Logic ─────────────────────────────────
# This section preserves the original app.py Groq logic for backward compatibility.
# NOTE: The /predict route is owned by routes/analysis.py blueprint.
#       This function is kept for internal reuse and backward compatibility.

def predict_with_groq(text):
    """
    Predict news authenticity using Groq API (Llama-3.3).
    PRESERVED from original app.py — do not modify this function.
    Used internally; the actual /predict endpoint is in routes/analysis.py.
    """
    if not client:
        return None

    system_prompt = (
        "You are an expert news fact-checker. Analyze the provided news text and determine if it is REAL or FAKE.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"label": "FAKE NEWS" or "REAL NEWS", "prediction": 1 or 0, "confidence": float, "reasons": [list of strings], "summary": "string"}\n'
        "Use prediction 1 for FAKE and 0 for REAL."
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "groq/compound-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news: {text[:4000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Groq API Error: %s", e)
        return None


# ─── Create the App ─────────────────────────────────────────────────────────────

try:
    app = create_app()
except Exception as startup_err:
    logger.exception("TruthCheck startup FAILED: %s", startup_err)
    raise


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not GROQ_API_KEY:
        print("⚠️  WARNING: GROQ_API_KEY not found in .env")
        print("   Add your key to .env: GROQ_API_KEY=gsk_your_key_here")
        print("   Get one at: https://console.groq.com/\n")

    print("+" + "-"*62 + "+")
    print("|           TruthCheck v2.0 -- Enterprise AI Platform          |")
    print("|          AI-Powered Misinformation Detection System          |")
    print("+" + "-"*62 + "+")
    print("|  Dashboard:  http://localhost:5000                           |")
    print("|  Landing:    http://localhost:5000/landing                   |")
    print("|  API Health: http://localhost:5000/health                    |")
    print("|  Image:      http://localhost:5000/image-detect              |")
    print("|  API Docs:   POST /predict, /api/analyze, /api/verify-image  |")
    print("+" + "-"*62 + "+\n")

    app.run(debug=True, host='localhost', port=5000)
