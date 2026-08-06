"""
TruthCheck — AI-Powered Misinformation Detection Platform
==========================================================
Main application entry point. Creates the Flask app, registers blueprints,
initializes extensions, and starts the development server.

Original app.py functionality (text analysis via Groq) is fully preserved
and accessible at the /predict endpoint. All new modules are added as
separate Flask Blueprints for clean separation of concerns.
"""
from flask import Flask, request, jsonify, render_template
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ─── Application Factory ───────────────────────────────────────────────────────

def create_app():
    """
    Application factory that creates and configures the Flask app.
    This pattern supports testing and multiple configurations.
    """
    app = Flask(__name__)

    # ── Load Configuration ──
    from config import Config
    app.config.from_object(Config)

    # Ensure required directories exist
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(app.instance_path), exist_ok=True)

    # ── Initialize Extensions ──
    from extensions import db, login_manager
    db.init_app(app)
    login_manager.init_app(app)

    # ── Create Database Tables ──
    with app.app_context():
        # Import models so SQLAlchemy knows about them
        from models.user import User
        from models.report import Report
        db.create_all()

    # ── Register Blueprints (all new modular routes) ──
    from routes import register_blueprints
    register_blueprints(app)

    # ── Inject Global Template Variables ──
    @app.context_processor
    def inject_globals():
        """Make config and utility data available to all templates."""
        return {
            'supported_languages': Config.SUPPORTED_LANGUAGES,
            'app_version': '2.0.0',
            'current_year': datetime.now().year,
        }

    # ── Register Preserved Routes ──
    @app.route('/landing', methods=['GET'])
    def landing():
        return render_template('index.html')

    @app.route('/result', methods=['GET'])
    def result_page():
        return render_template('result.html')

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'groq_active': client is not None
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'Missing "text" field'}), 400

            text = data['text'].strip()
            if len(text) < 10:
                return jsonify({'error': 'Text too short'}), 400

            result = predict_with_groq(text)
            if result:
                response_data = {
                    'success': True,
                    'source': 'Groq (Llama-3.3)',
                    'text': text[:150] + '...' if len(text) > 150 else text,
                    **result,
                    'timestamp': datetime.now().isoformat()
                }
                try:
                    from models.report import Report
                    report = Report(
                        input_type='text',
                        input_text=text[:500],
                        prediction=result.get('label', ''),
                        confidence=result.get('confidence'),
                        source='Groq (Llama-3.3)',
                    )
                    report.set_result(response_data)
                    db.session.add(report)
                    db.session.commit()
                    response_data['report_id'] = report.id
                except Exception as db_err:
                    print(f"DB save warning (non-critical): {db_err}")

                return jsonify(response_data)

            return jsonify({'error': 'Prediction service unavailable. Please check GROQ_API_KEY.'}), 503

        except Exception as e:
            return jsonify({'error': f'Server Error: {str(e)}'}), 500

    return app


# ─── PRESERVED: Original Groq Prediction Logic ─────────────────────────────────
# This section preserves the exact original app.py code for backward compatibility.
# The /predict endpoint is also registered via routes/analysis.py blueprint,
# but this standalone version ensures the original API contract is never broken.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)


def predict_with_groq(text):
    """
    Predict news authenticity using Groq API (Llama-3.3).
    PRESERVED from original app.py — do not modify this function.
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
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news: {text[:4000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None


# ─── Create the App ─────────────────────────────────────────────────────────────

app = create_app()


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
    print("|  API Docs:   POST /predict, /api/analyze, /api/youtube/...   |")
    print("+" + "-"*62 + "+\n")

    app.run(debug=True, host='localhost', port=5000)
