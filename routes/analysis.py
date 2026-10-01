"""
TruthCheck — Analysis Routes
============================
Core news text and headline authenticity detection endpoints.
Provides robust text analysis and preserves the /predict API.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from datetime import datetime
import logging

from extensions import db
from models.report import Report
from config import Config

logger = logging.getLogger(__name__)

analysis_bp = Blueprint('analysis', __name__)


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


def _extract_request_text(data):
    """Extract text/headline from various JSON or Form field names."""
    if not isinstance(data, dict):
        return ''
    return (
        data.get('text')
        or data.get('headline')
        or data.get('title')
        or data.get('content')
        or data.get('article')
        or data.get('claim')
        or data.get('query')
        or ''
    ).strip()


# ──────────────────────────────────────────────
# Original /predict endpoint (PRESERVED)
# ──────────────────────────────────────────────
@analysis_bp.route('/predict', methods=['POST'])
def predict():
    """Predict news authenticity — ORIGINAL API CONTRACT.

    Accepts JSON ``{"text": "..."}`` and returns the Groq prediction result.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        text = _extract_request_text(data)

        if not text:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        if len(text) < 5:
            return jsonify({'error': 'Text too short. Please provide at least 5 characters.'}), 400

        # --- Groq Prediction ---
        from services.groq_service import predict_news
        result = predict_news(text)

        if not result:
            logger.error("predict_news returned empty result for text: %s", text[:80])
            return jsonify({
                'error': 'Prediction service is currently unable to analyze this text. Please try again.',
            }), 500

        # Save analysis to database
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type='text',
                input_text=text[:5000],
                prediction=result.get('label', ''),
                confidence=result.get('confidence'),
                risk_level=result.get('risk_level', ''),
                source=result.get('source', 'Groq AI'),
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()
        except Exception as db_err:
            logger.debug("Database report save skipped: %s", db_err)
            db.session.rollback()

        return jsonify({
            'success': True,
            'source': result.get('source', 'Groq AI'),
            'text': text[:150] + '...' if len(text) > 150 else text,
            **result,
            'timestamp': datetime.now().isoformat(),
        }), 200

    except Exception as e:
        logger.exception("Error in /predict endpoint: %s", e)
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


# ──────────────────────────────────────────────
# Result page (preserved)
# ──────────────────────────────────────────────
@analysis_bp.route('/result', methods=['GET'])
def result_page():
    """Render the analysis result page."""
    return render_template('result.html')


# ──────────────────────────────────────────────
# Text Analysis Endpoints (/api/analyze and POST /analyze)
# ──────────────────────────────────────────────
@analysis_bp.route('/api/analyze', methods=['POST'])
def smart_analyze():
    """Analyze news text, headline, or claim for factual authenticity.

    Accepts JSON or Form:
        - ``text`` or ``headline``: The news text or claim to analyze.
        - ``response_lang`` (optional): Preferred response language code.

    Returns:
        JSON object with analysis results.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        text = _extract_request_text(data)
        response_lang = data.get('response_lang', getattr(Config, 'DEFAULT_LANGUAGE', 'en'))

        if not text:
            return jsonify({'error': 'Please provide news text, an article, or a headline to analyze.'}), 400

        if len(text) < 5:
            return jsonify({'error': 'Text too short (minimum 5 characters required)'}), 400

        from services.groq_service import predict_news
        result = predict_news(text, response_lang=response_lang)

        if not result:
            logger.error("Analysis service produced no result for input: %s", text[:80])
            return jsonify({'error': 'Analysis service was unable to process this request. Please try again.'}), 500

        # Persist report
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type='text',
                input_text=text[:5000],
                input_title=result.get('title', text[:100]),
                prediction=result.get('label', result.get('prediction', '')),
                confidence=result.get('confidence'),
                risk_level=result.get('risk_level', ''),
                response_language=response_lang,
                source=result.get('source', 'Groq AI'),
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()

            result['report_id'] = report.id
        except Exception as db_err:
            logger.debug("Database report save skipped: %s", db_err)
            db.session.rollback()

        return jsonify({
            'success': True,
            'input_type': 'text',
            **result,
            'timestamp': datetime.now().isoformat(),
        }), 200

    except Exception as e:
        logger.exception("Unexpected error in /api/analyze: %s", e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


# ──────────────────────────────────────────────
# Text Analysis Route (GET: Page, POST: Smart Analyze)
# ──────────────────────────────────────────────
@analysis_bp.route('/analyze', methods=['GET', 'POST'])
def analyze_text_page():
    """Render the text analysis form page on GET, or run analysis on POST."""
    if request.method == 'POST':
        return smart_analyze()

    return render_template(
        'analysis/text.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )
