"""
TruthCheck — Analysis Routes
Core text analysis and smart multi-type analysis endpoint.
Preserves the original /predict API contract.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from datetime import datetime
import json
import re

from extensions import db
from models.report import Report
from config import Config

analysis_bp = Blueprint('analysis', __name__)


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


def _detect_input_type(text):
    """Heuristically detect whether the input is a URL, YouTube link, or plain text.

    Returns:
        str: One of 'youtube', 'url', or 'text'.
    """
    text_stripped = text.strip()
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch',
        r'(?:https?://)?youtu\.be/',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/',
    ]
    for pattern in youtube_patterns:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            return 'youtube'
    if re.match(r'https?://', text_stripped, re.IGNORECASE):
        return 'url'
    return 'text'


# ──────────────────────────────────────────────
# Original /predict endpoint (PRESERVED)
# ──────────────────────────────────────────────
@analysis_bp.route('/predict', methods=['POST'])
# @limiter.limit("30 per minute")
def predict():
    """Predict news authenticity — ORIGINAL API CONTRACT.

    Accepts JSON ``{"text": "..."}`` and returns the Groq prediction result.
    This endpoint is backward-compatible with the original app.py implementation.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text'].strip()
        if len(text) < 10:
            return jsonify({'error': 'Text too short'}), 400

        # --- Groq Prediction ---
        from services.groq_service import predict_news
        result = predict_news(text)

        if result:
            # Save analysis to database
            try:
                report = Report(
                    user_id=_get_user_id(),
                    input_type='text',
                    input_text=text[:5000],
                    prediction=result.get('label', ''),
                    confidence=result.get('confidence'),
                    risk_level=result.get('risk_level', ''),
                    source='Groq (Llama-3.3)',
                )
                report.set_result(result)
                db.session.add(report)
                db.session.commit()
            except Exception:
                db.session.rollback()

            return jsonify({
                'success': True,
                'source': 'Groq (Llama-3.3)',
                'text': text[:150] + '...' if len(text) > 150 else text,
                **result,
                'timestamp': datetime.now().isoformat(),
            })

        return jsonify({
            'error': 'Prediction service unavailable. Please check GROQ_API_KEY.',
        }), 503

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


# ──────────────────────────────────────────────
# Result page (preserved)
# ──────────────────────────────────────────────
@analysis_bp.route('/result', methods=['GET'])
def result_page():
    """Render the analysis result page (preserved from original)."""
    return render_template('result.html')


# ──────────────────────────────────────────────
# Smart Analysis Endpoint
# ──────────────────────────────────────────────
@analysis_bp.route('/api/analyze', methods=['POST'])
# @limiter.limit("20 per minute")
def smart_analyze():
    """Smart analysis endpoint that auto-detects input type and routes accordingly.

    Accepts JSON with:
        - ``text``: The text, URL, or YouTube link to analyze.
        - ``response_lang`` (optional): Preferred response language code.

    Returns:
        JSON object with analysis results.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text'].strip()
        response_lang = data.get('response_lang', Config.DEFAULT_LANGUAGE)

        if not text:
            return jsonify({'error': 'Input text cannot be empty'}), 400

        input_type = _detect_input_type(text)
        result = None

        if input_type == 'youtube':
            from services.youtube_service import analyze_youtube
            result = analyze_youtube(text, response_lang=response_lang)
        elif input_type == 'url':
            from services.url_service import analyze_url
            result = analyze_url(text, response_lang=response_lang)
        else:
            if len(text) < 10:
                return jsonify({'error': 'Text too short (min 10 characters)'}), 400
            from services.groq_service import predict_news
            result = predict_news(text, response_lang=response_lang)

        if not result:
            return jsonify({'error': 'Analysis service unavailable'}), 503

        # Persist report
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type=input_type,
                input_text=text[:5000],
                input_title=result.get('title'),
                prediction=result.get('label', result.get('prediction', '')),
                confidence=result.get('confidence'),
                risk_level=result.get('risk_level', ''),
                response_language=response_lang,
                source=result.get('source', 'Groq (Llama-3.3)'),
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()

            result['report_id'] = report.id
        except Exception:
            db.session.rollback()

        return jsonify({
            'success': True,
            'input_type': input_type,
            **result,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


# ──────────────────────────────────────────────
# Text Analysis Page
# ──────────────────────────────────────────────
@analysis_bp.route('/analyze')
def analyze_text_page():
    """Render the text analysis form page."""
    return render_template(
        'analysis/text.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )
