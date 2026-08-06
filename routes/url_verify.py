"""
TruthCheck — URL Verification Routes
Analyze web articles/URLs for misinformation.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from datetime import datetime

from extensions import db
from models.report import Report
from config import Config

url_bp = Blueprint('url', __name__)


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


@url_bp.route('/url-verify')
def url_verify_page():
    """Render the URL verification form page."""
    return render_template(
        'analysis/url.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )


@url_bp.route('/api/url/analyze', methods=['POST'])
# @limiter.limit("15 per minute")
def analyze_url_content():
    """Analyze a web article/URL for misinformation.

    Accepts JSON:
        - ``url``: The web page URL to verify.
        - ``response_lang`` (optional): Preferred response language.

    Returns:
        JSON object with the analysis result.
    """
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing "url" field'}), 400

        url = data['url'].strip()
        response_lang = data.get('response_lang', Config.DEFAULT_LANGUAGE)

        if not url:
            return jsonify({'error': 'URL cannot be empty'}), 400

        # Validate URL format
        from utils.validators import validate_url_format
        is_valid, error_msg = validate_url_format(url)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Analyze
        from services.url_service import analyze_url
        result = analyze_url(url, response_lang=response_lang)

        if not result:
            return jsonify({'error': 'URL analysis service unavailable'}), 503

        # Persist report
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type='url',
                input_text=url,
                input_title=result.get('title'),
                prediction=result.get('label', result.get('prediction', '')),
                confidence=result.get('confidence'),
                risk_level=result.get('risk_level', ''),
                response_language=response_lang,
                detected_language=result.get('detected_language'),
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
            'input_type': 'url',
            **result,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
