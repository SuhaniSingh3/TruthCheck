"""
TruthCheck — YouTube Analysis Routes
Analyze YouTube videos for misinformation.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from datetime import datetime

from extensions import db
from models.report import Report
from config import Config

youtube_bp = Blueprint('youtube', __name__)


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


@youtube_bp.route('/youtube')
def youtube_page():
    """Render the YouTube analysis form page."""
    return render_template(
        'analysis/youtube.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )


@youtube_bp.route('/api/youtube/analyze', methods=['POST'])
# @limiter.limit("10 per minute")
def analyze_youtube_video():
    """Analyze a YouTube video for misinformation.

    Accepts JSON:
        - ``url``: YouTube video URL.
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

        # Validate YouTube URL format
        from utils.validators import validate_youtube_url
        is_valid, error_msg = validate_youtube_url(url)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Analyze
        from services.youtube_service import analyze_youtube
        result = analyze_youtube(url, response_lang=response_lang)

        if not result:
            return jsonify({'error': 'YouTube analysis service unavailable'}), 503

        # Persist report
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type='youtube',
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
            'input_type': 'youtube',
            **result,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
