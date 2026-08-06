"""
TruthCheck — Video Detection Routes
Analyze uploaded videos for misinformation / manipulated content.
"""
import os
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from werkzeug.utils import secure_filename
from datetime import datetime

from extensions import db
from models.report import Report
from config import Config

video_bp = Blueprint('video', __name__)


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


def _allowed_video(filename):
    """Check whether the file extension is in the allowed video set."""
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_VIDEO_EXTENSIONS
    )


@video_bp.route('/video-detect')
def video_detect_page():
    """Render the video analysis form page."""
    return render_template(
        'analysis/video.html',
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )


@video_bp.route('/api/video/analyze', methods=['POST'])
# @limiter.limit("5 per minute")
def analyze_video_upload():
    """Analyze an uploaded video for misinformation.

    Expects multipart/form-data with:
        - ``file``: The video file to analyze.
        - ``response_lang`` (optional): Preferred response language.

    Returns:
        JSON object with the analysis result.
    """
    saved_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not _allowed_video(file.filename):
            allowed = ', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)
            return jsonify({
                'error': f'Invalid file type. Allowed: {allowed}',
            }), 400

        response_lang = request.form.get('response_lang', Config.DEFAULT_LANGUAGE)

        # Validate file using shared validator
        from utils.validators import validate_file_upload
        is_valid, error_msg = validate_file_upload(file, allowed_extensions=Config.ALLOWED_VIDEO_EXTENSIONS)
        if not is_valid:
            return jsonify({'error': error_msg}), 400

        # Save file to uploads/
        filename = secure_filename(file.filename)
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_name = timestamp_prefix + filename
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        saved_path = os.path.join(Config.UPLOAD_FOLDER, safe_name)
        file.save(saved_path)

        # Analyze
        from services.video_service import analyze_video
        result = analyze_video(saved_path, response_lang=response_lang)

        if not result:
            return jsonify({'error': 'Video analysis service unavailable'}), 503

        # Persist report
        try:
            report = Report(
                user_id=_get_user_id(),
                input_type='video',
                input_text=None,
                input_filename=filename,
                input_title=result.get('title', filename),
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
            'input_type': 'video',
            'filename': filename,
            **result,
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
    finally:
        # Cleanup uploaded file
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
