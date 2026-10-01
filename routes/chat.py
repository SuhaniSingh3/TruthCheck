"""
TruthCheck — Chat Routes
========================
AI-powered conversational assistant for follow-up questions and analysis explanations.
Supports both /api/chat and /chat with rich report/verification context integration.
"""
from flask import Blueprint, request, jsonify
from flask_login import current_user
import json
import logging

from extensions import db
from models.report import Report
from config import Config

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


def _extract_context(context_input):
    """
    Parse context which may be a report ID (int/str), raw JSON string,
    dictionary (active frontend analysis), or plain text.
    """
    if not context_input:
        return None

    if isinstance(context_input, dict):
        return context_input

    if isinstance(context_input, int):
        try:
            report = Report.query.get(context_input)
            return report.to_dict() if report else None
        except Exception:
            return None

    if isinstance(context_input, str):
        context_str = context_input.strip()
        if context_str.isdigit():
            try:
                report = Report.query.get(int(context_str))
                return report.to_dict() if report else None
            except Exception:
                return None
        # Try parsing JSON
        if context_str.startswith('{') and context_str.endswith('}'):
            try:
                return json.loads(context_str)
            except Exception:
                pass
        return context_str

    return None


@chat_bp.route('/api/chat', methods=['POST'])
@chat_bp.route('/chat', methods=['POST'])
def chat():
    """Process a chat message with conversational history and active verification context.

    Accepts JSON or Form data:
        - ``message`` / ``prompt`` / ``query`` / ``text``: The user's query (required).
        - ``context`` / ``verification_context`` (optional): Analysis report object or ID.
        - ``chat_history`` / ``history`` (optional): List of past message objects.
        - ``response_lang`` (optional): Preferred language code.

    Returns:
        JSON ``{"success": true, "response": "..."}``.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}

        message = (
            data.get('message')
            or data.get('prompt')
            or data.get('query')
            or data.get('text')
            or ''
        ).strip()

        if not message:
            return jsonify({'error': 'Please provide a message or question.'}), 400

        raw_context = data.get('context') or data.get('verification_context') or data.get('active_analysis_data')
        context_data = _extract_context(raw_context)

        chat_history = data.get('chat_history') or data.get('history') or []
        response_lang = data.get('response_lang', getattr(Config, 'DEFAULT_LANGUAGE', 'en'))

        from services.chat_service import process_message
        response_text = process_message(
            message=message,
            chat_history=chat_history,
            context=context_data,
            response_lang=response_lang,
        )

        if not response_text:
            return jsonify({'error': 'Chat service is currently unavailable. Please try again.'}), 503

        return jsonify({
            'success': True,
            'response': response_text,
        }), 200

    except Exception as e:
        logger.exception("Error handling chat request: %s", e)
        return jsonify({'error': f'Chat service error: {str(e)}'}), 500


@chat_bp.route('/api/chat/suggestions', methods=['GET'])
def chat_suggestions():
    """Return a list of suggested follow-up questions.

    Query params:
        - ``context`` (optional): Report ID to base suggestions on.

    Returns:
        JSON ``{"suggestions": [...]}``.
    """
    try:
        context_id = request.args.get('context')
        suggestions = [
            "Why was this verdict chosen?",
            "Explain the uncertainty or confidence score.",
            "What are the main red flags in this content?",
            "How can I verify this claim independently?",
            "What sources corroborate or debunk this?"
        ]

        if context_id:
            try:
                report = Report.query.get(int(context_id))
                if report:
                    if report.is_fake:
                        suggestions = [
                            "Why was this flagged as potentially fake?",
                            "What are the main manipulation indicators found?",
                            "Explain the reasons behind this fake news classification.",
                            "Where can I find verified facts on this topic?"
                        ]
                    else:
                        suggestions = [
                            "What evidence makes this news credible?",
                            "Explain the confidence score for this real news verdict.",
                            "Are there any caveats or nuance to consider?",
                            "What reputable outlets have reported this?"
                        ]
            except Exception:
                pass

        return jsonify({
            'success': True,
            'suggestions': suggestions,
        }), 200

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
