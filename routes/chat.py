"""
TruthCheck — Chat Routes
AI-powered conversational assistant for follow-up questions about analyses.
"""
from flask import Blueprint, request, jsonify
from flask_login import current_user

from extensions import db
from models.report import Report
from config import Config

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/api/chat', methods=['POST'])
# @limiter.limit("30 per minute")
def chat():
    """Process a chat message, optionally in the context of a previous report.

    Accepts JSON:
        - ``message``: The user's chat message (required).
        - ``context`` (optional): Report ID to load as conversation context.
        - ``response_lang`` (optional): Preferred response language.

    Returns:
        JSON ``{"response": "..."}``.
    """
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" field'}), 400

        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        context_id = data.get('context')
        response_lang = data.get('response_lang', Config.DEFAULT_LANGUAGE)

        # Load context from a previous report if provided
        context_data = None
        if context_id:
            try:
                report = Report.query.get(int(context_id))
                if report:
                    context_data = report.to_dict()
            except (ValueError, TypeError):
                pass  # Ignore invalid context IDs

        from services.chat_service import process_message
        response_text = process_message(
            message=message,
            context=context_data,
            response_lang=response_lang,
        )

        if response_text is None:
            return jsonify({'error': 'Chat service unavailable'}), 503

        return jsonify({
            'success': True,
            'response': response_text,
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@chat_bp.route('/api/chat/suggestions', methods=['GET'])
# @limiter.limit("30 per minute")
def chat_suggestions():
    """Return a list of suggested follow-up questions.

    Query params:
        - ``context`` (optional): Report ID to base suggestions on.

    Returns:
        JSON ``{"suggestions": [...]}``.
    """
    try:
        context_id = request.args.get('context')

        # Default suggestions
        suggestions = [
            "What are the key indicators of fake news?",
            "How reliable is this source?",
            "Can you fact-check this claim?",
            "What should I look for when verifying news?",
            "Explain the confidence score in more detail.",
        ]

        # Contextual suggestions based on a specific report
        if context_id:
            try:
                report = Report.query.get(int(context_id))
                if report:
                    context_suggestions = []
                    if report.is_fake:
                        context_suggestions = [
                            "Why was this flagged as potentially fake?",
                            "What are the main red flags in this content?",
                            "Where can I find the original source for this?",
                            "What are common misinformation tactics used here?",
                            "How can I verify this information independently?",
                        ]
                    else:
                        context_suggestions = [
                            "What makes this source credible?",
                            "Are there any caveats to consider?",
                            "How was the authenticity determined?",
                            "Can you provide more details about the analysis?",
                            "What other sources corroborate this information?",
                        ]
                    suggestions = context_suggestions
            except (ValueError, TypeError):
                pass  # Use default suggestions

        return jsonify({
            'success': True,
            'suggestions': suggestions,
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
