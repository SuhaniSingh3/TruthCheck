"""
TruthCheck Interactive Fact-Checking AI Chat Assistant Service
==============================================================
Handles conversation turn logic, context injection, and Groq LLM inference.
"""
import logging
from services.groq_service import chat_response as groq_chat

logger = logging.getLogger(__name__)


def process_message(message=None, user_message=None, chat_history=None, context=None, response_lang='en'):
    """
    Process user chat input with conversational and report context.

    Args:
        message (str, optional): User's prompt/query.
        user_message (str, optional): Alias for message.
        chat_history (list, optional): Previous chat messages [{"role": "user"|"assistant", "content": "..."}].
        context (dict|str|int, optional): Active verification report/analysis context.
        response_lang (str, optional): Preferred language code.

    Returns:
        str: AI response text.
    """
    query = (message or user_message or "").strip()
    if not query:
        return "Please ask a question or enter a topic you would like me to explain."

    history = list(chat_history or [])
    history.append({"role": "user", "content": query})

    try:
        response = groq_chat(history, context=context, response_lang=response_lang)
        return response
    except Exception as e:
        logger.exception("Error in process_message: %s", e)
        return f"I encountered an error processing your query: {str(e)}"
