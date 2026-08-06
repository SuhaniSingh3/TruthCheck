"""
Interactive Fact-Checking AI Chat Assistant Service
"""
from services.groq_service import chat_response as groq_chat

def process_message(user_message, chat_history=None, context=None, response_lang='en'):
    """Process user chat input and return AI assistant response."""
    history = chat_history or []
    history.append({"role": "user", "content": user_message})
    response = groq_chat(history, context=context, response_lang=response_lang)
    return response
