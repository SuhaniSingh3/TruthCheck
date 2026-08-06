"""
Language detection and multilingual translation prompt builder for TruthCheck.
Supports 30+ national and international languages including English, Hindi, Spanish, Arabic, Chinese, etc.
"""
from langdetect import detect, DetectorFactory
from config import Config

DetectorFactory.seed = 0

def detect_language(text):
    """Detect ISO language code from input text."""
    if not text or len(text.strip()) < 5:
        return Config.DEFAULT_LANGUAGE
    try:
        lang = detect(text)
        return lang if lang in Config.SUPPORTED_LANGUAGES else Config.DEFAULT_LANGUAGE
    except Exception:
        return Config.DEFAULT_LANGUAGE

def get_language_name(code):
    """Return full language display name."""
    return Config.SUPPORTED_LANGUAGES.get(code, 'English')

def build_multilingual_prompt(base_prompt, source_lang, response_lang):
    """Inject multilingual analysis and response instructions into AI prompt."""
    target_lang_name = get_language_name(response_lang)
    source_lang_name = get_language_name(source_lang)
    instruction = (
        f"\n\nMULTILINGUAL INSTRUCTION:\n"
        f"- The input content may be in {source_lang_name} or another language.\n"
        f"- You MUST provide your final JSON values ('reasons', 'summary', 'explanation', 'claims') written in {target_lang_name}.\n"
        f"- Ensure cultural nuances, local idioms, and regional fact-checking contexts relevant to the source language are preserved."
    )
    return base_prompt + instruction
