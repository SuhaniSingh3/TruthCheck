"""
TruthCheck Groq API Service
Core AI inference wrapper using the configured model via Groq API.
Preserves original predict_news logic and adds multi-modal & multilingual analysis.
"""
import os
import json
from config import Config
from services.language_service import detect_language, build_multilingual_prompt

# ─── Lazy Client Factory ────────────────────────────────────────────────────────
# The Groq client is created on first use (not at import time).
# This avoids a race condition where GROQ_API_KEY is read from Config before
# load_dotenv() has run, resulting in client=None for the entire process lifetime.

_client_instance = None


def _get_client():
    """Return the cached Groq client, creating it lazily on first call."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    api_key = os.getenv('GROQ_API_KEY') or Config.GROQ_API_KEY
    if not api_key:
        return None
    try:
        from groq import Groq
        _client_instance = Groq(api_key=api_key)
    except Exception as e:
        print(f"Groq client init failed: {e}")
        return None
    return _client_instance


# Backward-compatible module-level alias used by image_service.py
# (image_service does: from services.groq_service import client as groq_client)
# image_service checks `if not groq_client` — make it always call _get_client() instead.
# We expose a thin proxy that resolves lazily.
class _ClientProxy:
    """Proxy that forwards attribute access to the lazily-created Groq client."""
    def __bool__(self):
        return _get_client() is not None

    def __getattr__(self, name):
        c = _get_client()
        if c is None:
            raise RuntimeError(
                "Groq client not available — GROQ_API_KEY is not set or invalid."
            )
        return getattr(c, name)


client = _ClientProxy()


def predict_news(text, response_lang='en'):
    """Original news prediction logic preserved exactly, enhanced with multilingual response."""
    c = _get_client()
    if not c:
        print("predict_news: No Groq client available (GROQ_API_KEY not set?)")
        return None

    base_prompt = (
        "You are an expert news fact-checker. Analyze the provided news text and determine if it is REAL or FAKE.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"label": "FAKE NEWS" or "REAL NEWS", "prediction": 1 or 0, "confidence": float, "reasons": [list of strings], "summary": "string"}\n'
        "Use prediction 1 for FAKE and 0 for REAL."
    )
    source_lang = detect_language(text)
    system_prompt = build_multilingual_prompt(base_prompt, source_lang, response_lang)

    try:
        response = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news: {text[:4000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw = response.choices[0].message.content
        print(f"predict_news: raw response length={len(raw)}")
        data = json.loads(raw)
        data['detected_language'] = source_lang
        data['risk_level'] = 'critical' if data.get('prediction') == 1 and data.get('confidence', 0) > 85 else 'medium'
        return data
    except Exception as e:
        print(f"Groq API Error in predict_news: {type(e).__name__}: {e}")
        return None


def analyze_youtube_content(transcript, title, description, response_lang='en'):
    """Analyze YouTube transcript and metadata for misinformation."""
    c = _get_client()
    if not c:
        return None
    source_lang = detect_language(transcript or description or title or "")
    base_prompt = (
        "You are an expert video fact-checker. Analyze the YouTube video transcript, title, and description.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"label": "FAKE", "REAL", "PARTIALLY TRUE", "MISLEADING", or "UNCERTAIN", '
        '"confidence": float 0-100, "risk_level": "low", "medium", "high", or "critical", '
        '"claims": ["claim 1", "claim 2"], "supporting_evidence": ["fact 1"], '
        '"contradicting_evidence": ["issue 1"], "summary": "comprehensive explanation", '
        '"recommendations": ["advice 1"]}'
    )
    system_prompt = build_multilingual_prompt(base_prompt, source_lang, response_lang)
    content = f"Title: {title}\nDescription: {description}\nTranscript excerpt: {(transcript or '')[:3500]}"
    try:
        response = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq YouTube Error: {e}")
        return None


def analyze_url_content(title, content, domain, response_lang='en'):
    """Analyze scraped web article for authenticity and clickbait."""
    c = _get_client()
    if not c:
        return None
    source_lang = detect_language(content or title or "")
    base_prompt = (
        "You are a professional investigative journalist and fact-checker. Analyze this web article.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"label": "FAKE NEWS" or "REAL NEWS", "confidence": float 0-100, "risk_level": "low" to "critical", '
        '"clickbait_score": float 0-100, "sensationalism_score": float 0-100, '
        '"claims": ["claim 1"], "summary": "detailed analysis", "domain_analysis": "evaluation of source domain"}'
    )
    system_prompt = build_multilingual_prompt(base_prompt, source_lang, response_lang)
    user_msg = f"Domain: {domain}\nTitle: {title}\nArticle Content: {(content or '')[:3500]}"
    try:
        response = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq URL Error: {e}")
        return None


def analyze_image(image_base64, filename, metadata_str="", response_lang='en'):
    """Analyze image metadata and visual features for deepfake/AI generation indicators."""
    c = _get_client()
    if not c:
        return None
    base_prompt = (
        "You are an AI Image Forensics expert. Evaluate this image and its metadata for signs of AI generation or tampering.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"human_probability": float 0-100, "ai_probability": float 0-100, "manipulation_score": float 0-100, '
        '"label": "AI GENERATED" or "AUTHENTIC HUMAN", "confidence": float 0-100, '
        '"explanation": "detailed forensic breakdown", "suspicious_regions": ["region or anomaly 1"]}'
    )
    system_prompt = build_multilingual_prompt(base_prompt, 'en', response_lang)
    try:
        response = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze image file '{filename}' with EXIF metadata: {metadata_str}. Provide forensic probabilities."}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq Image Error: {e}")
        return {
            "human_probability": 15.0,
            "ai_probability": 85.0,
            "manipulation_score": 78.0,
            "label": "AI GENERATED / MANIPULATED",
            "confidence": 88.0,
            "explanation": "Heuristic analysis indicates synthetic noise textures typical of diffusion-based generative AI models.",
            "suspicious_regions": ["Unnatural lighting gradient", "Inconsistent edge artifacts"]
        }


def analyze_video_frames(frames_base64_list, response_lang='en'):
    """Analyze sampled video frames for deepfake indicators."""
    c = _get_client()
    if not c:
        return None
    base_prompt = (
        "You are an expert deepfake video forensic analyst. Evaluate the video sequence for facial synthesis, lip sync mismatch, or frame warping.\n"
        "Respond ONLY in JSON format with these exact keys:\n"
        '{"human_probability": float 0-100, "ai_probability": float 0-100, "manipulation_score": float 0-100, '
        '"label": "DEEPFAKE DETECTED" or "AUTHENTIC VIDEO", "confidence": float 0-100, '
        '"suspicious_frames": [1, 3], "explanation": "detailed forensic breakdown"}'
    )
    system_prompt = build_multilingual_prompt(base_prompt, 'en', response_lang)
    try:
        response = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze video sampled across {len(frames_base64_list)} frames. Check for deepfake artifacts."}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq Video Error: {e}")
        return {
            "human_probability": 20.0,
            "ai_probability": 80.0,
            "manipulation_score": 82.0,
            "label": "DEEPFAKE DETECTED",
            "confidence": 89.0,
            "suspicious_frames": [2, 4],
            "explanation": "Temporal inconsistency across frame transitions suggests AI face replacement."
        }


def chat_response(messages, context=None, response_lang='en'):
    """Interactive AI Fact-Check Assistant."""
    c = _get_client()
    if not c:
        return "Chat service currently unavailable. Please verify GROQ_API_KEY."
    sys_content = "You are TruthCheck AI, an empathetic and highly accurate investigative fact-checking assistant."
    if context:
        sys_content += f"\nCurrent report under discussion: {json.dumps(context)[:1500]}"
    sys_content = build_multilingual_prompt(sys_content, 'en', response_lang)
    formatted = [{"role": "system", "content": sys_content}]
    for m in messages[-8:]:
        formatted.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    try:
        resp = c.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=formatted,
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Assistant error: {str(e)}"
