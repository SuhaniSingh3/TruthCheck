"""
YouTube Video Verification Service for TruthCheck
Extracts transcripts and metadata, and performs multi-claim fact verification.
"""
import re
from youtube_transcript_api import YouTubeTranscriptApi
from services.groq_service import analyze_youtube_content

def validate_youtube_url(url):
    """Extract YouTube video ID from URL."""
    if not url:
        return None
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def get_transcript(video_id, preferred_lang='en'):
    """Fetch transcript text for YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_lang, 'en', 'hi', 'es', 'fr', 'de'])
        text = " ".join([t['text'] for t in transcript_list])
        return text
    except Exception:
        return None

def analyze_youtube(url, response_lang='en'):
    """Full YouTube verification pipeline."""
    video_id = validate_youtube_url(url)
    if not video_id:
        return {"error": "Invalid YouTube URL provided."}

    transcript = get_transcript(video_id, preferred_lang=response_lang)
    title = f"YouTube Video ID: {video_id}"
    description = "Extracted from video link."
    if not transcript:
        description += " (No automated closed captions available; fallback heuristic analysis applied)."

    result = analyze_youtube_content(transcript, title, description, response_lang=response_lang)
    if result:
        result['video_id'] = video_id
        result['embed_url'] = f"https://www.youtube.com/embed/{video_id}"
        result['transcript_preview'] = (transcript or "No transcript available")[:500]
    return result
