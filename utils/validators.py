"""
Input, URL, and File Validators for TruthCheck
"""
import re
from werkzeug.utils import secure_filename
import html

def validate_text_input(text, min_length=10, max_length=50000):
    """Validate raw text input length."""
    if not text or not isinstance(text, str):
        return False, "Input text is required and must be a string."
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return False, f"Text too short (minimum {min_length} characters required)."
    if len(cleaned) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters."
    return True, None

def validate_url_format(url):
    """Validate URL syntax. Returns (is_valid: bool, error_message: str|None)."""
    if not url or not isinstance(url, str):
        return False, "URL is required and must be a string."
    url_pattern = re.compile(
        r'^(https?://)'
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,63})'
        r'(:[0-9]{1,5})?'
        r'(/.*)?$', re.IGNORECASE
    )
    if bool(url_pattern.match(url.strip())):
        return True, None
    return False, "Invalid URL format. Must start with http:// or https://."

def validate_youtube_url(url):
    """Validate if URL points to YouTube. Returns (is_valid: bool, error_message: str|None)."""
    if not url or not isinstance(url, str):
        return False, "URL is required and must be a string."
    youtube_pattern = re.compile(
        r'^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)[A-Za-z0-9_-]{11}(.*)?$',
        re.IGNORECASE
    )
    if bool(youtube_pattern.match(url.strip())):
        return True, None
    return False, "Invalid YouTube URL. Please provide a valid youtube.com or youtu.be link."

def validate_file_upload(file, allowed_extensions, max_size_bytes=104857600):
    """Validate uploaded file extension."""
    if not file or not file.filename:
        return False, "No file selected."
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return False, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}."
    return True, None

def sanitize_text(text):
    """Sanitize user text input."""
    if not text:
        return ""
    return html.escape(text.strip())

def sanitize_filename_custom(filename):
    """Secure filename helper."""
    return secure_filename(filename)
