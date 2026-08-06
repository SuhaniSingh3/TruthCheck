"""
Security and rate-limiting helpers for TruthCheck
"""
import os
from flask import request

MAGIC_BYTES = {
    'jpg': b'\xff\xd8\xff',
    'jpeg': b'\xff\xd8\xff',
    'png': b'\x89PNG\r\n\x1a\n',
    'webp': b'RIFF',
}

def check_file_magic_bytes(filepath, expected_ext):
    """Check magic bytes header of file to verify actual format."""
    if not os.path.exists(filepath):
        return False
    expected_ext = expected_ext.lower()
    magic = MAGIC_BYTES.get(expected_ext)
    if not magic:
        return True
    try:
        with open(filepath, 'rb') as f:
            header = f.read(12)
        if expected_ext == 'webp':
            return header.startswith(b'RIFF') and b'WEBP' in header
        return header.startswith(magic)
    except Exception:
        return False

def rate_limit_key():
    """Return key for IP rate limiting."""
    return request.remote_addr or '127.0.0.1'

def is_safe_url(url):
    """Basic safety check against private/local networks."""
    lower = url.lower()
    unsafe = ['localhost', '127.0.0.1', '0.0.0.0', '192.168.', '10.', '172.16.', 'file://']
    for u in unsafe:
        if u in lower:
            return False
    return True
