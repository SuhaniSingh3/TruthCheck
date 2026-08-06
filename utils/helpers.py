"""
Formatting and presentation helper functions for TruthCheck
"""
from datetime import datetime

def format_confidence(confidence):
    """Format float confidence into percentage string."""
    if confidence is None:
        return "--%"
    return f"{float(confidence):.1f}%"

def format_timestamp(dt):
    """Format timestamp into readable string."""
    if not dt:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except Exception:
            return dt
    return dt.strftime("%B %d, %Y at %I:%M %p")

def truncate_text(text, max_length=150):
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return (text[:max_length] + '...') if len(text) > max_length else text

def calculate_risk_level(confidence, prediction):
    """Derive risk category based on confidence and prediction string."""
    pred = (prediction or '').upper()
    conf = float(confidence or 50)
    if 'FAKE' in pred:
        if conf >= 85:
            return 'critical'
        elif conf >= 70:
            return 'high'
        return 'medium'
    elif 'REAL' in pred:
        if conf >= 85:
            return 'low'
        return 'medium'
    return 'medium'

def get_input_type_icon(input_type):
    """Return icon character for input type."""
    icons = {
        'text': '📝',
        'url': '🔗',
        'youtube': '▶️',
        'image': '🖼️',
        'video': '🎬',
    }
    return icons.get((input_type or '').lower(), '📝')

def get_verdict_color(prediction):
    """Return hex color code for prediction label."""
    pred = (prediction or '').upper()
    if 'FAKE' in pred:
        return '#ef4444'
    if 'REAL' in pred:
        return '#10b981'
    return '#f59e0b'
