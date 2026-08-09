"""
TruthCheck Configuration
Centralized application settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Vercel Detection ─────────────────────────────────────────────────────────
# Vercel sets the VERCEL environment variable automatically.
# We also detect read-only filesystem by checking if /tmp is writable.
IS_VERCEL = bool(os.getenv('VERCEL') or os.getenv('VERCEL_ENV'))


def _get_upload_folder():
    """
    Return a writable temp directory for uploaded images.
    On Vercel, the project root (/var/task) is read-only; only /tmp is writable.
    Locally, we use the project's 'uploads/' folder.
    """
    if IS_VERCEL:
        return '/tmp'
    # Local: use project uploads/ directory (created at startup)
    return os.path.join(BASE_DIR, 'uploads')


class Config:
    """Application configuration loaded from .env with sensible defaults."""

    # --- Flask Core ---
    SECRET_KEY = os.getenv('SECRET_KEY', 'truthcheck-dev-secret-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')

    # --- Database ---
    # LOCAL:  SQLite (auto-created in instance/ directory)
    # VERCEL with DATABASE_URL: PostgreSQL (set DATABASE_URL env var in Vercel dashboard)
    # VERCEL without DATABASE_URL: in-memory SQLite (DB features disabled, app still boots)
    _raw_db_url = os.getenv('DATABASE_URL', '')
    if _raw_db_url.startswith('postgres://'):
        # SQLAlchemy 1.4+ requires 'postgresql://' not 'postgres://'
        _raw_db_url = _raw_db_url.replace('postgres://', 'postgresql://', 1)

    if _raw_db_url:
        # Explicit DATABASE_URL (PostgreSQL on Vercel, or custom local)
        SQLALCHEMY_DATABASE_URI = _raw_db_url
    elif IS_VERCEL:
        # Vercel without DATABASE_URL — use in-memory SQLite so app boots.
        # db.create_all() is suppressed in app.py for this case.
        SQLALCHEMY_DATABASE_URI = 'sqlite://'   # ":memory:" — never writes to disk
    else:
        # Local development — persistent SQLite in instance/
        SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "instance", "truthcheck.db")}'

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        # Prevent stale connections on serverless cold starts
        'pool_pre_ping': True,
        'pool_recycle': 280,
    }

    # --- Groq AI API ---
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    GROQ_VISION_MODEL = os.getenv('GROQ_VISION_MODEL', 'llama-3.3-70b-versatile')
    GROQ_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
    GROQ_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '4096'))

    # --- File Uploads ---
    # Always /tmp on Vercel; 'uploads/' directory locally
    UPLOAD_FOLDER = _get_upload_folder()
    # Hard limit 20 MB for image uploads (matches IMAGE_VERIFY_MAX_SIZE)
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    ALLOWED_TEXT_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

    # --- AI Image Verification ---
    IMAGE_VERIFY_MAX_SIZE = 20 * 1024 * 1024   # 20 MB hard limit

    # Weighted pipeline scores — weights must sum to 1.0
    # Rationale:
    #   ela (0.22)       → Strong direct evidence of re-compression / editing
    #   noise (0.15)     → AI hallmark but also affected by compression
    #   frequency (0.15) → GAN upsampling artifacts
    #   jpeg (0.10)      → Compression block artifacts (JPEG only)
    #   metadata (0.13)  → Supporting evidence; not definitive alone
    #   copy_move (0.10) → Localized cloning evidence
    #   groq_vision (0.15) → LLM reasoning weight (when available)
    IMAGE_VERIFY_WEIGHTS = {
        'ela':          0.22,
        'noise':        0.15,
        'frequency':    0.15,
        'jpeg':         0.10,
        'metadata':     0.13,
        'copy_move':    0.10,
        'groq_vision':  0.15,
    }

    # --- Rate Limiting ---
    RATELIMIT_DEFAULT = os.getenv('RATE_LIMIT', '60 per minute')
    RATELIMIT_STORAGE_URI = 'memory://'

    # --- Multilingual Settings ---
    DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'en')
    SUPPORTED_LANGUAGES = {
        'en': 'English', 'hi': 'हिन्दी (Hindi)', 'es': 'Español (Spanish)',
        'fr': 'Français (French)', 'de': 'Deutsch (German)', 'ar': 'العربية (Arabic)',
        'zh': '中文 (Chinese)', 'ja': '日本語 (Japanese)', 'ko': '한국어 (Korean)',
        'pt': 'Português (Portuguese)', 'ru': 'Русский (Russian)', 'it': 'Italiano (Italian)',
        'bn': 'বাংলা (Bengali)', 'ta': 'தமிழ் (Tamil)', 'te': 'తెలుగు (Telugu)',
        'ur': 'اردو (Urdu)', 'mr': 'मराठी (Marathi)', 'gu': 'ગુજરાતી (Gujarati)',
        'kn': 'ಕನ್ನಡ (Kannada)', 'ml': 'മലയാളം (Malayalam)', 'pa': 'ਪੰਜਾਬੀ (Punjabi)',
        'th': 'ไทย (Thai)', 'vi': 'Tiếng Việt (Vietnamese)', 'tr': 'Türkçe (Turkish)',
        'pl': 'Polski (Polish)', 'nl': 'Nederlands (Dutch)', 'sv': 'Svenska (Swedish)',
        'id': 'Bahasa Indonesia', 'ms': 'Bahasa Melayu (Malay)', 'uk': 'Українська (Ukrainian)',
    }
