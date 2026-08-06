"""
TruthCheck Configuration
Centralized application settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Application configuration loaded from .env with sensible defaults."""

    # --- Flask Core ---
    SECRET_KEY = os.getenv('SECRET_KEY', 'truthcheck-dev-secret-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')

    # --- Database (SQLite for local, swap to PostgreSQL for production) ---
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "instance", "truthcheck.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Groq AI API ---
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    GROQ_VISION_MODEL = os.getenv('GROQ_VISION_MODEL', 'llama-3.3-70b-versatile')
    GROQ_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
    GROQ_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '4096'))

    # --- File Uploads ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload size
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    ALLOWED_TEXT_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

    # --- AI Image Verification ---
    IMAGE_VERIFY_MAX_SIZE = 20 * 1024 * 1024   # 20 MB hard limit for image uploads
    # Weighted pipeline: weights must sum to 1.0
    IMAGE_VERIFY_WEIGHTS = {
        'ela':          0.20,  # Error Level Analysis
        'noise':        0.15,  # Noise Pattern Fingerprinting
        'frequency':    0.15,  # FFT Frequency Domain
        'jpeg':         0.10,  # JPEG Artifact Scoring
        'metadata':     0.15,  # EXIF Metadata Analysis
        'ai_detection': 0.25,  # HuggingFace CNN (if available)
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
