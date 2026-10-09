"""
Vercel Serverless Entry Point for TruthCheck
============================================
Exports the Flask WSGI `app` callable for @vercel/python builder.
"""
import sys
import os

# Ensure the root project directory is on the Python module search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app  # noqa: F401 (WSGI handler exported for Vercel)

# Guarantee table initialization on Vercel serverless cold start
try:
    from extensions import db
    with app.app_context():
        db.create_all()
except Exception:
    pass
