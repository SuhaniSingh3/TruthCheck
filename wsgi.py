"""WSGI entrypoint for production servers.

Expose the Flask application as the module-level variable `app` so WSGI
servers (Waitress, Gunicorn) can import `wsgi:app`.
"""
from app import create_app

app = create_app()
