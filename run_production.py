"""Run the Flask app using Waitress for production-like serving.

Usage:
    python run_production.py

This binds to 0.0.0.0:8080 by default to avoid conflicting with the
development server on port 5000.
"""
from wsgi import app
from waitress import serve


def main():
    serve(app, host='0.0.0.0', port=8080, threads=8)


if __name__ == '__main__':
    main()
