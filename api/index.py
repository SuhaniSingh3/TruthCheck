"""
Vercel serverless entry point for TruthCheck.

Vercel Python runtime expects a WSGI-compatible `app` object at `api/index.py`.
This file simply imports the Flask application created by the application factory
in app.py — it does NOT duplicate any application logic.

Why api/index.py?
  Vercel's Python builder discovers serverless functions under the `api/`
  directory by convention. A function at `api/index.py` is mapped to the
  path `/api/index`, but when combined with a catch-all rewrite rule in
  vercel.json that sends every request to this function, the Flask router
  inside the function handles all URL dispatching correctly.
"""

import sys
import os

# Ensure the project root is on the Python path so that `app`, `config`,
# `routes`, `services`, etc. can all be imported without modification.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask `app` object created by the application factory.
# `app` is the WSGI callable that Vercel's Python runtime will invoke.
from app import app  # noqa: F401  (re-exported as the WSGI handler)
