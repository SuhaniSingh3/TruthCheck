Production deployment notes
==========================

Run with Waitress (recommended for Windows / simple deployments):

```bash
python -m pip install -r requirements.txt
python run_production.py
```

By default the production runner binds to `0.0.0.0:8080`.

For containerized deployment, use `wsgi:app` as the WSGI entrypoint and
start Waitress or Gunicorn accordingly.
