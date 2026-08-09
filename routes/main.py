"""
TruthCheck — Main Routes
Dashboard and landing page endpoints.
"""
from flask import Blueprint, render_template
from flask_login import current_user

from extensions import db
from models.report import Report

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def dashboard():
    """Render the main dashboard with analysis statistics.

    If the user is authenticated, statistics are scoped to their own
    reports.  Otherwise, show global / session-level stats.
    Database queries fail silently when DB is unavailable (Vercel without DB URL).
    """
    stats = {
        'total_analyses': 0,
        'url_analyses': 0,
        'video_analyses': 0,
        'image_analyses': 0,
        'fake_detected': 0,
        'real_detected': 0,
        'ai_images': 0,
        'deepfakes': 0,
    }
    total = 0
    fake_count = 0
    real_count = 0
    recent_reports = []

    try:
        if current_user.is_authenticated:
            base_query = Report.query.filter_by(user_id=current_user.id)
        else:
            base_query = Report.query

        total = base_query.count()
        fake_count = base_query.filter(
            Report.prediction.ilike('%fake%')
        ).count()
        real_count = base_query.filter(
            Report.prediction.ilike('%real%')
        ).count()
        recent_reports = (
            base_query
            .order_by(Report.created_at.desc())
            .limit(10)
            .all()
        )
        stats = {
            'total_analyses': total,
            'url_analyses': base_query.filter_by(input_type='url').count() if total else 0,
            'video_analyses': base_query.filter_by(input_type='youtube').count() if total else 0,
            'image_analyses': base_query.filter_by(input_type='image').count() if total else 0,
            'fake_detected': fake_count,
            'real_detected': real_count,
            'ai_images': base_query.filter_by(input_type='image').filter(
                Report.prediction.ilike('%AI%')).count() if total else 0,
            'deepfakes': base_query.filter_by(input_type='video').filter(
                Report.prediction.ilike('%DEEPFAKE%')).count() if total else 0,
        }
    except Exception:
        # DB unavailable (Vercel without DATABASE_URL) — render with zero stats
        pass

    return render_template(
        'dashboard.html',
        stats=stats,
        total=total,
        fake_count=fake_count,
        real_count=real_count,
        recent_reports=recent_reports,
    )


@main_bp.route('/landing')
def landing():
    """Render the original landing page (index.html), preserved as-is."""
    return render_template('index.html')
