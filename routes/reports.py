"""
TruthCheck — Report Export Routes
PDF, CSV, print, and JSON export of individual analysis reports.
"""
from flask import Blueprint, jsonify, make_response
from flask_login import current_user

from extensions import db
from models.report import Report

reports_bp = Blueprint('reports', __name__)


def _get_report_or_404(report_id):
    """Fetch a report by ID, scoping to the current user if authenticated.

    Returns:
        Tuple of (report, error_response). If report is None, return error_response
        directly.
    """
    if current_user.is_authenticated:
        report = Report.query.filter_by(id=report_id, user_id=current_user.id).first()
    else:
        report = Report.query.get(report_id)

    if not report:
        return None, (jsonify({'error': 'Report not found'}), 404)
    return report, None


@reports_bp.route('/api/report/<int:id>/pdf')
# @limiter.limit("10 per minute")
def download_pdf(id):
    """Generate and return a PDF download for a specific report.

    Args:
        id: The report primary key.

    Returns:
        PDF file as attachment.
    """
    try:
        report, error = _get_report_or_404(id)
        if error:
            return error

        from services.report_service import generate_pdf
        pdf_bytes = generate_pdf(report)

        if not pdf_bytes:
            return jsonify({'error': 'PDF generation failed'}), 500

        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = (
            f'attachment; filename=truthcheck_report_{id}.pdf'
        )
        return response

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@reports_bp.route('/api/report/<int:id>/csv')
# @limiter.limit("10 per minute")
def download_csv(id):
    """Generate and return a CSV download for a specific report.

    Args:
        id: The report primary key.

    Returns:
        CSV file as attachment.
    """
    try:
        report, error = _get_report_or_404(id)
        if error:
            return error

        from services.report_service import generate_csv
        csv_content = generate_csv(report)

        if not csv_content:
            return jsonify({'error': 'CSV generation failed'}), 500

        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = (
            f'attachment; filename=truthcheck_report_{id}.csv'
        )
        return response

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@reports_bp.route('/api/report/<int:id>/print')
# @limiter.limit("10 per minute")
def print_report(id):
    """Render a printer-friendly HTML page for a specific report.

    Args:
        id: The report primary key.

    Returns:
        HTML page optimized for printing.
    """
    try:
        report, error = _get_report_or_404(id)
        if error:
            return error

        from services.report_service import generate_print_html
        html_content = generate_print_html(report)

        if not html_content:
            return jsonify({'error': 'Print page generation failed'}), 500

        return html_content

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@reports_bp.route('/api/report/<int:id>')
# @limiter.limit("30 per minute")
def get_report_json(id):
    """Return the full report data as JSON.

    Args:
        id: The report primary key.

    Returns:
        JSON representation of the report.
    """
    try:
        report, error = _get_report_or_404(id)
        if error:
            return error

        return jsonify({
            'success': True,
            'report': report.to_dict(),
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
