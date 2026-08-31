"""
Report Generation Service for TruthCheck
Generates PDF, CSV, and printable HTML analysis summaries.

NOTE: reportlab is an optional dependency.
If not installed, PDF generation returns None and the route returns a 503.
All other report functions (CSV, HTML) work without reportlab.
"""
import io
import csv
import logging

logger = logging.getLogger(__name__)


def generate_pdf(report_dict):
    """Generate simple professional PDF report in bytes.

    Returns None if reportlab is not installed (non-fatal on Vercel).
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        logger.warning("reportlab not installed — PDF generation unavailable.")
        return None

    try:
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, 750, "TruthCheck — Enterprise AI Misinformation Report")
        p.setFont("Helvetica", 12)
        p.drawString(50, 710, f"Prediction: {report_dict.get('prediction', 'N/A')}")
        p.drawString(50, 690, f"Confidence: {report_dict.get('confidence', '--')}%")
        p.drawString(50, 670, f"Input Type: {report_dict.get('input_type', 'text').upper()}")
        p.drawString(50, 650, f"Source: {report_dict.get('source', 'Groq Llama-3.3')}")
        p.drawString(50, 610, "Summary:")
        summary = report_dict.get('result', {}).get('summary', 'No summary available.')
        p.drawString(50, 590, str(summary)[:80])
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        return None


def generate_csv(report_dict):
    """Generate CSV string of report."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Report ID', 'Input Type', 'Prediction', 'Confidence', 'Created At'])
    writer.writerow([
        report_dict.get('id'),
        report_dict.get('input_type'),
        report_dict.get('prediction'),
        report_dict.get('confidence'),
        report_dict.get('created_at')
    ])
    return output.getvalue()


def generate_print_html(report_dict):
    """Generate clean printable HTML summary."""
    return f"""
    <html><head><title>Print Report - TruthCheck</title></head>
    <body style="font-family: sans-serif; padding: 40px;">
        <h1>TruthCheck Analysis Report</h1>
        <h3>Verdict: {report_dict.get('prediction')} ({report_dict.get('confidence')}%)</h3>
        <p><strong>Input Type:</strong> {report_dict.get('input_type')}</p>
        <p><strong>Summary:</strong> {report_dict.get('result', {}).get('summary', '')}</p>
    </body></html>
    """
