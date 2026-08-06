"""
TruthCheck — History Routes
Browse, search, filter, and manage past analysis reports.
"""
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user

from extensions import db
from models.report import Report

history_bp = Blueprint('history', __name__)


@history_bp.route('/history')
def history_page():
    """Render the analysis history page."""
    return render_template('history.html')


@history_bp.route('/api/history', methods=['GET'])
# @limiter.limit("60 per minute")
def get_history():
    """Return paginated, filterable analysis history as JSON.

    Query params:
        - ``search``: Filter reports whose input_text or input_title contains this string.
        - ``type``: Filter by input_type (text, url, youtube, image, video).
        - ``sort``: Sort order — ``date_desc`` (default), ``date_asc``, or ``confidence``.
        - ``page``: Page number (default 1).
        - ``per_page``: Items per page (default 20, max 100).
        - ``favorites_only``: If ``true``, show only favorites.

    Returns:
        JSON ``{"reports": [...], "total": int, "page": int, "pages": int}``.
    """
    try:
        # Scope to authenticated user or all reports
        if current_user.is_authenticated:
            query = Report.query.filter_by(user_id=current_user.id)
        else:
            query = Report.query.filter_by(user_id=None)

        # --- Filters ---
        search = request.args.get('search', '').strip()
        if search:
            search_filter = f'%{search}%'
            query = query.filter(
                db.or_(
                    Report.input_text.ilike(search_filter),
                    Report.input_title.ilike(search_filter),
                    Report.prediction.ilike(search_filter),
                )
            )

        input_type = request.args.get('type', '').strip().lower()
        if input_type and input_type in ('text', 'url', 'youtube', 'image', 'video'):
            query = query.filter_by(input_type=input_type)

        favorites_only = request.args.get('favorites_only', '').lower() == 'true'
        if favorites_only:
            query = query.filter_by(is_favorite=True)

        # --- Sorting ---
        sort = request.args.get('sort', 'date_desc').strip().lower()
        if sort == 'date_asc':
            query = query.order_by(Report.created_at.asc())
        elif sort == 'confidence':
            query = query.order_by(Report.confidence.desc().nullslast())
        else:  # date_desc (default)
            query = query.order_by(Report.created_at.desc())

        # --- Pagination ---
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return jsonify({
            'success': True,
            'reports': [r.to_dict() for r in pagination.items],
            'total': pagination.total,
            'page': pagination.page,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev,
        })

    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@history_bp.route('/api/history/<int:id>', methods=['DELETE'])
def delete_report(id):
    """Delete a specific analysis report.

    Args:
        id: The report primary key.

    Returns:
        JSON confirmation or 404 if not found.
    """
    try:
        report = Report.query.filter_by(id=id, user_id=current_user.id).first()
        if not report:
            return jsonify({'error': 'Report not found'}), 404

        db.session.delete(report)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Report deleted'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@history_bp.route('/api/history/<int:id>/favorite', methods=['PUT'])
def toggle_favorite(id):
    """Toggle the favorite status of a report.

    Args:
        id: The report primary key.

    Returns:
        JSON with the new ``is_favorite`` value.
    """
    try:
        report = Report.query.filter_by(id=id, user_id=current_user.id).first()
        if not report:
            return jsonify({'error': 'Report not found'}), 404

        report.is_favorite = not report.is_favorite
        db.session.commit()

        return jsonify({
            'success': True,
            'is_favorite': report.is_favorite,
            'message': 'Added to favorites' if report.is_favorite else 'Removed from favorites',
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


@history_bp.route('/api/history/<int:id>/reanalyze', methods=['POST'])
def reanalyze_report(id):
    """Re-run analysis on a previously stored input.

    The original report is preserved; a new report is created with fresh results.

    Args:
        id: The report primary key.

    Returns:
        JSON with the new analysis result and new report ID.
    """
    try:
        original = Report.query.filter_by(id=id, user_id=current_user.id).first()
        if not original:
            return jsonify({'error': 'Report not found'}), 404

        input_type = original.input_type
        result = None

        if input_type == 'text':
            if not original.input_text:
                return jsonify({'error': 'No input text available for re-analysis'}), 400
            from services.groq_service import predict_news
            result = predict_news(
                original.input_text,
                response_lang=original.response_language,
            )
        elif input_type == 'youtube':
            if not original.input_text:
                return jsonify({'error': 'No URL available for re-analysis'}), 400
            from services.youtube_service import analyze_youtube
            result = analyze_youtube(
                original.input_text,
                response_lang=original.response_language,
            )
        elif input_type == 'url':
            if not original.input_text:
                return jsonify({'error': 'No URL available for re-analysis'}), 400
            from services.url_service import analyze_url
            result = analyze_url(
                original.input_text,
                response_lang=original.response_language,
            )
        elif input_type in ('image', 'video'):
            return jsonify({
                'error': 'Re-analysis of uploaded files is not supported (file not retained).',
            }), 400
        else:
            return jsonify({'error': f'Unsupported input type: {input_type}'}), 400

        if not result:
            return jsonify({'error': 'Analysis service unavailable'}), 503

        # Save new report
        new_report = Report(
            user_id=current_user.id,
            input_type=input_type,
            input_text=original.input_text,
            input_title=result.get('title', original.input_title),
            prediction=result.get('label', result.get('prediction', '')),
            confidence=result.get('confidence'),
            risk_level=result.get('risk_level', ''),
            response_language=original.response_language,
            source=result.get('source', 'Groq (Llama-3.3)'),
        )
        new_report.set_result(result)
        db.session.add(new_report)
        db.session.commit()

        return jsonify({
            'success': True,
            'report_id': new_report.id,
            'original_id': original.id,
            **result,
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Server Error: {str(e)}'}), 500
