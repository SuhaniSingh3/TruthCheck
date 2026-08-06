"""
TruthCheck — Image Verification Routes
=======================================
Provides two endpoints:
  POST /api/verify-image   — New full forensic pipeline (primary)
  POST /api/image/analyze  — Legacy endpoint (backward-compatible, delegates to pipeline)
  GET  /image-detect       — Renders the image verification UI page
"""
import os
import hashlib
import logging
from flask import Blueprint, render_template, request, jsonify
from flask_login import current_user
from werkzeug.utils import secure_filename
from datetime import datetime

from extensions import db
from models.report import Report
from config import Config

logger = logging.getLogger(__name__)
image_bp = Blueprint("image", __name__)

# Supported MIME types for security validation
ALLOWED_MIMETYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp"
}

# Maximum file size: 20 MB
IMAGE_MAX_BYTES = 20 * 1024 * 1024


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


def _allowed_image(filename: str) -> bool:
    """Check whether the file extension is in the allowed image set."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_IMAGE_EXTENSIONS
    )


def _validate_image_file(file) -> tuple[bool, str]:
    """
    Comprehensive image file validation:
    - Extension check
    - MIME type check
    - File size check (20 MB limit)
    - Minimal PIL open check (ensures it's a real image)
    """
    if not file or not file.filename:
        return False, "No file selected."

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file type '.{ext}'. Allowed: JPG, JPEG, PNG, WEBP."

    # Read into memory to check size and validate PIL
    file_bytes = file.read()
    if len(file_bytes) > IMAGE_MAX_BYTES:
        return False, f"File too large ({len(file_bytes) // (1024*1024)} MB). Maximum allowed: 20 MB."

    # Validate it's actually an image
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        return False, "File appears to be corrupted or is not a valid image."

    # Reset stream for saving
    file.stream = __import__("io").BytesIO(file_bytes)
    return True, ""


@image_bp.route("/image-detect")
def image_detect_page():
    """Render the AI Image Verification page."""
    return render_template(
        "analysis/image.html",
        supported_languages=Config.SUPPORTED_LANGUAGES,
    )


@image_bp.route("/api/verify-image", methods=["POST"])
def verify_image():
    """
    POST /api/verify-image
    Accept: multipart/form-data  (field name: 'file')
    Returns: JSON forensic analysis result

    Response schema:
    {
      "success": true,
      "prediction": "Fake" | "Suspicious" | "Real",
      "confidence": float,
      "reason": str,
      "details": {
          "ela_score": float,
          "noise_score": float,
          "freq_score": float,
          "jpeg_score": float,
          "metadata_score": float,
          "ai_detection_score": float,
          "gan_probability": float,
          "editing_probability": float,
          "metadata_completeness": float
      },
      "metadata": { "has_exif": bool, "has_gps": bool, ... },
      "image_hash": str,
      "filename": str,
      "timestamp": str
    }
    """
    saved_path = None
    try:
        # ── File presence check ──
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Please attach an image file."}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        # ── Validation ──
        is_valid, err_msg = _validate_image_file(file)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        # ── Save temporarily ──
        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
        safe_name = ts + filename
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        saved_path = os.path.join(Config.UPLOAD_FOLDER, safe_name)
        file.save(saved_path)

        # ── Run forensic pipeline ──
        from services.image_service import analyze_image_verification
        result = analyze_image_verification(saved_path)

        if not result:
            return jsonify({"error": "Image analysis service unavailable. Please try again."}), 503

        # ── Persist report (non-critical) ──
        try:
            prediction_label = result.get("prediction", "Unknown")
            confidence_val = result.get("confidence", 0.0)
            report = Report(
                user_id=_get_user_id(),
                input_type="image",
                input_text=None,
                input_filename=filename,
                input_title=f"Image Verification — {filename}",
                prediction=prediction_label,
                confidence=confidence_val,
                risk_level="critical" if prediction_label == "Fake" else
                           "medium" if prediction_label == "Suspicious" else "low",
                response_language="en",
                source="TruthCheck Forensic Pipeline",
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()
            result["report_id"] = report.id
        except Exception as db_err:
            logger.warning(f"Report save warning (non-critical): {db_err}")
            db.session.rollback()

        return jsonify({
            "success": True,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            **result,
        })

    except Exception as exc:
        logger.error(f"Image verification error: {exc}", exc_info=True)
        return jsonify({"error": f"Server error during analysis: {str(exc)}"}), 500

    finally:
        # Always clean up the temp file
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError as e:
                logger.warning(f"Could not delete temp file {saved_path}: {e}")


@image_bp.route("/api/image/analyze", methods=["POST"])
def analyze_image_upload():
    """
    POST /api/image/analyze  — Legacy endpoint kept for backward compatibility.
    Delegates to the new forensic pipeline under the hood.

    Expects multipart/form-data with:
        - ``file``: The image file to analyze.
        - ``response_lang`` (optional): Preferred response language.
    """
    saved_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not _allowed_image(file.filename):
            allowed = ", ".join(Config.ALLOWED_IMAGE_EXTENSIONS)
            return jsonify({"error": f"Invalid file type. Allowed: {allowed}"}), 400

        response_lang = request.form.get("response_lang", Config.DEFAULT_LANGUAGE)

        from utils.validators import validate_file_upload
        is_valid, error_msg = validate_file_upload(
            file, allowed_extensions=Config.ALLOWED_IMAGE_EXTENSIONS
        )
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        filename = secure_filename(file.filename)
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S_")
        safe_name = timestamp_prefix + filename
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        saved_path = os.path.join(Config.UPLOAD_FOLDER, safe_name)
        file.save(saved_path)

        from services.image_service import analyze_image
        result = analyze_image(saved_path, filename=filename, response_lang=response_lang)

        if not result:
            return jsonify({"error": "Image analysis service unavailable"}), 503

        try:
            report = Report(
                user_id=_get_user_id(),
                input_type="image",
                input_text=None,
                input_filename=filename,
                input_title=result.get("title", filename),
                prediction=result.get("label", result.get("prediction", "")),
                confidence=result.get("confidence"),
                risk_level=result.get("risk_level", ""),
                response_language=response_lang,
                source=result.get("source", "TruthCheck Forensic Pipeline"),
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()
            result["report_id"] = report.id
        except Exception:
            db.session.rollback()

        return jsonify({
            "success": True,
            "input_type": "image",
            "filename": filename,
            **result,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

    finally:
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
