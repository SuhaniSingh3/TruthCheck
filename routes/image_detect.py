"""
TruthCheck — Image Verification Routes (Vercel-Compatible)
==========================================================
Endpoints:
  POST /api/verify-image   — Full 3-part forensic pipeline (primary)
  POST /api/image/analyze  — Legacy endpoint (backward-compatible)
  GET  /image-detect       — Renders the image verification UI page

Vercel Compatibility:
  - Images are processed IN MEMORY — no disk writes to project root
  - The UPLOAD_FOLDER is only used if a temp file is absolutely needed;
    it resolves to /tmp on Vercel (the only writable path)
  - Database operations are wrapped in try/except (non-critical on Vercel)
"""
import os
import io
from werkzeug.exceptions import RequestEntityTooLarge
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

# Magic bytes for common image formats (path traversal & extension spoofing protection)
MAGIC_BYTES = {
    b'\xff\xd8\xff': 'jpeg',   # JPEG
    b'\x89PNG':       'png',    # PNG
    b'RIFF':          'webp',   # WEBP (RIFF....WEBP)
}


def _get_user_id():
    """Return the current user's id, or None for anonymous sessions."""
    return current_user.id if current_user.is_authenticated else None


def _allowed_image(filename: str) -> bool:
    """Check whether the file extension is in the allowed image set."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_IMAGE_EXTENSIONS
    )


def _validate_image_bytes(file_bytes: bytes, filename: str = "") -> tuple[bool, str]:
    """
    Comprehensive image validation — runs entirely in memory (no disk write).

    Checks:
    1. File size (max 20 MB)
    2. File extension (whitelist)
    3. Magic bytes / MIME type validation (prevents extension spoofing)
    4. PIL open sanity check (detects corrupted images)

    Returns (is_valid: bool, error_message: str)
    """
    # Size check
    if len(file_bytes) > IMAGE_MAX_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        return False, f"File too large ({mb:.1f} MB). Maximum allowed: 20 MB."

    if len(file_bytes) < 100:
        return False, "File is too small to be a valid image."

    # Extension check (only if filename is provided)
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Unsupported file type '.{ext}'. Allowed: JPG, JPEG, PNG, WEBP."

    # Magic bytes check — prevent extension spoofing
    header = file_bytes[:12]
    is_known_format = (
        header[:3] in (b'\xff\xd8\xff',) or   # JPEG
        header[:4] == b'\x89PNG' or             # PNG
        (header[:4] == b'RIFF' and header[8:12] == b'WEBP')  # WEBP
    )
    if not is_known_format:
        return False, "File does not appear to be a valid image (invalid file signature)."

    # PIL sanity check — catches corrupted/truncated images
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # verify() closes the image; re-open to access size etc.
    except Exception:
        return False, "File appears to be corrupted or is not a valid image."

    return True, ""


def _read_file_safely(file_obj) -> tuple[bytes, str]:
    """
    Read uploaded file into memory safely.
    Returns (file_bytes, safe_filename).
    Never trusts the uploaded filename — sanitizes it.
    """
    # Read into memory (never execute the file, never trust its name)
    file_bytes = file_obj.read()
    original_name = file_obj.filename or "upload"
    # secure_filename strips path traversal attempts, dangerous chars, etc.
    safe_name = secure_filename(original_name) or "image"
    return file_bytes, safe_name


# ─── Routes ─────────────────────────────────────────────────────────────────────

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
    Accept: multipart/form-data (field name: 'file' or 'image')

    Processes the image entirely in memory — Vercel-safe.

    Returns JSON structured as:
    {
      "success": true,
      "overall": { "status": str, "confidence": float },
      "ai_generated": { "status": str, "confidence": float, "probability": float, "reason": str },
      "manipulation": { "status": str, "confidence": float, "probability": float, "reason": str },
      "forensics": { "ela_score": float, "noise_score": float, ... },
      "metadata": { "available": bool, "camera": str, ... },
      "reason": [str, ...],
      "explanation": str,
      "model_information": { ... }
    }
    """
    try:
        # ── File presence check — accept 'file' or 'image' field names ──
        # Catch oversized files BEFORE accessing request.files (which raises 413)
        try:
            file = request.files.get("file") or request.files.get("image")
        except RequestEntityTooLarge:
            return jsonify({"error": "File too large. Maximum allowed size is 20 MB."}), 400

        if not file or not file.filename:
            return jsonify({"error": "No image file uploaded. Please attach an image."}), 400

        # ── Read into memory (Vercel-safe — no disk write) ──
        file_bytes, safe_name = _read_file_safely(file)

        # ── Comprehensive validation ──
        is_valid, err_msg = _validate_image_bytes(file_bytes, safe_name)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        # ── Run full forensic pipeline (in-memory) ──
        from services.image_service import analyze_image_verification
        result = analyze_image_verification(file_bytes, filename=safe_name)

        if not result:
            return jsonify({"error": "Image analysis service unavailable. Please try again."}), 503

        # ── Persist report (non-critical — silently skips if DB unavailable) ──
        try:
            overall_status = result.get("overall", {}).get("status", "Unknown")
            overall_conf   = result.get("overall", {}).get("confidence", 0.0)

            # Map overall status to risk level
            risk_map = {
                "AUTHENTIC":          "low",
                "LIKELY AUTHENTIC":   "low",
                "SUSPICIOUS":         "medium",
                "AI-GENERATED":       "medium",
                "LIKELY MANIPULATED": "high",
                "FAKE / DECEPTIVE":   "critical",
                "UNCERTAIN":          "low",
            }
            risk = risk_map.get(overall_status, "medium")

            report = Report(
                user_id=_get_user_id(),
                input_type="image",
                input_text=None,
                input_filename=safe_name,
                input_title=f"Image Verification — {safe_name}",
                prediction=overall_status,
                confidence=overall_conf,
                risk_level=risk,
                response_language="en",
                source="TruthCheck Forensic Pipeline",
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()
            result["report_id"] = report.id
        except Exception as db_err:
            logger.warning(f"Report save (non-critical): {db_err}")
            try:
                db.session.rollback()
            except Exception:
                pass

        return jsonify({
            **result,
            "filename":  safe_name,
            "timestamp": datetime.now().isoformat(),
        })

    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Maximum allowed size is 20 MB."}), 400

    except Exception as exc:
        logger.error(f"Image verification error: {exc}", exc_info=True)
        return jsonify({"error": f"Server error during analysis: {str(exc)}"}), 500


@image_bp.route("/api/image/analyze", methods=["POST"])
def analyze_image_upload():
    """
    POST /api/image/analyze — Legacy endpoint kept for backward compatibility.
    Delegates to the new forensic pipeline.
    Expects multipart/form-data with 'file' field.
    """
    try:
        file = request.files.get("file") or request.files.get("image")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        file_bytes, safe_name = _read_file_safely(file)
        is_valid, err_msg = _validate_image_bytes(file_bytes, safe_name)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        response_lang = request.form.get("response_lang", Config.DEFAULT_LANGUAGE)

        from services.image_service import analyze_image_verification
        result = analyze_image_verification(file_bytes, filename=safe_name)

        if not result:
            return jsonify({"error": "Image analysis service unavailable"}), 503

        # Map to legacy schema
        result["label"]       = result.get("overall", {}).get("status", "Unknown")
        result["prediction"]  = result["label"]
        result["confidence"]  = result.get("overall", {}).get("confidence", 50.0)
        result["filename"]    = safe_name

        try:
            report = Report(
                user_id=_get_user_id(),
                input_type="image",
                input_text=None,
                input_filename=safe_name,
                input_title=result.get("label", safe_name),
                prediction=result["label"],
                confidence=result["confidence"],
                response_language=response_lang,
                source="TruthCheck Forensic Pipeline",
            )
            report.set_result(result)
            db.session.add(report)
            db.session.commit()
            result["report_id"] = report.id
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass

        return jsonify({
            "success": True,
            "input_type": "image",
            **result,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Legacy image analyze error: {e}", exc_info=True)
        return jsonify({"error": f"Server Error: {str(e)}"}), 500
