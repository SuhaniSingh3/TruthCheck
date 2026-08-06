"""
TruthCheck — AI Image Verification Service
===========================================
Production-grade multi-signal forensic pipeline that combines:
  1. Error Level Analysis (ELA)          — re-compression artifact detection
  2. Noise Pattern Fingerprinting        — Laplacian variance analysis
  3. Frequency Domain Analysis           — FFT high-frequency anomaly scoring
  4. JPEG Artifact Scoring               — 8×8 DCT block discontinuity
  5. EXIF Metadata Analysis              — completeness & software-tag forensics
  6. Optional HuggingFace CNN            — umm-maybe/AI-image-detector (if available)
  7. Groq LLM                            — natural-language reason generation

Models are cached at module level and loaded only once per process.
HuggingFace model is stored in the default transformers cache:
  Windows: C:\\Users\\<user>\\.cache\\huggingface\\hub\\
"""

import os
import io
import hashlib
import logging
import tempfile
import base64
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# ─── Model Cache (module-level singletons) ──────────────────────────────────────

_hf_pipeline = None          # HuggingFace image-classification pipeline
_hf_loaded_attempt = False   # Prevent repeated failed loads


def _load_hf_model() -> Optional[Any]:
    """
    Lazily load the HuggingFace AI-image-detector model.
    Cached after first successful load. Returns None if transformers/torch
    are not installed — the pipeline degrades gracefully.

    Model: umm-maybe/AI-image-detector (~250 MB)
    Stored at: ~/.cache/huggingface/hub/ (default HF cache)
    """
    global _hf_pipeline, _hf_loaded_attempt
    if _hf_loaded_attempt:
        return _hf_pipeline
    _hf_loaded_attempt = True
    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading HuggingFace AI image detector (first run may download ~250 MB)…")
        _hf_pipeline = hf_pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector",
            device=-1  # CPU — change to 0 for GPU
        )
        logger.info("HuggingFace model loaded and cached.")
    except Exception as exc:
        logger.warning(f"HuggingFace model not available ({exc}). Forensic-only mode active.")
        _hf_pipeline = None
    return _hf_pipeline


# ─── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> Tuple[Image.Image, Image.Image]:
    """
    Load, resize, and normalize an image for forensic analysis.
    Returns (pil_rgb, pil_small) where pil_small is 512×512 max.
    Alpha channel is stripped; color space converted to RGB.
    """
    img = Image.open(image_path)
    # Remove alpha channel
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Capped resize for analysis (keep aspect ratio)
    img_small = img.copy()
    img_small.thumbnail((512, 512), Image.LANCZOS)
    return img, img_small


# ─── Signal 1: Error Level Analysis (ELA) ───────────────────────────────────────

def run_ela_analysis(image_path: str, quality: int = 75) -> float:
    """
    Error Level Analysis: Re-compress the image at a lower quality and
    compare it to the original. High difference → likely edited regions.

    Returns a fake_score (0–100) where higher = more suspicious.
    """
    try:
        img, _ = preprocess_image(image_path)
        # Re-save at reduced quality into a buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        orig_arr = np.array(img, dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)

        # Compute per-pixel absolute difference
        diff = np.abs(orig_arr - recomp_arr)
        # Mean difference normalized to 0-100 scale
        # Authentic images: ~10–25; heavily edited: >40
        mean_diff = float(np.mean(diff))
        # Map to 0–100 fake_score
        fake_score = min(100.0, (mean_diff / 50.0) * 100.0)
        return round(fake_score, 2)
    except Exception as exc:
        logger.warning(f"ELA analysis failed: {exc}")
        return 50.0  # neutral fallback


# ─── Signal 2: Noise Pattern Analysis ───────────────────────────────────────────

def run_noise_analysis(image_path: str) -> float:
    """
    Analyze the noise fingerprint using Laplacian variance.
    AI-generated images often have unnaturally smooth or uniform noise patterns.

    Returns a fake_score (0–100) where higher = more suspicious.
    """
    try:
        _, img_small = preprocess_image(image_path)
        gray = np.array(img_small.convert("L"), dtype=np.float64)

        # Laplacian kernel for edge/noise detection
        from scipy.ndimage import laplace
        lap = laplace(gray)
        variance = float(np.var(lap))

        # Natural images: variance typically 200–2000+
        # AI-generated images: often <100 (too smooth) or very high (GAN artifacts)
        if variance < 80:
            # Extremely smooth — AI hallmark
            fake_score = min(100.0, 75.0 + (80.0 - variance) * 0.3)
        elif variance > 5000:
            # Extremely noisy — possible GAN artifact
            fake_score = min(100.0, 60.0 + (variance - 5000) * 0.002)
        else:
            # Natural range — lower suspicion
            fake_score = max(0.0, 40.0 - (variance - 80.0) * 0.01)

        return round(fake_score, 2)
    except Exception as exc:
        logger.warning(f"Noise analysis failed: {exc}")
        return 40.0


# ─── Signal 3: Frequency Domain Analysis ────────────────────────────────────────

def run_frequency_analysis(image_path: str) -> float:
    """
    FFT-based frequency domain analysis.
    AI-generated images often have anomalous high-frequency energy patterns
    from upsampling artifacts in the GAN decoder.

    Returns a fake_score (0–100).
    """
    try:
        _, img_small = preprocess_image(image_path)
        gray = np.array(img_small.convert("L"), dtype=np.float64)

        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # High-frequency mask: outer 30% of the spectrum
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        high_freq_mask = dist > (max_dist * 0.7)
        low_freq_mask = dist < (max_dist * 0.3)

        high_energy = float(np.mean(magnitude[high_freq_mask]))
        low_energy = float(np.mean(magnitude[low_freq_mask]))

        if low_energy == 0:
            return 50.0

        # Natural images: ratio typically 0.01–0.05
        # GAN images: ratio often >0.08 (upsampling artifacts)
        ratio = high_energy / (low_energy + 1e-10)
        fake_score = min(100.0, ratio * 600.0)
        return round(fake_score, 2)
    except Exception as exc:
        logger.warning(f"Frequency analysis failed: {exc}")
        return 40.0


# ─── Signal 4: JPEG Artifact Scoring ────────────────────────────────────────────

def run_jpeg_artifact_analysis(image_path: str) -> float:
    """
    Analyze JPEG compression block artifacts (8×8 DCT blocks).
    Edited images show discontinuities at block boundaries.
    PNG or non-JPEG files get a neutral score.

    Returns a fake_score (0–100).
    """
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ('.jpg', '.jpeg'):
            return 30.0  # Non-JPEG: neutral

        _, img_small = preprocess_image(image_path)
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        h, w = gray.shape

        discontinuities = []
        for i in range(8, h, 8):
            row_diff = float(np.mean(np.abs(gray[i, :] - gray[i - 1, :])))
            discontinuities.append(row_diff)
        for j in range(8, w, 8):
            col_diff = float(np.mean(np.abs(gray[:, j] - gray[:, j - 1])))
            discontinuities.append(col_diff)

        if not discontinuities:
            return 30.0

        avg_discontinuity = float(np.mean(discontinuities))
        # Natural: ~2–8; edited: often >12
        fake_score = min(100.0, (avg_discontinuity / 15.0) * 100.0)
        return round(fake_score, 2)
    except Exception as exc:
        logger.warning(f"JPEG artifact analysis failed: {exc}")
        return 30.0


# ─── Signal 5: EXIF Metadata Analysis ───────────────────────────────────────────

def run_metadata_analysis(image_path: str) -> Dict[str, Any]:
    """
    Analyze EXIF metadata for authenticity signals:
    - Missing EXIF → suspicious (AI images rarely have complete EXIF)
    - Software tags (Adobe, GIMP, Photoshop) → editing detected
    - GPS data presence → authentic camera origin
    - Camera make/model → authentic hardware origin

    Returns dict with fake_score (0–100) and extracted metadata fields.
    """
    result = {
        "fake_score": 50.0,
        "has_exif": False,
        "has_gps": False,
        "has_camera": False,
        "software": None,
        "camera_make": None,
        "camera_model": None,
        "datetime": None,
        "raw_tags": {}
    }
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            if not exif_data:
                # No EXIF at all — very suspicious
                result["fake_score"] = 70.0
                return result

            result["has_exif"] = True
            raw_tags = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (str, int, float, bytes)):
                    raw_tags[tag_name] = str(value)[:200]

            result["raw_tags"] = raw_tags

            sw = raw_tags.get("Software", "")
            result["software"] = sw if sw else None
            result["camera_make"] = raw_tags.get("Make")
            result["camera_model"] = raw_tags.get("Model")
            result["datetime"] = raw_tags.get("DateTime") or raw_tags.get("DateTimeOriginal")

            # Check GPS IFD
            try:
                gps_ifd = exif_data.get_ifd(0x8825)
                result["has_gps"] = bool(gps_ifd)
            except Exception:
                pass

            # Scoring logic
            score = 50.0

            if result["camera_make"] and result["camera_model"]:
                score -= 20.0  # Has real camera — less suspicious
                result["has_camera"] = True

            if result["has_gps"]:
                score -= 15.0  # Has GPS location — authentic camera origin

            if result["datetime"]:
                score -= 5.0

            # Editing software detected
            editing_keywords = ["adobe", "photoshop", "gimp", "lightroom", "affinity",
                                 "paint", "canva", "midjourney", "stable diffusion",
                                 "dall-e", "firefly", "dall·e"]
            sw_lower = sw.lower() if sw else ""
            for kw in editing_keywords:
                if kw in sw_lower:
                    score += 30.0
                    break

            if not result["has_exif"] or len(raw_tags) < 3:
                score += 20.0

            result["fake_score"] = round(max(0.0, min(100.0, score)), 2)
    except Exception as exc:
        logger.warning(f"Metadata analysis failed: {exc}")
    return result


# ─── Signal 6: HuggingFace CNN ──────────────────────────────────────────────────

def run_ai_detection_model(image_path: str) -> Tuple[float, float]:
    """
    Run the optional HuggingFace AI-image-detector CNN.
    Returns (ai_probability, fake_score) both 0–100.
    Falls back to (50.0, 50.0) if model is unavailable.
    """
    model = _load_hf_model()
    if model is None:
        return 50.0, 50.0

    try:
        img, img_small = preprocess_image(image_path)
        results = model(img_small)
        # Model outputs labels like "artificial" / "human" with scores
        ai_score = 50.0
        for r in results:
            label = r["label"].lower()
            score = float(r["score"]) * 100.0
            if any(kw in label for kw in ["artificial", "fake", "generated", "ai"]):
                ai_score = score
                break
            elif any(kw in label for kw in ["human", "real", "authentic"]):
                ai_score = 100.0 - score
                break
        return round(ai_score, 2), round(ai_score, 2)
    except Exception as exc:
        logger.warning(f"HF model inference failed: {exc}")
        return 50.0, 50.0


# ─── Image Hash ─────────────────────────────────────────────────────────────────

def compute_image_hash(image_path: str) -> str:
    """Compute SHA-256 hash of the raw image file bytes."""
    try:
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ─── Groq Reason Generator ──────────────────────────────────────────────────────

def generate_reason(
    prediction: str,
    confidence: float,
    details: Dict[str, float],
    metadata: Dict[str, Any]
) -> str:
    """
    Use Groq LLM to generate a natural-language forensic explanation.
    Falls back to a template-based reason if Groq is unavailable.
    """
    try:
        from services.groq_service import client as groq_client
        if not groq_client:
            raise RuntimeError("No Groq client")

        ela = details.get("ela_score", 50)
        noise = details.get("noise_score", 50)
        freq = details.get("freq_score", 50)
        meta = details.get("metadata_score", 50)
        ai_det = details.get("ai_detection_score", 50)
        software = metadata.get("software") or "unknown"
        has_camera = metadata.get("has_camera", False)
        has_gps = metadata.get("has_gps", False)

        prompt = f"""You are an AI Image Forensics expert. Provide a concise (2–3 sentence) natural language explanation for why this image was classified as '{prediction}' with {confidence:.1f}% confidence.

Forensic signals detected:
- ELA (Error Level Analysis) score: {ela:.1f}/100 (higher = more editing artifacts)
- Noise Pattern score: {noise:.1f}/100 (higher = unnatural noise)
- Frequency Domain score: {freq:.1f}/100 (higher = AI upsampling artifacts)
- Metadata completeness score: {meta:.1f}/100 (higher = more suspicious metadata gaps)
- AI Detection CNN score: {ai_det:.1f}/100 (higher = more AI-like)
- Software tag: {software}
- Has real camera EXIF: {has_camera}
- Has GPS data: {has_gps}

Write 1–2 sentences explaining the most important signals in plain English for a non-technical user. Be specific about what was found."""

        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(f"Groq reason generation failed: {exc}")
        # Template fallback
        if prediction in ("Fake", "FAKE / MANIPULATED"):
            reasons = []
            details_ela = details.get("ela_score", 0)
            details_noise = details.get("noise_score", 0)
            if details_ela > 60:
                reasons.append("JPEG compression artifacts indicate editing or re-saving")
            if details_noise > 60:
                reasons.append("unnatural noise pattern detected — characteristic of AI generation")
            if details.get("metadata_score", 0) > 60:
                reasons.append("metadata is incomplete or missing camera information")
            if details.get("ai_detection_score", 0) > 70:
                reasons.append("AI generation probability is high based on visual patterns")
            if not reasons:
                reasons.append("multiple forensic signals indicate synthetic or manipulated content")
            return "Analysis detected: " + "; ".join(reasons) + "."
        else:
            return (
                f"The image shows natural noise patterns, consistent EXIF metadata"
                f"{', including GPS coordinates' if details.get('has_gps') else ''}"
                f" and camera information, with low ELA scores indicating minimal post-processing."
            )


# ─── Main Pipeline Orchestrator ─────────────────────────────────────────────────

# Weights for each signal (must sum to 1.0)
PIPELINE_WEIGHTS = {
    "ela":          0.20,
    "noise":        0.15,
    "frequency":    0.15,
    "jpeg":         0.10,
    "metadata":     0.15,
    "ai_detection": 0.25,
}


def analyze_image_verification(image_path: str) -> Dict[str, Any]:
    """
    Main image verification orchestrator.
    Runs all forensic signals, combines via weighted scoring, and returns
    a structured result dict compatible with the /api/verify-image endpoint.

    Args:
        image_path: Absolute path to the uploaded image file.

    Returns:
        {
          "prediction": "Fake" | "Suspicious" | "Real",
          "confidence": float,       # 0–100
          "reason": str,             # Natural language explanation
          "details": {               # Individual signal scores 0–100
              "ela_score": float,
              "noise_score": float,
              "freq_score": float,
              "jpeg_score": float,
              "metadata_score": float,
              "ai_detection_score": float,
              "gan_probability": float,
              "editing_probability": float,
              "metadata_completeness": float,
          },
          "metadata": {              # Raw EXIF fields
              "has_exif": bool, "has_gps": bool, "camera_make": str, ...
          },
          "image_hash": str,         # SHA-256 of original file
          "hf_model_used": bool,     # Whether HF CNN was used
        }
    """
    # --- Run all signals ---
    ela_score      = run_ela_analysis(image_path)
    noise_score    = run_noise_analysis(image_path)
    freq_score     = run_frequency_analysis(image_path)
    jpeg_score     = run_jpeg_artifact_analysis(image_path)
    meta_result    = run_metadata_analysis(image_path)
    meta_score     = meta_result["fake_score"]
    ai_score, _    = run_ai_detection_model(image_path)

    # --- Weighted composite fake score ---
    composite = (
        ela_score      * PIPELINE_WEIGHTS["ela"]          +
        noise_score    * PIPELINE_WEIGHTS["noise"]        +
        freq_score     * PIPELINE_WEIGHTS["frequency"]    +
        jpeg_score     * PIPELINE_WEIGHTS["jpeg"]         +
        meta_score     * PIPELINE_WEIGHTS["metadata"]     +
        ai_score       * PIPELINE_WEIGHTS["ai_detection"]
    )
    composite = round(composite, 2)

    # --- Verdict mapping ---
    if composite >= 65.0:
        prediction = "Fake"
        confidence = composite
    elif composite >= 45.0:
        prediction = "Suspicious"
        confidence = composite
    else:
        prediction = "Real"
        confidence = round(100.0 - composite, 2)

    # --- Build details dict ---
    details = {
        "ela_score":           ela_score,
        "noise_score":         noise_score,
        "freq_score":          freq_score,
        "jpeg_score":          jpeg_score,
        "metadata_score":      meta_score,
        "ai_detection_score":  ai_score,
        # Alias fields for API compatibility
        "gan_probability":     round(ai_score / 100.0, 4),
        "editing_probability": round((ela_score * 0.6 + jpeg_score * 0.4) / 100.0, 4),
        "metadata_completeness": round(1.0 - (meta_score / 100.0), 4),
    }

    # --- Metadata dict (for frontend display) ---
    metadata_display = {
        "has_exif":    meta_result.get("has_exif", False),
        "has_gps":     meta_result.get("has_gps", False),
        "has_camera":  meta_result.get("has_camera", False),
        "software":    meta_result.get("software"),
        "camera_make": meta_result.get("camera_make"),
        "camera_model": meta_result.get("camera_model"),
        "datetime":    meta_result.get("datetime"),
    }

    # --- Reason generation ---
    reason = generate_reason(prediction, confidence, details, metadata_display)

    # --- Image hash ---
    image_hash = compute_image_hash(image_path)

    return {
        "prediction":    prediction,
        "confidence":    confidence,
        "reason":        reason,
        "details":       details,
        "metadata":      metadata_display,
        "image_hash":    image_hash,
        "hf_model_used": _hf_pipeline is not None,
        "pipeline_weights": PIPELINE_WEIGHTS,
    }


# ─── Legacy compatibility (keep old interface working) ──────────────────────────

def extract_metadata(image_path: str) -> Dict[str, str]:
    """Legacy function — kept for backward compatibility with old routes."""
    result = run_metadata_analysis(image_path)
    return result.get("raw_tags", {})


def encode_image_base64(image_path: str) -> str:
    """Encode image to Base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def analyze_image(image_path: str, filename: str = "", response_lang: str = "en") -> Dict[str, Any]:
    """
    Legacy analyze_image function — now delegates to the full verification pipeline.
    Kept for backward compatibility with the old /api/image/analyze route.
    """
    result = analyze_image_verification(image_path)
    # Map to old schema expected by legacy route
    result["filename"] = filename
    result["label"] = result["prediction"]
    result["explanation"] = result["reason"]
    result["ai_probability"] = result["details"].get("ai_detection_score", 50.0)
    result["human_probability"] = round(100.0 - result["details"].get("ai_detection_score", 50.0), 2)
    result["manipulation_score"] = result["details"].get("editing_probability", 0.5) * 100.0
    result["suspicious_regions"] = []
    return result
