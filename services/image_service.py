"""
TruthCheck — AI Image Verification Service (Vercel-Compatible)
==============================================================
Production-grade multi-signal forensic pipeline using ONLY numpy + Pillow.
No scipy, no OpenCV, no torch required — fully compatible with Vercel Lambda.

Pipeline signals:
  1. Error Level Analysis (ELA)         — re-compression artifact detection
  2. Noise Pattern Analysis             — numpy Laplacian kernel (no scipy)
  3. Frequency Domain Analysis          — FFT high-frequency anomaly scoring
  4. JPEG Artifact Scoring              — 8×8 DCT block discontinuity
  5. EXIF Metadata Analysis             — completeness & software-tag forensics
  6. Copy-Move Detection (approx.)      — block-hash based similarity
  7. Groq Vision/Reason Generation      — LLM-based reason text (text only)
  8. HuggingFace CNN (optional)         — disabled on Vercel, local only

Results are structured in THREE separate parts per the specification:
  A. AI-Generated Detection
  B. Image Manipulation Detection
  C. Overall Authenticity Assessment

Models are cached at module level and loaded only once per process.
"""

import os
import io
import hashlib
import logging
import base64
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# ─── Model Cache (module-level singletons) ──────────────────────────────────────

_hf_pipeline = None          # HuggingFace image-classification pipeline
_hf_loaded_attempt = False   # Prevent repeated failed loads


def _load_hf_model() -> Optional[Any]:
    """
    Lazily load the HuggingFace AI-image-detector model.
    Cached after first successful load.
    DISABLED on Vercel (model is ~250MB and exceeds Lambda limits).
    Returns None if unavailable — pipeline degrades gracefully.
    """
    global _hf_pipeline, _hf_loaded_attempt
    if _hf_loaded_attempt:
        return _hf_pipeline
    _hf_loaded_attempt = True

    # Do not attempt to load on Vercel
    from config import IS_VERCEL
    if IS_VERCEL:
        logger.info("HuggingFace model disabled on Vercel — forensic-only mode.")
        return None

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading HuggingFace AI image detector (first run may download ~250 MB)…")
        _hf_pipeline = hf_pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector",
            device=-1  # CPU
        )
        logger.info("HuggingFace model loaded and cached.")
    except Exception as exc:
        logger.warning(f"HuggingFace model not available ({exc}). Forensic-only mode.")
        _hf_pipeline = None
    return _hf_pipeline


# ─── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(image_source) -> Tuple[Image.Image, Image.Image]:
    """
    Load and normalize an image for forensic analysis.
    Accepts a file path (str) or file-like bytes object.
    Returns (pil_rgb_original, pil_small_512) where pil_small is 512×512 max.
    Alpha channel is stripped; color space converted to RGB.
    """
    if isinstance(image_source, (str, os.PathLike)):
        img = Image.open(image_source)
    else:
        img = Image.open(io.BytesIO(image_source))

    # Remove alpha channel / normalize mode
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        mask = img.split()[-1] if img.mode in ("RGBA", "LA") else None
        background.paste(img, mask=mask)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img_small = img.copy()
    img_small.thumbnail((512, 512), Image.LANCZOS)
    return img, img_small


def _numpy_laplacian(gray_array: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 Laplacian kernel using pure numpy (no scipy needed).
    Equivalent to scipy.ndimage.laplace for variance-based noise analysis.

    Kernel:  0  1  0
             1 -4  1
             0  1  0
    """
    # Pad with edge values to handle borders
    padded = np.pad(gray_array, 1, mode='edge')
    lap = (
        padded[:-2, 1:-1] +   # top
        padded[2:,  1:-1] +   # bottom
        padded[1:-1, :-2] +   # left
        padded[1:-1, 2:]  +   # right
        -4 * gray_array
    )
    return lap


# ─── Signal 1: Error Level Analysis (ELA) ───────────────────────────────────────

def run_ela_analysis(image_source, quality: int = 75) -> float:
    """
    Error Level Analysis: Re-compress at lower quality and compare to original.
    High difference in specific regions → likely edited.

    Returns a manipulation_score (0–100) where higher = more suspicious.
    Authentic images: ~5–25; heavily edited: >45
    """
    try:
        img, _ = preprocess_image(image_source)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        orig_arr = np.array(img, dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)

        diff = np.abs(orig_arr - recomp_arr)
        mean_diff = float(np.mean(diff))

        # Standard deviation of pixel differences (high std = localized edits)
        std_diff = float(np.std(diff))

        # Combined score: mean diff weighted with std deviation
        # Mean: overall edit level; Std: localized anomalies (splicing)
        combined = (mean_diff * 0.6 + std_diff * 0.4)
        # Calibration: authentic ≈ 10-25, edited ≈ 40+
        score = min(100.0, (combined / 45.0) * 100.0)
        return round(score, 2)
    except Exception as exc:
        logger.warning(f"ELA analysis failed: {exc}")
        return 50.0  # neutral fallback


# ─── Signal 2: Noise Pattern Analysis ───────────────────────────────────────────

def run_noise_analysis(image_source) -> float:
    """
    Analyze noise using numpy Laplacian kernel (scipy-free).
    AI-generated images have unnaturally smooth or over-uniform noise.

    Returns manipulation_score (0–100) where higher = more suspicious (AI-like).
    """
    try:
        _, img_small = preprocess_image(image_source)
        gray = np.array(img_small.convert("L"), dtype=np.float64)

        lap = _numpy_laplacian(gray)
        variance = float(np.var(lap))

        # Natural images: variance typically 200–3000+
        # AI-generated: often <80 (too smooth) or >8000 (GAN ringing artifacts)
        if variance < 80:
            # Extremely smooth — strong AI hallmark
            score = min(100.0, 80.0 + (80.0 - variance) * 0.25)
        elif variance > 8000:
            # Extremely noisy — possible GAN artifact or heavy compression
            score = min(100.0, 55.0 + (variance - 8000) * 0.001)
        else:
            # Natural range — low suspicion
            score = max(0.0, 35.0 - (variance - 80.0) * 0.008)

        return round(score, 2)
    except Exception as exc:
        logger.warning(f"Noise analysis failed: {exc}")
        return 40.0


# ─── Signal 3: Frequency Domain Analysis ────────────────────────────────────────

def run_frequency_analysis(image_source) -> float:
    """
    FFT-based frequency domain analysis.
    AI upsampling (GAN decoders, diffusion upscalers) leaves characteristic
    high-frequency energy patterns.

    Returns a score (0–100) where higher = more AI-like frequency artifacts.
    """
    try:
        _, img_small = preprocess_image(image_source)
        gray = np.array(img_small.convert("L"), dtype=np.float64)

        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        # High-freq: outer 30% of spectrum | Low-freq: inner 30%
        high_freq_mask = dist > (max_dist * 0.7)
        low_freq_mask  = dist < (max_dist * 0.3)

        high_energy = float(np.mean(magnitude[high_freq_mask]))
        low_energy  = float(np.mean(magnitude[low_freq_mask]))

        if low_energy == 0:
            return 50.0

        ratio = high_energy / (low_energy + 1e-10)
        # Natural images: ratio ~0.01–0.06; GAN/diffusion: often >0.08
        score = min(100.0, ratio * 700.0)
        return round(score, 2)
    except Exception as exc:
        logger.warning(f"Frequency analysis failed: {exc}")
        return 40.0


# ─── Signal 4: JPEG Artifact Scoring ────────────────────────────────────────────

def run_jpeg_artifact_analysis(image_source, filename: str = "") -> float:
    """
    Analyze JPEG block artifact discontinuities (8×8 DCT blocks).
    Edited images show sharp discontinuities at block boundaries.
    PNG/WEBP files get a neutral score (30).

    Returns a score (0–100) where higher = more JPEG manipulation evidence.
    """
    try:
        ext = os.path.splitext(filename)[1].lower() if filename else ""
        if ext not in ('.jpg', '.jpeg'):
            return 30.0  # Non-JPEG: neutral

        _, img_small = preprocess_image(image_source)
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

        avg_disc = float(np.mean(discontinuities))
        # Natural: ~2–8; edited/re-saved: often >12
        score = min(100.0, (avg_disc / 15.0) * 100.0)
        return round(score, 2)
    except Exception as exc:
        logger.warning(f"JPEG artifact analysis failed: {exc}")
        return 30.0


# ─── Signal 5: EXIF Metadata Analysis ───────────────────────────────────────────

def run_metadata_analysis(image_source) -> Dict[str, Any]:
    """
    Analyze EXIF metadata for authenticity signals.
    - Missing EXIF → suspicious (AI images rarely have complete EXIF)
    - Editing software tags (Photoshop, GIMP) → manipulation detected
    - GPS data + camera make/model → authentic camera origin
    - AI generation software tags → AI-generated flag

    Returns dict with fake_score (0–100) and structured metadata fields.
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
        "editing_software_detected": False,
        "ai_software_detected": False,
        "raw_tags": {}
    }
    try:
        if isinstance(image_source, (str, os.PathLike)):
            img_ctx = Image.open(image_source)
        else:
            img_ctx = Image.open(io.BytesIO(image_source))

        with img_ctx as img:
            exif_data = img.getexif()
            if not exif_data:
                result["fake_score"] = 68.0  # No EXIF at all — very suspicious
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

            # Check GPS sub-IFD
            try:
                gps_ifd = exif_data.get_ifd(0x8825)
                result["has_gps"] = bool(gps_ifd)
            except Exception:
                pass

            # ── Scoring logic ──
            score = 50.0

            if result["camera_make"] and result["camera_model"]:
                score -= 20.0  # Real camera hardware — less suspicious
                result["has_camera"] = True

            if result["has_gps"]:
                score -= 12.0  # GPS = authentic camera origin

            if result["datetime"]:
                score -= 5.0

            # Editing software detection
            editing_keywords = [
                "photoshop", "gimp", "lightroom", "affinity", "paint.net",
                "canva", "capture one", "pixelmator", "darktable"
            ]
            ai_gen_keywords = [
                "midjourney", "stable diffusion", "dall-e", "dall·e",
                "firefly", "imagen", "bing image", "ideogram", "leonardo",
                "adobe firefly", "playground ai"
            ]
            sw_lower = sw.lower() if sw else ""

            for kw in ai_gen_keywords:
                if kw in sw_lower:
                    score += 40.0
                    result["ai_software_detected"] = True
                    break

            if not result["ai_software_detected"]:
                for kw in editing_keywords:
                    if kw in sw_lower:
                        score += 25.0
                        result["editing_software_detected"] = True
                        break

            # Very sparse EXIF (fewer than 3 real tags)
            if len(raw_tags) < 3:
                score += 18.0

            result["fake_score"] = round(max(0.0, min(100.0, score)), 2)
    except Exception as exc:
        logger.warning(f"Metadata analysis failed: {exc}")
    return result


# ─── Signal 6: Copy-Move Detection (approximation) ──────────────────────────────

def run_copy_move_detection(image_source) -> float:
    """
    Approximate copy-move (cloning) detection using block hashing.
    Divides image into overlapping 16×16 blocks and finds near-duplicates.
    High duplicate block ratio → possible cloning/copy-move manipulation.

    Returns probability (0–100) of copy-move manipulation.
    NOTE: This is an approximation. True copy-move detection (SIFT/SURF) requires
          OpenCV, which is excluded for Vercel compatibility.
    """
    try:
        _, img_small = preprocess_image(image_source)
        gray = np.array(img_small.convert("L"), dtype=np.uint8)
        h, w = gray.shape

        BLOCK = 16
        STEP  = 8  # overlapping blocks

        block_hashes = {}
        duplicate_count = 0
        total_blocks = 0

        for y in range(0, h - BLOCK, STEP):
            for x in range(0, w - BLOCK, STEP):
                block = gray[y:y+BLOCK, x:x+BLOCK]
                # Quantize to reduce noise sensitivity
                block_q = (block // 16).tobytes()
                h_val = hashlib.md5(block_q).hexdigest()[:8]
                if h_val in block_hashes:
                    # Check they're not adjacent blocks (allow 32px tolerance)
                    prev_y, prev_x = block_hashes[h_val]
                    if abs(y - prev_y) > 32 or abs(x - prev_x) > 32:
                        duplicate_count += 1
                else:
                    block_hashes[h_val] = (y, x)
                total_blocks += 1

        if total_blocks == 0:
            return 10.0

        ratio = duplicate_count / total_blocks
        # Typical authentic images: <2% | Copy-move: >5%
        score = min(100.0, ratio * 1500.0)
        return round(score, 2)
    except Exception as exc:
        logger.warning(f"Copy-move detection failed: {exc}")
        return 10.0


# ─── Signal 7: HuggingFace CNN ──────────────────────────────────────────────────

def run_ai_detection_model(image_source) -> Tuple[float, bool]:
    """
    Run the optional HuggingFace AI-image-detector CNN.
    Only available locally when torch + transformers are installed.
    Returns (ai_probability 0–100, model_was_used bool).
    """
    model = _load_hf_model()
    if model is None:
        return 50.0, False

    try:
        img, img_small = preprocess_image(image_source)
        results = model(img_small)
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
        return round(ai_score, 2), True
    except Exception as exc:
        logger.warning(f"HF model inference failed: {exc}")
        return 50.0, False


# ─── Image Hash ─────────────────────────────────────────────────────────────────

def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw image bytes."""
    try:
        return hashlib.sha256(image_bytes).hexdigest()
    except Exception:
        return ""


# ─── Groq Reason Generator ──────────────────────────────────────────────────────

def generate_forensic_reason(
    overall_status: str,
    overall_confidence: float,
    ai_generated: Dict,
    manipulation: Dict,
    forensics: Dict,
    metadata: Dict,
) -> str:
    """
    Generate a natural-language forensic explanation using Groq LLM.
    Falls back to a template-based explanation if Groq is unavailable.
    This uses text-only prompting — NOT an image vision model.
    """
    try:
        from services.groq_service import client as groq_client
        if not groq_client:
            raise RuntimeError("No Groq client configured")

        prompt = f"""You are a forensic image analyst. Provide a clear, 2–3 sentence natural-language explanation for the following image analysis result.

Overall Assessment: {overall_status} (Confidence: {overall_confidence:.1f}%)

AI Generation Analysis:
- Status: {ai_generated.get('status', 'UNCERTAIN')}
- Probability: {ai_generated.get('probability', 50.0):.1f}%

Manipulation Analysis:
- Status: {manipulation.get('status', 'UNCERTAIN')}
- Probability: {manipulation.get('probability', 50.0):.1f}%

Forensic Scores (0–100, higher = more suspicious):
- ELA (editing artifacts): {forensics.get('ela_score', 50):.1f}
- Noise pattern: {forensics.get('noise_score', 50):.1f}
- Frequency domain: {forensics.get('compression_score', 50):.1f}
- Splicing probability: {forensics.get('splicing_probability', 50):.1f}
- Copy-move probability: {forensics.get('copy_move_probability', 10):.1f}

Metadata:
- EXIF available: {metadata.get('available', False)}
- Camera detected: {metadata.get('camera', 'Unknown')}
- Editing software: {metadata.get('editing_software_detected', False)}

Write 2–3 sentences for a non-technical user explaining the most important findings. Be specific but accessible. Do NOT fabricate details not supported by the data."""

        resp = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "groq/compound-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(f"Groq reason generation failed: {exc}")
        return _template_reason(overall_status, ai_generated, manipulation, forensics)


def _template_reason(
    overall_status: str,
    ai_generated: Dict,
    manipulation: Dict,
    forensics: Dict,
) -> str:
    """Fallback template-based reason when Groq is unavailable."""
    parts = []

    ai_status = ai_generated.get("status", "UNCERTAIN")
    manip_status = manipulation.get("status", "UNCERTAIN")
    ela = forensics.get("ela_score", 50)
    noise = forensics.get("noise_score", 50)
    copy_move = forensics.get("copy_move_probability", 10)
    splicing = forensics.get("splicing_probability", 50)

    if ai_status == "YES":
        parts.append("Visual patterns are highly consistent with AI-generated content, including characteristic frequency artifacts and unnaturally smooth texture regions.")
    elif ai_status == "NO":
        parts.append("No strong evidence of AI generation was detected — the image shows natural noise and texture patterns typical of real photographs.")
    else:
        parts.append("AI generation probability is inconclusive based on the available forensic signals.")

    if manip_status == "YES":
        details = []
        if ela > 55:
            details.append("inconsistent re-compression levels")
        if splicing > 60:
            details.append("potential image splicing")
        if copy_move > 40:
            details.append("possible copy-move cloning")
        if noise > 60:
            details.append("unnatural noise patterns")
        if details:
            parts.append(f"Manipulation evidence was detected: {', '.join(details)}.")
        else:
            parts.append("Manipulation evidence was detected in pixel-level forensic analysis.")
    elif manip_status == "NO":
        parts.append("No significant manipulation artifacts were detected in the forensic analysis.")

    if overall_status in ("AUTHENTIC", "LIKELY AUTHENTIC"):
        parts.append("Overall, this image appears to be an authentic, unmodified photograph.")
    elif overall_status == "SUSPICIOUS":
        parts.append("The overall evidence is mixed — treat this image with caution and seek additional verification.")
    elif overall_status in ("LIKELY MANIPULATED", "FAKE / DECEPTIVE"):
        parts.append("Multiple forensic indicators suggest this image has been significantly altered or is deceptive.")
    elif overall_status == "AI-GENERATED":
        parts.append("This image was most likely created by an AI image generation system rather than a camera.")

    return " ".join(parts)


# ─── Three-Part Verdict Logic ────────────────────────────────────────────────────

def _determine_ai_verdict(
    hf_score: float,
    noise_score: float,
    freq_score: float,
    metadata: Dict,
) -> Dict[str, Any]:
    """
    Determine AI-generation verdict from available signals.

    Signals used:
    - HuggingFace CNN score (if model loaded)
    - Noise uniformity (AI images are too smooth)
    - Frequency domain ratio (GAN upsampling artifacts)
    - EXIF: AI generation software detected

    Returns structured dict with status, confidence, probability, reasons.
    """
    reasons = []
    evidence_scores = []

    # HF model (most reliable if available)
    if hf_score != 50.0:  # 50 = model not loaded / neutral
        evidence_scores.append(hf_score)
        if hf_score > 70:
            reasons.append("AI image classifier CNN indicates high probability of synthetic generation.")

    # Noise — strong AI indicator
    evidence_scores.append(noise_score * 0.9)
    if noise_score > 70:
        reasons.append("Noise patterns are unnaturally uniform, a hallmark of AI-generated images.")
    elif noise_score < 25:
        reasons.append("Natural noise fingerprint consistent with real camera capture.")

    # Frequency domain
    evidence_scores.append(freq_score * 0.8)
    if freq_score > 65:
        reasons.append("Frequency domain shows upsampling artifacts typical of GAN or diffusion decoders.")

    # Metadata: AI software
    if metadata.get("ai_software_detected"):
        evidence_scores.append(90.0)
        sw = metadata.get("software", "")
        reasons.append(f"EXIF metadata contains AI generation software tag: '{sw}'.")

    # Metadata: No EXIF at all (common for AI images)
    if not metadata.get("has_exif"):
        evidence_scores.append(60.0)
        reasons.append("No EXIF metadata found — common in AI-generated images.")

    # Calculate probability
    if evidence_scores:
        probability = float(np.mean(evidence_scores))
    else:
        probability = 50.0
    probability = round(min(100.0, max(0.0, probability)), 2)

    # Confidence in the verdict (certainty of the determination)
    confidence = round(abs(probability - 50.0) * 2.0, 2)
    confidence = max(10.0, min(99.0, confidence))

    # Status
    if probability >= 70:
        status = "YES"
    elif probability <= 30:
        status = "NO"
        confidence = round((1.0 - probability / 100.0) * 2.0 * 50.0, 2)
        confidence = max(10.0, min(99.0, confidence))
        if not reasons:
            reasons.append("No significant evidence of AI generation detected.")
    else:
        status = "UNCERTAIN"
        if not reasons:
            reasons.append("Mixed signals — AI generation probability is inconclusive.")

    return {
        "status": status,
        "confidence": confidence,
        "probability": probability,
        "reasons": reasons if reasons else ["Insufficient evidence to determine AI generation status."],
    }


def _determine_manipulation_verdict(
    ela_score: float,
    jpeg_score: float,
    copy_move_score: float,
    metadata: Dict,
) -> Dict[str, Any]:
    """
    Determine image manipulation/editing verdict.

    Signals used:
    - ELA score (re-compression artifacts = editing)
    - JPEG block artifacts
    - Copy-move detection
    - EXIF: editing software detected
    - EXIF: missing camera info

    Note: Minor edits (brightness, crop, resize) are NOT classified as deceptive.
    """
    reasons = []
    evidence_scores = []

    # ELA — primary manipulation signal
    evidence_scores.append(ela_score)
    if ela_score > 60:
        reasons.append("Error Level Analysis reveals inconsistent re-compression patterns, indicating localized edits.")
    elif ela_score > 40:
        reasons.append("Moderate ELA deviation detected — possible minor post-processing.")
    else:
        reasons.append("ELA shows consistent compression throughout the image, consistent with an unedited photo.")

    # JPEG artifacts
    if jpeg_score > 60:
        evidence_scores.append(jpeg_score)
        reasons.append("JPEG block boundary discontinuities indicate the image was re-saved multiple times, a common manipulation indicator.")
    elif jpeg_score > 30:
        evidence_scores.append(jpeg_score * 0.7)

    # Copy-move
    evidence_scores.append(copy_move_score * 0.8)
    if copy_move_score > 50:
        reasons.append("Duplicate image regions detected — possible copy-move cloning or object removal.")
    elif copy_move_score > 30:
        reasons.append("Some repeated patterns detected; may indicate minor cloning.")

    # Editing software in EXIF
    if metadata.get("editing_software_detected"):
        evidence_scores.append(70.0)
        sw = metadata.get("software", "")
        reasons.append(f"EXIF metadata contains editing software tag: '{sw}'.")
    elif metadata.get("ai_software_detected"):
        # AI-generated, not necessarily edited
        pass

    probability = round(float(np.mean(evidence_scores)), 2) if evidence_scores else 50.0
    probability = round(min(100.0, max(0.0, probability)), 2)

    # Splicing probability derived from ELA std deviation proxy
    splicing_probability = round(min(100.0, ela_score * 0.9), 2)

    confidence = round(abs(probability - 50.0) * 2.0, 2)
    confidence = max(10.0, min(99.0, confidence))

    if probability >= 65:
        status = "YES"
    elif probability <= 30:
        status = "NO"
        confidence = round((1.0 - probability / 100.0) * 2.0 * 50.0, 2)
        confidence = max(10.0, min(99.0, confidence))
    else:
        status = "UNCERTAIN"

    return {
        "status": status,
        "confidence": confidence,
        "probability": probability,
        "splicing_probability": splicing_probability,
        "reasons": reasons if reasons else ["No significant manipulation evidence detected."],
    }


def _determine_overall_verdict(
    ai_verdict: Dict,
    manip_verdict: Dict,
    metadata: Dict,
) -> Dict[str, Any]:
    """
    Determine the overall authenticity status.

    Important rules per specification:
    - AI-generated ≠ automatically fake
    - Minor editing (crop, brightness) ≠ deceptive
    - Must require strong evidence for FAKE / DECEPTIVE
    - UNCERTAIN when evidence is insufficient

    Status options:
        AUTHENTIC, LIKELY AUTHENTIC, SUSPICIOUS, LIKELY MANIPULATED,
        AI-GENERATED, FAKE / DECEPTIVE, UNCERTAIN
    """
    ai_prob = ai_verdict.get("probability", 50.0)
    ai_status = ai_verdict.get("status", "UNCERTAIN")
    manip_prob = manip_verdict.get("probability", 50.0)
    manip_status = manip_verdict.get("status", "UNCERTAIN")

    # AI-generated software confirmed in EXIF
    ai_software = metadata.get("ai_software_detected", False)

    # Calculate a combined "concern score"
    concern = (ai_prob * 0.45) + (manip_prob * 0.55)
    concern = round(concern, 2)

    # Overall confidence = how certain we are about the verdict
    # Weighted toward whichever signal is stronger
    confidence = round(max(ai_verdict.get("confidence", 50), manip_verdict.get("confidence", 50)), 2)

    if ai_software:
        status = "AI-GENERATED"
        confidence = 95.0
    elif ai_status == "YES" and manip_status == "YES":
        # Both AI and manipulated — likely deceptive
        status = "FAKE / DECEPTIVE"
        confidence = round(min(99.0, (ai_prob + manip_prob) / 2.0), 2)
    elif ai_status == "YES" and manip_status != "YES":
        # AI generated but not additionally manipulated
        status = "AI-GENERATED"
        confidence = ai_verdict.get("confidence", 70.0)
    elif manip_status == "YES" and ai_status != "YES":
        # Manipulated but not AI generated
        if manip_prob > 80:
            status = "FAKE / DECEPTIVE"
        else:
            status = "LIKELY MANIPULATED"
        confidence = manip_verdict.get("confidence", 70.0)
    elif concern > 65:
        status = "SUSPICIOUS"
        confidence = round(concern * 0.9, 2)
    elif concern < 30:
        status = "AUTHENTIC"
        confidence = round((1.0 - concern / 100.0) * 100.0, 2)
    elif concern < 45:
        status = "LIKELY AUTHENTIC"
        confidence = round((1.0 - concern / 100.0) * 80.0, 2)
    else:
        status = "SUSPICIOUS"
        confidence = round(concern * 0.85, 2)

    confidence = round(max(10.0, min(99.0, confidence)), 2)

    return {
        "status": status,
        "confidence": confidence,
    }


# ─── Main Pipeline Orchestrator ─────────────────────────────────────────────────

def analyze_image_verification(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Main image verification orchestrator.
    Accepts raw image bytes (NOT a file path) for Vercel compatibility.
    Processes entirely in memory — no temporary files written.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        filename:    Original filename (used for extension-based checks).

    Returns structured result dict with three-part analysis:
    {
      "success": True,
      "overall": { "status": str, "confidence": float },
      "ai_generated": { "status": str, "confidence": float, "probability": float, "reasons": list },
      "manipulation": { "status": str, "confidence": float, "probability": float, "reasons": list },
      "forensics": { "ela_score": float, "noise_score": float, "compression_score": float,
                     "splicing_probability": float, "copy_move_probability": float },
      "metadata": { "available": bool, "camera": str, "software": str, "editing_software_detected": bool },
      "reason": [str, ...],
      "model_information": { "ai_detector": str, "forensic_analysis": bool, "fallback_mode": bool }
    }
    """
    # ── Run all forensic signals ──
    ela_score    = run_ela_analysis(image_bytes)
    noise_score  = run_noise_analysis(image_bytes)
    freq_score   = run_frequency_analysis(image_bytes)
    jpeg_score   = run_jpeg_artifact_analysis(image_bytes, filename=filename)
    meta_result  = run_metadata_analysis(image_bytes)
    copy_move    = run_copy_move_detection(image_bytes)
    hf_score, hf_used = run_ai_detection_model(image_bytes)

    # ── Three-part verdicts ──
    ai_verdict   = _determine_ai_verdict(hf_score, noise_score, freq_score, meta_result)
    manip_verdict = _determine_manipulation_verdict(ela_score, jpeg_score, copy_move, meta_result)
    overall      = _determine_overall_verdict(ai_verdict, manip_verdict, meta_result)

    # ── Structured forensics block ──
    forensics = {
        "ela_score":            ela_score,
        "noise_score":          noise_score,
        "compression_score":    freq_score,
        "splicing_probability": manip_verdict.get("splicing_probability", round(ela_score * 0.85, 2)),
        "copy_move_probability": copy_move,
    }

    # ── Metadata block ──
    cam_parts = [meta_result.get("camera_make"), meta_result.get("camera_model")]
    metadata_block = {
        "available":                meta_result.get("has_exif", False),
        "camera":                   " ".join(p for p in cam_parts if p) or None,
        "software":                 meta_result.get("software"),
        "editing_software_detected": meta_result.get("editing_software_detected", False),
        "ai_software_detected":     meta_result.get("ai_software_detected", False),
        "has_gps":                  meta_result.get("has_gps", False),
        "datetime":                 meta_result.get("datetime"),
    }

    # ── Compile reason list ──
    reason_list: List[str] = []
    reason_list.extend(ai_verdict.get("reasons", [])[:2])
    reason_list.extend(manip_verdict.get("reasons", [])[:2])
    if not reason_list:
        reason_list.append("Forensic analysis complete — see individual sections for details.")

    # ── Natural-language summary ──
    nl_reason = generate_forensic_reason(
        overall["status"], overall["confidence"],
        ai_verdict, manip_verdict, forensics, metadata_block
    )

    # ── Model information ──
    model_info = {
        "ai_detector":     "umm-maybe/AI-image-detector" if hf_used else "forensic-only",
        "forensic_analysis": True,
        "fallback_mode":   not hf_used,
        "groq_reason":     True,  # always attempted
    }
    if not hf_used:
        reason_list.append("Advanced AI model unavailable; result based on forensic analysis only.")

    return {
        "success":       True,
        "overall":       overall,
        "ai_generated":  {
            "status":     ai_verdict["status"],
            "confidence": ai_verdict["confidence"],
            "probability": ai_verdict["probability"],
            "reason":     " ".join(ai_verdict.get("reasons", [])[:2]),
        },
        "manipulation":  {
            "status":     manip_verdict["status"],
            "confidence": manip_verdict["confidence"],
            "probability": manip_verdict["probability"],
            "reason":     " ".join(manip_verdict.get("reasons", [])[:2]),
        },
        "forensics":     forensics,
        "metadata":      metadata_block,
        "reason":        reason_list,
        "explanation":   nl_reason,
        "model_information": model_info,
        # Legacy compatibility fields
        "image_hash":    compute_image_hash(image_bytes),
        "hf_model_used": hf_used,
    }


# ─── Legacy compatibility wrappers ──────────────────────────────────────────────

def analyze_image_verification_from_path(image_path: str) -> Dict[str, Any]:
    """
    Legacy wrapper that accepts a file path instead of bytes.
    Kept for backward compatibility with old routes that still use paths.
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        filename = os.path.basename(image_path)
        return analyze_image_verification(image_bytes, filename=filename)
    except Exception as exc:
        logger.error(f"Failed to read image from path {image_path}: {exc}")
        return {}


def extract_metadata(image_path: str) -> Dict[str, str]:
    """Legacy function — kept for backward compatibility."""
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
    Legacy analyze_image function — delegates to the full verification pipeline.
    Kept for backward compatibility with the old /api/image/analyze route.
    """
    result = analyze_image_verification_from_path(image_path)
    if not result:
        return {}
    # Map to old schema
    result["filename"]          = filename or os.path.basename(image_path)
    result["label"]             = result.get("overall", {}).get("status", "Unknown")
    result["prediction"]        = result["label"]
    result["confidence"]        = result.get("overall", {}).get("confidence", 50.0)
    result["explanation"]       = result.get("explanation", "")
    result["ai_probability"]    = result.get("ai_generated", {}).get("probability", 50.0)
    result["human_probability"] = round(100.0 - result["ai_probability"], 2)
    result["manipulation_score"] = result.get("manipulation", {}).get("probability", 50.0)
    result["suspicious_regions"] = []
    return result
