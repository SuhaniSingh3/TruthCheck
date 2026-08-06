/**
 * TruthCheck — AI Image Verification Frontend
 * =============================================
 * Handles all client-side logic for the /image-detect page:
 *  - Drag-and-drop, file browse, clipboard paste (Ctrl+V), screenshot capture
 *  - Image preview with SHA-256 hash computation via Web Crypto API
 *  - FormData POST to /api/verify-image with animated loading steps
 *  - Confidence gauge animation (canvas)
 *  - Animated signal bar rendering
 *  - LocalStorage-based verification history (max 10 entries)
 *  - Duplicate image detection via hash comparison
 *  - PDF report via window.print()
 *
 * NOTE: This file is isolated and does NOT touch app.js globals.
 */

(function () {
  "use strict";

  /* ── DOM references ─────────────────────────────────────────────── */
  const uploadZone      = document.getElementById("iv-upload-zone");
  const fileInput       = document.getElementById("iv-file-input");
  const btnBrowse       = document.getElementById("iv-btn-browse");
  const btnPaste        = document.getElementById("iv-btn-paste");
  const btnScreenshot   = document.getElementById("iv-btn-screenshot");
  const previewPanel    = document.getElementById("iv-preview-panel");
  const previewImg      = document.getElementById("iv-preview-img");
  const previewFilename = document.getElementById("iv-preview-filename");
  const previewSize     = document.getElementById("iv-preview-size");
  const previewHashDisp = document.getElementById("iv-preview-hash-display");
  const hashRow         = document.getElementById("iv-hash-row");
  const dimensionsRow   = document.getElementById("iv-dimensions-row");
  const typeRow         = document.getElementById("iv-type-row");
  const btnRemove       = document.getElementById("iv-btn-remove");
  const btnVerify       = document.getElementById("iv-btn-verify");
  const verifyIcon      = document.getElementById("iv-verify-icon");
  const verifyLabel     = document.getElementById("iv-verify-label");
  const loader          = document.getElementById("iv-loader");
  const loaderLabel     = document.getElementById("iv-loader-label");
  const resultCard      = document.getElementById("iv-result-card");
  const resultHeader    = document.getElementById("iv-result-header");
  const verdictIcon     = document.getElementById("iv-verdict-icon");
  const verdictLabel    = document.getElementById("iv-verdict-label");
  const verdictSub      = document.getElementById("iv-verdict-sub");
  const reasonText      = document.getElementById("iv-reason-text");
  const gaugeCanvas     = document.getElementById("iv-gauge-canvas");
  const historyList     = document.getElementById("iv-history-list");
  const historyEmpty    = document.getElementById("iv-history-empty");
  const btnClearHistory = document.getElementById("iv-btn-clear-history");
  const btnDownloadRpt  = document.getElementById("iv-btn-download-report");
  const btnNew          = document.getElementById("iv-btn-new");
  const duplicateWarn   = document.getElementById("iv-duplicate-warn");

  const STORAGE_KEY = "truthcheck_iv_history";
  const MAX_HISTORY = 10;
  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  const MAX_BYTES = 20 * 1024 * 1024; // 20 MB

  /** Currently selected File object */
  let currentFile = null;
  /** SHA-256 hash of current file (hex string) */
  let currentHash = "";
  /** Last analysis result */
  let lastResult = null;

  /* ══════════════════════════════════════════════════════════════════
     INIT
  ══════════════════════════════════════════════════════════════════ */
  function init() {
    setupUploadZone();
    setupButtons();
    setupClipboardPaste();
    renderHistory();
  }

  /* ══════════════════════════════════════════════════════════════════
     UPLOAD ZONE
  ══════════════════════════════════════════════════════════════════ */
  function setupUploadZone() {
    // Click anywhere in zone → browse
    uploadZone.addEventListener("click", (e) => {
      if (e.target === fileInput) return;
      fileInput.click();
    });

    // Keyboard accessibility
    uploadZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") fileInput.click();
    });

    // File input change
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    // Drag & drop
    uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadZone.classList.add("iv-drag-over");
    });
    uploadZone.addEventListener("dragleave", (e) => {
      if (!uploadZone.contains(e.relatedTarget)) {
        uploadZone.classList.remove("iv-drag-over");
      }
    });
    uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadZone.classList.remove("iv-drag-over");
      const files = e.dataTransfer?.files;
      if (files?.length > 0) handleFile(files[0]);
    });
  }

  function setupButtons() {
    btnBrowse?.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    btnPaste?.addEventListener("click", async (e) => {
      e.stopPropagation();
      await pasteFromClipboard();
    });

    btnScreenshot?.addEventListener("click", async (e) => {
      e.stopPropagation();
      await captureScreenshot();
    });

    btnRemove?.addEventListener("click", resetUpload);
    btnVerify?.addEventListener("click", startVerification);
    btnNew?.addEventListener("click", resetAll);
    btnDownloadRpt?.addEventListener("click", downloadReport);
    btnClearHistory?.addEventListener("click", clearHistory);
  }

  /* ══════════════════════════════════════════════════════════════════
     CLIPBOARD PASTE (Ctrl+V on entire document)
  ══════════════════════════════════════════════════════════════════ */
  function setupClipboardPaste() {
    document.addEventListener("paste", async (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            e.preventDefault();
            showPasteHint();
            handleFile(file);
            return;
          }
        }
      }
    });
  }

  async function pasteFromClipboard() {
    try {
      const items = await navigator.clipboard.read();
      for (const item of items) {
        for (const type of item.types) {
          if (type.startsWith("image/")) {
            const blob = await item.getType(type);
            const file = new File([blob], `pasted-image.${type.split("/")[1]}`, { type });
            handleFile(file);
            return;
          }
        }
      }
      showToast("No image found in clipboard. Copy an image first, then try again.", "warn");
    } catch {
      showToast("Clipboard access denied. Use Ctrl+V instead.", "warn");
    }
  }

  async function captureScreenshot() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      const track = stream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(track);
      const bitmap = await imageCapture.grabFrame();
      track.stop();

      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      canvas.toBlob((blob) => {
        const file = new File([blob], "screenshot.png", { type: "image/png" });
        handleFile(file);
      }, "image/png");
    } catch {
      showToast("Screenshot capture not supported in this browser. Please upload a screenshot file instead.", "warn");
    }
  }

  function showPasteHint() {
    const hint = document.createElement("div");
    hint.className = "iv-paste-hint";
    hint.textContent = "📋 Image pasted from clipboard!";
    document.body.appendChild(hint);
    setTimeout(() => hint.remove(), 2500);
  }

  /* ══════════════════════════════════════════════════════════════════
     FILE HANDLING
  ══════════════════════════════════════════════════════════════════ */
  function handleFile(file) {
    // Validate type
    if (!ALLOWED_TYPES.includes(file.type) && !isAllowedByName(file.name)) {
      showToast(`Unsupported file type: ${file.type || file.name}. Please use JPG, JPEG, PNG, or WEBP.`, "error");
      return;
    }
    // Validate size
    if (file.size > MAX_BYTES) {
      showToast(`File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB. Maximum is 20 MB.`, "error");
      return;
    }

    currentFile = file;
    currentHash = "";
    lastResult = null;

    // Show preview
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewFilename.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    typeRow.textContent = file.type || "image/" + file.name.split(".").pop();
    hashRow.textContent = "Computing SHA-256…";
    dimensionsRow.textContent = "Loading…";
    previewHashDisp.textContent = "";

    // Get image dimensions
    const tempImg = new Image();
    tempImg.onload = () => {
      dimensionsRow.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
      URL.revokeObjectURL(url);
    };
    tempImg.src = url;

    // Compute SHA-256
    computeSHA256(file).then((hash) => {
      currentHash = hash;
      const short = hash.substring(0, 16) + "…";
      hashRow.textContent = hash;
      previewHashDisp.textContent = "SHA-256: " + short;

      // Check for duplicate
      if (isDuplicate(hash)) {
        duplicateWarn.style.display = "block";
        setTimeout(() => { duplicateWarn.style.display = "none"; }, 5000);
      } else {
        duplicateWarn.style.display = "none";
      }
    });

    // Show preview panel, hide result
    previewPanel.style.display = "block";
    loader.style.display = "none";
    resultCard.style.display = "none";
    resetSteps();
    btnVerify.disabled = false;
    verifyIcon.textContent = "🔬";
    verifyLabel.textContent = "Verify Image";

    // Scroll to preview
    previewPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function isAllowedByName(name) {
    const ext = name.split(".").pop().toLowerCase();
    return ["jpg", "jpeg", "png", "webp"].includes(ext);
  }

  async function computeSHA256(file) {
    try {
      const buffer = await file.arrayBuffer();
      const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    } catch {
      return "hash-unavailable";
    }
  }

  /* ══════════════════════════════════════════════════════════════════
     VERIFICATION
  ══════════════════════════════════════════════════════════════════ */
  async function startVerification() {
    if (!currentFile) return;

    btnVerify.disabled = true;
    verifyIcon.textContent = "⏳";
    verifyLabel.textContent = "Analyzing…";

    // Show loader
    loader.style.display = "flex";
    resultCard.style.display = "none";

    // Animate steps
    const steps = ["ela", "noise", "freq", "jpeg", "meta", "ai"];
    const stepLabels = [
      "Running Error Level Analysis…",
      "Analyzing noise patterns…",
      "Frequency domain analysis…",
      "JPEG artifact scoring…",
      "Extracting EXIF metadata…",
      "AI detection model…",
    ];
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      if (stepIdx < steps.length) {
        if (stepIdx > 0) markStepDone(steps[stepIdx - 1]);
        markStepActive(steps[stepIdx]);
        loaderLabel.textContent = stepLabels[stepIdx];
        stepIdx++;
      }
    }, 600);

    try {
      const formData = new FormData();
      formData.append("file", currentFile);

      const resp = await fetch("/api/verify-image", {
        method: "POST",
        body: formData,
      });

      clearInterval(stepInterval);
      // Mark all steps done
      steps.forEach(markStepDone);
      loaderLabel.textContent = "Rendering results…";

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Unknown server error" }));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      lastResult = data;

      // Short pause for UX smoothness
      await sleep(400);
      loader.style.display = "none";

      renderResult(data);
      addToHistory(data, currentFile.name, currentHash);
      renderHistory();

    } catch (err) {
      clearInterval(stepInterval);
      loader.style.display = "none";
      showToast(`Verification failed: ${err.message}`, "error");
      btnVerify.disabled = false;
      verifyIcon.textContent = "🔬";
      verifyLabel.textContent = "Verify Image";
      resetSteps();
    }
  }

  /* ══════════════════════════════════════════════════════════════════
     RESULT RENDERING
  ══════════════════════════════════════════════════════════════════ */
  function renderResult(data) {
    const prediction = data.prediction || "Unknown";
    const confidence = parseFloat(data.confidence) || 0;
    const reason     = data.reason || "Analysis complete.";
    const details    = data.details || {};
    const meta       = data.metadata || {};
    const timestamp  = data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString();

    // ── Verdict styling ──
    resultCard.className = "iv-result-card glass";
    resultHeader.className = "iv-result-header";

    let stateClass, icon, sub;
    if (prediction === "Fake") {
      stateClass = "iv-state-fake";
      icon = "🚨";
      sub  = `${confidence.toFixed(1)}% confidence — Image likely AI-generated or manipulated`;
    } else if (prediction === "Suspicious") {
      stateClass = "iv-state-suspicious";
      icon = "⚠️";
      sub  = `${confidence.toFixed(1)}% suspicion score — Proceed with caution`;
    } else {
      stateClass = "iv-state-real";
      icon = "✅";
      sub  = `${confidence.toFixed(1)}% confidence — Image appears authentic`;
    }

    resultCard.classList.add(stateClass);
    verdictIcon.textContent = icon;
    verdictLabel.textContent = prediction === "Fake"
      ? "FAKE / MANIPULATED"
      : prediction === "Suspicious"
        ? "SUSPICIOUS"
        : "REAL IMAGE";
    verdictSub.textContent = sub;
    reasonText.textContent = reason;

    // ── Gauge ──
    drawGauge(gaugeCanvas, confidence, prediction);

    // ── Signal bars ──
    animateBar("ela",   details.ela_score,           180);
    animateBar("noise", details.noise_score,          360);
    animateBar("freq",  details.freq_score,           540);
    animateBar("jpeg",  details.jpeg_score,           720);
    animateBar("meta",  details.metadata_score,       900);
    animateBar("ai",    details.ai_detection_score,  1080);

    // ── Metadata chips ──
    const exifEl = document.getElementById("iv-meta-exif");
    const gpsEl  = document.getElementById("iv-meta-gps");
    const camEl  = document.getElementById("iv-meta-camera");
    const swEl   = document.getElementById("iv-meta-sw");
    const ganEl  = document.getElementById("iv-meta-gan");
    const editEl = document.getElementById("iv-meta-edit");
    const hashEl = document.getElementById("iv-meta-hash");
    const tsEl   = document.getElementById("iv-meta-ts");

    setBoolChip(exifEl, meta.has_exif, "Present", "Missing");
    setBoolChip(gpsEl,  meta.has_gps,  "Present", "Not found");
    camEl.textContent  = [meta.camera_make, meta.camera_model].filter(Boolean).join(" ") || "Unknown";
    swEl.textContent   = meta.software || "None detected";
    ganEl.textContent  = details.gan_probability != null
      ? (details.gan_probability * 100).toFixed(1) + "%" : "—";
    editEl.textContent = details.editing_probability != null
      ? (details.editing_probability * 100).toFixed(1) + "%" : "—";
    hashEl.textContent = (data.image_hash || currentHash || "—").substring(0, 32) + "…";
    tsEl.textContent   = timestamp;

    // ── Show card ──
    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function setBoolChip(el, value, trueText, falseText) {
    if (value) {
      el.textContent = trueText;
      el.className = "iv-meta-chip-val iv-chip-yes";
    } else {
      el.textContent = falseText;
      el.className = "iv-meta-chip-val iv-chip-no";
    }
  }

  /* ── Confidence Gauge (canvas arc) ── */
  function drawGauge(canvas, value, prediction) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h - 8;
    const r  = h - 16;
    const startAngle = Math.PI;
    const endAngle   = Math.PI * 2;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.lineWidth = 10;
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineCap = "round";
    ctx.stroke();

    // Value arc
    const normalized = Math.min(100, Math.max(0, value)) / 100;
    const fillEnd    = startAngle + normalized * Math.PI;

    let color;
    if (prediction === "Fake")       color = "#ef4444";
    else if (prediction === "Suspicious") color = "#f59e0b";
    else                             color = "#10b981";

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillEnd);
    ctx.lineWidth = 10;
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.stroke();

    // Label
    ctx.fillStyle = color;
    ctx.font = "bold 18px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(1) + "%", cx, cy - 10);
  }

  /* ── Animated score bars ── */
  function animateBar(key, score, delay) {
    const barEl  = document.getElementById(`iv-bar-${key}`);
    const textEl = document.getElementById(`iv-score-${key}`);
    if (!barEl || !textEl) return;

    const value = parseFloat(score) || 0;
    barEl.style.width = "0%";
    barEl.className   = "iv-signal-bar-fill";

    let colorClass;
    if (value < 40)      colorClass = "iv-bar-low";
    else if (value < 65) colorClass = "iv-bar-medium";
    else                 colorClass = "iv-bar-high";

    setTimeout(() => {
      barEl.classList.add(colorClass);
      barEl.style.width = value + "%";
      textEl.textContent = value.toFixed(0);
    }, delay);
  }

  /* ══════════════════════════════════════════════════════════════════
     HISTORY
  ══════════════════════════════════════════════════════════════════ */
  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  }

  function saveHistory(hist) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(hist));
  }

  function addToHistory(data, filename, hash) {
    const hist = loadHistory();
    const entry = {
      id:         Date.now(),
      filename:   filename,
      hash:       hash,
      prediction: data.prediction,
      confidence: data.confidence,
      timestamp:  data.timestamp || new Date().toISOString(),
      thumbnail:  previewImg.src.startsWith("blob:") ? null : previewImg.src,
    };
    // Store thumbnail as data URL (max ~100KB compressed)
    try {
      const thumbCanvas = document.createElement("canvas");
      thumbCanvas.width  = 64;
      thumbCanvas.height = 48;
      thumbCanvas.getContext("2d").drawImage(previewImg, 0, 0, 64, 48);
      entry.thumbnail = thumbCanvas.toDataURL("image/jpeg", 0.5);
    } catch { /* ignore */ }

    hist.unshift(entry);
    if (hist.length > MAX_HISTORY) hist.splice(MAX_HISTORY);
    saveHistory(hist);
  }

  function isDuplicate(hash) {
    if (!hash) return false;
    const hist = loadHistory();
    return hist.some((e) => e.hash === hash);
  }

  function renderHistory() {
    const hist = loadHistory();
    if (hist.length === 0) {
      historyEmpty.style.display = "block";
      historyList.innerHTML = "";
      return;
    }
    historyEmpty.style.display = "none";
    historyList.innerHTML = hist.map((e) => {
      const vcClass  = e.prediction === "Fake"
        ? "iv-hv-fake"
        : e.prediction === "Suspicious"
          ? "iv-hv-suspicious"
          : "iv-hv-real";
      const ts = e.timestamp ? new Date(e.timestamp).toLocaleString() : "—";
      const thumb = e.thumbnail
        ? `<img class="iv-history-thumb" src="${e.thumbnail}" alt="thumb">`
        : `<div class="iv-history-thumb" style="display:flex;align-items:center;justify-content:center;font-size:1.2rem">🖼️</div>`;
      const hashShort = e.hash ? e.hash.substring(0, 12) + "…" : "—";
      return `
        <div class="iv-history-item glass">
          ${thumb}
          <div class="iv-history-info">
            <div class="iv-history-name">${escHtml(e.filename)}</div>
            <div class="iv-history-ts">${ts}</div>
            <div class="iv-history-hash">${hashShort}</div>
          </div>
          <span class="iv-history-verdict ${vcClass}">${e.prediction || "—"}</span>
          <span class="iv-history-conf">${parseFloat(e.confidence || 0).toFixed(1)}%</span>
        </div>`;
    }).join("");
  }

  function clearHistory() {
    if (!confirm("Clear all verification history?")) return;
    localStorage.removeItem(STORAGE_KEY);
    renderHistory();
  }

  /* ══════════════════════════════════════════════════════════════════
     PDF REPORT (via browser print)
  ══════════════════════════════════════════════════════════════════ */
  function downloadReport() {
    if (!lastResult) {
      showToast("No analysis result to export. Run a verification first.", "warn");
      return;
    }
    window.print();
  }

  /* ══════════════════════════════════════════════════════════════════
     RESET / UTILS
  ══════════════════════════════════════════════════════════════════ */
  function resetUpload() {
    currentFile = null;
    currentHash = "";
    lastResult  = null;
    fileInput.value = "";
    previewPanel.style.display = "none";
    loader.style.display = "none";
    resetSteps();
  }

  function resetAll() {
    resetUpload();
    resultCard.style.display = "none";
    uploadZone.scrollIntoView({ behavior: "smooth" });
  }

  function resetSteps() {
    document.querySelectorAll(".iv-step").forEach((s) => {
      s.className = "iv-step";
    });
    loaderLabel.textContent = "Initializing forensic pipeline…";
  }

  function markStepActive(key) {
    const el = document.querySelector(`[data-step="${key}"]`);
    if (el) { el.classList.remove("iv-step-done"); el.classList.add("iv-step-active"); }
  }

  function markStepDone(key) {
    const el = document.querySelector(`[data-step="${key}"]`);
    if (el) { el.classList.remove("iv-step-active"); el.classList.add("iv-step-done"); }
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(2) + " MB";
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* ── Toast notifications ── */
  function showToast(msg, type = "info") {
    const colors = {
      info:  "linear-gradient(135deg,#a855f7,#06b6d4)",
      warn:  "linear-gradient(135deg,#f59e0b,#fbbf24)",
      error: "linear-gradient(135deg,#ef4444,#f87171)",
    };
    const toast = document.createElement("div");
    toast.style.cssText = `
      position:fixed;bottom:32px;left:50%;transform:translateX(-50%);
      background:${colors[type] || colors.info};color:#fff;
      padding:0.8rem 1.6rem;border-radius:12px;font-size:0.88rem;
      font-weight:600;z-index:9999;box-shadow:0 8px 30px rgba(0,0,0,0.4);
      animation:iv-fadein 0.3s ease;max-width:90vw;text-align:center;`;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }

  /* ── Boot ── */
  document.addEventListener("DOMContentLoaded", init);
  // Also run immediately if DOM is already ready
  if (document.readyState !== "loading") init();

})();
