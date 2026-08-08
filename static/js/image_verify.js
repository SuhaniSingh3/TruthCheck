/**
 * TruthCheck — AI Image Verification Frontend (v2)
 * =================================================
 * Handles all client-side logic for the /image-detect page:
 *  - Drag-and-drop, file browse, clipboard paste (Ctrl+V), screenshot capture
 *  - Image preview with SHA-256 hash computation via Web Crypto API
 *  - FormData POST to /api/verify-image with animated loading steps
 *  - Renders the 3-part result dashboard:
 *      A. Overall Authenticity verdict
 *      B. AI-Generated Detection card
 *      C. Manipulation Detection card
 *  - Forensic signal bars (ELA, Noise, Frequency, Splicing, Copy-Move)
 *  - EXIF Metadata section
 *  - Natural-language explanation
 *  - Confidence gauge (canvas arc)
 *  - LocalStorage-based verification history (max 10 entries)
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
  const hashRow         = document.getElementById("iv-hash-row");
  const dimensionsRow   = document.getElementById("iv-dimensions-row");
  const typeRow         = document.getElementById("iv-type-row");
  const btnRemove       = document.getElementById("iv-btn-remove");
  const btnVerify       = document.getElementById("iv-btn-verify");
  const verifyIcon      = document.getElementById("iv-verify-icon");
  const verifyLabel     = document.getElementById("iv-verify-label");
  const loader          = document.getElementById("iv-loader");
  const loaderLabel     = document.getElementById("iv-loader-label");
  const resultDashboard = document.getElementById("iv-result-dashboard");
  const historyList     = document.getElementById("iv-history-list");
  const historyEmpty    = document.getElementById("iv-history-empty");
  const btnClearHistory = document.getElementById("iv-btn-clear-history");
  const btnDownloadRpt  = document.getElementById("iv-btn-download-report");
  const btnNew          = document.getElementById("iv-btn-new");
  const duplicateWarn   = document.getElementById("iv-duplicate-warn");
  const gaugeCanvas     = document.getElementById("iv-gauge-canvas");

  const STORAGE_KEY  = "truthcheck_iv_history_v2";
  const MAX_HISTORY  = 10;
  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  const MAX_BYTES    = 20 * 1024 * 1024; // 20 MB

  let currentFile = null;
  let currentHash = "";
  let lastResult  = null;

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
    uploadZone.addEventListener("click", (e) => {
      if (e.target === fileInput || e.target.closest("button")) return;
      fileInput.click();
    });
    uploadZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") fileInput.click();
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });
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
    btnBrowse?.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
    btnPaste?.addEventListener("click", async (e) => { e.stopPropagation(); await pasteFromClipboard(); });
    btnScreenshot?.addEventListener("click", async (e) => { e.stopPropagation(); await captureScreenshot(); });
    btnRemove?.addEventListener("click", resetUpload);
    btnVerify?.addEventListener("click", startVerification);
    btnNew?.addEventListener("click", resetAll);
    btnDownloadRpt?.addEventListener("click", downloadReport);
    btnClearHistory?.addEventListener("click", clearHistory);
  }

  /* ══════════════════════════════════════════════════════════════════
     CLIPBOARD PASTE (Ctrl+V on document)
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
      showToast("Screenshot capture not supported. Please upload a screenshot file instead.", "warn");
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
    if (!ALLOWED_TYPES.includes(file.type) && !isAllowedByName(file.name)) {
      showToast(`Unsupported file type: ${file.type || file.name}. Please use JPG, JPEG, PNG, or WEBP.`, "error");
      return;
    }
    if (file.size > MAX_BYTES) {
      showToast(`File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB. Maximum is 20 MB.`, "error");
      return;
    }

    currentFile = file;
    currentHash = "";
    lastResult  = null;

    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewFilename.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    typeRow.textContent = file.type || "image/" + file.name.split(".").pop();
    hashRow.textContent = "Computing SHA-256…";
    dimensionsRow.textContent = "Loading…";

    const tempImg = new Image();
    tempImg.onload = () => {
      dimensionsRow.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
      URL.revokeObjectURL(url);
    };
    tempImg.src = url;

    computeSHA256(file).then((hash) => {
      currentHash = hash;
      hashRow.textContent = hash;
      if (isDuplicate(hash)) {
        duplicateWarn.style.display = "block";
        setTimeout(() => { duplicateWarn.style.display = "none"; }, 5000);
      } else {
        duplicateWarn.style.display = "none";
      }
    });

    previewPanel.style.display = "block";
    loader.style.display = "none";
    resultDashboard.style.display = "none";
    resetSteps();
    btnVerify.disabled = false;
    verifyIcon.textContent = "🔬";
    verifyLabel.textContent = "Verify Image";

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

    loader.style.display = "flex";
    resultDashboard.style.display = "none";

    const steps = ["ela", "noise", "freq", "jpeg", "meta", "copymove", "ai"];
    const stepLabels = [
      "Running Error Level Analysis…",
      "Analyzing noise patterns…",
      "Frequency domain (FFT) analysis…",
      "JPEG artifact scoring…",
      "Extracting EXIF metadata…",
      "Copy-move detection…",
      "AI generation analysis…",
    ];
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      if (stepIdx < steps.length) {
        if (stepIdx > 0) markStepDone(steps[stepIdx - 1]);
        markStepActive(steps[stepIdx]);
        loaderLabel.textContent = stepLabels[stepIdx];
        stepIdx++;
      }
    }, 700);

    try {
      const formData = new FormData();
      formData.append("file", currentFile);

      const resp = await fetch("/api/verify-image", {
        method: "POST",
        body: formData,
      });

      clearInterval(stepInterval);
      steps.forEach(markStepDone);
      loaderLabel.textContent = "Rendering results…";

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Unknown server error" }));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      lastResult = data;

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
     RESULT RENDERING — 3-Part Dashboard
  ══════════════════════════════════════════════════════════════════ */
  function renderResult(data) {
    // Extract 3-part structure
    const overall   = data.overall   || {};
    const aiGen     = data.ai_generated || {};
    const manip     = data.manipulation || {};
    const forensics = data.forensics || {};
    const metadata  = data.metadata  || {};
    const timestamp = data.timestamp  ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString();

    // ── OVERALL RESULT ──
    renderOverallCard(overall);

    // ── AI-GENERATED CARD ──
    renderDetectCard({
      statusEl:  "iv-ai-status",
      badgeEl:   "iv-ai-badge",
      confBarEl: "iv-ai-conf-bar",
      confValEl: "iv-ai-conf-val",
      probBarEl: "iv-ai-prob-bar",
      probValEl: "iv-ai-prob-val",
      reasonEl:  "iv-ai-reason",
      status:    aiGen.status    || "UNCERTAIN",
      confidence: aiGen.confidence || 0,
      probability: aiGen.probability || 0,
      reason:    aiGen.reason    || "Analysis complete.",
      barColor:  aiGen.status === "YES" ? "iv-bar-purple" : (aiGen.status === "NO" ? "iv-bar-green" : "iv-bar-gray"),
    });

    // ── MANIPULATION CARD ──
    renderDetectCard({
      statusEl:  "iv-manip-status",
      badgeEl:   "iv-manip-badge",
      confBarEl: "iv-manip-conf-bar",
      confValEl: "iv-manip-conf-val",
      probBarEl: "iv-manip-prob-bar",
      probValEl: "iv-manip-prob-val",
      reasonEl:  "iv-manip-reason",
      status:    manip.status    || "UNCERTAIN",
      confidence: manip.confidence || 0,
      probability: manip.probability || 0,
      reason:    manip.reason    || "Analysis complete.",
      barColor:  manip.status === "YES" ? "iv-bar-red" : (manip.status === "NO" ? "iv-bar-green" : "iv-bar-gray"),
    });

    // ── FORENSIC BARS ──
    animateForensicBar("ela",       forensics.ela_score,            200);
    animateForensicBar("noise",     forensics.noise_score,          350);
    animateForensicBar("freq",      forensics.compression_score,    500);
    animateForensicBar("splice",    forensics.splicing_probability, 650);
    animateForensicBar("copymove",  forensics.copy_move_probability,800);

    // ── METADATA ──
    renderMetadata(metadata, timestamp);

    // ── EXPLANATION ──
    const expEl = document.getElementById("iv-explanation-text");
    if (expEl) expEl.textContent = data.explanation || (data.reason || []).join(" ") || "Analysis complete.";

    // ── MODEL BANNER ──
    renderModelBanner(data.model_information || {});

    // ── SHOW DASHBOARD ──
    resultDashboard.style.display = "block";
    resultDashboard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ── Overall Card ── */
  function renderOverallCard(overall) {
    const status     = overall.status     || "UNCERTAIN";
    const confidence = parseFloat(overall.confidence) || 0;

    const card       = document.getElementById("iv-overall-card");
    const iconEl     = document.getElementById("iv-overall-icon");
    const labelEl    = document.getElementById("iv-overall-label");
    const subEl      = document.getElementById("iv-overall-sub");

    // Map status → styling
    const STATUS_MAP = {
      "AUTHENTIC":          { cls: "iv-overall-authentic",   labelCls: "iv-label-authentic",   icon: "✅", sub: "This image appears to be authentic and unmodified." },
      "LIKELY AUTHENTIC":   { cls: "iv-overall-authentic",   labelCls: "iv-label-authentic",   icon: "✅", sub: "This image is likely authentic with minor or no modifications." },
      "SUSPICIOUS":         { cls: "iv-overall-suspicious",  labelCls: "iv-label-suspicious",  icon: "⚠️", sub: "Suspicious patterns detected — treat with caution." },
      "LIKELY MANIPULATED": { cls: "iv-overall-manipulated", labelCls: "iv-label-manipulated", icon: "🔴", sub: "This image shows strong signs of digital manipulation." },
      "FAKE / DECEPTIVE":   { cls: "iv-overall-fake",        labelCls: "iv-label-fake",        icon: "🚨", sub: "Multiple strong indicators of deceptive or fabricated content." },
      "AI-GENERATED":       { cls: "iv-overall-ai",          labelCls: "iv-label-ai",          icon: "🤖", sub: "This image was most likely created by an AI generation system." },
      "UNCERTAIN":          { cls: "iv-overall-uncertain",   labelCls: "iv-label-uncertain",   icon: "❓", sub: "Insufficient evidence to make a determination." },
    };

    const mapping = STATUS_MAP[status] || STATUS_MAP["UNCERTAIN"];

    // Remove all variant classes
    card.className = "iv-overall-card";
    card.classList.add(mapping.cls);

    iconEl.textContent  = mapping.icon;
    labelEl.textContent = status;
    labelEl.className   = `iv-overall-label ${mapping.labelCls}`;
    subEl.textContent   = `Confidence: ${confidence.toFixed(1)}% — ${mapping.sub}`;

    drawGauge(gaugeCanvas, confidence, status);
  }

  /* ── Detection Card (AI / Manipulation) ── */
  function renderDetectCard(opts) {
    const { status, confidence, probability } = opts;

    // Status element
    const statusEl = document.getElementById(opts.statusEl);
    if (statusEl) {
      statusEl.textContent  = status;
      statusEl.className    = "iv-detect-status " + statusClass(status);
    }

    // Badge
    const badgeEl = document.getElementById(opts.badgeEl);
    if (badgeEl) {
      badgeEl.textContent = status;
      badgeEl.className   = "iv-detect-badge " + badgeClass(status);
    }

    // Confidence bar
    animateMetricBar(opts.confBarEl, opts.confValEl, confidence, "iv-bar-gray");

    // Probability bar
    animateMetricBar(opts.probBarEl, opts.probValEl, probability, opts.barColor);

    // Reason
    const reasonEl = document.getElementById(opts.reasonEl);
    if (reasonEl) reasonEl.textContent = opts.reason || "—";
  }

  function statusClass(status) {
    if (status === "YES")       return "iv-status-yes";
    if (status === "NO")        return "iv-status-no";
    return "iv-status-uncertain";
  }

  function badgeClass(status) {
    if (status === "YES")       return "iv-badge-yes";
    if (status === "NO")        return "iv-badge-no";
    return "iv-badge-uncertain";
  }

  /* ── Metadata Section ── */
  function renderMetadata(meta, timestamp) {
    setBoolChip("iv-meta-exif",    meta.available, "Available",     "Not Available");
    setBoolChip("iv-meta-gps",     meta.has_gps,   "Present",       "Not found");
    setTextChip("iv-meta-camera",  meta.camera     || "Unknown");
    setTextChip("iv-meta-sw",      meta.software   || "None detected");
    setBoolChip("iv-meta-editing", meta.editing_software_detected, "⚠️ Detected", "Not detected", true);
    setBoolChip("iv-meta-aisoft",  meta.ai_software_detected,     "⚠️ Detected", "Not detected", true);
    setTextChip("iv-meta-dt",      meta.datetime   || "—");
    setTextChip("iv-meta-ts",      timestamp);
  }

  function setBoolChip(id, value, trueText, falseText, invertColors = false) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value ? trueText : falseText;
    if (invertColors) {
      el.className = value ? "iv-meta-chip-val iv-chip-warn" : "iv-meta-chip-val iv-chip-yes";
    } else {
      el.className = value ? "iv-meta-chip-val iv-chip-yes" : "iv-meta-chip-val iv-chip-no";
    }
  }

  function setTextChip(id, text) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.className = "iv-meta-chip-val"; }
  }

  /* ── Model Banner ── */
  function renderModelBanner(modelInfo) {
    const banner  = document.getElementById("iv-model-banner");
    const tag     = document.getElementById("iv-model-tag");
    const infoTxt = document.getElementById("iv-model-info-text");
    if (!banner || !tag) return;

    if (modelInfo.fallback_mode) {
      tag.textContent = "Forensic Analysis Only";
      tag.className   = "iv-model-tag fallback";
      if (infoTxt) infoTxt.textContent = "Advanced AI model unavailable — result based on forensic signal analysis.";
    } else {
      tag.textContent = modelInfo.ai_detector || "HuggingFace CNN";
      tag.className   = "iv-model-tag";
      if (infoTxt) infoTxt.textContent = "AI classification model + forensic signals.";
    }
    banner.style.display = "flex";
  }

  /* ── Confidence Gauge (canvas arc) ── */
  function drawGauge(canvas, value, status) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2, cy = h - 8, r = h - 16;
    const startAngle = Math.PI, endAngle = Math.PI * 2;

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.lineWidth = 10;
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineCap = "round";
    ctx.stroke();

    const normalized = Math.min(100, Math.max(0, value)) / 100;
    const fillEnd    = startAngle + normalized * Math.PI;

    const COLOR_MAP = {
      "AUTHENTIC":          "#10b981",
      "LIKELY AUTHENTIC":   "#34d399",
      "SUSPICIOUS":         "#f59e0b",
      "LIKELY MANIPULATED": "#ef4444",
      "FAKE / DECEPTIVE":   "#dc2626",
      "AI-GENERATED":       "#8b5cf6",
      "UNCERTAIN":          "#6b7280",
    };
    const color = COLOR_MAP[status] || "#6b7280";

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillEnd);
    ctx.lineWidth = 10;
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = "bold 18px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(1) + "%", cx, cy - 10);
  }

  /* ── Forensic Signal Bars ── */
  function animateForensicBar(key, score, delay) {
    const barEl  = document.getElementById(`iv-bar-${key}`);
    const textEl = document.getElementById(`iv-score-${key}`);
    if (!barEl || !textEl) return;

    const value = parseFloat(score) || 0;
    barEl.style.width = "0%";
    barEl.className   = "iv-signal-bar-fill";

    let colorClass;
    if (value < 35)      colorClass = "iv-bar-low";
    else if (value < 65) colorClass = "iv-bar-medium";
    else                 colorClass = "iv-bar-high";

    setTimeout(() => {
      barEl.classList.add(colorClass);
      barEl.style.width  = Math.min(100, value) + "%";
      textEl.textContent = value.toFixed(0) + "%";
    }, delay);
  }

  /* ── Detection Card Metric Bars ── */
  function animateMetricBar(barId, valId, score, colorClass) {
    const barEl  = document.getElementById(barId);
    const valEl  = document.getElementById(valId);
    if (!barEl || !valEl) return;

    const value = parseFloat(score) || 0;
    barEl.style.width = "0%";
    barEl.className   = `iv-metric-bar ${colorClass}`;

    setTimeout(() => {
      barEl.style.width  = Math.min(100, value) + "%";
      valEl.textContent  = value.toFixed(1) + "%";
    }, 150);
  }

  /* ══════════════════════════════════════════════════════════════════
     HISTORY
  ══════════════════════════════════════════════════════════════════ */
  function loadHistory() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
    catch { return []; }
  }

  function saveHistory(hist) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(hist));
  }

  function addToHistory(data, filename, hash) {
    const hist = loadHistory();
    const overallStatus = data.overall?.status || "UNCERTAIN";
    const overallConf   = data.overall?.confidence || 0;

    const entry = {
      id:         Date.now(),
      filename,
      hash,
      status:     overallStatus,
      confidence: overallConf,
      timestamp:  data.timestamp || new Date().toISOString(),
      thumbnail:  null,
    };

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
    return loadHistory().some((e) => e.hash === hash);
  }

  function renderHistory() {
    const hist = loadHistory();
    if (hist.length === 0) {
      historyEmpty.style.display = "block";
      historyList.innerHTML = "";
      return;
    }
    historyEmpty.style.display = "none";

    const STATUS_CLASSES = {
      "AUTHENTIC":          "iv-hv-authentic",
      "LIKELY AUTHENTIC":   "iv-hv-authentic",
      "SUSPICIOUS":         "iv-hv-suspicious",
      "LIKELY MANIPULATED": "iv-hv-manipulated",
      "FAKE / DECEPTIVE":   "iv-hv-fake",
      "AI-GENERATED":       "iv-hv-ai",
      "UNCERTAIN":          "iv-hv-uncertain",
    };

    historyList.innerHTML = hist.map((e) => {
      const vcCls  = STATUS_CLASSES[e.status] || "iv-hv-uncertain";
      const ts     = e.timestamp ? new Date(e.timestamp).toLocaleString() : "—";
      const thumb  = e.thumbnail
        ? `<img class="iv-history-thumb" src="${e.thumbnail}" alt="thumbnail">`
        : `<div class="iv-history-thumb" style="display:flex;align-items:center;justify-content:center;font-size:1.2rem">🖼️</div>`;
      return `
        <div class="iv-history-item glass">
          ${thumb}
          <div class="iv-history-info">
            <div class="iv-history-name">${escHtml(e.filename)}</div>
            <div class="iv-history-ts">${ts}</div>
          </div>
          <span class="iv-history-verdict ${vcCls}">${e.status || "—"}</span>
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
     PDF REPORT
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
    resultDashboard.style.display = "none";
    uploadZone.scrollIntoView({ behavior: "smooth" });
  }

  function resetSteps() {
    document.querySelectorAll(".iv-step").forEach((s) => { s.className = "iv-step"; });
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

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

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
    setTimeout(() => toast.remove(), 4500);
  }

  /* ── Boot ── */
  if (document.readyState !== "loading") {
    init();
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }

})();
