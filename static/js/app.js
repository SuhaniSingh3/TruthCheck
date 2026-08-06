/**
 * TruthCheck Enterprise Platform Frontend Logic
 * Includes dynamic particle background, smart universal detection, AJAX analysis, and AI assistant chatbot.
 */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initParticles();
  initChatWidget();
  initUniversalDetector();
  initLanguageSelector();
});

/* ─── 0. Dark / Light Theme Toggle ─── */
function initThemeToggle() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;

  // Ensure the saved theme is applied (backup if inline script missed)
  const saved = localStorage.getItem('truthcheck_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('truthcheck_theme', next);
  });
}

/* ─── 1. Particle Canvas Animation ─── */
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  for (let i = 0; i < 55; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      radius: Math.random() * 2 + 1
    });
  }

  function getThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
      fill: isDark ? 'rgba(168, 85, 247, 0.4)' : 'rgba(124, 58, 237, 0.25)',
      stroke: isDark ? 'rgba(6, 182, 212, 0.12)' : 'rgba(8, 145, 178, 0.1)'
    };
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    const { fill, stroke } = getThemeColors();
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;

    for (let i = 0; i < particles.length; i++) {
      let p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        let p2 = particles[j];
        let dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 110) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

/* ─── 2. AI Chatbot Assistant ─── */
function initChatWidget() {
  const toggleBtn = document.getElementById('chat-toggle');
  const panel = document.getElementById('chat-panel');
  const closeBtn = document.getElementById('chat-close');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const messagesBox = document.getElementById('chat-messages');

  if (!toggleBtn || !panel) return;

  toggleBtn.addEventListener('click', () => {
    panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
  });

  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      panel.style.display = 'none';
    });
  }

  const sendMessage = async () => {
    const text = input.value.trim();
    if (!text) return;

    // Append user message
    const uMsg = document.createElement('div');
    uMsg.className = 'msg user-msg';
    uMsg.textContent = text;
    messagesBox.appendChild(uMsg);
    input.value = '';
    messagesBox.scrollTop = messagesBox.scrollHeight;

    // Call chat API
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await resp.json();
      const aiMsg = document.createElement('div');
      aiMsg.className = 'msg ai-msg';
      aiMsg.textContent = data.response || "I analyzed the latest verification context.";
      messagesBox.appendChild(aiMsg);
      messagesBox.scrollTop = messagesBox.scrollHeight;
    } catch (e) {
      const errMsg = document.createElement('div');
      errMsg.className = 'msg ai-msg';
      errMsg.textContent = 'Error connecting to AI chat service.';
      messagesBox.appendChild(errMsg);
    }
  };

  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (input) {
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
  }
}

/* ─── 3. Universal Smart Detector ─── */
function initUniversalDetector() {
  const input = document.getElementById('smart-input');
  const badge = document.getElementById('detection-badge');
  const btn = document.getElementById('analyze-btn');
  const resultBox = document.getElementById('result-section');

  if (!input || !btn) return;

  input.addEventListener('input', () => {
    const val = input.value.trim();
    if (val.includes('youtube.com') || val.includes('youtu.be')) {
      badge.textContent = 'Detected: YouTube Video Link';
    } else if (val.startsWith('http://') || val.startsWith('https://')) {
      badge.textContent = 'Detected: Web Article URL';
    } else if (val.length > 0) {
      badge.textContent = 'Detected: News Text / Headline';
    } else {
      badge.textContent = 'Auto-Detecting Type...';
    }
  });

  btn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return alert("Please enter news text, article URL, or YouTube link.");

    btn.disabled = true;
    btn.textContent = "Analyzing & Fact-Checking...";
    if (resultBox) resultBox.style.display = 'none';

    const lang = document.getElementById('lang-select')?.value || 'en';

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, response_lang: lang })
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        alert("Verification error: " + (data.error || "Unable to complete analysis."));
      } else {
        renderAnalysisResult(data);
      }
    } catch (err) {
      alert("Network error occurred during verification.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Verify Authenticity →";
    }
  });
}

function renderAnalysisResult(data) {
  const resultBox = document.getElementById('result-section');
  if (!resultBox) return;

  const verdictEl = document.getElementById('result-verdict');
  const confEl = document.getElementById('result-confidence');
  const riskEl = document.getElementById('result-risk');
  const summaryEl = document.getElementById('result-summary');
  const reasonsEl = document.getElementById('result-reasons');

  verdictEl.textContent = `Verdict: ${data.label || data.prediction || 'CHECKED'}`;
  confEl.textContent = `${data.confidence || '--'}%`;
  riskEl.textContent = (data.risk_level || 'Medium').toUpperCase();
  summaryEl.textContent = data.summary || "Fact-checking complete.";

  reasonsEl.innerHTML = '';
  const listItems = data.reasons || data.claims || [];
  if (Array.isArray(listItems)) {
    listItems.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      reasonsEl.appendChild(li);
    });
  }

  resultBox.style.display = 'block';
  resultBox.scrollIntoView({ behavior: 'smooth' });
}

/* ─── 4. Multilingual Language Selector ─── */
function initLanguageSelector() {
  const select = document.getElementById('lang-select');
  if (!select) return;
  const saved = localStorage.getItem('truthcheck_lang') || 'en';
  select.value = saved;

  select.addEventListener('change', () => {
    localStorage.setItem('truthcheck_lang', select.value);
  });
}
