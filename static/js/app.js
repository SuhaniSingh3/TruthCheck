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
  initClearHistoryButtons();
  initHistoryManager();
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

// Global TruthCheck runtime state for contextual chat & verification tracking
window.TruthCheck = window.TruthCheck || {
  activeContext: null,
  chatHistory: []
};

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
    if (panel.style.display === 'flex' && input) {
      input.focus();
    }
  });

  if (closeBtn) {
    closeBtn.addEventListener('click', () => {
      panel.style.display = 'none';
    });
  }

  // Load existing analysis context from session storage if available
  try {
    const savedResult = sessionStorage.getItem("analysisResult");
    if (savedResult) {
      window.TruthCheck.activeContext = JSON.parse(savedResult);
    }
  } catch (_) {}

  const sendMessage = async () => {
    const text = input.value.trim();
    if (!text) return;

    // Append user message to UI
    const uMsg = document.createElement('div');
    uMsg.className = 'msg user-msg';
    uMsg.textContent = text;
    messagesBox.appendChild(uMsg);
    input.value = '';
    messagesBox.scrollTop = messagesBox.scrollHeight;

    // Create temporary typing placeholder for assistant
    const aiMsg = document.createElement('div');
    aiMsg.className = 'msg ai-msg';
    aiMsg.textContent = "Thinking & analyzing context...";
    messagesBox.appendChild(aiMsg);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    if (sendBtn) sendBtn.disabled = true;
    if (input) input.disabled = true;

    const lang = document.getElementById('lang-select')?.value || localStorage.getItem('truthcheck_lang') || 'en';
    const activeContext = window.TruthCheck.activeContext;

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          context: activeContext,
          chat_history: window.TruthCheck.chatHistory,
          response_lang: lang
        })
      });

      let data;
      try {
        data = await resp.json();
      } catch (jsonErr) {
        console.error("Failed to parse chat response JSON:", jsonErr);
        data = { error: `Server returned status ${resp.status}` };
      }

      if (resp.ok && data.response) {
        aiMsg.textContent = data.response;
        // Save to chat history
        window.TruthCheck.chatHistory.push({ role: "user", content: text });
        window.TruthCheck.chatHistory.push({ role: "assistant", content: data.response });
      } else {
        console.error("Chat API error:", resp.status, data);
        aiMsg.textContent = data.error || "Sorry, I was unable to generate a response for this query. Please try again.";
        aiMsg.style.borderColor = "rgba(239, 68, 68, 0.4)";
      }
    } catch (e) {
      console.error("Network error while connecting to chat service:", e);
      aiMsg.textContent = "Network error: Unable to reach TruthCheck AI service. Please check your connection.";
      aiMsg.style.borderColor = "rgba(239, 68, 68, 0.4)";
    } finally {
      if (sendBtn) sendBtn.disabled = false;
      if (input) {
        input.disabled = false;
        input.focus();
      }
      messagesBox.scrollTop = messagesBox.scrollHeight;
    }
  };

  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (input) {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
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
    if (val.length > 60) {
      badge.textContent = 'Detected: Full News Article';
    } else if (val.length > 0) {
      badge.textContent = 'Detected: News Headline / Claim';
    } else {
      badge.textContent = 'Ready to Verify';
    }
  });

  btn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return alert("Please enter a news article, headline, or claim to verify.");

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
      let data;
      try {
        data = await resp.json();
      } catch (jsonErr) {
        console.error("Failed to parse JSON response:", jsonErr);
        data = { error: `Server returned status ${resp.status}: ${resp.statusText}` };
      }

      if (!resp.ok || data.error) {
        console.error(`Verification API error (HTTP ${resp.status}):`, data);
        alert("Verification error: " + (data.error || "Unable to complete analysis."));
      } else {
        renderAnalysisResult(data);
      }
    } catch (err) {
      console.error("Network or execution error during verification:", err);
      alert("Network error occurred during verification. Please check your connection or server status.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Verify Authenticity →";
    }
  });
}

function renderAnalysisResult(data) {
  // Update runtime state and session storage for chatbot context
  window.TruthCheck.activeContext = data;
  try {
    sessionStorage.setItem("analysisResult", JSON.stringify(data));
  } catch (_) {}

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

/* ─── 5. Toast Notification Utility ─── */
function showToast(message, type = 'success') {
  let toast = document.getElementById('truthcheck-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'truthcheck-toast';
    toast.className = 'truthcheck-toast';
    document.body.appendChild(toast);
  }
  toast.className = `truthcheck-toast toast-${type} toast-show`;
  const icon = type === 'success' ? '✅' : '⚠️';
  toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;

  if (toast._timer) clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.classList.remove('toast-show');
  }, 3500);
}

/* ─── 6. Clear History Button Handler ─── */
function initClearHistoryButtons() {
  const clearBtns = document.querySelectorAll('.btn-clear-history, #clear-history-btn, #clear-dashboard-history-btn');
  if (!clearBtns.length) return;

  clearBtns.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      const confirmed = window.confirm("Are you sure you want to clear your verification history? This action cannot be undone.");
      if (!confirmed) return;

      const originalHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '⏳ Clearing...';

      try {
        const resp = await fetch('/api/clear-history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        const data = await resp.json().catch(() => ({}));

        if (resp.ok && data.success) {
          showToast(data.message || "Verification history cleared successfully.", "success");

          // 1. Clear history page container if present
          const historyList = document.getElementById('history-list');
          if (historyList) {
            historyList.innerHTML = '<p class="empty-state">No verification reports found. All history has been cleared.</p>';
          }

          // 2. Clear Dashboard recent feed if present
          const dashboardFeed = document.querySelector('.recent-feed');
          if (dashboardFeed && !historyList) {
            dashboardFeed.innerHTML = '<p class="empty-state">No analyses performed yet. Try scanning an article or URL above!</p>';
          }

          // Hide clear buttons
          document.querySelectorAll('.btn-clear-history, #clear-history-btn, #clear-dashboard-history-btn').forEach(b => {
            b.style.display = 'none';
          });

          // 3. Clear local storage / session storage contexts
          sessionStorage.removeItem("analysisResult");
          localStorage.removeItem("iv_history");
          if (window.TruthCheck) {
            window.TruthCheck.activeContext = null;
          }
        } else {
          console.error("Clear history error:", data);
          showToast(data.error || "Failed to clear verification history.", "error");
          btn.disabled = false;
          btn.innerHTML = originalHtml;
        }
      } catch (err) {
        console.error("Network error clearing history:", err);
        showToast("Network error while clearing history.", "error");
        btn.disabled = false;
        btn.innerHTML = originalHtml;
      }
    });
  });
}

/* ─── 7. History Page Dynamic Manager ─── */
function initHistoryManager() {
  const historyList = document.getElementById('history-list');
  if (!historyList) return;

  const searchInput = document.getElementById('history-search');
  const typeFilter = document.getElementById('history-filter-type');

  let debounceTimer = null;

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, m => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[m]);
  }

  async function loadHistory() {
    const search = searchInput ? searchInput.value.trim() : '';
    const type = typeFilter ? typeFilter.value : '';

    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (type) params.append('type', type);

    try {
      const resp = await fetch(`/api/history?${params.toString()}`);
      const data = await resp.json();

      if (!resp.ok || !data.success) {
        historyList.innerHTML = `<p class="empty-state">Error loading history: ${escapeHtml(data.error || 'Server error')}</p>`;
        return;
      }

      if (!data.reports || data.reports.length === 0) {
        historyList.innerHTML = '<p class="empty-state">No verification reports found.</p>';
        const clearBtn = document.getElementById('clear-history-btn');
        if (clearBtn) clearBtn.style.display = 'none';
        return;
      }

      const clearBtn = document.getElementById('clear-history-btn');
      if (clearBtn) clearBtn.style.display = 'inline-flex';

      let html = '<table class="history-table"><thead><tr><th>Type</th><th>Input Summary</th><th>Verdict</th><th>Confidence</th><th>Date</th><th>Actions</th></tr></thead><tbody>';
      data.reports.forEach(r => {
        const isFake = (r.prediction || '').toUpperCase().includes('FAKE');
        const badgeClass = isFake ? 'badge-fake' : 'badge-real';
        const typeBadge = (r.input_type || 'text').toUpperCase();
        const title = r.input_title || (r.input_text ? r.input_text.substring(0, 70) + '...' : 'Untitled Analysis');
        const dateStr = r.created_at ? new Date(r.created_at).toLocaleString() : '--';
        const conf = r.confidence ? `${r.confidence}%` : '--';

        html += `
          <tr id="report-row-${r.id}">
            <td><span class="badge glass">${escapeHtml(typeBadge)}</span></td>
            <td>${escapeHtml(title)}</td>
            <td><span class="badge ${badgeClass}">${escapeHtml(r.prediction || 'CHECKED')}</span></td>
            <td>${conf}</td>
            <td>${dateStr}</td>
            <td>
              <button class="btn-delete-report glass-btn-danger" data-id="${r.id}" title="Delete report" style="padding: 0.35rem 0.65rem; font-size: 0.8rem;">
                🗑️
              </button>
            </td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      historyList.innerHTML = html;

      // Bind single delete buttons
      historyList.querySelectorAll('.btn-delete-report').forEach(btn => {
        btn.addEventListener('click', async () => {
          const id = btn.getAttribute('data-id');
          if (!id) return;
          if (!window.confirm("Delete this report from verification history?")) return;

          btn.disabled = true;
          try {
            const delResp = await fetch(`/api/history/${id}`, { method: 'DELETE' });
            const delData = await delResp.json();
            if (delResp.ok && delData.success) {
              const row = document.getElementById(`report-row-${id}`);
              if (row) row.remove();
              showToast("Report deleted successfully", "success");
              if (historyList.querySelectorAll('tbody tr').length === 0) {
                historyList.innerHTML = '<p class="empty-state">No verification reports found.</p>';
                if (clearBtn) clearBtn.style.display = 'none';
              }
            } else {
              showToast(delData.error || "Failed to delete report", "error");
              btn.disabled = false;
            }
          } catch (e) {
            showToast("Network error deleting report", "error");
            btn.disabled = false;
          }
        });
      });

    } catch (err) {
      console.error("Failed to fetch history:", err);
      historyList.innerHTML = '<p class="empty-state">Failed to load history.</p>';
    }
  }

  if (searchInput) {
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(loadHistory, 300);
    });
  }

  if (typeFilter) {
    typeFilter.addEventListener('change', loadHistory);
  }

  loadHistory();
}

