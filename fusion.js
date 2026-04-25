// ============================================================
// FUSION 1.0 — FRONTEND JAVASCRIPT
// ============================================================
// This file talks to YOUR Python backend (main.py).
// It no longer calls Anthropic directly.
//
// Flow:
//   User types question
//       ↓
//   fetch('http://localhost:8000/ask')   ← calls YOUR backend
//       ↓
//   Python runs: classify → dispatch → fuse → log
//       ↓
//   Backend returns FusionResponse JSON
//       ↓
//   We display results in the UI
//
// To run:
//   1. Start backend: python main.py
//   2. Open this HTML file in browser
//   That's it. No other setup needed.
// ============================================================

// ── RENDER ANSWER (markdown + LaTeX) ────────────────────────
// Converts raw model output (with markdown and LaTeX) into
// properly rendered HTML. Called for the final fused answer.
function renderAnswer(el, text) {
  // 1. Protect LaTeX delimiters from Marked.js escaping
  // We double the backslashes so Marked leaves single backslashes for KaTeX
  const safeText = text
    .replace(/\\\[/g, '\\\\[')
    .replace(/\\\]/g, '\\\\]')
    .replace(/\\\(/g, '\\\\(')
    .replace(/\\\)/g, '\\\\)');

  // 2. Render markdown → HTML
  el.innerHTML = marked.parse(safeText);

  // 3. Render LaTeX inside the markdown HTML
  //    KaTeX auto-render is loaded deferred — wait for it
  const doRender = () => {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(el, {
        delimiters: [
          { left: '$$',  right: '$$',  display: true  },
          { left: '\\[', right: '\\]', display: true  },
          { left: '$',   right: '$',   display: false },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false   // Don't crash on malformed LaTeX
      });
    }
  };

  if (typeof renderMathInElement === 'function' || window.katexReady) {
    doRender();
  } else {
    // KaTeX not ready yet (deferred) — retry after scripts load
    document.addEventListener('DOMContentLoaded', doRender);
    setTimeout(doRender, 800);  // fallback safety net
  }
}

// ── BACKEND URL ─────────────────────────────────────────────
// This points to your running Python FastAPI server.
// Change this if you deploy the backend to a server later.
const BACKEND_URL = "https://daffodil-smite-frenzy.ngrok-free.dev";

// ── STATE ───────────────────────────────────────────────────
let queryHistory = [];   // Stores recent queries for history panel
let startTime   = null;  // Used to calculate elapsed time in UI

// ── CHAR COUNT ──────────────────────────────────────────────
document.getElementById('question-input').addEventListener('input', function() {
  document.getElementById('char-count').textContent = this.value.length + ' characters';
});

// ── UI HELPERS ──────────────────────────────────────────────

function setStep(stepId, state) {
  // state = '' (idle) | 'active' (running) | 'done' (complete)
  document.getElementById(stepId).className = 'step-box ' + state;
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = '⚠ ' + msg;
  el.className = 'error-msg visible';
}

function hideError() {
  document.getElementById('error-msg').className = 'error-msg';
}

function resetUI() {
  // Reset all panels to hidden before a new query
  document.getElementById('pipeline').className       = 'pipeline visible fade-in';
  document.getElementById('classify-result').className = 'classify-result';
  document.getElementById('models-panel').className   = 'models-panel';
  document.getElementById('fusion-output').className  = 'fusion-output';
  document.getElementById('stats-row').className      = 'stats-row';
  ['step-classify','step-route','step-dispatch','step-fuse','step-polish']
    .forEach(s => setStep(s, ''));
}

function showModelCards(modelsUsed) {
  // Show the 4 model cards in loading state while backend works
  const keys = ['a', 'b', 'c', 'd'];
  document.getElementById('models-panel').className = 'models-panel visible fade-in';

  keys.forEach((k, i) => {
    const modelName = modelsUsed[i] || `Model ${i + 1}`;
    document.getElementById('name-' + k).textContent    = modelName.toUpperCase();
    document.getElementById('card-' + k).className      = 'model-card loading';
    document.getElementById('answer-' + k).innerHTML    = '<span class="model-loading-text">thinking...</span>';
    document.getElementById('conf-' + k).textContent    = '—';
    document.getElementById('conf-bar-' + k).style.width = '0%';
  });
}

function fillModelCards(individualAnswers, confidenceScores, modelsUsed) {
  // Fill model cards with actual answers once backend responds
  const keys = ['a', 'b', 'c', 'd'];

  keys.forEach((k, i) => {
    if (i >= individualAnswers.length) return; // fewer than 4 models used

    const answer = individualAnswers[i] || '—';
    const conf   = confidenceScores[i]  || 0;
    const confPct = Math.round(conf * 100);

    const preview = answer
      .replace(/\*\*(.+?)\*\*/g, '$1')   // strip **bold**
      .replace(/\*(.+?)\*/g, '$1')        // strip *italic*
      .replace(/\\\[[\s\S]*?\\\]/g, '[math]')  // replace LaTeX blocks
      .replace(/\$\$[\s\S]*?\$\$/g, '[math]')
      .substring(0, 300) + (answer.length > 300 ? '...' : '');

    document.getElementById('card-' + k).className   = 'model-card done';
    document.getElementById('answer-' + k).textContent = preview;
    document.getElementById('conf-' + k).textContent       = confPct + '%';
    document.getElementById('conf-bar-' + k).style.width   = confPct + '%';
  });
}

// ── HEALTH CHECK ────────────────────────────────────────────
// Checks if backend is running when page loads.
// Updates the status pills in the header.
async function checkBackendHealth() {
  try {
    const res  = await fetch(`${BACKEND_URL}/health`);
    const data = await res.json();

    if (data.status === 'healthy') {
      // All status pills go green
      document.querySelectorAll('.status-pill').forEach(p => p.classList.add('active'));
      document.getElementById('pill-models').querySelector('.status-dot').style.background = 'var(--accent3)';
    }
  } catch (e) {
    // Backend not running — show all pills as inactive/red
    document.querySelectorAll('.status-pill').forEach(p => p.classList.remove('active'));
    showError('Backend not running.');
  }
}

// Run health check immediately when page loads
checkBackendHealth();

// ── MAIN FUNCTION ────────────────────────────────────────────
// This is called when user clicks "Run Fusion" button.
// It calls your Python backend and displays the results.

async function runFusion() {
  const question = document.getElementById('question-input').value.trim();
  if (!question) return;

  hideError();
  startTime = Date.now();

  const btn = document.getElementById('ask-btn');
  btn.disabled    = true;
  btn.textContent = '⏳ Running...';

  resetUI();

  try {
    // ── ANIMATE PIPELINE STEPS ──────────────────────────────
    // We animate the steps while waiting for backend response.
    // The backend does all steps internally in one call —
    // we just simulate the visual progress for better UX.

    setStep('step-classify', 'active');

    // ── CALL YOUR PYTHON BACKEND ─────────────────────────────
    // This single fetch replaces ALL the old Claude API calls.
    // Your Python backend handles:
    //   classify → route → dispatch → fuse → polish → log
    // We just send the question and receive the full result.

    const response = await fetch(`${BACKEND_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        user_id: 'student'   // Optional — for tracking in logs
      })
    });

    // Handle HTTP errors (4xx, 5xx)
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    // Parse the JSON response from your FusionResponse model
    // Shape: { final_answer, category, complexity, strategy,
    //          models_used, individual_answers, confidence_scores,
    //          fusion_weights, latency_seconds, cost_estimate }
    const data = await response.json();

    // ── UPDATE PIPELINE STEPS ───────────────────────────────
    // Animate through steps to show the pipeline visually
    setStep('step-classify', 'done');
    setStep('step-route', 'active');
    await new Promise(r => setTimeout(r, 200));
    setStep('step-route', 'done');
    setStep('step-dispatch', 'active');

    // ── SHOW CLASSIFICATION RESULT ───────────────────────────
    const cat        = data.category   || 'general';
    const complexity = data.complexity || 'medium';

    document.getElementById('res-category').textContent  = cat.toUpperCase();
    document.getElementById('res-strategy').textContent  = data.strategy || '—';
    document.getElementById('res-models').textContent    = (data.models_used || []).length + ' MODELS';

    const compEl      = document.getElementById('res-complexity');
    compEl.textContent = complexity.toUpperCase();
    compEl.className   = 'classify-value complexity-' + complexity;

    document.getElementById('classify-result').className = 'classify-result visible fade-in';

    // ── SHOW MODEL CARDS ─────────────────────────────────────
    // Show loading cards first, then fill with real data
    showModelCards(data.models_used || []);
    await new Promise(r => setTimeout(r, 300)); // brief pause for UX
    fillModelCards(
      data.individual_answers  || [],
      data.confidence_scores   || [],
      data.models_used         || []
    );

    setStep('step-dispatch', 'done');
    setStep('step-fuse', 'active');
    await new Promise(r => setTimeout(r, 200));
    setStep('step-fuse', 'done');
    setStep('step-polish', 'active');
    await new Promise(r => setTimeout(r, 200));
    setStep('step-polish', 'done');

    // ── SHOW FINAL FUSED ANSWER ──────────────────────────────
    const elapsed   = data.latency_seconds || ((Date.now() - startTime) / 1000).toFixed(1);
    const wordCount = (data.final_answer || '').split(' ').length;

    renderAnswer(document.getElementById('fusion-body'), data.final_answer || '—');
    document.getElementById('meta-time').textContent       = elapsed;
    document.getElementById('meta-tokens').textContent     = '~' + Math.round(wordCount * 1.3);
    document.getElementById('fusion-output').className     = 'fusion-output visible fade-in';

    // ── SHOW STATS ───────────────────────────────────────────
    const avgConf = data.confidence_scores && data.confidence_scores.length
      ? Math.round(data.confidence_scores.reduce((a, b) => a + b, 0) / data.confidence_scores.length * 100)
      : 0;

    document.getElementById('stat-quality').textContent    = avgConf + '%';
    document.getElementById('stat-models-used').textContent = (data.models_used || []).length;
    document.getElementById('stat-category').textContent   = cat.toUpperCase();
    document.getElementById('stat-cost').textContent       = data.cost_estimate || '~₹0';
    document.getElementById('stats-row').className         = 'stats-row visible fade-in';

    // ── ADD TO HISTORY ───────────────────────────────────────
    queryHistory.unshift({ question, category: cat, answer: data.final_answer });
    if (queryHistory.length > 5) queryHistory.pop();
    renderHistory();

  } catch (err) {
    // Show user-friendly error
    // Common errors:
    //   "Failed to fetch" → backend not running → python main.py
    //   "Server error: 500" → check terminal for Python error
    //   "Server error: 422" → question validation failed
    const msg = err.message.includes('fetch')
      ? 'Cannot connect to backend. Is python main.py running?'
      : err.message;

    showError(msg);
    ['step-classify','step-route','step-dispatch','step-fuse','step-polish']
      .forEach(s => setStep(s, ''));
  }

  btn.disabled   = false;
  btn.innerHTML  = '⚡ Run Fusion';
}

// ── HISTORY PANEL ────────────────────────────────────────────

function renderHistory() {
  const section = document.getElementById('history-section');
  const list    = document.getElementById('history-list');

  if (queryHistory.length === 0) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';
  list.innerHTML = queryHistory.map((h, i) => `
    <div class="history-item" onclick="loadHistory(${i})">
      <div class="history-q">${h.question}</div>
      <div class="history-cat">${h.category}</div>
    </div>
  `).join('');
}

function loadHistory(i) {
  // Clicking a history item loads that question back into the input
  document.getElementById('question-input').value = queryHistory[i].question;
  document.getElementById('char-count').textContent = queryHistory[i].question.length + ' characters';
}

// ── KEYBOARD SHORTCUT ────────────────────────────────────────
// Ctrl + Enter submits the question (no need to click button)
document.getElementById('question-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && e.ctrlKey) runFusion();
});
