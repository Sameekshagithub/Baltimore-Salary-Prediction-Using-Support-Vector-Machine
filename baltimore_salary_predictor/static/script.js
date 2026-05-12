/* ═══════════════════════════════════════════════════════════
   Baltimore Intelligence Hub v2 · script.js
   SVM + Random Forest + Linear Regression
═══════════════════════════════════════════════════════════ */
'use strict';

// ── DOM ───────────────────────────────────────────────────────
const form          = document.getElementById('predict-form');
const btn           = document.getElementById('predict-btn');
const btnText       = btn.querySelector('.btn-text');
const btnSpinner    = document.getElementById('btn-spinner');
const resultIdle    = document.getElementById('result-idle');
const resultActive  = document.getElementById('result-active');
const salaryEl      = document.getElementById('salary-amount');
const confidencePct = document.getElementById('confidence-pct');
const confidenceBar = document.getElementById('confidence-bar');
const salaryLow     = document.getElementById('salary-low');
const salaryHigh    = document.getElementById('salary-high');
const modelPill     = document.getElementById('result-model-pill');
const titleTag      = document.getElementById('result-title-tag');
const expTag        = document.getElementById('result-exp-tag');
const expDisplay    = document.getElementById('exp-display');
const slider        = document.getElementById('years_experience');
const selectedModel = document.getElementById('selected_model');
const modelInfoText = document.getElementById('model-info-text');
const compareCard   = document.getElementById('compare-card');
const compareRows   = document.getElementById('compare-rows');
const toast         = document.getElementById('toast');

// ── Model Info Descriptions ───────────────────────────────────
const MODEL_INFO = {
  svr:
    'SVR (Support Vector Regression) with RBF kernel. Excellent for non-linear salary patterns. Requires feature scaling via StandardScaler. C=100, γ=0.1, ε=0.1.',
  random_forest:
    'Random Forest Regressor with 200 trees, max_depth=12. Robust ensemble model that averages many decision trees. Provides native feature importances.',
  linear_regression:
    'Linear Regression — the baseline model. Fast and interpretable. Best for linear relationships. Use to benchmark against SVM and Random Forest.',
};

const MODEL_ICONS = {
  svr:               'fa-vector-square',
  random_forest:     'fa-tree',
  linear_regression: 'fa-chart-line',
};

// ── Model Tab Switching ───────────────────────────────────────
document.querySelectorAll('.model-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.model-tab').forEach(t => {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');

    const model = tab.dataset.model;
    selectedModel.value = model;
    modelInfoText.textContent = MODEL_INFO[model] || '';
  });
});

// ── Slider Live Update ────────────────────────────────────────
slider.addEventListener('input', () => {
  const val = slider.value;
  expDisplay.textContent = `${val} yrs`;
  const pct = (val / slider.max) * 100;
  slider.style.background =
    `linear-gradient(90deg, #6366f1 ${pct}%, rgba(99,130,255,0.15) ${pct}%)`;
});
slider.dispatchEvent(new Event('input'));

// ── Utilities ─────────────────────────────────────────────────
function formatCurrency(n) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(n);
}

function animateNumber(el, target, duration = 1200) {
  const startTs = performance.now();
  el.style.color = 'var(--indigo-400)';
  function step(ts) {
    const progress = Math.min((ts - startTs) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    el.textContent = formatCurrency(Math.round(target * eased));
    if (progress < 1) requestAnimationFrame(step);
    else {
      el.textContent = formatCurrency(target);
      el.style.color = '';
    }
  }
  requestAnimationFrame(step);
}

let toastTimer;
function showToast(msg, type = 'error') {
  toast.textContent = msg;
  toast.className   = `toast toast-${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4000);
}

function setLoading(on) {
  btn.disabled           = on;
  btnText.style.display  = on ? 'none' : 'flex';
  btnSpinner.classList.toggle('active', on);
}

// ── Render Result ─────────────────────────────────────────────
function showResult(data) {
  const salary    = data.predicted_salary;
  const conf      = data.confidence;
  const variance  = salary * 0.06;

  resultIdle.hidden   = true;
  resultActive.hidden = false;

  modelPill.textContent = data.model_label || data.model_type;
  titleTag.innerHTML    = `<i class="fa-solid fa-briefcase"></i> ${data.job_title}`;
  expTag.innerHTML      = `<i class="fa-solid fa-clock"></i> ${data.years_experience} yrs exp`;

  animateNumber(salaryEl, salary);

  confidencePct.textContent = `${conf}%`;
  setTimeout(() => { confidenceBar.style.width = `${conf}%`; }, 200);

  salaryLow.textContent  = `$${formatCurrency(Math.max(0, salary - variance))}`;
  salaryHigh.textContent = `$${formatCurrency(salary + variance)}`;

  if (data.feature_importance) updateXAI(data.feature_importance);

  document.getElementById('result-card')
    .scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Update XAI bars ───────────────────────────────────────────
function updateXAI(features) {
  const rows = document.querySelectorAll('.xai-row');
  features.forEach((feat, i) => {
    if (!rows[i]) return;
    const pctVal = Math.round(feat.importance * 100);
    setTimeout(() => {
      const fill = rows[i].querySelector('.xai-bar-fill');
      const pct  = rows[i].querySelector('.xai-pct');
      if (fill) fill.style.width = `${pctVal}%`;
      if (pct)  pct.textContent  = `${pctVal}%`;
    }, 300 + i * 80);
  });
}

// ── Render Compare Rows ───────────────────────────────────────
function renderCompare(results) {
  compareCard.hidden = false;
  compareRows.innerHTML = '';

  // Find best salary (closest to median)
  const salaries = results.map(r => r.salary);
  const median   = salaries.slice().sort((a,b)=>a-b)[Math.floor(salaries.length/2)];
  const bestIdx  = results.reduce((best, r, i) =>
    Math.abs(r.salary - median) < Math.abs(results[best].salary - median) ? i : best, 0);

  results.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = `compare-row${i === bestIdx ? ' best' : ''}`;
    row.innerHTML = `
      <div class="compare-row-label">
        <i class="fa-solid ${MODEL_ICONS[r.model] || 'fa-circle'}"></i>
        <span>${r.label}</span>
      </div>
      ${i === bestIdx ? '<span class="compare-best-tag">Median</span>' : '<span></span>'}
      <span class="compare-salary">$${formatCurrency(r.salary)}</span>
    `;
    compareRows.appendChild(row);
  });
}

// ── Single Predict call ───────────────────────────────────────
async function callPredict(jobTitle, agency, yearsExp, modelType) {
  const res = await fetch('/predict', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_title:        jobTitle,
      agency:           agency,
      years_experience: yearsExp,
      model_type:       modelType,
    }),
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({}));
    throw new Error(e.error || `Server ${res.status}`);
  }
  return res.json();
}

// ── Form Submit ───────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const jobTitle  = document.getElementById('job_title').value.trim();
  const agency    = document.getElementById('agency').value.trim();
  const yearsExp  = parseInt(slider.value, 10);
  const modelType = selectedModel.value;

  if (!jobTitle) { showToast('Please select a Job Title.'); return; }
  if (!agency)   { showToast('Please select an Agency.');   return; }

  setLoading(true);

  try {
    // ── Primary prediction (selected model)
    const data = await callPredict(jobTitle, agency, yearsExp, modelType);
    showResult(data);

    // ── Compare all 3 models in parallel
    const allModels = ['svr', 'random_forest', 'linear_regression'];
    const others    = allModels.filter(m => m !== modelType);

    const [r1, r2] = await Promise.all(
      others.map(m => callPredict(jobTitle, agency, yearsExp, m))
    );

    const compareData = [
      { model: modelType,   label: data.model_label, salary: data.predicted_salary },
      { model: others[0],   label: r1.model_label,   salary: r1.predicted_salary   },
      { model: others[1],   label: r2.model_label,   salary: r2.predicted_salary   },
    ].sort((a, b) => {
      const order = ['svr', 'random_forest', 'linear_regression'];
      return order.indexOf(a.model) - order.indexOf(b.model);
    });

    renderCompare(compareData);
    showToast('Prediction generated successfully!', 'success');

  } catch (err) {
    console.error('[Baltimore Hub]', err);
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
});
