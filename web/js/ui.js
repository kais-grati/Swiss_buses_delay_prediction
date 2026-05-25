// web/js/ui.js

let els = {};

function initUI() {
  els = {
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    stopSearch: document.getElementById('stop-search'),
    autocompleteList: document.getElementById('autocomplete-list'),
    departureList: document.getElementById('departure-list'),
    operatorInput: document.getElementById('operator'),
    lineInput: document.getElementById('line'),
    stopIndexInput: document.getElementById('stop-index'),
    prevDelayInput: document.getElementById('prev-delay'),
    datetimeInput: document.getElementById('datetime'),
    predictBtn: document.getElementById('predict-btn'),
    resultsSection: document.getElementById('results-section'),
    regDelay: document.getElementById('reg-delay'),
    regLabel: document.getElementById('reg-label'),
    regRange: document.getElementById('reg-range'),
    clsBucket: document.getElementById('cls-bucket'),
    clsProbBars: document.getElementById('cls-prob-bars'),
    summaryBanner: document.getElementById('summary-banner'),
    sbbCompare: document.getElementById('sbb-compare'),
    errorMsg: document.getElementById('error-msg'),
  };
}

function setLoading(msg) {
  els.loadingText.textContent = msg;
  els.loadingOverlay.style.display = 'flex';
}
function hideLoading() {
  els.loadingOverlay.style.display = 'none';
}
function showError(msg) {
  els.errorMsg.textContent = msg;
  els.errorMsg.style.display = 'block';
}
function hideError() {
  els.errorMsg.style.display = 'none';
}

function renderAutocomplete(stops, onSelect) {
  els.autocompleteList.innerHTML = '';
  if (!stops.length) { els.autocompleteList.style.display = 'none'; return; }
  stops.forEach(stop => {
    const li = document.createElement('li');
    li.textContent = stop.label || stop.name;
    li.addEventListener('click', () => {
      els.stopSearch.value = stop.label || stop.name;
      els.autocompleteList.style.display = 'none';
      onSelect(stop);
    });
    els.autocompleteList.appendChild(li);
  });
  els.autocompleteList.style.display = 'block';
}

function renderDepartures(departures, onSelect) {
  els.departureList.innerHTML = '';
  if (!departures.length) {
    els.departureList.innerHTML = '<p style="color:var(--text-dim);font-size:0.85rem;">No departures found.</p>';
    return;
  }
  departures.forEach(dep => {
    const div = document.createElement('div');
    div.style.cssText = 'padding:10px;border-bottom:1px solid var(--border);cursor:pointer;display:flex;justify-content:space-between;align-items:center;';
    div.addEventListener('click', () => onSelect(dep));
    const time = dep.scheduled.toLocaleTimeString('de-CH', { hour: '2-digit', minute: '2-digit' });
    let delayHtml = '';
    if (dep.hasRealtime) {
      const d = dep.delaySeconds;
      const color = d > 60 ? 'var(--red)' : d <= 0 ? 'var(--green)' : 'var(--yellow)';
      delayHtml = `<div style="font-size:0.8rem;color:${color}">${d >= 0 ? '+' : ''}${d}s</div>`;
    }
    div.innerHTML = `
      <div><strong>${dep.line}</strong> → ${dep.destination}
        <div style="font-size:0.75rem;color:var(--text-dim);">${dep.operator}</div></div>
      <div style="text-align:right;"><div style="font-size:1.1rem;">${time}</div>${delayHtml}</div>`;
    els.departureList.appendChild(div);
  });
}

function fillFormFromDeparture(dep) {
  els.operatorInput.value = dep.operator || '';
  els.lineInput.value = dep.line || '';
  els.datetimeInput.value = toLocalDatetimeString(dep.scheduled);
  if (dep.delaySeconds != null) {
    els.prevDelayInput.value = Math.max(0, dep.delaySeconds);
  }
}

const CLASS_NAMES = ['≤60s', '60–120s', '120–300s', '>300s'];
const BAR_COLORS = ['green', 'yellow', 'orange', 'red'];

function showRegression(delaySeconds) {
  els.regDelay.textContent = (delaySeconds >= 0 ? '+' : '') + Math.round(delaySeconds) + 's';
  if (delaySeconds <= 60) { els.regDelay.className = 'big-number green'; }
  else if (delaySeconds <= 120) { els.regDelay.className = 'big-number yellow'; }
  else { els.regDelay.className = 'big-number red'; }
  els.regLabel.textContent = delaySeconds <= 60 ? 'On time' : delaySeconds <= 120 ? 'Slight delay' : delaySeconds <= 300 ? 'Moderate delay' : 'Major delay';
  els.regRange.textContent = `≈${Math.round(Math.max(0, delaySeconds - 64))}s to ${Math.round(delaySeconds + 64)}s`;
}

function showClassification(probs) {
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) { if (probs[i] > probs[maxIdx]) maxIdx = i; }
  els.clsBucket.textContent = CLASS_NAMES[maxIdx];
  els.clsBucket.className = 'big-number ' + BAR_COLORS[maxIdx];
  els.clsProbBars.innerHTML = '';
  for (let i = 0; i < 4; i++) {
    const pct = Math.round(probs[i] * 100);
    els.clsProbBars.innerHTML +=
      `<div class="prob-row"><span class="prob-label">${CLASS_NAMES[i]}</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill ${BAR_COLORS[i]}" style="width:${pct}%"></div></div>
        <span class="prob-pct">${pct}%</span></div>`;
  }
}

function showSummary(delaySeconds) {
  let level, text;
  if (delaySeconds <= 60) { level = 'green'; text = 'Smooth trip expected'; }
  else if (delaySeconds <= 120) { level = 'yellow'; text = 'Minor delay likely'; }
  else if (delaySeconds <= 300) { level = 'red'; text = 'Significant delay expected'; }
  else { level = 'red'; text = 'Major delay — plan ahead'; }
  els.summaryBanner.textContent = text;
  els.summaryBanner.className = 'summary ' + level;
  els.sbbCompare.innerHTML = `Predicted arrival delay: <strong>${Math.round(delaySeconds) >= 0 ? '+' : ''}${Math.round(delaySeconds)}s</strong>`;
  els.resultsSection.style.display = 'block';
}

function toLocalDatetimeString(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  const h = String(date.getHours()).padStart(2, '0');
  const min = String(date.getMinutes()).padStart(2, '0');
  return `${y}-${m}-${d}T${h}:${min}`;
}

export { initUI, setLoading, hideLoading, showError, hideError, renderAutocomplete, renderDepartures, fillFormFromDeparture, showRegression, showClassification, showSummary };
