// web/js/app.js
import { model, loadModel, delayToClasses } from './model.js';
import { buildFeatures } from './preprocess.js';
import { searchStops, fetchDepartures } from './sbb.js';
import { fetchWeather } from './weather.js';
import {
  initUI, setLoading, hideLoading, showError, hideError,
  renderAutocomplete, renderDepartures, fillFormFromDeparture,
  showRegression, showClassification, showSummary,
} from './ui.js';

let encoderMap = {};
let trafficMedians = {};

async function init() {
  initUI();

  // Load static data
  try {
    const [encResp, trafficResp] = await Promise.all([
      fetch('data/encoder_mapping.json'),
      fetch('data/traffic_medians.json'),
    ]);
    encoderMap = await encResp.json();
    trafficMedians = await trafficResp.json();
  } catch (e) {
    console.warn('Failed to load encoder/traffic data', e);
  }

  // Load model
  setLoading('Loading prediction model...');
  try {
    await loadModel();
  } catch (e) {
    showError('Failed to load prediction model. Please refresh.');
    console.error(e);
    return;
  }
  hideLoading();

  wireEvents();
  document.getElementById('datetime').value = toLocalISO(new Date());
}

function wireEvents() {
  const stopSearch = document.getElementById('stop-search');
  const predictBtn = document.getElementById('predict-btn');

  let debounceTimer;
  stopSearch.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      const query = stopSearch.value.trim();
      if (query.length < 2) return;
      try {
        const stops = await searchStops(query);
        renderAutocomplete(stops, async (stop) => {
          try {
            const deps = await fetchDepartures(stop.name || stop.label);
            renderDepartures(deps, (dep) => {
              fillFormFromDeparture(dep);
              document.getElementById('departure-list').innerHTML = '';
            });
          } catch (e) {
            showError('Could not fetch departures. Try manual entry.');
          }
        });
      } catch (e) {
        console.warn('SBB autocomplete failed (CORS?):', e.message);
      }
    }, 300);
  });

  document.addEventListener('click', (e) => {
    if (!stopSearch.contains(e.target)) {
      document.getElementById('autocomplete-list').style.display = 'none';
    }
  });

  predictBtn.addEventListener('click', handlePredict);
}

async function handlePredict() {
  hideError();
  const operatorInput = document.getElementById('operator');
  const lineInput = document.getElementById('line');
  const stopIndexInput = document.getElementById('stop-index');
  const prevDelayInput = document.getElementById('prev-delay');
  const datetimeInput = document.getElementById('datetime');

  if (!operatorInput.value.trim() || !lineInput.value.trim()) {
    showError('Please enter operator and line (or search for a stop first).');
    return;
  }

  const predictBtn = document.getElementById('predict-btn');
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<span class="spinner"></span> Predicting...';

  try {
    const timestamp = new Date(datetimeInput.value || Date.now());
    const operator = operatorInput.value.trim();
    const line = lineInput.value.trim();
    const tripStopIndex = parseInt(stopIndexInput.value) || 1;
    const prevStopDelay = parseInt(prevDelayInput.value) || 0;

    const weather = await fetchWeather().catch(() => null);
    const trafficKey = `${operator}|${line}`;
    const traffic = trafficMedians[trafficKey] || null;

    const features = buildFeatures({
      timestamp, operator, line,
      stopId: 0, tripStopIndex, additionalTrip: false,
      prevStopDelay, weather, traffic, encoderMap,
    });

    const delay = model.predict(features);
    const probs = delayToClasses(delay);

    showRegression(delay);
    showClassification(probs);
    showSummary(delay);
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    showError('Prediction failed: ' + e.message);
    console.error(e);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict';
  }
}

function toLocalISO(date) {
  const tzOffset = date.getTimezoneOffset();
  const local = new Date(date.getTime() - tzOffset * 60000);
  return local.toISOString().slice(0, 16);
}

init();
