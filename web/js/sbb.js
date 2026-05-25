// web/js/sbb.js
// SBB timetable API via search.ch (public, no API key required)

const SBB_AUTOCOMPLETE_URL = 'https://timetable.search.ch/api/completion.json';
const SBB_STATIONBOARD_URL = 'https://timetable.search.ch/api/stationboard.json';

async function searchStops(query) {
  if (query.length < 2) return [];
  const url = `${SBB_AUTOCOMPLETE_URL}?query=${encodeURIComponent(query)}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`SBB autocomplete failed: ${response.status}`);
  const data = await response.json();
  return (data || []).slice(0, 10).map((item) => ({
    name: item.label,
    label: item.label,
    id: item.label,
  }));
}

async function fetchDepartures(stopName, limit = 20) {
  const url = `${SBB_STATIONBOARD_URL}?stop=${encodeURIComponent(stopName)}&limit=${limit}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`SBB stationboard failed: ${response.status}`);
  const data = await response.json();

  return (data.connections || []).map(entry => ({
    scheduled: new Date(entry.time),
    operator: entry.operator || '',
    line: entry.line || '',
    destination: entry.terminal ? entry.terminal.name : '',
    stopName: data.stop ? data.stop.name : stopName,
    stopId: data.stop ? data.stop.id : null,
  }));
}

export { searchStops, fetchDepartures };
