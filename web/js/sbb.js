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

  return (data.stationboard || []).map(entry => {
    const dep = entry.stop.departure;
    const scheduled = new Date(dep.scheduled);
    const prognosis = dep.prognosis;
    const estimated = prognosis ? new Date(prognosis.departure) : null;
    const delaySeconds = estimated ? Math.round((estimated - scheduled) / 1000) : null;
    const hasRealtime = !!(prognosis && prognosis.realtime);

    return {
      scheduled,
      estimated,
      delaySeconds: hasRealtime ? delaySeconds : null,
      operator: dep.operator || '',
      line: dep.line || entry.name || '',
      destination: entry.to || '',
      stopName: dep.station ? dep.station.name : stopName,
      stopId: dep.station ? dep.station.id : null,
      hasRealtime,
    };
  });
}

export { searchStops, fetchDepartures };
