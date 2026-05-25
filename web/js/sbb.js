// web/js/sbb.js
// Swiss public transport API (transport.opendata.ch) — free, no key, CORS-enabled

const API_BASE = 'https://transport.opendata.ch/v1';

async function searchStops(query) {
  if (query.length < 2) return [];
  const url = `${API_BASE}/locations?query=${encodeURIComponent(query)}&type=station`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Stop search failed: ${response.status}`);
  const data = await response.json();
  return (data.stations || [])
    .filter(s => s.id)  // only stops with real IDs (exclude address matches)
    .slice(0, 15)
    .map(s => ({
      name: s.name,
      label: s.name,
      id: s.id,
      icon: s.icon || 'station',
    }));
}

async function fetchDepartures(stopName, limit = 20) {
  const url = `${API_BASE}/stationboard?station=${encodeURIComponent(stopName)}&limit=${limit}&type=bus`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Departures failed: ${response.status}`);
  const data = await response.json();

  return (data.stationboard || []).map(entry => {
    const stop = entry.stop;
    const scheduled = new Date(stop.departure);
    const hasRealtime = stop.prognosis && stop.prognosis.departure;
    const estimated = hasRealtime ? new Date(stop.prognosis.departure) : null;
    const delaySeconds = estimated ? Math.round((estimated - scheduled) / 1000) : null;

    return {
      scheduled,
      estimated,
      delaySeconds: hasRealtime ? delaySeconds : null,
      hasRealtime: !!hasRealtime,
      operator: entry.operator || '',
      line: entry.number || '',
      destination: entry.to || '',
      stopName: stop.station ? stop.station.name : stopName,
      stopId: stop.station ? stop.station.id : null,
    };
  });
}

export { searchStops, fetchDepartures };
