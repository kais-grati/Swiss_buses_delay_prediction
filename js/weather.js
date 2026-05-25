// web/js/weather.js
// Open-Meteo free weather API (no key required)

const OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast';
const DEFAULT_LAT = 46.52;
const DEFAULT_LON = 6.63;

async function fetchWeather(lat = DEFAULT_LAT, lon = DEFAULT_LON) {
  try {
    const params = new URLSearchParams({
      latitude: lat,
      longitude: lon,
      hourly: 'temperature_2m,precipitation,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_gusts_10m,wind_direction_10m,snow_depth',
      forecast_hours: '1',
      timezone: 'Europe/Zurich',
    });
    const response = await fetch(`${OPEN_METEO_URL}?${params}`);
    if (!response.ok) return null;
    const data = await response.json();
    if (!data.hourly) return null;

    const i = 0;
    return {
      temperature: data.hourly.temperature_2m[i],
      precipitation: data.hourly.precipitation[i],
      humidity: data.hourly.relative_humidity_2m[i],
      pressure: data.hourly.surface_pressure[i],
      wind_speed: data.hourly.wind_speed_10m[i],
      wind_gust: data.hourly.wind_gusts_10m[i],
      wind_dir: data.hourly.wind_direction_10m[i],
      snow_depth: data.hourly.snow_depth[i] || 0,
      sunshine: null,
    };
  } catch {
    return null;
  }
}

export { fetchWeather };
