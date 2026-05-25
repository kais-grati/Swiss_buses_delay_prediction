// web/js/preprocess.js
import { isPublicHoliday, DEFAULT_VALUES } from './defaults.js';

/**
 * Build a Float64Array of 35 model features from user inputs and context.
 *
 * Feature order (from model training):
 *   [0] time_sin, [1] time_cos, [2] dow_sin, [3] dow_cos,
 *   [4] month_sin, [5] month_cos, [6] is_weekend, [7] operator,
 *   [8] line, [9] stop_id, [10] additional_trip, [11] prev_stop_delay,
 *   [12] dist_to_prev_stop, [13] is_public_holiday,
 *   [14] temperature, [15] precipitation, [16] sunshine,
 *   [17] humidity, [18] wind_dir, [19] pressure, [20] snow_depth,
 *   [21-29] traffic_*, [30] trip_stop_index,
 *   [31] hour, [32] dow, [33] month, [34] wind
 */
function buildFeatures(params) {
  const ts = params.timestamp;
  const feat = new Float64Array(35);

  // TemporalFeatureExtractor: timestamp -> hour, dow, month
  const hour = ts.getHours();
  const dow = (ts.getDay() + 6) % 7;  // JS 0=Sun -> 0=Mon
  const month = ts.getMonth() + 1;

  feat[31] = hour;
  feat[32] = dow;
  feat[33] = month;

  // Cyclical encodings
  feat[0] = Math.sin(2 * Math.PI * hour / 24);
  feat[1] = Math.cos(2 * Math.PI * hour / 24);
  feat[2] = Math.sin(2 * Math.PI * dow / 7);
  feat[3] = Math.cos(2 * Math.PI * dow / 7);
  feat[4] = Math.sin(2 * Math.PI * (month - 1) / 12);
  feat[5] = Math.cos(2 * Math.PI * (month - 1) / 12);

  // Boolean flags
  feat[6] = (dow >= 5) ? 1 : 0;
  feat[13] = isPublicHoliday(ts.toISOString().slice(0, 10)) ? 1 : 0;

  // StringEncoder: operator, line -> int codes
  const opMap = (params.encoderMap && params.encoderMap.operator) || {};
  const lineMap = (params.encoderMap && params.encoderMap.line) || {};
  feat[7] = (params.operator in opMap) ? opMap[params.operator] : Object.keys(opMap).length;
  feat[8] = (params.line in lineMap) ? lineMap[params.line] : Object.keys(lineMap).length;

  // Stop & trip identity
  feat[9] = params.stopId || 0;
  feat[10] = params.additionalTrip ? 1 : 0;
  feat[30] = params.tripStopIndex || 1;

  // Lag features
  feat[11] = params.prevStopDelay || 0;
  feat[12] = 0;  // dist_to_prev_stop - negligible per ablation

  // Weather (from Open-Meteo or defaults)
  const w = params.weather || {};
  feat[14] = w.temperature ?? DEFAULT_VALUES.temperature;
  feat[15] = w.precipitation ?? DEFAULT_VALUES.precipitation;
  feat[16] = w.sunshine ?? DEFAULT_VALUES.sunshine;
  feat[17] = w.humidity ?? DEFAULT_VALUES.humidity;
  feat[18] = w.wind_dir ?? DEFAULT_VALUES.wind_dir;
  feat[19] = w.pressure ?? DEFAULT_VALUES.pressure;
  feat[20] = w.snow_depth ?? DEFAULT_VALUES.snow_depth;

  // WindMerger: wind = (wind_speed + wind_gust) / 2
  feat[34] = ((w.wind_speed ?? 3.0) + (w.wind_gust ?? w.wind_speed ?? 3.0)) / 2;

  // Traffic (from medians or defaults)
  const t = params.traffic || {};
  feat[21] = t.traffic_dtv ?? DEFAULT_VALUES.traffic_dtv;
  feat[22] = t.traffic_dwv ?? DEFAULT_VALUES.traffic_dwv;
  feat[23] = t.traffic_pw ?? DEFAULT_VALUES.traffic_pw;
  feat[24] = t.traffic_lw ?? DEFAULT_VALUES.traffic_lw;
  feat[25] = t.traffic_lz ?? DEFAULT_VALUES.traffic_lz;
  feat[26] = t.traffic_li ?? DEFAULT_VALUES.traffic_li;
  feat[27] = t.traffic_heavy_share ?? DEFAULT_VALUES.traffic_heavy_share;
  feat[28] = t.traffic_peak_ratio ?? DEFAULT_VALUES.traffic_peak_ratio;
  feat[29] = t.traffic_peak ?? DEFAULT_VALUES.traffic_peak;

  return feat;
}

export { buildFeatures };
