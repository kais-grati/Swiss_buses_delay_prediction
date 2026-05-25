// web/js/defaults.js

/** Swiss public holidays 2026 (Vaud canton + federal). */
const SWISS_HOLIDAYS_2026 = [
  '2026-01-01', '2026-01-02', '2026-04-03', '2026-04-05',
  '2026-04-06', '2026-05-14', '2026-05-24', '2026-05-25',
  '2026-08-01', '2026-09-21', '2026-12-25',
];

function isPublicHoliday(dateStr) {
  return SWISS_HOLIDAYS_2026.includes(dateStr);
}

/** Default feature values when API data is unavailable (Lausanne region averages). */
const DEFAULT_VALUES = {
  time_sin: 0.0, time_cos: 1.0,
  dow_sin: 0.0, dow_cos: 1.0,
  month_sin: 0.0, month_cos: 1.0,
  is_weekend: 0, stop_id: 0, additional_trip: 0,
  prev_stop_delay: 0, dist_to_prev_stop: 0, is_public_holiday: 0,
  temperature: 10.5, precipitation: 0.0, sunshine: 0.4,
  humidity: 72.0, wind_dir: 220.0, pressure: 965.0, snow_depth: 0.0,
  wind: 3.0,
  traffic_dtv: 5000, traffic_dwv: 5200, traffic_pw: 90.0,
  traffic_lw: 5.0, traffic_lz: 3.0, traffic_li: 2.0,
  traffic_heavy_share: 5.0, traffic_peak_ratio: 10.0, traffic_peak: 550,
  hour: 12, dow: 1, month: 1,
};

export { SWISS_HOLIDAYS_2026, isPublicHoliday, DEFAULT_VALUES };
