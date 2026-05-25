"""Extract encoder mappings and traffic medians for JS preprocessing."""
import json
import joblib
from pathlib import Path


def extract_encoder_mappings():
    """Extract operator/line → integer code mappings from saved StringEncoder."""
    enc = joblib.load('saved_models/regression_catboost_705/preprocessor_2.joblib')
    mappings = getattr(enc, '_mappings', {})
    return mappings


def extract_traffic_medians():
    """Compute median traffic values per (operator, line) from Lausanne 50k dataset."""
    import duckdb
    traffic_cols = [
        'traffic_dtv', 'traffic_dwv', 'traffic_pw', 'traffic_lw',
        'traffic_lz', 'traffic_li', 'traffic_heavy_share',
        'traffic_peak_ratio', 'traffic_peak',
    ]
    cols_str = ', '.join(
        [f'MEDIAN({c}) AS {c}' for c in traffic_cols]
    )
    result = duckdb.query(f"""
        SELECT operator, line, {cols_str}
        FROM read_parquet('data/lausanne50k_bus_2025_weather_traffic.parquet')
        GROUP BY operator, line
    """).df()
    medians = {}
    for _, row in result.iterrows():
        key = f"{row['operator']}|{row['line']}"
        medians[key] = {c: (float(row[c]) if not (row[c] is None) else 0.0) for c in traffic_cols}
    return medians


def main():
    out_dir = Path('web/data')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Extracting encoder mappings...')
    mappings = extract_encoder_mappings()
    (out_dir / 'encoder_mapping.json').write_text(json.dumps(mappings, indent=2))
    print(f'  Operators: {len(mappings.get("operator", {}))}')
    print(f'  Lines: {len(mappings.get("line", {}))}')

    print('Extracting traffic medians...')
    medians = extract_traffic_medians()
    (out_dir / 'traffic_medians.json').write_text(json.dumps(medians, indent=2))
    print(f'  Unique (operator, line) combos: {len(medians)}')


if __name__ == '__main__':
    main()
