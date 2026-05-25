"""Convert saved CatBoost models to compact JSON for JS evaluation."""
import json
import gzip
import sys
import importlib.util
from pathlib import Path


def export_catboost_to_json(cbm_path, output_path):
    """Export a CatBoost .cbm model to compact JSON via Python export intermediate."""
    from catboost import CatBoostRegressor, CatBoostClassifier
    import tempfile, os

    # Determine model type by loading
    # Try regression first, then classification
    model = None
    is_classification = False
    try:
        model = CatBoostRegressor()
        model.load_model(str(cbm_path))
    except Exception:
        model = CatBoostClassifier()
        model.load_model(str(cbm_path))
        is_classification = True

    # Export to Python format
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        py_path = f.name
    model.save_model(py_path, format='python')

    # Import the Python model module to access arrays
    spec = importlib.util.spec_from_file_location("_cb_model", py_path)
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    m = cb.catboost_model()

    # Serialize the model arrays to JSON.
    # All attributes from the Python export are plain Python lists.
    model_data = {
        'float_features_index': m.float_features_index,
        'float_feature_borders': m.float_feature_borders,
        'tree_depth': [int(m.tree_depth[i]) for i in range(m.tree_count)],
        'tree_split_border': m.tree_split_border,
        'tree_split_feature_index': m.tree_split_feature_index,
        'tree_split_xor_mask': m.tree_split_xor_mask,
        'leaf_values': m.leaf_values,
        'tree_count': m.tree_count,
        'scale': m.scale,
        'biases': m.biases,
        'is_classification': is_classification,
    }

    json_bytes = json.dumps(model_data, separators=(',', ':')).encode('utf-8')

    # Gzip compress for serving
    compressed = gzip.compress(json_bytes, compresslevel=9)
    output_path = Path(output_path)
    output_path.write_bytes(compressed)

    uncompressed_mb = len(json_bytes) / 1024 / 1024
    compressed_mb = len(compressed) / 1024 / 1024
    print(f"  Uncompressed JSON: {uncompressed_mb:.1f} MB")
    print(f"  Gzipped: {compressed_mb:.1f} MB")
    print(f"  Trees: {m.tree_count}, Leaves: {len(m.leaf_values)}")

    os.unlink(py_path)
    return model_data


def main():
    base = Path('saved_models')
    out_dir = Path('web/models')
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        (base / 'regression_catboost_705' / 'model', out_dir / 'regression_model.json.gz'),
        (base / 'classification_catboost_705_4cls' / 'model', out_dir / 'classification_model.json.gz'),
    ]

    for cbm_path, out_path in models:
        print(f'Exporting {cbm_path}...')
        export_catboost_to_json(cbm_path, out_path)
        print()


if __name__ == '__main__':
    main()
