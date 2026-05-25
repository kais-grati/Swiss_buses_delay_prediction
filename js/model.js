// web/js/model.js
/**
 * CatBoost symmetric oblivious tree evaluator (regression only).
 *
 * Classification results are derived by bucketing the regression output
 * into 4 delay classes: ≤60s, 60-120s, 120-300s, >300s.
 *
 * Model JSON structure (decompressed from .json.gz):
 * {
 *   float_features_index: number[],     // which features are float
 *   float_feature_borders: number[][],  // quantization borders per float feature
 *   tree_depth: number[],               // depth of each tree
 *   tree_split_border: number[],        // border value for each split
 *   tree_split_feature_index: number[], // feature index for each split
 *   tree_split_xor_mask: number[],      // XOR mask (0 for float features)
 *   leaf_values: number[],              // all leaf values concatenated
 *   tree_count: number,
 *   scale: number,
 *   biases: number[],                   // [bias] for regression
 * }
 */
class CatBoostModel {
  constructor() {
    this.loaded = false;
  }

  async loadFromUrl(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load model: ${response.status}`);
    const buffer = await response.arrayBuffer();

    // Decompress gzip
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(buffer);
    writer.close();

    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }

    const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    const text = new TextDecoder().decode(combined);
    const data = JSON.parse(text);
    this.init(data);
    this.loaded = true;
  }

  init(data) {
    this.floatFeaturesIndex = data.float_features_index;
    this.floatFeatureBorders = data.float_feature_borders;
    this.treeDepth = data.tree_depth;
    this.treeSplitBorder = data.tree_split_border;
    this.treeSplitFeatureIndex = data.tree_split_feature_index;
    this.treeSplitXorMask = data.tree_split_xor_mask;
    this.leafValues = data.leaf_values;
    this.treeCount = data.tree_count;
    this.scale = data.scale;
    this.biases = data.biases;
  }

  /** Binary search for the first border > value. Returns position index. */
  quantizeFloat(value, borders) {
    let lo = 0, hi = borders.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (value > borders[mid]) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  /**
   * Run inference. features: Float64Array of 35 values.
   * Returns predicted delay in seconds.
   */
  predict(features) {
    if (!this.loaded) throw new Error('Model not loaded');

    // Quantize float features
    const binaryFeatures = new Uint8Array(this.floatFeaturesIndex.length);
    for (let i = 0; i < this.floatFeaturesIndex.length; i++) {
      const featIdx = this.floatFeaturesIndex[i];
      binaryFeatures[i] = this.quantizeFloat(features[featIdx], this.floatFeatureBorders[i]);
    }

    // Evaluate trees
    let result = this.biases[0];
    let treeSplitsIdx = 0;
    let leafValuesIdx = 0;

    for (let treeIdx = 0; treeIdx < this.treeCount; treeIdx++) {
      const depth = this.treeDepth[treeIdx];
      let index = 0;

      for (let d = 0; d < depth; d++) {
        const borderVal = this.treeSplitBorder[treeSplitsIdx + d];
        const featureIdx = this.treeSplitFeatureIndex[treeSplitsIdx + d];
        const xorMask = this.treeSplitXorMask[treeSplitsIdx + d];
        const cond = (binaryFeatures[featureIdx] ^ xorMask) >= borderVal;
        index |= (cond ? 1 : 0) << d;
      }

      treeSplitsIdx += depth;
      result += this.scale * this.leafValues[leafValuesIdx + index];
      leafValuesIdx += (1 << depth);
    }

    return result;
  }
}

const model = new CatBoostModel();

async function loadModel() {
  await model.loadFromUrl('models/regression_model.json.gz');
}

/** Derive 4-class probabilities from regression delay by soft bucketing. */
function delayToClasses(delaySeconds) {
  // Soft assignment: distance from each bucket center
  const centers = [30, 90, 210, 420]; // approximate centers of ≤60, 60-120, 120-300, >300
  const widths = [30, 30, 90, 180];
  const raw = centers.map((c, i) => Math.exp(-((delaySeconds - c) ** 2) / (2 * widths[i] * widths[i])));
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map(v => v / sum);
}

export { CatBoostModel, model, loadModel, delayToClasses };
