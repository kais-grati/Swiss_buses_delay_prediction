import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from ml.preprocessors.delay_binner import DelayBinner

pio.renderers.default = "browser"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Projection visualiser for bus delay data")
parser.add_argument("--bins", type=int, nargs="+", default=[0, 60, 120],
                    help="Delay class boundaries in seconds (default: 0 60 120)")
parser.add_argument("--sample", type=int, default=0,
                    help="Randomly downsample to N points (0 = no sampling)")
parser.add_argument("--method", type=str, default="pca",
                    choices=["pca", "kpca-rbf", "kpca-poly", "kpca-sigmoid", "tsne", "umap"],
                    help="Dimensionality reduction method (default: pca)")
parser.add_argument("--dims", type=int, default=3, choices=[2, 3],
                    help="Output dimensions: 2D or 3D plot (default: 3)")
parser.add_argument("--umap-n-neighbors", type=int, default=30,
                    help="UMAP n_neighbors parameter (default: 30)")
parser.add_argument("--umap-min-dist", type=float, default=0.1,
                    help="UMAP min_dist parameter (default: 0.1)")
parser.add_argument("--tsne-perplexity", type=float, default=50,
                    help="t-SNE perplexity parameter (default: 50)")
args = parser.parse_args()
bins = sorted(set(args.bins))
ndim = args.dims

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
_df = duckdb.sql("""
    SELECT * FROM 'data/dataset_705_echandens.parquet'
    WHERE arrival_delay_s IS NOT NULL
""").df()
print(f"  {len(_df):,} rows loaded")

FEATURE_COLS = [
    "time_sin", "time_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_public_holiday",
    "temperature", "precipitation", "sunshine", "humidity",
    "wind_speed", "wind_gust", "pressure", "snow_depth",
    "wind_dir_sin", "wind_dir_cos",
]

X = _df[FEATURE_COLS].astype(float).fillna(0).values
X_scaled = StandardScaler().fit_transform(X)

# ── Downsample ─────────────────────────────────────────────────────────────────
KERNEL_CAP = 5000
if args.method.startswith("kpca") and len(X_scaled) > KERNEL_CAP:
    if args.sample == 0:
        print(f"Kernel PCA on {len(X_scaled):,} rows would require a "
              f"{len(X_scaled):,}×{len(X_scaled):,} Gram matrix — likely OOM.")
        print(f"  Auto-downsampling to {KERNEL_CAP:,} points. "
              f"Use --sample to override.")
        args.sample = KERNEL_CAP
    elif args.sample > KERNEL_CAP:
        print(f"Warning: --sample {args.sample} > {KERNEL_CAP} — "
              f"kernel PCA may still OOM. Consider a lower value.")

if args.sample > 0 and args.sample < len(X_scaled):
    rng = pd.Series(range(len(X_scaled))).sample(n=args.sample, random_state=42).values
    X_scaled = X_scaled[rng]
    _df = _df.iloc[rng].reset_index(drop=True)
    print(f"  Downsampled to {args.sample:,} points before projection")

# ── Run projection ─────────────────────────────────────────────────────────────
default_axes = [f"Dim {i+1}" for i in range(ndim)]
axis_labels = list(default_axes)

if args.method == "pca":
    print(f"Running PCA (n_components={ndim})...")
    proj = PCA(n_components=ndim)
    X_proj = proj.fit_transform(X_scaled)
    ev = proj.explained_variance_ratio_
    parts = " · ".join(f"PC{i+1} {ev[i]:.1%}" for i in range(ndim))
    print(f"Explained variance — {parts}  total: {sum(ev):.1%}")
    subtitle = f"{parts} · total {sum(ev):.1%}"
    axis_labels = [f"PC{i+1} ({ev[i]:.1%})" for i in range(ndim)]

elif args.method.startswith("kpca"):
    kernel_map = {"kpca-rbf": "rbf", "kpca-poly": "poly", "kpca-sigmoid": "sigmoid"}
    kernel = kernel_map[args.method]
    title_kernel = {"rbf": "RBF", "poly": "Poly", "sigmoid": "Sigmoid"}
    print(f"Running Kernel PCA (kernel={kernel}, n_components={ndim})...")
    proj = KernelPCA(n_components=ndim, kernel=kernel, random_state=42, n_jobs=-1)
    X_proj = proj.fit_transform(X_scaled)
    ev = proj.eigenvalues_ / proj.eigenvalues_.sum()
    top_ev = ev[:ndim]
    sub_pct = top_ev / top_ev.sum()
    parts = " · ".join(f"λ{i+1} {sub_pct[i]:.1%}" for i in range(ndim))
    subtitle = f"Kernel PCA ({title_kernel[kernel]}) — {parts}"
    axis_labels = [f"KPC{i+1} ({sub_pct[i]:.1%})" for i in range(ndim)]

elif args.method == "tsne":
    print(f"Running t-SNE (n_components={ndim}, perplexity={args.tsne_perplexity})...")
    proj = TSNE(n_components=ndim, perplexity=args.tsne_perplexity,
                random_state=42, n_jobs=-1)
    X_proj = proj.fit_transform(X_scaled)
    kl = proj.kl_divergence_
    subtitle = f"t-SNE (perplexity={args.tsne_perplexity}, KL={kl:.2f})"

elif args.method == "umap":
    print(f"Running UMAP (n_components={ndim}, n_neighbors={args.umap_n_neighbors}, "
          f"min_dist={args.umap_min_dist})...")
    import umap
    proj = umap.UMAP(n_components=ndim, n_neighbors=args.umap_n_neighbors,
                     min_dist=args.umap_min_dist, random_state=42, verbose=True)
    X_proj = proj.fit_transform(X_scaled)
    subtitle = f"UMAP (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})"

else:
    raise ValueError(f"Unknown method: {args.method}")

# ── Build dataframe ───────────────────────────────────────────────────────────
col_names = [f"D{i+1}" for i in range(ndim)]
df_proj = pd.DataFrame(X_proj, columns=col_names)
df_proj["delay"] = _df["arrival_delay_s"].values

# ── Class distribution ────────────────────────────────────────────────────────
PALETTE = ["#FF1493", "#00FF00", "#FF4500", "#00BFFF", "#FFD700", "#FF00FF", "#1E90FF"]

binner = DelayBinner(bins=bins)
cls_series = binner.encode(df_proj["delay"]).values
n_classes = len(bins) + 1

print(f"\nClass distribution ({n_classes} classes, bins={bins}):")
for cls in range(n_classes):
    count = (cls_series == cls).sum()
    pct = count / len(cls_series) * 100
    print(f"  Class {cls} [{binner.class_names[cls]}]: {count:,} samples ({pct:.1f}%)")
print()

# ── Build figure (2D or 3D) ───────────────────────────────────────────────────
ScatterCls = go.Scatter3d if ndim == 3 else go.Scatter

fig = go.Figure()
for cls in range(n_classes):
    mask = cls_series == cls
    sub = df_proj[mask]
    count = mask.sum()
    label = f"Class {cls}: {binner.class_names[cls]} (n={count:,})"

    trace_kw = dict(
        mode="markers",
        name=label,
        marker=dict(size=6 if ndim == 2 else 3, color=PALETTE[cls % len(PALETTE)], opacity=0.5),
        customdata=sub["delay"],
    )

    if ndim == 3:
        trace_kw["x"] = sub["D1"]
        trace_kw["y"] = sub["D2"]
        trace_kw["z"] = sub["D3"]
        trace_kw["hovertemplate"] = (
            f"{label}<br>D1: %{{x:.2f}}<br>D2: %{{y:.2f}}<br>"
            f"D3: %{{z:.2f}}<br>Delay: %{{customdata}}s<extra></extra>"
        )
    else:
        trace_kw["x"] = sub["D1"]
        trace_kw["y"] = sub["D2"]
        trace_kw["hovertemplate"] = (
            f"{label}<br>D1: %{{x:.2f}}<br>D2: %{{y:.2f}}<br>"
            f"Delay: %{{customdata}}s<extra></extra>"
        )

    fig.add_trace(ScatterCls(**trace_kw))

title_dims = f"{ndim}D" if ndim == 2 else "3D"
fig.update_layout(
    title=(
        f"{title_dims} projection ({args.method}) — colored by delay class  [bins={bins}]<br>"
        f"<sup>{subtitle}</sup>"
    ),
    legend=dict(title="Delay class", itemsizing="constant"),
    width=1400, height=1050 if ndim == 3 else 900,
)

if ndim == 3:
    fig.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
    )
else:
    fig.update_layout(
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
    )

method_name = args.method.upper() if args.method == "pca" else args.method
print(f"Rendering {len(df_proj):,} points in {title_dims} ({method_name})...")
fig.show()
print("Done — plot opened in browser.")
