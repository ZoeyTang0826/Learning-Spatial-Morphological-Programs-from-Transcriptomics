"""
Xenium Baseline Analysis
Embeds cells using gene expression only (PCA, t-SNE, GMM, Leiden, scVI),
then evaluates how well each representation predicts morphology features.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from scipy import stats
from openTSNE import TSNE
import scvi
import json

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "preprocessing_xenium")
OUT_DIR   = os.path.dirname(__file__)
os.makedirs(OUT_DIR, exist_ok=True)
RAW_CSV   = os.path.join(DATA_DIR, "gene_expression_raw.csv")
ANNOT_CSV = os.path.join(DATA_DIR, "cell_annotations.csv")

N_PCA = 50
N_LATENT_SCVI = 20
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RANDOM_STATE = 42

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Xenium data...")
gene_expr = pd.read_csv(os.path.join(DATA_DIR, "gene_expression.csv"), index_col=0)
morphology = pd.read_csv(os.path.join(DATA_DIR, "morphology.csv"), index_col=0)
spatial    = pd.read_csv(os.path.join(DATA_DIR, "spatial.csv"), index_col=0)

common_idx = gene_expr.index.intersection(morphology.index).intersection(spatial.index)
gene_expr  = gene_expr.loc[common_idx]
morphology = morphology.loc[common_idx]
spatial    = spatial.loc[common_idx]

# Load cell type annotations
if os.path.exists(ANNOT_CSV):
    annot = pd.read_csv(ANNOT_CSV, index_col=0)
    annot = annot.reindex(common_idx)
    cell_type_labels = annot["cell_type"].fillna("Unlabeled").values
    cell_type_series = annot["cell_type"].fillna("Unlabeled")
    has_annotations  = True
    print(f"  Cell type annotations loaded: {len(np.unique(cell_type_labels))} types")
else:
    has_annotations  = False
    cell_type_labels = np.array(["Unknown"] * len(common_idx))
    cell_type_series = pd.Series(cell_type_labels, index=common_idx)
    print("  No cell type annotations found")

print(f"  {len(common_idx)} cells | {gene_expr.shape[1]} genes | {morphology.shape[1]} morphology features")

# Scale morphology once for all evaluations
morph_scaled = pd.DataFrame(
    StandardScaler().fit_transform(morphology),
    index=morphology.index,
    columns=morphology.columns,
)
X_expr = gene_expr.values.astype(np.float32)

# ── 1. PCA ────────────────────────────────────────────────────────────────────
print("\n[1/5] PCA...")
pca_model = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
X_pca = pca_model.fit_transform(X_expr)
pd.DataFrame(X_pca, index=common_idx,
             columns=[f"PC{i+1}" for i in range(N_PCA)]).to_csv(
    os.path.join(OUT_DIR, "pca_embedding.csv"))

var_exp = pca_model.explained_variance_ratio_
pd.DataFrame({"PC": range(1, N_PCA+1), "variance_explained": var_exp}).to_csv(
    os.path.join(OUT_DIR, "pca_variance_explained.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(range(1, N_PCA+1), np.cumsum(var_exp) * 100)
ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative variance (%)")
ax.set_title("PCA Scree Plot (Xenium gene expression)")
ax.axhline(80, color="red", linestyle="--", alpha=0.5, label="80%")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_scree.png"), dpi=150)
plt.close()

# ── 2. t-SNE (on top PCs for speed) ──────────────────────────────────────────
print("[2/5] t-SNE (openTSNE on top 50 PCs)...")
tsne_model = TSNE(
    n_components=2,
    perplexity=30,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    initialization="pca",
)
X_tsne = tsne_model.fit(X_pca)
pd.DataFrame(np.array(X_tsne), index=common_idx,
             columns=["tSNE1", "tSNE2"]).to_csv(
    os.path.join(OUT_DIR, "tsne_embedding.csv"))

# ── 3. GMM (BIC to choose n_components) ──────────────────────────────────────
print("[3/5] GMM (BIC selection on top 20 PCs)...")
X_pca20 = X_pca[:, :20]
bic_scores, n_range = [], range(2, 16)
for n in n_range:
    gmm = GaussianMixture(n_components=n, covariance_type="full",
                          random_state=RANDOM_STATE, n_init=3)
    gmm.fit(X_pca20)
    bic_scores.append(gmm.bic(X_pca20))

best_n = n_range[int(np.argmin(bic_scores))]
print(f"  Best n_components = {best_n}")
gmm_final = GaussianMixture(n_components=best_n, covariance_type="full",
                             random_state=RANDOM_STATE, n_init=5)
gmm_labels = gmm_final.fit_predict(X_pca20)
pd.DataFrame({"cluster": gmm_labels}, index=common_idx).to_csv(
    os.path.join(OUT_DIR, "gmm_clusters.csv"))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(n_range, bic_scores, marker="o")
ax.axvline(best_n, color="red", linestyle="--", label=f"best n={best_n}")
ax.set_xlabel("n_components"); ax.set_ylabel("BIC")
ax.set_title("GMM BIC curve"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gmm_bic.png"), dpi=150)
plt.close()

# ── 4. Leiden ─────────────────────────────────────────────────────────────────
print("[4/5] Leiden clustering...")
adata = ad.AnnData(X=X_expr)
adata.obsm["X_pca"] = X_pca
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=N_NEIGHBORS,
                random_state=RANDOM_STATE)
sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=RANDOM_STATE)
leiden_labels = adata.obs["leiden"].astype(int).values
pd.DataFrame({"cluster": leiden_labels}, index=common_idx).to_csv(
    os.path.join(OUT_DIR, "leiden_clusters.csv"))
print(f"  {len(np.unique(leiden_labels))} Leiden clusters")

# ── 5. scVI ───────────────────────────────────────────────────────────────────
print("[5/5] scVI...")
if os.path.exists(RAW_CSV):
    raw_counts = pd.read_csv(RAW_CSV, index_col=0).loc[common_idx]
    X_raw = raw_counts.values.astype(np.float32)
    print("  Using raw counts from gene_expression_raw.csv")
else:
    # Fallback: exponentiate log-normalized counts (less ideal)
    X_raw = np.expm1(X_expr).astype(np.float32)
    print("  WARNING: gene_expression_raw.csv not found, using expm1 fallback")
adata_scvi = ad.AnnData(X=X_raw)
adata_scvi.var_names = gene_expr.columns.tolist()
adata_scvi.obs_names = common_idx.astype(str).tolist()
adata_scvi.layers["counts"] = X_raw.copy()

scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts")
scvi_model = scvi.model.SCVI(adata_scvi, n_latent=N_LATENT_SCVI, n_layers=2)
scvi_model.train(max_epochs=150, early_stopping=True,
                 plan_kwargs={"lr": 1e-3},
                 check_val_every_n_epoch=5)
X_scvi = scvi_model.get_latent_representation()
pd.DataFrame(X_scvi, index=common_idx,
             columns=[f"scVI_{i+1}" for i in range(N_LATENT_SCVI)]).to_csv(
    os.path.join(OUT_DIR, "scvi_embedding.csv"))

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating representations against morphology...")

def r2_from_embedding(embedding, morph_df):
    """5-fold CV R² from Ridge regression: embedding → each morphology feature."""
    results = {}
    for col in morph_df.columns:
        scores = cross_val_score(Ridge(alpha=1.0), embedding, morph_df[col].values,
                                 cv=5, scoring="r2")
        results[col] = float(scores.mean())
    results["mean_R2"] = float(np.mean(list(results.values())))
    return results


def morphology_anova(labels, morph_df):
    """ANOVA F-statistic per morphology feature; silhouette in morphology space."""
    results = {}
    unique = np.unique(labels)
    for col in morph_df.columns:
        groups = [morph_df[col].values[labels == l] for l in unique]
        f, p = stats.f_oneway(*groups)
        # η² = SS_between / SS_total  (effect size)
        grand_mean = morph_df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total   = ((morph_df[col].values - grand_mean)**2).sum()
        results[f"{col}_eta2"] = float(ss_between / ss_total) if ss_total > 0 else 0.0
    sil_sample = min(10000, len(labels))
    idx = np.random.choice(len(labels), sil_sample, replace=False)
    results["silhouette_morphology"] = float(
        silhouette_score(morph_df.values[idx], labels[idx]))
    results["mean_eta2"] = float(np.mean([v for k, v in results.items()
                                          if k.endswith("_eta2")]))
    return results


all_metrics = {}
all_metrics["PCA_50"]  = r2_from_embedding(X_pca, morph_scaled)
all_metrics["tSNE"]    = r2_from_embedding(np.array(X_tsne), morph_scaled)
all_metrics["scVI"]    = r2_from_embedding(X_scvi, morph_scaled)
all_metrics["GMM"]     = morphology_anova(gmm_labels, morph_scaled)
all_metrics["Leiden"]  = morphology_anova(leiden_labels, morph_scaled)

# ── g + cell_type baseline ────────────────────────────────────────────────────
if has_annotations:
    from sklearn.preprocessing import LabelEncoder
    print("Evaluating g + cell_type baselines...")

    # One-hot encode cell type
    ct_dummies = pd.get_dummies(cell_type_series).values.astype(np.float32)

    # Concatenate cell type one-hot with each embedding
    all_metrics["PCA_50+CellType"] = r2_from_embedding(
        np.hstack([X_pca, ct_dummies]), morph_scaled)
    all_metrics["scVI+CellType"]   = r2_from_embedding(
        np.hstack([X_scvi, ct_dummies]), morph_scaled)

    # Cell type alone (upper bound of what annotation explains)
    all_metrics["CellType_only"]   = r2_from_embedding(ct_dummies, morph_scaled)

    # ── Within-cell-type stratified evaluation ────────────────────────────────
    print("Running within-cell-type stratified evaluation...")
    MIN_CELLS_PER_TYPE = 100  # skip very rare types
    unique_types = cell_type_series.value_counts()
    valid_types  = unique_types[unique_types >= MIN_CELLS_PER_TYPE].index.tolist()

    within_type_r2 = {}  # {cell_type: {method: mean_R2}}
    for ct in valid_types:
        mask = cell_type_series.values == ct
        if mask.sum() < MIN_CELLS_PER_TYPE:
            continue
        morph_ct = morph_scaled.iloc[mask]
        within_type_r2[ct] = {
            "PCA_50": r2_from_embedding(X_pca[mask], morph_ct)["mean_R2"],
            "scVI":   r2_from_embedding(X_scvi[mask], morph_ct)["mean_R2"],
            "n_cells": int(mask.sum()),
        }

    within_type_df = pd.DataFrame(within_type_r2).T
    within_type_df.index.name = "cell_type"
    within_type_df.to_csv(os.path.join(OUT_DIR, "within_celltype_r2.csv"))
    print(f"  Evaluated {len(valid_types)} cell types with ≥{MIN_CELLS_PER_TYPE} cells")

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for method, metrics in all_metrics.items():
    row = {"method": method}
    row.update(metrics)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "baseline_metrics_summary.csv"), index=False)

print("\n── Baseline Summary ──────────────────────────────")
embed_methods = ["PCA_50", "tSNE", "scVI"]
cluster_methods = ["GMM", "Leiden"]

print("Embedding methods (mean CV R² predicting morphology):")
for m in embed_methods:
    print(f"  {m:8s}  mean R² = {all_metrics[m]['mean_R2']:.4f}")

print("Clustering methods (mean η² morphology ANOVA, silhouette):")
for m in cluster_methods:
    print(f"  {m:8s}  mean η² = {all_metrics[m]['mean_eta2']:.4f} | "
          f"silhouette = {all_metrics[m]['silhouette_morphology']:.4f}")

# ── Figures ───────────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_STATE)
morph_cols = morphology.columns.tolist()
PLOT_MORPH = ["cell_area", "cell_elongation", "cell_eccentricity",
              "nucleus_area", "nucleus_elongation", "nucleus_cell_area_ratio"]
PLOT_MORPH = [c for c in PLOT_MORPH if c in morph_cols]

# UMAP on full scVI latent space (dim1-2 alone is not meaningful for scVI)
print("Computing UMAP on scVI latent space...")
adata_umap = ad.AnnData(X=X_scvi)
sc.pp.neighbors(adata_umap, use_rep="X", n_neighbors=15, random_state=RANDOM_STATE)
sc.tl.umap(adata_umap, random_state=RANDOM_STATE)
X_scvi_umap = adata_umap.obsm["X_umap"]

def clip_coords(coords, pct=1):
    """Clip outlier coordinates to [pct, 100-pct] percentile range."""
    coords = coords.copy()
    for dim in range(coords.shape[1]):
        lo, hi = np.percentile(coords[:, dim], [pct, 100 - pct])
        coords[:, dim] = np.clip(coords[:, dim], lo, hi)
    return coords

embeddings = {
    "PCA (PC1-2)": clip_coords(X_pca[:, :2]),
    "t-SNE":       clip_coords(np.array(X_tsne)),
    "scVI (UMAP)": clip_coords(X_scvi_umap),
}
cluster_sets = [("GMM", gmm_labels), ("Leiden", leiden_labels)]
n_embed = len(embeddings)

def scatter_morphology(ax, coords, morph_vals, cmap="viridis"):
    vmin, vmax = np.percentile(morph_vals, [2, 98])  # clip outliers in colormap
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=morph_vals, cmap=cmap,
                    s=2, alpha=0.5, linewidths=0, rasterized=True,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks([]); ax.set_yticks([])

def scatter_cluster(ax, coords, labels):
    n_cl = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", n_cl)
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap,
               s=2, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])

# ── Fig 1: Hexbin embeddings colored by mean morphology per bin ───────────────
print("Saving Fig 1: hexbin embeddings by morphology...")
n_feat = len(PLOT_MORPH)
fig, axes = plt.subplots(n_feat, n_embed, figsize=(4.5 * n_embed, 3.8 * n_feat))
for j, (emb_name, coords) in enumerate(embeddings.items()):
    for i, feat in enumerate(PLOT_MORPH):
        ax = axes[i, j]
        scatter_morphology(ax, coords, morph_scaled[feat].values)
        if i == 0:
            ax.set_title(emb_name, fontsize=10, fontweight="bold")
        if j == 0:
            ax.set_ylabel(feat, fontsize=8)

fig.suptitle("Gene Expression Embeddings Colored by Morphology Features",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_embeddings_by_morphology.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 2: Hexbin embeddings colored by dominant cluster per bin ──────────────
print("Saving Fig 2: hexbin embeddings by cluster...")
fig, axes = plt.subplots(2, n_embed, figsize=(4.5 * n_embed, 8))
for i, (cl_name, labels) in enumerate(cluster_sets):
    n_cl = len(np.unique(labels))
    for j, (emb_name, coords) in enumerate(embeddings.items()):
        ax = axes[i, j]
        scatter_cluster(ax, coords, labels)
        ax.set_title(f"{emb_name} — {cl_name} ({n_cl} clusters)", fontsize=8)
        if j == 0:
            ax.set_ylabel(cl_name, fontsize=10, fontweight="bold")

fig.suptitle("Gene Expression Embeddings Colored by Cluster Assignment",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_embeddings_by_cluster.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 3: Heatmap — shared color scale across both panels ────────────────────
print("Saving Fig 3: evaluation heatmap...")
r2_matrix = pd.DataFrame(
    {m: [all_metrics[m].get(c, np.nan) for c in morph_cols]
     for m in embed_methods},
    index=morph_cols,
)
eta2_matrix = pd.DataFrame(
    {m: [all_metrics[m].get(f"{c}_eta2", np.nan) for c in morph_cols]
     for m in cluster_methods},
    index=morph_cols,
)
# Shared vmax so R² and η² are on the same scale
shared_vmax = max(np.nanmax(r2_matrix.values), np.nanmax(eta2_matrix.values))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

def plot_heatmap(ax, matrix, title, vmax, fmt=".2f", cmap="YlOrRd"):
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, aspect="auto",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix.values[row, col]
            if not np.isnan(val):
                text_color = "white" if val > 0.6 * vmax else "black"
                ax.text(col, row, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color=text_color)
    ax.set_title(title, fontsize=11, pad=10)

plot_heatmap(axes[0], r2_matrix,   "Embedding Methods: CV R² → Morphology",  shared_vmax)
plot_heatmap(axes[1], eta2_matrix, "Clustering Methods: η² → Morphology",    shared_vmax)

fig.suptitle("Baseline Evaluation: Gene Expression → Morphology", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_evaluation_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 4: Summary — separate panels for η² and silhouette (silhouette is negative) ──
print("Saving Fig 4: summary bar chart...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel 1: Embedding mean R²
methods_r2 = embed_methods
vals_r2    = [all_metrics[m]["mean_R2"] for m in methods_r2]
colors_r2  = ["#4C72B0", "#DD8452", "#55A868"]
bars = axes[0].bar(methods_r2, vals_r2, color=colors_r2, edgecolor="k", width=0.5)
axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
axes[0].set_ylabel("Mean CV R²", fontsize=10)
axes[0].set_title("Embedding → Morphology\n(mean R² across 15 features)", fontsize=10)
axes[0].set_ylim(0, max(vals_r2) * 1.25)

# Panel 2: Clustering mean η²
vals_eta2  = [all_metrics[m]["mean_eta2"] for m in cluster_methods]
bars2 = axes[1].bar(cluster_methods, vals_eta2, color=["#4C72B0", "#DD8452"],
                    edgecolor="k", width=0.5)
axes[1].bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
axes[1].set_ylabel("Mean η²", fontsize=10)
axes[1].set_title("Clustering → Morphology\n(ANOVA η² effect size)", fontsize=10)
axes[1].set_ylim(0, max(vals_eta2) * 1.25)

# Panel 3: Silhouette in morphology space (can be negative — use symmetric axis)
vals_sil   = [all_metrics[m]["silhouette_morphology"] for m in cluster_methods]
bar_colors = ["#4C72B0" if v >= 0 else "#d62728" for v in vals_sil]
bars3 = axes[2].bar(cluster_methods, vals_sil, color=bar_colors, edgecolor="k", width=0.5)
axes[2].bar_label(bars3, fmt="%.3f", padding=3, fontsize=9)
axes[2].set_ylabel("Silhouette Score", fontsize=10)
axes[2].set_title("Clustering → Morphology\n(silhouette in morphology space)", fontsize=10)
axes[2].axhline(0, color="k", linewidth=0.8, linestyle="--")
sil_abs = max(abs(v) for v in vals_sil)
axes[2].set_ylim(-sil_abs * 1.4, sil_abs * 1.4)
axes[2].annotate("negative = clusters\nnot separated in\nmorphology space",
                 xy=(0.5, 0.05), xycoords="axes fraction",
                 ha="center", fontsize=7.5, color="gray")

fig.suptitle("Xenium Baseline Summary", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 5: Violin plots — cleaner than KDE for many clusters ──────────────────
print("Saving Fig 5: morphology violin plots by cluster...")

for cl_name, labels in cluster_sets:
    unique_labels = np.unique(labels)
    n_cl = len(unique_labels)

    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 11))
    axes = axes.flatten()

    for i, feat in enumerate(morph_cols):
        ax = axes[i]
        feat_vals = morphology[feat].values
        # Clip to 1–99th percentile to suppress outliers in violin width
        lo, hi = np.percentile(feat_vals, [1, 99])
        data_per_cluster = [
            np.clip(feat_vals[labels == l], lo, hi) for l in unique_labels
        ]
        parts = ax.violinplot(data_per_cluster, positions=range(n_cl),
                              showmedians=True, showextrema=False, widths=0.8)
        # Color each violin
        cmap_cl = plt.cm.get_cmap("tab10" if n_cl <= 10 else "tab20", n_cl)
        for li, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(cmap_cl(li))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        eta2_val = all_metrics[cl_name].get(f"{feat}_eta2", 0)
        ax.set_title(f"{feat}\n(η²={eta2_val:.3f})", fontsize=8)
        ax.set_xticks(range(n_cl))
        ax.set_xticklabels([f"C{l}" for l in unique_labels],
                           fontsize=5 if n_cl > 10 else 7, rotation=45)
        ax.set_ylabel("value" if i % n_cols == 0 else "", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(lo, hi)

    for j in range(len(morph_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{cl_name} Clustering: Morphology Distribution per Cluster "
                 f"({n_cl} clusters, {len(common_idx):,} cells)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig5_{cl_name.lower()}_morph_violin.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ── Fig 6: Embeddings colored by cell type ────────────────────────────────────
if has_annotations:
    print("Saving Fig 6: embeddings colored by cell type...")
    unique_ct   = sorted(cell_type_series.unique())
    n_ct        = len(unique_ct)
    ct_to_int   = {ct: i for i, ct in enumerate(unique_ct)}
    ct_int      = np.array([ct_to_int[ct] for ct in cell_type_series.values])
    cmap_ct     = plt.cm.get_cmap("tab20", n_ct)

    fig, axes = plt.subplots(1, n_embed, figsize=(5 * n_embed, 5))
    for j, (emb_name, coords) in enumerate(embeddings.items()):
        ax = axes[j]
        for i, ct in enumerate(unique_ct):
            mask = ct_int == i
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap_ct(i)], s=2, alpha=0.6, linewidths=0,
                       label=ct, rasterized=True)
        ax.set_title(emb_name, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    # Shared legend on last panel
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap_ct(i), markersize=7, label=ct)
               for i, ct in enumerate(unique_ct)]
    axes[-1].legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                    fontsize=7, framealpha=0.9)

    fig.suptitle("Gene Expression Embeddings Colored by Cell Type", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_embeddings_by_celltype.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ── Fig 7: Within-cell-type R² comparison ─────────────────────────────────────
if has_annotations:
    print("Saving Fig 7: within-cell-type R²...")
    wt = pd.read_csv(os.path.join(OUT_DIR, "within_celltype_r2.csv"), index_col=0)
    wt = wt.sort_values("PCA_50", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y      = np.arange(len(wt))
    height = 0.35
    bars1  = ax.barh(y - height/2, wt["PCA_50"], height,
                     label="PCA_50", color="#4C72B0", alpha=0.85)
    bars2  = ax.barh(y + height/2, wt["scVI"],   height,
                     label="scVI",   color="#55A868", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{ct} (n={int(wt.loc[ct,'n_cells'])})"
                        for ct in wt.index], fontsize=8)
    ax.set_xlabel("Mean CV R² (within cell type)", fontsize=10)
    ax.set_title("Within-Cell-Type: How Well Does Gene Expression\nPredict Morphology?",
                 fontsize=11)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.legend(fontsize=9)

    # Annotate overall mean for reference
    overall_pca  = all_metrics["PCA_50"]["mean_R2"]
    overall_scvi = all_metrics["scVI"]["mean_R2"]
    ax.axvline(overall_pca,  color="#4C72B0", linestyle="--", alpha=0.5,
               label=f"PCA overall ({overall_pca:.2f})")
    ax.axvline(overall_scvi, color="#55A868", linestyle="--", alpha=0.5,
               label=f"scVI overall ({overall_scvi:.2f})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_within_celltype_r2.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ── Fig 8: g+CellType vs g-only summary ───────────────────────────────────────
if has_annotations:
    print("Saving Fig 8: g+CellType vs g-only comparison...")
    compare_methods = ["CellType_only", "PCA_50", "PCA_50+CellType",
                       "scVI", "scVI+CellType"]
    compare_vals    = [all_metrics[m]["mean_R2"] for m in compare_methods]
    colors          = ["#8c8c8c", "#4C72B0", "#2a4d8f", "#55A868", "#2d6e42"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(compare_methods, compare_vals, color=colors, edgecolor="k", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Mean CV R² → Morphology", fontsize=10)
    ax.set_title("Effect of Adding Cell Type Annotation to Expression Baseline",
                 fontsize=11)
    ax.set_ylim(0, max(compare_vals) * 1.25)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_celltype_annotation_effect.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nAll figures and results saved to {OUT_DIR}/")
print("Output files:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith((".png", ".csv")):
        print(f"  {f}")
