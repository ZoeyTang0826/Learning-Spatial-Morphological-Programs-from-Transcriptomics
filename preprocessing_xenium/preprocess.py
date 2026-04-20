"""
Preprocessing pipeline for Xenium human breast cancer spatial transcriptomics data.

Extracts:
  - Gene expression (raw counts + log1p-normalized)
  - Morphology features from cell/nucleus segmentation masks
  - Spatial coordinates (x, y)

Output: preprocessed.h5ad (AnnData)
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from skimage.measure import regionprops_table
from insitupy.datasets import xenium_human_breast_cancer


# ── 1. Load data ──────────────────────────────────────────────────────────────

print("Loading xenium_human_breast_cancer ...")
isd = xenium_human_breast_cancer()
adata = isd.cells.table.copy()  # AnnData: cells × genes

print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
print("obs columns:", list(adata.obs.columns))


# ── 2. Gene expression preprocessing ─────────────────────────────────────────

# Keep raw counts in a layer
adata.layers["counts"] = adata.X.copy()

# Normalize to 10k counts per cell, then log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()

print("Gene expression: normalized and log1p-transformed.")


# ── 3. Spatial coordinates ────────────────────────────────────────────────────

# obsm['spatial'] holds (x, y) in microns — already present after loading
assert "spatial" in adata.obsm, "spatial coordinates not found in obsm"
print(f"Spatial coords shape: {adata.obsm['spatial'].shape}")


# ── 4. Morphology features from segmentation masks ───────────────────────────

def compute_morphology(mask_dask, cell_names, seg_mask_values, label="cell"):
    """
    Compute shape features from a segmentation mask.

    Parameters
    ----------
    mask_dask : dask array, shape (H, W), dtype uint32
        Segmentation mask where each pixel value is an integer cell label.
    cell_names : np.ndarray of str
        String cell IDs aligned with seg_mask_values.
    seg_mask_values : np.ndarray of int
        Integer mask labels aligned with cell_names.

    Returns a DataFrame indexed by cell string ID with prefixed columns:
        {label}_eccentricity, {label}_major_axis_length,
        {label}_minor_axis_length, {label}_elongation, {label}_perimeter,
        {label}_solidity
    """
    print(f"Computing {label} morphology features (loading mask into memory)...")
    mask_np = mask_dask.compute().astype(np.uint32)

    props = regionprops_table(
        mask_np,
        properties=[
            "label",
            "eccentricity",       # shape features not provided by Xenium
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
            "solidity",
        ],
    )
    df = pd.DataFrame(props).set_index("label")
    df["elongation"] = df["major_axis_length"] / df["minor_axis_length"].replace(0, np.nan)
    # Note: area is intentionally excluded — Xenium already provides cell_area /
    # nucleus_area in µm² (more accurate than pixel counts from regionprops).

    # Map integer mask label → string cell ID
    label_to_name = dict(zip(seg_mask_values, cell_names))
    df.index = df.index.map(label_to_name)
    df.index.name = "cell_id"

    df.columns = [f"{label}_{c}" for c in df.columns]
    return df


boundaries = isd.cells.boundaries

# Build the shared label ↔ name mapping once
cell_names = np.array([cn.compute() for cn in boundaries.cell_names])
seg_mask_values = boundaries.seg_mask_value.compute()

cell_morph = compute_morphology(boundaries["cells"], cell_names, seg_mask_values, label="cell")
nucleus_morph = compute_morphology(boundaries["nuclei"], cell_names, seg_mask_values, label="nucleus")

print(f"Cell morphology features: {list(cell_morph.columns)}")
print(f"Nucleus morphology features: {list(nucleus_morph.columns)}")


# ── 5. Align morphology features to adata cells ───────────────────────────────

cell_morph_aligned = cell_morph.reindex(adata.obs.index)
nucleus_morph_aligned = nucleus_morph.reindex(adata.obs.index)

# Merge into obs
for col in cell_morph_aligned.columns:
    adata.obs[col] = cell_morph_aligned[col].values

for col in nucleus_morph_aligned.columns:
    adata.obs[col] = nucleus_morph_aligned[col].values

# nucleus-to-cell area ratio
adata.obs["nucleus_cell_area_ratio"] = (
    adata.obs["nucleus_area"] / adata.obs["cell_area"].replace(0, np.nan)
)

print("Morphology features added to obs.")
print("Final obs columns:", list(adata.obs.columns))


# ── 6. Quality filtering ──────────────────────────────────────────────────────

# Keep cells with at least 5 transcripts
sc.pp.filter_cells(adata, min_counts=5)
# Keep genes detected in at least 10 cells
sc.pp.filter_genes(adata, min_cells=10)

print(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")


# ── 6b. Filter by morphological variability ───────────────────────────────────
# Keep top MORPH_TOP_FRAC of cells ranked by variance across morphology features.
# Standardize first so all features contribute equally regardless of scale.

MORPH_TOP_FRAC = 0.20   # keep top 20% most morphologically variable cells

morph_cols_all = [c for c in adata.obs.columns if any(
    c.startswith(p) for p in ("cell_", "nucleus_")
)]
morph_mat = adata.obs[morph_cols_all].copy().astype(float)

# Standardize each feature to zero mean / unit variance
morph_std = (morph_mat - morph_mat.mean()) / (morph_mat.std() + 1e-8)

# Per-cell morphological variance score = variance across standardized features
morph_var_score = morph_std.var(axis=1)

threshold = morph_var_score.quantile(1 - MORPH_TOP_FRAC)
keep_mask = morph_var_score >= threshold
adata = adata[keep_mask].copy()

print(f"After morphology variance filter (top {int(MORPH_TOP_FRAC*100)}%): "
      f"{adata.n_obs} cells × {adata.n_vars} genes")


# ── 7. Save ───────────────────────────────────────────────────────────────────

# Gene expression: log-normalized
gene_expr = adata.to_df()
gene_expr.to_csv("gene_expression.csv")
print("Saved gene_expression.csv")

# Gene expression: raw counts (needed for scVI)
raw_counts = pd.DataFrame(
    adata.layers["counts"].toarray()
    if hasattr(adata.layers["counts"], "toarray")
    else adata.layers["counts"],
    index=adata.obs.index,
    columns=adata.var_names,
)
raw_counts.to_csv("gene_expression_raw.csv")
print("Saved gene_expression_raw.csv")

# Morphology features
morph_cols = [c for c in adata.obs.columns if any(
    c.startswith(p) for p in ("cell_", "nucleus_")
)]
adata.obs[morph_cols].to_csv("morphology.csv")
print("Saved morphology.csv")

# Spatial coordinates
spatial_df = pd.DataFrame(
    adata.obsm["spatial"],
    index=adata.obs.index,
    columns=["x", "y"],
)
spatial_df.to_csv("spatial.csv")
print("Saved spatial.csv")

# Cell type annotations — load from Janesick et al. 2023 supplementary file
ANNOT_XLSX = os.path.expanduser(
    "~/Downloads/Cell_Barcode_Type_Matrices.xlsx"
)
if os.path.exists(ANNOT_XLSX):
    annot_df = pd.read_excel(ANNOT_XLSX,
                             sheet_name="Xenium R1 Fig1-5 (supervised)")
    # Barcodes are integer cell IDs matching adata.obs.index
    annot_df["Barcode"] = annot_df["Barcode"].astype(int)
    annot_df = annot_df.set_index("Barcode").rename(
        columns={"Cluster": "cell_type"})

    # Align to surviving cells after QC + morphology filter
    cell_ids = adata.obs.index.astype(int)
    cell_type_series = annot_df["cell_type"].reindex(cell_ids)
    cell_type_series.index = adata.obs.index

    adata.obs["cell_type"] = cell_type_series.values
    adata.obs[["cell_type"]].to_csv("cell_annotations.csv")

    print("\nCell type distribution after filtering:")
    print(adata.obs["cell_type"].value_counts().to_string())
else:
    print(f"\nAnnotation file not found at {ANNOT_XLSX} — skipping.")
