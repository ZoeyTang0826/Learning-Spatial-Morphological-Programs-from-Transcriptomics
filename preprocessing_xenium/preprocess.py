"""
Preprocessing pipeline for Xenium human breast cancer spatial transcriptomics data.

Extracts:
  - Gene expression (raw counts + log1p-normalized)
  - Morphology features from cell/nucleus segmentation masks
  - Spatial coordinates (x, y)

Output: preprocessed.h5ad (AnnData)
"""

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


# ── 7. Save ───────────────────────────────────────────────────────────────────

# Gene expression (cells × genes), log-normalized
gene_expr = adata.to_df()
gene_expr.to_csv("gene_expression.csv")
print("Saved gene_expression.csv")

# Morphology features
morph_cols = [c for c in adata.obs.columns if any(
    c.startswith(p) for p in ("cell_", "nucleus_")
)]
adata.obs[morph_cols].to_csv("morphology.csv")
print("Saved morphology.csv")

# Spatial coordinates
import pandas as pd
spatial_df = pd.DataFrame(
    adata.obsm["spatial"],
    index=adata.obs.index,
    columns=["x", "y"],
)
spatial_df.to_csv("spatial.csv")
print("Saved spatial.csv")
