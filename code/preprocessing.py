import pandas as pd
import numpy as np

# =========================
# 1. Load files
# =========================
meta_path = "mini-atlas/data/m1_patchseq_meta_data.csv"
morph_path = "mini-atlas/data/m1_patchseq_morph_features.csv"
expr_path = "mini-atlas/data/m1_patchseq_exon_counts.csv"

meta = pd.read_csv(meta_path, sep="\t")
morph = pd.read_csv(morph_path)
expr = pd.read_csv(expr_path, index_col=0)

# store raw shapes for summary table
raw_meta_shape = meta.shape
raw_morph_shape = morph.shape
raw_expr_shape = expr.shape

print("Raw shapes:")
print("metadata:", meta.shape)
print("morphology:", morph.shape)
print("expression:", expr.shape)

# =========================
# 2. Prepare expression matrix
# =========================
expr = expr.T

# =========================
# 3. Align cell IDs
# =========================
meta = meta.rename(columns={"Cell": "cell_id"})
meta = meta.set_index("cell_id")

morph = morph.rename(columns={"cell id": "cell_id"})
morph = morph.set_index("cell_id")

expr.index.name = "cell_id"

# keep only cells present in all datasets
common_cells = meta.index.intersection(morph.index).intersection(expr.index)

meta = meta.loc[common_cells]
morph = morph.loc[common_cells]
expr = expr.loc[common_cells]

aligned_meta_shape = meta.shape
aligned_morph_shape = morph.shape
aligned_expr_shape = expr.shape

print("\nAfter alignment:")
print("metadata:", meta.shape)
print("morphology:", morph.shape)
print("expression:", expr.shape)

# =========================
# 4. Clean morphology features
# =========================
# keep only numeric morphology columns
morph_numeric = morph.select_dtypes(include=[np.number])

print("\nMorphology numeric matrix:", morph_numeric.shape)

# drop columns with all missing values
morph_numeric = morph_numeric.dropna(axis=1, how="all")

# drop columns with too many missing values
missing_frac = morph_numeric.isna().mean()
morph_numeric = morph_numeric.loc[:, missing_frac < 0.2]

# drop cells with remaining missing values
valid_cells = morph_numeric.dropna().index

meta = meta.loc[valid_cells]
expr = expr.loc[valid_cells]
morph_numeric = morph_numeric.loc[valid_cells]

# remove near-constant morphology features
morph_std = morph_numeric.std(axis=0)
morph_numeric = morph_numeric.loc[:, morph_std > 1e-8]

clean_meta_shape = meta.shape
clean_morph_shape = morph_numeric.shape
clean_expr_shape = expr.shape

print("After morphology cleaning:", morph_numeric.shape)

# =========================
# 5. Standardize morphology
# =========================
morph_scaled = (morph_numeric - morph_numeric.mean()) / morph_numeric.std(ddof=0)

# remove low-variance morphology features AFTER scaling
# since standardized features have variance near 1, use original variance for selection
orig_var = morph_numeric.var(axis=0)
var_threshold = 0.01
selected_morph_features = orig_var[orig_var > var_threshold].index

morph_selected = morph_scaled[selected_morph_features]

print("Selected morphology features:", morph_selected.shape)

# =========================
# 6. Filtering genes
# =========================
print("\nFiltering genes...")

# remove genes with zero counts
expr = expr.loc[:, expr.sum(axis=0) > 0]

# keep genes expressed in at least 10 cells
min_cells = 10
gene_mask = (expr > 0).sum(axis=0) >= min_cells
expr = expr.loc[:, gene_mask]

gene_filtered_meta_shape = meta.shape
gene_filtered_morph_shape = morph_selected.shape
gene_filtered_expr_shape = expr.shape

print("Expression after gene filtering:", expr.shape)

# =========================
# 7. Normalize expression
# =========================
library_size = expr.sum(axis=1)

# remove zero-library cells just in case
valid_cells = library_size[library_size > 0].index
meta = meta.loc[valid_cells]
morph_selected = morph_selected.loc[valid_cells]
expr = expr.loc[valid_cells]

library_size = expr.sum(axis=1)

# normalize counts
expr_norm = expr.div(library_size, axis=0) * 1e4

# log transform
expr_log = np.log1p(expr_norm)

normalized_meta_shape = meta.shape
normalized_morph_shape = morph_selected.shape
normalized_expr_shape = expr_log.shape

print("Expression normalized:", expr_log.shape)

# =========================
# 8. Select highly variable genes
# =========================
gene_var = expr_log.var(axis=0)
top_genes = gene_var.sort_values(ascending=False).head(3000).index
expr_hvg = expr_log[top_genes]

hvg_meta_shape = meta.shape
hvg_morph_shape = morph_selected.shape
hvg_expr_shape = expr_hvg.shape

print("Highly variable genes:", expr_hvg.shape)

# =========================
# 9. Generate preprocessing summary table
# =========================
summary_table = pd.DataFrame({
    "Stage": [
        "Raw dataset",
        "After alignment",
        "After morphology cleaning",
        "After gene filtering",
        "After normalization",
        "After HVG selection"
    ],
    "Cells": [
        raw_meta_shape[0],
        aligned_meta_shape[0],
        clean_meta_shape[0],
        gene_filtered_meta_shape[0],
        normalized_meta_shape[0],
        hvg_meta_shape[0]
    ],
    "Morphology Features": [
        raw_morph_shape[1],
        aligned_morph_shape[1],
        clean_morph_shape[1],
        gene_filtered_morph_shape[1],
        normalized_morph_shape[1],
        hvg_morph_shape[1]
    ],
    "Genes": [
        raw_expr_shape[0],          # raw expression is genes x cells
        aligned_expr_shape[1],      # after transpose/alignment it is cells x genes
        clean_expr_shape[1],
        gene_filtered_expr_shape[1],
        normalized_expr_shape[1],
        hvg_expr_shape[1]
    ]
})

print("\nPreprocessing summary table:")
print(summary_table)

# =========================
# 10. Generate morphology feature summary table
# =========================
morph_feature_table = pd.DataFrame({
    "feature": morph_selected.columns,
    "mean_scaled": morph_selected.mean(axis=0).values,
    "std_scaled": morph_selected.std(axis=0, ddof=0).values,
    "original_variance": orig_var.loc[morph_selected.columns].values
}).sort_values(by="original_variance", ascending=False)

print("\nMorphology feature summary:")
print(morph_feature_table.head(10))

# =========================
# 11. Generate top HVG table
# =========================
top_hvg_table = pd.DataFrame({
    "gene": gene_var.sort_values(ascending=False).head(20).index,
    "variance": gene_var.sort_values(ascending=False).head(20).values
})

print("\nTop highly variable genes:")
print(top_hvg_table)

# =========================
# 12. Save outputs
# =========================

# metadata_clean.csv
# ------------------------------------------------
# Contains metadata/annotations for each neuron.
# Rows = cells (cell_id)
# Columns = metadata fields such as cell type, layer, specimen info, etc.
#
# Only cells that survived preprocessing are included:
#   - present in all datasets
#   - have valid morphology features
#   - have valid gene expression counts
#
# This file is mainly used for:
#   - labeling cells in plots
#   - grouping cells by type or layer
#   - biological interpretation of results
meta.to_csv("metadata_clean.csv")


# morphology_selected_scaled.csv
# ------------------------------------------------
# Cleaned and standardized morphology feature matrix.
#
# Rows = cells
# Columns = selected morphology features describing neuron shape
#
# Preprocessing steps applied:
#   1. removed non-numeric columns
#   2. removed features with many missing values
#   3. removed cells with missing morphology
#   4. removed near-constant features
#   5. standardized features (z-score scaling)
#
# This matrix represents the morphology vector "m" in the model.
# It will be used for:
#   - PCA / clustering
#   - predicting morphology from gene expression
#   - studying shape variability
morph_selected.to_csv("morphology_selected_scaled.csv")


# expression_hvg_log_normalized.csv
# ------------------------------------------------
# Final gene expression matrix used for modeling.
#
# Rows = cells
# Columns = genes (top 3000 highly variable genes)
#
# Preprocessing steps applied:
#   1. removed genes with zero counts
#   2. kept genes expressed in at least 10 cells
#   3. library-size normalization (counts per 10k)
#   4. log transformation (log1p)
#   5. selected highly variable genes
#
# This matrix represents the gene expression vector "g".
# It will be used for:
#   - gene → morphology modeling
#   - dimensionality reduction
#   - identifying genes related to neuron shape
expr_hvg.to_csv("expression_hvg_log_normalized.csv")


# preprocessing_summary_table.csv
# ------------------------------------------------
# Summary table describing how dataset dimensions change
# during preprocessing.
#
# Each row corresponds to a preprocessing stage.
# Columns report the number of cells, morphology features,
# and genes remaining at that stage.
#
# This table is useful for:
#   - documenting preprocessing steps
#   - reporting dataset statistics in the project report
summary_table.to_csv("preprocessing_summary_table.csv", index=False)


# morphology_feature_summary.csv
# ------------------------------------------------
# Statistics for the selected morphology features.
#
# Columns include:
#   - mean_scaled: mean after standardization
#   - std_scaled: standard deviation after scaling
#   - original_variance: variance before scaling
#
# This table helps verify that selected morphology
# features contain meaningful variation across neurons.
morph_feature_table.to_csv("morphology_feature_summary.csv", index=False)


# top_hvg_table.csv
# ------------------------------------------------
# List of the most highly variable genes in the dataset.
#
# Columns:
#   - gene: gene name
#   - variance: variance across cells
#
# Highly variable genes are selected because they capture
# the most biological variability and are informative for
# downstream analysis.
top_hvg_table.to_csv("top_hvg_table.csv", index=False)

print("\nSaved files:")
print("- metadata_clean.csv")
print("- morphology_selected_scaled.csv")
print("- expression_hvg_log_normalized.csv")
print("- preprocessing_summary_table.csv")
print("- morphology_feature_summary.csv")
print("- top_hvg_table.csv")