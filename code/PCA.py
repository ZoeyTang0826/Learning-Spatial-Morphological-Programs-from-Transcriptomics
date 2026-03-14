import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. Load preprocessed data
# -------------------------------

# morphology features
morph = pd.read_csv("morphology_selected_scaled.csv", index_col=0)

# gene expression
gene = pd.read_csv("expression_hvg_log_normalized.csv", index_col=0)

# metadata
meta = pd.read_csv("metadata_clean.csv", index_col=0)

print("Morphology shape:", morph.shape)
print("Gene expression shape:", gene.shape)
print("Metadata shape:", meta.shape)

print("\nMetadata columns:")
print(meta.columns.tolist())

# -------------------------------
# 2. Align all datasets by cell ID
# -------------------------------

common_cells = gene.index.intersection(morph.index).intersection(meta.index)

gene = gene.loc[common_cells]
morph = morph.loc[common_cells]
meta = meta.loc[common_cells]

print("\nAfter alignment:")
print("Morphology shape:", morph.shape)
print("Gene expression shape:", gene.shape)
print("Metadata shape:", meta.shape)

# -------------------------------
# 3. PCA on Morphology
# -------------------------------

pca_m = PCA(n_components=10)
morph_pca = pca_m.fit_transform(morph)

print("\nMorphology PCA variance explained:")
print(pca_m.explained_variance_ratio_)

# -------------------------------
# 4. PCA on Gene Expression
# -------------------------------

pca_g = PCA(n_components=10)
gene_pca = pca_g.fit_transform(gene)

print("\nGene PCA variance explained:")
print(pca_g.explained_variance_ratio_)

# -------------------------------
# 5. Joint PCA (gene + morphology)
# -------------------------------

X_joint = np.concatenate([gene.values, morph.values], axis=1)

pca_joint = PCA(n_components=10)
joint_pca = pca_joint.fit_transform(X_joint)

print("\nJoint PCA variance explained:")
print(pca_joint.explained_variance_ratio_)

# -------------------------------
# 6. Save PCA coordinates
# -------------------------------

morph_pca_df = pd.DataFrame(
    morph_pca,
    index=common_cells,
    columns=[f"PC{i+1}" for i in range(morph_pca.shape[1])]
)

gene_pca_df = pd.DataFrame(
    gene_pca,
    index=common_cells,
    columns=[f"PC{i+1}" for i in range(gene_pca.shape[1])]
)

joint_pca_df = pd.DataFrame(
    joint_pca,
    index=common_cells,
    columns=[f"PC{i+1}" for i in range(joint_pca.shape[1])]
)

morph_pca_df.to_csv("morphology_pca.csv")
gene_pca_df.to_csv("gene_pca.csv")
joint_pca_df.to_csv("joint_pca.csv")

# -------------------------------
# 7. Plot PCA (Morphology)
# -------------------------------

plt.figure(figsize=(6, 5))
plt.scatter(morph_pca[:, 0], morph_pca[:, 1], s=8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Morphology Features")
plt.tight_layout()
plt.savefig("PCA_morph.png", dpi=300)
plt.show()

# -------------------------------
# 8. Plot PCA (Gene Expression)
# -------------------------------

plt.figure(figsize=(6, 5))
plt.scatter(gene_pca[:, 0], gene_pca[:, 1], s=8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Gene Expression")
plt.tight_layout()
plt.savefig("PCA_gene.png", dpi=300)
plt.show()

# -------------------------------
# 9. Clean plotting helper
# -------------------------------

def plot_pca_with_labels(pca_array, labels, title, output_png, top_n=10):
    """
    Plot PCA with discrete legend using the top_n most frequent labels.
    All other labels are grouped into 'Other'.
    """
    labels = labels.astype(str)

    # keep only top N labels for readability
    top_labels = labels.value_counts().head(top_n).index
    labels_plot = labels.where(labels.isin(top_labels), other="Other")

    unique_labels = sorted(labels_plot.unique())
    cmap = plt.get_cmap("tab20", len(unique_labels))

    plt.figure(figsize=(8, 6))

    for i, lab in enumerate(unique_labels):
        idx = labels_plot == lab
        plt.scatter(
            pca_array[idx, 0],
            pca_array[idx, 1],
            s=18,
            alpha=0.8,
            color=cmap(i),
            label=lab
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(
        title="RNA type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        markerscale=1.2
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    # save label counts actually used in the plot
    label_count_df = labels_plot.value_counts().rename_axis("label").reset_index(name="count")
    return label_count_df

# -------------------------------
# 10. Plot gene/morphology PCA colored by metadata label
# -------------------------------

label_col = "RNA type"

if label_col in meta.columns:
    labels = meta[label_col]

    gene_label_counts = plot_pca_with_labels(
        gene_pca,
        labels,
        title=f"Gene Expression PCA Colored by {label_col}",
        output_png=f"PCA_gene_colored_by_{label_col.replace(' ', '_')}_clean.png",
        top_n=10
    )

    morph_label_counts = plot_pca_with_labels(
        morph_pca,
        labels,
        title=f"Morphology PCA Colored by {label_col}",
        output_png=f"PCA_morph_colored_by_{label_col.replace(' ', '_')}_clean.png",
        top_n=10
    )

    gene_label_counts.to_csv(
        f"PCA_label_counts_{label_col.replace(' ', '_')}.csv",
        index=False
    )

    print(f"\nSaved clean label-count table for {label_col}:")
    print(gene_label_counts)

else:
    print(f"\nColumn '{label_col}' not found in metadata.")
    print("Available columns:")
    print(meta.columns.tolist())

# -------------------------------
# 11. Scree plot for gene PCA
# -------------------------------

plt.figure(figsize=(6, 4))
plt.plot(
    range(1, len(pca_g.explained_variance_ratio_) + 1),
    pca_g.explained_variance_ratio_,
    marker="o"
)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Gene Expression PCA Scree Plot")
plt.tight_layout()
plt.savefig("PCA_scree.png", dpi=300)
plt.show()

# -------------------------------
# 12. Save explained variance tables
# -------------------------------

gene_var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca_g.explained_variance_ratio_))],
    "variance_explained": pca_g.explained_variance_ratio_
})

morph_var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca_m.explained_variance_ratio_))],
    "variance_explained": pca_m.explained_variance_ratio_
})

joint_var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca_joint.explained_variance_ratio_))],
    "variance_explained": pca_joint.explained_variance_ratio_
})

gene_var_df.to_csv("gene_pca_variance_explained.csv", index=False)
morph_var_df.to_csv("morphology_pca_variance_explained.csv", index=False)
joint_var_df.to_csv("joint_pca_variance_explained.csv", index=False)

print("\nSaved files:")
print("- morphology_pca.csv")
print("- gene_pca.csv")
print("- joint_pca.csv")
print("- PCA_morph.png")
print("- PCA_gene.png")
print(f"- PCA_gene_colored_by_{label_col.replace(' ', '_')}_clean.png")
print(f"- PCA_morph_colored_by_{label_col.replace(' ', '_')}_clean.png")
print("- PCA_scree.png")
print("- gene_pca_variance_explained.csv")
print("- morphology_pca_variance_explained.csv")
print("- joint_pca_variance_explained.csv")
if label_col in meta.columns:
    print(f"- PCA_label_counts_{label_col.replace(' ', '_')}.csv")