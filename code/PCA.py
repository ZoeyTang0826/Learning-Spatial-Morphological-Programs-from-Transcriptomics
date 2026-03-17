import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1. Load preprocessed data
# -------------------------------

# gene expression
gene = pd.read_csv("patch-seq_preprocess/expression_hvg_log_normalized.csv", index_col=0)

# metadata
meta = pd.read_csv("patch-seq_preprocess/metadata_clean.csv", index_col=0)

print("Gene expression shape:", gene.shape)
print("Metadata shape:", meta.shape)

print("\nMetadata columns:")
print(meta.columns.tolist())

# -------------------------------
# 2. Align datasets by cell ID
# -------------------------------

common_cells = gene.index.intersection(meta.index)

gene = gene.loc[common_cells]
meta = meta.loc[common_cells]

print("\nAfter alignment:")
print("Gene expression shape:", gene.shape)
print("Metadata shape:", meta.shape)

# -------------------------------
# 3. PCA on Gene Expression only
# -------------------------------

pca_g = PCA(n_components=10)
gene_pca = pca_g.fit_transform(gene)

print("\nGene PCA variance explained:")
print(pca_g.explained_variance_ratio_)

# -------------------------------
# 4. Save PCA coordinates
# -------------------------------

gene_pca_df = pd.DataFrame(
    gene_pca,
    index=common_cells,
    columns=[f"PC{i+1}" for i in range(gene_pca.shape[1])]
)

gene_pca_df.to_csv("gene_pca.csv")

# -------------------------------
# 5. Plot PCA (Gene Expression)
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
# 6. Clean plotting helper
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

    label_count_df = (
        labels_plot.value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    return label_count_df

# -------------------------------
# 7. Plot gene PCA colored by metadata label
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
# 8. Scree plot for gene PCA
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
# 9. Save explained variance table
# -------------------------------

gene_var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca_g.explained_variance_ratio_))],
    "variance_explained": pca_g.explained_variance_ratio_
})

gene_var_df.to_csv("gene_pca_variance_explained.csv", index=False)

print("\nSaved files:")
print("- gene_pca.csv")
print("- PCA_gene.png")
print(f"- PCA_gene_colored_by_{label_col.replace(' ', '_')}_clean.png")
print("- PCA_scree.png")
print("- gene_pca_variance_explained.csv")
if label_col in meta.columns:
    print(f"- PCA_label_counts_{label_col.replace(' ', '_')}.csv")