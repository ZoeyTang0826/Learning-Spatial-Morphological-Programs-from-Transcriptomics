import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------------
# 1. Load PCA results + metadata
# -------------------------------

gene_pca = pd.read_csv("PCA/gene_pca.csv", index_col=0)
meta = pd.read_csv("patch-seq_preprocess/metadata_clean.csv", index_col=0)

print("Gene PCA shape:", gene_pca.shape)
print("Metadata shape:", meta.shape)

# align cells
common_cells = gene_pca.index.intersection(meta.index)
gene_pca = gene_pca.loc[common_cells]
meta = meta.loc[common_cells]

print("\nAfter alignment:")
print("Gene PCA shape:", gene_pca.shape)
print("Metadata shape:", meta.shape)

print("\nMetadata columns:")
print(meta.columns.tolist())

# -------------------------------
# 2. Run t-SNE on PCA coordinates
# -------------------------------

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0
)

# use first 10 PCs
gene_tsne = tsne.fit_transform(gene_pca.iloc[:, :10])

print("t-SNE finished.")

# save embedding
tsne_df = pd.DataFrame(
    gene_tsne,
    index=common_cells,
    columns=["tSNE1", "tSNE2"]
)

tsne_df.to_csv("gene_tsne.csv")

# -------------------------------
# 3. Plot plain t-SNE
# -------------------------------

plt.figure(figsize=(7, 6))
plt.scatter(
    gene_tsne[:, 0],
    gene_tsne[:, 1],
    s=8
)

plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.title("t-SNE of Gene Expression")
plt.tight_layout()
plt.savefig("TSNE_gene.png", dpi=300)
plt.show()

# -------------------------------
# 4. Plot t-SNE colored by RNA type
#    using a DISCRETE legend
# -------------------------------

label_col = "RNA type"

if label_col in meta.columns:
    labels = meta[label_col].astype(str)

    # keep only top N most frequent labels for readability
    top_n = 10
    top_labels = labels.value_counts().head(top_n).index
    labels_plot = labels.where(labels.isin(top_labels), other="Other")

    unique_labels = sorted(labels_plot.unique())
    cmap = plt.get_cmap("tab20", len(unique_labels))

    plt.figure(figsize=(8, 6))

    for i, lab in enumerate(unique_labels):
        idx = labels_plot == lab
        plt.scatter(
            gene_tsne[idx, 0],
            gene_tsne[idx, 1],
            s=18,
            alpha=0.8,
            color=cmap(i),
            label=lab
        )

    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title(f"t-SNE of Gene Expression Colored by {label_col}")
    plt.legend(
        title=label_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        markerscale=1.2
    )
    plt.tight_layout()
    plt.savefig("TSNE_gene_RNA_type_discrete.png", dpi=300, bbox_inches="tight")
    plt.show()

    # save counts of labels shown in the legend
    label_count_df = labels_plot.value_counts().rename_axis("label").reset_index(name="count")
    label_count_df.to_csv("TSNE_RNA_type_label_counts.csv", index=False)

    print("\nRNA type labels used in discrete plot:")
    print(label_count_df)

else:
    print(f"\nColumn '{label_col}' not found in metadata.")
    print("Available columns:")
    print(meta.columns.tolist())

print("\nSaved files:")
print("- gene_tsne.csv")
print("- TSNE_gene.png")
print("- TSNE_gene_RNA_type_discrete.png")
if label_col in meta.columns:
    print("- TSNE_RNA_type_label_counts.csv")