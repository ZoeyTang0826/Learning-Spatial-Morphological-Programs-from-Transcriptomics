import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# -------------------------------
# 1. Load PCA results + metadata
# -------------------------------

gene_pca = pd.read_csv("PCA/gene_pca.csv", index_col=0)
meta = pd.read_csv("metadata_clean.csv", index_col=0)

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
# 2. Choose input + label column
# -------------------------------

# use first 10 PCs
X = gene_pca.iloc[:, :10].copy()

label_col = "RNA type"
if label_col not in meta.columns:
    raise ValueError(f"Column '{label_col}' not found in metadata.")

labels_true = meta[label_col].astype(str)

# -------------------------------
# 3. Build AnnData and run Leiden
# -------------------------------

adata = sc.AnnData(X)
adata.obs_names = common_cells
adata.obs[label_col] = labels_true.values

# neighbor graph on PCA coordinates
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")

# resolution controls number of clusters
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_cluster")

cluster_labels = adata.obs["leiden_cluster"].astype(str)

cluster_df = pd.DataFrame({
    "cell_id": common_cells,
    "Leiden_cluster": cluster_labels.values,
    label_col: labels_true.values
}).set_index("cell_id")

cluster_df.to_csv("leiden_clusters.csv")

print("\nCluster counts:")
print(cluster_df["Leiden_cluster"].value_counts().sort_index())

# -------------------------------
# 4. Evaluate clustering vs metadata
# -------------------------------

true_codes = labels_true.astype("category").cat.codes
pred_codes = cluster_labels.astype("category").cat.codes

ari = adjusted_rand_score(true_codes, pred_codes)
nmi = normalized_mutual_info_score(true_codes, pred_codes)

print(f"\nAdjusted Rand Index (ARI) vs {label_col}: {ari:.4f}")
print(f"Normalized Mutual Info (NMI) vs {label_col}: {nmi:.4f}")

metrics_df = pd.DataFrame({
    "metric": ["ARI", "NMI"],
    "value": [ari, nmi]
})
metrics_df.to_csv("leiden_clustering_metrics.csv", index=False)

# -------------------------------
# 5. Plot Leiden clusters on first two PCs
# -------------------------------

plt.figure(figsize=(7, 6))
plt.scatter(
    gene_pca.iloc[:, 0],
    gene_pca.iloc[:, 1],
    c=pred_codes,
    cmap="tab10",
    s=12
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Leiden Clusters on Gene PCA")
plt.colorbar(label="Leiden cluster")
plt.tight_layout()
plt.savefig("Leiden_clusters_on_gene_PCA.png", dpi=300)
plt.show()

# -------------------------------
# 6. Plot Leiden clusters on t-SNE if available
# -------------------------------

try:
    tsne = pd.read_csv("gene_tsne.csv", index_col=0)
    tsne = tsne.loc[common_cells]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        tsne.iloc[:, 0],
        tsne.iloc[:, 1],
        c=pred_codes,
        cmap="tab10",
        s=12
    )

    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title("Leiden Clusters on Gene t-SNE")
    plt.colorbar(label="Leiden cluster")
    plt.tight_layout()
    plt.savefig("Leiden_clusters_on_tSNE.png", dpi=300)
    plt.show()

except FileNotFoundError:
    print("\n gene_tsne.csv not found, skipping t-SNE cluster plot.")

# -------------------------------
# 7. Cluster vs RNA type table
# -------------------------------

cluster_vs_label = pd.crosstab(
    cluster_df["Leiden_cluster"],
    cluster_df[label_col]
)

cluster_vs_label.to_csv("leiden_cluster_vs_RNAtype_table.csv")

print("\nCluster vs RNA type table:")
print(cluster_vs_label)

print("\nSaved files:")
print("- leiden_clusters.csv")
print("- leiden_clustering_metrics.csv")
print("- Leiden_clusters_on_gene_PCA.png")
print("- Leiden_clusters_on_tSNE.png (if gene_tsne.csv exists)")
print("- leiden_cluster_vs_RNAtype_table.csv")