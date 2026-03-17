import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# -------------------------------
# 1. Load PCA results + metadata
# -------------------------------

gene_pca = pd.read_csv("PCA/gene_pca.csv", index_col=0)
meta = pd.read_csv("patch-seq_preprocess/metadata_clean.csv", index_col=0)
tsne = pd.read_csv("TSNE/gene_tsne.csv", index_col=0)


print("Gene PCA shape:", gene_pca.shape)
print("Metadata shape:", meta.shape)

# align cells
common_cells = gene_pca.index.intersection(meta.index)
gene_pca = gene_pca.loc[common_cells]
meta = meta.loc[common_cells]
tsne = tsne.loc[common_cells]

print("\nAfter alignment:")
print("Gene PCA shape:", gene_pca.shape)
print("Metadata shape:", meta.shape)

print("\nMetadata columns:")
print(meta.columns.tolist())

# -------------------------------
# 2. Choose input + label column
# -------------------------------

# use first 10 PCs for clustering
X = gene_pca.iloc[:, :10].values

# choose a biological label for evaluation
label_col = "RNA type"

if label_col not in meta.columns:
    raise ValueError(f"Column '{label_col}' not found in metadata.")

labels_true = meta[label_col].astype(str)

# -------------------------------
# 3. Run GMM clustering
# -------------------------------

# choose number of clusters
n_clusters = 8

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    random_state=0
)

cluster_labels = gmm.fit_predict(X)

# save clusters
cluster_df = pd.DataFrame({
    "cell_id": common_cells,
    "GMM_cluster": cluster_labels,
    label_col: labels_true.values
}).set_index("cell_id")

cluster_df.to_csv("gmm_clusters.csv")

print("\nCluster counts:")
print(cluster_df["GMM_cluster"].value_counts().sort_index())

# -------------------------------
# 4. Evaluate clustering vs metadata
# -------------------------------

# convert biological labels to integers
true_codes = labels_true.astype("category").cat.codes

ari = adjusted_rand_score(true_codes, cluster_labels)
nmi = normalized_mutual_info_score(true_codes, cluster_labels)

print(f"\nAdjusted Rand Index (ARI) vs {label_col}: {ari:.4f}")
print(f"Normalized Mutual Info (NMI) vs {label_col}: {nmi:.4f}")

metrics_df = pd.DataFrame({
    "metric": ["ARI", "NMI"],
    "value": [ari, nmi]
})

metrics_df.to_csv("gmm_clustering_metrics.csv", index=False)

# -------------------------------
# 5. Plot GMM clusters on first two PCs
# -------------------------------

plt.figure(figsize=(7, 6))

colors = plt.get_cmap("tab10").colors[:n_clusters]
cmap = ListedColormap(colors)
bounds = np.arange(n_clusters + 1)          # 0,1,2,...,8
norm = BoundaryNorm(bounds, cmap.N)

sc = plt.scatter(
    gene_pca.iloc[:, 0],
    gene_pca.iloc[:, 1],
    c=cluster_labels,
    cmap=cmap,
    norm=norm,
    s=12
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clusters on Gene PCA")

cbar = plt.colorbar(sc, boundaries=bounds)
cbar.set_label("Cluster ID")
cbar.set_ticks(np.arange(n_clusters) + 0.5)   # centers of blocks
cbar.set_ticklabels(range(n_clusters))

plt.tight_layout()
plt.savefig("GMM_clusters_on_gene_PCA.png", dpi=300)
plt.show()

# -------------------------------
# 6. Plot GMM clusters on t-SNE
# -------------------------------
plt.figure(figsize=(7, 6))

colors = plt.get_cmap("tab10").colors[:n_clusters]
cmap = ListedColormap(colors)
bounds = np.arange(n_clusters + 1)          # 0,1,2,...,8
norm = BoundaryNorm(bounds, cmap.N)

sc = plt.scatter(
    tsne.iloc[:, 0],
    tsne.iloc[:, 1],
    c=cluster_labels,
    cmap=cmap,
    norm=norm,
    s=12
)

plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.title("GMM Clusters on Gene t-SNE")

cbar = plt.colorbar(sc, boundaries=bounds)
cbar.set_label("Cluster ID")
cbar.set_ticks(np.arange(n_clusters) + 0.5)   # centers of blocks
cbar.set_ticklabels(range(n_clusters))

plt.tight_layout()
plt.savefig("GMM_clusters_on_tSNE.png", dpi=300)
plt.show()
# -------------------------------
# 7. Crosstab: clusters vs RNA type
# -------------------------------

cluster_vs_label = pd.crosstab(
    cluster_df["GMM_cluster"],
    cluster_df[label_col]
)

cluster_vs_label.to_csv("gmm_cluster_vs_RNAtype_table.csv")

print("\nCluster vs RNA type table:")
print(cluster_vs_label)

print("\nSaved files:")
print("- gmm_clusters.csv")
print("- gmm_clustering_metrics.csv")
print("- GMM_clusters_on_gene_PCA.png")
print("- GMM_clusters_on_tSNE.png (if gene_tsne.csv exists)")
print("- gmm_cluster_vs_RNAtype_table.csv")