from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pandas.api.types import CategoricalDtype


# CONFIG
REP_CSV = "path/to/conditional_alpha.csv"
METADATA_CSV = "path/to/metadata_clean.csv"
OUT_DIR = Path("/path/to/output/clustering")

TEST_SIZE = 0.2
RANDOM_STATE = 0
KNN_K = 10
MIN_CLASS_COUNT = 2
KMEANS_N_INIT = 20
MAX_LABELS_IN_PLOT = 12

# LABEL EXTRACTION
def extract_biological_labels(meta: pd.DataFrame) -> Dict[str, pd.Series]:
    labels: Dict[str, pd.Series] = {}
    priority = [
        "RNA type", "RNA family", "leiden", "t_type", "ttype", "t-type",
        "cell_type", "celltype", "annotation", "cluster", "subclass",
        "class", "layer"
    ]
    for c in priority:
        if c in meta.columns:
            labels[c] = meta[c].astype(str)

    for c in meta.columns:
        if c in labels:
            continue
        s = meta[c]
        nunique = s.astype(str).nunique(dropna=False)
        is_categorical_like = (
            pd.api.types.is_object_dtype(s) or isinstance(s.dtype, CategoricalDtype)
        )
        is_small_integer_label = pd.api.types.is_integer_dtype(s) and 2 <= nunique <= 100
        if (is_categorical_like or is_small_integer_label) and 2 <= nunique <= 100:
            labels[c] = s.astype(str)
    return labels


def choose_label(meta: pd.DataFrame) -> Optional[str]:
    labels = extract_biological_labels(meta)
    if not labels:
        return None
    preferred = ["RNA type", "leiden", "RNA family", "cell_type", "cluster"]
    for p in preferred:
        if p in labels:
            return p
    return list(labels.keys())[0]


def choose_depth_col(meta: pd.DataFrame) -> Optional[str]:
    preferred = [c for c in meta.columns if "depth" in c.lower()]
    return preferred[0] if preferred else None

# METRICS
def knn_accuracy_and_preds(X: np.ndarray, y: np.ndarray, k: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    k_eff = min(k, max(1, len(X_train) - 1))
    clf = KNeighborsClassifier(n_neighbors=k_eff)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(np.mean(y_pred == y_test))
    return acc, y_test, y_pred


def clustering_scores(X: np.ndarray, y: np.ndarray):
    n_clusters = len(np.unique(y))
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    pred = km.fit_predict(X)
    n_unique_pred = len(np.unique(pred))
    nmi = normalized_mutual_info_score(y, pred)
    return float(nmi), pred


def plot_pca_by_label(E: np.ndarray, labels: pd.Series, outpath: Path):
    top = labels.value_counts().index[:MAX_LABELS_IN_PLOT]
    keep = labels.isin(top)
    E2 = E[keep.to_numpy()]
    labels2 = labels[keep]

    plt.figure(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab20", len(top))
    for j, lab in enumerate(top):
        mask = labels2 == lab
        plt.scatter(E2[mask.to_numpy(), 0], E2[mask.to_numpy(), 1], s=12, alpha=0.7, label=lab, color=cmap(j))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA colored by label")
    plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_pca_by_kmeans(E: np.ndarray, pred: np.ndarray, outpath: Path):
    unique_clusters = np.unique(pred)
    n = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", max(1, n))

    plt.figure(figsize=(6, 5))
    for j, lab in enumerate(unique_clusters):
        mask = pred == lab
        plt.scatter(E[mask, 0], E[mask, 1], c=[cmap(j)], s=12, alpha=0.8, label=str(lab))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA colored by k-means cluster")
    if n <= MAX_LABELS_IN_PLOT:
        plt.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

# MAIN
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rep = pd.read_csv(REP_CSV, index_col=0)
    meta = pd.read_csv(METADATA_CSV, index_col=0)

    common = rep.index.intersection(meta.index)
    rep = rep.loc[common]
    meta = meta.loc[common]

    label_name = choose_label(meta)
    if label_name is None:
        raise RuntimeError("No valid biological label found in metadata.")

    depth_col = choose_depth_col(meta)

    y_all = meta[label_name].astype(str)
    counts_all = y_all.value_counts().sort_values(ascending=False)

    keep_classes = counts_all[counts_all >= MIN_CLASS_COUNT].index
    if len(keep_classes) == 0:
        keep_classes = counts_all.index
    keep = y_all.isin(keep_classes)

    rep_f = rep.loc[keep]
    meta_f = meta.loc[keep]
    y = y_all.loc[keep].to_numpy()

    X = rep_f.to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X)

    print(f"Loaded representation: {rep.shape}")
    print(f"Using label: {label_name}")
    print(f"Retained {len(keep_classes)} classes and {len(y)} cells with MIN_CLASS_COUNT={MIN_CLASS_COUNT}")

    # metrics
    knn_acc, y_test, y_pred = knn_accuracy_and_preds(X, y, KNN_K)
    ari, kmeans_pred = clustering_scores(X, y)

    # PCA for plots
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    E = pca.fit_transform(X)

    result = {
        "representation_file": REP_CSV,
        "label_used": label_name,
        "depth_col": depth_col,
        "min_class_count": MIN_CLASS_COUNT,
        "n_samples_before_filter": int(len(rep)),
        "n_classes_before_filter": int(y_all.nunique()),
        "n_samples_after_filter": int(len(X)),
        "n_classes_after_filter": int(len(np.unique(y))),
        "dim": int(X.shape[1]),
        "knn_k": KNN_K,
        "knn_accuracy": float(knn_acc),
        "kmeans_ari": float(ari),
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # plots
    plot_pca_by_label(E, pd.Series(y, index=rep_f.index), OUT_DIR / "pca_by_label.png")
    plot_pca_by_kmeans(E, kmeans_pred, OUT_DIR / "pca_by_kmeans.png")

    print(json.dumps(result, indent=2))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()