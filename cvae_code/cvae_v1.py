from __future__ import annotations

import json
import datetime
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# Config
@dataclass
class TrainConfig:
    metadata_path: str = "metadata_clean.csv"
    morphology_path: str = "morphology_selected_scaled.csv"
    expression_path: str = "expression_hvg_log_normalized.csv"

    zprofiles_path: Optional[str] = "mini-atlas/data/m1_patchseq_morph_zprofiles.csv"

    latent_dim: int = 8
    beta: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    weight_decay: float = 1e-5
    use_huber: bool = False

    # what condition source to use
    condition_mode: str = "g"            # none, g, gs, gsa
    cond_injection: str = "encode_decode"  # decode_only or encode_decode

    # architecture
    hidden_dims_enc: Tuple[int, ...] = (64, 32)
    hidden_dims_cond: Tuple[int, ...] = (128, 64)
    hidden_dims_dec: Tuple[int, ...] = (256, 128, 64)
    cond_emb_dim: int = 32

    # gene embedding reduction
    gene_pca_dim: int = 50

    # train / test
    test_size: float = 0.15
    random_state: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_subsample_indices(
    labels: Optional[pd.Series],
    n_total: int,
    max_points: int = 40000,
    random_state: int = 0,
):
    rng = np.random.default_rng(random_state)
    if n_total <= max_points:
        return np.arange(n_total)

    if labels is None:
        return np.sort(rng.choice(n_total, size=max_points, replace=False))

    labels = labels.astype(str)
    counts = labels.value_counts()
    classes = counts.index.tolist()
    per_class = max(50, max_points // max(1, len(classes)))

    idx_out = []
    arr = labels.to_numpy()
    for cls in classes:
        idx = np.where(arr == cls)[0]
        take = min(len(idx), per_class)
        idx_out.extend(rng.choice(idx, size=take, replace=False).tolist())

    idx_out = np.array(sorted(set(idx_out)))
    if len(idx_out) > max_points:
        idx_out = np.sort(rng.choice(idx_out, size=max_points, replace=False))
    return idx_out


# Patch-seq data loading
def read_indexed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def try_load_zprofiles(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # robust index handling
    candidate_cols = ["cell_id", "cell id", "Cell", "sample", "specimen_id"]
    idx_col = None
    for c in candidate_cols:
        if c in df.columns:
            idx_col = c
            break
    if idx_col is None:
        return None
    df = df.rename(columns={idx_col: "cell_id"}).set_index("cell_id")
    df.index = df.index.astype(str)
    return df


def load_patchseq_tables(cfg: TrainConfig):
    meta = read_indexed_csv(cfg.metadata_path)
    morph = read_indexed_csv(cfg.morphology_path)
    expr = read_indexed_csv(cfg.expression_path)

    zprof = try_load_zprofiles(cfg.zprofiles_path)

    common = meta.index.intersection(morph.index).intersection(expr.index)
    if zprof is not None:
        common = common.intersection(zprof.index)

    meta = meta.loc[common].copy()
    morph = morph.loc[common].copy()
    expr = expr.loc[common].copy()
    if zprof is not None:
        zprof = zprof.loc[common].copy()

    return meta, morph, expr, zprof

# Label / spatial extraction
def ensure_leiden_label(
    meta: pd.DataFrame,
    gene_pcs: np.ndarray,
    random_state: int = 0,
    resolution: float = 1.0,
    n_neighbors: int = 15,
) -> str:
    if "leiden" in meta.columns:
        meta["leiden"] = meta["leiden"].astype(str)
        return "leiden"

    if HAS_SCANPY:
        ad = sc.AnnData(X=gene_pcs.copy())
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")
        sc.tl.leiden(ad, resolution=resolution, key_added="leiden")
        meta["leiden"] = ad.obs["leiden"].astype(str).to_numpy()
        return "leiden"

    km = KMeans(n_clusters=12, random_state=random_state, n_init=10)
    pred = km.fit_predict(gene_pcs)
    meta["leiden"] = pd.Series([f"km_{x}" for x in pred], index=meta.index, dtype="object")
    return "leiden"


def extract_biological_labels(meta: pd.DataFrame) -> Dict[str, pd.Series]:
    labels: Dict[str, pd.Series] = {}

    priority = [
        "RNA type", "leiden", "t_type", "ttype", "t-type", "cell_type", "celltype",
        "annotation", "cluster", "subclass", "class", "layer"
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
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
        )
        is_small_integer_label = (
            pd.api.types.is_integer_dtype(s)
            and 2 <= nunique <= 100
        )

        if (is_categorical_like or is_small_integer_label) and 2 <= nunique <= 100:
            labels[c] = s.astype(str)

    if labels:
        keys = list(labels.keys())
        rna_keys = [k for k in keys if k.lower() in {"rna type", "rna family"}]
        leiden_key = next((k for k in keys if k.lower() == "leiden"), None)

        if rna_keys:
            new_keys: List[str] = []
            # add RNA-derived keys first (in original order)
            for k in keys:
                if k in rna_keys and k not in new_keys:
                    new_keys.append(k)
            # add the rest except leiden
            for k in keys:
                if k not in new_keys and (leiden_key is None or k != leiden_key):
                    new_keys.append(k)
            # finally, if leiden present, append it as fallback
            if leiden_key is not None and leiden_key not in new_keys:
                new_keys.append(leiden_key)

            labels = {k: labels[k] for k in new_keys}

    return labels


def build_spatial_proxy(
    meta: pd.DataFrame,
    zprof: Optional[pd.DataFrame],
) -> Optional[np.ndarray]:
    # best option: z-profiles
    if zprof is not None:
        znum = zprof.select_dtypes(include=[np.number]).copy()
        znum = znum.dropna(axis=1, how="all")
        if znum.shape[1] > 0:
            znum = znum.fillna(znum.median())
            return StandardScaler().fit_transform(znum.to_numpy(dtype=np.float32)).astype(np.float32)

    # fallback: continuous metadata columns that look spatial/depth-related
    candidates = []
    for c in meta.columns:
        cname = c.lower()
        if any(k in cname for k in ["pia", "x", "y", "position"]):
            if any(k in cname for k in ["pia", "x", "y","position"]):
                print("YES")
            if pd.api.types.is_numeric_dtype(meta[c]):
                candidates.append(c)

    if candidates:
        arr = meta[candidates].copy()
        arr = arr.fillna(arr.median())
        return StandardScaler().fit_transform(arr.to_numpy(dtype=np.float32)).astype(np.float32)

    return None


def make_annotation_condition(meta: pd.DataFrame) -> Optional[np.ndarray]:
    labels = extract_biological_labels(meta)
    if not labels:
        return None

    # Build a mapping from a normalized key -> original key to allow
    # robust, case/spacing/punctuation-insensitive matching.
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    norm_to_key = {norm(k): k for k in labels.keys()}

    # Preferred candidates in order. Keep RNA-derived labels first.
    preferred_candidates = ["RNA type", "RNA family", "t_type", "ttype", "t-type", "cell_type", "celltype", "leiden", "cluster", "layer"]

    preferred_key = None
    for cand in preferred_candidates:
        nc = norm(cand)
        if nc in norm_to_key:
            preferred_key = norm_to_key[nc]
            break
    
    # fall back 
    if preferred_key is None:
        for k in labels.keys():
            nk = norm(k)
            if "rna" in nk and "type" in nk:
                preferred_key = k
                break
        if preferred_key is None:
            for k in labels.keys():
                nk = norm(k)
                if "rna" in nk and "family" in nk:
                    preferred_key = k
                    break

    if preferred_key is None:
        preferred_key = list(labels.keys())[0]

    return pd.get_dummies(labels[preferred_key].astype(str), dummy_na=True).to_numpy(dtype=np.float32)


# Condition construction
def make_gene_condition(expr: pd.DataFrame, pca_dim: int = 50) -> np.ndarray:
    X = expr.to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(pca_dim, X.shape[1], X.shape[0] - 1), random_state=0)
    return pca.fit_transform(X).astype(np.float32)


def build_condition_matrix(
    meta: pd.DataFrame,
    expr: pd.DataFrame,
    zprof: Optional[pd.DataFrame],
    mode: str = "none",
    gene_pca_dim: int = 50,
) -> Optional[np.ndarray]:
    mode = mode.lower()
    if mode == "none":
        return None

    parts = []
    gene_pcs = None

    if mode == "a":
        ann = make_annotation_condition(meta)
        if ann is not None:
            parts.append(ann)
    if mode in {"g", "gs", "gsa"}:
        gene_pcs = make_gene_condition(expr, pca_dim=gene_pca_dim)
        parts.append(gene_pcs)

    if mode in {"gs", "gsa"}:
        spatial = build_spatial_proxy(meta, zprof)
        if spatial is not None:
            parts.append(spatial)

    if mode == "gsa":
        ann = make_annotation_condition(meta)
        if ann is not None:
            parts.append(ann)

    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32)


# Data and Model
class MorphologyDataset(Dataset):
    def __init__(self, x: np.ndarray, c: Optional[np.ndarray] = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = None if c is None else torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.c is None:
            return self.x[idx], torch.empty(0)
        return self.x[idx], self.c[idx]


class BetaCVAE(nn.Module):
    """
    Modes:
      - decode_only:   q(z|x),     p(x|z,c)
      - encode_decode: q(z|x,c),   p(x|z,c)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        cond_dim: int = 0,
        cond_injection: str = "encode_decode",
        hidden_dims_enc: Tuple[int, ...] = (64, 32),
        hidden_dims_cond: Tuple[int, ...] = (128, 64),
        hidden_dims_dec: Tuple[int, ...] = (256, 128, 64),
        cond_emb_dim: int = 32,
        use_huber: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.cond_injection = cond_injection
        self.use_huber = use_huber
        self.cond_emb_dim = cond_emb_dim if cond_dim > 0 else 0

        # x encoder
        x_layers = []
        prev = input_dim
        for h in hidden_dims_enc:
            x_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.x_encoder = nn.Sequential(*x_layers)
        self.x_enc_out_dim = prev

        # condition encoder
        if cond_dim > 0:
            c_layers = []
            prev = cond_dim
            for h in hidden_dims_cond:
                c_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            c_layers.append(nn.Linear(prev, self.cond_emb_dim))
            c_layers.append(nn.ReLU())
            self.cond_encoder = nn.Sequential(*c_layers)
        else:
            self.cond_encoder = None

        # posterior head
        if cond_injection == "encode_decode" and cond_dim > 0:
            post_in = self.x_enc_out_dim + self.cond_emb_dim
        else:
            post_in = self.x_enc_out_dim

        self.fc_mu = nn.Linear(post_in, latent_dim)
        self.fc_logvar = nn.Linear(post_in, latent_dim)

        # decoder
        dec_in = latent_dim + (self.cond_emb_dim if cond_dim > 0 else 0)
        dec_layers = []
        prev = dec_in
        for h in hidden_dims_dec:
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode_condition(self, c: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cond_encoder is None:
            return None
        if c is None or c.numel() == 0:
            raise ValueError("Condition tensor required but missing.")
        return self.cond_encoder(c)

    def encode(self, x: torch.Tensor, c: Optional[torch.Tensor] = None):
        hx = self.x_encoder(x)
        if self.cond_injection == "encode_decode" and self.cond_encoder is not None:
            hc = self.encode_condition(c)
            h = torch.cat([hx, hc], dim=-1)
        else:
            h = hx
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: Optional[torch.Tensor] = None):
        if self.cond_encoder is not None:
            hc = self.encode_condition(c)
            h = torch.cat([z, hc], dim=-1)
        else:
            h = z
        return self.decoder(h)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar, z

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor):
        if self.use_huber:
            return F.smooth_l1_loss(x_hat, x, reduction="mean")
        return F.mse_loss(x_hat, x, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


# Training
def train_one_epoch(model, loader, optimizer, beta, device):
    model.train()
    total_loss = total_rec = total_kl = 0.0
    n = 0

    for x, c in loader:
        x = x.to(device)
        c = c.to(device) if c.numel() > 0 else None

        optimizer.zero_grad()
        x_hat, mu, logvar, _ = model(x, c)
        rec = model.reconstruction_loss(x_hat, x)
        kl = kl_divergence(mu, logvar)
        loss = rec + beta * kl
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rec += rec.item() * bs
        total_kl += kl.item() * bs
        n += bs

    return total_loss / n, total_rec / n, total_kl / n


@torch.no_grad()
def eval_one_epoch(model, loader, beta, device):
    model.eval()
    total_loss = total_rec = total_kl = 0.0
    n = 0

    for x, c in loader:
        x = x.to(device)
        c = c.to(device) if c.numel() > 0 else None

        x_hat, mu, logvar, _ = model(x, c)
        rec = model.reconstruction_loss(x_hat, x)
        kl = kl_divergence(mu, logvar)
        loss = rec + beta * kl

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rec += rec.item() * bs
        total_kl += kl.item() * bs
        n += bs

    return total_loss / n, total_rec / n, total_kl / n


@torch.no_grad()
def encode_all(model, X: np.ndarray, C: Optional[np.ndarray], device: str):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    c = None if C is None else torch.tensor(C, dtype=torch.float32, device=device)
    mu, logvar = model.encode(x, c)
    return mu.cpu().numpy(), logvar.cpu().numpy()


# Plotting / metrics
def plot_embedding_numeric(
    E: np.ndarray,
    values: np.ndarray,
    outpath: Path,
    title: str,
    cmap: str = "viridis",
    max_points: int = 40000,
    random_state: int = 0,
):
    idx = balanced_subsample_indices(None, len(values), max_points=max_points, random_state=random_state)
    E2 = E[idx]
    vals = values[idx]
    lo, hi = np.nanpercentile(vals, [1, 99])
    vals = np.clip(vals, lo, hi)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(E2[:, 0], E2[:, 1], c=vals, s=4, alpha=0.6, cmap=cmap)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.colorbar(sc, shrink=0.8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_embedding_categorical(
    E: np.ndarray,
    labels: np.ndarray,
    outpath: Path,
    title: str,
    max_classes: int = 15,
    max_points: int = 40000,
    random_state: int = 0,
):
    labels = pd.Series(labels.astype(str))
    top = labels.value_counts().index[:max_classes]
    keep = labels.isin(top)

    labels2 = labels[keep]
    E2 = E[keep.to_numpy()]
    idx = balanced_subsample_indices(labels2, len(labels2), max_points=max_points, random_state=random_state)
    labels3 = labels2.iloc[idx]
    E3 = E2[idx]

    plt.figure(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab20", len(top))
    for j, lab in enumerate(top):
        mask = labels3 == lab
        plt.scatter(E3[mask, 0], E3[mask, 1], s=5, alpha=0.6, label=lab, color=cmap(j))

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def compute_embedding_metrics(
    Z_mean: np.ndarray,
    labels: pd.Series,
    label_name: str,
    random_state: int = 0,
) -> dict:
    metrics = {}
    labels = labels.astype(str)
    vc = labels.value_counts()
    keep_classes = vc[vc >= 20].index
    keep = labels.isin(keep_classes)

    if keep.sum() < 200 or len(keep_classes) < 2:
        return metrics

    Zk = Z_mean[keep.to_numpy()]
    y = labels[keep].to_numpy()

    try:
        metrics[f"{label_name}_silhouette"] = float(silhouette_score(Zk, y))
    except Exception:
        pass

    try:
        Xtr, Xte, ytr, yte = sk_train_test_split(
            Zk, y, test_size=0.2, random_state=random_state, stratify=y
        )
        clf = KNeighborsClassifier(n_neighbors=15)
        clf.fit(Xtr, ytr)
        metrics[f"{label_name}_knn_accuracy"] = float(clf.score(Xte, yte))
    except Exception:
        pass

    try:
        n_classes = len(np.unique(y))
        km = KMeans(n_clusters=n_classes, random_state=random_state, n_init=10)
        pred = km.fit_predict(Zk)
        metrics[f"{label_name}_latent_kmeans_nmi"] = float(normalized_mutual_info_score(y, pred))
        metrics[f"{label_name}_latent_kmeans_ari"] = float(adjusted_rand_score(y, pred))
    except Exception:
        pass

    return metrics


def save_gaussian_diagnostics(mu_all, logvar_all, outdir: Path, tag: str):
    std_all = np.exp(0.5 * logvar_all)
    latent_dim = mu_all.shape[1]

    fig, axes = plt.subplots(1, latent_dim, figsize=(4 * latent_dim, 3.5))
    if latent_dim == 1:
        axes = [axes]
    for i in range(latent_dim):
        axes[i].hist(mu_all[:, i], bins=80, alpha=0.85)
        axes[i].set_title(f"mu dim {i}")
    plt.tight_layout()
    plt.savefig(outdir / f"gaussian_mu_hist_{tag}.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, latent_dim, figsize=(4 * latent_dim, 3.5))
    if latent_dim == 1:
        axes = [axes]
    for i in range(latent_dim):
        axes[i].hist(std_all[:, i], bins=80, alpha=0.85)
        axes[i].set_title(f"std dim {i}")
    plt.tight_layout()
    plt.savefig(outdir / f"gaussian_std_hist_{tag}.png", dpi=220)
    plt.close()

    return {
        "latent_std_by_dim": mu_all.std(axis=0).tolist(),
        "latent_mean_by_dim": mu_all.mean(axis=0).tolist(),
        "mean_posterior_logvar_by_dim": logvar_all.mean(axis=0).tolist(),
        "mean_posterior_std_by_dim": std_all.mean(axis=0).tolist(),
    }


def save_latent_mean_analysis(
    meta: pd.DataFrame,
    Z_mean: np.ndarray,
    label_dict: Dict[str, pd.Series],
    spatial_proxy: Optional[np.ndarray],
    outdir: Path,
    tag: str,
    random_state: int = 0,
):
    metrics = {}

    pca = PCA(n_components=2, random_state=random_state)
    E_pca = pca.fit_transform(Z_mean)

    plt.figure(figsize=(6, 5))
    idx_bg = balanced_subsample_indices(None, len(E_pca), max_points=40000, random_state=random_state)
    plt.scatter(E_pca[idx_bg, 0], E_pca[idx_bg, 1], s=4, alpha=0.25)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Latent mean PCA ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"latent_mean_pca_{tag}.png", dpi=220)
    plt.close()

    # numeric overlays from metadata
    for col in meta.columns:
        if pd.api.types.is_numeric_dtype(meta[col]):
            if any(k in col.lower() for k in ["depth", "layer", "pia"]):
                plot_embedding_numeric(
                    E_pca,
                    meta[col].to_numpy(dtype=float),
                    outdir / f"latent_mean_pca_{tag}_{col}.png",
                    f"Latent mean PCA ({tag}) colored by {col}",
                    random_state=random_state,
                )

    # spatial proxy overlays if present
    if spatial_proxy is not None and spatial_proxy.shape[1] >= 1:
        plot_embedding_numeric(
            E_pca,
            spatial_proxy[:, 0],
            outdir / f"latent_mean_pca_{tag}_spatial_proxy_0.png",
            f"Latent mean PCA ({tag}) colored by spatial proxy 0",
            random_state=random_state,
        )

    # label overlays
    for name, series in label_dict.items():
        plot_embedding_categorical(
            E_pca,
            series.to_numpy(),
            outdir / f"latent_mean_pca_{tag}_{name}.png",
            f"Latent mean PCA ({tag}) colored by {name}",
            max_classes=20 if name == "leiden" else 15,
            random_state=random_state,
        )
        metrics.update(compute_embedding_metrics(Z_mean, series, name, random_state=random_state))

    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.15,
            metric="euclidean",
            random_state=random_state,
        )
        E_umap = reducer.fit_transform(Z_mean)

        plt.figure(figsize=(6, 5))
        idx_bg = balanced_subsample_indices(None, len(E_umap), max_points=40000, random_state=random_state)
        plt.scatter(E_umap[idx_bg, 0], E_umap[idx_bg, 1], s=4, alpha=0.25)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title(f"Latent mean UMAP ({tag})")
        plt.tight_layout()
        plt.savefig(outdir / f"latent_mean_umap_{tag}.png", dpi=220)
        plt.close()

        for name, series in label_dict.items():
            plot_embedding_categorical(
                E_umap,
                series.to_numpy(),
                outdir / f"latent_mean_umap_{tag}_{name}.png",
                f"Latent mean UMAP ({tag}) colored by {name}",
                max_classes=20 if name == "leiden" else 15,
                random_state=random_state,
            )

    with open(outdir / f"latent_mean_metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# Main
def main():
    cfg = TrainConfig(
        metadata_path = "metadata_clean.csv",
        morphology_path = "morphology_selected_scaled.csv",
        expression_path = "expression_hvg_log_normalized.csv",
        zprofiles_path="mini-atlas/data/m1_patchseq_morph_zprofiles.csv",
        latent_dim=8,
        beta=1e-3,
        lr=1e-3,
        batch_size=256,
        epochs=1500,
        condition_mode="gsa",          # none, g, gs, gsa
        cond_injection="encode_decode",  # decode_only or encode_decode
        hidden_dims_enc=(256, 128, 64),
        hidden_dims_cond=(256, 128, 64),
        hidden_dims_dec=(256, 128, 64),
        cond_emb_dim=32,
        use_huber=False,
        gene_pca_dim=50,
    )

    set_seed(cfg.random_state)

    meta, morph, expr, zprof = load_patchseq_tables(cfg)

    gene_pcs = make_gene_condition(expr, pca_dim=cfg.gene_pca_dim)
    ensure_leiden_label(meta, gene_pcs, random_state=cfg.random_state)

    condition = build_condition_matrix(
        meta=meta,
        expr=expr,
        zprof=zprof,
        mode=cfg.condition_mode,
        gene_pca_dim=cfg.gene_pca_dim,
    )

    spatial_proxy = build_spatial_proxy(meta, zprof)
    label_dict = extract_biological_labels(meta)

    # morphology input
    M = morph.to_numpy(dtype=np.float32)

    idx = np.arange(M.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    X_train = M[train_idx]
    X_test = M[test_idx]
    X_all = M

    C_train = C_test = C_all = None
    if condition is not None:
        C_train = condition[train_idx]
        C_test = condition[test_idx]
        C_all = condition

    train_ds = MorphologyDataset(X_train, C_train)
    test_ds = MorphologyDataset(X_test, C_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    cond_dim = 0 if C_train is None else C_train.shape[1]
    model = BetaCVAE(
        input_dim=X_train.shape[1],
        latent_dim=cfg.latent_dim,
        cond_dim=cond_dim,
        cond_injection=cfg.cond_injection,
        hidden_dims_enc=cfg.hidden_dims_enc,
        hidden_dims_cond=cfg.hidden_dims_cond,
        hidden_dims_dec=cfg.hidden_dims_dec,
        cond_emb_dim=cfg.cond_emb_dim,
        use_huber=cfg.use_huber,
    ).to(cfg.device)

    print(
        f"Model config: input_dim={X_train.shape[1]}, latent_dim={cfg.latent_dim}, "
        f"beta={cfg.beta}, lr={cfg.lr}, cond_dim={cond_dim}, "
        f"condition_mode={cfg.condition_mode}, cond_injection={cfg.cond_injection}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_rec, tr_kl = train_one_epoch(model, train_loader, optimizer, cfg.beta, cfg.device)
        va_loss, va_rec, va_kl = eval_one_epoch(model, test_loader, cfg.beta, cfg.device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_rec": tr_rec,
            "train_kl": tr_kl,
            "val_loss": va_loss,
            "val_rec": va_rec,
            "val_kl": va_kl,
        })

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | train loss {tr_loss:.4f} rec {tr_rec:.4f} kl {tr_kl:.4f} | "
                f"val loss {va_loss:.4f} rec {va_rec:.4f} kl {va_kl:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    mu_all, logvar_all = encode_all(model, X_all, C_all, cfg.device)

    tag = f"patchseq_{cfg.condition_mode}_{cfg.cond_injection}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results_cvae_celltype") / tag / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "morph_features": morph.columns.tolist(),
        },
        run_dir / f"{tag}.pt",
    )

    latent_cols = [f"morph_latent_mean_{i}" for i in range(cfg.latent_dim)]
    latent_df = pd.DataFrame(mu_all, index=meta.index, columns=latent_cols)
    latent_df.to_csv(run_dir / f"latent_vectors_{tag}.csv")

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / f"train_history_{tag}.csv", index=False)

    with open(run_dir / f"detected_biological_labels_{tag}.json", "w") as f:
        json.dump({k: int(v.nunique()) for k, v in label_dict.items()}, f, indent=2)

    gaussian_metrics = save_gaussian_diagnostics(mu_all, logvar_all, run_dir, tag)
    latent_metrics = save_latent_mean_analysis(meta, mu_all, label_dict, spatial_proxy, run_dir, tag, cfg.random_state)

    metrics = {
        "condition_mode": cfg.condition_mode,
        "cond_injection": cfg.cond_injection,
        "latent_dim": cfg.latent_dim,
        "beta": cfg.beta,
        "learning_rate": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "n_cells": int(meta.shape[0]),
        "n_morph_features": int(morph.shape[1]),
        "n_genes": int(expr.shape[1]),
        "cond_dim": int(cond_dim),
        "best_val_loss": float(history_df["val_loss"].min()),
        "best_val_rec": float(history_df.loc[history_df["val_loss"].idxmin(), "val_rec"]),
        "best_val_kl": float(history_df.loc[history_df["val_loss"].idxmin(), "val_kl"]),
        "detected_label_names": list(label_dict.keys()),
    }
    metrics.update(gaussian_metrics)
    metrics.update(latent_metrics)

    with open(run_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved PatchSeq run to: {run_dir}")


if __name__ == "__main__":
    main()