from __future__ import annotations

import json
import copy
import datetime
import itertools
import random
from dataclasses import dataclass, asdict
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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

    condition_mode: str = "sa"   # use sa based on your results
    gene_pca_dim: int = 50

    latent_dim: int = 8
    beta: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 1000
    weight_decay: float = 1e-5

    encoder_hidden_dims: Tuple[int, ...] = (256, 128, 64)
    decoder_hidden_dims: Tuple[int, ...] = (256, 128, 64)
    cond_hidden_dims: Tuple[int, ...] = (128, 64)

    # residual branch regularization
    lambda_alpha_l2: float = 1e-4
    lambda_residual_scale: float = 0.0

    # feature weighting
    use_feature_weights: bool = False
    feature_weight_mode: str = "inverse_predictability"
    predictability_csv: Optional[str] = "results_morph_diagnostics/feature_predictability_sa.csv"

    test_size: float = 0.15
    random_state: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AblationConfig:
    pc_sets: Tuple[Tuple[int, ...], ...] = (
        (0,),          # PC1
        (0, 1, 2),     # PC1-3
        (0, 1, 2, 4),  # PC1,2,3,5
        (0, 1, 2, 4, 9) # PC1,2,3,5,10
    )

    warm_start_epochs_list: Tuple[int, ...] = (
        0,
        200,
    )

    residual_scale_modes: Tuple[str, ...] = (
        "none",       # x_hat = x_base + x_cond
        "scalar",     # x_hat = x_base + tau * x_cond
        "featurewise" # x_hat = x_base + tau_vec ⊙ x_cond
    )

    # initial scale value; small is safer
    residual_scale_init: float = 0.1

    # Optional: reduce runs during debugging
    max_runs: Optional[int] = None


# Utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

# Condition construction
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

    bins = pd.qcut(gene_pcs[:, 0], q=min(12, max(2, len(gene_pcs) // 30)), duplicates="drop")
    meta["leiden"] = bins.astype(str)
    return "leiden"


def extract_biological_labels(meta: pd.DataFrame) -> Dict[str, pd.Series]:
    labels: Dict[str, pd.Series] = {}
    priority = [
        "RNA type", "RNA family", "leiden", "t_type", "ttype", "t-type", "cell_type",
        "celltype", "annotation", "cluster", "subclass", "class", "layer"
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
            pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)
        )
        is_small_integer_label = pd.api.types.is_integer_dtype(s) and 2 <= nunique <= 100
        if (is_categorical_like or is_small_integer_label) and 2 <= nunique <= 100:
            labels[c] = s.astype(str)

    return labels


def choose_annotation_label(meta: pd.DataFrame) -> Optional[str]:
    labels = extract_biological_labels(meta)
    if not labels:
        return None

    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    norm_to_key = {norm(k): k for k in labels.keys()}
    preferred_candidates = [
        "RNA type", "RNA family", "t_type", "ttype", "t-type",
        "cell_type", "celltype", "leiden", "cluster", "layer"
    ]
    for cand in preferred_candidates:
        nc = norm(cand)
        if nc in norm_to_key:
            return norm_to_key[nc]
    return list(labels.keys())[0]


def build_spatial_proxy(meta: pd.DataFrame, zprof: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    if zprof is not None:
        znum = zprof.select_dtypes(include=[np.number]).copy()
        znum = znum.dropna(axis=1, how="all")
        if znum.shape[1] > 0:
            znum = znum.fillna(znum.median())
            return StandardScaler().fit_transform(
                znum.to_numpy(dtype=np.float32)
            ).astype(np.float32)

    candidates = []
    for c in meta.columns:
        cname = c.lower()
        if any(k in cname for k in ["pia", "x", "y", "coord", "position"]):
            if pd.api.types.is_numeric_dtype(meta[c]):
                candidates.append(c)

    if candidates:
        arr = meta[candidates].copy()
        arr = arr.fillna(arr.median())
        return StandardScaler().fit_transform(
            arr.to_numpy(dtype=np.float32)
        ).astype(np.float32)

    return None


def make_annotation_condition(meta: pd.DataFrame) -> Optional[np.ndarray]:
    label_name = choose_annotation_label(meta)
    if label_name is None:
        return None
    return pd.get_dummies(meta[label_name].astype(str), dummy_na=True).to_numpy(dtype=np.float32)


def make_gene_condition(expr: pd.DataFrame, pca_dim: int = 50) -> np.ndarray:
    X = expr.to_numpy(dtype=np.float32)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(pca_dim, X.shape[1], X.shape[0] - 1), random_state=0)
    return pca.fit_transform(X).astype(np.float32)


def build_condition_matrix(
    meta: pd.DataFrame,
    expr: pd.DataFrame,
    zprof: Optional[pd.DataFrame],
    mode: str,
    gene_pca_dim: int,
) -> Optional[np.ndarray]:
    mode = mode.lower()
    if mode == "none":
        return None

    gene = make_gene_condition(expr, pca_dim=gene_pca_dim)
    spatial = build_spatial_proxy(meta, zprof)
    annot = make_annotation_condition(meta)

    parts = []
    if mode == "g":
        parts = [gene]
    elif mode == "s":
        parts = [spatial]
    elif mode == "a":
        parts = [annot]
    elif mode == "gs":
        parts = [gene, spatial]
    elif mode == "ga":
        parts = [gene, annot]
    elif mode == "sa":
        parts = [spatial, annot]
    elif mode == "gsa":
        parts = [gene, spatial, annot]
    else:
        raise ValueError(f"Unknown condition mode: {mode}")

    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32)

# Feature weights
def build_feature_weights(cfg: TrainConfig, morph_columns: List[str]) -> np.ndarray:
    if not cfg.use_feature_weights:
        return np.ones(len(morph_columns), dtype=np.float32)

    if cfg.feature_weight_mode == "uniform" or cfg.predictability_csv is None:
        return np.ones(len(morph_columns), dtype=np.float32)

    df = pd.read_csv(cfg.predictability_csv)
    if "feature" not in df.columns or "r2" not in df.columns:
        raise ValueError("predictability_csv must contain feature,r2")

    r2_map = dict(zip(df["feature"].astype(str), df["r2"].astype(float)))
    vals = []
    for feat in morph_columns:
        r2 = r2_map.get(feat, 0.0)
        vals.append(1.0 - max(-0.5, min(0.95, r2)))
    w = np.array(vals, dtype=np.float32)
    w = w / np.mean(w)
    return w

# Dataset
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


# Model
class ResidualPCVAE(nn.Module):
    """
    x_hat = x_base + scale(x_cond)

    scale mode:
      none:       x_hat = x_base + x_cond
      scalar:     x_hat = x_base + tau * x_cond
      featurewise:x_hat = x_base + tau_vec ⊙ x_cond
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        cond_dim: int,
        shared_basis: np.ndarray,   # [D, k]
        encoder_hidden_dims: Tuple[int, ...],
        decoder_hidden_dims: Tuple[int, ...],
        cond_hidden_dims: Tuple[int, ...],
        residual_scale_mode: str = "none",
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.k_shared = shared_basis.shape[1]
        self.residual_scale_mode = residual_scale_mode

        basis = torch.tensor(shared_basis, dtype=torch.float32)
        self.register_buffer("shared_basis", basis)

        # encoder
        enc_layers = []
        prev = input_dim
        for h in encoder_hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # base decoder
        dec_layers = []
        prev = latent_dim
        for h in decoder_hidden_dims:
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.base_decoder = nn.Sequential(*dec_layers)

        # conditional branch
        if cond_dim > 0:
            cond_layers = []
            prev = cond_dim
            for h in cond_hidden_dims:
                cond_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            cond_layers.append(nn.Linear(prev, self.k_shared))
            self.cond_net = nn.Sequential(*cond_layers)
        else:
            self.cond_net = None

        # residual scaling parameterization
        # parameterize raw scale and pass through softplus for positivity
        if residual_scale_mode == "none":
            self.raw_tau = None
        elif residual_scale_mode == "scalar":
            init_raw = inverse_softplus(torch.tensor(residual_scale_init, dtype=torch.float32))
            self.raw_tau = nn.Parameter(init_raw.view(1))
        elif residual_scale_mode == "featurewise":
            init_raw = inverse_softplus(torch.full((input_dim,), residual_scale_init, dtype=torch.float32))
            self.raw_tau = nn.Parameter(init_raw)
        else:
            raise ValueError(f"Unknown residual_scale_mode: {residual_scale_mode}")

    def get_tau(self):
        if self.raw_tau is None:
            return None
        return F.softplus(self.raw_tau)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_base(self, z: torch.Tensor):
        return self.base_decoder(z)

    def decode_conditional(self, c: Optional[torch.Tensor]):
        if self.cond_net is None or c is None or c.numel() == 0:
            batch = 1 if c is None else c.shape[0]
            device = self.shared_basis.device
            alpha = torch.zeros((batch, self.k_shared), device=device)
            x_cond = torch.zeros((batch, self.input_dim), device=device)
            return alpha, x_cond

        alpha = self.cond_net(c)
        x_cond = alpha @ self.shared_basis.T
        return alpha, x_cond

    def apply_residual_scale(self, x_cond: torch.Tensor):
        tau = self.get_tau()
        if tau is None:
            return x_cond
        if self.residual_scale_mode == "scalar":
            return tau.view(1, 1) * x_cond
        if self.residual_scale_mode == "featurewise":
            return tau.view(1, -1) * x_cond
        return x_cond

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_base = self.decode_base(z)
        alpha, x_cond_raw = self.decode_conditional(c)
        x_cond = self.apply_residual_scale(x_cond_raw)
        x_hat = x_base + x_cond

        return {
            "x_hat": x_hat,
            "x_base": x_base,
            "x_cond": x_cond,
            "x_cond_raw": x_cond_raw,
            "alpha": alpha,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "tau": self.get_tau(),
        }


def inverse_softplus(x: torch.Tensor):
    return torch.log(torch.expm1(x))


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def weighted_mse(x_hat: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    diff2 = (x_hat - x) ** 2
    return torch.mean(diff2 * w.view(1, -1))


# PCA shared basis
def fit_shared_basis_from_train(
    X_train: np.ndarray,
    shared_pc_indices: Tuple[int, ...],
    random_state: int = 0,
):
    pca = PCA(n_components=min(X_train.shape[1], X_train.shape[0] - 1), random_state=random_state)
    pca.fit(X_train)

    comps = pca.components_
    valid_idx = [i for i in shared_pc_indices if 0 <= i < comps.shape[0]]
    if len(valid_idx) == 0:
        raise ValueError("No valid shared_pc_indices.")

    U_k = comps[valid_idx].T.astype(np.float32)
    pc_names = [f"PC{i+1}" for i in valid_idx]
    return pca, U_k, pc_names

# Training / eval
def train_one_epoch(
    model: ResidualPCVAE,
    loader: DataLoader,
    optimizer,
    beta: float,
    device: str,
    feature_weights: torch.Tensor,
    lambda_alpha_l2: float,
    lambda_residual_scale: float,
    freeze_conditional: bool = False,
):
    model.train()
    total_loss = total_rec = total_kl = total_alpha = total_resid = 0.0
    n = 0

    for x, c in loader:
        x = x.to(device)
        c = c.to(device) if c.numel() > 0 else None

        optimizer.zero_grad()
        out = model(x, c)

        rec = weighted_mse(out["x_hat"], x, feature_weights)
        kl = kl_divergence(out["mu"], out["logvar"])
        alpha_pen = torch.mean(out["alpha"] ** 2) if out["alpha"].numel() > 0 else torch.tensor(0.0, device=device)
        resid_pen = torch.mean(out["x_cond"] ** 2)

        loss = rec + beta * kl + lambda_alpha_l2 * alpha_pen + lambda_residual_scale * resid_pen
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rec += rec.item() * bs
        total_kl += kl.item() * bs
        total_alpha += alpha_pen.item() * bs
        total_resid += resid_pen.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "rec": total_rec / n,
        "kl": total_kl / n,
        "alpha_pen": total_alpha / n,
        "resid_pen": total_resid / n,
    }


@torch.no_grad()
def eval_one_epoch(
    model: ResidualPCVAE,
    loader: DataLoader,
    beta: float,
    device: str,
    feature_weights: torch.Tensor,
    lambda_alpha_l2: float,
    lambda_residual_scale: float,
):
    model.eval()
    total_loss = total_rec = total_kl = total_alpha = total_resid = 0.0
    n = 0

    for x, c in loader:
        x = x.to(device)
        c = c.to(device) if c.numel() > 0 else None

        out = model(x, c)

        rec = weighted_mse(out["x_hat"], x, feature_weights)
        kl = kl_divergence(out["mu"], out["logvar"])
        alpha_pen = torch.mean(out["alpha"] ** 2) if out["alpha"].numel() > 0 else torch.tensor(0.0, device=device)
        resid_pen = torch.mean(out["x_cond"] ** 2)

        loss = rec + beta * kl + lambda_alpha_l2 * alpha_pen + lambda_residual_scale * resid_pen

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_rec += rec.item() * bs
        total_kl += kl.item() * bs
        total_alpha += alpha_pen.item() * bs
        total_resid += resid_pen.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "rec": total_rec / n,
        "kl": total_kl / n,
        "alpha_pen": total_alpha / n,
        "resid_pen": total_resid / n,
    }


def set_warm_start_mode(model: ResidualPCVAE, warm_start_on: bool):
    """
    During warm start:
      - train encoder + base decoder only
      - freeze cond net and tau
    """
    cond_modules = []
    if model.cond_net is not None:
        cond_modules.append(model.cond_net)
    if model.raw_tau is not None:
        model.raw_tau.requires_grad = not warm_start_on

    for mod in cond_modules:
        for p in mod.parameters():
            p.requires_grad = not warm_start_on


@torch.no_grad()
def predict_all(model: ResidualPCVAE, X: np.ndarray, C: Optional[np.ndarray], device: str):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    c = None if C is None else torch.tensor(C, dtype=torch.float32, device=device)
    out = model(x, c)
    tau = out["tau"]
    tau_np = None if tau is None else tau.detach().cpu().numpy()
    return {
        "x_hat": out["x_hat"].cpu().numpy(),
        "x_base": out["x_base"].cpu().numpy(),
        "x_cond": out["x_cond"].cpu().numpy(),
        "x_cond_raw": out["x_cond_raw"].cpu().numpy(),
        "alpha": out["alpha"].cpu().numpy(),
        "mu": out["mu"].cpu().numpy(),
        "logvar": out["logvar"].cpu().numpy(),
        "tau": tau_np,
    }


# ============================================================
# Metrics / plots
# ============================================================

def compute_feature_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str]):
    rows = []
    for j, feat in enumerate(feature_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append({
            "feature": feat,
            "r2": float(r2_score(yt, yp)),
            "mse": float(np.mean((yt - yp) ** 2)),
            "pearson": float(np.corrcoef(yt, yp)[0, 1]) if (np.std(yt) > 1e-12 and np.std(yp) > 1e-12) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_history(history_df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.plot(history_df["epoch"], history_df["train_rec"], label="train_rec", linestyle="--")
    plt.plot(history_df["epoch"], history_df["val_rec"], label="val_rec", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_alpha_hist(alpha: np.ndarray, pc_names: List[str], outpath: Path):
    k = alpha.shape[1]
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 3.5))
    if k == 1:
        axes = [axes]
    for i in range(k):
        axes[i].hist(alpha[:, i], bins=60, alpha=0.85)
        axes[i].set_title(f"alpha {pc_names[i]}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# ============================================================
# Single run
# ============================================================

def run_single_experiment(
    base_cfg: TrainConfig,
    shared_pc_indices: Tuple[int, ...],
    warm_start_epochs: int,
    residual_scale_mode: str,
    residual_scale_init: float,
    run_root: Path,
    X_all: np.ndarray,
    C_all: Optional[np.ndarray],
    meta_index: pd.Index,
    morph_columns: List[str],
):
    cfg = copy.deepcopy(base_cfg)
    set_seed(cfg.random_state)

    idx = np.arange(X_all.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    C_train = None if C_all is None else C_all[train_idx]
    C_test = None if C_all is None else C_all[test_idx]

    pca_train, U_k, pc_names = fit_shared_basis_from_train(
        X_train, shared_pc_indices, random_state=cfg.random_state
    )

    feature_weights_np = build_feature_weights(cfg, morph_columns)
    feature_weights = torch.tensor(feature_weights_np, dtype=torch.float32, device=cfg.device)

    train_ds = MorphologyDataset(X_train, C_train)
    test_ds = MorphologyDataset(X_test, C_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    cond_dim = 0 if C_train is None else C_train.shape[1]
    model = ResidualPCVAE(
        input_dim=X_train.shape[1],
        latent_dim=cfg.latent_dim,
        cond_dim=cond_dim,
        shared_basis=U_k,
        encoder_hidden_dims=cfg.encoder_hidden_dims,
        decoder_hidden_dims=cfg.decoder_hidden_dims,
        cond_hidden_dims=cfg.cond_hidden_dims,
        residual_scale_mode=residual_scale_mode,
        residual_scale_init=residual_scale_init,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_name = (
        f"pcs_{'_'.join(str(i+1) for i in shared_pc_indices)}"
        f"__warm_{warm_start_epochs}"
        f"__scale_{residual_scale_mode}"
    )
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== RUN {run_name} ===\n"
        f"cond_mode={cfg.condition_mode} | cond_dim={cond_dim} | latent_dim={cfg.latent_dim} | "
        f"shared_pcs={pc_names}"
    )

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        warm_active = epoch <= warm_start_epochs
        set_warm_start_mode(model, warm_active)

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            beta=cfg.beta,
            device=cfg.device,
            feature_weights=feature_weights,
            lambda_alpha_l2=cfg.lambda_alpha_l2,
            lambda_residual_scale=cfg.lambda_residual_scale,
        )

        va = eval_one_epoch(
            model=model,
            loader=test_loader,
            beta=cfg.beta,
            device=cfg.device,
            feature_weights=feature_weights,
            lambda_alpha_l2=cfg.lambda_alpha_l2,
            lambda_residual_scale=cfg.lambda_residual_scale,
        )

        row = {
            "epoch": epoch,
            "warm_start_active": int(warm_active),
            "train_loss": tr["loss"],
            "train_rec": tr["rec"],
            "train_kl": tr["kl"],
            "train_alpha_pen": tr["alpha_pen"],
            "train_resid_pen": tr["resid_pen"],
            "val_loss": va["loss"],
            "val_rec": va["rec"],
            "val_kl": va["kl"],
            "val_alpha_pen": va["alpha_pen"],
            "val_resid_pen": va["resid_pen"],
        }
        history.append(row)

        if va["loss"] < best_val:
            best_val = va["loss"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch == warm_start_epochs or epoch % 50 == 0 or epoch == cfg.epochs:
            tau = model.get_tau()
            tau_summary = "none"
            if tau is not None:
                tau_cpu = tau.detach().cpu().numpy()
                tau_summary = (
                    f"tau_mean={float(np.mean(tau_cpu)):.4f}, "
                    f"tau_min={float(np.min(tau_cpu)):.4f}, "
                    f"tau_max={float(np.max(tau_cpu)):.4f}"
                )
            print(
                f"Epoch {epoch:04d} | warm={warm_active} | "
                f"train loss {tr['loss']:.4f} rec {tr['rec']:.4f} kl {tr['kl']:.4f} | "
                f"val loss {va['loss']:.4f} rec {va['rec']:.4f} kl {va['kl']:.4f} | "
                f"{tau_summary}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_all = predict_all(model, X_all, C_all, cfg.device)
    pred_test = predict_all(model, X_test, C_test, cfg.device)

    feat_full = compute_feature_metrics(X_test, pred_test["x_hat"], morph_columns)
    feat_base = compute_feature_metrics(X_test, pred_test["x_base"], morph_columns)

    merged = feat_full.merge(
        feat_base[["feature", "r2", "mse", "pearson"]],
        on="feature",
        suffixes=("_full", "_base")
    )
    merged["delta_r2_conditional"] = merged["r2_full"] - merged["r2_base"]

    history_df = pd.DataFrame(history)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": asdict(cfg),
            "shared_pc_indices_zero_based": list(shared_pc_indices),
            "shared_pc_names": pc_names,
            "shared_basis": U_k,
            "pca_explained_variance_ratio": pca_train.explained_variance_ratio_.tolist(),
            "feature_weights": feature_weights_np.tolist(),
            "warm_start_epochs": warm_start_epochs,
            "residual_scale_mode": residual_scale_mode,
            "residual_scale_init": residual_scale_init,
        },
        run_dir / "model.pt",
    )

    pd.DataFrame(pred_all["mu"], index=meta_index, columns=[f"latent_mu_{i}" for i in range(cfg.latent_dim)]).to_csv(
        run_dir / "latent_mu.csv"
    )
    pd.DataFrame(pred_all["alpha"], index=meta_index, columns=[f"alpha_{name}" for name in pc_names]).to_csv(
        run_dir / "conditional_alpha.csv"
    )
    pd.DataFrame(pred_all["x_hat"], index=meta_index, columns=morph_columns).to_csv(
        run_dir / "reconstruction_full.csv"
    )
    pd.DataFrame(pred_all["x_base"], index=meta_index, columns=morph_columns).to_csv(
        run_dir / "reconstruction_base.csv"
    )
    pd.DataFrame(pred_all["x_cond"], index=meta_index, columns=morph_columns).to_csv(
        run_dir / "reconstruction_conditional_component.csv"
    )

    if pred_all["tau"] is not None:
        tau_arr = pred_all["tau"]
        if tau_arr.ndim == 0:
            tau_df = pd.DataFrame({"tau": [float(tau_arr)]})
        elif tau_arr.size == 1:
            tau_df = pd.DataFrame({"tau": [float(tau_arr.reshape(-1)[0])]})
        else:
            tau_df = pd.DataFrame({"feature": morph_columns, "tau": tau_arr.reshape(-1)})
        tau_df.to_csv(run_dir / "tau.csv", index=False)

    feat_full.to_csv(run_dir / "test_feature_metrics_full.csv", index=False)
    feat_base.to_csv(run_dir / "test_feature_metrics_base.csv", index=False)
    merged.to_csv(run_dir / "test_feature_metrics_delta.csv", index=False)

    plot_alpha_hist(pred_all["alpha"], pc_names, run_dir / "alpha_hist.png")

    tau = model.get_tau()
    tau_mean = tau_min = tau_max = None
    if tau is not None:
        tau_np = tau.detach().cpu().numpy().reshape(-1)
        tau_mean = float(np.mean(tau_np))
        tau_min = float(np.min(tau_np))
        tau_max = float(np.max(tau_np))

    summary = {
        "condition_mode": cfg.condition_mode,
        "latent_dim": cfg.latent_dim,
        "shared_pc_indices_zero_based": list(shared_pc_indices),
        "shared_pc_names": pc_names,
        "shared_pc_explained_variance_ratio": [
            float(pca_train.explained_variance_ratio_[i]) for i in shared_pc_indices
        ],
        "n_cells": int(len(meta_index)),
        "n_morph_features": int(len(morph_columns)),
        "cond_dim": int(cond_dim),
        "warm_start_epochs": int(warm_start_epochs),
        "residual_scale_mode": residual_scale_mode,
        "residual_scale_init": float(residual_scale_init),
        "best_val_loss": float(best_val),
        "test_mean_r2_full": float(feat_full["r2"].mean()),
        "test_mean_r2_base": float(feat_base["r2"].mean()),
        "test_mean_delta_r2_conditional": float(merged["delta_r2_conditional"].mean()),
        "test_median_delta_r2_conditional": float(merged["delta_r2_conditional"].median()),
        "tau_mean": tau_mean,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "num_features_improved": int((merged["delta_r2_conditional"] > 0).sum()),
        "num_features_hurt": int((merged["delta_r2_conditional"] < 0).sum()),
    }

    with open(run_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# Main
def main():
    base_cfg = TrainConfig(
        condition_mode="gs",
        latent_dim=8,
        beta=1e-3,
        lr=1e-3,
        batch_size=256,
        epochs=1000,
        encoder_hidden_dims=(256, 128, 64),
        decoder_hidden_dims=(256, 128, 64),
        cond_hidden_dims=(128, 64),
        lambda_alpha_l2=1e-4,
        lambda_residual_scale=0.0,
        use_feature_weights=False,
        predictability_csv="results_morph_diagnostics/feature_predictability_a.csv",
    )

    ab_cfg = AblationConfig()

    set_seed(base_cfg.random_state)

    meta, morph, expr, zprof = load_patchseq_tables(base_cfg)

    gene_pcs = make_gene_condition(expr, pca_dim=base_cfg.gene_pca_dim)
    ensure_leiden_label(meta, gene_pcs, random_state=base_cfg.random_state)

    condition = build_condition_matrix(
        meta=meta,
        expr=expr,
        zprof=zprof,
        mode=base_cfg.condition_mode,
        gene_pca_dim=base_cfg.gene_pca_dim,
    )

    X_all = morph.to_numpy(dtype=np.float32)
    C_all = condition

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("results_residual_pc_vae_ablation") / f"{base_cfg.condition_mode}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    with open(run_root / "base_train_config.json", "w") as f:
        json.dump(asdict(base_cfg), f, indent=2)
    with open(run_root / "ablation_config.json", "w") as f:
        json.dump(asdict(ab_cfg), f, indent=2)

    combos = list(itertools.product(
        ab_cfg.pc_sets,
        ab_cfg.warm_start_epochs_list,
        ab_cfg.residual_scale_modes,
    ))

    if ab_cfg.max_runs is not None:
        combos = combos[:ab_cfg.max_runs]

    all_rows = []
    for shared_pc_indices, warm_start_epochs, residual_scale_mode in combos:
        summary = run_single_experiment(
            base_cfg=base_cfg,
            shared_pc_indices=shared_pc_indices,
            warm_start_epochs=warm_start_epochs,
            residual_scale_mode=residual_scale_mode,
            residual_scale_init=ab_cfg.residual_scale_init,
            run_root=run_root,
            X_all=X_all,
            C_all=C_all,
            meta_index=meta.index,
            morph_columns=morph.columns.tolist(),
        )
        all_rows.append(summary)
        pd.DataFrame(all_rows).sort_values(
            ["test_mean_r2_full", "test_mean_delta_r2_conditional", "best_val_loss"],
            ascending=[False, False, True]
        ).to_csv(run_root / "ablation_summary_running.csv", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df = summary_df.sort_values(
        ["test_mean_r2_full", "test_mean_delta_r2_conditional", "best_val_loss"],
        ascending=[False, False, True]
    )
    summary_df.to_csv(run_root / "ablation_summary.csv", index=False)

    print("\nFinished all ablations.")
    print(f"Saved to: {run_root.resolve()}")
    print(summary_df[[
        "shared_pc_indices_zero_based",
        "warm_start_epochs",
        "residual_scale_mode",
        "best_val_loss",
        "test_mean_r2_full",
        "test_mean_r2_base",
        "test_mean_delta_r2_conditional",
        "num_features_improved",
        "num_features_hurt",
        "tau_mean",
    ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()