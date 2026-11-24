#!/usr/bin/env python3
"""
run_reviewer_ablations.py
=========================
Standalone, reproducible reviewer-ablation script for the Time-Causal
Graph-Tabular BGNN paper.

Usage (from the project root where data/, Scripts/experiments/ live):
    python run_reviewer_ablations.py

Outputs (written next to this script unless --output-dir is passed):
    reviewer_ablations_and_seeds.json
    reviewer_ablations_and_seeds.csv

What it runs
------------
  Ablation A : Degree-Aware GNN aggregation (W4)
      Three variants of Stage-3 Res-GNN neighbour pooling:
        A1 - log1p_scaled  : mean(Z_j) weighted by log(1+degree_i)
        A2 - sum           : sum(Z_j)  (count-aware, no normalisation)
        A3 - concat_logdeg : original mean-pool PLUS log(1+degree) appended
                             as extra feature channels
      All three are trained with the original Stage-3 protocol (SELU,
      1-hop, K_card=10, K_dev=5, 30-day lookback, focal loss, isotonic
      calibration, N_SEEDS=3, expanding Pass A + frozen Pass B).

  Ablation B : 30-day windowed Stage-2 relational features (W4)
      Rebuilds Stage-2's 19 relational features using the same 30-day
      sliding window (2 592 000 s) that the graph uses, trains a new
      LightGBM model (identical hyper-params to Stage 2), and evaluates
      on sealed Block 5.

  Multi-seed stability (W6)
      Trains Stage-3 (real graph variant), Stage-3n (no-graph ablation),
      Stage-4 (joint BGNN) and Stage-4n (no-boost ablation) each across
      seeds = [42, 102, 202, 302, 402] and reports per-seed and
      mean ± std for AP, Cal-AP, ROC-AUC, Queue-Loss (1 % cap), and
      Corrected Bayes Loss.

Protocol constraints (identical to sealed evaluation)
    - 5-block equal-row chronological partition (Blocks 1-4 development,
      Block 5 sealed holdout)
    - 1-day (86 400 s) embargo between training window and evaluated rows
    - Causality assertions before every training call
    - Block 5 target column is never read during training or scoring
    - All development metrics evaluated on the embargoed validation rows
    - Block 5 metrics evaluated on the embargoed 115 111-row subset only
"""

# =============================================================================
# 0. Imports & Argument Parsing
# =============================================================================
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm is required.  pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit("PyTorch is required.  pip install torch")

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("scikit-learn is required.  pip install scikit-learn")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Reviewer Ablations for BGNN Paper")
parser.add_argument(
    "--data-dir",
    default=None,
    help="Directory containing train_transaction.csv and train_identity.csv. "
         "Searched automatically if omitted.",
)
parser.add_argument(
    "--output-dir",
    default=".",
    help="Directory to write reviewer_ablations_and_seeds.{json,csv}.",
)
parser.add_argument(
    "--skip-ablation-a",
    action="store_true",
    help="Skip Ablation A (degree-aware GNN).",
)
parser.add_argument(
    "--skip-ablation-b",
    action="store_true",
    help="Skip Ablation B (30-day windowed Stage-2).",
)
parser.add_argument(
    "--skip-seed-sweep",
    action="store_true",
    help="Skip multi-seed neural stability sweep.",
)
parser.add_argument(
    "--n-bootstrap",
    type=int,
    default=400,
    help="Bootstrap resamples for confidence intervals (default 400).",
)
ARGS = parser.parse_args()

# =============================================================================
# 1. Configuration (sealed — matches Stages 1–4 v2 exactly)
# =============================================================================
SEED = 42
N_SEEDS_ORIG = 3        # Stage 3 original
N_SEEDS_S4 = 2          # Stage 4 original
STABILITY_SEEDS = [42, 102, 202, 302, 402]

TIME_COL = "TransactionDT"
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
AMT_COL = "TransactionAmt"

N_BLOCKS = 5
DEV_BLOCKS = [1, 2, 3, 4]
OOT_BLOCK = 5
SECONDS_PER_DAY = 86_400
EMBARGO_SECONDS = 86_400
LOOKBACK_WINDOW_SECS = 30 * SECONDS_PER_DAY   # 2 592 000 s

# Neural hyper-params (Stage 3 baseline)
HIDDEN_DIM = 64
DROPOUT = 0.20
LR = 3e-3
WEIGHT_DECAY = 1e-4
N_PCA = 15
CARD_MAX_NEIGHBORS = 10
DEVICE_MAX_NEIGHBORS = 5
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
MAX_EPOCHS = 20
PATIENCE_S3 = 4
INNER_VAL_FRAC = 0.20
MIN_STOP_ROWS = 5_000
MIN_STOP_POSITIVES = 50
BATCH_SIZE = 4096
B1_SUBFOLDS = 4

# BGNN hyper-params (Stage 4)
OUTER_ROUNDS = 4
INNER_EPOCHS = 2
PATIENCE_S4 = 2
BOOSTING_LR = 0.10
DISPLACEMENT_CLIP = 0.20

# Cost model — matches Stage-3/4 compute_metrics()
C_REVIEW = 10.0
C_FRICTION = 25.0

# Corrected Bayes Loss cost model (W5 audit)
C_REVIEW_CORR = 5.0
C_FRICTION_CORR = 15.0

INVESTIGATION_CAP = 0.01   # 1 % queue capacity
N_BOOTSTRAP = ARGS.n_bootstrap

LGB_PARAMS_S2 = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "max_depth": -1,
    "min_child_samples": 50,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.7,
    "seed": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "data_random_seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEAKAGE_COLUMNS = [
    "isFraud", "TransactionID", "TransactionDT",
    "elapsed_days", "day_index", "chrono_block",
]
AUDIT_ONLY_COLUMNS = [
    "uid_card", "uid_device", "card_first_seen", "device_first_seen",
    "stage1_prob_raw", "stage1_prob_calibrated", "stage1_prob_filled",
    "stage1_available", "stage1_missing", "stage1_source", "stage1_oof_prob",
]

CANDIDATE_DATA_DIRS = [
    "/kaggle/input/competitions/ieee-fraud-detection",
    "/kaggle/input/ieee-fraud-detection",
    "./data/ieee-fraud-detection",
    "./data",
    ".",
]
CANDIDATE_DECISION_PATHS = [
    "/kaggle/input/notebooks/iemanihsan/ieee-cis-eda-leakage-audit/artifacts/feature_risk_table.csv",
    "Scripts/experiments/EDA/artifacts/feature_risk_table.csv",
    "./artifacts/feature_risk_table.csv",
    "feature_risk_table.csv",
]
CANDIDATE_STAGE1_OOF = [
    "/kaggle/input/notebooks/iemanihsan/fork-of-stage-1-standalone-tabular-lightgbm-basel/artifacts/stage1_oof_predictions.csv",
    "/kaggle/input/notebooks/iemanihsan/stage-1-standalone-tabular-lightgbm-baseline/artifacts/stage1_oof_predictions.csv",
    "Scripts/experiments/Stage 1/artifacts/stage1_oof_predictions.csv",
    "./artifacts/stage1_oof_predictions.csv",
    "stage1_oof_predictions.csv",
]

# =============================================================================
# 2. Seeding & Device Utilities
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(SEED)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


# =============================================================================
# 3. Data Loading & Temporal Structuring
# =============================================================================

def resolve_first_existing(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exist:\n  " + "\n  ".join(candidates))


def load_data(data_dir: str | None) -> pd.DataFrame:
    if data_dir is None:
        for d in CANDIDATE_DATA_DIRS:
            if os.path.exists(os.path.join(d, "train_transaction.csv")):
                data_dir = d
                break
        if data_dir is None:
            raise FileNotFoundError(
                "train_transaction.csv not found. Pass --data-dir or place data "
                "under data/ieee-fraud-detection/"
            )

    print(f"[data] Loading from {data_dir}")
    trans = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    id_path = os.path.join(data_dir, "train_identity.csv")
    if os.path.exists(id_path):
        ident = pd.read_csv(id_path)
        df = trans.merge(ident, on=ID_COL, how="left")
        del ident
    else:
        warnings.warn("train_identity.csv not found — identity features absent.")
        df = trans.copy()
    del trans
    gc.collect()

    assert df[ID_COL].is_unique, "Duplicate TransactionIDs after merge"
    assert df[TARGET_COL].notna().all(), "Null isFraud values"

    df = df.sort_values([TIME_COL, ID_COL], kind="mergesort").reset_index(drop=True)
    assert df[TIME_COL].is_monotonic_increasing, "Frame not chronologically sorted"

    t0 = int(df[TIME_COL].min())
    df["elapsed_days"] = ((df[TIME_COL] - t0) / SECONDS_PER_DAY).astype(np.float32)
    df["day_index"] = ((df[TIME_COL] - t0) // SECONDS_PER_DAY).astype(np.int16)

    n = len(df)
    block_arr = (np.minimum(np.arange(n) * N_BLOCKS // n, N_BLOCKS - 1) + 1).astype(np.int16)
    df["chrono_block"] = block_arr

    print(f"[data] {len(df):,} rows | blocks: {df.groupby('chrono_block').size().to_dict()}")
    return df


def build_entity_uids(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    anchor = np.where(
        frame["D1"].isna(),
        "na",
        (frame["day_index"] - frame["D1"].fillna(0)).astype("int64").astype(str),
    )

    def col_str(name: str, fill) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(["na"] * len(frame), index=frame.index)
        return frame[name].fillna(fill).astype(str)

    uid_card = (
        col_str("card1", -1) + "_" + col_str("card2", -1) + "_"
        + col_str("card3", -1) + "_" + col_str("card4", "na") + "_"
        + col_str("card5", -1) + "_" + col_str("card6", "na") + "_"
        + col_str("addr1", -1) + "_" + pd.Series(anchor, index=frame.index)
    )
    uid_device = (
        col_str("DeviceInfo", "no_dev") + "_"
        + col_str("id_30", "no_os") + "_"
        + col_str("id_31", "no_brw")
    )
    return uid_card, uid_device


def first_seen_flag(uid_series: pd.Series) -> np.ndarray:
    seen: set = set()
    flags = np.zeros(len(uid_series), dtype=bool)
    for i, v in enumerate(uid_series):
        if v not in seen:
            flags[i] = True
            seen.add(v)
    return flags


# =============================================================================
# 4. Causality Assertions
# =============================================================================

def assert_causal(df: pd.DataFrame, train_idx, score_idx, label: str) -> None:
    train_max = df.loc[train_idx, TIME_COL].max()
    score_min = df.loc[score_idx, TIME_COL].min()
    assert train_max <= score_min, (
        f"CAUSALITY VIOLATION [{label}]: train_max={train_max} > score_min={score_min}"
    )


# =============================================================================
# 5. Graph Construction (strict causal predecessor extraction)
# =============================================================================

def extract_causal_predecessors(
    transaction_dt: np.ndarray,
    entity_series: np.ndarray,
    max_neighbors: int = 10,
    lookback: int = LOOKBACK_WINDOW_SECS,
) -> tuple[list[list[int]], dict]:
    histories: dict = {}
    predecessors: list[list[int]] = []
    n_tie_skipped = 0

    for idx in range(len(transaction_dt)):
        dt = transaction_dt[idx]
        ent = entity_series[idx]
        h = histories.setdefault(ent, deque())

        while h and (dt - h[0][0] > lookback):
            h.popleft()

        selected: list[int] = []
        for t_prev, j in reversed(h):
            if t_prev >= dt:        # strict time: skip same-second edges
                n_tie_skipped += 1
                continue
            selected.append(j)
            if len(selected) >= max_neighbors:
                break
        selected.reverse()

        predecessors.append(selected)
        h.append((dt, idx))        # insert AFTER reading

    stats = {
        "tie_edges_skipped": int(n_tie_skipped),
        "total_edges": int(sum(len(p) for p in predecessors)),
    }
    return predecessors, stats


# =============================================================================
# 6. Neighbour Aggregation — baseline (mean) and degree-aware variants
# =============================================================================

def aggregate_mean(Z: np.ndarray, hist: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Standard unweighted mean pooling (original Stage-3 baseline)."""
    n, d = len(hist), Z.shape[1]
    agg = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n, 1), dtype=np.float32)

    counts = np.fromiter((len(h) for h in hist), dtype=np.int64, count=n)
    total = int(counts.sum())
    if total == 0:
        return agg, mask

    src = np.empty(total, dtype=np.int64)
    pos = 0
    for h in hist:
        k = len(h)
        if k:
            src[pos:pos + k] = h
            pos += k

    offsets = np.concatenate(([0], np.cumsum(counts)))
    valid = counts > 0
    sums = np.add.reduceat(Z[src], offsets[:-1][valid], axis=0)
    agg[valid] = (sums / counts[valid, None]).astype(np.float32)
    mask[valid] = 1.0
    return agg, mask


def aggregate_sum(Z: np.ndarray, hist: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Sum aggregation (count-aware, no normalisation)."""
    n, d = len(hist), Z.shape[1]
    agg = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n, 1), dtype=np.float32)

    counts = np.fromiter((len(h) for h in hist), dtype=np.int64, count=n)
    total = int(counts.sum())
    if total == 0:
        return agg, mask

    src = np.empty(total, dtype=np.int64)
    pos = 0
    for h in hist:
        k = len(h)
        if k:
            src[pos:pos + k] = h
            pos += k

    offsets = np.concatenate(([0], np.cumsum(counts)))
    valid = counts > 0
    sums = np.add.reduceat(Z[src], offsets[:-1][valid], axis=0)
    agg[valid] = sums.astype(np.float32)
    mask[valid] = 1.0
    return agg, mask


def aggregate_log1p_scaled(Z: np.ndarray, hist: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean pooling weighted by log(1 + degree_i)."""
    n, d = len(hist), Z.shape[1]
    agg = np.zeros((n, d), dtype=np.float32)
    mask = np.zeros((n, 1), dtype=np.float32)

    counts = np.fromiter((len(h) for h in hist), dtype=np.int64, count=n)
    total = int(counts.sum())
    if total == 0:
        return agg, mask

    src = np.empty(total, dtype=np.int64)
    pos = 0
    for h in hist:
        k = len(h)
        if k:
            src[pos:pos + k] = h
            pos += k

    offsets = np.concatenate(([0], np.cumsum(counts)))
    valid = counts > 0
    sums = np.add.reduceat(Z[src], offsets[:-1][valid], axis=0)
    # Multiply normalised mean by log(1+degree)
    log_w = np.log1p(counts[valid].astype(np.float32)).reshape(-1, 1)
    agg[valid] = ((sums / counts[valid, None]) * log_w).astype(np.float32)
    mask[valid] = 1.0
    return agg, mask


def aggregate_concat_logdeg(
    Z: np.ndarray, hist: list[list[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool PLUS log(1+degree) appended as an extra channel.

    Returns (agg [n, d], logdeg [n, 1], mask [n, 1]).
    The caller concatenates logdeg into Z before passing to the model.
    """
    agg, mask = aggregate_mean(Z, hist)
    counts = np.fromiter((len(h) for h in hist), dtype=np.int64, count=len(hist))
    logdeg = np.log1p(counts.astype(np.float32)).reshape(-1, 1)
    return agg, logdeg, mask


def neighbour_means(
    Z: np.ndarray,
    card_hist: list[list[int]],
    dev_hist: list[list[int]],
    agg_mode: str = "mean",   # "mean" | "sum" | "log1p_scaled" | "concat_logdeg"
) -> tuple[np.ndarray, ...]:
    """Return (card_agg, card_mask, dev_agg, dev_mask) [+ logdeg channels]."""
    if agg_mode == "mean":
        ca, cm = aggregate_mean(Z, card_hist)
        da, dm = aggregate_mean(Z, dev_hist)
        return ca, cm, da, dm
    elif agg_mode == "sum":
        ca, cm = aggregate_sum(Z, card_hist)
        da, dm = aggregate_sum(Z, dev_hist)
        return ca, cm, da, dm
    elif agg_mode == "log1p_scaled":
        ca, cm = aggregate_log1p_scaled(Z, card_hist)
        da, dm = aggregate_log1p_scaled(Z, dev_hist)
        return ca, cm, da, dm
    elif agg_mode == "concat_logdeg":
        ca, cld, cm = aggregate_concat_logdeg(Z, card_hist)
        da, dld, dm = aggregate_concat_logdeg(Z, dev_hist)
        return ca, cm, da, dm, cld, dld   # extra channels passed as bonus
    else:
        raise ValueError(f"Unknown agg_mode '{agg_mode}'")


def blank_aggs(n: int, d: int, agg_mode: str = "mean") -> tuple[np.ndarray, ...]:
    z = np.zeros((n, d), dtype=np.float32)
    m = np.zeros((n, 1), dtype=np.float32)
    if agg_mode == "concat_logdeg":
        return z, m, z, m, np.zeros((n, 1), dtype=np.float32), np.zeros((n, 1), dtype=np.float32)
    return z, m, z, m


def slice_aggs(agg_tuple: tuple[np.ndarray, ...], lo: int, hi: int) -> tuple[np.ndarray, ...]:
    return tuple(a[lo:hi] for a in agg_tuple)


# =============================================================================
# 7. Neural Feature Pipeline
# =============================================================================
LOG_COLS = ["TransactionAmt", "dist1", "dist2", "C1", "C2", "C13"]


def build_neural_features(
    df_fit: pd.DataFrame,
    df_apply: pd.DataFrame,
    v_columns: list[str],
    num_columns: list[str],
    n_pca: int = N_PCA,
    extra_input_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned continuous node representation Z0.

    extra_input_dim: extra columns appended (e.g. log-degree channels in
    concat_logdeg variant). The returned arrays are shape [n, D+extra_input_dim].
    For non-concat_logdeg modes extra_input_dim=0 and nothing changes.
    """
    X_fit = df_fit[num_columns].copy()
    X_app = df_apply[num_columns].copy()

    for col in LOG_COLS:
        if col in num_columns:
            X_fit[col] = np.log1p(np.maximum(0.0, X_fit[col].fillna(0.0)))
            X_app[col] = np.log1p(np.maximum(0.0, X_app[col].fillna(0.0)))

    fill_values = X_fit.median().fillna(0.0)
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit.fillna(fill_values))
    X_app_s = scaler.transform(X_app.fillna(fill_values))

    v_scaler = StandardScaler()
    V_fit = v_scaler.fit_transform(df_fit[v_columns].fillna(0.0))
    V_app = v_scaler.transform(df_apply[v_columns].fillna(0.0))

    n_comp = int(min(n_pca, V_fit.shape[1], max(1, V_fit.shape[0] - 1)))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=SEED)
    V_fit_pca = pca.fit_transform(V_fit)
    V_app_pca = pca.transform(V_app)

    fit_prior = float(df_fit[TARGET_COL].mean())
    eps = 1e-6

    def to_logit(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = sub.get("stage1_prob_raw", pd.Series(np.nan, index=sub.index))
        avail = raw.notna().to_numpy().astype(np.float32)
        p = np.clip(raw.fillna(fit_prior).to_numpy(dtype=float), eps, 1 - eps)
        return np.log(p / (1 - p)).astype(np.float32), avail

    logit_fit, avail_fit = to_logit(df_fit)
    logit_app, avail_app = to_logit(df_apply)

    Z_fit = np.hstack([X_fit_s, V_fit_pca,
                       logit_fit.reshape(-1, 1), avail_fit.reshape(-1, 1)]).astype(np.float32)
    Z_app = np.hstack([X_app_s, V_app_pca,
                       logit_app.reshape(-1, 1), avail_app.reshape(-1, 1)]).astype(np.float32)

    assert np.isfinite(Z_fit).all(), "Non-finite values in Z_fit"
    assert np.isfinite(Z_app).all(), "Non-finite values in Z_app"
    return Z_fit, Z_app


def rows_positional_split(n: int, dt: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Positional 80/20 time-ordered inner split."""
    n_fit = max(1, int(n * (1 - frac)))
    fit_m = np.zeros(n, dtype=bool)
    stop_m = np.zeros(n, dtype=bool)
    fit_m[:n_fit] = True
    stop_m[n_fit:] = True
    assert dt[fit_m].max() <= dt[stop_m].min(), "Inner split is not time-ordered."
    return fit_m, stop_m


# =============================================================================
# 8. Metric Utilities
# =============================================================================

def bayes_threshold(amounts: np.ndarray, c_rev: float, c_fric: float) -> np.ndarray:
    return np.minimum((c_rev + c_fric) / (np.asarray(amounts, dtype=float) + c_fric), 1.0)


def queue_loss_at_cap(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    cap: float = INVESTIGATION_CAP,
    c_miss: float = 1.0,
    c_rev: float = 5.0,
    c_fric: float = 15.0,
) -> float:
    """Queue loss at a fixed capacity cap (H3c definition)."""
    n = len(y)
    n_alert = max(1, int(np.floor(cap * n)))
    order = np.argsort(-p)
    alerted = np.zeros(n, dtype=bool)
    alerted[order[:n_alert]] = True

    # Missed frauds cost c_miss * amount
    missed_frauds = (y == 1) & (~alerted)
    missed_cost = float((amounts[missed_frauds] * c_miss).sum())
    # Review cost per alert
    review_cost = float(n_alert * c_rev)
    # Friction: alerted non-frauds
    friction_cost = float(((~y.astype(bool)) & alerted).sum() * c_fric)
    return missed_cost + review_cost + friction_cost


def corrected_bayes_loss(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    c_rev: float = C_REVIEW_CORR,
    c_fric: float = C_FRICTION_CORR,
) -> dict[str, float]:
    """Corrected Bayes loss under τ* = 20/(A+15)."""
    tau = bayes_threshold(amounts, c_rev, c_fric)
    alerted = p > tau
    n_alert = int(alerted.sum())
    stolen = float(amounts[y.astype(bool) & ~alerted].sum())   # missed frauds
    rev = float(n_alert * c_rev)
    fric = float((~y.astype(bool) & alerted).sum() * c_fric)
    return {
        "n_alerts": n_alert,
        "alert_pct": float(n_alert / max(len(y), 1) * 100),
        "total_loss": stolen + rev + fric,
    }


def compute_metrics(y: np.ndarray, p: np.ndarray, amounts: np.ndarray) -> dict[str, float]:
    if y.sum() == 0:
        return {k: float("nan") for k in [
            "AP", "ROC_AUC", "queue_loss_1pct", "bayes_loss_corr", "n_alerts_bayes"
        ]}
    ap = float(average_precision_score(y, p))
    roc = float(roc_auc_score(y, p))
    ql = queue_loss_at_cap(y, p, amounts)
    bl = corrected_bayes_loss(y, p, amounts)
    return {
        "AP": ap,
        "ROC_AUC": roc,
        "queue_loss_1pct": ql,
        "bayes_loss_corr": bl["total_loss"],
        "n_alerts_bayes": bl["n_alerts"],
        "alert_pct_bayes": bl["alert_pct"],
    }


def bootstrap_ap(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    aps = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y[idx].sum() > 0:
            aps.append(average_precision_score(y[idx], p[idx]))
    if not aps:
        return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    aps_arr = np.array(aps)
    return {
        "mean": float(np.mean(aps_arr)),
        "std": float(np.std(aps_arr, ddof=1)),
        "ci_lo": float(np.percentile(aps_arr, 2.5)),
        "ci_hi": float(np.percentile(aps_arr, 97.5)),
    }


# =============================================================================
# 9. Neural Architectures
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * ((1 - p_t) ** self.gamma) * bce).mean()


class FrozenResGNN(nn.Module):
    """Stage-3 baseline architecture (1-hop SELU mean-pool)."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.proj_self = nn.Linear(in_dim, hidden_dim)
        self.proj_card = nn.Linear(in_dim, hidden_dim)
        self.proj_dev = nn.Linear(in_dim, hidden_dim)
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
        )
        self.classifier = nn.Linear(hidden_dim + in_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        card_h: torch.Tensor, card_mask: torch.Tensor,
        dev_h: torch.Tensor, dev_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_self = F.selu(self.proj_self(z))
        h_card = F.selu(self.proj_card(card_h)) * card_mask
        h_dev = F.selu(self.proj_dev(dev_h)) * dev_mask
        h_graph = self.agg(torch.cat([h_self, h_card, h_dev], dim=1))
        return self.classifier(torch.cat([h_graph, z], dim=1)).squeeze(1)


class FrozenResGNN_ConcatLogDeg(nn.Module):
    """Ablation A3: original mean-pool PLUS appended log-degree channels."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        # z has in_dim dims; card_h has in_dim dims; dev_h has in_dim dims
        # + 2 extra log-degree channels (card_logdeg, dev_logdeg) appended to z
        self.proj_self = nn.Linear(in_dim + 2, hidden_dim)   # +2 logdeg channels
        self.proj_card = nn.Linear(in_dim, hidden_dim)
        self.proj_dev = nn.Linear(in_dim, hidden_dim)
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
        )
        self.classifier = nn.Linear(hidden_dim + in_dim + 2, 1)

    def forward(
        self,
        z: torch.Tensor,       # [B, in_dim+2]  (last 2 dims are logdeg)
        card_h: torch.Tensor, card_mask: torch.Tensor,
        dev_h: torch.Tensor, dev_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_self = F.selu(self.proj_self(z))
        # card_h and dev_h are still in_dim (pre-logdeg original embeddings)
        h_card = F.selu(self.proj_card(card_h)) * card_mask
        h_dev = F.selu(self.proj_dev(dev_h)) * dev_mask
        h_graph = self.agg(torch.cat([h_self, h_card, h_dev], dim=1))
        return self.classifier(torch.cat([h_graph, z], dim=1)).squeeze(1)


class MonotonicGatedBGNN(nn.Module):
    """Stage-4 architecture with monotonic blending gate."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.proj_self = nn.Linear(in_dim, hidden_dim)
        self.proj_card = nn.Linear(in_dim, hidden_dim)
        self.proj_dev = nn.Linear(in_dim, hidden_dim)
        self.agg = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.SELU(),
        )
        self.gnn_head = nn.Linear(hidden_dim + in_dim, 1)
        self.w_card = nn.Parameter(torch.tensor([1.5]))
        self.w_dev = nn.Parameter(torch.tensor([1.5]))
        self.b_gate = nn.Parameter(torch.tensor([-0.5]))

    def forward(
        self,
        z: torch.Tensor,
        card_h: torch.Tensor, card_mask: torch.Tensor,
        dev_h: torch.Tensor, dev_mask: torch.Tensor,
        gbdt_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_self = F.selu(self.proj_self(z))
        h_card = F.selu(self.proj_card(card_h)) * card_mask
        h_dev = F.selu(self.proj_dev(dev_h)) * dev_mask
        h_graph = self.agg(torch.cat([h_self, h_card, h_dev], dim=1))
        gnn_logit = self.gnn_head(torch.cat([h_graph, z], dim=1)).squeeze(1)
        cm = card_mask.squeeze(1)
        dm = dev_mask.squeeze(1)
        gate = F.softplus(self.w_card) * cm + F.softplus(self.w_dev) * dm + self.b_gate
        active = ((cm + dm) > 0).float()
        beta = torch.sigmoid(gate) * active
        return beta * gnn_logit + (1.0 - beta) * gbdt_logit, beta


# =============================================================================
# 10. Stage-3 Training Helpers
# =============================================================================

@torch.no_grad()
def infer_s3(
    model: nn.Module,
    Z: np.ndarray,
    agg: tuple[np.ndarray, ...],
    batch_size: int = 32768,
) -> np.ndarray:
    model.eval()
    outputs = []
    for i in range(0, len(Z), batch_size):
        s = slice(i, i + batch_size)
        args = [torch.from_numpy(np.ascontiguousarray(Z[s])).to(DEVICE)]
        args += [torch.from_numpy(np.ascontiguousarray(a[s])).to(DEVICE) for a in agg]
        outputs.append(torch.sigmoid(model(*args)).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_one_model_s3(
    Z_fit: np.ndarray,
    agg_fit: tuple,
    y_fit: np.ndarray,
    Z_stop: np.ndarray,
    agg_stop: tuple,
    y_stop: np.ndarray,
    seed: int,
    agg_mode: str = "mean",
    label: str = "",
) -> tuple[nn.Module, int, float]:
    set_global_seed(seed)
    in_dim = Z_fit.shape[1]

    if agg_mode == "concat_logdeg":
        model = FrozenResGNN_ConcatLogDeg(in_dim, HIDDEN_DIM, DROPOUT).to(DEVICE)
    else:
        model = FrozenResGNN(in_dim, HIDDEN_DIM, DROPOUT).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)
    use_amp = DEVICE.type == "cuda"
    scaler_amp = make_grad_scaler(use_amp)

    gen = torch.Generator()
    gen.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(Z_fit)),
        *[torch.from_numpy(np.ascontiguousarray(a)) for a in agg_fit],
        torch.from_numpy(y_fit.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    best_ap, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in loader:
            *x_batch, b_y = batch
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                # Last tensor in x_batch is label; rest are (z, ca, cm, da, dm [, cld, dld])
                logits = model(*[t.to(DEVICE) for t in x_batch])
                loss = criterion(logits, b_y.to(DEVICE))
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        stop_probs = infer_s3(model, Z_stop, agg_stop)
        cur_ap = float(average_precision_score(y_stop, stop_probs))
        if cur_ap > best_ap:
            best_ap, best_epoch, stale = cur_ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE_S3:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def fit_calibrator(probs_stop: np.ndarray, y_stop: np.ndarray) -> IsotonicRegression | None:
    if len(y_stop) == 0 or y_stop.sum() < MIN_STOP_POSITIVES:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(probs_stop, y_stop)
    return iso


def apply_calibrator(cal, probs: np.ndarray) -> np.ndarray:
    if cal is None:
        return probs.copy()
    return np.clip(cal.predict(probs), 0.0, 1.0).astype(np.float32)


def train_multiseed_s3(
    Z_fit, agg_fit, y_fit,
    Z_stop, agg_stop, y_stop,
    Z_score, agg_score,
    n_seeds: int,
    base_seed: int = SEED,
    agg_mode: str = "mean",
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, IsotonicRegression | None]:
    score_stack, stop_stack = [], []
    for k in range(n_seeds):
        seed = base_seed + 1000 * k
        model, ep, ap = train_one_model_s3(
            Z_fit, agg_fit, y_fit, Z_stop, agg_stop, y_stop,
            seed=seed, agg_mode=agg_mode, label=f"{label}/s{k}"
        )
        score_stack.append(infer_s3(model, Z_score, agg_score))
        stop_stack.append(infer_s3(model, Z_stop, agg_stop))
        print(f"      [{label}/s{k}] epoch={ep} inner_AP={ap:.4f}")
        del model
        gc.collect()

    p_score = np.mean(np.vstack(score_stack), axis=0).astype(np.float32)
    p_stop = np.mean(np.vstack(stop_stack), axis=0).astype(np.float32)
    cal = fit_calibrator(p_stop, y_stop)
    return p_score, p_stop, cal


# =============================================================================
# 11. Stage-4 (BGNN) Training Helpers
# =============================================================================

@torch.no_grad()
def infer_s4(
    model: MonotonicGatedBGNN,
    Z: np.ndarray,
    agg: tuple[np.ndarray, ...],
    logit: np.ndarray,
    batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, betas = [], []
    for i in range(0, len(Z), batch_size):
        s = slice(i, i + batch_size)
        out, beta = model(
            torch.from_numpy(np.ascontiguousarray(Z[s])).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(agg[0][s])).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(agg[1][s])).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(agg[2][s])).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(agg[3][s])).to(DEVICE),
            torch.from_numpy(np.ascontiguousarray(logit[s])).to(DEVICE),
        )
        probs.append(torch.sigmoid(out).float().cpu().numpy())
        betas.append(beta.float().cpu().numpy())
    p = np.concatenate(probs) if probs else np.empty(0)
    b = np.concatenate(betas) if betas else np.empty(0)
    return p, b


def run_one_bgnn(
    Z_tr0: np.ndarray, Z_va0: np.ndarray,
    logit_tr: np.ndarray, logit_va: np.ndarray,
    y_fit: np.ndarray, y_stop: np.ndarray,
    n_fit: int, n_tr: int,
    card_hist: list[list[int]], dev_hist: list[list[int]],
    total_len: int,
    boosting_lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    set_global_seed(seed)
    Z_tr, Z_va = Z_tr0.copy(), Z_va0.copy()

    def compute_all_aggs():
        Z_stacked = np.vstack([Z_tr, Z_va])
        a_all = neighbour_means(Z_stacked, card_hist, dev_hist, agg_mode="mean")
        return (
            tuple(a[:n_fit] for a in a_all),
            tuple(a[n_fit:n_tr] for a in a_all),
            tuple(a[n_tr:total_len] for a in a_all),
        )

    agg_fit, agg_stop, agg_va = compute_all_aggs()
    model = MonotonicGatedBGNN(Z_tr.shape[1], HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)
    use_amp = DEVICE.type == "cuda"
    scaler_amp = make_grad_scaler(use_amp)

    best_ap, best_round, stale = -np.inf, 0, 0
    best_state = None
    best_Z_va, best_agg_va, best_logit_va = Z_va.copy(), agg_va, logit_va.copy()

    for m_round in range(1, OUTER_ROUNDS + 1):
        gen = torch.Generator(); gen.manual_seed(seed + m_round)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(Z_tr[:n_fit])),
                *[torch.from_numpy(np.ascontiguousarray(a)) for a in agg_fit],
                torch.from_numpy(np.ascontiguousarray(logit_tr[:n_fit])),
                torch.from_numpy(y_fit.astype(np.float32)),
            ),
            batch_size=BATCH_SIZE, shuffle=True, generator=gen, pin_memory=False,
        )
        model.train()
        for _ in range(INNER_EPOCHS):
            for b_z, b_ca, b_cm, b_da, b_dm, b_lg, b_y in loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(use_amp):
                    out, _ = model(b_z.to(DEVICE), b_ca.to(DEVICE), b_cm.to(DEVICE),
                                   b_da.to(DEVICE), b_dm.to(DEVICE), b_lg.to(DEVICE))
                    loss = criterion(out, b_y.to(DEVICE))
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

        if boosting_lr > 0:
            p_fit, _ = infer_s4(model, Z_tr[:n_fit], agg_fit, logit_tr[:n_fit])
            residual = (y_fit - p_fit).astype(np.float32)
            reg = lgb.LGBMRegressor(
                objective="regression", learning_rate=0.08, num_leaves=31,
                n_estimators=30, random_state=seed + m_round, n_jobs=-1, verbose=-1
            )
            reg.fit(Z_tr[:n_fit], residual)
            direction = (Z_tr[:n_fit] * residual.reshape(-1, 1)).mean(axis=0)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-8 else direction
            disp_tr = np.clip(reg.predict(Z_tr).reshape(-1, 1) * direction,
                              -DISPLACEMENT_CLIP, DISPLACEMENT_CLIP)
            disp_va = np.clip(reg.predict(Z_va).reshape(-1, 1) * direction,
                              -DISPLACEMENT_CLIP, DISPLACEMENT_CLIP)
            Z_tr = (Z_tr + boosting_lr * disp_tr).astype(np.float32)
            Z_va = (Z_va + boosting_lr * disp_va).astype(np.float32)
            agg_fit, agg_stop, agg_va = compute_all_aggs()

        p_stop, _ = infer_s4(model, Z_tr[n_fit:n_tr], agg_stop, logit_tr[n_fit:n_tr])
        cur_ap = float(average_precision_score(y_stop, p_stop))
        if cur_ap > best_ap:
            best_ap, best_round, stale = cur_ap, m_round, 0
            best_state = copy.deepcopy(model.state_dict())
            best_Z_va, best_agg_va, best_logit_va = Z_va.copy(), agg_va, logit_va.copy()
        else:
            stale += 1
            if stale >= PATIENCE_S4:
                break

    model.load_state_dict(best_state)
    p_va, _ = infer_s4(model, best_Z_va, best_agg_va, best_logit_va)
    p_stop_final, _ = infer_s4(model, Z_tr[n_fit:n_tr], agg_stop, logit_tr[n_fit:n_tr])
    del model; gc.collect()
    return p_va, p_stop_final, best_round, best_ap


def train_multiseed_bgnn(
    Z_tr0, Z_va0, logit_tr, logit_va,
    y_fit, y_stop, n_fit, n_tr,
    card_hist, dev_hist, total_len,
    boosting_lr: float,
    n_seeds: int,
    base_seed: int = SEED,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, IsotonicRegression | None]:
    p_va_list, p_stop_list = [], []
    for k in range(n_seeds):
        seed = base_seed + 1000 * k
        p_va, p_stop, rnd, ap = run_one_bgnn(
            Z_tr0, Z_va0, logit_tr, logit_va,
            y_fit, y_stop, n_fit, n_tr,
            card_hist, dev_hist, total_len,
            boosting_lr, seed,
        )
        p_va_list.append(p_va)
        p_stop_list.append(p_stop)
        print(f"      [{label}/s{k}] round={rnd} inner_AP={ap:.4f}")

    p_va_mean = np.mean(np.vstack(p_va_list), axis=0).astype(np.float32)
    p_stop_mean = np.mean(np.vstack(p_stop_list), axis=0).astype(np.float32)
    cal = fit_calibrator(p_stop_mean, y_stop)
    return p_va_mean, p_stop_mean, cal


# =============================================================================
# 12. Stage-2 Windowed Relational Feature Builder (Ablation B)
# =============================================================================

def build_windowed_relational_features(
    df_in: pd.DataFrame,
    card_col: str,
    device_col: str,
    lookback_secs: int = LOOKBACK_WINDOW_SECS,
) -> pd.DataFrame:
    """Stage-2 relational features with 30-day sliding window constraint.

    Identical feature set as Stage-2 (19 features) except that history
    aggregations are truncated to the lookback_secs window, mirroring the
    graph's time-causal predecessor extraction exactly.
    """
    out: dict[str, np.ndarray] = {}
    n = len(df_in)
    dt = df_in[TIME_COL].to_numpy()
    amt = df_in[AMT_COL].to_numpy(dtype=np.float32)

    for name, col in [("card", card_col), ("device", device_col)]:
        key = f"uid_{name}"
        entities = df_in[col].to_numpy()

        # Per-entity sliding history
        hist: dict[Any, deque] = {}
        seen_before = np.zeros(n, dtype=np.float32)
        dt_diff = np.full(n, -1.0, dtype=np.float32)
        running_sum_amt = np.zeros(n, dtype=np.float32)

        for i in range(n):
            ent = entities[i]
            t = dt[i]
            a = float(amt[i])

            h = hist.setdefault(ent, deque())
            # Expire entries outside window
            while h and (t - h[0][0] > lookback_secs):
                h.popleft()

            cnt = len(h)
            seen_before[i] = float(cnt)

            if cnt > 0:
                prev_t, prev_sum = h[-1][0], sum(e[2] for e in h)
                dt_diff[i] = float(t - prev_t)
                running_sum_amt[i] = float(prev_sum)
            else:
                dt_diff[i] = -1.0
                running_sum_amt[i] = 0.0

            h.append((t, i, a))

        with np.errstate(divide="ignore", invalid="ignore"):
            running_mean_amt = np.where(
                seen_before > 0, running_sum_amt / seen_before, amt
            ).astype(np.float32)

        out[f"{key}_seen_before"] = seen_before
        out[f"{key}_dt_diff"] = dt_diff
        out[f"{key}_amt_running_mean"] = running_mean_amt
        out[f"{key}_amt_ratio"] = (amt / (running_mean_amt + 1.0)).astype(np.float32)
        out[f"{key}_amt_diff"] = (amt - running_mean_amt).astype(np.float32)

    # Card-anchored C-series windowed running means
    card_entities = df_in[card_col].to_numpy()
    card_seen = out["uid_card_seen_before"]
    for c_col in ["C1", "C2", "C13", "C14"]:
        if c_col not in df_in.columns:
            continue
        c_vals = df_in[c_col].fillna(0.0).to_numpy(dtype=np.float32)
        hist_c: dict[Any, deque] = {}
        c_running_mean = np.zeros(n, dtype=np.float32)
        for i in range(n):
            ent = card_entities[i]
            t = dt[i]
            h = hist_c.setdefault(ent, deque())
            while h and (t - h[0][0] > lookback_secs):
                h.popleft()
            cnt = len(h)
            if cnt > 0:
                c_running_mean[i] = float(sum(e[1] for e in h)) / cnt
            h.append((t, c_vals[i]))
        out[f"uid_card_{c_col}_running_mean"] = c_running_mean

    # Expanding count features for categorical proxies (unchanged — these are
    # frequency statistics, not time-aggregations, so windowing doesn't apply)
    for cat in ["card1", "card2", "addr1", "P_emaildomain", "R_emaildomain"]:
        if cat in df_in.columns:
            out[f"{cat}_seen_before"] = (
                df_in.groupby(cat, sort=False, dropna=False)
                     .cumcount().to_numpy(dtype=np.float32)
            )

    return pd.DataFrame(out, index=df_in.index)


# =============================================================================
# 13. Load Shared Data & Feature Ledger
# =============================================================================

print("\n" + "=" * 80)
print("LOADING DATA AND FEATURE LEDGER")
print("=" * 80)

df = load_data(ARGS.data_dir)

# Entity UIDs
df["uid_card"], df["uid_device"] = build_entity_uids(df)
df["card_first_seen"] = first_seen_flag(df["uid_card"])
df["device_first_seen"] = first_seen_flag(df["uid_device"])

# Sealed OOT quarantine
dev_mask_arr = (df["chrono_block"] < OOT_BLOCK).to_numpy()
oot_mask_arr = (df["chrono_block"] == OOT_BLOCK).to_numpy()
assert df.loc[dev_mask_arr, TIME_COL].max() <= df.loc[oot_mask_arr, TIME_COL].min(), (
    "FATAL: OOT Block 5 overlaps development Blocks 1-4."
)
OOT_INDEX = set(df.index[oot_mask_arr].tolist())
print(f"[protocol] Block 5 OOT quarantine: PASSED ({oot_mask_arr.sum():,} rows sealed)")

# Feature ledger
decision_file = resolve_first_existing(CANDIDATE_DECISION_PATHS)
print(f"[features] Loaded decisions ledger: {decision_file}")
decisions_df = pd.read_csv(decision_file)
action_col = "action" if "action" in decisions_df.columns else "verdict"
drop_filter = decisions_df[action_col].astype(str).str.lower().str.startswith("drop")
accepted_features = decisions_df.loc[~drop_filter, "feature"].tolist()

base_features = [
    col for col in accepted_features
    if col in df.columns
    and col not in TARGET_LEAKAGE_COLUMNS
    and col not in AUDIT_ONLY_COLUMNS
]
v_cols = sorted(
    [c for c in base_features if c.startswith("V") and c[1:].isdigit()],
    key=lambda x: int(x[1:]),
)
num_cols = [c for c in base_features if c not in v_cols and df[c].dtype.kind in "if"]
print(f"[features] Continuous: {len(num_cols)} | V-columns: {len(v_cols)}")

# Stage-1 OOF (needed for two-stage Z representation)
stage1_oof_path = resolve_first_existing(CANDIDATE_STAGE1_OOF)
s1_oof = pd.read_csv(stage1_oof_path)
print(f"[stage1]   Loaded Stage-1 OOF: {stage1_oof_path} ({len(s1_oof):,} rows)")

prob_col = next(
    (c for c in ["stage1_prob_calibrated", "stage1_prob_raw", "stage1_prob_filled"]
     if c in s1_oof.columns), None
)
assert prob_col is not None, "No recognisable probability column in Stage-1 OOF"
df = df.merge(s1_oof[[ID_COL, prob_col]].rename(columns={prob_col: "stage1_prob_raw"}),
              on=ID_COL, how="left")
print(f"[stage1]   Stage-1 logit coverage: {df['stage1_prob_raw'].notna().mean():.1%}")

# OOT embargo mask (for Block 5 metric evaluation)
oot_idx_arr = df.index[oot_mask_arr].to_numpy()
oot_dt = df.loc[oot_idx_arr, TIME_COL].to_numpy()
dev_max_dt = df.loc[dev_mask_arr, TIME_COL].max()
oot_embargo_keep = oot_dt >= (dev_max_dt + EMBARGO_SECONDS)
oot_eval_idx = oot_idx_arr[oot_embargo_keep]
print(f"[protocol] OOT evaluation rows (after 1-day embargo): {len(oot_eval_idx):,}")

y_oot_eval = df.loc[oot_eval_idx, TARGET_COL].to_numpy()
amt_oot_eval = df.loc[oot_eval_idx, AMT_COL].to_numpy()

# =============================================================================
# 14. Shared Pass-B Setup Helpers
# =============================================================================

def make_dev_oot_partition(agg_mode: str = "mean"):
    """Build Z, histories, and agg arrays for the frozen OOT pass."""
    dev_idx = df.index[dev_mask_arr].to_numpy()
    oot_idx = df.index[oot_mask_arr].to_numpy()
    assert_causal(df, dev_idx, oot_idx, "frozen_oot")

    df_dev = df.loc[dev_idx].reset_index(drop=True)
    df_oot = df.loc[oot_idx].reset_index(drop=True)
    Z_dev, Z_oot = build_neural_features(df_dev, df_oot, v_cols, num_cols)

    dev_oot = pd.concat([df_dev, df_oot], ignore_index=True)
    dt_devoot = dev_oot[TIME_COL].to_numpy()

    ch_oot, _ = extract_causal_predecessors(
        dt_devoot, dev_oot["uid_card"].to_numpy(), CARD_MAX_NEIGHBORS)
    dh_oot, _ = extract_causal_predecessors(
        dt_devoot, dev_oot["uid_device"].to_numpy(), DEVICE_MAX_NEIGHBORS)

    agg_devoot = neighbour_means(np.vstack([Z_dev, Z_oot]), ch_oot, dh_oot, agg_mode=agg_mode)
    n_dev = len(df_dev)
    agg_dev = slice_aggs(agg_devoot, 0, n_dev)
    agg_oot = slice_aggs(agg_devoot, n_dev, n_dev + len(df_oot))

    dt_dev = df_dev[TIME_COL].to_numpy()
    fit_m, stop_m = rows_positional_split(len(df_dev), dt_dev, INNER_VAL_FRAC)
    y_dev = df_dev[TARGET_COL].to_numpy()

    return (Z_dev, Z_oot, agg_dev, agg_oot, y_dev, fit_m, stop_m,
            df_dev, df_oot, ch_oot, dh_oot, dev_oot, n_dev)


def eval_on_oot(p_oot_full: np.ndarray) -> dict[str, float]:
    """Evaluate on the embargoed OOT evaluation subset."""
    p_eval = p_oot_full[oot_embargo_keep]
    return compute_metrics(y_oot_eval, p_eval, amt_oot_eval)


# =============================================================================
# 15. ABLATION A — Degree-Aware GNN Aggregation
# =============================================================================

all_results: list[dict] = []

if not ARGS.skip_ablation_a:
    print("\n" + "=" * 80)
    print("ABLATION A: DEGREE-AWARE GNN AGGREGATION")
    print("=" * 80)

    DEG_VARIANTS = [
        ("A1_log1p_scaled", "log1p_scaled",
         "Mean pooling weighted by log(1+degree_i): counts as a scale factor on the aggregated vector"),
        ("A2_sum",          "sum",
         "Sum aggregation (count-aware, no normalisation)"),
        ("A3_concat_logdeg","concat_logdeg",
         "Original mean-pool PLUS log(1+degree) appended as extra feature channels (card+dev)"),
    ]

    for variant_id, agg_mode, description in DEG_VARIANTS:
        print(f"\n--- Ablation {variant_id}: {description} ---")
        t0 = time.time()

        (Z_dev, Z_oot, agg_dev, agg_oot, y_dev,
         fit_m, stop_m, df_dev, df_oot,
         ch_oot, dh_oot, dev_oot, n_dev) = make_dev_oot_partition(agg_mode=agg_mode)

        # For concat_logdeg: append extra log-degree channels into Z
        if agg_mode == "concat_logdeg":
            # agg_dev has 6 elements: (ca, cm, da, dm, cld, dld)
            cld_dev, dld_dev = agg_dev[4], agg_dev[5]
            cld_oot, dld_oot = agg_oot[4], agg_oot[5]
            # Augment Z with log-degree channels
            Z_dev_aug = np.hstack([Z_dev, cld_dev, dld_dev]).astype(np.float32)
            Z_oot_aug = np.hstack([Z_oot, cld_oot, dld_oot]).astype(np.float32)
            # Trim agg tuples to standard 4-tuple for the model's card/dev args
            agg_dev_trim = agg_dev[:4]
            agg_oot_trim = agg_oot[:4]
        else:
            Z_dev_aug = Z_dev
            Z_oot_aug = Z_oot
            agg_dev_trim = agg_dev
            agg_oot_trim = agg_oot

        agg_fit = slice_aggs(agg_dev_trim, 0, int(fit_m.sum()))
        # We need positional indexing matching fit/stop masks
        # Rebuild positional slices
        fit_indices = np.where(fit_m)[0]
        stop_indices = np.where(stop_m)[0]
        agg_fit_pos = tuple(a[fit_m] for a in agg_dev_trim)
        agg_stop_pos = tuple(a[stop_m] for a in agg_dev_trim)

        y_fit = y_dev[fit_m]
        y_stop = y_dev[stop_m]

        p_oot_raw, p_stop_raw, cal = train_multiseed_s3(
            Z_dev_aug[fit_m], agg_fit_pos, y_fit,
            Z_dev_aug[stop_m], agg_stop_pos, y_stop,
            Z_oot_aug, agg_oot_trim,
            n_seeds=N_SEEDS_ORIG,
            base_seed=SEED,
            agg_mode=agg_mode,
            label=variant_id,
        )
        p_oot_cal = apply_calibrator(cal, p_oot_raw)

        m_raw = eval_on_oot(p_oot_raw)
        m_cal = eval_on_oot(p_oot_cal)
        boot = bootstrap_ap(y_oot_eval, p_oot_raw[oot_embargo_keep])
        elapsed = time.time() - t0

        rec = {
            "ablation": "A",
            "variant_id": variant_id,
            "description": description,
            "agg_mode": agg_mode,
            "n_seeds": N_SEEDS_ORIG,
            "elapsed_secs": round(elapsed, 1),
            "AP_raw": m_raw["AP"],
            "AP_calibrated": m_cal["AP"],
            "AP_boot_mean": boot["mean"],
            "AP_boot_std": boot["std"],
            "AP_boot_ci_lo": boot["ci_lo"],
            "AP_boot_ci_hi": boot["ci_hi"],
            "ROC_AUC_raw": m_raw["ROC_AUC"],
            "queue_loss_1pct_raw": m_raw["queue_loss_1pct"],
            "bayes_loss_corr_raw": m_raw["bayes_loss_corr"],
            "n_alerts_bayes_raw": m_raw["n_alerts_bayes"],
        }
        all_results.append(rec)

        print(f"  AP_raw={m_raw['AP']:.4f} AP_cal={m_cal['AP']:.4f} "
              f"ROC={m_raw['ROC_AUC']:.4f} QL={m_raw['queue_loss_1pct']:,.0f} "
              f"BL={m_raw['bayes_loss_corr']:,.0f} [{elapsed:.0f}s]")

        del Z_dev, Z_oot, agg_devoot, ch_oot, dh_oot, dev_oot
        gc.collect()

else:
    print("\n[SKIP] Ablation A skipped (--skip-ablation-a)")


# =============================================================================
# 16. ABLATION B — 30-Day Windowed Stage-2 Relational Features
# =============================================================================

if not ARGS.skip_ablation_b:
    print("\n" + "=" * 80)
    print("ABLATION B: 30-DAY WINDOWED STAGE-2 RELATIONAL FEATURES")
    print("=" * 80)
    t0 = time.time()

    # Build Stage-2 windowed features over the ENTIRE dataset (chronological order)
    # and then train/evaluate using expanding block folds + sealed OOT.
    print("[B] Building 30-day windowed relational features over full dataset...")
    rel_win = build_windowed_relational_features(df, "uid_card", "uid_device",
                                                  lookback_secs=LOOKBACK_WINDOW_SECS)
    stage2w_features = base_features + list(rel_win.columns)

    # Merge features temporarily for GBDT training
    df_s2w = pd.concat([df[base_features + [TIME_COL, TARGET_COL, ID_COL, AMT_COL,
                                             "chrono_block", "uid_card", "uid_device",
                                             "card_first_seen"]], rel_win], axis=1)
    del rel_win; gc.collect()

    # Expanding fold CV (Blocks 1-3 as val folds) + sealed OOT
    folds_b = [
        ([1], 2),
        ([1, 2], 3),
        ([1, 2, 3], 4),
    ]

    fold_ap_b: list[float] = []
    final_rounds_b: list[int] = []

    for train_blocks, val_block in folds_b:
        print(f"  [B] Fold train={train_blocks} -> val Block {val_block}")
        tr_idx = df_s2w.index[df_s2w["chrono_block"].isin(train_blocks)].to_numpy()
        va_idx = df_s2w.index[df_s2w["chrono_block"] == val_block].to_numpy()
        assert_causal(df_s2w, tr_idx, va_idx, f"ablB_fold_val{val_block}")

        # Embargo
        tr_max_dt = df_s2w.loc[tr_idx, TIME_COL].max()
        va_embargo = df_s2w.loc[va_idx, TIME_COL].to_numpy() >= tr_max_dt + EMBARGO_SECONDS

        x_tr = df_s2w.loc[tr_idx, stage2w_features].to_numpy(dtype=np.float32)
        y_tr = df_s2w.loc[tr_idx, TARGET_COL].to_numpy()
        x_va = df_s2w.loc[va_idx, stage2w_features].to_numpy(dtype=np.float32)
        y_va = df_s2w.loc[va_idx, TARGET_COL].to_numpy()

        # Inner split for early stopping
        n_tr = len(x_tr)
        dt_tr = df_s2w.loc[tr_idx, TIME_COL].to_numpy()
        fit_m, stop_m = rows_positional_split(n_tr, dt_tr, INNER_VAL_FRAC)

        dtrain = lgb.Dataset(x_tr[fit_m], label=y_tr[fit_m], free_raw_data=True)
        dstop = lgb.Dataset(x_tr[stop_m], label=y_tr[stop_m], reference=dtrain, free_raw_data=True)

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        bst = lgb.train(
            LGB_PARAMS_S2,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dstop],
            callbacks=callbacks,
        )
        final_rounds_b.append(bst.best_iteration)
        p_va = bst.predict(x_va)[va_embargo]
        ap_v = float(average_precision_score(y_va[va_embargo], p_va)) if y_va[va_embargo].sum() > 0 else float("nan")
        fold_ap_b.append(ap_v)
        print(f"       AP={ap_v:.4f} rounds={bst.best_iteration}")
        del dtrain, dstop, bst
        gc.collect()

    # Sealed OOT evaluation
    print("  [B] Training frozen Stage-2w model on Blocks 1-4 -> scoring Block 5")
    dev_b_idx = df_s2w.index[df_s2w["chrono_block"].isin(DEV_BLOCKS)].to_numpy()
    oot_b_idx = df_s2w.index[df_s2w["chrono_block"] == OOT_BLOCK].to_numpy()
    assert_causal(df_s2w, dev_b_idx, oot_b_idx, "ablB_oot")
    # OOT target must NOT be read in this block
    assert all(i not in OOT_INDEX for i in dev_b_idx), "OOT rows leaked into dev"

    n_rounds_final = int(np.median(final_rounds_b))
    x_dev_b = df_s2w.loc[dev_b_idx, stage2w_features].to_numpy(dtype=np.float32)
    y_dev_b = df_s2w.loc[dev_b_idx, TARGET_COL].to_numpy()
    x_oot_b = df_s2w.loc[oot_b_idx, stage2w_features].to_numpy(dtype=np.float32)

    dt_dev_b = df_s2w.loc[dev_b_idx, TIME_COL].to_numpy()
    fit_m_b, stop_m_b = rows_positional_split(len(x_dev_b), dt_dev_b, INNER_VAL_FRAC)

    dtrain_b = lgb.Dataset(x_dev_b[fit_m_b], label=y_dev_b[fit_m_b], free_raw_data=True)
    dstop_b = lgb.Dataset(x_dev_b[stop_m_b], label=y_dev_b[stop_m_b],
                          reference=dtrain_b, free_raw_data=True)
    bst_b = lgb.train(
        LGB_PARAMS_S2, dtrain_b, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dstop_b],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    p_oot_b = bst_b.predict(x_oot_b)

    # Isotonic calibration
    p_stop_b = bst_b.predict(x_dev_b[stop_m_b])
    cal_b = fit_calibrator(p_stop_b, y_dev_b[stop_m_b])
    p_oot_b_cal = apply_calibrator(cal_b, p_oot_b)

    # Evaluate on embargoed OOT
    m_b_raw = eval_on_oot(p_oot_b)
    m_b_cal = eval_on_oot(p_oot_b_cal)
    boot_b = bootstrap_ap(y_oot_eval, p_oot_b[oot_embargo_keep])
    elapsed_b = time.time() - t0

    rec_b = {
        "ablation": "B",
        "variant_id": "B_windowed_stage2",
        "description": f"Stage-2 LightGBM with 30-day windowed relational features (lookback={LOOKBACK_WINDOW_SECS}s)",
        "n_relational_features": len(rel_win.columns) if "rel_win" in dir() else -1,
        "fold_APs": fold_ap_b,
        "mean_dev_AP": float(np.nanmean(fold_ap_b)),
        "n_boost_rounds_oot": int(bst_b.best_iteration),
        "n_boost_rounds_median_dev": n_rounds_final,
        "elapsed_secs": round(elapsed_b, 1),
        "AP_raw": m_b_raw["AP"],
        "AP_calibrated": m_b_cal["AP"],
        "AP_boot_mean": boot_b["mean"],
        "AP_boot_std": boot_b["std"],
        "AP_boot_ci_lo": boot_b["ci_lo"],
        "AP_boot_ci_hi": boot_b["ci_hi"],
        "ROC_AUC_raw": m_b_raw["ROC_AUC"],
        "queue_loss_1pct_raw": m_b_raw["queue_loss_1pct"],
        "bayes_loss_corr_raw": m_b_raw["bayes_loss_corr"],
        "n_alerts_bayes_raw": m_b_raw["n_alerts_bayes"],
    }
    all_results.append(rec_b)

    print(f"  AP_raw={m_b_raw['AP']:.4f} AP_cal={m_b_cal['AP']:.4f} "
          f"ROC={m_b_raw['ROC_AUC']:.4f} QL={m_b_raw['queue_loss_1pct']:,.0f} "
          f"BL={m_b_raw['bayes_loss_corr']:,.0f} [{elapsed_b:.0f}s]")
    del df_s2w, x_dev_b, x_oot_b, dtrain_b, dstop_b, bst_b
    gc.collect()

else:
    print("\n[SKIP] Ablation B skipped (--skip-ablation-b)")


# =============================================================================
# 17. MULTI-SEED NEURAL STABILITY SWEEP (W6)
# =============================================================================

if not ARGS.skip_seed_sweep:
    print("\n" + "=" * 80)
    print("MULTI-SEED NEURAL STABILITY SWEEP (5 seeds)")
    print("=" * 80)

    SWEEP_VARIANTS = [
        # (variant_id, model_type, boosting_lr, n_seeds_orig, description)
        ("S3_real",    "s3",   0.0, N_SEEDS_ORIG, "Stage 3 Frozen Res-GNN (real graph, mean-pool)"),
        ("S3n_no_graph","s3n", 0.0, N_SEEDS_ORIG, "Stage 3n No-Graph Ablation (zeros agg)"),
        ("S4_joint",   "s4",   BOOSTING_LR, N_SEEDS_S4, "Stage 4 Joint BGNN (M=4, J=2)"),
        ("S4n_no_boost","s4n", 0.0,          N_SEEDS_S4, "Stage 4n No-Boost Gate Ablation"),
    ]

    (Z_dev_sw, Z_oot_sw, agg_dev_sw, agg_oot_sw, y_dev_sw,
     fit_m_sw, stop_m_sw, df_dev_sw, df_oot_sw,
     ch_oot_sw, dh_oot_sw, dev_oot_sw, n_dev_sw) = make_dev_oot_partition(agg_mode="mean")

    dt_dev_sw = df_dev_sw[TIME_COL].to_numpy()
    logit_dev_sw = np.where(
        df_dev_sw["stage1_prob_raw"].notna(),
        np.log(np.clip(df_dev_sw["stage1_prob_raw"], 1e-6, 1 - 1e-6)
               / (1 - np.clip(df_dev_sw["stage1_prob_raw"], 1e-6, 1 - 1e-6))),
        np.log(y_dev_sw.mean() / (1 - y_dev_sw.mean() + 1e-9)),
    ).astype(np.float32)
    logit_oot_sw = np.where(
        df_oot_sw["stage1_prob_raw"].notna(),
        np.log(np.clip(df_oot_sw["stage1_prob_raw"], 1e-6, 1 - 1e-6)
               / (1 - np.clip(df_oot_sw["stage1_prob_raw"], 1e-6, 1 - 1e-6))),
        np.log(y_dev_sw.mean() / (1 - y_dev_sw.mean() + 1e-9)),
    ).astype(np.float32)

    n_fit_sw = int(fit_m_sw.sum())
    n_tr_sw = len(Z_dev_sw)

    for var_id, model_type, blr, n_seeds_orig, description in SWEEP_VARIANTS:
        print(f"\n--- Sweep: {var_id} | {description} ---")
        per_seed_records: list[dict] = []

        for seed in STABILITY_SEEDS:
            print(f"  Seed {seed}...")
            t0 = time.time()

            if model_type in ("s3", "s3n"):
                use_blank = (model_type == "s3n")
                agg_fit_v = blank_aggs(n_fit_sw, Z_dev_sw.shape[1]) if use_blank else tuple(a[fit_m_sw] for a in agg_dev_sw)
                agg_stop_v = blank_aggs(int(stop_m_sw.sum()), Z_dev_sw.shape[1]) if use_blank else tuple(a[stop_m_sw] for a in agg_dev_sw)
                agg_oot_v = blank_aggs(len(Z_oot_sw), Z_oot_sw.shape[1]) if use_blank else agg_oot_sw

                model, ep, ap = train_one_model_s3(
                    Z_dev_sw[fit_m_sw], agg_fit_v, y_dev_sw[fit_m_sw],
                    Z_dev_sw[stop_m_sw], agg_stop_v, y_dev_sw[stop_m_sw],
                    seed=seed, agg_mode="mean", label=f"{var_id}/seed{seed}",
                )
                p_oot_raw = infer_s3(model, Z_oot_sw, agg_oot_v)
                p_stop_v = infer_s3(model, Z_dev_sw[stop_m_sw], agg_stop_v)
                cal_v = fit_calibrator(p_stop_v, y_dev_sw[stop_m_sw])
                del model

            else:  # s4 or s4n
                p_oot_raw, p_stop_v, rnd, ap = run_one_bgnn(
                    Z_dev_sw, Z_oot_sw, logit_dev_sw, logit_oot_sw,
                    y_dev_sw[fit_m_sw], y_dev_sw[stop_m_sw],
                    n_fit_sw, n_tr_sw, ch_oot_sw, dh_oot_sw,
                    n_tr_sw + len(Z_oot_sw),
                    boosting_lr=blr, seed=seed,
                )
                cal_v = fit_calibrator(p_stop_v, y_dev_sw[stop_m_sw])

            p_oot_cal = apply_calibrator(cal_v, p_oot_raw)
            m_raw = eval_on_oot(p_oot_raw)
            m_cal = eval_on_oot(p_oot_cal)
            elapsed_v = time.time() - t0

            per_seed_records.append({
                "ablation": "W6_seed_sweep",
                "variant_id": var_id,
                "seed": seed,
                "description": description,
                "elapsed_secs": round(elapsed_v, 1),
                "AP_raw": m_raw["AP"],
                "AP_calibrated": m_cal["AP"],
                "ROC_AUC_raw": m_raw["ROC_AUC"],
                "queue_loss_1pct_raw": m_raw["queue_loss_1pct"],
                "bayes_loss_corr_raw": m_raw["bayes_loss_corr"],
                "n_alerts_bayes_raw": m_raw["n_alerts_bayes"],
            })
            all_results.extend(per_seed_records[-1:])
            print(f"    AP_raw={m_raw['AP']:.4f} AP_cal={m_cal['AP']:.4f} "
                  f"ROC={m_raw['ROC_AUC']:.4f} [{elapsed_v:.0f}s]")
            gc.collect()

        # Summary row for this variant
        vals_ap = [r["AP_raw"] for r in per_seed_records if not np.isnan(r["AP_raw"])]
        vals_apc = [r["AP_calibrated"] for r in per_seed_records if not np.isnan(r["AP_calibrated"])]
        vals_roc = [r["ROC_AUC_raw"] for r in per_seed_records if not np.isnan(r["ROC_AUC_raw"])]
        vals_ql = [r["queue_loss_1pct_raw"] for r in per_seed_records]
        vals_bl = [r["bayes_loss_corr_raw"] for r in per_seed_records]

        summary_rec = {
            "ablation": "W6_seed_sweep_SUMMARY",
            "variant_id": var_id + "_SUMMARY",
            "description": description,
            "n_seeds": len(STABILITY_SEEDS),
            "seeds": STABILITY_SEEDS,
            "AP_raw_mean": float(np.mean(vals_ap)),
            "AP_raw_std": float(np.std(vals_ap, ddof=1)) if len(vals_ap) > 1 else 0.0,
            "AP_raw_min": float(np.min(vals_ap)),
            "AP_raw_max": float(np.max(vals_ap)),
            "AP_calibrated_mean": float(np.mean(vals_apc)),
            "AP_calibrated_std": float(np.std(vals_apc, ddof=1)) if len(vals_apc) > 1 else 0.0,
            "ROC_AUC_mean": float(np.mean(vals_roc)),
            "ROC_AUC_std": float(np.std(vals_roc, ddof=1)) if len(vals_roc) > 1 else 0.0,
            "queue_loss_mean": float(np.mean(vals_ql)),
            "queue_loss_std": float(np.std(vals_ql, ddof=1)) if len(vals_ql) > 1 else 0.0,
            "bayes_loss_mean": float(np.mean(vals_bl)),
            "bayes_loss_std": float(np.std(vals_bl, ddof=1)) if len(vals_bl) > 1 else 0.0,
        }
        all_results.append(summary_rec)
        print(f"\n  SUMMARY {var_id}: AP_raw={summary_rec['AP_raw_mean']:.4f}±{summary_rec['AP_raw_std']:.4f} "
              f"AP_cal={summary_rec['AP_calibrated_mean']:.4f}±{summary_rec['AP_calibrated_std']:.4f} "
              f"ROC={summary_rec['ROC_AUC_mean']:.4f}±{summary_rec['ROC_AUC_std']:.4f}")

    del Z_dev_sw, Z_oot_sw, agg_dev_sw, agg_oot_sw
    gc.collect()

else:
    print("\n[SKIP] Multi-seed sweep skipped (--skip-seed-sweep)")


# =============================================================================
# 18. Save Results
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

json_path = OUTPUT_DIR / "reviewer_ablations_and_seeds.json"
csv_path = OUTPUT_DIR / "reviewer_ablations_and_seeds.csv"

with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"[output] JSON: {json_path}")

flat_rows = []
for r in all_results:
    flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
            for k, v in r.items()}
    flat_rows.append(flat)

pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
print(f"[output] CSV:  {csv_path}")


# =============================================================================
# 19. Print Summary Table
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"\n{'Variant':35s} {'AP_raw':>8s} {'AP_cal':>8s} {'ROC_AUC':>8s} "
      f"{'QL_1pct':>12s} {'BL_corr':>12s}")
print("-" * 90)
for r in all_results:
    vid = r.get("variant_id", "?")
    ap_r = r.get("AP_raw", r.get("AP_raw_mean", float("nan")))
    ap_c = r.get("AP_calibrated", r.get("AP_calibrated_mean", float("nan")))
    roc  = r.get("ROC_AUC_raw", r.get("ROC_AUC_mean", float("nan")))
    ql   = r.get("queue_loss_1pct_raw", r.get("queue_loss_mean", float("nan")))
    bl   = r.get("bayes_loss_corr_raw", r.get("bayes_loss_mean", float("nan")))
    ap_std = r.get("AP_raw_std", None)
    suffix = f" ±{ap_std:.4f}" if ap_std is not None else ""

    def _f(v, fmt=".4f"):
        return format(v, fmt) if isinstance(v, float) and not np.isnan(v) else "   —"

    print(f"{vid[:35]:35s} {_f(ap_r)+suffix:>14s} {_f(ap_c):>8s} {_f(roc):>8s} "
          f"{_f(ql, ',.0f'):>12s} {_f(bl, ',.0f'):>12s}")

print("\n[done] All ablations complete.")
