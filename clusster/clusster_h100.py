from __future__ import annotations
import os
import sys
import glob
import json
import math
import argparse
from typing import List, Tuple, Dict, Any

# --- жестко фиксируем 16 CPU threads ---
CPU_THREADS = 16
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

set_num_threads(CPU_THREADS)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.decomposition import PCA
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
    Birch,
    DBSCAN,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import optuna


def parse_args():
    ap = argparse.ArgumentParser(
        "AE clustering for early companies, then assign later companies by nearest centroid in AE latent space."
    )

    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--all-csv", default="data/prices_all.csv")
    ap.add_argument("--price-col", default="auto")

    ap.add_argument("--base-year-start", default="2008-01-01")
    ap.add_argument("--min-rows", type=int, default=50)

    # Optuna
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--study-name", default="ae_clustering_2008_centroid_extend")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

    # AE search space
    ap.add_argument("--latent-dim-min", type=int, default=2)
    ap.add_argument("--latent-dim-max", type=int, default=16)
    ap.add_argument("--hidden1-choices", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--hidden2-choices", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--batch-size-choices", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--epochs-min", type=int, default=40)
    ap.add_argument("--epochs-max", type=int, default=200)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=5e-3)

    # clustering
    ap.add_argument(
        "--algo",
        nargs="+",
        default=["kmeans", "agglo", "gmm", "spectral", "birch", "dbscan"],
        choices=["kmeans", "agglo", "gmm", "spectral", "birch", "dbscan"],
    )
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=12)

    # penalties
    ap.add_argument("--penalty-small-clusters-weight", type=float, default=35.0)
    ap.add_argument("--penalty-imbalance-weight", type=float, default=20.0)
    ap.add_argument("--penalty-smallest-cluster-weight", type=float, default=80.0)
    ap.add_argument("--target-min-clusters", type=int, default=4)
    ap.add_argument("--max-largest-share-without-penalty", type=float, default=0.45)
    ap.add_argument("--max-size-ratio-without-penalty", type=float, default=3.0)
    ap.add_argument("--min-cluster-size-without-penalty", type=int, default=3)

    # output dir
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default=None)

    # outputs
    ap.add_argument("--pdf-file", default="clusters_report.pdf")
    ap.add_argument("--out-csv", default="cluster_fullstart_assignments.csv")
    ap.add_argument("--out-html", default="cluster_table.html")
    ap.add_argument("--trials-csv", default="optuna_trials.csv")
    ap.add_argument("--best-json", default="optuna_best_params.json")
    ap.add_argument("--run-params-json", default="run_params.json")
    ap.add_argument("--gap-report-csv", default="company_missing_points_vs_etalon.csv")
    ap.add_argument("--late-details-csv", default="late_assignment_details.csv")
    ap.add_argument("--all-metrics-json", default="all_metrics.json")

    args, _unk = ap.parse_known_args(sys.argv[1:])
    return args


def safe_float(x: Any) -> float:
    try:
        item = getattr(x, "item", None)
        if callable(item):
            return float(item())
    except Exception:
        pass
    try:
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return float(np.mean(arr))
    except Exception:
        return float(x)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    return str(value)


OUTPUT_FILE_ARG_NAMES = [
    "pdf_file",
    "out_csv",
    "out_html",
    "trials_csv",
    "best_json",
    "run_params_json",
    "gap_report_csv",
    "late_details_csv",
    "all_metrics_json",
]


def make_safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name).strip())
    safe = safe.strip("._-")
    return safe or "run"


def make_unique_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        return path
    for i in range(1, 1000):
        p = f"{path}_{i:03d}"
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=False)
            return p
    raise SystemExit(f"Не удалось создать уникальную папку результатов для {path}")


def prepare_run_output_dir(args: argparse.Namespace) -> str:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        requested_dir = args.output_dir
    else:
        run_name = make_safe_run_name(args.run_name or f"run_{timestamp}")
        requested_dir = os.path.join(args.runs_root, run_name)

    output_dir = make_unique_dir(requested_dir)
    args.output_dir = output_dir

    for attr in OUTPUT_FILE_ARG_NAMES:
        value = getattr(args, attr)
        filename = os.path.basename(str(value))
        setattr(args, attr, os.path.join(output_dir, filename))

    return output_dir


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cuda(device: torch.device):
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(min(4, CPU_THREADS))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def load_prices(path_all_csv: str, data_dir: str) -> pd.DataFrame:
    if os.path.exists(path_all_csv):
        df = pd.read_csv(path_all_csv)
    else:
        parts = []
        for p in glob.glob(os.path.join(data_dir, "*.csv")):
            try:
                dfi = pd.read_csv(p)
                if "Ticker" in dfi.columns:
                    parts.append(dfi)
            except Exception:
                pass
        if not parts:
            raise SystemExit(f"Нет CSV с колонкой 'Ticker' в {data_dir}")
        df = pd.concat(parts, ignore_index=True)

    ren = {}
    for want in ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                ren[c] = want
                break

    df = df.rename(columns=ren)

    if "date" not in df.columns or "Ticker" not in df.columns:
        raise SystemExit("Нужны колонки: date, Ticker (+ цены).")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "Ticker"]).reset_index(drop=True)
    return df


def pick_price_col(df: pd.DataFrame, pref: str) -> str:
    if pref != "auto":
        if pref in df.columns:
            return pref
        raise SystemExit(f"--price-col={pref} не найден.")
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Close" in df.columns:
        return "Close"
    raise SystemExit("Не найдена колонка цены: ни 'Adj Close', ни 'Close'.")


def build_first_dates(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    dff = df[["Ticker", "date", price_col]].dropna(subset=[price_col]).copy()
    return (
        dff.groupby("Ticker", as_index=False)["date"]
        .min()
        .rename(columns={"date": "first_date"})
        .sort_values(["first_date", "Ticker"])
        .reset_index(drop=True)
    )


def build_etalon_calendar(df: pd.DataFrame, price_col: str) -> Tuple[str, np.ndarray]:
    dff = df[["Ticker", "date", price_col]].dropna(subset=[price_col]).copy()
    counts = dff.groupby("Ticker")["date"].nunique().sort_values(ascending=False)
    etalon_ticker = counts.index[0]
    etalon_dates = np.array(
        sorted(dff.loc[dff["Ticker"] == etalon_ticker, "date"].unique()),
        dtype="datetime64[ns]",
    )
    return etalon_ticker, etalon_dates


@njit
def fill_inside_gaps_midpoint_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    n = out.shape[0]
    i = 0
    while i < n:
        if np.isnan(out[i]):
            start = i
            while i < n and np.isnan(out[i]):
                i += 1
            end = i - 1
            left_idx = start - 1
            right_idx = i
            if left_idx >= 0 and right_idx < n:
                if (not np.isnan(out[left_idx])) and (not np.isnan(out[right_idx])):
                    fill_value = 0.5 * (out[left_idx] + out[right_idx])
                    for j in range(start, end + 1):
                        out[j] = fill_value
        else:
            i += 1
    return out


@njit(parallel=True)
def fill_inside_gaps_midpoint_2d(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    n_rows, n_cols = out.shape
    for col in prange(n_cols):
        i = 0
        while i < n_rows:
            if np.isnan(out[i, col]):
                start = i
                while i < n_rows and np.isnan(out[i, col]):
                    i += 1
                end = i - 1
                left_idx = start - 1
                right_idx = i
                if left_idx >= 0 and right_idx < n_rows:
                    lv = out[left_idx, col]
                    rv = out[right_idx, col]
                    if (not np.isnan(lv)) and (not np.isnan(rv)):
                        fill_value = 0.5 * (lv + rv)
                        for j in range(start, end + 1):
                            out[j, col] = fill_value
            else:
                i += 1
    return out


def build_gap_report(
    df: pd.DataFrame,
    price_col: str,
    etalon_dates: np.ndarray,
    out_csv: str,
) -> pd.DataFrame:
    dff = df[["Ticker", "date", price_col]].dropna(subset=[price_col]).copy()

    rows = []
    for ticker, grp in dff.groupby("Ticker"):
        company_dates = np.array(sorted(grp["date"].unique()), dtype="datetime64[ns]")
        first_date = pd.Timestamp(company_dates.min())
        last_date = pd.Timestamp(company_dates.max())
        actual_points = int(company_dates.size)

        expected_dates = etalon_dates[etalon_dates >= np.datetime64(first_date)]
        if expected_dates.size == 0:
            expected_points = 0
            missing_points = 0
            missing_ratio = np.nan
        else:
            expected_points = int(expected_dates.size)
            actual_on_grid = int(np.isin(expected_dates, company_dates).sum())
            missing_points = expected_points - actual_on_grid
            missing_ratio = missing_points / expected_points if expected_points > 0 else np.nan

        rows.append({
            "Ticker": ticker,
            "first_date": first_date,
            "last_date": last_date,
            "actual_points": actual_points,
            "expected_points_to_etalon_end": expected_points,
            "missing_points": missing_points,
            "missing_ratio": missing_ratio,
        })

    result = pd.DataFrame(rows).sort_values(
        ["missing_points", "missing_ratio", "Ticker"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    result.to_csv(out_csv, index=False)
    return result


def build_base_and_late_groups(
    first_dates: pd.DataFrame,
    base_trade_date: pd.Timestamp,
) -> Tuple[List[str], List[str]]:
    base_tickers = first_dates.loc[first_dates["first_date"] <= base_trade_date, "Ticker"].tolist()
    late_tickers = first_dates.loc[first_dates["first_date"] > base_trade_date, "Ticker"].tolist()
    return base_tickers, late_tickers


def build_wide_on_calendar(
    df: pd.DataFrame,
    price_col: str,
    tickers: List[str],
    calendar_dates: np.ndarray,
) -> pd.DataFrame:
    dff = df[df["Ticker"].isin(set(tickers))][["date", "Ticker", price_col]].copy()
    wide = dff.pivot(index="date", columns="Ticker", values=price_col).sort_index()
    wide = wide.reindex(pd.to_datetime(calendar_dates))
    return wide


def fill_wide_internal_gaps(wide: pd.DataFrame) -> pd.DataFrame:
    mat = wide.to_numpy(dtype=np.float64)
    filled = fill_inside_gaps_midpoint_2d(mat)
    return pd.DataFrame(filled, index=wide.index, columns=wide.columns)


def scale_minmax_wide(wide: pd.DataFrame) -> pd.DataFrame:
    wmin = wide.min(axis=0)
    wmax = wide.max(axis=0)
    denom = (wmax - wmin).replace(0, np.nan)
    scaled = (wide - wmin) / denom
    scaled = scaled.fillna(0.0)
    return scaled


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, z: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, z),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


def train_ae_return_model(
    X: np.ndarray,
    zdim: int,
    h1: int,
    h2: int,
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    trial: optuna.trial.Trial | None = None,
) -> Tuple[nn.Module, np.ndarray, float]:
    seed_everything(seed)

    _, d = X.shape
    batch = min(batch, X.shape[0])

    model = Autoencoder(d, h1, h2, zdim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(X)
    if device.type == "cuda":
        X_tensor = X_tensor.pin_memory()

    loader = torch.utils.data.DataLoader(
        X_tensor,
        batch_size=batch,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
        drop_last=False,
    )

    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        seen = 0

        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                recon = model(batch_x)
                loss = loss_fn(recon, batch_x)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.detach().item()) * batch_x.shape[0]
            seen += batch_x.shape[0]

        epoch_loss /= max(seen, 1)

        if trial is not None:
            trial.report(epoch_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.eval()
    with torch.no_grad():
        X_device = torch.from_numpy(X).to(device)
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            Z_t = model.encoder(X_device)
            R_t = model(X_device)

        Z = Z_t.detach().float().cpu().numpy()
        recon_mse = float(torch.mean((R_t.float() - X_device.float()) ** 2).detach().cpu().item())

    return model, Z, recon_mse


def encode_with_model(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    if X.shape[0] == 0:
        return np.empty((0, model.encoder[-2].out_features), dtype=np.float32)

    X_tensor = torch.from_numpy(X)
    if device.type == "cuda":
        X_tensor = X_tensor.pin_memory()

    loader = torch.utils.data.DataLoader(
        X_tensor,
        batch_size=min(batch_size, X.shape[0]),
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
        drop_last=False,
    )

    parts = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z = model.encoder(batch_x)
            parts.append(z.detach().float().cpu().numpy())

    return np.vstack(parts)


def get_cluster_size_stats(labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels)
    labels = labels[labels != -1]

    if labels.size == 0:
        return {
            "n_clusters_found": 0,
            "largest_cluster_size": 0,
            "smallest_cluster_size": 0,
            "largest_share": 1.0,
            "size_ratio": np.inf,
        }

    uniq, counts = np.unique(labels, return_counts=True)
    largest = int(np.max(counts))
    smallest = int(np.min(counts))
    total = int(np.sum(counts))

    largest_share = largest / total if total > 0 else 1.0
    size_ratio = (largest / smallest) if smallest > 0 else np.inf

    return {
        "n_clusters_found": int(len(uniq)),
        "largest_cluster_size": largest,
        "smallest_cluster_size": smallest,
        "largest_share": float(largest_share),
        "size_ratio": float(size_ratio),
    }


def score_clustering(Z: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels)
    noise_ratio = float(np.mean(labels == -1)) if np.any(labels == -1) else 0.0

    mask = labels != -1
    Z_eval = Z[mask]
    labels_eval = labels[mask]

    uniq_eval = np.unique(labels_eval) if len(labels_eval) > 0 else np.array([])
    size_stats = get_cluster_size_stats(labels)

    if len(labels_eval) < 3 or len(uniq_eval) < 2:
        return {
            "silhouette": -np.inf,
            "calinski": -np.inf,
            "davies": np.inf,
            "noise_ratio": noise_ratio,
            **size_stats,
        }

    sil = silhouette_score(Z_eval, labels_eval)
    cal = calinski_harabasz_score(Z_eval, labels_eval)
    dav = davies_bouldin_score(Z_eval, labels_eval)

    return {
        "silhouette": safe_float(sil),
        "calinski": safe_float(cal),
        "davies": safe_float(dav),
        "noise_ratio": noise_ratio,
        **size_stats,
    }


def small_cluster_count_penalty(n_clusters_found: int, target_min_clusters: int) -> float:
    if n_clusters_found >= target_min_clusters:
        return 0.0
    if n_clusters_found <= 1:
        return float(target_min_clusters ** 2)
    return float((target_min_clusters - n_clusters_found) ** 2)


def imbalance_penalty(
    largest_share: float,
    size_ratio: float,
    max_largest_share_without_penalty: float,
    max_size_ratio_without_penalty: float,
) -> float:
    penalty = 0.0
    if np.isfinite(largest_share):
        penalty += max(0.0, largest_share - max_largest_share_without_penalty)
    if np.isfinite(size_ratio):
        penalty += max(0.0, np.log(size_ratio / max_size_ratio_without_penalty))
    return float(penalty)


def smallest_cluster_penalty(
    smallest_cluster_size: int,
    min_cluster_size_without_penalty: int,
) -> float:
    if smallest_cluster_size >= min_cluster_size_without_penalty:
        return 0.0
    if smallest_cluster_size <= 0:
        return float(min_cluster_size_without_penalty ** 2)
    return float((min_cluster_size_without_penalty - smallest_cluster_size) ** 2)


def calinski_objective_with_penalties(
    calinski: float,
    n_clusters_found: int,
    largest_share: float,
    size_ratio: float,
    smallest_cluster_size: int,
    small_clusters_weight: float,
    imbalance_weight: float,
    smallest_cluster_weight: float,
    target_min_clusters: int,
    max_largest_share_without_penalty: float,
    max_size_ratio_without_penalty: float,
    min_cluster_size_without_penalty: int,
) -> Tuple[float, float, float, float, float]:
    if not np.isfinite(calinski) or calinski <= 0:
        return -1e18, -np.inf, np.inf, np.inf, np.inf

    base_score = float(np.log1p(calinski))

    p_small = small_cluster_count_penalty(
        n_clusters_found=n_clusters_found,
        target_min_clusters=target_min_clusters,
    )

    p_imb = imbalance_penalty(
        largest_share=largest_share,
        size_ratio=size_ratio,
        max_largest_share_without_penalty=max_largest_share_without_penalty,
        max_size_ratio_without_penalty=max_size_ratio_without_penalty,
    )

    p_tiny = smallest_cluster_penalty(
        smallest_cluster_size=smallest_cluster_size,
        min_cluster_size_without_penalty=min_cluster_size_without_penalty,
    )

    value = float(
        base_score
        - small_clusters_weight * p_small
        - imbalance_weight * p_imb
        - smallest_cluster_weight * p_tiny
    )

    return value, base_score, float(p_small), float(p_imb), float(p_tiny)


def cluster_and_score(
    Z: np.ndarray,
    algo: str,
    trial: optuna.trial.Trial,
    seed: int,
    k_min: int,
    k_max: int,
) -> Dict[str, Any]:
    n_samples = Z.shape[0]
    k_hi = min(k_max, max(k_min, n_samples - 1))

    if algo == "kmeans":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(Z)
        params = {"n_clusters": k}

    elif algo == "agglo":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        linkage = trial.suggest_categorical("agglo_linkage", ["ward", "complete", "average", "single"])
        if linkage == "ward":
            metric = "euclidean"
        else:
            metric = trial.suggest_categorical("agglo_metric", ["euclidean", "manhattan", "cosine"])
        labels = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric).fit_predict(Z)
        params = {"n_clusters": k, "linkage": linkage, "metric": metric}

    elif algo == "gmm":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        covariance_type = trial.suggest_categorical("gmm_covariance_type", ["full", "tied", "diag", "spherical"])
        reg_covar = trial.suggest_float("gmm_reg_covar", 1e-6, 1e-2, log=True)
        Z64 = np.asarray(Z, dtype=np.float64)
        try:
            labels = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=seed,
                n_init=2,
            ).fit_predict(Z64)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            raise optuna.TrialPruned()
        params = {"n_clusters": k, "covariance_type": covariance_type, "reg_covar": reg_covar}

    elif algo == "spectral":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        affinity = trial.suggest_categorical("spectral_affinity", ["nearest_neighbors", "rbf"])
        if affinity == "nearest_neighbors":
            max_nn = max(2, min(15, n_samples - 1))
            n_neighbors = trial.suggest_int("spectral_n_neighbors", 2, max_nn)
            labels = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                n_neighbors=n_neighbors,
                random_state=seed,
                assign_labels="kmeans",
            ).fit_predict(Z)
            params = {"n_clusters": k, "affinity": affinity, "n_neighbors": n_neighbors}
        else:
            gamma = trial.suggest_float("spectral_gamma", 1e-2, 10.0, log=True)
            labels = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                gamma=gamma,
                random_state=seed,
                assign_labels="kmeans",
            ).fit_predict(Z)
            params = {"n_clusters": k, "affinity": affinity, "gamma": gamma}

    elif algo == "birch":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        threshold = trial.suggest_float("birch_threshold", 0.1, 1.5)
        branching_factor = trial.suggest_int("birch_branching_factor", 20, 100)
        labels = Birch(n_clusters=k, threshold=threshold, branching_factor=branching_factor).fit_predict(Z)
        params = {"n_clusters": k, "threshold": threshold, "branching_factor": branching_factor}

    elif algo == "dbscan":
        eps = trial.suggest_float("dbscan_eps", 0.05, 3.0, log=True)
        min_samples = trial.suggest_int("dbscan_min_samples", 2, 10)
        metric = trial.suggest_categorical("dbscan_metric", ["euclidean", "manhattan", "cosine"])
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(Z)
        params = {"eps": eps, "min_samples": min_samples, "metric": metric}

    else:
        raise ValueError(f"Unknown algo: {algo}")

    metrics = score_clustering(Z, labels)
    return {"labels": labels, "metrics": metrics, "params": params}


def save_pdf(pdf_path: str, best: dict, companies: List[str], scaled: pd.DataFrame, title_suffix: str):
    Z = best["Z"]
    labels = best["labels"]
    algo = best["algo"]
    zdim = best["latent_dim"]
    m = best["metrics"]

    with PdfPages(pdf_path) as pdf:
        pca2 = PCA(n_components=2).fit_transform(Z)
        fig = plt.figure(figsize=(7.2, 6.2))
        sc = plt.scatter(
            pca2[:, 0],
            pca2[:, 1],
            c=labels,
            cmap="tab10",
            s=70,
            edgecolor="k",
            linewidths=0.3,
        )
        plt.title(
            f"{algo.upper()}  z={zdim}  "
            f"CH={m['calinski']:.1f}  Sil={m['silhouette']:.3f}  "
            f"DB={m['davies']:.3f}  {title_suffix}"
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.35)
        plt.colorbar(sc, label="cluster")
        pdf.savefig(fig)
        plt.close(fig)

        for cl in sorted(set(labels)):
            if cl == -1:
                continue
            idx = [i for i, lab in enumerate(labels) if lab == cl]
            show = idx[: min(12, len(idx))]
            if not show:
                continue

            cols = [companies[i] for i in show]
            fig2 = plt.figure(figsize=(12, 5))
            plt.plot(scaled.index, scaled[cols].values, linewidth=1.1)
            plt.title(f"{algo.upper()}  Cluster {cl}  (first {len(cols)} of {len(idx)})")
            plt.xlabel("Date")
            plt.ylabel("Normalised price (0..1)")
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

        cluster_map = pd.DataFrame({"Company": companies, "Cluster": labels}).sort_values(["Cluster", "Company"])
        fig3, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Company → Cluster", fontsize=14, pad=18)
        tbl = ax.table(cellText=cluster_map.values, colLabels=cluster_map.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.15)
        pdf.savefig(fig3)
        plt.close(fig3)


def save_interactive_table_html(path: str, cluster_map: pd.DataFrame, info: Dict[str, str]):
    table_html = cluster_map.to_html(index=False, classes="display compact", table_id="clusters")
    meta = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in info.items())

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Cluster assignments</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
<style>body{{font-family:Arial, sans-serif; margin:18px}}</style>
</head><body>
<h2>Cluster assignments</h2>
<ul>{meta}</ul>
{table_html}
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>
$(function(){{
  $('#clusters').DataTable({{
    pageLength: 25,
    lengthMenu: [10,25,50,100,200],
    order: [[1,'asc']],
    stateSave: true
  }});
}});
</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


class FixedTrialAdapter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def suggest_int(self, name, low, high):
        return int(self.params[name])

    def suggest_float(self, name, low, high, log=False):
        return float(self.params[name])

    def suggest_categorical(self, name, choices):
        return self.params[name]


@njit
def squared_euclidean_to_centroids(x: np.ndarray, centroids: np.ndarray) -> int:
    best_idx = -1
    best_val = 1e308
    for i in range(centroids.shape[0]):
        s = 0.0
        for j in range(centroids.shape[1]):
            d = x[j] - centroids[i, j]
            s += d * d
        if s < best_val:
            best_val = s
            best_idx = i
    return best_idx


def main():
    args = parse_args()
    output_dir = prepare_run_output_dir(args)
    device = pick_device()
    configure_cuda(device)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] CPU threads fixed to: {CPU_THREADS}")
    print("[INFO] Device:", device)
    if device.type == "cuda":
        print("[INFO] CUDA build:", torch.version.cuda)
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    df = load_prices(args.all_csv, args.data_dir)
    price_col = pick_price_col(df, args.price_col)

    first_dates = build_first_dates(df, price_col)
    etalon_ticker, etalon_dates = build_etalon_calendar(df, price_col)
    base_year_start = pd.Timestamp(args.base_year_start)

    etalon_dates_ts = pd.to_datetime(etalon_dates)
    etalon_2008_dates = etalon_dates_ts[etalon_dates_ts >= base_year_start]
    if len(etalon_2008_dates) == 0:
        raise SystemExit(f"На эталонном календаре нет дат >= {base_year_start.date()}")

    base_trade_date = pd.Timestamp(etalon_2008_dates.min())
    base_calendar = np.array(etalon_2008_dates, dtype="datetime64[ns]")

    _ = build_gap_report(df, price_col, etalon_dates, args.gap_report_csv)

    base_tickers, late_tickers = build_base_and_late_groups(first_dates, base_trade_date)

    print(f"[INFO] Etalon ticker: {etalon_ticker}")
    print(f"[INFO] Etalon points: {len(etalon_dates)}")
    print(f"[INFO] First trading date in 2008 on etalon calendar: {base_trade_date.date()}")
    print(f"[INFO] Base tickers (first_date <= first trading date 2008): {len(base_tickers)}")
    print(f"[INFO] Late tickers: {len(late_tickers)}")

    # Базовая группа
    base_wide = build_wide_on_calendar(df, price_col, base_tickers, base_calendar)

    raw_counts = (
        df[df["Ticker"].isin(set(base_tickers))]
        .dropna(subset=[price_col])
        .groupby("Ticker")["date"]
        .nunique()
    )
    good_base = raw_counts[raw_counts >= args.min_rows].index.tolist()
    base_wide = base_wide[good_base]

    if base_wide.shape[1] == 0:
        raise SystemExit("После min_rows в базовой группе не осталось тикеров.")

    base_wide_filled = fill_wide_internal_gaps(base_wide)

    still_bad = base_wide_filled.columns[base_wide_filled.isna().any()].tolist()
    if still_bad:
        print(f"[WARN] Удаляем базовые тикеры, где NaN остались после midpoint-fill: {still_bad[:20]}{' ...' if len(still_bad) > 20 else ''}")
        base_wide_filled = base_wide_filled.loc[:, ~base_wide_filled.isna().any()]

    if base_wide_filled.shape[1] == 0:
        raise SystemExit("После удаления тикеров с остаточными NaN базовая группа пуста.")

    base_scaled = scale_minmax_wide(base_wide_filled)
    X_base = base_scaled.T.values.astype(np.float32)
    base_companies = list(base_scaled.columns)

    print(f"[INFO] AE base matrix: companies={len(base_companies)}, time_points={X_base.shape[1]}")

    def objective(trial: optuna.trial.Trial) -> float:
        latent_dim = trial.suggest_int("latent_dim", args.latent_dim_min, args.latent_dim_max)
        hidden1 = trial.suggest_categorical("hidden1", args.hidden1_choices)
        hidden2 = trial.suggest_categorical("hidden2", args.hidden2_choices)

        if hidden2 > hidden1:
            raise optuna.TrialPruned()

        batch_size = trial.suggest_categorical("batch_size", args.batch_size_choices)
        epochs = trial.suggest_int("epochs", args.epochs_min, args.epochs_max)
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        algo = trial.suggest_categorical("algo", args.algo)

        _, Z, recon_mse = train_ae_return_model(
            X=X_base,
            zdim=latent_dim,
            h1=hidden1,
            h2=hidden2,
            epochs=epochs,
            batch=batch_size,
            lr=lr,
            seed=args.seed,
            device=device,
            trial=trial,
        )

        cluster_res = cluster_and_score(
            Z=Z,
            algo=algo,
            trial=trial,
            seed=args.seed,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        metrics = cluster_res["metrics"]

        if metrics["n_clusters_found"] < 3:
            raise optuna.TrialPruned()

        if metrics["smallest_cluster_size"] < 2:
            raise optuna.TrialPruned()

        if np.isfinite(metrics["largest_share"]) and metrics["largest_share"] > 0.85:
            raise optuna.TrialPruned()

        objective_value, log_calinski, p_small, p_imb, p_tiny = calinski_objective_with_penalties(
            calinski=metrics["calinski"],
            n_clusters_found=metrics["n_clusters_found"],
            largest_share=metrics["largest_share"],
            size_ratio=metrics["size_ratio"],
            smallest_cluster_size=metrics["smallest_cluster_size"],
            small_clusters_weight=args.penalty_small_clusters_weight,
            imbalance_weight=args.penalty_imbalance_weight,
            smallest_cluster_weight=args.penalty_smallest_cluster_weight,
            target_min_clusters=args.target_min_clusters,
            max_largest_share_without_penalty=args.max_largest_share_without_penalty,
            max_size_ratio_without_penalty=args.max_size_ratio_without_penalty,
            min_cluster_size_without_penalty=args.min_cluster_size_without_penalty,
        )

        trial.set_user_attr("recon_mse", float(recon_mse))
        trial.set_user_attr("silhouette", float(metrics["silhouette"]))
        trial.set_user_attr("calinski", float(metrics["calinski"]))
        trial.set_user_attr("log_calinski", float(log_calinski))
        trial.set_user_attr("davies", float(metrics["davies"]))
        trial.set_user_attr("n_clusters_found", int(metrics["n_clusters_found"]))
        trial.set_user_attr("noise_ratio", float(metrics["noise_ratio"]))
        trial.set_user_attr("largest_cluster_size", int(metrics["largest_cluster_size"]))
        trial.set_user_attr("smallest_cluster_size", int(metrics["smallest_cluster_size"]))
        trial.set_user_attr("largest_share", float(metrics["largest_share"]))
        trial.set_user_attr("size_ratio", float(metrics["size_ratio"]))
        trial.set_user_attr("penalty_small_k", float(p_small))
        trial.set_user_attr("penalty_imbalance", float(p_imb))
        trial.set_user_attr("penalty_tiny_cluster", float(p_tiny))
        trial.set_user_attr("objective_value", float(objective_value))

        return float(objective_value)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        catch=(ValueError, np.linalg.LinAlgError, RuntimeError, TypeError),
    )

    print("\n=== BEST TRIAL ===")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    bp = study.best_trial.params

    best_model, Z_base, recon_mse_best = train_ae_return_model(
        X=X_base,
        zdim=int(bp["latent_dim"]),
        h1=int(bp["hidden1"]),
        h2=int(bp["hidden2"]),
        epochs=int(bp["epochs"]),
        batch=int(bp["batch_size"]),
        lr=float(bp["lr"]),
        seed=args.seed,
        device=device,
        trial=None,
    )

    fixed_trial = FixedTrialAdapter(bp)
    cluster_best = cluster_and_score(
        Z=Z_base,
        algo=bp["algo"],
        trial=fixed_trial,
        seed=args.seed,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    metrics_best = cluster_best["metrics"]

    best_objective_value, best_log_calinski, p_small_best, p_imb_best, p_tiny_best = calinski_objective_with_penalties(
        calinski=metrics_best["calinski"],
        n_clusters_found=metrics_best["n_clusters_found"],
        largest_share=metrics_best["largest_share"],
        size_ratio=metrics_best["size_ratio"],
        smallest_cluster_size=metrics_best["smallest_cluster_size"],
        small_clusters_weight=args.penalty_small_clusters_weight,
        imbalance_weight=args.penalty_imbalance_weight,
        smallest_cluster_weight=args.penalty_smallest_cluster_weight,
        target_min_clusters=args.target_min_clusters,
        max_largest_share_without_penalty=args.max_largest_share_without_penalty,
        max_size_ratio_without_penalty=args.max_size_ratio_without_penalty,
        min_cluster_size_without_penalty=args.min_cluster_size_without_penalty,
    )

    best = {
        "algo": bp["algo"],
        "latent_dim": int(bp["latent_dim"]),
        "Z": Z_base,
        "labels": cluster_best["labels"],
        "metrics": metrics_best,
        "recon_mse": recon_mse_best,
        "best_params": bp,
        "best_objective_value": best_objective_value,
        "log_calinski": best_log_calinski,
        "penalty_small_k": p_small_best,
        "penalty_imbalance": p_imb_best,
        "penalty_tiny_cluster": p_tiny_best,
    }

    print("\n=== ЛУЧШАЯ КОНФИГУРАЦИЯ ДЛЯ AE-ГРУППЫ ===")
    print(f"algo={best['algo']}, z={best['latent_dim']}")
    print(f"Objective value: {safe_float(best['best_objective_value']):.6f}")
    print(f"log1p(Calinski-Harabasz): {safe_float(best['log_calinski']):.6f}")
    print(f"Calinski-Harabasz raw: {safe_float(best['metrics']['calinski']):.6f}")
    print(f"Silhouette: {safe_float(best['metrics']['silhouette']):.6f}")
    print(f"Davies-Bouldin: {safe_float(best['metrics']['davies']):.6f}")
    print(f"Recon MSE: {safe_float(best['recon_mse']):.6f}")
    print(f"Clusters found: {best['metrics']['n_clusters_found']}")
    print(f"Largest cluster size: {best['metrics']['largest_cluster_size']}")
    print(f"Smallest cluster size: {best['metrics']['smallest_cluster_size']}")
    print(f"Largest share: {safe_float(best['metrics']['largest_share']):.6f}")
    print(f"Size ratio: {safe_float(best['metrics']['size_ratio']):.6f}")
    print(f"Penalty small K: {safe_float(best['penalty_small_k']):.6f}")
    print(f"Penalty imbalance: {safe_float(best['penalty_imbalance']):.6f}")
    print(f"Penalty tiny cluster: {safe_float(best['penalty_tiny_cluster']):.6f}")

    # центроиды по базовой группе в latent-space
    base_cluster_map = pd.DataFrame({"Company": base_companies, "Cluster": best["labels"]})
    cluster_ids_sorted = sorted(base_cluster_map["Cluster"].unique().tolist())
    centroids = []
    centroid_cluster_ids = []
    for cid in cluster_ids_sorted:
        idx = np.where(best["labels"] == cid)[0]
        centroids.append(np.mean(Z_base[idx], axis=0))
        centroid_cluster_ids.append(cid)
    centroids = np.array(centroids, dtype=np.float64)

    # поздние компании: до старта торговли будут нули после minmax/fillna
    late_wide = build_wide_on_calendar(df, price_col, late_tickers, base_calendar)
    late_wide_filled = fill_wide_internal_gaps(late_wide)
    late_scaled = scale_minmax_wide(late_wide_filled)

    late_companies_present = list(late_scaled.columns)
    X_late = late_scaled.T.values.astype(np.float32) if len(late_companies_present) > 0 else np.empty((0, X_base.shape[1]), dtype=np.float32)
    Z_late = encode_with_model(best_model, X_late, device=device, batch_size=512) if X_late.shape[0] > 0 else np.empty((0, Z_base.shape[1]), dtype=np.float32)

    late_assign_rows = []
    late_labels = []

    for i, company in enumerate(late_companies_present):
        best_centroid_idx = squared_euclidean_to_centroids(Z_late[i].astype(np.float64), centroids)
        assigned_cluster = int(centroid_cluster_ids[best_centroid_idx])
        late_labels.append(assigned_cluster)

        dists = np.sum((centroids - Z_late[i].astype(np.float64)) ** 2, axis=1)
        order = np.argsort(dists)
        best_dist = float(dists[order[0]]) if len(order) >= 1 else np.nan
        second_best = float(dists[order[1]]) if len(order) >= 2 else np.nan
        margin = second_best - best_dist if np.isfinite(best_dist) and np.isfinite(second_best) else np.nan

        late_assign_rows.append({
            "Company": company,
            "assigned_cluster": assigned_cluster,
            "best_distance_to_centroid": best_dist,
            "second_best_distance_to_centroid": second_best,
            "distance_margin": margin,
        })

    late_details = pd.DataFrame(late_assign_rows).sort_values(["assigned_cluster", "Company"]).reset_index(drop=True)
    late_details.to_csv(args.late_details_csv, index=False)

    late_cluster_map = pd.DataFrame({"Company": late_companies_present, "Cluster": late_labels})

    final_cluster_map = pd.concat([base_cluster_map, late_cluster_map], ignore_index=True)
    final_cluster_map = (
        final_cluster_map.drop_duplicates(subset=["Company"], keep="last")
        .sort_values(["Cluster", "Company"])
        .reset_index(drop=True)
    )
    final_cluster_map.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved final cluster assignments → {args.out_csv}")

    # Метрики 2 раза
    overall_companies = base_companies + late_companies_present
    overall_labels = np.concatenate([best["labels"], np.array(late_labels, dtype=int)]) if len(late_labels) > 0 else best["labels"].copy()
    Z_all = np.vstack([Z_base, Z_late]) if Z_late.shape[0] > 0 else Z_base.copy()

    overall_metrics = score_clustering(Z_all, overall_labels)

    print("\n=== МЕТРИКИ ПО ВСЕМ КОМПАНИЯМ В ОБЩЕМ LATENT-SPACE ===")
    print(f"Companies used in overall metrics: {len(overall_companies)}")
    print(f"Overall Calinski-Harabasz: {safe_float(overall_metrics['calinski']):.6f}")
    print(f"Overall log1p(Calinski-Harabasz): {safe_float(np.log1p(overall_metrics['calinski'])) if np.isfinite(overall_metrics['calinski']) and overall_metrics['calinski'] > 0 else float('nan'):.6f}")
    print(f"Overall Silhouette: {safe_float(overall_metrics['silhouette']):.6f}")
    print(f"Overall Davies-Bouldin: {safe_float(overall_metrics['davies']):.6f}")

    save_pdf(
        args.pdf_file,
        best,
        base_companies,
        base_scaled,
        title_suffix=f"(AE base group from {base_trade_date.date()})",
    )
    print(f"[INFO] PDF report saved → {args.pdf_file}")

    info = {
        "Algorithm": best["algo"].upper(),
        "Latent dim": str(best["latent_dim"]),
        "Base trade date": str(base_trade_date.date()),
        "Base companies clustered by AE": str(len(base_companies)),
        "Late companies assigned by centroid": str(len(late_companies_present)),
        "Objective": "log1p(calinski)_with_penalties",
        "Objective value (AE group)": f"{safe_float(best['best_objective_value']):.6f}",
        "AE-group log1p(Calinski-Harabasz)": f"{safe_float(best['log_calinski']):.4f}",
        "AE-group Calinski-Harabasz raw": f"{safe_float(best['metrics']['calinski']):.4f}",
        "AE-group Silhouette": f"{safe_float(best['metrics']['silhouette']):.4f}",
        "AE-group Davies-Bouldin": f"{safe_float(best['metrics']['davies']):.4f}",
        "Overall Calinski-Harabasz raw": f"{safe_float(overall_metrics['calinski']):.4f}",
        "Overall log1p(Calinski-Harabasz)": f"{safe_float(np.log1p(overall_metrics['calinski'])) if np.isfinite(overall_metrics['calinski']) and overall_metrics['calinski'] > 0 else float('nan'):.4f}",
        "Overall Silhouette": f"{safe_float(overall_metrics['silhouette']):.4f}",
        "Overall Davies-Bouldin": f"{safe_float(overall_metrics['davies']):.4f}",
        "Price column": price_col,
        "Etalon ticker": etalon_ticker,
        "CPU threads": str(CPU_THREADS),
        "Trials": str(len(study.trials)),
    }
    save_interactive_table_html(args.out_html, final_cluster_map, info)
    print(f"[INFO] Interactive table saved → {args.out_html}")

    trials_rows = []
    for t in study.trials:
        row = {
            "number": t.number,
            "state": str(t.state),
            "value": t.value if t.value is not None else np.nan,
        }
        row.update(t.params)
        row["recon_mse"] = t.user_attrs.get("recon_mse", np.nan)
        row["silhouette"] = t.user_attrs.get("silhouette", np.nan)
        row["calinski"] = t.user_attrs.get("calinski", np.nan)
        row["log_calinski"] = t.user_attrs.get("log_calinski", np.nan)
        row["davies"] = t.user_attrs.get("davies", np.nan)
        row["n_clusters_found"] = t.user_attrs.get("n_clusters_found", np.nan)
        row["noise_ratio"] = t.user_attrs.get("noise_ratio", np.nan)
        row["largest_cluster_size"] = t.user_attrs.get("largest_cluster_size", np.nan)
        row["smallest_cluster_size"] = t.user_attrs.get("smallest_cluster_size", np.nan)
        row["largest_share"] = t.user_attrs.get("largest_share", np.nan)
        row["size_ratio"] = t.user_attrs.get("size_ratio", np.nan)
        row["penalty_small_k"] = t.user_attrs.get("penalty_small_k", np.nan)
        row["penalty_imbalance"] = t.user_attrs.get("penalty_imbalance", np.nan)
        row["penalty_tiny_cluster"] = t.user_attrs.get("penalty_tiny_cluster", np.nan)
        row["objective_value"] = t.user_attrs.get("objective_value", np.nan)
        trials_rows.append(row)

    trials_df = pd.DataFrame(trials_rows).sort_values(by=["value"], ascending=False, na_position="last")
    trials_df.to_csv(args.trials_csv, index=False)

    best_json_payload = {
        "objective": "log1p(calinski)_with_penalties",
        "best_value": safe_float(study.best_value),
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "base_trade_date_2008": str(base_trade_date.date()),
        "etalon_ticker": etalon_ticker,
        "cpu_threads": CPU_THREADS,
    }
    with open(args.best_json, "w", encoding="utf-8") as f:
        json.dump(best_json_payload, f, ensure_ascii=False, indent=2)

    metrics_payload = {
        "base_group_metrics_latent_space": make_json_safe(best["metrics"]),
        "base_group_objective_value": safe_float(best["best_objective_value"]),
        "base_group_log_calinski": safe_float(best["log_calinski"]),
        "base_group_recon_mse": safe_float(best["recon_mse"]),
        "overall_metrics_latent_space": make_json_safe(overall_metrics),
        "overall_log_calinski": safe_float(np.log1p(overall_metrics["calinski"])) if np.isfinite(overall_metrics["calinski"]) and overall_metrics["calinski"] > 0 else None,
        "n_base_companies": int(len(base_companies)),
        "n_late_companies": int(len(late_companies_present)),
        "n_total_companies_final": int(final_cluster_map.shape[0]),
    }
    with open(args.all_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    run_params = {
        "args": make_json_safe(vars(args)),
        "cpu_threads": CPU_THREADS,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "etalon_ticker": etalon_ticker,
        "etalon_points": int(len(etalon_dates)),
        "base_trade_date_2008": str(base_trade_date.date()),
        "n_base_initial": int(len(base_tickers)),
        "n_base_after_min_rows_and_fill": int(len(base_companies)),
        "n_late_initial": int(len(late_tickers)),
        "n_late_encoded": int(len(late_companies_present)),
        "dropped_base_after_fill": still_bad,
    }
    with open(args.run_params_json, "w", encoding="utf-8") as f:
        json.dump(run_params, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Gap report saved → {args.gap_report_csv}")
    print(f"[INFO] Late details saved → {args.late_details_csv}")
    print(f"[INFO] Trials saved → {args.trials_csv}")
    print(f"[INFO] Best params saved → {args.best_json}")
    print(f"[INFO] All metrics saved → {args.all_metrics_json}")
    print(f"[INFO] Run params saved → {args.run_params_json}")

    print("\n=== TOP-10 TRIALS ===")
    cols_show = [
        "number",
        "state",
        "value",
        "algo",
        "latent_dim",
        "log_calinski",
        "calinski",
        "silhouette",
        "davies",
        "n_clusters_found",
        "largest_cluster_size",
        "smallest_cluster_size",
        "largest_share",
        "size_ratio",
        "penalty_small_k",
        "penalty_imbalance",
        "penalty_tiny_cluster",
    ]
    existing_cols = [c for c in cols_show if c in trials_df.columns]
    print(trials_df[existing_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()