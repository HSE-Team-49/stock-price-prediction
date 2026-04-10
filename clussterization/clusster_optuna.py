from __future__ import annotations
import os
import glob
import sys
import argparse
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

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
        "AE clustering with Optuna for tickers starting at the global first date."
    )

    ap.add_argument("--data-dir", default="data", help="Каталог с CSV, если нет --all-csv")
    ap.add_argument("--all-csv", default="data/prices_all.csv", help="Общий CSV (tidy)")
    ap.add_argument("--price-col", default="auto", help="auto | Adj Close | Close | Open | High | Low")
    ap.add_argument("--min-rows", type=int, default=50, help="Мин. наблюдений на тикер")

    # Optuna
    ap.add_argument("--n-trials", type=int, default=80, help="Количество trial")
    ap.add_argument("--study-name", default="ae_clustering_optuna")
    ap.add_argument("--storage", default=None, help="Напр. sqlite:///ae_clustering_optuna.db")
    ap.add_argument("--timeout", type=int, default=None, help="Лимит времени в секундах")
    ap.add_argument("--seed", type=int, default=42)

    # search space AE
    ap.add_argument("--latent-dim-min", type=int, default=2)
    ap.add_argument("--latent-dim-max", type=int, default=16)
    ap.add_argument("--hidden1-choices", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--hidden2-choices", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--batch-size-choices", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--epochs-min", type=int, default=40)
    ap.add_argument("--epochs-max", type=int, default=200)
    ap.add_argument("--lr-min", type=float, default=1e-4)
    ap.add_argument("--lr-max", type=float, default=5e-3)

    # clustering methods
    ap.add_argument(
        "--algo",
        nargs="+",
        default=["kmeans", "agglo", "gmm", "spectral", "birch", "dbscan"],
        choices=["kmeans", "agglo", "gmm", "spectral", "birch", "dbscan"],
        help="Разрешённые методы кластеризации",
    )
    ap.add_argument("--k-min", type=int, default=3, help="Минимальное число кластеров")
    ap.add_argument("--k-max", type=int, default=12, help="Максимальное число кластеров")

    # penalties
    ap.add_argument(
        "--penalty-small-clusters-weight",
        type=float,
        default=35.0,
        help="Вес штрафа за слишком маленькое число кластеров",
    )
    ap.add_argument(
        "--penalty-imbalance-weight",
        type=float,
        default=20.0,
        help="Вес штрафа за дисбаланс размеров кластеров",
    )
    ap.add_argument(
        "--penalty-smallest-cluster-weight",
        type=float,
        default=80.0,
        help="Вес штрафа за слишком маленький минимальный размер кластера",
    )
    ap.add_argument(
        "--target-min-clusters",
        type=int,
        default=4,
        help="Желаемое минимальное число найденных кластеров без штрафа",
    )
    ap.add_argument(
        "--max-largest-share-without-penalty",
        type=float,
        default=0.45,
        help="Максимальная доля крупнейшего кластера без штрафа",
    )
    ap.add_argument(
        "--max-size-ratio-without-penalty",
        type=float,
        default=3.0,
        help="Максимальное отношение largest/smallest без штрафа",
    )
    ap.add_argument(
        "--min-cluster-size-without-penalty",
        type=int,
        default=3,
        help="Минимальный размер кластера без штрафа",
    )

    # outputs
    ap.add_argument("--pdf-file", default="clusters_fullstart_report.pdf")
    ap.add_argument("--out-csv", default="cluster_fullstart_assignments.csv")
    ap.add_argument("--out-html", default="cluster_fullstart_table.html")
    ap.add_argument("--trials-csv", default="optuna_trials.csv")
    ap.add_argument("--best-json", default="optuna_best_params.json")

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


def filter_tickers_start_at_global_min(df: pd.DataFrame) -> List[str]:
    gmin = df["date"].min()
    firsts = df.groupby("Ticker")["date"].min()
    keep = firsts[firsts == gmin].index.tolist()
    return keep


def build_matrix_full(
    df: pd.DataFrame,
    price_col: str,
    min_rows: int,
    tickers_keep: List[str],
) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex, pd.DataFrame]:
    dff = df[df["Ticker"].isin(set(tickers_keep))].copy()
    wide = dff.pivot(index="date", columns="Ticker", values=price_col).sort_index()

    good_cols = [c for c in wide.columns if wide[c].notna().sum() >= min_rows]
    wide = wide[good_cols]

    if wide.shape[1] == 0:
        raise SystemExit("После фильтрации по min_rows не осталось тикеров.")

    if wide.isna().any().any():
        bad = wide.columns[wide.isna().any()].tolist()
        print(f"[WARN] Удаляем тикеры с NaN: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
        wide = wide.loc[:, ~wide.isna().any()]

    if wide.shape[1] == 0:
        raise SystemExit("После удаления тикеров с NaN не осталось тикеров.")

    wmin = wide.min(axis=0)
    wmax = wide.max(axis=0)
    denom = (wmax - wmin).replace(0, np.nan)

    scaled = (wide - wmin) / denom
    scaled = scaled.fillna(0.0)

    X = scaled.T.values.astype(np.float32)
    companies = list(scaled.columns)
    dates = scaled.index
    return X, companies, dates, scaled


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, z: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, z),
            nn.ReLU(),  # возвращено как в исходной версии
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


def train_ae(
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
) -> Tuple[np.ndarray, float]:
    seed_everything(seed)

    _, d = X.shape
    model = Autoencoder(d, h1, h2, zdim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(X)
    if device.type == "cuda":
        X_tensor = X_tensor.pin_memory()
        torch.backends.cudnn.benchmark = True

    loader = torch.utils.data.DataLoader(
        X_tensor,
        batch_size=batch,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        seen = 0

        for batch_x in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
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
        infer_loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch, shuffle=False)

        Z_parts = []
        for batch_x in infer_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            z = model.encoder(batch_x).detach().cpu().numpy()
            Z_parts.append(z)

        Z = np.vstack(Z_parts)

        recon_err = 0.0
        count = 0
        for batch_x in infer_loader:
            bx = batch_x.to(device, non_blocking=True)
            r = model(bx).detach().cpu().numpy()
            recon_err += float(np.sum((r - batch_x.numpy()) ** 2))
            count += batch_x.shape[0]

        recon_mse = recon_err / (count * d)

    return Z, recon_mse


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
) -> Tuple[float, float, float, float]:
    if not np.isfinite(calinski):
        return -1e18, np.inf, np.inf, np.inf

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
        calinski
        - small_clusters_weight * p_small
        - imbalance_weight * p_imb
        - smallest_cluster_weight * p_tiny
    )
    return value, float(p_small), float(p_imb), float(p_tiny)


def cluster_and_score(
    Z: np.ndarray,
    algo: str,
    trial: optuna.trial.Trial,
    seed: int,
    k_min: int,
    k_max: int,
) -> Dict:
    n_samples = Z.shape[0]
    k_hi = min(k_max, max(k_min, n_samples - 1))

    if algo == "kmeans":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = model.fit_predict(Z)
        params = {"n_clusters": k}

    elif algo == "agglo":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        linkage = trial.suggest_categorical("agglo_linkage", ["ward", "complete", "average", "single"])

        if linkage == "ward":
            metric = "euclidean"
        else:
            metric = trial.suggest_categorical("agglo_metric", ["euclidean", "manhattan", "cosine"])

        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
        labels = model.fit_predict(Z)
        params = {"n_clusters": k, "linkage": linkage, "metric": metric}

    elif algo == "gmm":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        covariance_type = trial.suggest_categorical(
            "gmm_covariance_type", ["full", "tied", "diag", "spherical"]
        )
        reg_covar = trial.suggest_float("gmm_reg_covar", 1e-6, 1e-2, log=True)
        Z64 = np.asarray(Z, dtype=np.float64)

        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=seed,
                n_init=2,
            )
            labels = model.fit_predict(Z64)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            raise optuna.TrialPruned()

        params = {
            "n_clusters": k,
            "covariance_type": covariance_type,
            "reg_covar": reg_covar,
        }

    elif algo == "spectral":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        affinity = trial.suggest_categorical("spectral_affinity", ["nearest_neighbors", "rbf"])

        if affinity == "nearest_neighbors":
            max_nn = max(2, min(15, n_samples - 1))
            n_neighbors = trial.suggest_int("spectral_n_neighbors", 2, max_nn)
            model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                n_neighbors=n_neighbors,
                random_state=seed,
                assign_labels="kmeans",
            )
            labels = model.fit_predict(Z)
            params = {"n_clusters": k, "affinity": affinity, "n_neighbors": n_neighbors}
        else:
            gamma = trial.suggest_float("spectral_gamma", 1e-2, 10.0, log=True)
            model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                gamma=gamma,
                random_state=seed,
                assign_labels="kmeans",
            )
            labels = model.fit_predict(Z)
            params = {"n_clusters": k, "affinity": affinity, "gamma": gamma}

    elif algo == "birch":
        k = trial.suggest_int("n_clusters", k_min, k_hi)
        threshold = trial.suggest_float("birch_threshold", 0.1, 1.5)
        branching_factor = trial.suggest_int("birch_branching_factor", 20, 100)

        model = Birch(
            n_clusters=k,
            threshold=threshold,
            branching_factor=branching_factor,
        )
        labels = model.fit_predict(Z)
        params = {
            "n_clusters": k,
            "threshold": threshold,
            "branching_factor": branching_factor,
        }

    elif algo == "dbscan":
        eps = trial.suggest_float("dbscan_eps", 0.05, 3.0, log=True)
        min_samples = trial.suggest_int("dbscan_min_samples", 2, 10)
        metric = trial.suggest_categorical("dbscan_metric", ["euclidean", "manhattan", "cosine"])

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        labels = model.fit_predict(Z)
        params = {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
        }

    else:
        raise ValueError(f"Unknown algo: {algo}")

    metrics = score_clustering(Z, labels)

    return {
        "labels": labels,
        "metrics": metrics,
        "params": params,
    }


def save_pdf(pdf_path: str, best: dict, companies: List[str], scaled: pd.DataFrame, price_col: str):
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
            f"DB={m['davies']:.3f}  ({price_col})"
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

        cluster_map = pd.DataFrame({"Company": companies, "Cluster": labels}).sort_values(
            ["Cluster", "Company"]
        )
        fig3, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Company → Cluster", fontsize=14, pad=18)
        tbl = ax.table(
            cellText=cluster_map.values,
            colLabels=cluster_map.columns,
            loc="center",
        )
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
<h2>Cluster assignments (tickers starting at global first date)</h2>
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


def main():
    args = parse_args()

    device = pick_device()
    print("[INFO] Device:", device)
    if device.type == "cuda":
        print("[INFO] CUDA build:", torch.version.cuda)
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    df = load_prices(args.all_csv, args.data_dir)
    price_col = pick_price_col(df, args.price_col)
    keep = filter_tickers_start_at_global_min(df)

    if not keep:
        raise SystemExit("Не найдено тикеров, начинающихся с первой даты в данных.")

    X, companies, dates, scaled = build_matrix_full(df, price_col, args.min_rows, keep)

    print(f"[INFO] Using price column: {price_col}")
    print(f"[INFO] Global first date: {df['date'].min().date()}")
    print(f"[INFO] Tickers starting at first date: {len(companies)}")
    print(f"[INFO] Matrix: companies={len(companies)}, time_points={X.shape[1]}")

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

        Z, recon_mse = train_ae(
            X=X,
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

        objective_value, p_small, p_imb, p_tiny = calinski_objective_with_penalties(
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
        catch=(ValueError, np.linalg.LinAlgError),
    )

    print("\n=== BEST TRIAL ===")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    bp = study.best_trial.params

    Z_best, recon_mse_best = train_ae(
        X=X,
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
        Z=Z_best,
        algo=bp["algo"],
        trial=fixed_trial,
        seed=args.seed,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    metrics_best = cluster_best["metrics"]

    best_objective_value, p_small_best, p_imb_best, p_tiny_best = calinski_objective_with_penalties(
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
        "Z": Z_best,
        "labels": cluster_best["labels"],
        "metrics": metrics_best,
        "recon_mse": recon_mse_best,
        "best_params": bp,
        "best_objective_value": best_objective_value,
        "penalty_small_k": p_small_best,
        "penalty_imbalance": p_imb_best,
        "penalty_tiny_cluster": p_tiny_best,
    }

    print("\n=== ЛУЧШАЯ КОНФИГУРАЦИЯ ===")
    print(f"algo={best['algo']}, z={best['latent_dim']}")
    print(f"Objective value: {safe_float(best['best_objective_value']):.6f}")
    print(f"Calinski-Harabasz: {safe_float(best['metrics']['calinski']):.6f}")
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
    print(f"Noise ratio: {safe_float(best['metrics']['noise_ratio']):.6f}")

    cluster_map = (
        pd.DataFrame({"Company": companies, "Cluster": best["labels"]})
        .sort_values(["Cluster", "Company"])
        .reset_index(drop=True)
    )
    cluster_map.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved cluster assignments → {args.out_csv}")

    save_pdf(args.pdf_file, best, companies, scaled, price_col)
    print(f"[INFO] PDF report saved → {args.pdf_file}")

    info = {
        "Algorithm": best["algo"].upper(),
        "Latent dim": str(best["latent_dim"]),
        "Objective": "calinski_with_penalties",
        "Objective value": f"{safe_float(best['best_objective_value']):.6f}",
        "Calinski-Harabasz": f"{safe_float(best['metrics']['calinski']):.4f}",
        "Silhouette": f"{safe_float(best['metrics']['silhouette']):.4f}",
        "Davies-Bouldin": f"{safe_float(best['metrics']['davies']):.4f}",
        "Reconstruction MSE": f"{safe_float(best['recon_mse']):.6f}",
        "Clusters found": str(best["metrics"]["n_clusters_found"]),
        "Largest cluster size": str(best["metrics"]["largest_cluster_size"]),
        "Smallest cluster size": str(best["metrics"]["smallest_cluster_size"]),
        "Largest share": f"{safe_float(best['metrics']['largest_share']):.4f}",
        "Size ratio": f"{safe_float(best['metrics']['size_ratio']):.4f}",
        "Penalty small K": f"{safe_float(best['penalty_small_k']):.6f}",
        "Penalty imbalance": f"{safe_float(best['penalty_imbalance']):.6f}",
        "Penalty tiny cluster": f"{safe_float(best['penalty_tiny_cluster']):.6f}",
        "Noise ratio": f"{safe_float(best['metrics']['noise_ratio']):.4f}",
        "Penalty small clusters weight": str(args.penalty_small_clusters_weight),
        "Penalty imbalance weight": str(args.penalty_imbalance_weight),
        "Penalty smallest cluster weight": str(args.penalty_smallest_cluster_weight),
        "Target min clusters": str(args.target_min_clusters),
        "Max largest share without penalty": str(args.max_largest_share_without_penalty),
        "Max size ratio without penalty": str(args.max_size_ratio_without_penalty),
        "Min cluster size without penalty": str(args.min_cluster_size_without_penalty),
        "Price column": price_col,
        "Companies": str(len(companies)),
        "Time points": str(X.shape[1]),
        "Global first date": str(df["date"].min().date()),
        "Trials": str(len(study.trials)),
    }
    save_interactive_table_html(args.out_html, cluster_map, info)
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

    trials_df = pd.DataFrame(trials_rows).sort_values(
        by=["value"], ascending=False, na_position="last"
    )
    trials_df.to_csv(args.trials_csv, index=False)
    print(f"[INFO] Optuna trials saved → {args.trials_csv}")

    with open(args.best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": "calinski_with_penalties",
                "best_value": safe_float(study.best_value),
                "best_params": study.best_trial.params,
                "best_user_attrs": study.best_trial.user_attrs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Best params saved → {args.best_json}")

    print("\n=== TOP-10 TRIALS ===")
    cols_show = [
        "number",
        "state",
        "value",
        "algo",
        "latent_dim",
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