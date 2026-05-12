import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn

import optuna

from joblib import Parallel, delayed
from numba import njit, prange, set_num_threads, get_num_threads


USER_INPUT_PATH = Path("data/prices_all.csv")
USER_OUT_ROOT = Path("results_stocks")

USER_DATE_COL = "date"
USER_TICKER_COL = "Ticker"
USER_PRICE_COL = "Close"

USER_CPU_N_JOBS = 16
USER_NUMBA_NUM_THREADS = 16
USER_JOBLIB_BATCH_SIZE = 16

USER_PARALLEL_PREPARE_SERIES = True
USER_PARALLEL_BUILD_WINDOWS = True
USER_PARALLEL_FFT_BY_TICKER = True

# GPU / Torch
USER_USE_AMP = True
USER_AMP_DTYPE = "bf16"
USER_USE_TORCH_COMPILE = True
USER_RECON_BATCH_SIZE = 32768

# AE / Optuna
USER_WINDOW_SIZE = 252
USER_VAL_FRACTION = 0.2
USER_N_TRIALS = 300
USER_OPTUNA_N_JOBS = 1
USER_SEED = 42

# FFT
USER_MIN_PERIOD = 2
USER_MAX_PERIOD = 500
USER_TOP_K = 5

USER_MAX_WINDOWS = None

# Ветки анализа
USER_RUN_PRICE_BRANCH = True
USER_RUN_RETURNS_BRANCH = True
USER_RUN_AE_TRAIN = True
USER_RUN_AE_FFT = True
USER_RUN_RAW_FFT = True

USER_EXAMPLE_TICKERS = []


@dataclass
class Config:
    INPUT_PATH: Path = USER_INPUT_PATH
    OUT_ROOT: Path = USER_OUT_ROOT
    OUT_DIR: Path = field(init=False)

    DATE_COL: str = USER_DATE_COL
    TICKER_COL: str = USER_TICKER_COL
    PRICE_COL: str = USER_PRICE_COL

    DROP_UNNAMED_INDEX_COLS: bool = True

    WINDOW_SIZE: int = USER_WINDOW_SIZE
    VAL_FRACTION: float = USER_VAL_FRACTION
    N_TRIALS: int = USER_N_TRIALS
    OPTUNA_N_JOBS: int = USER_OPTUNA_N_JOBS
    SEED: int = USER_SEED

    MIN_PERIOD: int = USER_MIN_PERIOD
    MAX_PERIOD: int = USER_MAX_PERIOD
    TOP_K: int = USER_TOP_K

    EXAMPLE_TICKERS: List[str] = field(default_factory=lambda: list(USER_EXAMPLE_TICKERS))

    RUN_PRICE_BRANCH: bool = USER_RUN_PRICE_BRANCH
    RUN_RETURNS_BRANCH: bool = USER_RUN_RETURNS_BRANCH
    RUN_AE_TRAIN: bool = USER_RUN_AE_TRAIN
    RUN_AE_FFT: bool = USER_RUN_AE_FFT
    RUN_RAW_FFT: bool = USER_RUN_RAW_FFT

    USE_AMP: bool = USER_USE_AMP
    AMP_DTYPE: str = USER_AMP_DTYPE
    USE_TORCH_COMPILE: bool = USER_USE_TORCH_COMPILE
    RECON_BATCH_SIZE: int = USER_RECON_BATCH_SIZE

    MAX_WINDOWS: Optional[int] = USER_MAX_WINDOWS

    CPU_N_JOBS: int = USER_CPU_N_JOBS
    NUMBA_NUM_THREADS: int = USER_NUMBA_NUM_THREADS
    JOBLIB_BATCH_SIZE: int = USER_JOBLIB_BATCH_SIZE

    PARALLEL_PREPARE_SERIES: bool = USER_PARALLEL_PREPARE_SERIES
    PARALLEL_BUILD_WINDOWS: bool = USER_PARALLEL_BUILD_WINDOWS
    PARALLEL_FFT_BY_TICKER: bool = USER_PARALLEL_FFT_BY_TICKER

    PRICE_BEST_MODEL_PATH: Path = field(init=False)
    PRICE_FFT_SUMMARY_PATH: Path = field(init=False)
    PRICE_COMMON_PERIODS_PATH: Path = field(init=False)
    PRICE_SPECTRA_CSV_PATH: Path = field(init=False)
    PRICE_SPECTRA_PDF_PATH: Path = field(init=False)
    PRICE_TS_CSV_PATH: Path = field(init=False)
    PRICE_TS_PDF_PATH: Path = field(init=False)
    PRICE_BEST_PARAMS_PATH: Path = field(init=False)

    RETURNS_BEST_MODEL_PATH: Path = field(init=False)
    RETURNS_FFT_SUMMARY_PATH: Path = field(init=False)
    RETURNS_COMMON_PERIODS_PATH: Path = field(init=False)
    RETURNS_SPECTRA_CSV_PATH: Path = field(init=False)
    RETURNS_SPECTRA_PDF_PATH: Path = field(init=False)
    RETURNS_TS_CSV_PATH: Path = field(init=False)
    RETURNS_TS_PDF_PATH: Path = field(init=False)
    RETURNS_BEST_PARAMS_PATH: Path = field(init=False)

    def __post_init__(self):
        stem = Path(self.INPUT_PATH).stem
        safe = "".join(
            ch if ch.isalnum() or ch in (" ", "_", "-") else "_"
            for ch in stem
        ).strip()

        if not safe:
            safe = "run"

        self.OUT_DIR = self.OUT_ROOT / safe
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.PRICE_BEST_MODEL_PATH = self.OUT_DIR / "price_autoencoder_best.pt"
        self.PRICE_FFT_SUMMARY_PATH = self.OUT_DIR / "price_fft_extended_summary.xlsx"
        self.PRICE_COMMON_PERIODS_PATH = self.OUT_DIR / "price_common_periods.xlsx"
        self.PRICE_SPECTRA_CSV_PATH = self.OUT_DIR / "price_fft_spectra.csv"
        self.PRICE_SPECTRA_PDF_PATH = self.OUT_DIR / "price_fft_spectra.pdf"
        self.PRICE_TS_CSV_PATH = self.OUT_DIR / "price_ae_original_vs_recon_timeseries.csv"
        self.PRICE_TS_PDF_PATH = self.OUT_DIR / "price_ae_original_vs_recon_timeseries.pdf"
        self.PRICE_BEST_PARAMS_PATH = self.OUT_DIR / "price_best_params.json"

        self.RETURNS_BEST_MODEL_PATH = self.OUT_DIR / "returns_autoencoder_best.pt"
        self.RETURNS_FFT_SUMMARY_PATH = self.OUT_DIR / "returns_fft_extended_summary.xlsx"
        self.RETURNS_COMMON_PERIODS_PATH = self.OUT_DIR / "returns_common_periods.xlsx"
        self.RETURNS_SPECTRA_CSV_PATH = self.OUT_DIR / "returns_fft_spectra.csv"
        self.RETURNS_SPECTRA_PDF_PATH = self.OUT_DIR / "returns_fft_spectra.pdf"
        self.RETURNS_TS_CSV_PATH = self.OUT_DIR / "returns_ae_original_vs_recon_timeseries.csv"
        self.RETURNS_TS_PDF_PATH = self.OUT_DIR / "returns_ae_original_vs_recon_timeseries.pdf"
        self.RETURNS_BEST_PARAMS_PATH = self.OUT_DIR / "returns_best_params.json"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def setup_torch_for_gpu() -> None:
    if not torch.cuda.is_available():
        print("[TORCH] CUDA недоступна, AE будет считаться на CPU.")
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("[TORCH] CUDA:", torch.version.cuda)
    print("[TORCH] GPU:", torch.cuda.get_device_name(0))
    print("[TORCH] TF32 enabled")


def setup_cpu_numba(cfg: Config) -> None:
    n_threads = int(cfg.NUMBA_NUM_THREADS)

    if n_threads <= 0:
        n_threads = 1

    set_num_threads(n_threads)

    print(f"[CPU] os.cpu_count(): {os.cpu_count()}")
    print(f"[CPU] joblib n_jobs: {cfg.CPU_N_JOBS}")
    print(f"[CPU] numba threads: {get_num_threads()}")


def get_amp_dtype(cfg: Config):
    if not cfg.USE_AMP or not torch.cuda.is_available():
        return None

    if cfg.AMP_DTYPE.lower() == "bf16":
        return torch.bfloat16

    if cfg.AMP_DTYPE.lower() == "fp16":
        return torch.float16

    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def normalize_input_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    if cfg.DROP_UNNAMED_INDEX_COLS:
        unnamed_cols = [
            c for c in df.columns
            if str(c).lower().startswith("unnamed")
        ]

        if unnamed_cols:
            print(f"[LOAD] Удаляю индексные колонки: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols)

    if cfg.DATE_COL not in df.columns and "Date" in df.columns:
        print("[LOAD] Колонка date не найдена, использую Date.")
        df = df.rename(columns={"Date": cfg.DATE_COL})

    required = [cfg.DATE_COL, cfg.TICKER_COL, cfg.PRICE_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"[LOAD] Не найдены обязательные колонки: {missing}. "
            f"Фактические колонки: {list(df.columns)}"
        )

    return df


def load_stocks_long(cfg: Config) -> pd.DataFrame:
    print(f"[LOAD] Читаю файл: {cfg.INPUT_PATH}")
    print(f"[LOAD] Абсолютный путь: {cfg.INPUT_PATH.resolve()}")
    print(f"[LOAD] exists: {cfg.INPUT_PATH.exists()}")
    print(f"[LOAD] cwd: {Path.cwd()}")

    df = pd.read_csv(
        cfg.INPUT_PATH,
        sep=None,
        engine="python",
        encoding="utf-8",
    )

    df = normalize_input_columns(df, cfg)

    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce")
    df[cfg.PRICE_COL] = pd.to_numeric(df[cfg.PRICE_COL], errors="coerce")
    df[cfg.TICKER_COL] = df[cfg.TICKER_COL].astype(str)

    df = df.dropna(subset=[cfg.DATE_COL, cfg.TICKER_COL, cfg.PRICE_COL]).copy()
    df = df.sort_values([cfg.TICKER_COL, cfg.DATE_COL]).reset_index(drop=True)

    print(f"[LOAD] Строк: {len(df)}")
    print(f"[LOAD] Тикеров: {df[cfg.TICKER_COL].nunique()}")
    print(f"[LOAD] Дата min: {df[cfg.DATE_COL].min()}")
    print(f"[LOAD] Дата max: {df[cfg.DATE_COL].max()}")

    return df



def build_group_payloads(df: pd.DataFrame, cfg: Config) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    payloads = []

    for ticker, sub in df.groupby(cfg.TICKER_COL, sort=False):
        dates = sub[cfg.DATE_COL].to_numpy()
        prices = pd.to_numeric(sub[cfg.PRICE_COL], errors="coerce").to_numpy(dtype=np.float64)
        payloads.append((str(ticker), dates, prices))

    return payloads


def _prepare_one_ticker_payload(
    ticker: str,
    dates: np.ndarray,
    prices: np.ndarray,
    mode: str,
) -> Optional[Tuple[str, pd.Series]]:
    mask = np.isfinite(prices) & (prices > 0)

    if not np.any(mask):
        return None

    dates = dates[mask]
    prices = prices[mask]

    if len(prices) == 0:
        return None

    tmp = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "price": prices,
    })

    tmp = tmp.dropna(subset=["date", "price"])
    tmp = tmp.sort_values("date")
    tmp = tmp.drop_duplicates(subset=["date"], keep="last")

    if tmp.empty:
        return None

    price_series = pd.Series(
        tmp["price"].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(tmp["date"]),
        name="Close",
    )

    if mode == "price":
        x = np.log(price_series.astype(float))
    elif mode == "returns":
        x = np.log(price_series.astype(float)).diff().dropna()
    else:
        raise ValueError(f"Неизвестный mode={mode}")

    if x.empty:
        return None

    mu = x.mean()
    sigma = x.std(ddof=0)

    if not np.isfinite(sigma) or sigma == 0:
        s = pd.Series(np.zeros(len(x), dtype=np.float32), index=x.index)
    else:
        s = ((x - mu) / sigma).astype(np.float32)

    s.name = ticker
    s.index.name = "date"

    if s.empty:
        return None

    return ticker, s


def prepare_series_map_parallel(
    df: pd.DataFrame,
    cfg: Config,
    mode: str,
) -> Dict[str, pd.Series]:
    payloads = build_group_payloads(df, cfg)

    print(f"[PREP:{mode}] Подготовка рядов. Тикеров: {len(payloads)}")

    if cfg.PARALLEL_PREPARE_SERIES:
        results = Parallel(
            n_jobs=cfg.CPU_N_JOBS,
            backend="loky",
            batch_size=cfg.JOBLIB_BATCH_SIZE,
            verbose=5,
        )(
            delayed(_prepare_one_ticker_payload)(
                ticker=ticker,
                dates=dates,
                prices=prices,
                mode=mode,
            )
            for ticker, dates, prices in payloads
        )
    else:
        results = [
            _prepare_one_ticker_payload(ticker, dates, prices, mode)
            for ticker, dates, prices in payloads
        ]

    series_map: Dict[str, pd.Series] = {}

    for item in results:
        if item is None:
            continue

        ticker, s = item
        series_map[ticker] = s

    print(f"[PREP:{mode}] Готовых рядов: {len(series_map)}")

    return series_map



def _build_windows_one_series(
    x: np.ndarray,
    window_size: int,
) -> Optional[np.ndarray]:
    if len(x) < window_size:
        return None

    w = np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size)
    return np.ascontiguousarray(w, dtype=np.float32)


def build_windows_from_series_list(
    series_list: List[np.ndarray],
    window_size: int,
    max_windows: Optional[int] = None,
    seed: int = 42,
    n_jobs: int = 1,
    joblib_batch_size: int = 8,
) -> np.ndarray:
    if n_jobs > 1:
        parts = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size=joblib_batch_size,
            verbose=5,
        )(
            delayed(_build_windows_one_series)(x, window_size)
            for x in series_list
        )
    else:
        parts = [
            _build_windows_one_series(x, window_size)
            for x in series_list
        ]

    parts = [p for p in parts if p is not None]

    if not parts:
        raise RuntimeError("[AE] Не удалось собрать ни одного окна.")

    X = np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    if max_windows is not None and X.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_windows, replace=False)
        idx.sort()
        X = X[idx]

    return X


def prepare_branch_windows(
    df: pd.DataFrame,
    cfg: Config,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, pd.Series]]:
    series_map = prepare_series_map_parallel(df, cfg, mode=mode)
    series_list = [s.values.astype(np.float32) for s in series_map.values()]

    n_jobs = cfg.CPU_N_JOBS if cfg.PARALLEL_BUILD_WINDOWS else 1

    X = build_windows_from_series_list(
        series_list=series_list,
        window_size=cfg.WINDOW_SIZE,
        max_windows=cfg.MAX_WINDOWS,
        seed=cfg.SEED,
        n_jobs=n_jobs,
        joblib_batch_size=cfg.JOBLIB_BATCH_SIZE,
    )

    print(f"[PREP:{mode}] Всего рядов: {len(series_map)}")
    print(f"[PREP:{mode}] Всего окон: {X.shape[0]}")
    print(f"[PREP:{mode}] Длина окна: {cfg.WINDOW_SIZE}")
    print(f"[PREP:{mode}] X memory: {X.nbytes / 1024 ** 2:.2f} MB")

    return X, series_map


def train_val_split(X: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    n_val = int(n * val_fraction)
    n_train = n - n_val

    if n_train <= 0 or n_val <= 0:
        raise RuntimeError(
            f"[AE] Некорректное разбиение train/val: n={n}, val_fraction={val_fraction}"
        )

    return X[:n_train], X[n_train:]



@njit(cache=True)
def _estimate_prominence_simple(power: np.ndarray, idx: int) -> float:
    n = power.shape[0]
    peak = power[idx]

    left_min = peak

    for i in range(idx - 1, -1, -1):
        if power[i] > peak:
            break

        if power[i] < left_min:
            left_min = power[i]

    right_min = peak

    for i in range(idx + 1, n):
        if power[i] > peak:
            break

        if power[i] < right_min:
            right_min = power[i]

    base = left_min if left_min > right_min else right_min
    prom = peak - base

    if prom < 0:
        prom = 0.0

    return prom


@njit(cache=True)
def _top_peaks_numba(
    periods: np.ndarray,
    power: np.ndarray,
    top_k: int,
):
    n = power.shape[0]

    peak_idx = np.empty(n, dtype=np.int64)
    peak_power = np.empty(n, dtype=np.float64)
    peak_prom = np.empty(n, dtype=np.float64)

    count = 0

    for i in range(1, n - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            peak_idx[count] = i
            peak_power[count] = power[i]
            peak_prom[count] = _estimate_prominence_simple(power, i)
            count += 1

    if count == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    k = min(top_k, count)

    out_periods = np.empty(k, dtype=np.float64)
    out_powers = np.empty(k, dtype=np.float64)
    out_idx = np.empty(k, dtype=np.int64)
    out_prom = np.empty(k, dtype=np.float64)

    selected = np.zeros(count, dtype=np.bool_)

    for r in range(k):
        best_j = -1
        best_power = -1.0

        for j in range(count):
            if not selected[j] and peak_power[j] > best_power:
                best_power = peak_power[j]
                best_j = j

        selected[best_j] = True

        idx = peak_idx[best_j]
        out_periods[r] = periods[idx]
        out_powers[r] = power[idx]
        out_idx[r] = idx
        out_prom[r] = peak_prom[best_j]

    return out_periods, out_powers, out_idx, out_prom


@njit(parallel=True, cache=True)
def _overlap_average_from_window_outputs(
    outputs: np.ndarray,
    original_len: int,
    window_size: int,
) -> np.ndarray:
    n_windows = outputs.shape[0]
    recon = np.empty(original_len, dtype=np.float32)

    for t in prange(original_len):
        j_min = t - window_size + 1

        if j_min < 0:
            j_min = 0

        j_max = t

        if j_max > n_windows - 1:
            j_max = n_windows - 1

        s = 0.0
        c = 0

        for j in range(j_min, j_max + 1):
            local_pos = t - j
            s += outputs[j, local_pos]
            c += 1

        if c > 0:
            recon[t] = s / c
        else:
            recon[t] = 0.0

    return recon



def period_band(period: float) -> str:
    if not np.isfinite(period):
        return "unknown"

    if period <= 7:
        return "very_short"

    if period <= 14:
        return "weekly"

    if period <= 35:
        return "monthly"

    if period <= 80:
        return "quarterly"

    if period <= 140:
        return "semiannual"

    if period <= 280:
        return "annual"

    return "long_cycle"


def compute_periodogram(
    series: np.ndarray,
    min_period: Optional[float],
    max_period: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    x = series.astype(np.float64)
    x = x[np.isfinite(x)]

    n = len(x)

    if n < 10:
        return None, None

    x = x - np.mean(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = (np.abs(X) ** 2) / n

    mask = freqs > 0

    if max_period is not None:
        mask &= freqs >= 1.0 / max_period

    if min_period is not None:
        mask &= freqs <= 1.0 / min_period

    freqs = freqs[mask]
    power = power[mask]

    if len(freqs) == 0:
        return None, None

    periods = 1.0 / freqs

    return periods, power


def find_dominant_periods(
    periods: np.ndarray,
    power: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    periods = np.asarray(periods, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)

    out_periods, out_powers, out_idx, out_prom = _top_peaks_numba(
        periods,
        power,
        int(top_k),
    )

    if len(out_periods) == 0:
        return pd.DataFrame(columns=[
            "period",
            "power",
            "rank_by_power",
            "fft_index",
            "prominence",
            "period_band",
        ])

    df = pd.DataFrame({
        "period": out_periods,
        "power": out_powers,
        "rank_by_power": np.arange(1, len(out_periods) + 1, dtype=int),
        "fft_index": out_idx,
        "prominence": out_prom,
    })

    df["period_band"] = df["period"].apply(period_band)

    return df



class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()

        enc_layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            enc_layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ]

            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = latent_dim

        for _ in range(n_layers):
            dec_layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ]

            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def maybe_compile_model(model: nn.Module, cfg: Config) -> nn.Module:
    if not cfg.USE_TORCH_COMPILE:
        return model

    if not torch.cuda.is_available():
        return model

    if not hasattr(torch, "compile"):
        return model

    try:
        print("[TORCH] torch.compile enabled")
        return torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"[TORCH] torch.compile недоступен: {e}")
        return model



def to_gpu_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).to(DEVICE, non_blocking=True)


def iterate_gpu_batches(
    X: torch.Tensor,
    batch_size: int,
    shuffle: bool,
):
    n = X.shape[0]

    if shuffle:
        idx = torch.randperm(n, device=X.device)
    else:
        idx = None

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        if idx is None:
            xb = X[start:end]
        else:
            xb = X[idx[start:end]]

        yield xb, xb


def train_one_epoch_gpu_tensor(
    model: nn.Module,
    X_train: torch.Tensor,
    optimizer,
    criterion,
    batch_size: int,
    amp_dtype,
) -> float:
    model.train()

    total = 0.0
    n_batches = 0

    for xb, yb in iterate_gpu_batches(
        X_train,
        batch_size=batch_size,
        shuffle=True,
    ):
        optimizer.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(xb)
                loss = criterion(preds, yb)
        else:
            preds = model(xb)
            loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        n_batches += 1

    return total / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch_gpu_tensor(
    model: nn.Module,
    X_val: torch.Tensor,
    criterion,
    batch_size: int,
    amp_dtype,
) -> float:
    model.eval()

    total = 0.0
    n_batches = 0

    for xb, yb in iterate_gpu_batches(
        X_val,
        batch_size=batch_size,
        shuffle=False,
    ):
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(xb)
                loss = criterion(preds, yb)
        else:
            preds = model(xb)
            loss = criterion(preds, yb)

        total += float(loss.detach().cpu())
        n_batches += 1

    return total / max(1, n_batches)


def optuna_objective(
    trial,
    X_train_np: np.ndarray,
    X_val_np: np.ndarray,
    cfg: Config,
) -> float:
    input_dim = X_train_np.shape[1]
    amp_dtype = get_amp_dtype(cfg)

    latent_dim = trial.suggest_int("latent_dim", 8, 128, step=8)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 1024, step=128)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.35)
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)

    batch_size = trial.suggest_categorical(
        "batch_size",
        [1024, 2048, 4096, 8192, 16384, 32768],
    )

    n_epochs = trial.suggest_int("n_epochs", 20, 80)

    X_train = to_gpu_tensor(X_train_np)
    X_val = to_gpu_tensor(X_val_np)

    raw_model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(DEVICE)

    model = maybe_compile_model(raw_model, cfg)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 10
    bad_epochs = 0

    for epoch in range(n_epochs):
        _ = train_one_epoch_gpu_tensor(
            model=model,
            X_train=X_train,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )

        val_loss = eval_one_epoch_gpu_tensor(
            model=model,
            X_val=X_val,
            criterion=criterion,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    del model, raw_model, optimizer, X_train, X_val

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val


def train_autoencoder_with_optuna(
    X: np.ndarray,
    model_path: Path,
    params_path: Path,
    cfg: Config,
    branch_name: str,
) -> Dict[str, Any]:
    print(f"\n=== [AE:{branch_name}] Optuna обучение автоэнкодера ===")

    set_seed(cfg.SEED)

    X_train, X_val = train_val_split(X, cfg.VAL_FRACTION)

    print(f"[AE:{branch_name}] Train windows: {X_train.shape[0]}")
    print(f"[AE:{branch_name}] Val windows: {X_val.shape[0]}")
    print(f"[AE:{branch_name}] Window size: {X_train.shape[1]}")

    sampler = optuna.samplers.TPESampler(seed=cfg.SEED)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=8,
        interval_steps=2,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_val, cfg),
        n_trials=cfg.N_TRIALS,
        n_jobs=cfg.OPTUNA_N_JOBS,
        show_progress_bar=True,
    )

    best = study.best_trial

    print(f"[AE:{branch_name}] Best trial:")
    print(f"  val MSE: {best.value}")
    print(f"  params: {best.params}")

    bp = best.params
    input_dim = X_train.shape[1]
    amp_dtype = get_amp_dtype(cfg)

    X_train_gpu = to_gpu_tensor(X_train)
    X_val_gpu = to_gpu_tensor(X_val)

    raw_model = Autoencoder(
        input_dim=input_dim,
        latent_dim=bp["latent_dim"],
        hidden_dim=bp["hidden_dim"],
        n_layers=bp["n_layers"],
        dropout=bp["dropout"],
    ).to(DEVICE)

    model = maybe_compile_model(raw_model, cfg)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=bp["lr"],
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(bp["n_epochs"]):
        tr = train_one_epoch_gpu_tensor(
            model=model,
            X_train=X_train_gpu,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=bp["batch_size"],
            amp_dtype=amp_dtype,
        )

        vl = eval_one_epoch_gpu_tensor(
            model=model,
            X_val=X_val_gpu,
            criterion=criterion,
            batch_size=bp["batch_size"],
            amp_dtype=amp_dtype,
        )

        print(
            f"[AE:{branch_name} FINAL] "
            f"epoch {epoch + 1}/{bp['n_epochs']} "
            f"train={tr:.6f} val={vl:.6f}"
        )

        if vl < best_val:
            best_val = vl
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in raw_model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"[AE:{branch_name}] Не удалось сохранить best_state.")

    torch.save(best_state, model_path)

    payload = {
        "best_value": float(best.value),
        "final_best_val": float(best_val),
        **best.params,
        "window_size": int(cfg.WINDOW_SIZE),
        "amp": bool(cfg.USE_AMP),
        "amp_dtype": cfg.AMP_DTYPE,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[AE:{branch_name}] Лучшая модель сохранена: {model_path}")
    print(f"[AE:{branch_name}] Лучшие параметры сохранены: {params_path}")

    del model, raw_model, optimizer, X_train_gpu, X_val_gpu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_value": best.value,
        "best_params": dict(best.params),
    }


def load_trained_model(
    model_path: Path,
    input_dim: int,
    params: Dict[str, Any],
) -> Autoencoder:
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=int(params["latent_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        n_layers=int(params["n_layers"]),
        dropout=float(params["dropout"]),
    ).to(DEVICE)

    print(f"[LOAD MODEL] Гружу веса: {model_path}")

    state = torch.load(model_path, map_location=DEVICE)

    cleaned_state = {}

    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned_state[k.replace("_orig_mod.", "", 1)] = v
        else:
            cleaned_state[k] = v

    model.load_state_dict(cleaned_state)
    model.eval()

    return model


@torch.no_grad()
def reconstruct_series_with_ae(
    model: Autoencoder,
    x: np.ndarray,
    window_size: int,
    batch_size: int,
    cfg: Config,
) -> np.ndarray:
    x = x.astype(np.float32)
    n = len(x)

    if n < window_size:
        return x.copy()

    windows = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=window_size,
    )

    windows = np.ascontiguousarray(windows, dtype=np.float32)
    outputs = np.empty_like(windows, dtype=np.float32)

    amp_dtype = get_amp_dtype(cfg)
    model.eval()

    for start in range(0, windows.shape[0], batch_size):
        end = min(start + batch_size, windows.shape[0])

        xb = torch.from_numpy(windows[start:end]).to(
            DEVICE,
            non_blocking=True,
        )

        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(xb)
        else:
            out = model(xb)

        outputs[start:end] = out.detach().float().cpu().numpy()

    recon = _overlap_average_from_window_outputs(
        outputs=outputs,
        original_len=n,
        window_size=window_size,
    )

    return recon



def add_fft_rows_for_series(
    rows: List[Dict[str, Any]],
    ticker: str,
    branch_name: str,
    source_name: str,
    x: np.ndarray,
    cfg: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    periods, power = compute_periodogram(
        x,
        cfg.MIN_PERIOD,
        cfg.MAX_PERIOD,
    )

    if periods is None or power is None:
        return None, None

    dom = find_dominant_periods(
        periods,
        power,
        cfg.TOP_K,
    )

    total_power = float(np.nansum(power))

    if not np.isfinite(total_power) or total_power <= 0:
        total_power = np.nan

    if (not dom.empty) and np.isfinite(total_power) and total_power > 0:
        signal_power = float(np.nansum(dom["power"].to_numpy(dtype=float)))
        noise_power = float(total_power - signal_power)
        signal_share = signal_power / total_power
        noise_share = noise_power / total_power
    else:
        signal_power = np.nan
        noise_power = np.nan
        signal_share = np.nan
        noise_share = np.nan

    if dom.empty:
        rows.append({
            "ticker": ticker,
            "mode": branch_name,
            "source": source_name,
            "rank": np.nan,
            "period_days": np.nan,
            "period_band": "unknown",
            "peak_power": np.nan,
            "prominence": np.nan,
            "n_points": int(len(x)),
            "total_power_band": float(total_power) if np.isfinite(total_power) else np.nan,
            "signal_power_topk": float(signal_power) if np.isfinite(signal_power) else np.nan,
            "noise_power_band": float(noise_power) if np.isfinite(noise_power) else np.nan,
            "signal_share": float(signal_share) if np.isfinite(signal_share) else np.nan,
            "noise_share": float(noise_share) if np.isfinite(noise_share) else np.nan,
        })
    else:
        for _, r in dom.iterrows():
            rows.append({
                "ticker": ticker,
                "mode": branch_name,
                "source": source_name,
                "rank": int(r["rank_by_power"]),
                "period_days": float(r["period"]),
                "period_band": str(r["period_band"]),
                "peak_power": float(r["power"]),
                "prominence": float(r["prominence"]),
                "n_points": int(len(x)),
                "total_power_band": float(total_power) if np.isfinite(total_power) else np.nan,
                "signal_power_topk": float(signal_power) if np.isfinite(signal_power) else np.nan,
                "noise_power_band": float(noise_power) if np.isfinite(noise_power) else np.nan,
                "signal_share": float(signal_share) if np.isfinite(signal_share) else np.nan,
                "noise_share": float(noise_share) if np.isfinite(noise_share) else np.nan,
            })

    return periods, power


def _fft_from_ready_arrays_job(
    ticker: str,
    branch_name: str,
    x_vals: np.ndarray,
    x_recon: np.ndarray,
    min_period: int,
    max_period: int,
    top_k: int,
    run_raw_fft: bool,
):
    class SimpleCfg:
        pass

    cfg = SimpleCfg()
    cfg.MIN_PERIOD = min_period
    cfg.MAX_PERIOD = max_period
    cfg.TOP_K = top_k
    cfg.RUN_RAW_FFT = run_raw_fft

    rows = []

    raw_spectrum = None
    ae_spectrum = None

    if run_raw_fft:
        raw_periods, raw_power = add_fft_rows_for_series(
            rows=rows,
            ticker=ticker,
            branch_name=branch_name,
            source_name="raw",
            x=x_vals,
            cfg=cfg,
        )

        if raw_periods is not None and raw_power is not None:
            raw_spectrum = (raw_periods, raw_power)

    ae_periods, ae_power = add_fft_rows_for_series(
        rows=rows,
        ticker=ticker,
        branch_name=branch_name,
        source_name="ae_recon",
        x=x_recon,
        cfg=cfg,
    )

    if ae_periods is not None and ae_power is not None:
        ae_spectrum = (ae_periods, ae_power)

    return {
        "ticker": ticker,
        "summary_rows": rows,
        "raw_spectrum": raw_spectrum,
        "ae_spectrum": ae_spectrum,
    }


def finalize_fft_summary(fft_summary: pd.DataFrame) -> pd.DataFrame:
    if fft_summary.empty:
        fft_summary["peak_share"] = pd.Series(dtype=float)
        fft_summary["norm_peak_share"] = pd.Series(dtype=float)
        return fft_summary

    fft_summary["peak_share"] = (
        pd.to_numeric(fft_summary["peak_power"], errors="coerce")
        / pd.to_numeric(fft_summary["total_power_band"], errors="coerce")
    )

    fft_summary.loc[
        ~np.isfinite(fft_summary["peak_share"]),
        "peak_share",
    ] = np.nan

    ps = pd.to_numeric(fft_summary["peak_share"], errors="coerce")
    ps_min = ps.min(skipna=True)
    ps_max = ps.max(skipna=True)
    denom = ps_max - ps_min

    if np.isfinite(denom) and denom > 0:
        fft_summary["norm_peak_share"] = (ps - ps_min) / denom
    else:
        fft_summary["norm_peak_share"] = np.nan

    return fft_summary


def build_common_periods_table(fft_summary: pd.DataFrame) -> pd.DataFrame:
    if fft_summary.empty:
        return pd.DataFrame()

    df = fft_summary.copy()
    df = df.dropna(subset=["period_days", "peak_share"])

    if df.empty:
        return pd.DataFrame()

    df["period_rounded"] = df["period_days"].round().astype(int)

    common = (
        df.groupby(
            ["mode", "source", "period_rounded", "period_band"],
            as_index=False,
        )
        .agg(
            tickers_count=("ticker", "nunique"),
            mean_peak_share=("peak_share", "mean"),
            median_peak_share=("peak_share", "median"),
            mean_prominence=("prominence", "mean"),
            median_signal_share=("signal_share", "median"),
        )
        .sort_values(
            ["source", "tickers_count", "mean_peak_share"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

    return common


def run_fft_pipeline_for_branch(
    series_map: Dict[str, pd.Series],
    cfg: Config,
    best_params: Dict[str, Any],
    model_path: Path,
    summary_path: Path,
    common_periods_path: Path,
    spectra_csv_path: Path,
    spectra_pdf_path: Path,
    ts_csv_path: Path,
    ts_pdf_path: Path,
    branch_name: str,
) -> None:
    print(f"\n=== [FFT:{branch_name}] AE-реконструкция на GPU + FFT на CPU ===")

    model = load_trained_model(
        model_path=model_path,
        input_dim=cfg.WINDOW_SIZE,
        params=best_params,
    )

    print(f"[FFT:{branch_name}] Фаза 1: AE-реконструкция на GPU без параллелизма")

    reconstructed_map: Dict[str, Dict[str, Any]] = {}

    for j, (ticker, s) in enumerate(series_map.items(), start=1):
        x_vals = s.values.astype(np.float32)

        x_recon = reconstruct_series_with_ae(
            model=model,
            x=x_vals,
            window_size=cfg.WINDOW_SIZE,
            batch_size=cfg.RECON_BATCH_SIZE,
            cfg=cfg,
        )

        reconstructed_map[ticker] = {
            "dates": s.index.to_pydatetime(),
            "orig": x_vals,
            "recon": x_recon,
        }

        if j % 25 == 0 or j == len(series_map):
            print(
                f"[FFT:{branch_name}] AE-реконструкция: "
                f"{j}/{len(series_map)}"
            )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[FFT:{branch_name}] Фаза 2: FFT + пики на CPU")

    items = list(reconstructed_map.items())

    if cfg.PARALLEL_FFT_BY_TICKER:
        fft_results = Parallel(
            n_jobs=cfg.CPU_N_JOBS,
            backend="loky",
            batch_size=cfg.JOBLIB_BATCH_SIZE,
            verbose=5,
        )(
            delayed(_fft_from_ready_arrays_job)(
                ticker=ticker,
                branch_name=branch_name,
                x_vals=data["orig"],
                x_recon=data["recon"],
                min_period=cfg.MIN_PERIOD,
                max_period=cfg.MAX_PERIOD,
                top_k=cfg.TOP_K,
                run_raw_fft=cfg.RUN_RAW_FFT,
            )
            for ticker, data in items
        )
    else:
        fft_results = [
            _fft_from_ready_arrays_job(
                ticker=ticker,
                branch_name=branch_name,
                x_vals=data["orig"],
                x_recon=data["recon"],
                min_period=cfg.MIN_PERIOD,
                max_period=cfg.MAX_PERIOD,
                top_k=cfg.TOP_K,
                run_raw_fft=cfg.RUN_RAW_FFT,
            )
            for ticker, data in items
        ]

    spectra_for_plots: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    ts_for_plots: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for res in fft_results:
        ticker = res["ticker"]
        data = reconstructed_map[ticker]

        summary_rows.extend(res["summary_rows"])

        need_plot = (not cfg.EXAMPLE_TICKERS) or (ticker in cfg.EXAMPLE_TICKERS)

        if need_plot:
            ts_for_plots[ticker] = (
                data["dates"],
                data["orig"],
                data["recon"],
            )

            if res["raw_spectrum"] is not None:
                spectra_for_plots[(ticker, "raw")] = res["raw_spectrum"]

            if res["ae_spectrum"] is not None:
                spectra_for_plots[(ticker, "ae_recon")] = res["ae_spectrum"]

    fft_summary = pd.DataFrame(summary_rows)
    fft_summary = finalize_fft_summary(fft_summary)
    common_periods = build_common_periods_table(fft_summary)

    with pd.ExcelWriter(summary_path, engine="xlsxwriter") as writer:
        fft_summary.to_excel(
            writer,
            sheet_name=f"fft_{branch_name}",
            index=False,
        )

    with pd.ExcelWriter(common_periods_path, engine="xlsxwriter") as writer:
        common_periods.to_excel(
            writer,
            sheet_name=f"common_{branch_name}",
            index=False,
        )

    print(f"[FFT:{branch_name}] Summary сохранён: {summary_path}")
    print(f"[FFT:{branch_name}] Common periods сохранён: {common_periods_path}")

    if spectra_for_plots:
        rows = []

        for (ticker, source), (periods, power) in spectra_for_plots.items():
            pmax = power.max()
            power_norm = power / pmax if pmax > 0 else power

            for p, pw, pwn in zip(periods, power, power_norm):
                rows.append({
                    "ticker": ticker,
                    "mode": branch_name,
                    "source": source,
                    "period_days": float(p),
                    "power": float(pw),
                    "power_norm": float(pwn),
                })

        pd.DataFrame(rows).to_csv(spectra_csv_path, index=False)
        print(f"[FFT:{branch_name}] Спектры CSV: {spectra_csv_path}")

        with PdfPages(spectra_pdf_path) as pdf:
            for (ticker, source), (periods, power) in spectra_for_plots.items():
                plt.figure(figsize=(8, 4))

                power_plot = power / power.max() if power.max() > 0 else power

                plt.plot(periods, power_plot)
                plt.xlabel("Период, наблюдений")
                plt.ylabel("Нормированная мощность")
                plt.title(
                    f"FFT spectrum, {branch_name}, {source}, ticker: {ticker}"
                )
                plt.xscale("log")
                plt.grid(True, which="both", alpha=0.3)

                pdf.savefig()
                plt.close()

        print(f"[FFT:{branch_name}] Спектры PDF: {spectra_pdf_path}")

    if ts_for_plots:
        rows = []

        value_col_orig = f"orig_{branch_name}"
        value_col_recon = f"recon_{branch_name}"

        for ticker, (dates, orig, recon) in ts_for_plots.items():
            for d, o, r in zip(dates, orig, recon):
                rows.append({
                    "ticker": ticker,
                    "mode": branch_name,
                    "date": d,
                    value_col_orig: float(o),
                    value_col_recon: float(r),
                })

        pd.DataFrame(rows).to_csv(ts_csv_path, index=False)
        print(f"[FFT:{branch_name}] Временные ряды CSV: {ts_csv_path}")

        with PdfPages(ts_pdf_path) as pdf:
            for ticker, (dates, orig, recon) in ts_for_plots.items():
                plt.figure(figsize=(10, 4))

                plt.plot(
                    dates,
                    orig,
                    label=f"Original {branch_name}",
                    alpha=0.7,
                )

                plt.plot(
                    dates,
                    recon,
                    label=f"AE reconstruction {branch_name}",
                    alpha=0.7,
                )

                plt.xlabel("Дата")
                plt.ylabel("Нормированное значение")
                plt.title(
                    f"Original vs AE, {branch_name}, ticker: {ticker}"
                )
                plt.grid(True, alpha=0.3)
                plt.legend()

                pdf.savefig()
                plt.close()

        print(f"[FFT:{branch_name}] Временные ряды PDF: {ts_pdf_path}")



def run_price_branch(df: pd.DataFrame, cfg: Config) -> None:
    print("\n" + "=" * 70)
    print("[BRANCH] PRICE")
    print("=" * 70)

    X_price, series_map_price = prepare_branch_windows(
        df,
        cfg,
        mode="price",
    )

    best_params = None

    if cfg.RUN_AE_TRAIN:
        result = train_autoencoder_with_optuna(
            X=X_price,
            model_path=cfg.PRICE_BEST_MODEL_PATH,
            params_path=cfg.PRICE_BEST_PARAMS_PATH,
            cfg=cfg,
            branch_name="price",
        )

        best_params = result["best_params"]

    if cfg.RUN_AE_FFT:
        if best_params is None:
            raise RuntimeError("[FFT:price] Не найдены best_params.")

        run_fft_pipeline_for_branch(
            series_map=series_map_price,
            cfg=cfg,
            best_params=best_params,
            model_path=cfg.PRICE_BEST_MODEL_PATH,
            summary_path=cfg.PRICE_FFT_SUMMARY_PATH,
            common_periods_path=cfg.PRICE_COMMON_PERIODS_PATH,
            spectra_csv_path=cfg.PRICE_SPECTRA_CSV_PATH,
            spectra_pdf_path=cfg.PRICE_SPECTRA_PDF_PATH,
            ts_csv_path=cfg.PRICE_TS_CSV_PATH,
            ts_pdf_path=cfg.PRICE_TS_PDF_PATH,
            branch_name="price",
        )


def run_returns_branch(df: pd.DataFrame, cfg: Config) -> None:
    print("\n" + "=" * 70)
    print("[BRANCH] RETURNS")
    print("=" * 70)

    X_returns, series_map_returns = prepare_branch_windows(
        df,
        cfg,
        mode="returns",
    )

    best_params = None

    if cfg.RUN_AE_TRAIN:
        result = train_autoencoder_with_optuna(
            X=X_returns,
            model_path=cfg.RETURNS_BEST_MODEL_PATH,
            params_path=cfg.RETURNS_BEST_PARAMS_PATH,
            cfg=cfg,
            branch_name="returns",
        )

        best_params = result["best_params"]

    if cfg.RUN_AE_FFT:
        if best_params is None:
            raise RuntimeError("[FFT:returns] Не найдены best_params.")

        run_fft_pipeline_for_branch(
            series_map=series_map_returns,
            cfg=cfg,
            best_params=best_params,
            model_path=cfg.RETURNS_BEST_MODEL_PATH,
            summary_path=cfg.RETURNS_FFT_SUMMARY_PATH,
            common_periods_path=cfg.RETURNS_COMMON_PERIODS_PATH,
            spectra_csv_path=cfg.RETURNS_SPECTRA_CSV_PATH,
            spectra_pdf_path=cfg.RETURNS_SPECTRA_PDF_PATH,
            ts_csv_path=cfg.RETURNS_TS_CSV_PATH,
            ts_pdf_path=cfg.RETURNS_TS_PDF_PATH,
            branch_name="returns",
        )


# ============================================================
# RUN
# ============================================================

def run_one_file():
    cfg = Config()

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_torch_for_gpu()
    setup_cpu_numba(cfg)

    print("\n==============================")
    print(f"[RUN] FILE: {cfg.INPUT_PATH}")
    print(f"[RUN] OUT : {cfg.OUT_DIR}")
    print(f"[RUN] DEVICE: {DEVICE}")

    if torch.cuda.is_available():
        print(f"[RUN] GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[RUN] GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
        )

    print("==============================\n")

    set_seed(cfg.SEED)

    df = load_stocks_long(cfg)

    if cfg.RUN_PRICE_BRANCH:
        run_price_branch(df, cfg)

    if cfg.RUN_RETURNS_BRANCH:
        run_returns_branch(df, cfg)

    print(f"\n[DONE] {cfg.INPUT_PATH} -> {cfg.OUT_DIR}")


def main():
    run_one_file()


if __name__ == "__main__":
    main()
