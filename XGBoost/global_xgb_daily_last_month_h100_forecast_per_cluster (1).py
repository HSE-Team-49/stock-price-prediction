from __future__ import annotations

"""
global_xgb_daily_h100_forecast_last_month_two_models.py

Clean daily XGBoost-модель для прогноза последнего месяца по дням. Версия с облегчёнными market/cluster/macro-interaction признаками для устойчивости по RAM.

Текущая постановка:
- История до 2025-12-20 лежит в основном файле prices_all.csv.
- Факт после 2025-12-20 лежит в отдельном файле prices_all_2025_12_20_today.csv.
- Файл после 2025-12-20 используется для построения полного panel и для проверки факта.
- Последний месяц до 2026-05-08 включительно используется только для расчёта метрик:
  WAPE, MAPE, MSE, MAE, RMSE.
- Модель обучается только на данных до начала тестового окна с запасом forecast_horizon,
  чтобы target_h не залезал в период оценки.
- Прогноз direct multi-horizon:
  target_h = log(price[t+h]) - log(price[t]), h = 1..21.
- XGBoost использует GPU/H100 через device=cuda и tree_method=hist.
- Optuna подбирает параметры.
- CPU-часть частично распараллелена через joblib и numba.

Пример запуска:

python3 global_xgb_daily_h100_forecast_last_month_two_models.py \
  --prices-csv data/prices_all.csv \
  --future-prices-csv data/prices_all_2025_12_20_today.csv \
  --eval-end-date 2026-05-08 \
  --eval-months 1 \
  --macro-dir data \
  --cluster-csv cluster_fullstart_assignments.csv \
  --season-dir results_stocks/prices_all \
  --out-root results_xgb_daily_h100 \
  --forecast-horizon 21 \
  --n-trials 300 \
  --n-jobs 16 \
  --numba-threads 16 \
  --use-gpu 1 \
  --gpu-id 0
"""

import os

# Важно задавать до тяжёлых импортов, чтобы не было размножения потоков.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import json
import time
import math
import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from joblib import Parallel, delayed
from numba import njit, set_num_threads, get_num_threads

import optuna
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# CONSTANTS
# ============================================================

MACRO_FILES = {
    "CPIAUCSL.csv": "cpi",
    "DCOILBRENTEU.csv": "brent",
    "DEXUSEU.csv": "usd_eur",
    "DGS2.csv": "dgs2",
    "DGS10.csv": "dgs10",
    "DGS30.csv": "dgs30",
    "EFFR.csv": "effr",
    "M2SL.csv": "m2",
    "UNRATE.csv": "unrate",
}

MONTHLY_MACRO = {"cpi", "m2", "unrate"}

RET_LAGS = [1, 2, 3, 5, 10, 21, 42, 63]
PRICE_LAGS = [1, 5, 10, 21, 42, 63, 126, 252]
ROLLS = [5, 10, 21, 42, 63, 126, 252]
ATR_WINDOWS = [10, 21, 63]
RSI_PERIODS = [6, 12, 14, 21]
VOL_WINDOWS = [5, 21, 63, 126]
BETA_WINDOWS = [21, 63, 126, 252]
MACRO_LAGS = [1, 5, 21, 63, 126, 252]
MACRO_ROLLS = [5, 21, 63, 126, 252]


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    prices_csv: str
    macro_dir: str
    cluster_csv: Optional[str]
    season_dir: Optional[str]
    out_root: str

    # Дополнительный файл с фактическими ценами после 2025-12-20.
    future_prices_csv: Optional[str] = None

    # Последние N месяцев до eval_end_date включительно — только для метрик.
    eval_end_date: str = "2026-05-08"
    eval_months: int = 1
    macro_known_until: str = "2025-12-20"

    # Горизонты 1..short_horizon_days используют макро.
    # Горизонты short_horizon_days+1..forecast_horizon обучаются без макро.
    short_horizon_days: int = 5

    date_col: str = "date"
    ticker_col: str = "Ticker"
    price_col: str = "auto"

    forecast_horizon: int = 21
    val_days: int = 252
    min_rows_per_ticker: int = 300
    filter_global_first_date: bool = False

    n_trials: int = 80
    optuna_timeout: Optional[int] = None
    early_stopping_rounds: int = 80
    random_state: int = 42

    n_jobs: int = 64
    numba_threads: int = 64
    joblib_batch_size: int = 8

    use_gpu: int = 1
    gpu_id: int = 0

    max_optuna_rows: int = 2_500_000
    max_train_rows: Optional[int] = None
    save_panel: int = 1
    save_validation_predictions: int = 1
    save_future_forecast: int = 1

    # Per-cluster settings.
    cluster_ids: Optional[str] = None
    include_noise_cluster: int = 1
    min_train_rows_per_cluster: int = 5000
    min_valid_rows_per_cluster: int = 500
    min_test_rows_per_cluster: int = 500

    macro_publication_lag_days_monthly: int = 21
    macro_publication_lag_days_daily: int = 1


# ============================================================
# NUMBA HELPERS
# ============================================================

@njit(cache=True)
def atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = close.shape[0]
    tr = np.empty(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        if i == 0 or not np.isfinite(close[i - 1]):
            if np.isfinite(high[i]) and np.isfinite(low[i]):
                tr[i] = high[i] - low[i]
            else:
                tr[i] = np.nan
        else:
            a = high[i] - low[i] if np.isfinite(high[i]) and np.isfinite(low[i]) else np.nan
            b = abs(high[i] - close[i - 1]) if np.isfinite(high[i]) else np.nan
            c = abs(low[i] - close[i - 1]) if np.isfinite(low[i]) else np.nan

            v = a
            if np.isfinite(b) and (not np.isfinite(v) or b > v):
                v = b
            if np.isfinite(c) and (not np.isfinite(v) or c > v):
                v = c

            tr[i] = v

    alpha = 1.0 / float(window)
    prev = np.nan

    for i in range(n):
        if not np.isfinite(tr[i]):
            out[i] = prev
        else:
            if not np.isfinite(prev):
                prev = tr[i]
            else:
                prev = alpha * tr[i] + (1.0 - alpha) * prev
            out[i] = prev

    return out


@njit(cache=True)
def rsi_numba(ret: np.ndarray, period: int) -> np.ndarray:
    n = ret.shape[0]
    out = np.empty(n, dtype=np.float64)

    alpha = 1.0 / float(period)
    gain_avg = np.nan
    loss_avg = np.nan

    for i in range(n):
        r = ret[i]

        if not np.isfinite(r):
            out[i] = np.nan
            continue

        gain = r if r > 0 else 0.0
        loss = -r if r < 0 else 0.0

        if not np.isfinite(gain_avg):
            gain_avg = gain
            loss_avg = loss
        else:
            gain_avg = alpha * gain + (1.0 - alpha) * gain_avg
            loss_avg = alpha * loss + (1.0 - alpha) * loss_avg

        if loss_avg == 0.0:
            out[i] = 100.0
        else:
            rs = gain_avg / loss_avg
            out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


@njit(cache=True)
def rolling_corr_beta_numba(x: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    corr = np.empty(n, dtype=np.float64)
    beta = np.empty(n, dtype=np.float64)

    for i in range(n):
        start = i - window + 1
        if start < 0:
            start = 0

        cnt = 0
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0

        for j in range(start, i + 1):
            xv = x[j]
            yv = y[j]

            if np.isfinite(xv) and np.isfinite(yv):
                cnt += 1
                sx += xv
                sy += yv
                sxx += xv * xv
                syy += yv * yv
                sxy += xv * yv

        if cnt <= 2:
            corr[i] = np.nan
            beta[i] = np.nan
        else:
            mx = sx / cnt
            my = sy / cnt

            cov = sxy / cnt - mx * my
            vx = sxx / cnt - mx * mx
            vy = syy / cnt - my * my

            if vx > 0.0 and vy > 0.0:
                corr[i] = cov / math.sqrt(vx * vy)
            else:
                corr[i] = np.nan

            if vy > 0.0:
                beta[i] = cov / vy
            else:
                beta[i] = np.nan

    return corr, beta


# ============================================================
# BASIC HELPERS
# ============================================================

def run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def save_df(df: pd.DataFrame, path_no_ext: Path, csv_also: bool = False) -> None:
    try:
        df.to_parquet(path_no_ext.with_suffix(".parquet"), index=False)
        print(f"[SAVE] {path_no_ext.with_suffix('.parquet')}")

        if csv_also:
            df.to_csv(path_no_ext.with_suffix(".csv"), index=False)
            print(f"[SAVE] {path_no_ext.with_suffix('.csv')}")

    except Exception as e:
        print(f"[WARN] parquet не сохранился: {e}")
        df.to_csv(path_no_ext.with_suffix(".csv"), index=False)
        print(f"[SAVE] {path_no_ext.with_suffix('.csv')}")


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}

    wanted = ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for want in wanted:
        for c in df.columns:
            if str(c).strip().lower() == want.lower():
                ren[c] = want
                break

    return df.rename(columns=ren)


def pick_price_col(df: pd.DataFrame, pref: str) -> str:
    if pref != "auto":
        if pref not in df.columns:
            raise ValueError(f"Нет price_col={pref}. Колонки: {list(df.columns)}")
        non_null = pd.to_numeric(df[pref], errors="coerce").notna().sum()
        if non_null == 0:
            raise ValueError(f"Колонка price_col={pref} есть, но в ней нет числовых значений.")
        return pref
    candidates = ["Adj Close", "Close", "Price", "price"]
    stats = []
    for c in candidates:
        if c in df.columns:
            stats.append((c, pd.to_numeric(df[c], errors="coerce").notna().sum()))
    if not stats:
        raise ValueError("Не найдено ни одной ценовой колонки из: Adj Close, Close, Price, price.")
    stats = sorted(stats, key=lambda x: x[1], reverse=True)
    best_col, best_count = stats[0]
    if best_count == 0:
        raise ValueError(f"Ценовые колонки найдены, но все пустые: {stats}")
    print(f"[PRICE COL] selected={best_col}, non_null={best_count:,}, candidates={stats}")
    return best_col

def _load_one_prices_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = normalize_price_columns(df)

    if "date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError(f"В {path} нужны колонки date и Ticker")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str)

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_source_file"] = path.name

    return df


def load_prices(cfg: Config) -> Tuple[pd.DataFrame, str]:
    paths = [Path(cfg.prices_csv)]

    if cfg.future_prices_csv:
        p2 = Path(cfg.future_prices_csv)
        if p2.exists():
            paths.append(p2)
        else:
            print(f"[WARN] future_prices_csv не найден: {p2}")

    parts = []

    for path in paths:
        d = _load_one_prices_file(path)
        print(
            f"[LOAD] {path}: rows={len(d):,}, "
            f"dates={d['date'].min().date()}..{d['date'].max().date()}"
        )
        parts.append(d)

    df = pd.concat(parts, ignore_index=True, sort=False)
    df = normalize_price_columns(df)

    price_col = pick_price_col(df, cfg.price_col)

    df = df.dropna(subset=["date", "Ticker", price_col]).copy()
    df = df.sort_values(["Ticker", "date", "_source_file"])
    df = df.drop_duplicates(["Ticker", "date"], keep="last")

    if cfg.filter_global_first_date:
        gmin = df["date"].min().normalize()
        firsts = df.groupby("Ticker")["date"].min().dt.normalize()
        keep = set(firsts[firsts == gmin].index)
        df = df[df["Ticker"].isin(keep)].copy()

    counts = df.groupby("Ticker").size()
    keep = counts[counts >= cfg.min_rows_per_ticker].index
    df = df[df["Ticker"].isin(keep)].copy().reset_index(drop=True)

    print(f"[LOAD] TOTAL rows={len(df):,}, tickers={df['Ticker'].nunique():,}, price_col={price_col}")
    print(f"[LOAD] TOTAL dates={df['date'].min().date()}..{df['date'].max().date()}")

    return df, price_col


# ============================================================
# MACRO FEATURES
# ============================================================

def read_macro_csv(path: Path, name: str) -> pd.Series:
    df = pd.read_csv(path)

    date_col = None
    for c in df.columns:
        if str(c).lower() in {"date", "observation_date", "time"}:
            date_col = c
            break

    if date_col is None:
        date_col = df.columns[0]

    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"В macro csv нет value col: {path}")

    value_col = value_cols[0]

    for c in value_cols:
        if pd.to_numeric(df[c].replace(".", np.nan), errors="coerce").notna().sum() > 0:
            value_col = c
            break

    s = pd.Series(
        pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce").values,
        index=pd.to_datetime(df[date_col], errors="coerce"),
        name=name,
    )

    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]

    return s.dropna()


def is_monthly(s: pd.Series, name: str) -> bool:
    if name in MONTHLY_MACRO:
        return True

    if len(s) < 5:
        return False

    diffs = pd.Series(s.index).diff().dropna().dt.days
    return float(diffs.median()) >= 20


def make_macro_features(cfg: Config, dates: pd.DatetimeIndex) -> pd.DataFrame:
    macro_dir = Path(cfg.macro_dir)
    base = pd.DataFrame(index=pd.DatetimeIndex(sorted(dates.unique())))

    if not macro_dir.exists():
        print(f"[WARN] macro_dir не найден: {macro_dir}")
        return pd.DataFrame({"date": base.index})

    loaded = []

    for fname, name in MACRO_FILES.items():
        path = macro_dir / fname

        if not path.exists():
            continue

        s = read_macro_csv(path, name)

        # Будущие макро после macro_known_until считаются неизвестными.
        macro_known_until = pd.Timestamp(cfg.macro_known_until)
        s = s[s.index <= macro_known_until].copy()
        if s.empty:
            continue

        monthly = is_monthly(s, name)
        shift = (
            cfg.macro_publication_lag_days_monthly
            if monthly
            else cfg.macro_publication_lag_days_daily
        )

        # Сдвиг индекса — простая защита от утечки будущей публикации.
        s = s.copy()
        s.index = s.index + pd.offsets.BDay(shift)

        aligned = s.reindex(base.index).ffill()
        avail_aligned = pd.Series(s.index, index=s.index).reindex(base.index).ffill()
        base[name] = aligned
        base[f"{name}_age_available_days"] = (
            pd.Series(base.index, index=base.index) - pd.to_datetime(avail_aligned)
        ).dt.days.astype(float)
        base[f"{name}_after_macro_known_until"] = (base.index > pd.Timestamp(cfg.macro_known_until)).astype(np.float32)

        loaded.append((fname, name, "monthly" if monthly else "daily", shift, str((s.index - pd.offsets.BDay(shift)).max().date())))

    print("[MACRO] loaded:")
    for x in loaded:
        print("  ", x)

    if len(base.columns) == 0:
        return pd.DataFrame({"date": base.index})

    f = pd.DataFrame(index=base.index)

    for col in base.columns:
        s = base[col].astype(float)

        f[col] = s

        for L in MACRO_LAGS:
            f[f"{col}_lag{L}"] = s.shift(L)

        for L in [1, 5, 21, 63, 126, 252]:
            f[f"{col}_diff_{L}"] = s.diff(L)
            f[f"{col}_pct_change_{L}"] = s.pct_change(L)

        for W in MACRO_ROLLS:
            m = s.rolling(W, min_periods=max(3, W // 4)).mean()
            sd = s.rolling(W, min_periods=max(3, W // 4)).std()

            f[f"{col}_roll_mean_{W}"] = m
            f[f"{col}_roll_std_{W}"] = sd
            f[f"{col}_zscore_{W}"] = (s - m) / sd.replace(0, np.nan)

    # Derived macro.
    if {"dgs10", "dgs2"}.issubset(f.columns):
        f["yield_spread_10y_2y"] = f["dgs10"] - f["dgs2"]
        f["yield_curve_inverted_flag"] = (f["yield_spread_10y_2y"] < 0).astype(np.float32)

    if {"dgs30", "dgs10"}.issubset(f.columns):
        f["yield_spread_30y_10y"] = f["dgs30"] - f["dgs10"]

    if {"dgs30", "dgs2"}.issubset(f.columns):
        f["yield_spread_30y_2y"] = f["dgs30"] - f["dgs2"]

    if {"dgs2", "effr"}.issubset(f.columns):
        f["dgs2_minus_effr"] = f["dgs2"] - f["effr"]

    if {"dgs10", "effr"}.issubset(f.columns):
        f["dgs10_minus_effr"] = f["dgs10"] - f["effr"]

    if {"dgs30", "effr"}.issubset(f.columns):
        f["dgs30_minus_effr"] = f["dgs30"] - f["effr"]

    if "cpi" in f.columns:
        f["cpi_mom"] = f["cpi"].pct_change(21)
        f["cpi_yoy"] = f["cpi"].pct_change(252)
        f["cpi_acceleration_21"] = f["cpi_mom"].diff(21)

    if "m2" in f.columns:
        f["m2_mom"] = f["m2"].pct_change(21)
        f["m2_yoy"] = f["m2"].pct_change(252)

    if {"m2_yoy", "cpi_yoy"}.issubset(f.columns):
        f["m2_real_growth"] = f["m2_yoy"] - f["cpi_yoy"]

    if "unrate" in f.columns:
        f["unrate_diff_21"] = f["unrate"].diff(21)
        f["unrate_diff_63"] = f["unrate"].diff(63)
        f["unrate_rising_flag"] = (f["unrate_diff_21"] > 0).astype(np.float32)

    for spread in [
        "yield_spread_10y_2y",
        "yield_spread_30y_10y",
        "yield_spread_30y_2y",
        "dgs10_minus_effr",
    ]:
        if spread in f.columns:
            f[f"{spread}_lag1"] = f[spread].shift(1)
            f[f"{spread}_diff1"] = f[spread].diff(1)
            f[f"{spread}_diff21"] = f[spread].diff(21)
            f[f"{spread}_roll_mean_63"] = f[spread].rolling(63, min_periods=15).mean()
            f[f"{spread}_roll_std_63"] = f[spread].rolling(63, min_periods=15).std()

    return (
        f.replace([np.inf, -np.inf], np.nan)
        .reset_index()
        .rename(columns={"index": "date"})
    )


# ============================================================
# CLUSTER / SEASON FEATURES
# ============================================================

def load_cluster_features(path: Optional[str]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["Ticker", "cluster_id"])

    df = pd.read_csv(path)

    ren = {}
    for c in df.columns:
        lc = str(c).lower()

        if lc in {"company", "ticker", "symbol"}:
            ren[c] = "Ticker"

        elif lc in {"cluster", "cluster_id", "label"}:
            ren[c] = "cluster_id"

    df = df.rename(columns=ren)

    if "Ticker" not in df.columns or "cluster_id" not in df.columns:
        print(f"[WARN] В cluster csv нужны Company/Ticker и Cluster. Колонки: {list(df.columns)}")
        return pd.DataFrame(columns=["Ticker", "cluster_id"])

    out = df[["Ticker", "cluster_id"]].copy()
    out["Ticker"] = out["Ticker"].astype(str)
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce").fillna(-999).astype(int)

    sizes = out["cluster_id"].value_counts().rename("cluster_size")

    out = out.join(sizes, on="cluster_id")
    out["cluster_share"] = out["cluster_size"] / max(1, len(out))
    out["cluster_is_noise"] = (out["cluster_id"] == -1).astype(np.float32)

    print(f"[CLUSTER] rows={len(out)}")

    return out


def load_season_features(season_dir: Optional[str]) -> pd.DataFrame:
    if not season_dir or not Path(season_dir).exists():
        return pd.DataFrame(columns=["Ticker"])

    base = Path(season_dir)
    files = []

    for pat in [
        "*fft_extended_summary.xlsx",
        "*fft_summary.xlsx",
        "*fft_extended_summary.csv",
        "*fft_spectra.csv",
    ]:
        files += list(base.glob(pat))

    files = sorted(set(files))

    if not files:
        return pd.DataFrame(columns=["Ticker"])

    parts = []

    for p in files:
        try:
            d = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)

        except Exception as e:
            print(f"[WARN] season read failed {p}: {e}")
            continue

        if "ticker" in [str(c).lower() for c in d.columns]:
            for c in d.columns:
                if str(c).lower() == "ticker":
                    d = d.rename(columns={c: "Ticker"})
                    break

            d["_file"] = p.name
            parts.append(d)

    if not parts:
        return pd.DataFrame(columns=["Ticker"])

    raw = pd.concat(parts, ignore_index=True)
    raw["Ticker"] = raw["Ticker"].astype(str)

    if "mode" not in raw.columns:
        raw["mode"] = raw["_file"].str.extract(r"(price|returns)", expand=False).fillna("unknown")

    if "source" not in raw.columns:
        raw["source"] = "unknown"

    if "rank" in raw.columns:
        raw["rank"] = pd.to_numeric(raw["rank"], errors="coerce")
    else:
        raw["rank"] = np.nan

    num_cols = [
        "period_days",
        "peak_share",
        "prominence",
        "signal_share",
        "noise_share",
        "norm_peak_share",
    ]

    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    rows = []

    for ticker, g in raw.groupby("Ticker", sort=False):
        row: Dict[str, Any] = {"Ticker": ticker}

        for c in num_cols:
            if c in g.columns:
                row[f"season_{c}_mean"] = float(g[c].mean(skipna=True))
                row[f"season_{c}_max"] = float(g[c].max(skipna=True))

        for (mode, source), gs in g.groupby(["mode", "source"], dropna=False):
            prefix = f"season_{str(mode)}_{str(source)}".replace(" ", "_")

            if gs["rank"].notna().any():
                top = gs.sort_values(["rank", "peak_share"], ascending=[True, False]).head(1)

            elif "peak_share" in gs.columns:
                top = gs.sort_values("peak_share", ascending=False).head(1)

            else:
                top = gs.head(1)

            if len(top):
                tr = top.iloc[0]

                for c in num_cols:
                    if c in top.columns:
                        row[f"{prefix}_top1_{c}"] = tr.get(c, np.nan)

                if "period_band" in top.columns:
                    row[f"{prefix}_top1_period_band"] = str(tr.get("period_band", "unknown"))

        rows.append(row)

    out = pd.DataFrame(rows)

    if "period_band" in raw.columns:
        bands = (
            raw.assign(period_band=raw["period_band"].astype(str))
            .groupby(["Ticker", "period_band"])
            .size()
            .unstack(fill_value=0)
        )

        flags = pd.DataFrame(index=bands.index)

        for b in bands.columns:
            flags[f"has_{b}_cycle"] = (bands[b] > 0).astype(np.float32)

        out = out.merge(flags.reset_index(), on="Ticker", how="left")

    print(f"[SEASON] tickers={len(out)}")

    return out


# ============================================================
# STOCK FEATURES
# ============================================================

def make_one_ticker_features(
    ticker: str,
    sub: pd.DataFrame,
    price_col: str,
    horizons: List[int],
) -> pd.DataFrame:
    sub = sub.sort_values("date").drop_duplicates("date", keep="last")

    p = pd.to_numeric(sub[price_col], errors="coerce").astype(float).reset_index(drop=True)
    date = pd.to_datetime(sub["date"]).reset_index(drop=True)

    high = (
        pd.to_numeric(sub["High"], errors="coerce").astype(float).reset_index(drop=True)
        if "High" in sub.columns else p.copy()
    )

    low = (
        pd.to_numeric(sub["Low"], errors="coerce").astype(float).reset_index(drop=True)
        if "Low" in sub.columns else p.copy()
    )

    close = (
        pd.to_numeric(sub["Close"], errors="coerce").astype(float).reset_index(drop=True)
        if "Close" in sub.columns else p.copy()
    )

    open_ = (
        pd.to_numeric(sub["Open"], errors="coerce").astype(float).reset_index(drop=True)
        if "Open" in sub.columns else p.copy()
    )

    volume = (
        pd.to_numeric(sub["Volume"], errors="coerce").astype(float).reset_index(drop=True)
        if "Volume" in sub.columns else pd.Series(np.nan, index=p.index)
    )

    # Чтобы не фрагментировать DataFrame, основные признаки копим в словарь.
    data: Dict[str, Any] = {
        "date": date,
        "Ticker": ticker,
        "price": p,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }

    df = pd.DataFrame(data)
    df = df.dropna(subset=["date", "price"]).reset_index(drop=True)

    p = df["price"].astype(float)
    logp = np.log(p.replace(0, np.nan))
    ret = logp.diff()

    feat: Dict[str, Any] = {}
    feat["log_price"] = logp
    feat["ret"] = ret

    for h in horizons:
        feat[f"target_logret_h{h}"] = logp.shift(-h) - logp
        feat[f"target_price_h{h}"] = p.shift(-h)
        feat[f"target_date_h{h}"] = df["date"].shift(-h)

    for L in RET_LAGS:
        feat[f"ret_lag{L}"] = ret.shift(L)

    feat["ret_abs_lag1"] = ret.shift(1).abs()
    feat["ret_sign_lag1"] = np.sign(ret.shift(1))
    feat["ret_positive_lag1"] = (ret.shift(1) > 0).astype(np.float32)
    feat["ret_negative_lag1"] = (ret.shift(1) < 0).astype(np.float32)

    for L in PRICE_LAGS:
        feat[f"price_ratio_{L}"] = p / p.shift(L)
        feat[f"price_mom_{L}"] = logp - logp.shift(L)

    for W in ROLLS:
        r = ret.rolling(W, min_periods=max(3, W // 4))

        feat[f"ret_roll_mean_{W}"] = r.mean()
        feat[f"ret_roll_std_{W}"] = r.std()
        feat[f"ret_roll_min_{W}"] = r.min()
        feat[f"ret_roll_max_{W}"] = r.max()
        feat[f"ret_roll_median_{W}"] = r.median()
        feat[f"ret_roll_skew_{W}"] = r.skew()
        feat[f"ret_roll_kurt_{W}"] = r.kurt()
        feat[f"ret_mom_{W}"] = ret.rolling(W, min_periods=max(3, W // 4)).sum()
        feat[f"realized_vol_{W}"] = r.std() * np.sqrt(252.0)

        down = ret.where(ret < 0.0, 0.0)
        up = ret.where(ret > 0.0, 0.0)

        feat[f"downside_vol_{W}"] = (
            down.rolling(W, min_periods=max(3, W // 4)).std() * np.sqrt(252.0)
        )

        feat[f"upside_vol_{W}"] = (
            up.rolling(W, min_periods=max(3, W // 4)).std() * np.sqrt(252.0)
        )

        hi = p.rolling(W, min_periods=max(3, W // 4)).max()
        lo = p.rolling(W, min_periods=max(3, W // 4)).min()

        feat[f"distance_from_{W}d_high"] = p / hi - 1.0
        feat[f"distance_from_{W}d_low"] = p / lo - 1.0

    for a, b in [(5, 21), (21, 63), (63, 252)]:
        feat[f"volatility_ratio_{a}_{b}"] = (
            feat[f"ret_roll_std_{a}"] / feat[f"ret_roll_std_{b}"].replace(0, np.nan)
        )

    high_arr = df["high"].to_numpy(np.float64)
    low_arr = df["low"].to_numpy(np.float64)
    close_arr = df["close"].to_numpy(np.float64)
    ret_arr = ret.to_numpy(np.float64)

    for W in ATR_WINDOWS:
        atr = atr_numba(high_arr, low_arr, close_arr, W)

        feat[f"atr_{W}"] = atr
        feat[f"atr_to_price_{W}"] = atr / p.replace(0, np.nan)

    for P in RSI_PERIODS:
        feat[f"rsi_{P}"] = rsi_numba(ret_arr, P)

    feat["rsi_delta_1"] = pd.Series(feat["rsi_14"]).diff()
    feat["rsi_overbought_flag"] = (pd.Series(feat["rsi_14"]) > 70).astype(np.float32)
    feat["rsi_oversold_flag"] = (pd.Series(feat["rsi_14"]) < 30).astype(np.float32)

    for W in [5, 10, 21, 42, 63, 126, 252]:
        sma = p.rolling(W, min_periods=max(3, W // 4)).mean()
        ema = p.ewm(span=W, adjust=False).mean()

        feat[f"sma_{W}"] = sma
        feat[f"ema_{W}"] = ema
        feat[f"price_to_sma_{W}"] = p / sma.replace(0, np.nan) - 1.0
        feat[f"price_to_ema_{W}"] = p / ema.replace(0, np.nan) - 1.0

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    feat["macd"] = macd
    feat["macd_signal"] = signal
    feat["macd_hist"] = macd - signal

    mid = p.rolling(21, min_periods=7).mean()
    sd = p.rolling(21, min_periods=7).std()

    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd

    feat["bollinger_mid_21"] = mid
    feat["bollinger_upper_21"] = upper
    feat["bollinger_lower_21"] = lower
    feat["bollinger_width_21"] = (upper - lower) / mid.replace(0, np.nan)
    feat["bollinger_position_21"] = (p - lower) / (upper - lower).replace(0, np.nan)

    vol = df["volume"].astype(float)
    logv = np.log(vol.replace(0, np.nan))

    feat["vol_dln_1"] = logv.diff()

    for L in [1, 5, 21, 63]:
        feat[f"volume_lag{L}"] = vol.shift(L)
        feat[f"volume_change_{L}"] = vol.pct_change(L)

    for W in VOL_WINDOWS:
        m = vol.rolling(W, min_periods=max(3, W // 4)).mean()
        s = vol.rolling(W, min_periods=max(3, W // 4)).std()

        feat[f"volume_roll_mean_{W}"] = m
        feat[f"volume_roll_std_{W}"] = s
        feat[f"volume_zscore_{W}"] = (vol - m) / s.replace(0, np.nan)
        feat[f"vol_dln_roll_std_{W}"] = (
            feat["vol_dln_1"].rolling(W, min_periods=max(3, W // 4)).std()
        )

    for W in [21, 63, 126]:
        feat[f"ret_volume_corr_{W}"] = (
            ret.rolling(W, min_periods=max(5, W // 4)).corr(feat["vol_dln_1"])
        )

    dt = pd.to_datetime(df["date"])

    feat["day_of_week"] = dt.dt.dayofweek
    feat["day_of_month"] = dt.dt.day
    feat["day_of_year"] = dt.dt.dayofyear
    feat["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    feat["month"] = dt.dt.month
    feat["quarter"] = dt.dt.quarter
    feat["year"] = dt.dt.year

    feat["is_month_start"] = dt.dt.is_month_start.astype(np.float32)
    feat["is_month_end"] = dt.dt.is_month_end.astype(np.float32)
    feat["is_quarter_start"] = dt.dt.is_quarter_start.astype(np.float32)
    feat["is_quarter_end"] = dt.dt.is_quarter_end.astype(np.float32)
    feat["is_year_start"] = dt.dt.is_year_start.astype(np.float32)
    feat["is_year_end"] = dt.dt.is_year_end.astype(np.float32)

    feat["month_sin"] = np.sin(2.0 * np.pi * feat["month"] / 12.0)
    feat["month_cos"] = np.cos(2.0 * np.pi * feat["month"] / 12.0)
    feat["dow_sin"] = np.sin(2.0 * np.pi * feat["day_of_week"] / 7.0)
    feat["dow_cos"] = np.cos(2.0 * np.pi * feat["day_of_week"] / 7.0)
    feat["doy_sin"] = np.sin(2.0 * np.pi * feat["day_of_year"] / 365.25)
    feat["doy_cos"] = np.cos(2.0 * np.pi * feat["day_of_year"] / 365.25)

    df = pd.concat([df, pd.DataFrame(feat)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    return df


def optimize_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Снижает RAM перед сохранением/склейкой panel.
    float64 -> float32, int64 -> int32 там, где это безопасно.
    """
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[c]):
            mn = df[c].min(skipna=True)
            mx = df[c].max(skipna=True)
            if pd.notna(mn) and pd.notna(mx):
                if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
                    df[c] = df[c].astype(np.int16)
                elif mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
                    df[c] = df[c].astype(np.int32)
    return df


def make_stock_features(
    raw: pd.DataFrame,
    price_col: str,
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Streaming/batched версия построения дневных признаков.

    Старая версия делала так:
        parts = Parallel(...)(... все тикеры ...)
        panel = pd.concat(parts)

    Для дневных данных это даёт огромный пик RAM: одновременно живут
    все датафреймы по тикерам + итоговый concat.

    Здесь признаки считаются батчами, каждый батч сразу сохраняется в parquet,
    затем батчи перечитываются уже с оптимизированными dtype.
    """
    horizons = list(range(1, cfg.forecast_horizon + 1))
    groups = list(raw.groupby("Ticker", sort=False))

    print(f"[FEATURES] tickers={len(groups)}, n_jobs={cfg.n_jobs}")

    tmp_dir = out_dir / "_daily_ticker_feature_batches"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Чистим старые батчи, если такой run_dir переиспользуется.
    for old in tmp_dir.glob("features_batch_*.parquet"):
        old.unlink()

    batch_size = max(1, int(getattr(cfg, "joblib_batch_size", 8)))
    batch_paths: List[Path] = []

    for batch_id, start_i in enumerate(range(0, len(groups), batch_size)):
        batch = groups[start_i:start_i + batch_size]
        print(
            f"[FEATURES] batch {batch_id + 1}/{math.ceil(len(groups) / batch_size)} "
            f"tickers={start_i + 1}..{start_i + len(batch)}"
        )

        # Для дневной версии используем threading, чтобы не плодить копии raw/sub в процессах.
        parts = Parallel(
            n_jobs=cfg.n_jobs,
            backend="threading",
            batch_size=1,
            verbose=0,
        )(
            delayed(make_one_ticker_features)(ticker, sub, price_col, horizons)
            for ticker, sub in batch
        )

        batch_df = pd.concat(parts, ignore_index=True, copy=False)
        batch_df = optimize_panel_dtypes(batch_df)
        batch_df = batch_df.replace([np.inf, -np.inf], np.nan)

        path = tmp_dir / f"features_batch_{batch_id:04d}.parquet"
        batch_df.to_parquet(path, index=False)
        batch_paths.append(path)

        print(
            f"[FEATURES] saved {path.name}: "
            f"rows={len(batch_df):,}, cols={len(batch_df.columns):,}, "
            f"size={path.stat().st_size / (1024 ** 2):.1f} MB"
        )

        del parts, batch_df
        gc.collect()

    print(f"[FEATURES] reading {len(batch_paths)} parquet batches")

    frames = []
    for path in batch_paths:
        frames.append(pd.read_parquet(path))

    out = pd.concat(frames, ignore_index=True, copy=False)
    out = optimize_panel_dtypes(out)

    del frames
    gc.collect()

    print(f"[FEATURES] stock panel rows={len(out):,}, cols={len(out.columns):,}")

    return out


# ============================================================
# PANEL-LEVEL FEATURES
# ============================================================

def add_market_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Облегчённая daily-версия market features.
    В старой daily-версии здесь считались rolling beta/correlation через joblib по всем тикерам,
    что создавало большой пик RAM. Для последнего месячного backtest оставляем устойчивые
    рыночные лаги/rolling-признаки и excess return, без тяжёлых beta-фичей.
    """
    print("[FEATURES] market lightweight")

    mkt = panel.groupby("date")["ret"].mean().sort_index().rename("mkt_ret").to_frame()

    for L in [1, 2, 3, 5, 10, 21, 42, 63]:
        mkt[f"mkt_ret_lag{L}"] = mkt["mkt_ret"].shift(L)

    for W in [5, 21, 63, 126, 252]:
        mkt[f"mkt_ret_roll_mean_{W}"] = mkt["mkt_ret"].rolling(W, min_periods=max(3, W // 4)).mean()
        mkt[f"mkt_ret_roll_std_{W}"] = mkt["mkt_ret"].rolling(W, min_periods=max(3, W // 4)).std()
        mkt[f"market_regime_bull_{W}"] = (mkt[f"mkt_ret_roll_mean_{W}"] > 0).astype(np.float32)
        mkt[f"market_regime_bear_{W}"] = (mkt[f"mkt_ret_roll_mean_{W}"] < 0).astype(np.float32)

    panel = panel.merge(mkt.reset_index(), on="date", how="left")
    panel["excess_ret"] = panel["ret"] - panel["mkt_ret"]

    for L in [1, 5, 21]:
        panel[f"excess_ret_lag{L}"] = panel.groupby("Ticker", sort=False)["excess_ret"].shift(L)

    panel = optimize_panel_dtypes(panel.replace([np.inf, -np.inf], np.nan))
    return panel

def add_cluster_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Облегчённая cluster dynamics для daily: только агрегаты текущего дня по кластеру.
    Rolling cluster-признаки на дневной full-panel сильно увеличивают память.
    """
    if "cluster_id" not in panel.columns:
        return panel

    print("[FEATURES] cluster dynamics lightweight")

    cl = panel.groupby(["date", "cluster_id"], as_index=False).agg(
        cluster_ret_mean_1=("ret", "mean"),
        cluster_ret_median_1=("ret", "median"),
        cluster_ret_std_1=("ret", "std"),
        cluster_positive_share_1=("ret", lambda x: float(np.mean(np.asarray(x) > 0))),
        cluster_negative_share_1=("ret", lambda x: float(np.mean(np.asarray(x) < 0))),
    )

    panel = panel.merge(cl, on=["date", "cluster_id"], how="left")
    panel["ret_minus_cluster_mean"] = panel["ret"] - panel["cluster_ret_mean_1"]
    panel["ret_zscore_in_cluster"] = panel["ret_minus_cluster_mean"] / panel["cluster_ret_std_1"].replace(0, np.nan)

    panel = optimize_panel_dtypes(panel.replace([np.inf, -np.inf], np.nan))
    return panel

def add_rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    print("[FEATURES] ranks")

    for col in ["ret", "ret_roll_std_21", "ret_mom_21", "volume"]:
        if col in panel.columns:
            panel[f"{col}_rank_market"] = panel.groupby("date")[col].rank(pct=False)
            panel[f"{col}_percentile_market"] = panel.groupby("date")[col].rank(pct=True)

            if "cluster_id" in panel.columns:
                panel[f"{col}_rank_cluster"] = panel.groupby(["date", "cluster_id"])[col].rank(pct=False)
                panel[f"{col}_percentile_cluster"] = panel.groupby(["date", "cluster_id"])[col].rank(pct=True)

    return panel


def add_macro_interactions(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Облегчённые macro interactions: только простые произведения текущей доходности
    на уже известные macro-change признаки. Без rolling corr/beta по тикерам,
    чтобы не раздувать RAM на дневной версии.
    """
    print("[FEATURES] macro interactions lightweight")

    pairs = [
        ("brent_pct_change_1", "ret_x_brent_return_1"),
        ("dgs10_diff_1", "ret_x_dgs10_diff_1"),
        ("yield_spread_10y_2y", "ret_x_yield_spread_10y_2y"),
        ("usd_eur_pct_change_1", "ret_x_usd_eur_return_1"),
    ]

    for c, out in pairs:
        if c in panel.columns:
            panel[out] = panel["ret"] * panel[c]

    panel = optimize_panel_dtypes(panel.replace([np.inf, -np.inf], np.nan))
    return panel

def add_season_phase(panel: pd.DataFrame) -> pd.DataFrame:
    period_cols = [c for c in panel.columns if c.endswith("_top1_period_days")]

    if not period_cols:
        return panel

    print("[FEATURES] season phases")

    date_to_idx = {d: i for i, d in enumerate(sorted(panel["date"].unique()))}
    t = panel["date"].map(date_to_idx).astype(float)

    for col in period_cols[:8]:
        per = pd.to_numeric(panel[col], errors="coerce").replace(0, np.nan)
        prefix = col.replace("_top1_period_days", "")

        panel[f"{prefix}_phase_sin"] = np.sin(2.0 * np.pi * t / per)
        panel[f"{prefix}_phase_cos"] = np.cos(2.0 * np.pi * t / per)

    return panel


def encode_cats(panel: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    mappings = {}
    cat_cols = []

    for c in panel.columns:
        if c in {"Ticker", "cluster_id"} or panel[c].dtype == "object":
            cat_cols.append(c)

    for c in cat_cols:
        vals = panel[c].astype(str).fillna("__NA__")
        cats = sorted(vals.unique().tolist())
        mp = {v: i for i, v in enumerate(cats)}

        panel[f"{c}_code"] = vals.map(mp).astype(np.int32)
        mappings[c] = mp

    with open(out_dir / "categorical_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    return panel, mappings


def build_panel(
    raw: pd.DataFrame,
    price_col: str,
    cfg: Config,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]]]:
    panel = make_stock_features(raw, price_col, cfg, out_dir)

    cluster = load_cluster_features(cfg.cluster_csv)

    if not cluster.empty:
        panel = panel.merge(cluster, on="Ticker", how="left")
        panel["cluster_id"] = panel["cluster_id"].fillna(-999).astype(int)
        panel["cluster_is_noise"] = panel["cluster_is_noise"].fillna(0).astype(np.float32)

    season = load_season_features(cfg.season_dir)

    if not season.empty:
        panel = panel.merge(season, on="Ticker", how="left")

    macro = make_macro_features(cfg, pd.DatetimeIndex(sorted(panel["date"].unique())))

    if not macro.empty:
        panel = panel.merge(macro, on="date", how="left")

    panel = add_market_features(panel, cfg)
    panel = add_cluster_dynamics(panel)
    panel = add_rank_features(panel)
    panel = add_macro_interactions(panel, cfg)
    panel = add_season_phase(panel)

    panel = panel.replace([np.inf, -np.inf], np.nan)

    panel, mappings = encode_cats(panel, out_dir)

    target_cols = [f"target_logret_h{h}" for h in range(1, cfg.forecast_horizon + 1)]

    drop = {"date", "Ticker", "price", "log_price"}

    drop.update(target_cols)
    drop.update([f"target_price_h{h}" for h in range(1, cfg.forecast_horizon + 1)])
    drop.update([f"target_date_h{h}" for h in range(1, cfg.forecast_horizon + 1)])

    feats = []

    for c in panel.columns:
        if c in drop or c.startswith("target_") or panel[c].dtype == "object":
            continue

        if panel[c].notna().sum() > 0:
            feats.append(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

    print(f"[PANEL] rows={len(panel):,}, features={len(feats):,}")

    return panel, feats, mappings


# ============================================================
# METRICS
# ============================================================

def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(y) > 2 else np.nan,
        "DA": float(np.mean(np.sign(y) == np.sign(p))),
    }


def price_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "WAPE": np.nan,
            "MAPE": np.nan,
            "MSE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "n": 0,
        }

    abs_err = np.abs(y_true - y_pred)

    denom = np.sum(np.abs(y_true))
    nonzero = np.abs(y_true) > 1e-12

    return {
        "WAPE": float(100.0 * np.sum(abs_err) / denom) if denom > 0 else np.nan,
        "MAPE": float(100.0 * np.mean(abs_err[nonzero] / np.abs(y_true[nonzero]))) if np.any(nonzero) else np.nan,
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


# ============================================================
# DATA STACKING
# ============================================================

def stack_horizons(
    df: pd.DataFrame,
    feat_cols: List[str],
    horizons: List[int],
    max_rows: Optional[int],
    seed: int,
    meta: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
    xs = []
    ys = []
    ms = []

    for h in horizons:
        tcol = f"target_logret_h{h}"

        sub = df.dropna(subset=[tcol])

        if sub.empty:
            continue

        x = sub[feat_cols].copy()
        x["horizon"] = np.float32(h)

        xs.append(x)
        ys.append(sub[tcol].to_numpy(np.float32))

        if meta:
            m = sub[["date", "Ticker", "price"]].copy()
            m["horizon"] = h
            m["target_logret"] = sub[tcol].values

            pcol = f"target_price_h{h}"
            dcol = f"target_date_h{h}"

            if pcol in sub.columns:
                m["target_price"] = sub[pcol].values

            if dcol in sub.columns:
                m["target_date"] = sub[dcol].values

            ms.append(m)

    if not xs:
        raise RuntimeError("stack_horizons: пустой набор горизонтов/targets.")

    X = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys).astype(np.float32)

    M = pd.concat(ms, ignore_index=True) if meta and ms else None

    if max_rows is not None and len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        idx.sort()

        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

        if M is not None:
            M = M.iloc[idx].reset_index(drop=True)

    for c in X.columns:
        if X[c].dtype.kind in "fciu":
            X[c] = X[c].astype(np.float32)

    return X, y, M



# ============================================================
# FEATURE SET SPLIT: SHORT WITH MACRO / LONG WITHOUT MACRO
# ============================================================

def is_macro_feature(col: str) -> bool:
    macro_roots = {
        "cpi", "brent", "usd_eur", "dgs2", "dgs10", "dgs30", "effr", "m2", "unrate",
    }
    for root in macro_roots:
        if col == root or col.startswith(root + "_"):
            return True
    macro_prefixes = (
        "yield_spread_", "yield_curve_", "dgs2_minus_effr", "dgs10_minus_effr", "dgs30_minus_effr",
        "ret_x_brent", "ret_x_dgs10", "ret_x_yield_spread", "ret_x_usd_eur",
        "rolling_corr_ret_brent", "rolling_corr_ret_dgs10", "rolling_corr_ret_usd_eur",
        "rolling_corr_ret_cpi", "rolling_corr_ret_m2",
        "beta_to_brent", "beta_to_dgs10", "beta_to_usd_eur", "beta_to_cpi", "beta_to_m2",
    )
    return col.startswith(macro_prefixes)


def split_feature_sets(feat_cols: List[str], out_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    macro_features = [c for c in feat_cols if is_macro_feature(c)]
    short_features = list(feat_cols)
    long_features = [c for c in feat_cols if not is_macro_feature(c)]
    payload = {
        "short_features_count": len(short_features),
        "long_features_count": len(long_features),
        "macro_features_excluded_from_long_count": len(macro_features),
        "short_features": short_features,
        "long_features": long_features,
        "macro_features_excluded_from_long": macro_features,
    }
    with open(out_dir / "feature_sets_short_macro_long_no_macro.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[FEATURE SET] short with macro: {len(short_features):,}")
    print(f"[FEATURE SET] long without macro: {len(long_features):,}")
    print(f"[FEATURE SET] macro excluded from long: {len(macro_features):,}")
    return short_features, long_features, macro_features


def horizon_groups(cfg: Config) -> Tuple[List[int], List[int]]:
    short_h = list(range(1, min(cfg.short_horizon_days, cfg.forecast_horizon) + 1))
    long_h = list(range(min(cfg.short_horizon_days, cfg.forecast_horizon) + 1, cfg.forecast_horizon + 1))
    return short_h, long_h

# ============================================================
# XGBOOST / OPTUNA
# ============================================================

def fixed_params(raw: Dict[str, Any], attrs: Dict[str, Any], cfg: Config) -> Tuple[Dict[str, Any], int]:
    params = {
        "objective": raw.get("objective", "reg:squarederror"),
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": f"cuda:{cfg.gpu_id}" if cfg.use_gpu else "cpu",
        "max_depth": int(raw["max_depth"]),
        "grow_policy": raw.get("grow_policy", "depthwise"),
        "learning_rate": float(raw["learning_rate"]),
        "min_child_weight": float(raw["min_child_weight"]),
        "subsample": float(raw["subsample"]),
        "colsample_bytree": float(raw["colsample_bytree"]),
        "colsample_bynode": float(raw["colsample_bynode"]),
        "lambda": float(raw["lambda"]),
        "alpha": float(raw["alpha"]),
        "gamma": float(raw["gamma"]),
        "max_bin": int(raw["max_bin"]),
        "seed": cfg.random_state,
        "nthread": max(1, min(cfg.n_jobs, os.cpu_count() or cfg.n_jobs)),
    }

    if params["grow_policy"] == "lossguide" and "max_leaves" in raw:
        params["max_leaves"] = int(raw["max_leaves"])

    if cfg.use_gpu:
        params["sampling_method"] = raw.get("sampling_method", "uniform")

    rounds = int(attrs.get("best_iteration", raw.get("num_boost_round", 1000)))
    rounds = max(300, int(rounds * 1.1) + 1)

    return params, rounds


def suggest_params(trial: optuna.Trial, cfg: Config) -> Tuple[Dict[str, Any], int]:
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    params = {
        "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:pseudohubererror"]),
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": f"cuda:{cfg.gpu_id}" if cfg.use_gpu else "cpu",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "grow_policy": grow_policy,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.45, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-4, 100.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        "seed": cfg.random_state,
        "nthread": max(1, min(cfg.n_jobs, os.cpu_count() or cfg.n_jobs)),
    }

    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 32, 512)

    if cfg.use_gpu:
        params["sampling_method"] = trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])

    rounds = trial.suggest_int("num_boost_round", 300, 3000)

    return params, rounds


def run_optuna(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
    horizons: List[int],
    model_name: str,
) -> Dict[str, Any]:
    print(f"[OPTUNA] model={model_name}, horizons={horizons[0]}..{horizons[-1]}, n_horizons={len(horizons)}")

    Xtr, ytr, _ = stack_horizons(train, feat_cols, horizons, cfg.max_optuna_rows, cfg.random_state)

    valid_max = min(cfg.max_optuna_rows // 3, max(1000, len(valid) * len(horizons)))

    Xva, yva, _ = stack_horizons(valid, feat_cols, horizons, valid_max, cfg.random_state + 1)

    print(f"[OPTUNA] Xtr={Xtr.shape}, Xva={Xva.shape}")

    dtr = xgb.DMatrix(Xtr, label=ytr, missing=np.nan, nthread=cfg.n_jobs)
    dva = xgb.DMatrix(Xva, label=yva, missing=np.nan, nthread=cfg.n_jobs)

    def obj(trial: optuna.Trial) -> float:
        params, rounds = suggest_params(trial, cfg)

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=rounds,
            evals=[(dva, "valid")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=False,
        )

        pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        m = metrics(yva, pred)

        trial.set_user_attr("best_iteration", int(bst.best_iteration))
        trial.set_user_attr("rmse", m["RMSE"])
        trial.set_user_attr("mae", m["MAE"])
        trial.set_user_attr("directional_accuracy", m["DA"])
        trial.set_user_attr("r2", m["R2"])

        return m["RMSE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    study.optimize(
        obj,
        n_trials=cfg.n_trials,
        timeout=cfg.optuna_timeout,
        show_progress_bar=True,
        catch=(ValueError, RuntimeError, xgb.core.XGBoostError),
    )

    study.trials_dataframe(
        attrs=("number", "value", "params", "state", "user_attrs")
    ).to_csv(out_dir / f"optuna_trials_{model_name}.csv", index=False)

    best = study.best_trial

    params, rounds = fixed_params(best.params, best.user_attrs, cfg)

    payload = {
        "best_value_rmse": float(study.best_value),
        "best_trial_number": int(best.number),
        "best_params": params,
        "num_boost_round": int(rounds),
        "best_user_attrs": best.user_attrs,
        "raw_optuna_params": best.params,
    }

    with open(out_dir / f"best_params_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    del Xtr, ytr, Xva, yva, dtr, dva
    gc.collect()

    print(f"[OPTUNA] best RMSE={study.best_value:.6f}")

    return payload


def train_final(
    df: pd.DataFrame,
    feat_cols: List[str],
    best: Dict[str, Any],
    cfg: Config,
    out_dir: Path,
    horizons: List[int],
    model_name: str,
) -> xgb.Booster:
    X, y, _ = stack_horizons(
        df,
        feat_cols,
        horizons,
        cfg.max_train_rows,
        cfg.random_state,
    )

    print(f"[TRAIN] model={model_name}, horizons={horizons[0]}..{horizons[-1]}, X={X.shape}")

    dtr = xgb.DMatrix(X, label=y, missing=np.nan, nthread=cfg.n_jobs)

    bst = xgb.train(
        best["best_params"],
        dtr,
        num_boost_round=int(best["num_boost_round"]),
        evals=[(dtr, "train")],
        verbose_eval=100,
    )

    bst.save_model(out_dir / f"global_xgb_daily_model_{model_name}.json")

    imp = bst.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": list(imp.keys()), "gain": list(imp.values())})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    imp_df.to_csv(out_dir / f"feature_importance_gain_{model_name}.csv", index=False)

    del X, y, dtr
    gc.collect()

    return bst


# ============================================================
# VALIDATION / BACKTEST / FORECAST
# ============================================================

def validate(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    valid: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
) -> None:
    short_horizons, long_horizons = horizon_groups(cfg)
    metas = []
    if short_horizons:
        Xs, ys, ms = stack_horizons(valid, short_feat_cols, short_horizons, None, cfg.random_state, meta=True)
        ds = xgb.DMatrix(Xs, label=ys, missing=np.nan, nthread=cfg.n_jobs)
        ps = bst_short.predict(ds)
        ms = ms.copy()
        ms["y_true"] = ys
        ms["y_pred"] = ps
        ms["price_pred"] = ms["price"] * np.exp(ms["y_pred"])
        ms["model_group"] = "short_macro"
        metas.append(ms)
        del Xs, ys, ds
    if long_horizons and bst_long is not None:
        Xl, yl, ml = stack_horizons(valid, long_feat_cols, long_horizons, None, cfg.random_state, meta=True)
        dl = xgb.DMatrix(Xl, label=yl, missing=np.nan, nthread=cfg.n_jobs)
        pl = bst_long.predict(dl)
        ml = ml.copy()
        ml["y_true"] = yl
        ml["y_pred"] = pl
        ml["price_pred"] = ml["price"] * np.exp(ml["y_pred"])
        ml["model_group"] = "long_no_macro"
        metas.append(ml)
        del Xl, yl, dl
    meta = pd.concat(metas, ignore_index=True)
    rows = []
    for h, g in meta.groupby("horizon"):
        rows.append({"horizon": int(h), "model_group": g["model_group"].iloc[0], **metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy()), "n": int(len(g))})
    pd.DataFrame(rows).sort_values("horizon").to_csv(out_dir / "validation_metrics_by_horizon.csv", index=False)
    with open(out_dir / "validation_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(metrics(meta["y_true"].to_numpy(), meta["y_pred"].to_numpy()), f, ensure_ascii=False, indent=2)
    if cfg.save_validation_predictions:
        save_df(meta, out_dir / "validation_predictions")
    gc.collect()

def evaluate_last_months_backtest(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    panel: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    """Rolling daily backtest на последнем cfg.eval_months месяце до cfg.eval_end_date."""
    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(months=int(cfg.eval_months))
    short_horizons, long_horizons = horizon_groups(cfg)

    def collect_predict(horizons: List[int], feat_cols: List[str], bst: xgb.Booster, model_group: str):
        xs, metas = [], []
        for h in horizons:
            tcol = f"target_logret_h{h}"
            pcol = f"target_price_h{h}"
            dcol = f"target_date_h{h}"
            if tcol not in panel.columns or pcol not in panel.columns or dcol not in panel.columns:
                continue
            sub = panel.dropna(subset=[tcol, pcol, dcol]).copy()
            sub[dcol] = pd.to_datetime(sub[dcol])
            sub = sub[(sub[dcol] >= eval_start) & (sub[dcol] <= eval_end) & (sub["date"] < sub[dcol])].copy()
            if sub.empty:
                continue
            x = sub[feat_cols].copy()
            x["horizon"] = np.float32(h)
            m = sub[["date", "Ticker", "price", pcol, dcol, tcol]].copy()
            m = m.rename(columns={"date": "feature_date", "price": "feature_price", pcol: "actual_price", dcol: "target_date", tcol: "actual_logret"})
            m["horizon"] = h
            m["model_group"] = model_group
            xs.append(x); metas.append(m)
        if not xs:
            return None
        X = pd.concat(xs, ignore_index=True)
        meta = pd.concat(metas, ignore_index=True)
        for c in X.columns:
            if X[c].dtype.kind in "fciu":
                X[c] = X[c].astype(np.float32)
        d = xgb.DMatrix(X, missing=np.nan, nthread=cfg.n_jobs)
        pred_logret = bst.predict(d)
        meta["pred_logret"] = pred_logret
        meta["pred_price"] = meta["feature_price"].astype(float) * np.exp(meta["pred_logret"].astype(float))
        return meta

    metas = []
    if short_horizons:
        m = collect_predict(short_horizons, short_feat_cols, bst_short, "short_macro")
        if m is not None:
            metas.append(m)
    if long_horizons and bst_long is not None:
        m = collect_predict(long_horizons, long_feat_cols, bst_long, "long_no_macro")
        if m is not None:
            metas.append(m)
    if not metas:
        raise RuntimeError("Не удалось собрать backtest-выборку: проверь eval_end_date/eval_months/target_date_h.")
    meta = pd.concat(metas, ignore_index=True)
    meta["abs_error"] = np.abs(meta["actual_price"].astype(float) - meta["pred_price"].astype(float))
    meta["ape"] = meta["abs_error"] / meta["actual_price"].replace(0, np.nan).abs()
    meta = meta.sort_values(["target_date", "Ticker", "horizon"]).reset_index(drop=True)

    by_h = []
    for h, g in meta.groupby("horizon"):
        by_h.append({"horizon": int(h), "model_group": g["model_group"].iloc[0], **price_metrics(g["actual_price"].values, g["pred_price"].values)})
    metrics_h = pd.DataFrame(by_h).sort_values("horizon")
    by_group = []
    for group, g in meta.groupby("model_group"):
        row = {"model_group": group}; row.update(price_metrics(g["actual_price"].values, g["pred_price"].values)); by_group.append(row)
    metrics_group = pd.DataFrame(by_group).sort_values("model_group")
    by_date = []
    for dte, g in meta.groupby("target_date"):
        row = {"target_date": pd.Timestamp(dte).date().isoformat()}; row.update(price_metrics(g["actual_price"].values, g["pred_price"].values)); by_date.append(row)
    metrics_d = pd.DataFrame(by_date).sort_values("target_date")
    by_ticker = []
    for tkr, g in meta.groupby("Ticker"):
        row = {"Ticker": tkr}; row.update(price_metrics(g["actual_price"].values, g["pred_price"].values)); by_ticker.append(row)
    metrics_t = pd.DataFrame(by_ticker).sort_values("WAPE")
    overall = {"eval_start": eval_start.date().isoformat(), "eval_end": eval_end.date().isoformat(), "eval_months": int(cfg.eval_months), "short_horizon_days": int(cfg.short_horizon_days), **price_metrics(meta["actual_price"].values, meta["pred_price"].values)}

    meta.to_csv(out_dir / "backtest_last_month_daily_predictions.csv", index=False)
    metrics_h.to_csv(out_dir / "backtest_last_month_daily_metrics_by_horizon.csv", index=False)
    metrics_group.to_csv(out_dir / "backtest_last_month_daily_metrics_by_model_group.csv", index=False)
    metrics_d.to_csv(out_dir / "backtest_last_month_daily_metrics_by_date.csv", index=False)
    metrics_t.to_csv(out_dir / "backtest_last_month_daily_metrics_by_ticker.csv", index=False)
    with open(out_dir / "backtest_last_month_daily_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print("[BACKTEST] last month daily:", overall)
    return meta

def forecast_future(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    panel: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    last = panel.sort_values(["Ticker", "date"]).groupby("Ticker", as_index=False).tail(1).reset_index(drop=True)
    short_horizons, long_horizons = horizon_groups(cfg)
    metas = []
    def predict_last(horizons, feat_cols, bst, model_group):
        xs, ms = [], []
        for h in horizons:
            x = last[feat_cols].copy(); x["horizon"] = np.float32(h)
            m = last[["Ticker", "date", "price"]].copy(); m["horizon"] = h; m["model_group"] = model_group
            xs.append(x); ms.append(m)
        X = pd.concat(xs, ignore_index=True); M = pd.concat(ms, ignore_index=True)
        for c in X.columns:
            if X[c].dtype.kind in "fciu": X[c] = X[c].astype(np.float32)
        d = xgb.DMatrix(X, missing=np.nan, nthread=cfg.n_jobs)
        M["pred_logret"] = bst.predict(d)
        return M
    if short_horizons:
        metas.append(predict_last(short_horizons, short_feat_cols, bst_short, "short_macro"))
    if long_horizons and bst_long is not None:
        metas.append(predict_last(long_horizons, long_feat_cols, bst_long, "long_no_macro"))
    M = pd.concat(metas, ignore_index=True)
    M["last_price"] = M["price"].astype(float)
    M["pred_price"] = M["last_price"] * np.exp(M["pred_logret"])
    global_last = pd.Timestamp(panel["date"].max())
    fdates = pd.bdate_range(global_last + pd.offsets.BDay(1), periods=cfg.forecast_horizon)
    hmap = {i + 1: d for i, d in enumerate(fdates)}
    M["forecast_date"] = M["horizon"].map(hmap)
    out = M[["Ticker", "date", "forecast_date", "horizon", "model_group", "last_price", "pred_logret", "pred_price"]].rename(columns={"date": "feature_date"}).sort_values(["Ticker", "horizon"])
    out.to_csv(out_dir / "forecast_daily_next_month.csv", index=False)
    out.pivot(index="forecast_date", columns="Ticker", values="pred_price").to_csv(out_dir / "forecast_daily_next_month_wide_prices.csv")
    return out

def split_dates(panel: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    dates = np.array(sorted(panel["date"].dropna().unique()))

    if len(dates) <= cfg.val_days + cfg.forecast_horizon + 10:
        return pd.Timestamp(dates[max(1, int(len(dates) * 0.8))])

    return pd.Timestamp(dates[-cfg.val_days])



# ============================================================
# PER-CLUSTER HELPERS
# ============================================================

def parse_cluster_ids_arg(value: Optional[str]) -> Optional[List[int]]:
    if value is None or str(value).strip() == "":
        return None
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def cluster_name(cluster_id: Any) -> str:
    try:
        cid = int(cluster_id)
    except Exception:
        cid = str(cluster_id)
    return f"cluster_{str(cid).replace('-', 'minus_')}"


def target_mask_any(df: pd.DataFrame, cfg: Config) -> pd.Series:
    target_cols = [f"target_logret_h{h}" for h in range(1, cfg.forecast_horizon + 1) if f"target_logret_h{h}" in df.columns]
    if not target_cols:
        return pd.Series(False, index=df.index)
    return df[target_cols].notna().any(axis=1)


def count_test_rows_for_cluster(panel_cluster: pd.DataFrame, cfg: Config) -> int:
    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(months=int(cfg.eval_months))
    total = 0
    for h in range(1, cfg.forecast_horizon + 1):
        tcol = f"target_logret_h{h}"
        pcol = f"target_price_h{h}"
        dcol = f"target_date_h{h}"
        if tcol not in panel_cluster.columns or pcol not in panel_cluster.columns or dcol not in panel_cluster.columns:
            continue
        d = pd.to_datetime(panel_cluster[dcol])
        total += int((panel_cluster[tcol].notna() & panel_cluster[pcol].notna() & d.notna() & (d >= eval_start) & (d <= eval_end) & (panel_cluster["date"] < d)).sum())
    return total


def save_aggregate_backtest_outputs(all_meta: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    if all_meta.empty:
        raise RuntimeError("Пустой общий per-cluster backtest.")

    all_meta = all_meta.sort_values(["target_date", "cluster_id", "Ticker", "horizon"]).reset_index(drop=True)
    all_meta.to_csv(out_dir / "backtest_last_month_daily_predictions.csv", index=False)

    overall = {
        "eval_start": (pd.Timestamp(cfg.eval_end_date) - pd.DateOffset(months=int(cfg.eval_months))).date().isoformat(),
        "eval_end": pd.Timestamp(cfg.eval_end_date).date().isoformat(),
        "eval_months": int(cfg.eval_months),
        "short_horizon_days": int(cfg.short_horizon_days),
        **price_metrics(all_meta["actual_price"].values, all_meta["pred_price"].values),
    }
    with open(out_dir / "backtest_last_month_daily_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    rows = []
    for cid, g in all_meta.groupby("cluster_id", dropna=False):
        row = {"cluster_id": int(cid) if pd.notna(cid) else cid}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        row["tickers"] = int(g["Ticker"].nunique())
        rows.append(row)
    pd.DataFrame(rows).sort_values("WAPE").to_csv(out_dir / "backtest_last_month_daily_metrics_by_cluster.csv", index=False)

    rows = []
    for (cid, group), g in all_meta.groupby(["cluster_id", "model_group"], dropna=False):
        row = {"cluster_id": int(cid) if pd.notna(cid) else cid, "model_group": group}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        row["tickers"] = int(g["Ticker"].nunique())
        rows.append(row)
    pd.DataFrame(rows).sort_values(["cluster_id", "model_group"]).to_csv(out_dir / "backtest_last_month_daily_metrics_by_cluster_and_model_group.csv", index=False)

    rows = []
    for group, g in all_meta.groupby("model_group", dropna=False):
        row = {"model_group": group}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        rows.append(row)
    pd.DataFrame(rows).sort_values("model_group").to_csv(out_dir / "backtest_last_month_daily_metrics_by_model_group.csv", index=False)

    rows = []
    for h, g in all_meta.groupby("horizon", dropna=False):
        row = {"horizon": int(h), "model_group": g["model_group"].iloc[0]}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        rows.append(row)
    pd.DataFrame(rows).sort_values("horizon").to_csv(out_dir / "backtest_last_month_daily_metrics_by_horizon.csv", index=False)

    rows = []
    for dte, g in all_meta.groupby("target_date", dropna=False):
        row = {"target_date": pd.Timestamp(dte).date().isoformat()}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        rows.append(row)
    pd.DataFrame(rows).sort_values("target_date").to_csv(out_dir / "backtest_last_month_daily_metrics_by_date.csv", index=False)

    rows = []
    for ticker, g in all_meta.groupby("Ticker", dropna=False):
        row = {"Ticker": ticker, "cluster_id": int(g["cluster_id"].iloc[0]) if pd.notna(g["cluster_id"].iloc[0]) else g["cluster_id"].iloc[0]}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        rows.append(row)
    pd.DataFrame(rows).sort_values("WAPE").to_csv(out_dir / "backtest_last_month_daily_metrics_by_ticker.csv", index=False)

    print("[BACKTEST] per-cluster aggregate:", overall, flush=True)

# ============================================================
# CLI / MAIN
# ============================================================

def parse_args() -> Config:
    ap = argparse.ArgumentParser("Global XGBoost daily stock forecast optimized for H100")

    ap.add_argument("--prices-csv", default="data/prices_all.csv")
    ap.add_argument("--future-prices-csv", default=None)
    ap.add_argument("--eval-end-date", default="2026-05-08")
    ap.add_argument("--eval-months", type=int, default=1)
    ap.add_argument("--macro-known-until", default="2025-12-20")

    ap.add_argument("--macro-dir", default="data")
    ap.add_argument("--cluster-csv", default=None)
    ap.add_argument("--season-dir", default=None)
    ap.add_argument("--out-root", default="results_xgb_daily_h100")

    ap.add_argument("--date-col", default="date")
    ap.add_argument("--ticker-col", default="Ticker")
    ap.add_argument("--price-col", default="auto")

    ap.add_argument("--forecast-horizon", type=int, default=21)
    ap.add_argument("--short-horizon-days", type=int, default=5)
    ap.add_argument("--val-days", type=int, default=252)
    ap.add_argument("--min-rows-per-ticker", type=int, default=300)
    ap.add_argument("--filter-global-first-date", type=int, default=0)

    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--optuna-timeout", type=int, default=None)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--n-jobs", type=int, default=64)
    ap.add_argument("--numba-threads", type=int, default=64)
    ap.add_argument("--joblib-batch-size", type=int, default=8)

    ap.add_argument("--use-gpu", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)

    ap.add_argument("--max-optuna-rows", type=int, default=2_500_000)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--save-panel", type=int, default=1)
    ap.add_argument("--save-validation-predictions", type=int, default=1)
    ap.add_argument("--save-future-forecast", type=int, default=1)

    ap.add_argument("--cluster-ids", default=None, help="Comma-separated cluster ids to train, e.g. 0,1,2")
    ap.add_argument("--include-noise-cluster", type=int, default=1)
    ap.add_argument("--min-train-rows-per-cluster", type=int, default=5000)
    ap.add_argument("--min-valid-rows-per-cluster", type=int, default=500)
    ap.add_argument("--min-test-rows-per-cluster", type=int, default=500)

    ap.add_argument("--macro-publication-lag-days-monthly", type=int, default=21)
    ap.add_argument("--macro-publication-lag-days-daily", type=int, default=1)

    a = ap.parse_args()
    d = vars(a)

    d["filter_global_first_date"] = bool(d.get("filter_global_first_date", 0))

    config_fields = set(Config.__dataclass_fields__.keys())
    d = {k: v for k, v in d.items() if k in config_fields}

    return Config(**d)


def main() -> None:
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore", PerformanceWarning)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = parse_args()
    set_num_threads(max(1, cfg.numba_threads))

    out_dir = Path(cfg.out_root) / run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[RUN] Per-cluster XGBoost daily forecast")
    print("[RUN] out_dir:", out_dir.resolve())
    print("[RUN] XGBoost:", xgb.__version__)
    print("[RUN] use_gpu:", cfg.use_gpu, "gpu_id:", cfg.gpu_id)
    print("[RUN] n_jobs:", cfg.n_jobs, "numba_threads:", get_num_threads())
    print("[RUN] eval_end_date:", cfg.eval_end_date, "eval_months:", cfg.eval_months)
    print("[RUN] macro_known_until:", cfg.macro_known_until)
    print("[RUN] short_horizon_days:", cfg.short_horizon_days, "(macro used only up to this horizon)")
    print("[RUN] cluster_ids:", cfg.cluster_ids if cfg.cluster_ids else "ALL")
    print("=" * 80)

    raw, price_col = load_prices(cfg)

    panel, feat_cols, mappings = build_panel(raw, price_col, cfg, out_dir)
    short_feat_cols, long_feat_cols, macro_excluded = split_feature_sets(feat_cols, out_dir)
    short_horizons, long_horizons = horizon_groups(cfg)

    if "cluster_id" not in panel.columns:
        raise RuntimeError("В panel нет cluster_id. Для per-cluster версии нужен --cluster-csv.")

    if cfg.save_panel:
        save_df(panel, out_dir / "forecast_panel_features")

    has_target = target_mask_any(panel, cfg)

    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(months=int(cfg.eval_months))
    train_target_cutoff = eval_start - pd.offsets.BDay(cfg.forecast_horizon)

    print(f"[EVAL] eval_start={eval_start.date()}, eval_end={eval_end.date()}, eval_months={cfg.eval_months}")
    print(f"[SPLIT] train_target_cutoff={train_target_cutoff.date()}")
    print(f"[MODEL SPLIT] short horizons with macro: {short_horizons[0]}..{short_horizons[-1] if short_horizons else 'none'}")
    print(f"[MODEL SPLIT] long horizons without macro: {long_horizons[0] if long_horizons else 'none'}..{long_horizons[-1] if long_horizons else 'none'}")

    selected = parse_cluster_ids_arg(cfg.cluster_ids)
    cluster_ids = sorted(pd.to_numeric(panel["cluster_id"], errors="coerce").dropna().astype(int).unique().tolist())
    if selected is not None:
        cluster_ids = [cid for cid in cluster_ids if cid in set(selected)]
    if not int(cfg.include_noise_cluster):
        cluster_ids = [cid for cid in cluster_ids if cid != -1]

    print(f"[CLUSTERS] to train: {cluster_ids}")

    all_predictions = []
    report_rows = []

    for cid in cluster_ids:
        cname = cluster_name(cid)
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)

        cp = panel[panel["cluster_id"].astype(int) == int(cid)].copy()
        c_has_target = has_target.loc[cp.index]

        historical = cp[(cp["date"] < train_target_cutoff) & c_has_target].copy()
        test_rows = count_test_rows_for_cluster(cp, cfg)

        row = {
            "cluster_id": int(cid),
            "cluster_name": cname,
            "tickers": int(cp["Ticker"].nunique()),
            "panel_rows": int(len(cp)),
            "historical_rows": int(len(historical)),
            "test_rows": int(test_rows),
            "status": "pending",
            "reason": "",
        }

        print("-" * 80)
        print(f"[CLUSTER] {cid} tickers={row['tickers']} panel_rows={row['panel_rows']:,} historical_rows={row['historical_rows']:,} test_rows={test_rows:,}")

        if historical.empty:
            row["status"] = "skipped"
            row["reason"] = "empty historical"
            report_rows.append(row)
            continue

        if len(historical) < cfg.min_train_rows_per_cluster:
            row["status"] = "skipped"
            row["reason"] = f"historical_rows < {cfg.min_train_rows_per_cluster}"
            report_rows.append(row)
            continue

        if test_rows < cfg.min_test_rows_per_cluster:
            row["status"] = "skipped"
            row["reason"] = f"test_rows < {cfg.min_test_rows_per_cluster}"
            report_rows.append(row)
            continue

        cutoff = split_dates(historical, cfg)
        train = historical[historical["date"] < cutoff].copy()
        valid = historical[historical["date"] >= cutoff].copy()

        row["optuna_cutoff"] = cutoff.date().isoformat()
        row["train_rows"] = int(len(train))
        row["valid_rows"] = int(len(valid))

        print(f"[CLUSTER {cid}] optuna cutoff={cutoff.date()}, train={len(train):,}, valid={len(valid):,}")

        if len(train) < cfg.min_train_rows_per_cluster:
            row["status"] = "skipped"
            row["reason"] = f"train_rows < {cfg.min_train_rows_per_cluster}"
            report_rows.append(row)
            continue

        if len(valid) < cfg.min_valid_rows_per_cluster:
            row["status"] = "skipped"
            row["reason"] = f"valid_rows < {cfg.min_valid_rows_per_cluster}"
            report_rows.append(row)
            continue

        try:
            best_short = run_optuna(train, valid, short_feat_cols, cfg, cdir, short_horizons, "short_macro")
            final_train = historical
            bst_short = train_final(final_train, short_feat_cols, best_short, cfg, cdir, short_horizons, "short_macro")

            bst_long = None
            if long_horizons:
                best_long = run_optuna(train, valid, long_feat_cols, cfg, cdir, long_horizons, "long_no_macro")
                bst_long = train_final(final_train, long_feat_cols, best_long, cfg, cdir, long_horizons, "long_no_macro")

            validate(bst_short, bst_long, valid, short_feat_cols, long_feat_cols, cfg, cdir)
            pred = evaluate_last_months_backtest(bst_short, bst_long, cp, short_feat_cols, long_feat_cols, cfg, cdir)
            pred["cluster_id"] = int(cid)
            pred["cluster_name"] = cname
            all_predictions.append(pred)

            if cfg.save_future_forecast:
                forecast_future(bst_short, bst_long, cp, short_feat_cols, long_feat_cols, cfg, cdir)

            row["status"] = "trained"
            row["reason"] = ""
            row["prediction_rows"] = int(len(pred))

        except Exception as e:
            row["status"] = "failed"
            row["reason"] = repr(e)
            print(f"[CLUSTER {cid}] FAILED: {e}", flush=True)

        report_rows.append(row)
        pd.DataFrame(report_rows).to_csv(out_dir / "cluster_training_report.csv", index=False)
        gc.collect()

    report = pd.DataFrame(report_rows)
    report.to_csv(out_dir / "cluster_training_report.csv", index=False)

    if not all_predictions:
        raise RuntimeError("Ни один кластер не дал predictions. Смотри cluster_training_report.csv")

    all_meta = pd.concat(all_predictions, ignore_index=True)
    save_aggregate_backtest_outputs(all_meta, cfg, out_dir)

    print("\n[DONE]")
    print("Results:", out_dir.resolve())
    print("Backtest overall:", (out_dir / "backtest_last_month_daily_metrics_overall.json").resolve())
    print("Backtest predictions:", (out_dir / "backtest_last_month_daily_predictions.csv").resolve())


if __name__ == "__main__":
    main()
