from __future__ import annotations
import os
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

RET_LAGS_M = [1, 2, 3, 6, 12]
PRICE_LAGS_M = [1, 2, 3, 6, 12, 24]
ROLLS_M = [3, 6, 12, 24, 36]
ATR_WINDOWS_M = [3, 6, 12]
RSI_PERIODS_M = [3, 6, 12]
VOL_WINDOWS_M = [3, 6, 12, 24]
BETA_WINDOWS_M = [6, 12, 24, 36]

MACRO_LAGS_M = [1, 2, 3, 6, 12, 24]
MACRO_ROLLS_M = [3, 6, 12, 24]

@dataclass
class Config:
    prices_csv: str
    macro_dir: str
    cluster_csv: Optional[str]
    season_dir: Optional[str]
    out_root: str
    resume_run_dir: Optional[str] = None

    future_prices_csv: Optional[str] = None

    eval_end_date: str = "2026-05-08"
    eval_years: int = 1

    macro_known_until: str = "2025-12-20"

    date_col: str = "date"
    ticker_col: str = "Ticker"
    price_col: str = "auto"

    month_rule: str = "M"
    forecast_horizon_months: int = 12

    short_horizon_months: int = 3

    val_months: int = 12
    min_months_per_ticker: int = 36
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

    max_optuna_rows: Optional[int] = 0
    max_train_rows: Optional[int] = 0

    save_panel: int = 1
    save_validation_predictions: int = 1
    save_future_forecast: int = 1

    macro_publication_lag_days_monthly: int = 21
    macro_publication_lag_days_daily: int = 1

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

            corr[i] = cov / math.sqrt(vx * vy) if vx > 0.0 and vy > 0.0 else np.nan
            beta[i] = cov / vy if vy > 0.0 else np.nan

    return corr, beta

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

    for want in ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        for c in df.columns:
            if str(c).strip().lower() == want.lower():
                ren[c] = want
                break

    return df.rename(columns=ren)


def pick_price_col(df: pd.DataFrame, pref: str) -> str:
    if pref != "auto":
        if pref not in df.columns:
            raise ValueError(f"Нет price_col={pref}. Колонки: {list(df.columns)}")
        return pref

    if "Adj Close" in df.columns:
        return "Adj Close"

    if "Close" in df.columns:
        return "Close"

    raise ValueError("Нет ни Adj Close, ни Close")


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

    print(f"[LOAD] TOTAL daily rows={len(df):,}, tickers={df['Ticker'].nunique():,}, price_col={price_col}")
    print(f"[LOAD] TOTAL daily dates={df['date'].min().date()}..{df['date'].max().date()}")

    return df.reset_index(drop=True), price_col

def make_monthly_prices(raw: pd.DataFrame, price_col: str, cfg: Config) -> pd.DataFrame:
    rows = []

    for ticker, sub in raw.groupby("Ticker", sort=False):
        sub = sub.sort_values("date").drop_duplicates("date", keep="last").copy()
        if sub.empty:
            continue

        price = pd.to_numeric(sub[price_col], errors="coerce")
        o = pd.to_numeric(sub["Open"], errors="coerce") if "Open" in sub.columns else price
        h = pd.to_numeric(sub["High"], errors="coerce") if "High" in sub.columns else price
        l = pd.to_numeric(sub["Low"], errors="coerce") if "Low" in sub.columns else price
        c = pd.to_numeric(sub["Close"], errors="coerce") if "Close" in sub.columns else price
        v = pd.to_numeric(sub["Volume"], errors="coerce") if "Volume" in sub.columns else pd.Series(np.nan, index=sub.index)

        d = pd.DataFrame({
            "date": pd.to_datetime(sub["date"]),
            "price": price.values,
            "open": o.values,
            "high": h.values,
            "low": l.values,
            "close": c.values,
            "volume": v.values,
        }).dropna(subset=["date", "price"])

        if d.empty:
            continue

        d["month_period"] = d["date"].dt.to_period("M")
        d["daily_log_price"] = np.log(d["price"].replace(0, np.nan))
        d["daily_ret"] = d["daily_log_price"].diff()

        g = d.groupby("month_period", sort=True)

        mo = pd.DataFrame({
            "month_date": g["date"].max(),
            "Ticker": ticker,
            "month_price": g["price"].last(),
            "month_open": g["open"].first(),
            "month_high": g["high"].max(),
            "month_low": g["low"].min(),
            "month_close": g["close"].last(),
            "month_volume": g["volume"].sum(min_count=1),
            "trading_days_in_month": g["price"].count(),
            "daily_price_last": g["price"].last(),
            "daily_price_mean": g["price"].mean(),
            "daily_price_std": g["price"].std(),
            "daily_price_min": g["price"].min(),
            "daily_price_max": g["price"].max(),
            "daily_ret_mean_in_month": g["daily_ret"].mean(),
            "daily_ret_std_in_month": g["daily_ret"].std(),
            "daily_ret_min_in_month": g["daily_ret"].min(),
            "daily_ret_max_in_month": g["daily_ret"].max(),
            "daily_ret_sum_in_month": g["daily_ret"].sum(min_count=1),
            "daily_volume_mean_in_month": g["volume"].mean(),
            "daily_volume_std_in_month": g["volume"].std(),
            "daily_volume_max_in_month": g["volume"].max(),
        }).reset_index(drop=True)

        mo = mo.dropna(subset=["month_price"])
        rows.append(mo)

    if not rows:
        raise RuntimeError("Не удалось собрать месячные цены.")

    monthly = pd.concat(rows, ignore_index=True)
    monthly = monthly.sort_values(["Ticker", "month_date"]).reset_index(drop=True)

    counts = monthly.groupby("Ticker").size()
    keep = counts[counts >= cfg.min_months_per_ticker].index
    monthly = monthly[monthly["Ticker"].isin(keep)].copy().reset_index(drop=True)

    print(f"[MONTHLY] rows={len(monthly):,}, tickers={monthly['Ticker'].nunique():,}")
    print(f"[MONTHLY] dates={monthly['month_date'].min().date()}..{monthly['month_date'].max().date()}")

    return monthly

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


def make_macro_features(cfg: Config, month_dates: pd.DatetimeIndex) -> pd.DataFrame:
    macro_dir = Path(cfg.macro_dir)
    month_dates = pd.DatetimeIndex(sorted(month_dates.unique()))

    base = pd.DataFrame(index=month_dates)

    if not macro_dir.exists():
        print(f"[WARN] macro_dir не найден: {macro_dir}")
        return pd.DataFrame({"month_date": month_dates})

    loaded = []
    macro_known_until = pd.Timestamp(cfg.macro_known_until)

    for fname, name in MACRO_FILES.items():
        path = macro_dir / fname

        if not path.exists():
            continue

        s_raw = read_macro_csv(path, name)

        s_raw = s_raw[s_raw.index <= macro_known_until].copy()

        if s_raw.empty:
            continue

        monthly = is_monthly(s_raw, name)
        lag_days = cfg.macro_publication_lag_days_monthly if monthly else cfg.macro_publication_lag_days_daily

        avail_index = s_raw.index + pd.offsets.BDay(lag_days)

        avail_df = pd.DataFrame({
            "available_date": avail_index,
            "observation_date": s_raw.index,
            name: s_raw.values,
        }).sort_values("available_date")

        avail_df = avail_df.drop_duplicates("available_date", keep="last")

        tmp = avail_df.set_index("available_date")

        value_aligned = tmp[name].reindex(month_dates).ffill()
        obs_aligned = tmp["observation_date"].reindex(month_dates).ffill()
        avail_aligned = pd.Series(tmp.index, index=tmp.index).reindex(month_dates).ffill()

        base[name] = value_aligned.astype(float)

        base[f"{name}_age_available_days"] = (pd.Series(month_dates, index=month_dates) - pd.to_datetime(avail_aligned)).dt.days.astype(float)
        base[f"{name}_age_observation_days"] = (pd.Series(month_dates, index=month_dates) - pd.to_datetime(obs_aligned)).dt.days.astype(float)

        base[f"{name}_after_macro_known_until"] = (month_dates > macro_known_until).astype(np.float32)

        loaded.append((fname, name, "monthly" if monthly else "daily", lag_days, str(s_raw.index.max().date())))

    print("[MACRO] loaded with no future macro after", macro_known_until.date())
    for row in loaded:
        print("  ", row)

    if len([c for c in base.columns if not c.endswith("_days")]) == 0:
        return pd.DataFrame({"month_date": month_dates})

    f = pd.DataFrame(index=month_dates)

    macro_names = [name for _, name in MACRO_FILES.items() if name in base.columns]

    for col in macro_names:
        s = base[col].astype(float)
        f[col] = s

        for age_col in [
            f"{col}_age_available_days",
            f"{col}_age_observation_days",
            f"{col}_after_macro_known_until",
        ]:
            if age_col in base.columns:
                f[age_col] = base[age_col]

        for L in MACRO_LAGS_M:
            f[f"{col}_lag{L}m"] = s.shift(L)

        for L in [1, 2, 4, 8, 13, 26, 52]:
            f[f"{col}_diff_{L}m"] = s.diff(L)
            f[f"{col}_pct_change_{L}m"] = s.pct_change(L)

        for W in MACRO_ROLLS_M:
            m = s.rolling(W, min_periods=max(2, W // 4)).mean()
            sd = s.rolling(W, min_periods=max(2, W // 4)).std()

            f[f"{col}_roll_mean_{W}m"] = m
            f[f"{col}_roll_std_{W}m"] = sd
            f[f"{col}_zscore_{W}m"] = (s - m) / sd.replace(0, np.nan)

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
        f["cpi_mom_4m"] = f["cpi"].pct_change(4)
        f["cpi_yoy_52m"] = f["cpi"].pct_change(52)
        f["cpi_acceleration_4m"] = f["cpi_mom_4m"].diff(4)

    if "m2" in f.columns:
        f["m2_mom_4m"] = f["m2"].pct_change(4)
        f["m2_yoy_52m"] = f["m2"].pct_change(52)

    if {"m2_yoy_52m", "cpi_yoy_52m"}.issubset(f.columns):
        f["m2_real_growth_52m"] = f["m2_yoy_52m"] - f["cpi_yoy_52m"]

    if "unrate" in f.columns:
        f["unrate_diff_4m"] = f["unrate"].diff(4)
        f["unrate_diff_13m"] = f["unrate"].diff(13)
        f["unrate_rising_flag_4m"] = (f["unrate_diff_4m"] > 0).astype(np.float32)

    for spread in [
        "yield_spread_10y_2y",
        "yield_spread_30y_10y",
        "yield_spread_30y_2y",
        "dgs10_minus_effr",
    ]:
        if spread in f.columns:
            f[f"{spread}_lag1m"] = f[spread].shift(1)
            f[f"{spread}_diff1m"] = f[spread].diff(1)
            f[f"{spread}_diff4m"] = f[spread].diff(4)
            f[f"{spread}_roll_mean_13m"] = f[spread].rolling(13, min_periods=4).mean()
            f[f"{spread}_roll_std_13m"] = f[spread].rolling(13, min_periods=4).std()

    return (
        f.replace([np.inf, -np.inf], np.nan)
        .reset_index()
        .rename(columns={"index": "month_date"})
    )

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

def make_one_ticker_monthly_features(
    ticker: str,
    sub: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    sub = sub.sort_values("month_date").drop_duplicates("month_date", keep="last").reset_index(drop=True)

    p = sub["month_price"].astype(float)
    logp = np.log(p.replace(0, np.nan))
    ret = logp.diff()

    feat: Dict[str, Any] = {}
    feat["log_price"] = logp
    feat["ret_m"] = ret

    feat["price_feature"] = p
    feat["log_price_feature"] = logp

    for h in horizons:
        feat[f"target_logret_m{h}"] = logp.shift(-h) - logp
        feat[f"target_price_m{h}"] = p.shift(-h)
        feat[f"target_month_m{h}"] = sub["month_date"].shift(-h)

    for L in RET_LAGS_M:
        feat[f"ret_lag{L}m"] = ret.shift(L)

    feat["ret_abs_lag1m"] = ret.shift(1).abs()
    feat["ret_sign_lag1m"] = np.sign(ret.shift(1))
    feat["ret_positive_lag1m"] = (ret.shift(1) > 0).astype(np.float32)
    feat["ret_negative_lag1m"] = (ret.shift(1) < 0).astype(np.float32)

    for L in PRICE_LAGS_M:
        feat[f"price_ratio_{L}m"] = p / p.shift(L)
        feat[f"price_mom_{L}m"] = logp - logp.shift(L)

    for W in ROLLS_M:
        r = ret.rolling(W, min_periods=max(2, W // 4))

        feat[f"ret_roll_mean_{W}m"] = r.mean()
        feat[f"ret_roll_std_{W}m"] = r.std()
        feat[f"ret_roll_min_{W}m"] = r.min()
        feat[f"ret_roll_max_{W}m"] = r.max()
        feat[f"ret_roll_median_{W}m"] = r.median()
        feat[f"ret_roll_skew_{W}m"] = r.skew()
        feat[f"ret_roll_kurt_{W}m"] = r.kurt()
        feat[f"ret_mom_{W}m"] = ret.rolling(W, min_periods=max(2, W // 4)).sum()

        feat[f"realized_vol_{W}m"] = r.std() * np.sqrt(12.0)

        down = ret.where(ret < 0.0, 0.0)
        up = ret.where(ret > 0.0, 0.0)

        feat[f"downside_vol_{W}m"] = down.rolling(W, min_periods=max(2, W // 4)).std() * np.sqrt(12.0)
        feat[f"upside_vol_{W}m"] = up.rolling(W, min_periods=max(2, W // 4)).std() * np.sqrt(12.0)

        hi = p.rolling(W, min_periods=max(2, W // 4)).max()
        lo = p.rolling(W, min_periods=max(2, W // 4)).min()

        feat[f"distance_from_{W}m_high"] = p / hi - 1.0
        feat[f"distance_from_{W}m_low"] = p / lo - 1.0

    for a, b in [(4, 13), (13, 52), (26, 104)]:
        if f"ret_roll_std_{a}m" in feat and f"ret_roll_std_{b}m" in feat:
            feat[f"volatility_ratio_{a}_{b}m"] = feat[f"ret_roll_std_{a}m"] / feat[f"ret_roll_std_{b}m"].replace(0, np.nan)

    high_arr = sub["month_high"].to_numpy(np.float64)
    low_arr = sub["month_low"].to_numpy(np.float64)
    close_arr = sub["month_close"].to_numpy(np.float64)
    ret_arr = ret.to_numpy(np.float64)

    for W in ATR_WINDOWS_M:
        atr = atr_numba(high_arr, low_arr, close_arr, W)
        feat[f"atr_{W}m"] = atr
        feat[f"atr_to_price_{W}m"] = atr / p.replace(0, np.nan)

    for P in RSI_PERIODS_M:
        feat[f"rsi_{P}m"] = rsi_numba(ret_arr, P)

    feat["rsi_delta_1m"] = pd.Series(feat["rsi_12m"]).diff()
    feat["rsi_overbought_flag"] = (pd.Series(feat["rsi_12m"]) > 70).astype(np.float32)
    feat["rsi_oversold_flag"] = (pd.Series(feat["rsi_12m"]) < 30).astype(np.float32)

    for W in [3, 6, 12, 24, 36]:
        sma = p.rolling(W, min_periods=max(2, W // 4)).mean()
        ema = p.ewm(span=W, adjust=False).mean()

        feat[f"sma_{W}m"] = sma
        feat[f"ema_{W}m"] = ema
        feat[f"price_to_sma_{W}m"] = p / sma.replace(0, np.nan) - 1.0
        feat[f"price_to_ema_{W}m"] = p / ema.replace(0, np.nan) - 1.0

    ema3 = p.ewm(span=3, adjust=False).mean()
    ema6 = p.ewm(span=6, adjust=False).mean()
    macd = ema3 - ema6
    signal = macd.ewm(span=3, adjust=False).mean()

    feat["macd_m"] = macd
    feat["macd_signal_m"] = signal
    feat["macd_hist_m"] = macd - signal

    mid = p.rolling(6, min_periods=2).mean()
    sd = p.rolling(6, min_periods=2).std()
    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd

    feat["bollinger_mid_6m"] = mid
    feat["bollinger_upper_6m"] = upper
    feat["bollinger_lower_6m"] = lower
    feat["bollinger_width_6m"] = (upper - lower) / mid.replace(0, np.nan)
    feat["bollinger_position_6m"] = (p - lower) / (upper - lower).replace(0, np.nan)

    vol = sub["month_volume"].astype(float)
    logv = np.log(vol.replace(0, np.nan))

    feat["vol_dln_1m"] = logv.diff()

    for L in [1, 2, 3, 6, 12, 24]:
        feat[f"volume_lag{L}m"] = vol.shift(L)
        feat[f"volume_change_{L}m"] = vol.pct_change(L)

    for W in VOL_WINDOWS_M:
        m = vol.rolling(W, min_periods=max(2, W // 4)).mean()
        sdv = vol.rolling(W, min_periods=max(2, W // 4)).std()

        feat[f"volume_roll_mean_{W}m"] = m
        feat[f"volume_roll_std_{W}m"] = sdv
        feat[f"volume_zscore_{W}m"] = (vol - m) / sdv.replace(0, np.nan)
        feat[f"vol_dln_roll_std_{W}m"] = feat["vol_dln_1m"].rolling(W, min_periods=max(2, W // 4)).std()

    for W in [6, 12, 24]:
        feat[f"ret_volume_corr_{W}m"] = ret.rolling(W, min_periods=max(3, W // 4)).corr(feat["vol_dln_1m"])

    dt = pd.to_datetime(sub["month_date"])

    feat["month_of_year"] = dt.dt.month
    feat["month"] = dt.dt.month
    feat["quarter"] = dt.dt.quarter
    feat["year"] = dt.dt.year
    feat["is_month_end_month"] = dt.dt.is_month_end.astype(np.float32)
    feat["is_quarter_end_month"] = dt.dt.is_quarter_end.astype(np.float32)
    feat["is_year_end_month"] = dt.dt.is_year_end.astype(np.float32)

    feat["month_sin"] = np.sin(2.0 * np.pi * feat["month"] / 12.0)
    feat["month_cos"] = np.cos(2.0 * np.pi * feat["month"] / 12.0)
    feat["month_of_year_sin"] = np.sin(2.0 * np.pi * feat["month_of_year"] / 12.0)
    feat["month_of_year_cos"] = np.cos(2.0 * np.pi * feat["month_of_year"] / 12.0)

    out = pd.concat([sub.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).copy()

    return out


def make_monthly_stock_features(monthly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    horizons = list(range(1, cfg.forecast_horizon_months + 1))
    groups = list(monthly.groupby("Ticker", sort=False))

    print(f"[FEATURES] monthly tickers={len(groups)}, n_jobs={cfg.n_jobs}")

    parts = Parallel(
        n_jobs=cfg.n_jobs,
        backend="loky",
        batch_size=cfg.joblib_batch_size,
        verbose=5,
    )(
        delayed(make_one_ticker_monthly_features)(ticker, sub, horizons)
        for ticker, sub in groups
    )

    out = pd.concat(parts, ignore_index=True)

    del parts
    gc.collect()

    print(f"[FEATURES] monthly stock panel rows={len(out):,}, cols={len(out.columns):,}")

    return out

def add_market_features(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("[FEATURES] monthly market")

    mkt = panel.groupby("month_date")["ret_m"].mean().sort_index().rename("mkt_ret_m").to_frame()

    for L in [1, 2, 4, 8, 13, 26, 52]:
        mkt[f"mkt_ret_lag{L}m"] = mkt["mkt_ret_m"].shift(L)

    for W in [4, 13, 26, 52]:
        mkt[f"mkt_ret_roll_mean_{W}m"] = mkt["mkt_ret_m"].rolling(W, min_periods=max(2, W // 4)).mean()
        mkt[f"mkt_ret_roll_std_{W}m"] = mkt["mkt_ret_m"].rolling(W, min_periods=max(2, W // 4)).std()
        mkt[f"market_regime_bull_{W}m"] = (mkt[f"mkt_ret_roll_mean_{W}m"] > 0).astype(np.float32)
        mkt[f"market_regime_bear_{W}m"] = (mkt[f"mkt_ret_roll_mean_{W}m"] < 0).astype(np.float32)

    panel = panel.merge(mkt.reset_index(), on="month_date", how="left")

    panel["excess_ret_m"] = panel["ret_m"] - panel["mkt_ret_m"]

    for L in [1, 4, 13]:
        panel[f"excess_ret_lag{L}m"] = panel.groupby("Ticker", sort=False)["excess_ret_m"].shift(L)

    mkt_map = mkt["mkt_ret_m"].to_dict()

    def one(ticker: str, sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values("month_date")

        x = sub["ret_m"].to_numpy(np.float64)
        y = sub["month_date"].map(mkt_map).to_numpy(np.float64)

        res = pd.DataFrame({"_idx": sub.index.values})

        for W in BETA_WINDOWS_M:
            corr, beta = rolling_corr_beta_numba(x, y, W)
            res[f"corr_mith_market_{W}m"] = corr
            res[f"beta_to_mkt_{W}m"] = beta

        return res

    parts = Parallel(
        n_jobs=min(cfg.n_jobs, 16),
        backend="loky",
        batch_size=16,
    )(
        delayed(one)(t, s)
        for t, s in panel.groupby("Ticker", sort=False)
    )

    rb = pd.concat(parts, ignore_index=True).set_index("_idx")

    for c in rb.columns:
        panel.loc[rb.index, c] = rb[c].values

    return panel.replace([np.inf, -np.inf], np.nan)


def add_cluster_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    if "cluster_id" not in panel.columns:
        return panel

    print("[FEATURES] monthly cluster dynamics")

    cl = panel.groupby(["month_date", "cluster_id"], as_index=False).agg(
        cluster_ret_mean_1m=("ret_m", "mean"),
        cluster_ret_median_1m=("ret_m", "median"),
        cluster_ret_std_1m=("ret_m", "std"),
        cluster_positive_share_1m=("ret_m", lambda x: float(np.mean(np.asarray(x) > 0))),
        cluster_negative_share_1m=("ret_m", lambda x: float(np.mean(np.asarray(x) < 0))),
    )

    panel = panel.merge(cl, on=["month_date", "cluster_id"], how="left")

    panel["ret_minus_cluster_mean_m"] = panel["ret_m"] - panel["cluster_ret_mean_1m"]
    panel["ret_zscore_in_cluster_m"] = (
        panel["ret_minus_cluster_mean_m"] / panel["cluster_ret_std_1m"].replace(0, np.nan)
    )

    cl_ts = cl.sort_values(["cluster_id", "month_date"]).copy()

    for W in [4, 13, 26, 52]:
        cl_ts[f"cluster_ret_mean_roll_{W}m"] = (
            cl_ts.groupby("cluster_id")["cluster_ret_mean_1m"]
            .transform(lambda s: s.rolling(W, min_periods=max(2, W // 4)).mean())
        )

        cl_ts[f"cluster_ret_std_roll_{W}m"] = (
            cl_ts.groupby("cluster_id")["cluster_ret_mean_1m"]
            .transform(lambda s: s.rolling(W, min_periods=max(2, W // 4)).std())
        )

        cl_ts[f"cluster_momentum_{W}m"] = (
            cl_ts.groupby("cluster_id")["cluster_ret_mean_1m"]
            .transform(lambda s: s.rolling(W, min_periods=max(2, W // 4)).sum())
        )

    add_cols = [
        c for c in cl_ts.columns
        if c.startswith("cluster_ret_mean_roll_")
        or c.startswith("cluster_ret_std_roll_")
        or c.startswith("cluster_momentum_")
    ]

    panel = panel.merge(cl_ts[["month_date", "cluster_id"] + add_cols], on=["month_date", "cluster_id"], how="left")

    return panel.replace([np.inf, -np.inf], np.nan)


def add_rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    print("[FEATURES] monthly ranks")

    for col in ["ret_m", "ret_roll_std_13m", "ret_mom_13m", "month_volume", "price_feature"]:
        if col in panel.columns:
            panel[f"{col}_rank_market"] = panel.groupby("month_date")[col].rank(pct=False)
            panel[f"{col}_percentile_market"] = panel.groupby("month_date")[col].rank(pct=True)

            if "cluster_id" in panel.columns:
                panel[f"{col}_rank_cluster"] = panel.groupby(["month_date", "cluster_id"])[col].rank(pct=False)
                panel[f"{col}_percentile_cluster"] = panel.groupby(["month_date", "cluster_id"])[col].rank(pct=True)

    return panel


def add_macro_interactions(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("[FEATURES] monthly macro interactions")

    pairs = [
        ("brent_pct_change_1m", "ret_x_brent_return_1m"),
        ("dgs10_diff_1m", "ret_x_dgs10_diff_1m"),
        ("yield_spread_10y_2y", "ret_x_yield_spread_10y_2y"),
        ("usd_eur_pct_change_1m", "ret_x_usd_eur_return_1m"),
    ]

    for c, out in pairs:
        if c in panel.columns:
            panel[out] = panel["ret_m"] * panel[c]

    factor_cols = [
        c for c in [
            "brent_pct_change_1m",
            "dgs10_diff_1m",
            "usd_eur_pct_change_1m",
            "cpi_pct_change_4m",
            "m2_pct_change_4m",
        ]
        if c in panel.columns
    ]

    if not factor_cols:
        return panel

    def one(ticker: str, sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values("month_date")

        x = sub["ret_m"].to_numpy(np.float64)

        res = pd.DataFrame({"_idx": sub.index.values})

        for fc in factor_cols:
            y = sub[fc].to_numpy(np.float64)

            safe = (
                fc.replace("_pct_change_1m", "")
                .replace("_pct_change_4m", "")
                .replace("_diff_1m", "")
            )

            for W in [26, 52]:
                corr, beta = rolling_corr_beta_numba(x, y, W)
                res[f"rolling_corr_ret_{safe}_{W}m"] = corr
                res[f"beta_to_{safe}_{W}m"] = beta

        return res

    parts = Parallel(
        n_jobs=min(cfg.n_jobs, 16),
        backend="loky",
        batch_size=16,
    )(
        delayed(one)(t, s)
        for t, s in panel.groupby("Ticker", sort=False)
    )

    rb = pd.concat(parts, ignore_index=True).set_index("_idx")

    for c in rb.columns:
        panel.loc[rb.index, c] = rb[c].values

    return panel.replace([np.inf, -np.inf], np.nan)


def add_season_phase(panel: pd.DataFrame) -> pd.DataFrame:
    period_cols = [c for c in panel.columns if c.endswith("_top1_period_days")]

    if not period_cols:
        return panel

    print("[FEATURES] monthly season phases")

    date_to_idx = {d: i for i, d in enumerate(sorted(panel["month_date"].unique()))}
    t_months = panel["month_date"].map(date_to_idx).astype(float)

    for col in period_cols[:8]:
        per_days = pd.to_numeric(panel[col], errors="coerce").replace(0, np.nan)
        per_months = per_days / 30.4375
        prefix = col.replace("_top1_period_days", "")

        panel[f"{prefix}_phase_sin_m"] = np.sin(2.0 * np.pi * t_months / per_months)
        panel[f"{prefix}_phase_cos_m"] = np.cos(2.0 * np.pi * t_months / per_months)

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
    monthly: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]]]:
    panel = make_monthly_stock_features(monthly, cfg)

    cluster = load_cluster_features(cfg.cluster_csv)

    if not cluster.empty:
        panel = panel.merge(cluster, on="Ticker", how="left")
        panel["cluster_id"] = panel["cluster_id"].fillna(-999).astype(int)
        panel["cluster_is_noise"] = panel["cluster_is_noise"].fillna(0).astype(np.float32)

    season = load_season_features(cfg.season_dir)

    if not season.empty:
        panel = panel.merge(season, on="Ticker", how="left")

    macro = make_macro_features(cfg, pd.DatetimeIndex(sorted(panel["month_date"].unique())))

    if not macro.empty:
        panel = panel.merge(macro, on="month_date", how="left")

    panel = add_market_features(panel, cfg)
    panel = add_cluster_dynamics(panel)
    panel = add_rank_features(panel)
    panel = add_macro_interactions(panel, cfg)
    panel = add_season_phase(panel)

    panel = panel.replace([np.inf, -np.inf], np.nan)

    panel, mappings = encode_cats(panel, out_dir)

    target_cols = [f"target_logret_m{h}" for h in range(1, cfg.forecast_horizon_months + 1)]

    drop = {"month_date", "Ticker"}

    drop.update(target_cols)
    drop.update([f"target_price_m{h}" for h in range(1, cfg.forecast_horizon_months + 1)])
    drop.update([f"target_month_m{h}" for h in range(1, cfg.forecast_horizon_months + 1)])

    feats = []

    for c in panel.columns:
        if c in drop or c.startswith("target_") or panel[c].dtype == "object":
            continue

        if panel[c].notna().sum() > 0:
            feats.append(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

    print(f"[PANEL] monthly rows={len(panel):,}, features={len(feats):,}")

    return panel, feats, mappings

def is_macro_feature(col: str) -> bool:
    macro_roots = {
        "cpi",
        "brent",
        "usd_eur",
        "dgs2",
        "dgs10",
        "dgs30",
        "effr",
        "m2",
        "unrate",
    }

    for root in macro_roots:
        if col == root or col.startswith(root + "_"):
            return True

    macro_prefixes = (
        "yield_spread_",
        "yield_curve_",
        "dgs2_minus_effr",
        "dgs10_minus_effr",
        "dgs30_minus_effr",
        "ret_x_brent",
        "ret_x_dgs10",
        "ret_x_yield_spread",
        "ret_x_usd_eur",
        "rolling_corr_ret_brent",
        "rolling_corr_ret_dgs10",
        "rolling_corr_ret_usd_eur",
        "rolling_corr_ret_cpi",
        "rolling_corr_ret_m2",
        "beta_to_brent",
        "beta_to_dgs10",
        "beta_to_usd_eur",
        "beta_to_cpi",
        "beta_to_m2",
    )

    return col.startswith(macro_prefixes)


def split_feature_sets(feat_cols: List[str], out_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    macro_features = [c for c in feat_cols if is_macro_feature(c)]
    long_features = [c for c in feat_cols if not is_macro_feature(c)]
    short_features = list(feat_cols)

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
    short_h = list(range(1, min(cfg.short_horizon_months, cfg.forecast_horizon_months) + 1))
    long_h = list(range(min(cfg.short_horizon_months, cfg.forecast_horizon_months) + 1, cfg.forecast_horizon_months + 1))
    return short_h, long_h

def logret_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
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

def stack_horizons(
    df: pd.DataFrame,
    feat_cols: List[str],
    horizons: List[int],
    max_rows: Optional[int],
    seed: int,
    meta: bool = False,
    target_before: Optional[pd.Timestamp] = None,
    target_from: Optional[pd.Timestamp] = None,
    target_to: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
    xs = []
    ys = []
    ms = []

    for h in horizons:
        tcol = f"target_logret_m{h}"
        pcol = f"target_price_m{h}"
        dcol = f"target_month_m{h}"

        if tcol not in df.columns or dcol not in df.columns:
            continue

        sub = df.dropna(subset=[tcol, dcol]).copy()
        sub[dcol] = pd.to_datetime(sub[dcol])

        if target_before is not None:
            sub = sub[sub[dcol] < target_before]

        if target_from is not None:
            sub = sub[sub[dcol] >= target_from]

        if target_to is not None:
            sub = sub[sub[dcol] <= target_to]

        if sub.empty:
            continue

        x = sub[feat_cols].copy()
        x["horizon_months"] = np.float32(h)

        xs.append(x)
        ys.append(sub[tcol].to_numpy(np.float32))

        if meta:
            m = sub[["month_date", "Ticker", "month_price"]].copy()
            m["horizon_months"] = h
            m["target_logret"] = sub[tcol].values
            m["target_month"] = sub[dcol].values

            if pcol in sub.columns:
                m["target_price"] = sub[pcol].values

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

def fixed_params(raw: Dict[str, Any], attrs: Dict[str, Any], cfg: Config) -> Tuple[Dict[str, Any], int]:
    params = {
        "objective": raw.get("objective", "reg:squarederror"),
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": f"cuda:{cfg.gpu_id}" if cfg.use_gpu else "cpu",
        "max_depth": int(raw["max_depth"]),
        "grow_policy": raw.get("grow_policy", "depthwise"),
        "learning_rate": float(raw["learning_rate"]),
        "min_child_meight": float(raw["min_child_meight"]),
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
        "min_child_meight": trial.suggest_float("min_child_meight", 1e-3, 50.0, log=True),
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
    train_pairs: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
    train_target_before: pd.Timestamp,
    valid_target_from: pd.Timestamp,
    valid_target_to: pd.Timestamp,
    horizons: List[int],
    model_name: str,
) -> Dict[str, Any]:
    optuna_max_rows = None if (not cfg.max_optuna_rows or cfg.max_optuna_rows <= 0) else cfg.max_optuna_rows
    print(f"[OPTUNA] model={model_name}, horizons={horizons[0]}..{horizons[-1]}, n_horizons={len(horizons)}")

    best_params_path = out_dir / f"best_params_{model_name}.json"
    if best_params_path.exists():
        print(f"[RESUME] loading existing Optuna params: {best_params_path}")
        with open(best_params_path, "r", encoding="utf-8") as f:
            return json.load(f)

    Xtr, ytr, _ = stack_horizons(
        train_pairs,
        feat_cols,
        horizons,
        optuna_max_rows,
        cfg.random_state,
        target_before=train_target_before,
    )

    valid_max = None if (not cfg.max_optuna_rows or cfg.max_optuna_rows <= 0) else min(cfg.max_optuna_rows // 3, max(1000, len(valid_pairs) * len(horizons)))

    Xva, yva, _ = stack_horizons(
        valid_pairs,
        feat_cols,
        horizons,
        valid_max,
        cfg.random_state + 1,
        target_from=valid_target_from,
        target_to=valid_target_to,
    )

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
        m = logret_metrics(yva, pred)

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
    target_before: pd.Timestamp,
    horizons: List[int],
    model_name: str,
) -> xgb.Booster:
    model_path = out_dir / f"global_xgb_monthly_model_{model_name}.json"
    if model_path.exists():
        print(f"[RESUME] loading existing model: {model_path}")
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        return bst

    train_max_rows = None if (not cfg.max_train_rows or cfg.max_train_rows <= 0) else cfg.max_train_rows

    X, y, _ = stack_horizons(
        df,
        feat_cols,
        horizons,
        train_max_rows,
        cfg.random_state,
        target_before=target_before,
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

    bst.save_model(model_path)

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

def validate(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    valid: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
    target_from: pd.Timestamp,
    target_to: pd.Timestamp,
) -> None:
    short_horizons, long_horizons = horizon_groups(cfg)

    metas = []

    if short_horizons:
        Xs, ys, ms = stack_horizons(
            valid,
            short_feat_cols,
            short_horizons,
            None,
            cfg.random_state,
            meta=True,
            target_from=target_from,
            target_to=target_to,
        )
        ds = xgb.DMatrix(Xs, label=ys, missing=np.nan, nthread=cfg.n_jobs)
        ps = bst_short.predict(ds)
        ms = ms.copy()
        ms["y_true"] = ys
        ms["y_pred"] = ps
        ms["price_pred"] = ms["month_price"] * np.exp(ms["y_pred"])
        ms["model_group"] = "short_macro"
        metas.append(ms)
        del Xs, ys, ds

    if long_horizons and bst_long is not None:
        Xl, yl, ml = stack_horizons(
            valid,
            long_feat_cols,
            long_horizons,
            None,
            cfg.random_state,
            meta=True,
            target_from=target_from,
            target_to=target_to,
        )
        dl = xgb.DMatrix(Xl, label=yl, missing=np.nan, nthread=cfg.n_jobs)
        pl = bst_long.predict(dl)
        ml = ml.copy()
        ml["y_true"] = yl
        ml["y_pred"] = pl
        ml["price_pred"] = ml["month_price"] * np.exp(ml["y_pred"])
        ml["model_group"] = "long_no_macro"
        metas.append(ml)
        del Xl, yl, dl

    meta = pd.concat(metas, ignore_index=True)

    rows = []
    for h, g in meta.groupby("horizon_months"):
        rows.append({
            "horizon_months": int(h),
            "model_group": g["model_group"].iloc[0],
            **logret_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
            "n": int(len(g)),
        })

    pd.DataFrame(rows).sort_values("horizon_months").to_csv(
        out_dir / "pretest_validation_metrics_by_horizon.csv",
        index=False,
    )

    with open(out_dir / "pretest_validation_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(logret_metrics(meta["y_true"].to_numpy(), meta["y_pred"].to_numpy()), f, ensure_ascii=False, indent=2)

    if cfg.save_validation_predictions:
        save_df(meta, out_dir / "pretest_validation_predictions")

    gc.collect()

def evaluate_last_year_backtest(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    panel: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(years=int(cfg.eval_years))

    short_horizons, long_horizons = horizon_groups(cfg)

    metas = []

    if short_horizons:
        Xs, _, ms = stack_horizons(
            panel,
            short_feat_cols,
            short_horizons,
            None,
            cfg.random_state,
            meta=True,
            target_from=eval_start,
            target_to=eval_end,
        )
        ds = xgb.DMatrix(Xs, missing=np.nan, nthread=cfg.n_jobs)
        ps = bst_short.predict(ds)
        ms = ms.copy()
        ms["pred_logret"] = ps
        ms["pred_price"] = ms["month_price"].astype(float) * np.exp(ms["pred_logret"].astype(float))
        ms["model_group"] = "short_macro"
        metas.append(ms)
        del Xs, ds

    if long_horizons and bst_long is not None:
        Xl, _, ml = stack_horizons(
            panel,
            long_feat_cols,
            long_horizons,
            None,
            cfg.random_state,
            meta=True,
            target_from=eval_start,
            target_to=eval_end,
        )
        dl = xgb.DMatrix(Xl, missing=np.nan, nthread=cfg.n_jobs)
        pl = bst_long.predict(dl)
        ml = ml.copy()
        ml["pred_logret"] = pl
        ml["pred_price"] = ml["month_price"].astype(float) * np.exp(ml["pred_logret"].astype(float))
        ml["model_group"] = "long_no_macro"
        metas.append(ml)
        del Xl, dl

    meta = pd.concat(metas, ignore_index=True)

    meta["abs_error"] = np.abs(meta["target_price"].astype(float) - meta["pred_price"].astype(float))
    meta["ape"] = meta["abs_error"] / meta["target_price"].replace(0, np.nan).abs()

    meta = meta.rename(columns={
        "month_date": "feature_month",
        "month_price": "feature_price",
        "target_month": "target_month",
        "target_price": "actual_price",
    })

    meta = meta.sort_values(["target_month", "Ticker", "horizon_months"]).reset_index(drop=True)

    by_h = []
    for h, g in meta.groupby("horizon_months"):
        by_h.append({
            "horizon_months": int(h),
            "model_group": g["model_group"].iloc[0],
            **price_metrics(g["actual_price"].values, g["pred_price"].values),
        })
    metrics_h = pd.DataFrame(by_h).sort_values("horizon_months")

    by_group = []
    for group, g in meta.groupby("model_group"):
        row = {"model_group": group}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_group.append(row)
    metrics_group = pd.DataFrame(by_group).sort_values("model_group")

    by_month = []
    for dte, g in meta.groupby("target_month"):
        row = {"target_month": pd.Timestamp(dte).date().isoformat()}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_month.append(row)
    metrics_m = pd.DataFrame(by_month).sort_values("target_month")

    by_ticker = []
    for tkr, g in meta.groupby("Ticker"):
        row = {"Ticker": tkr}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_ticker.append(row)
    metrics_t = pd.DataFrame(by_ticker).sort_values("WAPE")

    overall = {
        "eval_start": eval_start.date().isoformat(),
        "eval_end": eval_end.date().isoformat(),
        "eval_years": int(cfg.eval_years),
        "short_horizon_months": int(cfg.short_horizon_months),
        "short_model": "uses_macro",
        "long_model": "excludes_macro",
        **price_metrics(meta["actual_price"].values, meta["pred_price"].values),
    }

    meta.to_csv(out_dir / "backtest_last_year_monthly_predictions.csv", index=False)
    metrics_h.to_csv(out_dir / "backtest_last_year_monthly_metrics_by_horizon.csv", index=False)
    metrics_group.to_csv(out_dir / "backtest_last_year_monthly_metrics_by_model_group.csv", index=False)
    metrics_m.to_csv(out_dir / "backtest_last_year_monthly_metrics_by_month.csv", index=False)
    metrics_t.to_csv(out_dir / "backtest_last_year_monthly_metrics_by_ticker.csv", index=False)

    with open(out_dir / "backtest_last_year_monthly_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("[BACKTEST] last year monthly:", overall)
    print(f"[BACKTEST] predictions: {out_dir / 'backtest_last_year_monthly_predictions.csv'}")
    print(f"[BACKTEST] metrics by horizon: {out_dir / 'backtest_last_year_monthly_metrics_by_horizon.csv'}")
    print(f"[BACKTEST] metrics by model group: {out_dir / 'backtest_last_year_monthly_metrics_by_model_group.csv'}")
    print(f"[BACKTEST] metrics by month: {out_dir / 'backtest_last_year_monthly_metrics_by_month.csv'}")
    print(f"[BACKTEST] metrics by ticker: {out_dir / 'backtest_last_year_monthly_metrics_by_ticker.csv'}")

    gc.collect()

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
    last = (
        panel.sort_values(["Ticker", "month_date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    short_horizons, long_horizons = horizon_groups(cfg)

    metas = []

    if short_horizons:
        xs, ms = [], []
        for h in short_horizons:
            x = last[short_feat_cols].copy()
            x["horizon_months"] = np.float32(h)
            m = last[["Ticker", "month_date", "month_price"]].copy()
            m["horizon_months"] = h
            xs.append(x)
            ms.append(m)

        Xs = pd.concat(xs, ignore_index=True)
        Ms = pd.concat(ms, ignore_index=True)

        for c in Xs.columns:
            if Xs[c].dtype.kind in "fciu":
                Xs[c] = Xs[c].astype(np.float32)

        ds = xgb.DMatrix(Xs, missing=np.nan, nthread=cfg.n_jobs)
        ps = bst_short.predict(ds)

        Ms["pred_logret"] = ps
        Ms["model_group"] = "short_macro"
        metas.append(Ms)

    if long_horizons and bst_long is not None:
        xl, ml = [], []
        for h in long_horizons:
            x = last[long_feat_cols].copy()
            x["horizon_months"] = np.float32(h)
            m = last[["Ticker", "month_date", "month_price"]].copy()
            m["horizon_months"] = h
            xl.append(x)
            ml.append(m)

        Xl = pd.concat(xl, ignore_index=True)
        Ml = pd.concat(ml, ignore_index=True)

        for c in Xl.columns:
            if Xl[c].dtype.kind in "fciu":
                Xl[c] = Xl[c].astype(np.float32)

        dl = xgb.DMatrix(Xl, missing=np.nan, nthread=cfg.n_jobs)
        pl = bst_long.predict(dl)

        Ml["pred_logret"] = pl
        Ml["model_group"] = "long_no_macro"
        metas.append(Ml)

    M = pd.concat(metas, ignore_index=True)

    M["last_price"] = M["month_price"].astype(float)
    M["pred_price"] = M["last_price"] * np.exp(M["pred_logret"])

    global_last = pd.Timestamp(panel["month_date"].max())
    first_forecast_month = (global_last + pd.offsets.MonthEnd(0)) + pd.offsets.MonthEnd(1)
    fdates = pd.date_range(first_forecast_month, periods=cfg.forecast_horizon_months, freq="M")
    hmap = {i + 1: d for i, d in enumerate(fdates)}

    M["forecast_month"] = M["horizon_months"].map(hmap)

    out = (
        M[["Ticker", "month_date", "forecast_month", "horizon_months", "model_group", "last_price", "pred_logret", "pred_price"]]
        .rename(columns={"month_date": "feature_month"})
        .sort_values(["Ticker", "horizon_months"])
    )

    out.to_csv(out_dir / "forecast_monthly_next_year.csv", index=False)
    out.pivot(index="forecast_month", columns="Ticker", values="pred_price").to_csv(
        out_dir / "forecast_monthly_next_year_wide_prices.csv"
    )

    return out

def evaluate_fixed_origin_forecast(
    bst_short: xgb.Booster,
    bst_long: Optional[xgb.Booster],
    panel: pd.DataFrame,
    short_feat_cols: List[str],
    long_feat_cols: List[str],
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(years=int(cfg.eval_years))
    origin_date = pd.Timestamp(panel.loc[pd.to_datetime(panel["month_date"]) < eval_start, "month_date"].max())
    if pd.isna(origin_date):
        raise RuntimeError(f"Не найден origin month < {eval_start.date()}")

    origin = (
        panel[pd.to_datetime(panel["month_date"]) == origin_date]
        .sort_values(["Ticker", "month_date"])
        .drop_duplicates("Ticker", keep="last")
        .copy()
    )
    short_horizons, long_horizons = horizon_groups(cfg)

    def collect_predict(horizons: List[int], feat_cols: List[str], bst: xgb.Booster, model_group: str):
        xs, metas = [], []
        for h in horizons:
            pcol = f"target_price_m{h}"
            dcol = f"target_month_m{h}"
            tcol = f"target_logret_m{h}"
            if pcol not in origin.columns or dcol not in origin.columns:
                continue
            sub = origin.dropna(subset=[pcol, dcol]).copy()
            sub[dcol] = pd.to_datetime(sub[dcol])
            sub = sub[(sub[dcol] >= eval_start) & (sub[dcol] <= eval_end)].copy()
            if sub.empty:
                continue
            x = sub[feat_cols].copy()
            x["horizon_months"] = np.float32(h)
            m = sub[["month_date", "Ticker", "month_price", pcol, dcol]].copy()
            rename = {"month_date": "origin_month", "month_price": "origin_price", pcol: "actual_price", dcol: "target_month"}
            if tcol in sub.columns:
                m[tcol] = sub[tcol]
                rename[tcol] = "actual_logret"
            m = m.rename(columns=rename)
            m["horizon_months"] = h
            m["model_group"] = model_group
            xs.append(x)
            metas.append(m)
        if not xs:
            return None
        X = pd.concat(xs, ignore_index=True)
        meta = pd.concat(metas, ignore_index=True)
        for c in X.columns:
            if X[c].dtype.kind in "fciu":
                X[c] = X[c].astype(np.float32)
        d = xgb.DMatrix(X, missing=np.nan, nthread=cfg.n_jobs)
        meta["pred_logret"] = bst.predict(d)
        meta["pred_price"] = meta["origin_price"].astype(float) * np.exp(meta["pred_logret"].astype(float))
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
        raise RuntimeError("Не удалось собрать fixed-origin monthly выборку")

    meta = pd.concat(metas, ignore_index=True)
    meta["abs_error"] = np.abs(meta["actual_price"].astype(float) - meta["pred_price"].astype(float))
    meta["ape"] = meta["abs_error"] / meta["actual_price"].replace(0, np.nan).abs()
    meta = meta.sort_values(["target_month", "Ticker", "horizon_months"]).reset_index(drop=True)

    by_h = []
    for h, g in meta.groupby("horizon_months"):
        by_h.append({"horizon_months": int(h), "model_group": g["model_group"].iloc[0], **price_metrics(g["actual_price"].values, g["pred_price"].values)})
    metrics_h = pd.DataFrame(by_h).sort_values("horizon_months")

    by_group = []
    for group, g in meta.groupby("model_group"):
        row = {"model_group": group}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_group.append(row)
    metrics_group = pd.DataFrame(by_group).sort_values("model_group")

    by_date = []
    for dte, g in meta.groupby("target_month"):
        row = {"target_month": pd.Timestamp(dte).date().isoformat()}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_date.append(row)
    metrics_d = pd.DataFrame(by_date).sort_values("target_month")

    by_ticker = []
    for tkr, g in meta.groupby("Ticker"):
        row = {"Ticker": tkr}
        row.update(price_metrics(g["actual_price"].values, g["pred_price"].values))
        by_ticker.append(row)
    metrics_t = pd.DataFrame(by_ticker).sort_values("WAPE")

    overall = {
        "mode": "fixed_origin",
        "eval_start": eval_start.date().isoformat(),
        "eval_end": eval_end.date().isoformat(),
        "origin_month": origin_date.date().isoformat(),
        "eval_years": int(cfg.eval_years),
        "short_horizon_months": int(cfg.short_horizon_months),
        **price_metrics(meta["actual_price"].values, meta["pred_price"].values),
    }

    meta.to_csv(out_dir / "fixed_origin_monthly_predictions.csv", index=False)
    metrics_h.to_csv(out_dir / "fixed_origin_monthly_metrics_by_horizon.csv", index=False)
    metrics_group.to_csv(out_dir / "fixed_origin_monthly_metrics_by_model_group.csv", index=False)
    metrics_d.to_csv(out_dir / "fixed_origin_monthly_metrics_by_date.csv", index=False)
    metrics_t.to_csv(out_dir / "fixed_origin_monthly_metrics_by_ticker.csv", index=False)
    with open(out_dir / "fixed_origin_monthly_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print("[FIXED ORIGIN] monthly:", overall)
    return meta

def parse_args() -> Config:
    ap = argparse.ArgumentParser("Global XGBoost monthly stock forecast optimized for H100")

    ap.add_argument("--prices-csv", default="data/prices_all.csv")
    ap.add_argument("--future-prices-csv", default=None)

    ap.add_argument("--eval-end-date", default="2026-05-08")
    ap.add_argument("--eval-years", type=int, default=1)
    ap.add_argument("--macro-known-until", default="2025-12-20")

    ap.add_argument("--macro-dir", default="data")
    ap.add_argument("--cluster-csv", default=None)
    ap.add_argument("--season-dir", default=None)
    ap.add_argument("--out-root", default="results_xgb_monthly_h100")
    ap.add_argument("--resume-run-dir", default=None)

    ap.add_argument("--date-col", default="date")
    ap.add_argument("--ticker-col", default="Ticker")
    ap.add_argument("--price-col", default="auto")

    ap.add_argument("--month-rule", default="M")
    ap.add_argument("--forecast-horizon-months", type=int, default=12)
    ap.add_argument("--short-horizon-months", type=int, default=3)

    ap.add_argument("--val-months", type=int, default=12)
    ap.add_argument("--min-months-per-ticker", type=int, default=36)
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

    ap.add_argument("--max-optuna-rows", type=int, default=0, help="0 means no row limit")
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 means no row limit")

    ap.add_argument("--save-panel", type=int, default=1)
    ap.add_argument("--save-validation-predictions", type=int, default=1)
    ap.add_argument("--save-future-forecast", type=int, default=1)

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

    if cfg.resume_run_dir:
        out_dir = Path(cfg.resume_run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[RESUME] using existing run_dir:", out_dir.resolve())
    else:
        out_dir = Path(cfg.out_root) / run_id()
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[RUN] Global XGBoost MONTHLY forecast")
    print("[RUN] out_dir:", out_dir.resolve())
    print("[RUN] XGBoost:", xgb.__version__)
    print("[RUN] use_gpu:", cfg.use_gpu, "gpu_id:", cfg.gpu_id)
    print("[RUN] n_jobs:", cfg.n_jobs, "numba_threads:", get_num_threads())
    print("[RUN] eval_end_date:", cfg.eval_end_date, "eval_years:", cfg.eval_years)
    print("[RUN] macro_known_until:", cfg.macro_known_until)
    print("[RUN] short_horizon_months:", cfg.short_horizon_months, "(macro used only up to this horizon)")
    print("=" * 80)

    raw, price_col = load_prices(cfg)
    monthly = make_monthly_prices(raw, price_col, cfg)

    panel, feat_cols, mappings = build_panel(monthly, cfg, out_dir)
    short_feat_cols, long_feat_cols, macro_excluded = split_feature_sets(feat_cols, out_dir)
    short_horizons, long_horizons = horizon_groups(cfg)

    if cfg.save_panel:
        save_df(panel, out_dir / "monthly_forecast_panel_features")

    eval_end = pd.Timestamp(cfg.eval_end_date)
    eval_start = eval_end - pd.DateOffset(years=int(cfg.eval_years))

    valid_target_to = eval_start - pd.offsets.MonthEnd(1)
    valid_target_from = valid_target_to - pd.DateOffset(months=int(cfg.val_months))

    train_target_before = valid_target_from
    final_train_target_before = eval_start

    print(f"[EVAL] target eval_start={eval_start.date()}, eval_end={eval_end.date()}")
    print(f"[VALID] target valid_from={valid_target_from.date()}, valid_to={valid_target_to.date()}")
    print(f"[TRAIN] optuna train target_before={train_target_before.date()}")
    print(f"[TRAIN] final train target_before={final_train_target_before.date()}")
    print(f"[MODEL SPLIT] short horizons with macro: {short_horizons[0]}..{short_horizons[-1] if short_horizons else 'none'}")
    print(f"[MODEL SPLIT] long horizons without macro: {long_horizons[0] if long_horizons else 'none'}..{long_horizons[-1] if long_horizons else 'none'}")

    best_short = run_optuna(
        train_pairs=panel,
        valid_pairs=panel,
        feat_cols=short_feat_cols,
        cfg=cfg,
        out_dir=out_dir,
        train_target_before=train_target_before,
        valid_target_from=valid_target_from,
        valid_target_to=valid_target_to,
        horizons=short_horizons,
        model_name="short_macro",
    )

    bst_short = train_final(
        df=panel,
        feat_cols=short_feat_cols,
        best=best_short,
        cfg=cfg,
        out_dir=out_dir,
        target_before=final_train_target_before,
        horizons=short_horizons,
        model_name="short_macro",
    )

    bst_long = None
    if long_horizons:
        best_long = run_optuna(
            train_pairs=panel,
            valid_pairs=panel,
            feat_cols=long_feat_cols,
            cfg=cfg,
            out_dir=out_dir,
            train_target_before=train_target_before,
            valid_target_from=valid_target_from,
            valid_target_to=valid_target_to,
            horizons=long_horizons,
            model_name="long_no_macro",
        )

        bst_long = train_final(
            df=panel,
            feat_cols=long_feat_cols,
            best=best_long,
            cfg=cfg,
            out_dir=out_dir,
            target_before=final_train_target_before,
            horizons=long_horizons,
            model_name="long_no_macro",
        )

    validate(
        bst_short=bst_short,
        bst_long=bst_long,
        valid=panel,
        short_feat_cols=short_feat_cols,
        long_feat_cols=long_feat_cols,
        cfg=cfg,
        out_dir=out_dir,
        target_from=valid_target_from,
        target_to=valid_target_to,
    )

    evaluate_fixed_origin_forecast(
        bst_short=bst_short,
        bst_long=bst_long,
        panel=panel,
        short_feat_cols=short_feat_cols,
        long_feat_cols=long_feat_cols,
        cfg=cfg,
        out_dir=out_dir,
    )

    if cfg.save_future_forecast:
        forecast_future(
            bst_short=bst_short,
            bst_long=bst_long,
            panel=panel,
            short_feat_cols=short_feat_cols,
            long_feat_cols=long_feat_cols,
            cfg=cfg,
            out_dir=out_dir,
        )

    print("\n[DONE]")
    print("Results:", out_dir.resolve())
    print("Backtest overall:", (out_dir / "backtest_last_year_monthly_metrics_overall.json").resolve())
    print("Backtest predictions:", (out_dir / "backtest_last_year_monthly_predictions.csv").resolve())


if __name__ == "__main__":
    main()
