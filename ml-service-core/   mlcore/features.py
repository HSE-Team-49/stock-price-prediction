from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


LAGS = [1, 2, 3, 6, 9, 12]
ROLL_WINDOWS = [3, 6, 12]
ATR_WINDOWS = [6, 12]
RSI_PERIODS = [6, 12]


def load_prices_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    ren = {}
    for want in ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                ren[c] = want
                break
    df = df.rename(columns=ren)

    if "date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("CSV must contain columns: 'date' and 'Ticker'")

    df["date"] = pd.to_datetime(df["date"])
    return df


def pick_price_col(df: pd.DataFrame, pref: str = "auto") -> str:
    if pref != "auto":
        if pref in df.columns:
            return pref
        raise ValueError(f"PRICE_COL_PREF='{pref}' not in columns")
    return "Adj Close" if "Adj Close" in df.columns else "Close"


def filter_tickers_starting_at_global_min(df: pd.DataFrame) -> pd.DataFrame:
    gmin = df["date"].min().normalize()
    firsts = df.groupby("Ticker")["date"].min().dt.normalize()
    keep = set(firsts[firsts == gmin].index)
    return df[df["Ticker"].isin(keep)].copy()


def compute_atr(hi: pd.Series, lo: pd.Series, close: pd.Series, ws: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ws, adjust=False).mean()
    return atr


def compute_rsi(ret: pd.Series, period: int) -> pd.Series:
    delta = ret.copy()
    gain = (delta.clip(lower=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def make_monthly_ohlcv(df: pd.DataFrame, price_col: str) -> Dict[str, pd.DataFrame]:
    piv = {
        col: df.pivot(index="date", columns="Ticker", values=col)
        for col in ["Open", "High", "Low", "Close", "Volume"]
        if col in df.columns
    }
    if "Adj Close" in df.columns:
        piv["Adj Close"] = df.pivot(index="date", columns="Ticker", values="Adj Close")

    for k in piv:
        piv[k] = piv[k].sort_index()

    monthly: Dict[str, pd.DataFrame] = {}
    if "Open" in piv:
        monthly["Open"] = piv["Open"].resample("ME").first()
    if "High" in piv:
        monthly["High"] = piv["High"].resample("ME").max()
    if "Low" in piv:
        monthly["Low"] = piv["Low"].resample("ME").min()
    if "Close" in piv:
        monthly["Close"] = piv["Close"].resample("ME").last()
    if "Adj Close" in piv:
        monthly["Adj Close"] = piv["Adj Close"].resample("ME").last()
    if "Volume" in piv:
        monthly["Volume"] = piv["Volume"].resample("ME").sum()

    if price_col not in monthly:
        raise ValueError(f"price_col '{price_col}' not available in monthly OHLCV")
    monthly["Price"] = monthly[price_col].copy()
    return monthly


def make_market_factor(rets: pd.DataFrame) -> pd.Series:
    return rets.mean(axis=1)


def month_cyc_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    m = index.month
    return pd.DataFrame(
        {"month_sin": np.sin(2 * np.pi * m / 12.0),
         "month_cos": np.cos(2 * np.pi * m / 12.0)},
        index=index,
    )


def make_monthly_panel(df_daily: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Возвращает panel со строками по (Ticker, target_month)
    feature_month = месяц t (из которого сделаны фичи)
    target_next  = ret(t+1)
    target_month = t+1 (месяц, который прогнозируем)

    Важно: target_month — Period('YYYY-MM')
    """
    M = make_monthly_ohlcv(df_daily, price_col)

    price = M["Price"]
    hi = M.get("High", None)
    lo = M.get("Low", None)
    close = M.get("Close", price)
    vol = M.get("Volume", None)

    lnp = np.log(price)
    rets = lnp.diff().dropna(how="all")  

    mkt = make_market_factor(rets)
    cal = month_cyc_features(rets.index)

    atr_map = {W: pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float) for W in ATR_WINDOWS}
    rsi_map = {P: pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float) for P in RSI_PERIODS}

    if (hi is not None) and (lo is not None) and (close is not None):
        for tkr in rets.columns:
            h = hi[tkr].reindex(rets.index)
            l = lo[tkr].reindex(rets.index)
            c = close[tkr].reindex(rets.index)
            for W in ATR_WINDOWS:
                atr_map[W][tkr] = compute_atr(h, l, c, W)
            for P in RSI_PERIODS:
                rsi_map[P][tkr] = compute_rsi(rets[tkr], P)

    dlnv = None
    if vol is not None:
        lnv = np.log(vol.replace(0, np.nan))
        dlnv = lnv.diff()

    rows = []
    for tkr in rets.columns.tolist():
        s = rets[tkr].copy()
        feat = pd.DataFrame({"date": s.index, "Ticker": tkr, "ret": s.values}).set_index("date")

        # lag features
        for L in LAGS:
            feat[f"ret_lag{L}"] = feat["ret"].shift(L)

        # rolling
        for W in ROLL_WINDOWS:
            feat[f"ret_roll_mean_{W}"] = feat["ret"].rolling(W).mean()
            feat[f"ret_roll_std_{W}"] = feat["ret"].rolling(W).std()
            feat[f"ret_mom_{W}"] = feat["ret"].rolling(W).sum()

        # ATR / RSI
        for W in ATR_WINDOWS:
            feat[f"atr_{W}"] = atr_map[W][tkr]
        for P in RSI_PERIODS:
            feat[f"rsi_{P}"] = rsi_map[P][tkr]

        # volume features
        if dlnv is not None:
            feat["vol_dln_1"] = dlnv[tkr]
            for W in ROLL_WINDOWS:
                feat[f"vol_roll_std_{W}"] = dlnv[tkr].rolling(W).std()

        # market factor + lags
        feat["mkt_ret"] = mkt.reindex(feat.index)
        for L in [1, 3, 6, 12]:
            feat[f"mkt_ret_lag{L}"] = feat["mkt_ret"].shift(L)

        # calendar
        feat = feat.join(cal)

        # target
        feat["target_next"] = feat["ret"].shift(-1)

        feat = feat.dropna().reset_index()
        rows.append(feat)

    panel = pd.concat(rows, ignore_index=True)
    panel["target_month"] = (panel["date"] + pd.offsets.MonthEnd(1)).dt.to_period("M")
    panel["feature_month"] = panel["date"].dt.to_period("M")

    drop_cols = {"date", "ret"}
    feat_cols = [c for c in panel.columns if c not in drop_cols | {"Ticker", "target_next", "target_month", "feature_month"}]

    panel = panel[["Ticker", "feature_month", "target_month", "target_next"] + feat_cols] \
        .sort_values(["Ticker", "target_month"]) \
        .reset_index(drop=True)

    return panel


def get_feature_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns if c not in {"Ticker", "feature_month", "target_month", "target_next"}]
