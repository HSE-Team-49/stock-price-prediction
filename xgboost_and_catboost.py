from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor, Pool



DATA_DIR = "data"
ALL_CSV = os.path.join(DATA_DIR, "prices_all.csv")

PRICE_COL_PREF = "auto"

# Тестируем целевые месяцы (target_month) 2024
TEST_YEAR = 2024

RANDOM_STATE = 13
np.random.seed(RANDOM_STATE)

# Фичи
LAGS = [1, 2, 3, 6, 9, 12]
ROLL_WINDOWS = [3, 6, 12]
ATR_WINDOWS = [6, 12]
RSI_PERIODS = [6, 12]

# Модель: "xgb" или "cat"
MODEL_KIND = "xgb"

# GPU/CPU
USE_GPU_XGB = True     
USE_GPU_CAT = True     

# Optuna
N_TRIALS_XGB = 25
N_TRIALS_CAT = 25
HOLDOUT_MONTHS = 12

OUT_DIR = "forecasts"
PARAMS_DIR = os.path.join(OUT_DIR, "optuna_params")
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Логи
LOG_EVERY_TICKER = 20
LOG_EVERY_MONTH = 4



def load_prices() -> pd.DataFrame:
    if not os.path.exists(ALL_CSV):
        raise SystemExit(f"Не найден файл {ALL_CSV}")

    df = pd.read_csv(ALL_CSV)

    ren = {}
    for want in ["date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                ren[c] = want
                break
    df = df.rename(columns=ren)

    if "date" not in df.columns or "Ticker" not in df.columns:
        raise SystemExit("Нужны колонки 'date' и 'Ticker'")

    df["date"] = pd.to_datetime(df["date"])
    return df


def pick_price_col(df: pd.DataFrame) -> str:
    if PRICE_COL_PREF != "auto":
        if PRICE_COL_PREF in df.columns:
            return PRICE_COL_PREF
        raise SystemExit(f"PRICE_COL_PREF='{PRICE_COL_PREF}' нет в данных")
    return "Adj Close" if "Adj Close" in df.columns else "Close"


def filter_tickers_starting_at_global_min(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем тикеры, чья первая дата совпадает с глобальной минимальной датой."""
    gmin = df["date"].min().normalize()
    firsts = df.groupby("Ticker")["date"].min().dt.normalize()
    keep = set(firsts[firsts == gmin].index)
    return df[df["Ticker"].isin(keep)].copy()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _params_path(model_kind: str, ticker: str) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(PARAMS_DIR, f"{model_kind}_params_{safe}.json")


def load_cached_params(model_kind: str, ticker: str) -> Optional[Dict]:
    path = _params_path(model_kind, ticker)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_params(model_kind: str, ticker: str, params: Dict) -> None:
    path = _params_path(model_kind, ticker)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)



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


def make_monthly_panel(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    date = feature_month (месяц t)
    target_next = ret(t+1)
    target_month = t+1 (целевой месяц, который мы оцениваем)
    """
    M = make_monthly_ohlcv(df, price_col)

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

        for L in LAGS:
            feat[f"ret_lag{L}"] = feat["ret"].shift(L)

        for W in ROLL_WINDOWS:
            feat[f"ret_roll_mean_{W}"] = feat["ret"].rolling(W).mean()
            feat[f"ret_roll_std_{W}"] = feat["ret"].rolling(W).std()
            feat[f"ret_mom_{W}"] = feat["ret"].rolling(W).sum()

        for W in ATR_WINDOWS:
            feat[f"atr_{W}"] = atr_map[W][tkr]
        for P in RSI_PERIODS:
            feat[f"rsi_{P}"] = rsi_map[P][tkr]

        if dlnv is not None:
            feat["vol_dln_1"] = dlnv[tkr]
            for W in ROLL_WINDOWS:
                feat[f"vol_roll_std_{W}"] = dlnv[tkr].rolling(W).std()

        feat["mkt_ret"] = mkt.reindex(feat.index)
        for L in [1, 3, 6, 12]:
            feat[f"mkt_ret_lag{L}"] = feat["mkt_ret"].shift(L)

        feat = feat.join(cal)

        feat["target_next"] = feat["ret"].shift(-1)

        feat = feat.dropna().reset_index()
        rows.append(feat)

    panel = pd.concat(rows, ignore_index=True)
    panel["target_month"] = (panel["date"] + pd.offsets.MonthEnd(1)).dt.to_period("M")

    feat_cols = [c for c in panel.columns if c not in {"date", "Ticker", "ret", "target_next", "target_month"}]

    panel = panel[["date", "target_month", "Ticker", "target_next"] + feat_cols] \
        .sort_values(["Ticker", "target_month"]) \
        .reset_index(drop=True)

    return panel



def split_train_holdout_by_target_month(df_train: pd.DataFrame, holdout_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    df_train содержит только target_month < TEST_YEAR.
    Делим по последним holdout_months целевых месяцев.
    """
    if df_train.empty:
        raise ValueError("df_train empty")

    last_tm = df_train["target_month"].max()
    cutoff_tm = last_tm - holdout_months  # Period arithmetic

    tr = df_train[df_train["target_month"] <= cutoff_tm].copy()
    va = df_train[df_train["target_month"] > cutoff_tm].copy()

    if tr.empty or va.empty:
        raise SystemExit("Слишком мало месяцев в train для hold-out. Уменьши HOLDOUT_MONTHS или проверь данные.")
    return tr, va



def optuna_xgb_one_ticker(df_train: pd.DataFrame, feat_cols: List[str]) -> Dict:
    tr, va = split_train_holdout_by_target_month(df_train, HOLDOUT_MONTHS)

    dtr = xgb.DMatrix(tr[feat_cols], label=tr["target_next"])
    dva = xgb.DMatrix(va[feat_cols], label=va["target_next"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cuda" if USE_GPU_XGB else "cpu",

            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 1e-3, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 64, 512),
        }
        rounds = trial.suggest_int("num_boost_round", 200, 1200)

        bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "valid")], verbose_eval=False)
        pred = bst.predict(dva)
        return float(np.sqrt(mean_squared_error(va["target_next"].values, pred)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS_XGB, show_progress_bar=False)

    best = study.best_trial.params
    cfg = {
        "params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cuda" if USE_GPU_XGB else "cpu",
            **{k: v for k, v in best.items() if k != "num_boost_round"},
        },
        "num_boost_round": int(best["num_boost_round"]),
        "best_rmse_holdout": float(study.best_value),
    }
    return cfg


def optuna_cat_one_ticker(df_train: pd.DataFrame, feat_cols: List[str]) -> Dict:
    tr, va = split_train_holdout_by_target_month(df_train, HOLDOUT_MONTHS)

    pool_tr = Pool(tr[feat_cols], tr["target_next"])
    pool_va = Pool(va[feat_cols], va["target_next"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "RMSE",
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
            "logging_level": "Silent",

            "task_type": "GPU" if USE_GPU_CAT else "CPU",
            "devices": "0" if USE_GPU_CAT else None,

            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Poisson"]),
            "border_count": trial.suggest_int("border_count", 64, 255),

            "od_type": "Iter",
            "od_wait": trial.suggest_int("od_wait", 50, 200),
        }
        iterations = trial.suggest_int("iterations", 500, 2000)

        if params.get("devices", None) is None:
            params.pop("devices", None)

        model = CatBoostRegressor(**params, iterations=iterations)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        pred = model.predict(pool_va)
        return float(np.sqrt(mean_squared_error(va["target_next"].values, pred)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS_CAT, show_progress_bar=False)

    best = study.best_trial.params
    cfg = {
        "params": {
            "loss_function": "RMSE",
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
            "logging_level": "Silent",
            "task_type": "GPU" if USE_GPU_CAT else "CPU",
            **{k: v for k, v in best.items() if k != "iterations"},
        },
        "iterations": int(best["iterations"]),
        "best_rmse_holdout": float(study.best_value),
    }
    # если CPU — уберём devices
    if not USE_GPU_CAT:
        cfg["params"].pop("devices", None)
    else:
        cfg["params"]["devices"] = "0"
    return cfg


def walk_forward_xgb_ticker(df_t: pd.DataFrame, feat_cols: List[str], best_cfg: Dict) -> pd.DataFrame:
    df_t = df_t.sort_values("target_month").reset_index(drop=True)
    test_months = [m for m in df_t["target_month"].unique().tolist() if m.year == TEST_YEAR]

    preds = []
    tkr = df_t["Ticker"].iloc[0]

    for i, tm in enumerate(test_months, 1):
        train_df = df_t[df_t["target_month"] < tm]
        cur_df = df_t[df_t["target_month"] == tm]

        if train_df.empty or cur_df.empty:
            continue

        dtr = xgb.DMatrix(train_df[feat_cols], label=train_df["target_next"])
        model = xgb.train(best_cfg["params"], dtr, num_boost_round=int(best_cfg["num_boost_round"]))

        dcur = xgb.DMatrix(cur_df[feat_cols])
        yhat = model.predict(dcur)

        preds.append(pd.DataFrame({
            "Ticker": tkr,
            "target_month": cur_df["target_month"].astype(str).values,
            "feature_month": cur_df["date"].dt.to_period("M").astype(str).values,
            "y_true": cur_df["target_next"].values,
            "y_pred": yhat,
        }))

        if LOG_EVERY_MONTH and (i % LOG_EVERY_MONTH == 0):
            m = metrics(cur_df["target_next"].values, yhat)
            print(f"  [{tkr}] {tm}  RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def walk_forward_cat_ticker(df_t: pd.DataFrame, feat_cols: List[str], best_cfg: Dict) -> pd.DataFrame:
    df_t = df_t.sort_values("target_month").reset_index(drop=True)
    test_months = [m for m in df_t["target_month"].unique().tolist() if m.year == TEST_YEAR]

    preds = []
    tkr = df_t["Ticker"].iloc[0]

    base_params = dict(best_cfg["params"])
    iterations = int(best_cfg["iterations"])

    for i, tm in enumerate(test_months, 1):
        train_df = df_t[df_t["target_month"] < tm]
        cur_df = df_t[df_t["target_month"] == tm]

        if train_df.empty or cur_df.empty:
            continue

        model = CatBoostRegressor(**base_params, iterations=iterations)
        model.fit(train_df[feat_cols], train_df["target_next"], verbose=False)

        yhat = model.predict(cur_df[feat_cols])

        preds.append(pd.DataFrame({
            "Ticker": tkr,
            "target_month": cur_df["target_month"].astype(str).values,
            "feature_month": cur_df["date"].dt.to_period("M").astype(str).values,
            "y_true": cur_df["target_next"].values,
            "y_pred": yhat,
        }))

        if LOG_EVERY_MONTH and (i % LOG_EVERY_MONTH == 0):
            m = metrics(cur_df["target_next"].values, yhat)
            print(f"  [{tkr}] {tm}  RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("[INFO] Загружаем данные…")
    raw = load_prices()
    price_col = pick_price_col(raw)

    raw = filter_tickers_starting_at_global_min(raw)
    tickers = sorted(raw["Ticker"].unique().tolist())
    print(f"[INFO] Тикеров с началом в глобальной дате: {len(tickers)}")

    print("[INFO] Формируем месячный панель и фичи…")
    panel = make_monthly_panel(raw, price_col)

    feat_cols = [c for c in panel.columns if c not in {"date", "Ticker", "target_next", "target_month"}]
    print(f"[INFO] #Features = {len(feat_cols)}  пример: {feat_cols[:8]}")
    print(f"[INFO] TEST_YEAR={TEST_YEAR}  HOLDOUT_MONTHS={HOLDOUT_MONTHS}")
    print(f"[INFO] MODEL_KIND={MODEL_KIND}  N_TRIALS_XGB={N_TRIALS_XGB}  N_TRIALS_CAT={N_TRIALS_CAT}")

    all_preds = []
    per_ticker_metrics = []

    for idx, tkr in enumerate(tickers, 1):
        df_t = panel[panel["Ticker"] == tkr].copy()
        if df_t.empty:
            continue

        df_train = df_t[df_t["target_month"].dt.year < TEST_YEAR].copy()
        if df_train.empty:
            continue

        if not (df_t["target_month"].dt.year == TEST_YEAR).any():
            continue

        if idx == 1 or (LOG_EVERY_TICKER and idx % LOG_EVERY_TICKER == 0):
            print(f"\n[INFO] ({idx}/{len(tickers)}) Ticker={tkr}  train_rows={len(df_train)}  total_rows={len(df_t)}")

        cached = load_cached_params(MODEL_KIND, tkr)
        if cached is not None:
            best_cfg = cached
        else:
            if MODEL_KIND == "xgb":
                best_cfg = optuna_xgb_one_ticker(df_train, feat_cols)
            elif MODEL_KIND == "cat":
                best_cfg = optuna_cat_one_ticker(df_train, feat_cols)
            else:
                raise SystemExit(f"Unknown MODEL_KIND={MODEL_KIND}")

            save_cached_params(MODEL_KIND, tkr, best_cfg)

        if MODEL_KIND == "xgb":
            preds_t = walk_forward_xgb_ticker(df_t, feat_cols, best_cfg)
        else:
            preds_t = walk_forward_cat_ticker(df_t, feat_cols, best_cfg)

        if preds_t.empty:
            continue

        m = metrics(preds_t["y_true"].values, preds_t["y_pred"].values)
        per_ticker_metrics.append({
            "Ticker": tkr,
            "n_test": int(len(preds_t)),
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "R2": m["R2"],
            "best_holdout_rmse": float(best_cfg.get("best_rmse_holdout", np.nan)),
        })

        all_preds.append(preds_t)

    if not all_preds:
        raise SystemExit("Не получилось собрать предсказания. Проверь данные/фичи/периоды.")

    preds = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(per_ticker_metrics).sort_values("RMSE").reset_index(drop=True)

    overall = metrics(preds["y_true"].values, preds["y_pred"].values)
    print("\n[INFO] Overall (pooled):", overall)
    print("[INFO] Best tickers by RMSE:")
    print(metrics_df.head(10).to_string(index=False))

    # Save
    preds_path = os.path.join(OUT_DIR, f"{MODEL_KIND}_per_ticker_optuna_walkforward_target_{TEST_YEAR}.csv")
    metrics_path = os.path.join(OUT_DIR, f"{MODEL_KIND}_per_ticker_optuna_metrics_target_{TEST_YEAR}.csv")
    summary_path = os.path.join(OUT_DIR, f"{MODEL_KIND}_optuna_summary_target_{TEST_YEAR}.csv")

    preds.to_csv(preds_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame([{
        "model": MODEL_KIND,
        "test_year": TEST_YEAR,
        "RMSE": overall["RMSE"],
        "MAE": overall["MAE"],
        "R2": overall["R2"],
        "n_rows": int(len(preds)),
        "n_tickers": int(preds["Ticker"].nunique()),
        "holdout_months": HOLDOUT_MONTHS,
        "n_trials": N_TRIALS_XGB if MODEL_KIND == "xgb" else N_TRIALS_CAT,
    }]).to_csv(summary_path, index=False)

    print("\n[INFO] Saved:")
    print(" ", preds_path)
    print(" ", metrics_path)
    print(" ", summary_path)
    print(" ", f"params cache dir: {PARAMS_DIR}")


if __name__ == "__main__":
    main()
