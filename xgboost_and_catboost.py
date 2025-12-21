from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor, Pool


DATA_DIR = "data"
ALL_CSV = os.path.join(DATA_DIR, "prices_all.csv")

PRICE_COL_PREF = "auto"

TRAIN_END = pd.Timestamp("2022-12-31")
TEST_YEAR = 2023
VAL_YEAR  = 2024

# Optuna
N_TRIALS_XGB = 40
N_TRIALS_CAT = 40
RANDOM_STATE = 13

XGB_UPDATE_ROUNDS = 100
CAT_UPDATE_ROUNDS = 200

# Фичи
LAGS = [1, 2, 3, 6, 9, 12]
ROLL_WINDOWS = [3, 6, 12]
ATR_WINDOWS = [6, 12]
RSI_PERIODS = [6, 12]


def load_prices() -> pd.DataFrame:
    if not os.path.exists(ALL_CSV):
        raise SystemExit(f"Не найден файл {ALL_CSV}")
    df = pd.read_csv(ALL_CSV)

    ren = {}
    for want in ["date","Ticker","Open","High","Low","Close","Adj Close","Volume"]:
        for c in df.columns:
            if c.strip().lower() == want.lower():
                ren[c] = want; break
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
    """Оставляем только тикеры, чья первая дата совпадает с глобальной минимальной датой (обычно 2008-01-xx)."""
    gmin = df["date"].min().normalize()
    firsts = df.groupby("Ticker")["date"].min().dt.normalize()
    keep = set(firsts[firsts == gmin].index)
    return df[df["Ticker"].isin(keep)].copy()

# ---------- Технические индикаторы (на месячных OHLC) ----------
def compute_atr(hi: pd.Series, lo: pd.Series, close: pd.Series, ws: int) -> pd.Series:
    """
    ATR (Wilder). Для месячных: TR_t = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EMA(TR, alpha=1/ws) или rolling mean — возьмём EMA (классика Уайлдера).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev_close).abs(),
        (lo - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder smoothing ~= EMA(alpha=1/ws)
    atr = tr.ewm(alpha=1.0/ws, adjust=False).mean()
    return atr

def compute_rsi(ret: pd.Series, period: int) -> pd.Series:
    """
    RSI Уайлдера на базе месячных лог-ретов: используем приросты ретов как «доходности».
    Классический RSI считается от delta цены; здесь от delta ln(P) эквивалентно.
    """
    delta = ret.copy()
    gain = (delta.clip(lower=0)).ewm(alpha=1.0/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------- Формирование месячного панеля ----------
def make_monthly_ohlcv(df: pd.DataFrame, price_col: str) -> Dict[str, pd.DataFrame]:
    piv = {col: df.pivot(index="date", columns="Ticker", values=col)
           for col in ["Open","High","Low","Close","Volume"] if col in df.columns}
    if "Adj Close" in df.columns:
        piv["Adj Close"] = df.pivot(index="date", columns="Ticker", values="Adj Close")
    for k in piv:
        piv[k] = piv[k].sort_index()

    monthly = {}
    if "Open" in piv:       monthly["Open"]      = piv["Open"].resample("ME").first()
    if "High" in piv:       monthly["High"]      = piv["High"].resample("ME").max()
    if "Low" in piv:        monthly["Low"]       = piv["Low"].resample("ME").min()
    if "Close" in piv:      monthly["Close"]     = piv["Close"].resample("ME").last()
    if "Adj Close" in piv:  monthly["Adj Close"] = piv["Adj Close"].resample("ME").last()
    if "Volume" in piv:     monthly["Volume"]    = piv["Volume"].resample("ME").sum()

    monthly["Price"] = monthly[price_col].copy()
    return monthly


def make_market_factor(rets: pd.DataFrame) -> pd.Series:
    """Простой рыночный фактор: средняя поперёк тикеров месячная доходность (equal-weight)."""
    return rets.mean(axis=1)

def month_cyc_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Циклические календарные: sin/cos по номеру месяца."""
    m = index.month
    return pd.DataFrame({
        "month_sin": np.sin(2*np.pi * m/12.0),
        "month_cos": np.cos(2*np.pi * m/12.0),
    }, index=index)

def make_monthly_panel(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    1) Месячные OHLCV
    2) Лог-цены Price -> лог-реты ret
    3) Фичи: лаги ретов, скользящие статистики, ATR, RSI, объёмы, рыночный фактор, календарные син/кос
    4) Target = next_month ret
    """
    M = make_monthly_ohlcv(df, price_col)

    price = M["Price"]
    hi = M.get("High", None)
    lo = M.get("Low", None)
    close = M.get("Close", price)
    vol = M.get("Volume", None)

    # базовые ряды
    lnp = np.log(price)
    rets = lnp.diff()
    rets = rets.dropna(how="all")

    # рыночный фактор (equal-weight)
    mkt = make_market_factor(rets)

    # календарные
    cal = month_cyc_features(rets.index)

    atr_map = {W: pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float) for W in ATR_WINDOWS}
    rsi_map = {P: pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float) for P in RSI_PERIODS}
    if (hi is not None) and (lo is not None) and (close is not None):
        for t in rets.columns:
            h = hi[t].reindex(rets.index)
            l = lo[t].reindex(rets.index)
            c = close[t].reindex(rets.index)
            for W in ATR_WINDOWS:
                atr_map[W][t] = compute_atr(h, l, c, W)
            for P in RSI_PERIODS:
                rsi_map[P][t] = compute_rsi(rets[t], P)

    # объёмы (лог-изменение и скользящая std)
    dlnv = None
    if vol is not None:
        lnv = np.log(vol.replace(0, np.nan))
        dlnv = lnv.diff()  # лог-изменение объёма


    rows = []
    tickers = rets.columns.tolist()
    for t in tickers:
        s = rets[t].copy()  # ret
        feat = pd.DataFrame({"date": s.index, "Ticker": t, "ret": s.values}).set_index("date")

        # лаги ретов
        for L in LAGS:
            feat[f"ret_lag{L}"] = feat["ret"].shift(L)

        # скользящие по ретам
        for W in ROLL_WINDOWS:
            feat[f"ret_roll_mean_{W}"] = feat["ret"].rolling(W).mean()
            feat[f"ret_roll_std_{W}"]  = feat["ret"].rolling(W).std()
            feat[f"ret_mom_{W}"]       = feat["ret"].rolling(W).sum()

        # ATR/RSI
        for W in ATR_WINDOWS:
            feat[f"atr_{W}"] = atr_map[W][t]
        for P in RSI_PERIODS:
            feat[f"rsi_{P}"] = rsi_map[P][t]

        # объёмы
        if dlnv is not None:
            feat["vol_dln_1"] = dlnv[t]
            for W in ROLL_WINDOWS:
                feat[f"vol_roll_std_{W}"] = dlnv[t].rolling(W).std()

        # рыночный фактор и его лаги
        feat["mkt_ret"] = mkt.reindex(feat.index)
        for L in [1, 3, 6, 12]:
            feat[f"mkt_ret_lag{L}"] = feat["mkt_ret"].shift(L)

        # календарные (одни и те же для всех тикеров)
        feat = feat.join(cal)

        # target_next
        feat["target_next"] = feat["ret"].shift(-1)

        # чистим NaN
        feat = feat.dropna().reset_index()
        rows.append(feat)

    panel = pd.concat(rows, ignore_index=True)
    # финальные колонки
    cols = ["date","Ticker","target_next"] + [c for c in panel.columns if c not in {"date","Ticker","ret","target_next"}]
    panel = panel[cols].sort_values(["date","Ticker"]).reset_index(drop=True)
    return panel

@dataclass
class Splits:
    train: pd.DataFrame
    test: pd.DataFrame
    valid: pd.DataFrame
    feat_cols: List[str]

def split_periods(panel: pd.DataFrame) -> Splits:
    train = panel[panel["date"] <= TRAIN_END].copy()
    test  = panel[panel["date"].dt.year == TEST_YEAR].copy()
    valid = panel[panel["date"].dt.year == VAL_YEAR].copy()
    feat_cols = [c for c in panel.columns if c not in {"date","Ticker","target_next"}]
    return Splits(train, test, valid, feat_cols)

# ---------- Метрики ----------
def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }

# ---------- Optuna: XGBoost ----------
def optuna_xgb(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict:
    last_train_date = train_df["date"].max()
    tr = train_df[train_df["date"] <= last_train_date - pd.offsets.MonthEnd(12)]
    va = train_df[train_df["date"] >  last_train_date - pd.offsets.MonthEnd(12)]
    dtr = xgb.DMatrix(tr[feat_cols], label=tr["target_next"])
    dva = xgb.DMatrix(va[feat_cols], label=va["target_next"])

    def objective(trial: optuna.Trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "device": "cuda",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 1e-3, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha":  trial.suggest_float("alpha",  1e-3, 10.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 64, 512),
        }
        rounds = trial.suggest_int("num_boost_round", 300, 1500)
        bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "valid")], verbose_eval=False)
        pred = bst.predict(dva)
        return np.sqrt(mean_squared_error(va["target_next"], pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)
    best_params = study.best_trial.params

    return {
        "params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "device": "cuda",
            "tree_method": "hist",
            **{k: v for k, v in best_params.items() if k != "num_boost_round"},
        },
        "num_boost_round": int(best_params["num_boost_round"]),
    }


# ---------- Optuna: CatBoost ----------
def optuna_cat(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict:
    # hold-out на последние 12 месяцев train для подбора гиперпараметров
    last_train_date = train_df["date"].max()
    tr = train_df[train_df["date"] <= last_train_date - pd.offsets.MonthEnd(12)]
    va = train_df[train_df["date"] >  last_train_date - pd.offsets.MonthEnd(12)]

    pool_tr = Pool(tr[feat_cols], tr["target_next"])
    pool_va = Pool(va[feat_cols], va["target_next"])

    def objective(trial: optuna.Trial):
        params = {
            "loss_function": "RMSE",
            "task_type": "GPU",
            "devices": "0",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "grow_policy": "SymmetricTree",
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Poisson"]),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "allow_writing_files": False,
            "logging_level": "Silent",
            "gpu_ram_part": trial.suggest_float("gpu_ram_part", 0.7, 0.98),
            "od_type": "Iter",
            "od_wait": trial.suggest_int("od_wait", 50, 200),
        }
        iterations = trial.suggest_int("iterations", 700, 2500)

        model = CatBoostRegressor(**params, iterations=iterations, random_seed=RANDOM_STATE)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

        pred = model.predict(pool_va)
        return np.sqrt(mean_squared_error(va["target_next"], pred))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS_CAT, timeout=3600, show_progress_bar=True)

    best = study.best_trial.params
    best_fixed = {
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "logging_level": "Silent",
        "grow_policy": "SymmetricTree",
        "bootstrap_type": best["bootstrap_type"],
        "depth": best["depth"],
        "learning_rate": best["learning_rate"],
        "l2_leaf_reg": best["l2_leaf_reg"],
        "bagging_temperature": best["bagging_temperature"],
        "random_strength": best["random_strength"],
        "border_count": best["border_count"],
        "gpu_ram_part": best["gpu_ram_part"],
        "od_type": "Iter",
        "od_wait": best["od_wait"],
    }
    return {"params": best_fixed, "iterations": int(best["iterations"])}

def fit_xgb_full(train_df: pd.DataFrame, feat_cols: List[str], best_cfg: Dict) -> xgb.Booster:
    dtr = xgb.DMatrix(train_df[feat_cols], label=train_df["target_next"])
    return xgb.train(best_cfg["params"], dtr, num_boost_round=best_cfg["num_boost_round"])

def fit_cat_full(train_df: pd.DataFrame, feat_cols: List[str], best_cfg: Dict) -> CatBoostRegressor:
    pool_tr = Pool(train_df[feat_cols], train_df["target_next"])
    model = CatBoostRegressor(**best_cfg["params"], iterations=best_cfg["iterations"],
                          random_seed=RANDOM_STATE)
    model.fit(pool_tr)

    return model

def online_predict_xgb(model, base_train, feat_cols, panel_full, years, best_params):
    preds = []
    train_cum = base_train.copy()

    for y in years:
        months = sorted(panel_full[panel_full["date"].dt.year == y]["date"].dt.to_period("M").unique())
        for m in months:
            cur = panel_full[panel_full["date"].dt.to_period("M") == m]
            if cur.empty:
                continue

            dcur = xgb.DMatrix(cur[feat_cols])
            yhat = model.predict(dcur)
            preds.append(pd.DataFrame({
                "date": cur["date"].values,
                "Ticker": cur["Ticker"].values,
                "y_true": cur["target_next"].values,
                "y_pred": yhat,
                "phase": "pred_before_update"
            }))

            train_cum = pd.concat([train_cum, cur], ignore_index=True)
            dtr_new = xgb.DMatrix(train_cum[feat_cols], label=train_cum["target_next"])
            model = xgb.train(best_params, dtr_new,
                              num_boost_round=100,
                              xgb_model=model,
                              verbose_eval=False)
    return pd.concat(preds, ignore_index=True), model


def online_predict_cat(model: CatBoostRegressor,
                       base_train: pd.DataFrame,
                       feat_cols: List[str],
                       panel_full: pd.DataFrame,
                       years: List[int]) -> Tuple[pd.DataFrame, CatBoostRegressor]:
    preds = []
    train_cum = base_train.copy()

    base_params = model.get_params()
    for k in ["iterations", "verbose", "verbose_eval", "silent"]:
        base_params.pop(k, None)
    base_params.update({
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "logging_level": "Silent",
        "grow_policy": "SymmetricTree",
        "od_type": "Iter",
        "od_wait": base_params.get("od_wait", 100),
    })

    best_iters = getattr(model, "tree_count_", None)
    if best_iters is None:
        best_iters = 1500

    for y in years:
        months = sorted(panel_full[panel_full["date"].dt.year == y]["date"].dt.to_period("M").unique())
        for m in months:
            cur = panel_full[panel_full["date"].dt.to_period("M") == m]
            if cur.empty:
                continue

            yhat = model.predict(cur[feat_cols])
            preds.append(pd.DataFrame({
                "date": cur["date"].values,
                "Ticker": cur["Ticker"].values,
                "y_true": cur["target_next"].values,
                "y_pred": yhat,
                "phase": "pred_before_update"
            }))

            train_cum = pd.concat([train_cum, cur], ignore_index=True)

            tc = train_cum.copy()
            last_date = tc["date"].max()
            cutoff = last_date - pd.offsets.MonthEnd(12)
            tr = tc[tc["date"] <= cutoff]
            va = tc[tc["date"] >  cutoff]
            if len(va) < 1:
                pool_tr = Pool(tc[feat_cols], tc["target_next"])
                new_model = CatBoostRegressor(**base_params, iterations=int(best_iters))
                new_model.fit(pool_tr, verbose=False)
            else:
                pool_tr = Pool(tr[feat_cols], tr["target_next"])
                pool_va = Pool(va[feat_cols], va["target_next"])
                new_model = CatBoostRegressor(**base_params, iterations=int(best_iters))
                new_model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)

            model = new_model

    return pd.concat(preds, ignore_index=True), model



def main():
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("[INFO] Загружаем данные…")
    raw = load_prices()
    price_col = pick_price_col(raw)
    raw = filter_tickers_starting_at_global_min(raw)
    print(f"[INFO] Тикеров с началом в глобальной дате: {raw['Ticker'].nunique()}")

    print("[INFO] Формируем месячный панель и расширенные фичи…")
    panel = make_monthly_panel(raw, price_col)

    sp = split_periods(panel)
    print(f"[INFO] Train ≤ {TRAIN_END.date()}: rows={len(sp.train)}")
    print(f"[INFO] Test {TEST_YEAR}: rows={len(sp.test)}")
    print(f"[INFO] Valid {VAL_YEAR}: rows={len(sp.valid)}")
    print(f"[INFO] #Features = {len(sp.feat_cols)}  пример: {sp.feat_cols[:8]}")

    # ============== XGBoost ==============
    print("\n[INFO] Optuna → XGBoost (GPU)…")
    best_xgb = optuna_xgb(sp.train, sp.feat_cols)
    print("[INFO] Best XGB:", best_xgb)

    print("[INFO] Fit XGB on full train…")
    xgb_model = fit_xgb_full(sp.train, sp.feat_cols, best_xgb)

    print("[INFO] Online XGB 2023 → 2024…")
    xgb_preds_test, xgb_model = online_predict_xgb(
    xgb_model, sp.train, sp.feat_cols, sp.test, [TEST_YEAR], best_xgb["params"]
    )
    xgb_preds_val,  xgb_model = online_predict_xgb(
        xgb_model, pd.concat([sp.train, sp.test]), sp.feat_cols, sp.valid, [VAL_YEAR], best_xgb["params"]
    )

    xgb_preds = pd.concat([xgb_preds_test, xgb_preds_val], ignore_index=True)

    xgb_test_m = metrics(xgb_preds[xgb_preds["date"].dt.year == TEST_YEAR]["y_true"],
                         xgb_preds[xgb_preds["date"].dt.year == TEST_YEAR]["y_pred"])
    xgb_val_m  = metrics(xgb_preds[xgb_preds["date"].dt.year == VAL_YEAR]["y_true"],
                         xgb_preds[xgb_preds["date"].dt.year == VAL_YEAR]["y_pred"])
    print("[XGB] Test:", xgb_test_m)
    print("[XGB] Valid:", xgb_val_m)

    # ============== CatBoost ==============
    print("\n[INFO] Optuna → CatBoost (GPU)…")
    best_cat = optuna_cat(sp.train, sp.feat_cols)
    print("[INFO] Best CAT:", best_cat)

    print("[INFO] Fit CAT on full train…")
    cat_model = fit_cat_full(sp.train, sp.feat_cols, best_cat)

    print("[INFO] Online CAT 2023 → 2024…")
    cat_preds_test, cat_model = online_predict_cat(cat_model, sp.train, sp.feat_cols, sp.test, [TEST_YEAR])
    cat_preds_val,  cat_model = online_predict_cat(cat_model,  pd.concat([sp.train, sp.test]), sp.feat_cols, sp.valid, [VAL_YEAR])
    cat_preds = pd.concat([cat_preds_test, cat_preds_val], ignore_index=True)

    cat_test_m = metrics(cat_preds[cat_preds["date"].dt.year == TEST_YEAR]["y_true"],
                         cat_preds[cat_preds["date"].dt.year == TEST_YEAR]["y_pred"])
    cat_val_m  = metrics(cat_preds[cat_preds["date"].dt.year == VAL_YEAR]["y_true"],
                         cat_preds[cat_preds["date"].dt.year == VAL_YEAR]["y_pred"])
    print("[CAT] Test:", cat_test_m)
    print("[CAT] Valid:", cat_val_m)

    # Сохранение
    os.makedirs("forecasts", exist_ok=True)
    xgb_preds.to_csv("forecasts/xgb_monthly_preds.csv", index=False)
    cat_preds.to_csv("forecasts/cat_monthly_preds.csv", index=False)
    pd.DataFrame({
        "model": ["XGB-Test","XGB-Valid","CAT-Test","CAT-Valid"],
        "RMSE": [xgb_test_m["RMSE"], xgb_val_m["RMSE"], cat_test_m["RMSE"], cat_val_m["RMSE"]],
        "MAE":  [xgb_test_m["MAE"],  xgb_val_m["MAE"],  cat_test_m["MAE"],  cat_val_m["MAE"]],
        "R2":   [xgb_test_m["R2"],   xgb_val_m["R2"],   cat_test_m["R2"],   cat_val_m["R2"]],
    }).to_csv("forecasts/summary_metrics.csv", index=False)

    print("\n[INFO] Saved:")
    print("  forecasts/xgb_monthly_preds.csv")
    print("  forecasts/cat_monthly_preds.csv")
    print("  forecasts/summary_metrics.csv")

if __name__ == "__main__":
    main()
