from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor, Pool

DATA_DIR = "data"
ALL_CSV = os.path.join(DATA_DIR, "prices_all.csv")

PRICE_COL_PREF = "auto"

TEST_YEAR = 2024
TRAIN_END = pd.Timestamp(f"{TEST_YEAR-1}-12-31")

# Optuna
N_TRIALS_XGB = 40
N_TRIALS_CAT = 40
RANDOM_STATE = 13

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



def compute_atr(hi: pd.Series, lo: pd.Series, close: pd.Series, ws: int) -> pd.Series:
    """
    ATR (Wilder).
    TR_t = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EMA(TR, alpha=1/ws) (Wilder smoothing)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [hi - lo, (hi - prev_close).abs(), (lo - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ws, adjust=False).mean()
    return atr


def compute_rsi(ret: pd.Series, period: int) -> pd.Series:
    """
    RSI (Wilder) на базе месячных лог-доходностей.
    """
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
    """Рыночный фактор: equal-weight средняя месячная доходность по тикерам."""
    return rets.mean(axis=1)


def month_cyc_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    m = index.month
    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * m / 12.0),
            "month_cos": np.cos(2 * np.pi * m / 12.0),
        },
        index=index,
    )


def make_monthly_panel(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    1) Дневные OHLCV -> месячные OHLCV
    2) Лог-цены -> лог-доходности ret
    3) Фичи: лаги, rolling, ATR, RSI, объём, рыночный фактор, календарные sin/cos
    4) target_next = ret.shift(-1)
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

    # ATR/RSI матрицы на тикеры
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

    # объёмы
    dlnv = None
    if vol is not None:
        lnv = np.log(vol.replace(0, np.nan))
        dlnv = lnv.diff()

    rows = []
    for t in rets.columns.tolist():
        s = rets[t].copy()

        feat = pd.DataFrame({"date": s.index, "Ticker": t, "ret": s.values}).set_index("date")

        # лаги
        for L in LAGS:
            feat[f"ret_lag{L}"] = feat["ret"].shift(L)

        # rolling по ретам
        for W in ROLL_WINDOWS:
            feat[f"ret_roll_mean_{W}"] = feat["ret"].rolling(W).mean()
            feat[f"ret_roll_std_{W}"] = feat["ret"].rolling(W).std()
            feat[f"ret_mom_{W}"] = feat["ret"].rolling(W).sum()

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

        # рыночный фактор + лаги
        feat["mkt_ret"] = mkt.reindex(feat.index)
        for L in [1, 3, 6, 12]:
            feat[f"mkt_ret_lag{L}"] = feat["mkt_ret"].shift(L)

        # календарные
        feat = feat.join(cal)

        # таргет
        feat["target_next"] = feat["ret"].shift(-1)

        feat = feat.dropna().reset_index()
        rows.append(feat)

    panel = pd.concat(rows, ignore_index=True)

    cols = ["date", "Ticker", "target_next"] + [
        c for c in panel.columns if c not in {"date", "Ticker", "ret", "target_next"}
    ]
    panel = panel[cols].sort_values(["date", "Ticker"]).reset_index(drop=True)
    return panel


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }



def _split_last_12m_holdout(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_train_date = train_df["date"].max()
    cutoff = last_train_date - pd.offsets.MonthEnd(12)
    tr = train_df[train_df["date"] <= cutoff].copy()
    va = train_df[train_df["date"] > cutoff].copy()
    if tr.empty or va.empty:
        raise SystemExit("Слишком мало данных в train для hold-out на последние 12 месяцев.")
    return tr, va


def optuna_xgb(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict:
    tr, va = _split_last_12m_holdout(train_df)

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
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
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


def optuna_cat(train_df: pd.DataFrame, feat_cols: List[str]) -> Dict:
    tr, va = _split_last_12m_holdout(train_df)

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



def walk_forward_xgb_expanding(panel_all: pd.DataFrame,
                               feat_cols: List[str],
                               xgb_params: Dict,
                               num_boost_round: int,
                               test_year: int) -> pd.DataFrame:
    """
    Expanding window retrain:
    на месяце m:
      - обучаем XGB с нуля на всех месяцах < m (накопленные с самого начала)
      - предсказываем для месяца m
      - затем добавляем месяц m в накопление
    Сохраняем предсказания только для месяцев test_year.
    """
    panel_all = panel_all.sort_values(["date", "Ticker"]).reset_index(drop=True)
    months = sorted(panel_all["date"].dt.to_period("M").unique())

    preds = []
    train_cum = []

    for m in months:
        cur = panel_all[panel_all["date"].dt.to_period("M") == m]
        if cur.empty:
            continue

        if len(train_cum) > 0:
            train_df = pd.concat(train_cum, ignore_index=True)
            dtr = xgb.DMatrix(train_df[feat_cols], label=train_df["target_next"])
            model = xgb.train(xgb_params, dtr, num_boost_round=num_boost_round)

            dcur = xgb.DMatrix(cur[feat_cols])
            yhat = model.predict(dcur)

            if int(m.year) == int(test_year):
                preds.append(pd.DataFrame({
                    "date": cur["date"].values,
                    "Ticker": cur["Ticker"].values,
                    "y_true": cur["target_next"].values,
                    "y_pred": yhat,
                    "phase": "pred_before_update"
                }))

        # "прошли месяц" -> добавили в накопление (веса изменятся на следующем шаге)
        train_cum.append(cur)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def walk_forward_cat_expanding(panel_all: pd.DataFrame,
                               feat_cols: List[str],
                               cat_cfg: Dict,
                               test_year: int,
                               holdout_months: int = 12,
                               max_iters_wf: int = 1200,
                               log_every: int = 1) -> pd.DataFrame:
    """
    Expanding window retrain для CatBoost (каждый месяц переобучаемся с нуля на 1..t-1),
    но с ускорением:
      - hold-out последние `holdout_months` месяцев внутри накопленного train
      - early stopping через use_best_model + od_type/od_wait
      - потолок iterations для walk-forward: max_iters_wf

    Прозрачный лог:
      - месяц шага
      - месяцев в train (уникальных)
      - строк в train
      - использовано итераций (model.tree_count_)
      - время шага (сек)
    """
    import time

    panel_all = panel_all.sort_values(["date", "Ticker"]).reset_index(drop=True)
    months = sorted(panel_all["date"].dt.to_period("M").unique())

    preds = []
    train_cum = []

    base_params = dict(cat_cfg["params"])
    iterations = int(min(cat_cfg["iterations"], max_iters_wf))

    base_params.setdefault("od_type", "Iter")
    base_params.setdefault("od_wait", 100)
    base_params.setdefault("allow_writing_files", False)
    base_params.setdefault("logging_level", "Silent")
    base_params.setdefault("task_type", "GPU")
    base_params.setdefault("devices", "0")

    for step_i, m in enumerate(months, 1):
        cur = panel_all[panel_all["date"].dt.to_period("M") == m]
        if cur.empty:
            continue

        # учимся на 1..t-1
        if len(train_cum) > 0:
            train_df = pd.concat(train_cum, ignore_index=True)
            n_months = train_df["date"].dt.to_period("M").nunique()
            n_rows = len(train_df)

            t0 = time.perf_counter()
            model = CatBoostRegressor(**base_params, iterations=iterations, random_seed=RANDOM_STATE)

            used_holdout = False
            if n_months >= (holdout_months + 2):
                used_holdout = True
                last_date = train_df["date"].max()
                cutoff = last_date - pd.offsets.MonthEnd(holdout_months)

                tr = train_df[train_df["date"] <= cutoff]
                va = train_df[train_df["date"] > cutoff]

                if (len(tr) > 0) and (len(va) > 0):
                    pool_tr = Pool(tr[feat_cols], tr["target_next"])
                    pool_va = Pool(va[feat_cols], va["target_next"])
                    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
                else:
                    used_holdout = False
                    pool_all = Pool(train_df[feat_cols], train_df["target_next"])
                    model.fit(pool_all, verbose=False)
            else:
                pool_all = Pool(train_df[feat_cols], train_df["target_next"])
                model.fit(pool_all, verbose=False)

            # predict
            yhat = model.predict(cur[feat_cols])

            # сохранить только test_year
            if int(m.year) == int(test_year):
                preds.append(pd.DataFrame({
                    "date": cur["date"].values,
                    "Ticker": cur["Ticker"].values,
                    "y_true": cur["target_next"].values,
                    "y_pred": yhat,
                    "phase": "pred_before_update"
                }))

            dt = time.perf_counter() - t0
            used_iters = getattr(model, "tree_count_", None)

            if log_every and (step_i % log_every == 0):
                holdout_tag = "holdout+ES" if used_holdout else "no-holdout"
                iters_tag = f"{used_iters}" if used_iters is not None else "?"
                print(
                    f"[WF CAT] step={step_i:3d}/{len(months)} "
                    f"month={m} train_months={n_months:3d} train_rows={n_rows:6d} "
                    f"iters_used={iters_tag:>4s}/{iterations} "
                    f"time={dt:6.2f}s mode={holdout_tag}"
                )

        # добавляем текущий месяц в историю
        train_cum.append(cur)

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()



def main():
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("[INFO] Загружаем данные…")
    raw = load_prices()
    price_col = pick_price_col(raw)

    raw = filter_tickers_starting_at_global_min(raw)
    print(f"[INFO] Тикеров с началом в глобальной дате: {raw['Ticker'].nunique()}")

    print("[INFO] Формируем месячный панель и фичи…")
    panel = make_monthly_panel(raw, price_col)

    # TRAIN = всё до 2024, TEST = 2024
    train = panel[panel["date"] <= TRAIN_END].copy()
    test = panel[panel["date"].dt.year == TEST_YEAR].copy()

    feat_cols = [c for c in panel.columns if c not in {"date", "Ticker", "target_next"}]

    print(f"[INFO] Train ≤ {TRAIN_END.date()}: rows={len(train)}")
    print(f"[INFO] Test {TEST_YEAR}: rows={len(test)}")
    print(f"[INFO] #Features = {len(feat_cols)}  пример: {feat_cols[:8]}")

    print("\n[INFO] Optuna → XGBoost (GPU) на train…")
    best_xgb = optuna_xgb(train, feat_cols)
    print("[INFO] Best XGB:", best_xgb)

    print("\n[INFO] Optuna → CatBoost (GPU) на train…")
    best_cat = optuna_cat(train, feat_cols)
    print("[INFO] Best CAT:", best_cat)

    print(f"\n[INFO] Walk-forward expanding (XGB) → сохраняем предикты только за {TEST_YEAR}…")
    xgb_preds = walk_forward_xgb_expanding(
        panel_all=panel,
        feat_cols=feat_cols,
        xgb_params=best_xgb["params"],
        num_boost_round=best_xgb["num_boost_round"],
        test_year=TEST_YEAR
    )

    print(f"[INFO] Walk-forward expanding (CAT) → сохраняем предикты только за {TEST_YEAR}…")
    cat_preds = walk_forward_cat_expanding(
    panel_all=panel,
    feat_cols=feat_cols,
    cat_cfg=best_cat,
    test_year=TEST_YEAR,
    holdout_months=12,
    max_iters_wf=1200,
    log_every=1      
    )


    if xgb_preds.empty or cat_preds.empty:
        raise SystemExit("Не получилось собрать предсказания (слишком мало месяцев/данных после dropna).")

    xgb_test_m = metrics(xgb_preds["y_true"], xgb_preds["y_pred"])
    cat_test_m = metrics(cat_preds["y_true"], cat_preds["y_pred"])

    print("\n[XGB] Test:", xgb_test_m)
    print("[CAT] Test:", cat_test_m)

    os.makedirs("forecasts", exist_ok=True)

    xgb_path = f"forecasts/xgb_monthly_preds_test_{TEST_YEAR}.csv"
    cat_path = f"forecasts/cat_monthly_preds_test_{TEST_YEAR}.csv"
    sum_path = f"forecasts/summary_metrics_test_{TEST_YEAR}.csv"

    xgb_preds.to_csv(xgb_path, index=False)
    cat_preds.to_csv(cat_path, index=False)

    pd.DataFrame({
        "model": ["XGB-Test", "CAT-Test"],
        "RMSE": [xgb_test_m["RMSE"], cat_test_m["RMSE"]],
        "MAE":  [xgb_test_m["MAE"],  cat_test_m["MAE"]],
        "R2":   [xgb_test_m["R2"],   cat_test_m["R2"]],
        "test_year": [TEST_YEAR, TEST_YEAR],
        "train_end": [str(TRAIN_END.date()), str(TRAIN_END.date())],
    }).to_csv(sum_path, index=False)

    print("\n[INFO] Saved:")
    print(f"  {xgb_path}")
    print(f"  {cat_path}")
    print(f"  {sum_path}")


if __name__ == "__main__":
    main()
