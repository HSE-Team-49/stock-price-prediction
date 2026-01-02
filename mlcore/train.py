from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow

import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from .model_bundle import ModelBundle
from .store_sqlite import SQLiteStore
from .utils_git import get_commit_hash


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _split_holdout_by_target_month(df_train: pd.DataFrame, holdout_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df_train: только target_month < test_year
    if df_train.empty:
        raise ValueError("train df empty")

    # target_month хранится как 'YYYY-MM'
    all_months = sorted(df_train["target_month"].unique().tolist())
    if len(all_months) <= holdout_months + 1:
        raise ValueError("too few months for holdout")

    cutoff = all_months[-holdout_months - 1]
    tr = df_train[df_train["target_month"] <= cutoff].copy()
    va = df_train[df_train["target_month"] > cutoff].copy()
    if tr.empty or va.empty:
        raise ValueError("holdout split empty")
    return tr, va


def optuna_xgb_one_ticker(
    df_train: pd.DataFrame,
    feat_cols: List[str],
    n_trials: int,
    seed: int,
    use_gpu: bool,
    holdout_months: int,
) -> Dict[str, Any]:
    tr, va = _split_holdout_by_target_month(df_train, holdout_months)

    dtr = xgb.DMatrix(tr[feat_cols], label=tr["target_next"])
    dva = xgb.DMatrix(va[feat_cols], label=va["target_next"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
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

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    return {
        "params": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            **{k: v for k, v in best.items() if k != "num_boost_round"},
        },
        "num_boost_round": int(best["num_boost_round"]),
        "best_rmse_holdout": float(study.best_value),
    }


def optuna_cat_one_ticker(
    df_train: pd.DataFrame,
    feat_cols: List[str],
    n_trials: int,
    seed: int,
    use_gpu: bool,
    holdout_months: int,
) -> Dict[str, Any]:
    tr, va = _split_holdout_by_target_month(df_train, holdout_months)

    pool_tr = Pool(tr[feat_cols], tr["target_next"])
    pool_va = Pool(va[feat_cols], va["target_next"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "loss_function": "RMSE",
            "random_seed": seed,
            "allow_writing_files": False,
            "logging_level": "Silent",
            "task_type": "GPU" if use_gpu else "CPU",
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
        if use_gpu:
            params["devices"] = "0"

        iterations = trial.suggest_int("iterations", 500, 2000)

        model = CatBoostRegressor(**params, iterations=iterations)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
        pred = model.predict(pool_va)
        return float(np.sqrt(mean_squared_error(va["target_next"].values, pred)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    cfg = {
        "params": {
            "loss_function": "RMSE",
            "random_seed": seed,
            "allow_writing_files": False,
            "logging_level": "Silent",
            "task_type": "GPU" if use_gpu else "CPU",
            **{k: v for k, v in best.items() if k != "iterations"},
        },
        "iterations": int(best["iterations"]),
        "best_rmse_holdout": float(study.best_value),
    }
    if use_gpu:
        cfg["params"]["devices"] = "0"
    return cfg


def walk_forward_predict_one_ticker_xgb(
    df_t: pd.DataFrame,
    feat_cols: List[str],
    best_cfg: Dict[str, Any],
    test_year: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Для каждого target_month в test_year:
      train = target_month < current
      fit from scratch
      predict current month
    Возвращаем preds_df и fitted_models (последний обученный на всей истории до конца) для deploy.
    """
    df_t = df_t.sort_values("target_month").reset_index(drop=True)
    months = sorted(df_t["target_month"].unique().tolist())
    test_months = [m for m in months if m.startswith(f"{test_year}-")]

    preds = []
    fitted_last_model = None

    for tm in test_months:
        train_df = df_t[df_t["target_month"] < tm]
        cur_df = df_t[df_t["target_month"] == tm]
        if train_df.empty or cur_df.empty:
            continue

        dtr = xgb.DMatrix(train_df[feat_cols], label=train_df["target_next"])
        model = xgb.train(best_cfg["params"], dtr, num_boost_round=int(best_cfg["num_boost_round"]))
        fitted_last_model = model

        dcur = xgb.DMatrix(cur_df[feat_cols])
        yhat = model.predict(dcur)

        preds.append(pd.DataFrame({
            "ticker": cur_df["ticker"].values,
            "target_month": cur_df["target_month"].values,
            "y_true": cur_df["target_next"].values,
            "y_pred": yhat,
        }))

    preds_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return preds_df, {"model": fitted_last_model}


def walk_forward_predict_one_ticker_cat(
    df_t: pd.DataFrame,
    feat_cols: List[str],
    best_cfg: Dict[str, Any],
    test_year: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_t = df_t.sort_values("target_month").reset_index(drop=True)
    months = sorted(df_t["target_month"].unique().tolist())
    test_months = [m for m in months if m.startswith(f"{test_year}-")]

    preds = []
    fitted_last_model = None

    base_params = dict(best_cfg["params"])
    iterations = int(best_cfg["iterations"])

    for tm in test_months:
        train_df = df_t[df_t["target_month"] < tm]
        cur_df = df_t[df_t["target_month"] == tm]
        if train_df.empty or cur_df.empty:
            continue

        model = CatBoostRegressor(**base_params, iterations=iterations)
        model.fit(train_df[feat_cols], train_df["target_next"], verbose=False)
        fitted_last_model = model

        yhat = model.predict(cur_df[feat_cols])

        preds.append(pd.DataFrame({
            "ticker": cur_df["ticker"].values,
            "target_month": cur_df["target_month"].values,
            "y_true": cur_df["target_next"].values,
            "y_pred": yhat,
        }))

    preds_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return preds_df, {"model": fitted_last_model}


@dataclass
class RetrainOutputs:
    experiment_id: int
    run_id: Optional[str]
    model_path: str
    overall: Dict[str, float]
    per_ticker: pd.DataFrame
    preds: pd.DataFrame


def retrain_and_log(
    store: SQLiteStore,
    model_kind: str,                 # "xgb" or "cat"
    test_year: int,
    holdout_months: int,
    n_trials: int,
    use_gpu_xgb: bool,
    use_gpu_cat: bool,
    random_state: int,
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment_name: str,
    out_dir: str,
) -> RetrainOutputs:
    """
    1) читаем feature_store из SQLite
    2) тюним по тикерам (только target_month < test_year)
    3) делаем walk-forward предикты на test_year
    4) логируем в MLflow: params/metrics/artifacts + сохраняем model bundle
    5) пишем experiments row в SQLite
    """
    os.makedirs(out_dir, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    commit_hash = get_commit_hash()

    panel = store.load_panel_for_training()
    if panel.empty:
        raise ValueError("feature_store empty: run scripts/extract_data.py first")

    # target_month как TEXT 'YYYY-MM' уже в базе
    feat_cols = [c for c in panel.columns if c not in {"ticker", "feature_month", "target_month", "target_next"}]

    tickers = sorted(panel["ticker"].unique().tolist())

    all_preds = []
    per_ticker_rows = []
    per_ticker_cfg: Dict[str, Dict[str, Any]] = {}
    per_ticker_models: Dict[str, Any] = {}

    # MLflow setup
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{model_kind}_retrain_{created_at}") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "model_kind": model_kind,
            "test_year": test_year,
            "holdout_months": holdout_months,
            "n_trials": n_trials,
            "random_state": random_state,
            "use_gpu_xgb": use_gpu_xgb,
            "use_gpu_cat": use_gpu_cat,
            "commit_hash": commit_hash,
        })

        t0_all = time.perf_counter()

        for tkr in tickers:
            df_t = panel[panel["ticker"] == tkr].copy()
            # train only: target_month < test_year
            df_train = df_t[df_t["target_month"] < f"{test_year}-01"].copy()

            # если у тикера нет 2024 — пропускаем
            has_test = df_t["target_month"].str.startswith(f"{test_year}-").any()
            if df_train.empty or not has_test:
                continue

            # Optuna
            if model_kind == "xgb":
                cfg = optuna_xgb_one_ticker(
                    df_train=df_train,
                    feat_cols=feat_cols,
                    n_trials=n_trials,
                    seed=random_state,
                    use_gpu=use_gpu_xgb,
                    holdout_months=holdout_months,
                )
                preds_t, fitted = walk_forward_predict_one_ticker_xgb(
                    df_t=df_t, feat_cols=feat_cols, best_cfg=cfg, test_year=test_year
                )
                model_obj = fitted["model"]
            elif model_kind == "cat":
                cfg = optuna_cat_one_ticker(
                    df_train=df_train,
                    feat_cols=feat_cols,
                    n_trials=n_trials,
                    seed=random_state,
                    use_gpu=use_gpu_cat,
                    holdout_months=holdout_months,
                )
                preds_t, fitted = walk_forward_predict_one_ticker_cat(
                    df_t=df_t, feat_cols=feat_cols, best_cfg=cfg, test_year=test_year
                )
                model_obj = fitted["model"]
            else:
                raise ValueError(f"Unknown model_kind={model_kind}")

            if preds_t.empty or model_obj is None:
                continue

            per_ticker_cfg[tkr] = cfg
            per_ticker_models[tkr] = model_obj

            m = metrics(preds_t["y_true"].values, preds_t["y_pred"].values)
            per_ticker_rows.append({
                "ticker": tkr,
                "n_test": int(len(preds_t)),
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "R2": m["R2"],
                "best_holdout_rmse": float(cfg.get("best_rmse_holdout", np.nan)),
            })

            all_preds.append(preds_t)

        if not all_preds:
            raise ValueError("No predictions produced. Check data coverage / test_year.")

        preds = pd.concat(all_preds, ignore_index=True)
        per_ticker_df = pd.DataFrame(per_ticker_rows).sort_values("RMSE").reset_index(drop=True)
        overall = metrics(preds["y_true"].values, preds["y_pred"].values)

        mlflow.log_metrics({
            "overall_rmse": overall["RMSE"],
            "overall_mae": overall["MAE"],
            "overall_r2": overall["R2"],
            "n_rows": int(len(preds)),
            "n_tickers": int(preds["ticker"].nunique()),
        })

        # save artifacts
        preds_path = os.path.join(out_dir, f"{model_kind}_preds_test_{test_year}.csv")
        per_ticker_path = os.path.join(out_dir, f"{model_kind}_per_ticker_metrics_test_{test_year}.csv")
        summary_path = os.path.join(out_dir, f"{model_kind}_summary_test_{test_year}.json")

        preds.to_csv(preds_path, index=False)
        per_ticker_df.to_csv(per_ticker_path, index=False)

        summary = {
            "model_kind": model_kind,
            "test_year": test_year,
            "overall": overall,
            "n_rows": int(len(preds)),
            "n_tickers": int(preds["ticker"].nunique()),
            "holdout_months": holdout_months,
            "n_trials": n_trials,
            "created_at": created_at,
            "commit_hash": commit_hash,
            "run_id": run_id,
        }
        import json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(preds_path, artifact_path="artifacts")
        mlflow.log_artifact(per_ticker_path, artifact_path="artifacts")
        mlflow.log_artifact(summary_path, artifact_path="artifacts")

        # save model bundle (deployable)
        bundle = ModelBundle(
            model_kind=model_kind,
            created_at=created_at,
            commit_hash=commit_hash,
            experiment_name=mlflow_experiment_name,
            feature_columns=feat_cols,
            per_ticker_params=per_ticker_cfg,
            models=per_ticker_models,
        )
        model_path = os.path.join(out_dir, f"{model_kind}_bundle_{test_year}_{int(time.time())}.joblib")
        bundle.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        dt_all = time.perf_counter() - t0_all
        mlflow.log_metric("train_total_seconds", float(dt_all))

    # write experiment to SQLite
    experiment_id = store.insert_experiment(
        created_at=created_at,
        model_kind=model_kind,
        test_year=test_year,
        holdout_months=holdout_months,
        n_trials=n_trials,
        run_id=run_id,
        experiment_name=mlflow_experiment_name,
        commit_hash=commit_hash,
        model_path=model_path,
        metrics={"overall": overall, "n_rows": int(len(preds)), "n_tickers": int(preds["ticker"].nunique())},
    )

    return RetrainOutputs(
        experiment_id=experiment_id,
        run_id=run_id,
        model_path=model_path,
        overall=overall,
        per_ticker=per_ticker_df,
        preds=preds,
    )
