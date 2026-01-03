from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .model_bundle import ModelBundle
from .store_sqlite import SQLiteStore


class ModelNotDeployedError(RuntimeError):
    pass


class FeatureNotFoundError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def predict_one(store: SQLiteStore, ticker: str, target_month: str) -> Dict[str, Any]:
    """
    target_month: 'YYYY-MM'
    """
    
    active = store.get_active_model()
    if not active.active_model_path or active.active_experiment_id is None:
        raise ModelNotDeployedError("Active model is not set. Run scripts/set_active.py")
    
    t0 = time.perf_counter()
    payload = {"ticker": ticker, "target_month": target_month}

    try:
        import xgboost as xgb
        bundle = ModelBundle.load(active.active_model_path)
        
        df = store.fetch_features_for_requests([(ticker, target_month)])
        if df.empty:
            print(f"Features not found for ({ticker}, {target_month})")
            raise FeatureNotFoundError(f"Features not found for ({ticker}, {target_month})")
        
        # feature columns are saved inside bundle to avoid drift
        X = df[bundle.feature_columns]
        
        model = bundle.models.get(ticker)
        
        if model is None:
            raise FeatureNotFoundError(f"No model for ticker={ticker} in active bundle")
        
        dmatrix = xgb.DMatrix(
                data=X.values,  # Конвертируем DataFrame в numpy array
                feature_names=list(X.columns)
            )
        y_pred = float(model.predict(dmatrix)[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        store.insert_request_log(
            ts=_now_iso(),
            experiment_id=int(active.active_experiment_id),
            model_kind=bundle.model_kind,
            ticker=ticker,
            target_month=target_month,
            payload=payload,
            y_pred=y_pred,
            latency_ms=float(latency_ms),
            error=None,
        )
        
        return {
            "ticker": ticker,
            "target_month": target_month,
            "y_pred": y_pred,
            "experiment_id": int(active.active_experiment_id),
            "model_kind": bundle.model_kind,
            "latency_ms": float(latency_ms),
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        store.insert_request_log(
            ts=_now_iso(),
            experiment_id=int(active.active_experiment_id) if active.active_experiment_id is not None else None,
            model_kind=None,
            ticker=ticker,
            target_month=target_month,
            payload=payload,
            y_pred=None,
            latency_ms=float(latency_ms),
            error=str(e),
        )
        raise


def predict_batch(store: SQLiteStore, requests: List[Tuple[str, str]]) -> list:
    """
    requests: list of (ticker, target_month 'YYYY-MM')
    """
    rows = []
    for tkr, tm in requests:
        res = predict_one(store, tkr, tm)
        rows.append(res)
    return rows
