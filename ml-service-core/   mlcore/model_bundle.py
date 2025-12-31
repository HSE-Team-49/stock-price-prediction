from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib


@dataclass
class ModelBundle:
    model_kind: str                  # "xgb" or "cat"
    created_at: str
    commit_hash: str
    experiment_name: str

    feature_columns: List[str]
    per_ticker_params: Dict[str, Dict[str, Any]]   # best_cfg per ticker
    models: Dict[str, Any]                          # fitted model objects per ticker

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError("Loaded object is not a ModelBundle")
        return obj
