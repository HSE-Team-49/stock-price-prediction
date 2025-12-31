from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class ActiveModelInfo:
    active_experiment_id: Optional[int]
    active_model_path: Optional[str]


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  model_kind TEXT NOT NULL,
                  test_year INTEGER NOT NULL,
                  holdout_months INTEGER NOT NULL,
                  n_trials INTEGER NOT NULL,
                  run_id TEXT,
                  experiment_name TEXT,
                  commit_hash TEXT,
                  model_path TEXT,
                  metrics_json TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                  id INTEGER PRIMARY KEY CHECK (id = 1),
                  active_experiment_id INTEGER,
                  active_model_path TEXT,
                  updated_at TEXT
                );
            """)
            conn.execute("""
                INSERT OR IGNORE INTO model_registry (id, active_experiment_id, active_model_path, updated_at)
                VALUES (1, NULL, NULL, NULL);
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  experiment_id INTEGER,
                  model_kind TEXT,
                  ticker TEXT,
                  target_month TEXT,
                  payload_json TEXT,
                  y_pred REAL,
                  latency_ms REAL,
                  error TEXT
                );
            """)

            conn.commit()

    def ensure_feature_store_table(self, feature_cols: List[str]) -> None:
        cols_sql = ",\n".join([f'"{c}" REAL' for c in feature_cols])

        ddl = f"""
        CREATE TABLE IF NOT EXISTS feature_store (
          ticker TEXT NOT NULL,
          feature_month TEXT NOT NULL,
          target_month TEXT NOT NULL,
          target_next REAL,
          {cols_sql},
          PRIMARY KEY (ticker, target_month)
        );
        """
        with self.connect() as conn:
            conn.execute(ddl)
            conn.commit()

    def replace_features(self, panel: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Полная перезаливка (быстро и прозрачно).
        panel columns: Ticker, feature_month (Period), target_month (Period), target_next, feature_cols...
        """
        self.ensure_feature_store_table(feature_cols)

        df = panel.copy()
        df["ticker"] = df["Ticker"].astype(str)
        df["feature_month"] = df["feature_month"].astype(str)
        df["target_month"] = df["target_month"].astype(str)
        df = df.drop(columns=["Ticker"])

        cols = ["ticker", "feature_month", "target_month", "target_next"] + feature_cols
        df = df[cols]

        with self.connect() as conn:
            conn.execute("DELETE FROM feature_store;")
            df.to_sql("feature_store", conn, if_exists="append", index=False)
            conn.commit()

    def fetch_features_for_requests(self, requests: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        requests: list of (ticker, target_month 'YYYY-MM')
        """
        if not requests:
            return pd.DataFrame()

        placeholders = ",".join(["(?, ?)"] * len(requests))
        params: List[Any] = []
        for t, m in requests:
            params.extend([t, m])

        sql = f"""
        SELECT * FROM feature_store
        WHERE (ticker, target_month) IN ({placeholders})
        ORDER BY ticker, target_month;
        """
        with self.connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def load_panel_for_training(self) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query("SELECT * FROM feature_store ORDER BY ticker, target_month;", conn)

    def insert_experiment(
        self,
        created_at: str,
        model_kind: str,
        test_year: int,
        holdout_months: int,
        n_trials: int,
        run_id: Optional[str],
        experiment_name: Optional[str],
        commit_hash: str,
        model_path: str,
        metrics: Dict[str, Any],
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments
                  (created_at, model_kind, test_year, holdout_months, n_trials,
                   run_id, experiment_name, commit_hash, model_path, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at, model_kind, test_year, holdout_months, n_trials,
                    run_id, experiment_name, commit_hash, model_path, json.dumps(metrics, ensure_ascii=False),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_experiment(self, experiment_id: int) -> Dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Experiment id={experiment_id} not found")

        with self.connect() as conn:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(experiments);").fetchall()]
        data = dict(zip(cols, row))
        if data.get("metrics_json"):
            data["metrics"] = json.loads(data["metrics_json"])
        return data

    def set_active_model(self, experiment_id: int, model_path: str, updated_at: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE model_registry
                SET active_experiment_id = ?, active_model_path = ?, updated_at = ?
                WHERE id = 1
                """,
                (experiment_id, model_path, updated_at),
            )
            conn.commit()

    def get_active_model(self) -> ActiveModelInfo:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT active_experiment_id, active_model_path FROM model_registry WHERE id = 1"
            ).fetchone()
        if row is None:
            return ActiveModelInfo(None, None)
        return ActiveModelInfo(row[0], row[1])

    def insert_request_log(
        self,
        ts: str,
        experiment_id: Optional[int],
        model_kind: Optional[str],
        ticker: Optional[str],
        target_month: Optional[str],
        payload: Dict[str, Any],
        y_pred: Optional[float],
        latency_ms: Optional[float],
        error: Optional[str],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO request_logs
                  (ts, experiment_id, model_kind, ticker, target_month, payload_json, y_pred, latency_ms, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, experiment_id, model_kind, ticker, target_month,
                    json.dumps(payload, ensure_ascii=False),
                    y_pred, latency_ms, error,
                ),
            )
            conn.commit()
