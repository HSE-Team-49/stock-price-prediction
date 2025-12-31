from __future__ import annotations

import argparse
import os

from mlcore.store_sqlite import SQLiteStore
from mlcore.train import retrain_and_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite db path")
    ap.add_argument("--model", required=True, choices=["xgb", "cat"], help="Model kind")
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--use-gpu-xgb", action="store_true")
    ap.add_argument("--use-gpu-cat", action="store_true")

    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""), help="MLflow tracking URI")
    ap.add_argument("--mlflow-exp", default="stocks-monthly", help="MLflow experiment name")
    ap.add_argument("--out-dir", default="storage/models", help="Dir to store artifacts/models")
    args = ap.parse_args()

    store = SQLiteStore(args.db)

    out = retrain_and_log(
        store=store,
        model_kind=args.model,
        test_year=args.test_year,
        holdout_months=args.holdout,
        n_trials=args.trials,
        use_gpu_xgb=args.use_gpu_xgb,
        use_gpu_cat=args.use_gpu_cat,
        random_state=args.seed,
        mlflow_tracking_uri=args.mlflow_uri if args.mlflow_uri else None,
        mlflow_experiment_name=args.mlflow_exp,
        out_dir=args.out_dir,
    )

    print("\n[retrain] DONE")
    print("  experiment_id:", out.experiment_id)
    print("  run_id:", out.run_id)
    print("  model_path:", out.model_path)
    print("  overall:", out.overall)


if __name__ == "__main__":
    main()
