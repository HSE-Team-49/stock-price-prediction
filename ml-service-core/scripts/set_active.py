from __future__ import annotations

import argparse
from datetime import datetime, timezone

from mlcore.store_sqlite import SQLiteStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite db path")
    ap.add_argument("--experiment-id", type=int, required=True)
    args = ap.parse_args()

    store = SQLiteStore(args.db)
    exp = store.get_experiment(args.experiment_id)
    model_path = exp["model_path"]

    ts = datetime.now(timezone.utc).isoformat()
    store.set_active_model(args.experiment_id, model_path, updated_at=ts)

    print("[deploy] active model set")
    print("  experiment_id:", args.experiment_id)
    print("  model_path:", model_path)


if __name__ == "__main__":
    main()
