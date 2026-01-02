from __future__ import annotations

import argparse
import pandas as pd

from mlcore.store_sqlite import SQLiteStore
from mlcore.infer import predict_one, predict_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite db path")
    ap.add_argument("--ticker", default="", help="Ticker, e.g. AAPL")
    ap.add_argument("--target-month", default="", help="YYYY-MM, e.g. 2024-06")
    ap.add_argument("--csv", default="", help="CSV with columns ticker,target_month")
    args = ap.parse_args()

    store = SQLiteStore(args.db)

    if args.csv:
        req = pd.read_csv(args.csv)
        requests = list(zip(req["ticker"].astype(str).tolist(), req["target_month"].astype(str).tolist()))
        out = predict_batch(store, requests)
        print(out.to_csv(index=False))
        return

    if not args.ticker or not args.target_month:
        raise SystemExit("Provide either --csv OR (--ticker AND --target-month)")

    res = predict_one(store, args.ticker, args.target_month)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
