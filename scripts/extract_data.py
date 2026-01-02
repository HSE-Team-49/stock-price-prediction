from __future__ import annotations

import argparse

from mlcore.features import (
    load_prices_csv,
    pick_price_col,
    filter_tickers_starting_at_global_min,
    make_monthly_panel,
    get_feature_columns,
)
from mlcore.store_sqlite import SQLiteStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to prices_all.csv")
    ap.add_argument("--db", required=True, help="Path to SQLite db, e.g. storage/app.db")
    ap.add_argument("--price-col", default="auto", help="auto | Close | Adj Close")
    args = ap.parse_args()

    print("[extract] load csv...")
    df = load_prices_csv(args.csv)
    price_col = pick_price_col(df, pref=args.price_col)

    print("[extract] filter tickers by global start...")
    df = filter_tickers_starting_at_global_min(df)
    print("[extract] tickers:", df["Ticker"].nunique())

    print("[extract] build monthly panel...")
    panel = make_monthly_panel(df, price_col=price_col)
    feat_cols = get_feature_columns(panel)
    print("[extract] rows:", len(panel), "features:", len(feat_cols))

    store = SQLiteStore(args.db)
    store.replace_features(panel, feat_cols)
    print("[extract] done. feature_store replaced.")


if __name__ == "__main__":
    main()
