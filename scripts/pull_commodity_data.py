#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_sources import (
    build_raw_metadata,
    clean_raw_prices,
    compute_daily_features,
    compute_monthly_features,
    fetch_fred_series,
    prepare_daily_prices,
    resample_monthly_prices,
    write_dataframe_csv,
    write_raw_metadata,
    write_raw_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull WTI data from FRED.")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--series", default="DCOILWTICO", help="FRED series ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end or datetime.now(timezone.utc).date().isoformat()
    start_date = args.start
    series_id = args.series

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_response = fetch_fred_series(series_id, start_date, end_date)
    raw_clean, duplicate_count = clean_raw_prices(raw_response)

    pull_timestamp = datetime.now(timezone.utc)
    raw_path = write_raw_snapshot(
        raw_clean, series_id, start_date, end_date, raw_dir, pull_timestamp
    )

    metadata = build_raw_metadata(
        series_id=series_id,
        source="FRED",
        start_date=start_date,
        end_date=end_date,
        pull_timestamp=pull_timestamp,
        raw_data=raw_clean,
        output_filename=raw_path.name,
    )
    metadata_path = write_raw_metadata(metadata, raw_dir)

    daily = prepare_daily_prices(raw_clean)
    daily_features = compute_daily_features(daily)
    monthly = resample_monthly_prices(daily)
    monthly_features = compute_monthly_features(monthly)

    daily_path = write_dataframe_csv(daily, processed_dir / "wti_daily.csv")
    daily_features_path = write_dataframe_csv(
        daily_features, processed_dir / "wti_features_daily.csv"
    )
    monthly_path = write_dataframe_csv(monthly, processed_dir / "wti_monthly.csv")
    monthly_features_path = write_dataframe_csv(
        monthly_features, processed_dir / "wti_features_monthly.csv"
    )

    missing_count = int(raw_clean["price"].isna().sum())
    min_date = raw_clean["date"].min()
    max_date = raw_clean["date"].max()
    date_range = (
        f"{min_date.date().isoformat()} to {max_date.date().isoformat()}"
        if pd.notna(min_date) and pd.notna(max_date)
        else "no dates"
    )

    print("WTI pull summary")
    print(f"- rows pulled: {len(raw_clean)} (duplicates dropped: {duplicate_count})")
    print(f"- date range: {date_range}")
    print(f"- missing prices: {missing_count}")
    print("- files written:")
    print(f"  - {raw_path}")
    print(f"  - {metadata_path}")
    print(f"  - {daily_path}")
    print(f"  - {daily_features_path}")
    print(f"  - {monthly_path}")
    print(f"  - {monthly_features_path}")


if __name__ == "__main__":
    main()
