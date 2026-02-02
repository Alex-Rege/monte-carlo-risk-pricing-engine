#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.crypto_data_sources import (
    compute_crypto_daily_features,
    compute_crypto_hourly_features,
    compute_hourly_gap_report,
    fetch_binance_klines,
    prepare_crypto_daily,
    prepare_crypto_hourly,
    prepare_raw_snapshot,
    write_crypto_metadata,
    write_daily_csv,
    write_hourly_csv,
    write_raw_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull BTC spot data from Binance.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (default: BTCUSDT).")
    parser.add_argument("--start", default="2017-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--intervals",
        default="1d,1h",
        help="Comma-separated intervals to pull (default: 1d,1h).",
    )
    return parser.parse_args()


def _interval_list(value: str) -> list[str]:
    intervals = [item.strip() for item in value.split(",") if item.strip()]
    if not intervals:
        raise ValueError("At least one interval is required.")
    return intervals


def _symbol_prefix(symbol: str) -> str:
    if symbol.upper().endswith("USDT"):
        return symbol[:-4].lower()
    return symbol.lower()


def main() -> None:
    args = parse_args()
    end_date = args.end or datetime.now(timezone.utc).date().isoformat()
    start_date = args.start
    symbol = args.symbol.upper()
    intervals = _interval_list(args.intervals)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    reports_dir = Path("data/reports")

    pull_timestamp = datetime.now(timezone.utc)
    metadata: dict[str, object] = {
        "source": "Binance",
        "symbol": symbol,
        "pull_timestamp_utc": pull_timestamp.isoformat(),
        "intervals": {},
    }

    prefix = _symbol_prefix(symbol)

    for interval in intervals:
        klines = fetch_binance_klines(symbol, interval, start_date, end_date)
        raw_snapshot, duplicate_count = prepare_raw_snapshot(klines)
        raw_path = write_raw_snapshot(
            raw_snapshot,
            symbol,
            interval,
            start_date,
            end_date,
            raw_dir,
            pull_timestamp,
        )

        timestamp_series = pd.to_datetime(raw_snapshot["timestamp"], errors="coerce", utc=True)
        actual_min = timestamp_series.min()
        actual_max = timestamp_series.max()
        metadata_entry = {
            "interval": interval,
            "requested_start": start_date,
            "requested_end": end_date,
            "actual_min_timestamp": actual_min.isoformat() if pd.notna(actual_min) else None,
            "actual_max_timestamp": actual_max.isoformat() if pd.notna(actual_max) else None,
            "row_count": int(len(raw_snapshot)),
            "duplicate_count": int(duplicate_count),
            "output_filename": raw_path.name,
        }
        metadata["intervals"][interval] = metadata_entry

        if interval == "1d":
            daily = prepare_crypto_daily(raw_snapshot)
            daily_features = compute_crypto_daily_features(daily)
            daily_path = write_daily_csv(daily, processed_dir / f"{prefix}_daily.csv")
            daily_features_path = write_daily_csv(
                daily_features, processed_dir / f"{prefix}_features_daily.csv"
            )
            missing_price_count = int(raw_snapshot["price"].isna().sum())
            date_range = (
                f"{daily['date'].min().date().isoformat()} to {daily['date'].max().date().isoformat()}"
                if not daily.empty
                else "no dates"
            )
            print(f"Binance {symbol} {interval} summary")
            print(f"- rows pulled: {len(raw_snapshot)} (duplicates dropped: {duplicate_count})")
            print(f"- date range: {date_range}")
            print(f"- missing prices: {missing_price_count}")
            print("- files written:")
            print(f"  - {raw_path}")
            print(f"  - {daily_path}")
            print(f"  - {daily_features_path}")
        elif interval == "1h":
            hourly = prepare_crypto_hourly(raw_snapshot)
            hourly_features = compute_crypto_hourly_features(hourly)
            hourly_path = write_hourly_csv(hourly, processed_dir / f"{prefix}_hourly.csv")
            hourly_features_path = write_hourly_csv(
                hourly_features, processed_dir / f"{prefix}_features_hourly.csv"
            )
            gap_report = compute_hourly_gap_report(hourly)
            reports_dir.mkdir(parents=True, exist_ok=True)
            gap_path = reports_dir / f"{prefix}_hourly_gaps.json"
            with gap_path.open("w", encoding="utf-8") as handle:
                json.dump(gap_report, handle, indent=2, sort_keys=True)

            missing_price_count = int(raw_snapshot["price"].isna().sum())
            date_range = (
                f"{hourly['timestamp'].min().isoformat()} to {hourly['timestamp'].max().isoformat()}"
                if not hourly.empty
                else "no timestamps"
            )
            print(f"Binance {symbol} {interval} summary")
            print(f"- rows pulled: {len(raw_snapshot)} (duplicates dropped: {duplicate_count})")
            print(f"- timestamp range: {date_range}")
            print(f"- missing prices: {missing_price_count}")
            print(
                f"- hourly gaps: expected {gap_report['expected_count']}, "
                f"missing {gap_report['missing_count']}"
            )
            print("- files written:")
            print(f"  - {raw_path}")
            print(f"  - {hourly_path}")
            print(f"  - {hourly_features_path}")
            print(f"  - {gap_path}")
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    metadata_path = write_crypto_metadata(metadata, raw_dir)
    print(f"- metadata updated: {metadata_path}")


if __name__ == "__main__":
    main()
