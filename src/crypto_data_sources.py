from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

LOGGER = logging.getLogger(__name__)


def _date_to_str(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _looks_like_date_only(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _parse_timestamp(value: str | date | datetime, *, is_end: bool) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.to_datetime(value, errors="raise")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    is_date_only = False
    if isinstance(value, str) and _looks_like_date_only(value):
        is_date_only = True
    if isinstance(value, date) and not isinstance(value, datetime):
        is_date_only = True

    if is_date_only and is_end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return ts


def _interval_to_millis(interval: str) -> int:
    mapping = {"1d": 24 * 60 * 60 * 1000, "1h": 60 * 60 * 1000}
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def _klines_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    data = pd.DataFrame(rows, columns=columns)
    data = data[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    data["open_time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    data["close_time"] = pd.to_datetime(data["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def fetch_binance_klines(
    symbol: str, interval: str, start: str | date | datetime, end: str | date | datetime
) -> pd.DataFrame:
    start_ts = _parse_timestamp(start, is_end=False)
    end_ts = _parse_timestamp(end, is_end=True)
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    if end_ms < start_ms:
        raise ValueError("End timestamp must be after start timestamp.")

    interval_ms = _interval_to_millis(interval)
    all_rows: list[list[Any]] = []
    next_start = start_ms
    session = requests.Session()

    while next_start <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": next_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        response = session.get(BINANCE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        next_start = last_open_time + interval_ms
        if next_start <= last_open_time:
            raise RuntimeError("Pagination did not advance while fetching klines.")

    data = _klines_to_dataframe(all_rows)
    if interval == "1d":
        data["date"] = data["open_time"].dt.strftime("%Y-%m-%d")
    elif interval == "1h":
        data["datetime"] = data["open_time"].dt.strftime("%Y-%m-%dT%H:00:00Z")
    return data


def prepare_raw_snapshot(klines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = klines.copy()
    data = data.sort_values("open_time")
    duplicate_count = int(data.duplicated("open_time").sum())
    if duplicate_count:
        LOGGER.info("Dropped %s duplicate timestamps from Binance data.", duplicate_count)
        data = data.drop_duplicates(subset=["open_time"], keep="last")
    data = data.reset_index(drop=True)
    snapshot = pd.DataFrame(
        {
            "timestamp": data["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": data["close"].astype(float),
            "open": data["open"].astype(float),
            "high": data["high"].astype(float),
            "low": data["low"].astype(float),
            "volume": data["volume"].astype(float),
        }
    )
    return snapshot, duplicate_count


def write_raw_snapshot(
    snapshot: pd.DataFrame,
    symbol: str,
    interval: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    output_dir: Path,
    pull_timestamp: datetime | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pull_timestamp = pull_timestamp or datetime.now(timezone.utc)
    stamp = pull_timestamp.strftime("%Y-%m-%dT%H%M%SZ")
    start_label = _date_to_str(start_date)
    end_label = _date_to_str(end_date)
    filename = f"binance_{symbol}_{interval}_{start_label}_to_{end_label}_{stamp}.csv"
    path = output_dir / filename
    snapshot.to_csv(path, index=False)
    return path


def write_crypto_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "crypto_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path


def prepare_crypto_daily(raw_snapshot: pd.DataFrame) -> pd.DataFrame:
    data = raw_snapshot.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    data["date"] = data["timestamp"].dt.floor("D")
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.reset_index(drop=True)
    return data[["date", "price"]]


def prepare_crypto_hourly(raw_snapshot: pd.DataFrame) -> pd.DataFrame:
    data = raw_snapshot.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    data = data.drop_duplicates(subset=["timestamp"], keep="last")
    data = data.reset_index(drop=True)
    return data[["timestamp", "price"]]


def compute_crypto_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    prev_price = data["price"].shift(1)
    valid_mask = (data["price"] > 0) & (prev_price > 0)
    data["ret_1d"] = np.where(valid_mask, np.log(data["price"] / prev_price), np.nan)
    for window in (7, 30, 90):
        data[f"vol_{window}d"] = data["ret_1d"].rolling(window=window).std() * np.sqrt(365.0)
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    vol_30d = data["vol_30d"]
    q33 = vol_30d.quantile(0.33)
    q66 = vol_30d.quantile(0.66)

    def label_regime(value: float) -> str | float:
        if pd.isna(value) or pd.isna(q33) or pd.isna(q66):
            return np.nan
        if value < q33:
            return "low"
        if value < q66:
            return "mid"
        return "high"

    data["vol_regime"] = vol_30d.apply(label_regime)
    return data[
        [
            "date",
            "price",
            "ret_1d",
            "vol_7d",
            "vol_30d",
            "vol_90d",
            "month",
            "day_of_week",
            "vol_regime",
        ]
    ]


def compute_crypto_hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    prev_price = data["price"].shift(1)
    valid_mask = (data["price"] > 0) & (prev_price > 0)
    data["ret_1h"] = np.where(valid_mask, np.log(data["price"] / prev_price), np.nan)
    for window in (24, 168, 720):
        data[f"vol_{window}h"] = data["ret_1h"].rolling(window=window).std() * np.sqrt(24.0 * 365.0)
    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    vol_168h = data["vol_168h"]
    q33 = vol_168h.quantile(0.33)
    q66 = vol_168h.quantile(0.66)

    def label_regime(value: float) -> str | float:
        if pd.isna(value) or pd.isna(q33) or pd.isna(q66):
            return np.nan
        if value < q33:
            return "low"
        if value < q66:
            return "mid"
        return "high"

    data["vol_regime"] = vol_168h.apply(label_regime)
    return data[
        [
            "timestamp",
            "price",
            "ret_1h",
            "vol_24h",
            "vol_168h",
            "vol_720h",
            "hour_of_day",
            "day_of_week",
            "vol_regime",
        ]
    ]


def write_daily_csv(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True).dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False)
    return path


def write_hourly_csv(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output.to_csv(path, index=False)
    return path


def compute_hourly_gap_report(hourly: pd.DataFrame) -> dict[str, Any]:
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
    if data.empty:
        return {
            "expected_count": 0,
            "actual_count": 0,
            "missing_count": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
    start = data["timestamp"].iloc[0].floor("h")
    end = data["timestamp"].iloc[-1].floor("h")
    expected = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    expected_count = int(len(expected))
    actual_count = int(data["timestamp"].drop_duplicates().shape[0])
    missing_count = expected_count - actual_count
    return {
        "expected_count": expected_count,
        "actual_count": actual_count,
        "missing_count": missing_count,
        "first_timestamp": start.isoformat(),
        "last_timestamp": end.isoformat(),
    }
