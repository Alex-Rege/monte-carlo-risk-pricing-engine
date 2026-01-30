from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

LOGGER = logging.getLogger(__name__)


def _date_to_str(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def build_fred_url(series_id: str, start_date: str, end_date: str) -> str:
    params = {"id": series_id, "cosd": start_date, "coed": end_date}
    return f"{FRED_BASE_URL}?{urlencode(params)}"


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = build_fred_url(series_id, start_date, end_date)
    raw = pd.read_csv(url)
    if raw.shape[1] < 2:
        raise ValueError("FRED response missing expected columns.")
    raw = raw.rename(columns={raw.columns[0]: "date", raw.columns[1]: "price"})
    return raw[["date", "price"]]


def clean_raw_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.sort_values("date")
    duplicate_count = int(data.duplicated("date").sum())
    if duplicate_count:
        LOGGER.info("Dropped %s duplicate dates from raw data.", duplicate_count)
        data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.reset_index(drop=True)
    return data, duplicate_count


def write_raw_snapshot(
    raw: pd.DataFrame,
    series_id: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    pull_timestamp: datetime | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pull_timestamp = pull_timestamp or datetime.now(timezone.utc)
    stamp = pull_timestamp.strftime("%Y-%m-%dT%H%M%SZ")
    filename = f"fred_{series_id}_{start_date}_to_{end_date}_{stamp}.csv"
    path = output_dir / filename
    to_write = raw.copy()
    to_write["date"] = pd.to_datetime(to_write["date"]).dt.strftime("%Y-%m-%d")
    to_write.to_csv(path, index=False)
    return path


def write_raw_metadata(metadata: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return path


def prepare_daily_prices(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["date", "price"]).sort_values("date")
    data = data.drop_duplicates(subset=["date"], keep="last")
    data = data.reset_index(drop=True)
    return data[["date", "price"]]


def compute_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    prev_price = data["price"].shift(1)
    valid_mask = (data["price"] > 0) & (prev_price > 0)
    data["ret_1d"] = np.where(valid_mask, np.log(data["price"] / prev_price), np.nan)
    for window in (20, 60, 252):
        data[f"vol_{window}d"] = (
            data["ret_1d"].rolling(window=window).std() * np.sqrt(252.0)
        )
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    vol_60d = data["vol_60d"]
    q33 = vol_60d.quantile(0.33)
    q66 = vol_60d.quantile(0.66)

    def label_regime(value: float) -> str | float:
        if pd.isna(value) or pd.isna(q33) or pd.isna(q66):
            return np.nan
        if value < q33:
            return "low"
        if value < q66:
            return "mid"
        return "high"

    data["vol_regime"] = vol_60d.apply(label_regime)

    return data[
        [
            "date",
            "price",
            "ret_1d",
            "vol_20d",
            "vol_60d",
            "vol_252d",
            "month",
            "day_of_week",
            "vol_regime",
        ]
    ]


def resample_monthly_prices(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.copy()
    data = data.set_index("date").sort_index()
    monthly = data[["price"]].resample("ME").last()
    monthly = monthly.dropna(subset=["price"]).reset_index()
    prev_price = monthly["price"].shift(1)
    valid_mask = (monthly["price"] > 0) & (prev_price > 0)
    monthly["ret_1m"] = np.where(valid_mask, np.log(monthly["price"] / prev_price), np.nan)
    return monthly[["date", "price", "ret_1m"]]


def compute_monthly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    data = monthly.copy()
    for window in (3, 12):
        data[f"vol_{window}m"] = (
            data["ret_1m"].rolling(window=window).std() * np.sqrt(12.0)
        )
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    return data[
        ["date", "price", "ret_1m", "vol_3m", "vol_12m", "month", "quarter"]
    ]


def write_dataframe_csv(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = data.copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False)
    return path


def build_raw_metadata(
    *,
    series_id: str,
    source: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    pull_timestamp: datetime,
    raw_data: pd.DataFrame,
    output_filename: str,
) -> dict[str, Any]:
    start_str = _date_to_str(start_date)
    end_str = _date_to_str(end_date)
    min_date = raw_data["date"].min()
    max_date = raw_data["date"].max()
    return {
        "source": source,
        "series_id": series_id,
        "pull_timestamp": pull_timestamp.astimezone(timezone.utc).isoformat(),
        "requested_start": start_str,
        "requested_end": end_str,
        "actual_min_date": min_date.date().isoformat() if pd.notna(min_date) else None,
        "actual_max_date": max_date.date().isoformat() if pd.notna(max_date) else None,
        "row_count": int(len(raw_data)),
        "missing_price_count": int(raw_data["price"].isna().sum()),
        "output_filename": output_filename,
    }
