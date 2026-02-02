import numpy as np
import pandas as pd

from src.crypto_data_sources import (
    compute_crypto_daily_features,
    compute_crypto_hourly_features,
    prepare_crypto_hourly,
)


def test_log_return_correctness_daily():
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    daily = pd.DataFrame({"date": dates, "price": [100.0, 110.0, 121.0]})
    features = compute_crypto_daily_features(daily)
    expected = np.log(110.0 / 100.0)
    assert np.isclose(features["ret_1d"].iloc[1], expected)


def test_log_return_correctness_hourly():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    hourly = pd.DataFrame({"timestamp": timestamps, "price": [200.0, 220.0, 242.0]})
    features = compute_crypto_hourly_features(hourly)
    expected = np.log(220.0 / 200.0)
    assert np.isclose(features["ret_1h"].iloc[1], expected)


def test_rolling_vol_window_behavior():
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    prices = np.linspace(100.0, 110.0, num=10)
    daily = pd.DataFrame({"date": dates, "price": prices})
    features = compute_crypto_daily_features(daily)
    assert features["vol_7d"].iloc[:6].isna().all()
    assert features["vol_7d"].iloc[6:].notna().any()


def test_regime_labels_valid():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    prices = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=100)))
    daily = pd.DataFrame({"date": dates, "price": prices})
    features = compute_crypto_daily_features(daily)
    labels = set(features["vol_regime"].dropna().unique())
    assert labels.issubset({"low", "mid", "high"})


def test_sort_unique_timestamps():
    raw = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T01:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "price": [101.0, 100.0, 102.0],
        }
    )
    hourly = prepare_crypto_hourly(raw)
    assert hourly["timestamp"].is_monotonic_increasing
    assert hourly["timestamp"].is_unique
