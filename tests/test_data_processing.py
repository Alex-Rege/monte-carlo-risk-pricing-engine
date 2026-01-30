import numpy as np
import pandas as pd

from src.data_sources import compute_daily_features, prepare_daily_prices, resample_monthly_prices


def test_log_return_correctness():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    daily = pd.DataFrame({"date": dates, "price": [100.0, 110.0, 121.0]})
    features = compute_daily_features(daily)
    expected = np.log(110.0 / 100.0)
    assert np.isclose(features["ret_1d"].iloc[1], expected)


def test_rolling_vol_window_behavior():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    prices = np.linspace(100.0, 150.0, num=25)
    daily = pd.DataFrame({"date": dates, "price": prices})
    features = compute_daily_features(daily)
    assert features["vol_20d"].iloc[:19].isna().all()
    assert features["vol_20d"].iloc[19:].notna().any()


def test_dates_unique_and_sorted():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-01", "bad-date"],
            "price": [101.0, 100.0, 102.0, 99.0],
        }
    )
    daily = prepare_daily_prices(raw)
    assert daily["date"].is_monotonic_increasing
    assert daily["date"].is_unique


def test_vol_regime_labels_valid():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=80)))
    daily = pd.DataFrame({"date": dates, "price": prices})
    features = compute_daily_features(daily)
    labels = set(features["vol_regime"].dropna().unique())
    assert labels.issubset({"low", "mid", "high"})


def test_monthly_resample_length_reasonable():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    prices = np.linspace(90.0, 110.0, num=90)
    daily = pd.DataFrame({"date": dates, "price": prices})
    monthly = resample_monthly_prices(daily)
    assert len(monthly) < len(daily)
    months = monthly["date"].dt.to_period("M")
    assert months.is_unique
