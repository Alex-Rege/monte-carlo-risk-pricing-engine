# Data outputs

This project stores lightweight datasets used for notebooks. The commodity volatility
and seasonality notebooks (notebook 40 and possibly notebook 50) rely on WTI crude oil
spot prices from FRED (series ID `DCOILWTICO`). Notebook 50 also relies on BTC spot
prices from Binance.

## WTI crude oil data (FRED)

**Source:** Federal Reserve Economic Data (FRED), series `DCOILWTICO` (WTI spot price).

**Raw snapshot**
- Location: `data/raw/`
- File pattern: `fred_DCOILWTICO_<start>_to_<end>_<pull_date>.csv`
- Columns: `date`, `price`
- Notes: no imputation or smoothing; missing prices are preserved in the raw snapshot.
- Metadata: `data/raw/metadata.json` (overwritten each pull).

**Processed daily**
- `data/processed/wti_daily.csv`: clean daily prices (no missing prices).
- `data/processed/wti_features_daily.csv`: log returns, rolling vol, seasonality fields,
  and a simple volatility regime label.

**Processed monthly**
- `data/processed/wti_monthly.csv`: month-end prices with monthly log returns.
- `data/processed/wti_features_monthly.csv`: rolling monthly vol plus month/quarter fields.

## BTC spot data (Binance)

**Source:** Binance spot klines for `BTCUSDT`.

**Raw snapshot**
- Location: `data/raw/`
- File patterns:
  - `binance_BTCUSDT_1d_<start>_to_<end>_<pull_date>.csv`
  - `binance_BTCUSDT_1h_<start>_to_<end>_<pull_date>.csv`
- Columns: `timestamp`, `price`, `open`, `high`, `low`, `volume`
- Notes: timestamps are UTC; hourly data is 24/7; no imputation or smoothing.
- Metadata: `data/raw/crypto_metadata.json` (overwritten each pull).

**Processed daily**
- `data/processed/btc_daily.csv`: clean daily prices (no missing prices).
- `data/processed/btc_features_daily.csv`: log returns, rolling vol, seasonality fields,
  and a simple volatility regime label.

**Processed hourly**
- `data/processed/btc_hourly.csv`: clean hourly prices (no missing prices).
- `data/processed/btc_features_hourly.csv`: log returns, rolling vol, time-of-day fields,
  and a simple volatility regime label.
- `data/reports/btc_hourly_gaps.json`: missing-hour diagnostics (expected vs actual hours).

**Annualization conventions**
- Daily volatility uses `sqrt(365)`.
- Hourly volatility uses `sqrt(24*365)`.

## How to pull the data

Run the scripts from the repository root:

```
python scripts/pull_commodity_data.py --start 2000-01-01 --end 2025-12-31 --series DCOILWTICO
python scripts/pull_crypto_data.py --symbol BTCUSDT --start 2017-01-01 --end 2026-02-02
```

The scripts write raw snapshots plus deterministic processed datasets. No imputation,
interpolation, or smoothing is applied at any stage.

## Intended use

These datasets are meant for illustrative volatility and seasonality analysis in the
commodity and crypto notebooks, not for production trading systems.
