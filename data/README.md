# Data outputs

This project stores lightweight datasets used for notebooks. The commodity volatility
and seasonality notebooks (notebook 40 and possibly notebook 50) rely on WTI crude oil
spot prices from FRED (series ID `DCOILWTICO`).

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

## How to pull the data

Run the script from the repository root:

```
python scripts/pull_commodity_data.py --start 2000-01-01 --end 2025-12-31 --series DCOILWTICO
```

The script writes a raw snapshot plus deterministic processed datasets. No imputation,
interpolation, or smoothing is applied at any stage.

## Intended use

These datasets are meant for illustrative volatility and seasonality analysis in the
commodity notebooks, not for production trading systems.
