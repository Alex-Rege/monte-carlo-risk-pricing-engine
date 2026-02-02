# Monte Carlo Risk & Pricing Engine

This repository implements a **general-purpose Monte Carlo engine for stochastic simulation, pricing, and risk analysis**, designed as a reusable core that can be applied across derivatives pricing, portfolio risk, stress testing, and scenario analysis.

The engine is built around Geometric Brownian Motion (single- and multi-asset), Monte Carlo estimators with confidence intervals, variance-reduction techniques, and analytical benchmarks (Black–Scholes). Emphasis is placed on **numerical correctness, convergence diagnostics, and transparent validation**, rather than on product- or sector-specific customization.

The project follows a layered structure: a **core engine** complemented by **application notebooks** that illustrate how the same machinery adapts to different subfields (trading, risk management, scenario analysis, and stress testing).


*Core idea:* a single stochastic engine feeds multiple estimators (pricing, sensitivities, tail risk), which are then reused across validation and application notebooks.

---

## Where to start (by reviewer profile)

- **General quant / first-time reviewer**  
  → `notebooks/00_core_validation.ipynb`  
  End-to-end validation of the engine: stochastic simulation, pricing, Greeks, tail risk, and PDE benchmarks.

- **Derivatives / volatility trading / hedging**  
  → `00_core_validation.ipynb` → *Greeks section*  
  → `10_volatility_and_greeks.ipynb`  
  Focus on Delta/Vega/Gamma estimation, estimator noise, variance reduction, and sensitivity to volatility assumptions.

- **Market risk / tail risk / stress testing**  
  → `00_core_validation.ipynb` → *VaR & ES section*  
  → `11_tail_risk_and_stress.ipynb`  
  Emphasis on PnL distributions, VaR vs ES behavior, correlation and volatility stress, and tail amplification mechanisms.

- **Private markets / long-horizon risk / discounting**  
  → `20_scenario_and_discount_rate_risk.ipynb`  
  Scenario-based analysis over long horizons, fan charts, downside risk metrics, and discount-rate sensitivity of cashflows.  
  Focus on structural uncertainty rather than backtesting.

- **Insurance / regulatory capital / model risk**  
  → `30_capital_and_model_risk.ipynb`  
  Capital quantiles (VaR/ES), correlation stress impacts, estimation uncertainty, and explicit model-risk framing using alternative assumptions and benchmarks.

- **Energy trading / commodities**  
  → `40_commodity_vol_and_seasonality.ipynb`  
  Empirical analysis of WTI crude oil volatility, seasonality, and regime effects using FRED data, linked back to Monte Carlo pricing and risk metrics.

- **Crypto markets / short-dated options / model limits**  
  → `50_crypto_short_dated_options.ipynb`  
  BTC short-horizon volatility instability, tail diagnostics vs Normal/GBM baselines, and calibration-window sensitivity using Binance spot data.

---

## Core validation

Notebook `notebooks/00_core_validation.ipynb` is the primary entry point and provides a comprehensive validation of the core engine:

* **GBM simulation correctness**  
  Empirical mean and variance of simulated GBM paths are compared against closed-form theoretical moments across multiple time horizons.

* **Correlated multi-asset dynamics**  
  Correlation targeting is validated using PSD-safe correlation handling and factorization, with scatter plots and empirical correlation heatmaps confirming consistency with the input correlation matrix.

* **Monte Carlo pricing vs analytical benchmarks**  
  European call and put prices computed via Monte Carlo are compared against Black–Scholes closed-form prices across a grid of strikes and maturities, highlighting convergence behavior.

* **Statistical error and confidence intervals**  
  Confidence interval width is analyzed as a function of the number of paths, making estimator uncertainty explicit rather than implicit.

* **Variance reduction techniques**  
  Antithetic sampling and control variates are evaluated against plain Monte Carlo, with direct comparisons of convergence rates and confidence interval shrinkage.

* **Greeks estimation and stability**  
  Delta, Vega, and Gamma are estimated using both finite-difference and pathwise Monte Carlo estimators where applicable. Results are validated against analytical Black–Scholes Greeks, with dedicated analysis of estimator noise, bump-size effects, and convergence across seeds.

* **Portfolio VaR and Expected Shortfall (ES)**  
  Correlated asset simulations are aggregated into portfolio PnL distributions, from which empirical VaR and ES are computed across confidence levels. Plots illustrate tail behavior, diversification effects, and the relationship between VaR and ES.

* **PDE pricing benchmark (finite differences)**  
  A Crank–Nicolson finite-difference solver for the Black–Scholes PDE is used as an independent benchmark. PDE prices and sensitivities are compared against both closed-form and Monte Carlo results, with diagnostics highlighting discretization error, grid convergence, and payoff non-smoothness effects.

All figures generated in this notebook are saved under `figures/00_core_validation/` and serve as the quantitative credibility backbone of the project.

Continuous integration runs `pytest` (unit tests) and `ruff` (lint/format) on every push or pull request.

---

## Repository structure

* `src/` — Core simulation, pricing, Greeks estimation, variance reduction, PDE solvers, and risk metric modules.
* `tests/` — Unit tests and numerical validation checks.
* `notebooks/` — Validation and application notebooks (Notebook 00 is the canonical entry point).
* `data/` — Optional datasets (raw and processed) for data-driven notebooks.
* `figures/` — Saved plots generated by notebooks for quick review.

---

## Quickstart

Install dependencies:

```bash
python -m pip install -r requirements.txt

```

Run tests:
```bash
pytest -q
```

Run lint/format checks:
```bash
ruff check .
ruff format --check .
```

Open the core validation notebook:
```bash
jupyter lab notebooks/00_core_validation.ipynb
```

---

## Notebooks overview

* `00_core_validation.ipynb`  
  Canonical entry point. End-to-end validation of the Monte Carlo engine: GBM moments, correlation handling, MC vs Black–Scholes pricing, confidence intervals, variance reduction, Greeks estimation, portfolio VaR/ES, and PDE benchmarks.

* `10_volatility_and_greeks.ipynb`  
  **Derivatives, volatility trading, hedging.**  
  Greeks estimation (finite-difference and pathwise) with detailed noise and stability analysis, variance-reduction effects, and sensitivity to volatility mis-specification. Focus on estimator behavior rather than trading signals.

* `11_tail_risk_and_stress.ipynb`  
  **Market risk, tail risk, stress testing.**  
  Scenario-based stress analysis under volatility and correlation shocks, empirical VaR/ES estimation, and discussion of estimator instability in the tails. Emphasis on tail amplification mechanisms.

* `20_scenario_and_discount_rate_risk.ipynb`  
  **Private markets, long-horizon risk, discounting.**  
  Synthetic long-horizon scenario generation, fan charts, downside-focused risk metrics, and sensitivity of cashflow valuation to discount-rate assumptions. Emphasizes scenario logic and structural uncertainty rather than backtesting.

* `30_capital_and_model_risk.ipynb`  
  **Insurance, regulatory capital, model risk.**  
  Capital quantiles (VaR/ES across confidence levels), correlation stress impacts, and estimation uncertainty (e.g. via repeated simulations). Includes explicit model-risk framing by comparing outputs under alternative parameter assumptions and benchmarks.

* `40_commodity_vol_and_seasonality.ipynb`  
  **Energy trading, commodities, real-data diagnostics.**  
  Analysis of WTI crude oil (FRED `DCOILWTICO`) return volatility, seasonality patterns, and regime proxies. Includes rolling volatility, seasonality diagnostics, and regime labeling with links back to Monte Carlo pricing and risk estimates. Data details: `data/README.md`. Pull script: `scripts/pull_commodity_data.py`.

* `50_crypto_short_dated_options.ipynb`  
  **Crypto markets, short-dated risk, model breakdown.**  
  Uses BTC spot data from Binance (daily + hourly) to show volatility instability, tail exceedance vs Normal/GBM baselines, and calibration-window sensitivity. Includes rolling no-look-ahead calibration, short-horizon MC VaR/ES across evaluation dates, empirical vs MC tail risk comparisons, and lightweight coverage and sigma-shock stress diagnostics. Data details: `data/README.md`. Pull script: `scripts/pull_crypto_data.py`.
