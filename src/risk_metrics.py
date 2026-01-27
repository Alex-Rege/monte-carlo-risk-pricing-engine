from __future__ import annotations

import numpy as np

from src.gbm import simulate_correlated_gbm_paths


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    return alpha


def empirical_var(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical Value-at-Risk (VaR) from a PnL sample at confidence level alpha.

    VaR is reported as a positive number representing a loss threshold:
        VaR_alpha = -quantile_{1 - alpha}(PnL).
    """
    alpha = _validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a 1D array.")
    if pnl.size == 0:
        raise ValueError("pnl must be non-empty.")

    q = np.quantile(pnl, 1.0 - alpha)
    return -float(q)


def empirical_es(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical Expected Shortfall (ES) from a PnL sample at confidence level alpha.

    ES is reported as a positive number:
        ES_alpha = -mean(PnL | PnL <= quantile_{1 - alpha}(PnL)).
    """
    alpha = _validate_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a 1D array.")
    if pnl.size == 0:
        raise ValueError("pnl must be non-empty.")

    q = np.quantile(pnl, 1.0 - alpha)
    tail = pnl[pnl <= q]
    if tail.size == 0:
        return -float(q)
    return -float(tail.mean())


def simulate_portfolio_pnl(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    weights: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate portfolio PnL using correlated GBM paths over horizon T.

    The portfolio value is computed as the weighted sum of asset prices.
    PnL is defined as V_T - V_0 for each path.
    """
    S0 = np.asarray(S0, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if S0.ndim != 1:
        raise ValueError("S0 must be a 1D array.")
    if weights.shape != S0.shape:
        raise ValueError("weights must have the same shape as S0.")

    _, paths = simulate_correlated_gbm_paths(
        S0=S0,
        mu=np.asarray(mu, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        corr_matrix=np.asarray(corr_matrix, dtype=float),
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    S_T = paths[:, -1, :]
    v0 = float(S0 @ weights)
    vT = S_T @ weights
    return vT - v0


def estimate_portfolio_var_es(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    weights: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    alpha: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Simulate portfolio PnL and estimate empirical VaR and ES at confidence alpha.

    Returns (VaR, ES, pnl).
    """
    pnl = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    var = empirical_var(pnl, alpha=alpha)
    es = empirical_es(pnl, alpha=alpha)
    return var, es, pnl


def var_es_curve(pnl: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute VaR and ES arrays for a list of confidence levels.
    """
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a 1D array.")
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1:
        raise ValueError("alphas must be a 1D array.")
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty.")

    vars_ = np.empty(alphas.shape, dtype=float)
    ess = np.empty(alphas.shape, dtype=float)
    for i, alpha in enumerate(alphas):
        alpha = _validate_alpha(float(alpha))
        vars_[i] = empirical_var(pnl, alpha=alpha)
        ess[i] = empirical_es(pnl, alpha=alpha)
    return vars_, ess


def plot_pnl_histogram(
    pnl: np.ndarray,
    alpha: float = 0.95,
    var: float | None = None,
    es: float | None = None,
    bins: int = 50,
    ax=None,
):
    """
    Plot a PnL histogram with VaR and ES markers. Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1:
        raise ValueError("pnl must be a 1D array.")

    if var is None:
        var = empirical_var(pnl, alpha=alpha)
    if es is None:
        es = empirical_es(pnl, alpha=alpha)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.hist(pnl, bins=bins, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(-var, color="crimson", linestyle="--", linewidth=2, label=f"VaR {alpha:.1%}")
    ax.axvline(-es, color="darkorange", linestyle="-", linewidth=2, label=f"ES {alpha:.1%}")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig, ax


def plot_var_es_vs_alpha(
    pnl: np.ndarray,
    alphas: np.ndarray,
    ax=None,
):
    """
    Plot VaR and ES as a function of confidence level. Returns (fig, ax).
    """
    import matplotlib.pyplot as plt

    vars_, ess = var_es_curve(pnl, alphas)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(alphas, vars_, color="crimson", linewidth=2, label="VaR")
    ax.plot(alphas, ess, color="darkorange", linewidth=2, label="ES")
    ax.set_xlabel("Confidence level (alpha)")
    ax.set_ylabel("Risk measure")
    ax.legend()
    return fig, ax
