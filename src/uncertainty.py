from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_statistic(
    x, stat_fn: Callable[[np.ndarray], float], n_boot: int = 500, seed: int | None = None
) -> np.ndarray:
    """
    Bootstrap a statistic by sampling with replacement from x.

    Parameters
    ----------
    x:
        Sample data. Interpreted as a 1D sample.
    stat_fn:
        Function mapping a 1D array to a scalar statistic (e.g., mean, VaR, ES).
    n_boot:
        Number of bootstrap replicates.
    seed:
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Bootstrap replicates of stat_fn(x), shape (n_boot,).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if x.size == 0:
        raise ValueError("x must be non-empty.")
    if n_boot <= 0:
        raise ValueError("n_boot must be a positive integer.")

    rng = np.random.default_rng(seed)
    n = x.size

    reps = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        reps[b] = float(stat_fn(x[idx]))
    return reps


def ci_from_samples(
    samples, alpha: float = 0.05, method: str = "percentile"
) -> tuple[float, float]:
    """
    Confidence interval from a set of samples.

    Parameters
    ----------
    samples:
        1D array of bootstrap replicates (or other sampling distribution).
    alpha:
        Significance level. alpha=0.05 yields a 95% CI.
    method:
        Currently supports 'percentile'.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper CI bounds.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1D array.")
    if samples.size == 0:
        raise ValueError("samples must be non-empty.")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    method = str(method).lower()
    if method != "percentile":
        raise ValueError("method must be 'percentile'.")

    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_bootstrap(samples, alpha: float = 0.05) -> dict[str, float]:
    """
    Convenience summary of bootstrap samples: mean/std and percentile CI.
    """
    samples = np.asarray(samples, dtype=float)
    lo, hi = ci_from_samples(samples, alpha=alpha, method="percentile")
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "lo": lo,
        "hi": hi,
    }
