from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    variance: float
    stderr: float
    ci: tuple[float, float]
    n_paths: int
    confidence: float = 0.95
    runtime: float | None = None



def time_call(func: Callable[..., Any], *args, **kwargs):
    """
    Measure execution time of a function call.
    Returns (result, runtime_in_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    runtime = time.perf_counter() - start
    return result, runtime


def summarize_terminal_distribution(S_T, S0, mu, sigma, T, label: str = ""):
    """
    Print and return empirical vs theoretical mean/variance for GBM terminal values.
    """
    S_T = np.asarray(S_T, dtype=float)
    emp_mean = float(S_T.mean())
    emp_var = float(S_T.var(ddof=1))

    theo_mean = float(S0 * np.exp(mu * T))
    theo_var = float((S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0))

    print(f"{label}Empirical mean: {emp_mean:.6f}, Theoretical mean: {theo_mean:.6f}")
    print(f"{label}Empirical var : {emp_var:.6f}, Theoretical var : {theo_var:.6f}")
    print(f"{label}Rel. error mean: {(emp_mean - theo_mean) / theo_mean:.4%}")
    print(f"{label}Rel. error var : {(emp_var - theo_var) / theo_var:.4%}")
    print("-" * 50)

    return {
        "empirical_mean": emp_mean,
        "theoretical_mean": theo_mean,
        "empirical_var": emp_var,
        "theoretical_var": theo_var,
    }
