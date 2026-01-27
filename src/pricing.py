from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm

from src.gbm import simulate_gbm_terminal
from src.options import payoff_call, payoff_put
from src.utils import MCResult, time_call

def _validate_mc_inputs(S0: float, K: float, T: float, sigma: float, n_paths: int):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2.")



def black_scholes_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes price for a European call under constant volatility.
    """
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        # deterministic under Q: S_T = S0 e^{rT}
        return max(S0 - K * math.exp(-r * T), 0.0)

    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return float(S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes price for a European put under constant volatility.
    """
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S0, 0.0)

    vol_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def _mc_price_from_terminal_payoffs(
    discounted_payoffs: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float, tuple[float, float]]:
    """
    Compute mean, sample variance (ddof=1), standard error, and normal-approx CI
    for the MC estimator. discounted_payoffs are already e^{-rT} * payoff.
    """
    x = np.asarray(discounted_payoffs, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 paths to estimate variance/CI reliably.")

    price = float(x.mean())
    var = float(x.var(ddof=1))
    stderr = math.sqrt(var / n)

    if confidence == 0.95:
        z = 1.959963984540054
    else:
        z = float(norm.ppf(0.5 + confidence / 2.0))

    ci = (price - z * stderr, price + z * stderr)
    return price, var, stderr, ci



def mc_european_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    confidence: float = 0.95,
    return_details: bool = True,
) -> MCResult | float:
    """
    Monte Carlo price of a European call under risk-neutral GBM (drift = r).

    Returns MCResult by default; set return_details=False to return just the price.
    """
    _validate_mc_inputs(S0, K, T, sigma, n_paths)

    # T == 0: option is worth intrinsic value immediately (no randomness)
    if T == 0.0:
        price0 = float(payoff_call(np.array([S0]), K)[0])
        result = MCResult(
            price=price0,
            variance=0.0,
            stderr=0.0,
            ci=(price0, price0),
            n_paths=n_paths,
            confidence=confidence,
            runtime=0.0,
        )
        return result if return_details else result.price

    S_T, runtime = time_call(
        simulate_gbm_terminal, S0, r, sigma, T, n_paths, seed=seed
    )

    payoffs = payoff_call(S_T, K)
    discounted = np.exp(-r * T) * payoffs

    price, var, stderr, ci = _mc_price_from_terminal_payoffs(discounted, confidence=confidence)
    result = MCResult(
        price=price,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=runtime,
    )

    return result if return_details else result.price



def mc_european_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    confidence: float = 0.95,
    return_details: bool = True,
) -> MCResult | float:
    """
    Monte Carlo price of a European put under risk-neutral GBM (drift = r).

    Returns MCResult by default; set return_details=False to return just the price.
    """
    _validate_mc_inputs(S0, K, T, sigma, n_paths)

    # T == 0: option is worth intrinsic value immediately (no randomness)
    if T == 0.0:
        price0 = float(payoff_put(np.array([S0]), K)[0])
        result = MCResult(
            price=price0,
            variance=0.0,
            stderr=0.0,
            ci=(price0, price0),
            n_paths=n_paths,
            confidence=confidence,
            runtime=0.0,
        )
        return result if return_details else result.price

    S_T, runtime = time_call(
        simulate_gbm_terminal, S0, r, sigma, T, n_paths, seed=seed
    )

    payoffs = payoff_put(S_T, K)
    discounted = np.exp(-r * T) * payoffs

    price, var, stderr, ci = _mc_price_from_terminal_payoffs(discounted, confidence=confidence)
    result = MCResult(
        price=price,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=runtime,
    )

    return result if return_details else result.price

