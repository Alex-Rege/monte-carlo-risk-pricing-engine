from __future__ import annotations

import math
from typing import Literal

import numpy as np

from src.pricing import black_scholes_call, black_scholes_put

try:
    from scipy.optimize import brentq

    _HAS_BRENTQ = True
except Exception:  # pragma: no cover - fallback when scipy isn't available
    brentq = None
    _HAS_BRENTQ = False


def _validate_iv_inputs(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    _ = float(r)
    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    return option_type


def _bs_price(option_type: str, S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if option_type == "call":
        return black_scholes_call(S0, K, T, r, sigma)
    return black_scholes_put(S0, K, T, r, sigma)


def bs_implied_vol(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"],
    initial_guess: float = 0.2,
    max_vol: float = 5.0,
    price_eps_abs: float = 1e-4,
    price_eps_rel: float = 1e-8,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Implied volatility from Black-Scholes price using Brent's method or bisection.

    Returns NaN if the price is outside no-arbitrage bounds.
    """
    option_type = _validate_iv_inputs(price, S0, K, T, r, option_type)
    price = float(price)
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)

    if T == 0.0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        return 0.0 if abs(price - intrinsic) <= 1e-12 else float("nan")

    disc = math.exp(-r * T)
    if option_type == "call":
        lower, upper = max(0.0, S0 - K * disc), S0
    else:
        lower, upper = max(0.0, K * disc - S0), K * disc

    scale = max(S0, K * disc, 1.0)
    eps = max(price_eps_abs, price_eps_rel * scale)

    if price <= lower + eps:
        return float("nan")
    if price >= upper - eps:
        return float("nan")
    if not (lower - 1e-12 <= price <= upper + 1e-12):
        return float("nan")

    low = 1e-6
    high = max(max_vol, initial_guess)
    f_low = _bs_price(option_type, S0, K, T, r, low) - price
    f_high = _bs_price(option_type, S0, K, T, r, high) - price

    while f_high < 0.0 and high < 10.0:
        high *= 2.0
        f_high = _bs_price(option_type, S0, K, T, r, high) - price

    if f_low > 0.0 or f_high < 0.0:
        return float("nan")

    if _HAS_BRENTQ:
        sigma = float(brentq(lambda s: _bs_price(option_type, S0, K, T, r, s) - price, low, high))
        return float("nan") if sigma <= 1.5 * low else sigma

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = _bs_price(option_type, S0, K, T, r, mid) - price
        if abs(f_mid) <= tol or (high - low) <= tol:
            return float("nan") if mid <= 1.5 * low else float(mid)
        if f_mid > 0.0:
            high = mid
        else:
            low = mid

    mid = 0.5 * (low + high)
    return float("nan") if mid <= 1.5 * low else float(mid)


def implied_vol_surface(
    prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    S0: float,
    r: float,
    option_type: Literal["call", "put"],
    initial_guess: float = 0.2,
    price_eps_abs: float = 1e-4,
    price_eps_rel: float = 1e-8,
) -> np.ndarray:
    """
    Compute implied vols for a grid of prices with shape (len(maturities), len(strikes)).
    """
    prices = np.asarray(prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if prices.shape != (maturities.size, strikes.size):
        raise ValueError("prices must have shape (len(maturities), len(strikes)).")

    vols = np.empty_like(prices)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            vols[i, j] = bs_implied_vol(
                prices[i, j],
                S0=S0,
                K=K,
                T=T,
                r=r,
                option_type=option_type,
                initial_guess=initial_guess,
                price_eps_abs=price_eps_abs,
                price_eps_rel=price_eps_rel,
            )
    return vols
