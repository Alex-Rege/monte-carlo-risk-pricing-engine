from __future__ import annotations

import math
from typing import Literal

import numpy as np

from src.payoffs import payoff_call, payoff_put
from src.utils import MCResult


def _validate_heston_params(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if v0 < 0:
        raise ValueError("v0 must be non-negative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    if xi <= 0:
        raise ValueError("xi must be positive.")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be strictly between -1 and 1.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1.")
    _ = float(r)


def simulate_heston_qe_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Heston paths using the Quadratic-Exponential (QE) scheme for variance.

    Risk-neutral dynamics:
        dS_t = r S_t dt + sqrt(V_t) S_t dW1_t
        dV_t = kappa (theta - V_t) dt + xi sqrt(V_t) dW2_t
        corr(dW1, dW2) = rho

    Returns (t_grid, S_paths, V_paths) with shapes:
        S_paths.shape == (n_paths, n_steps + 1)
        V_paths.shape == (n_paths, n_steps + 1)
    """
    _validate_heston_params(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths)

    S0 = float(S0)
    v0 = float(v0)
    r = float(r)
    kappa = float(kappa)
    theta = float(theta)
    xi = float(xi)
    rho = float(rho)
    T = float(T)

    dt = T / n_steps
    exp_kdt = math.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_kdt
    psi_c = 1.5
    eps = 1e-12

    t_grid = np.linspace(0.0, T, n_steps + 1)
    S_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    V_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    S_paths[:, 0] = S0
    V_paths[:, 0] = v0

    rng = np.random.default_rng(seed)
    sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)

    for step in range(n_steps):
        v_t = V_paths[:, step]

        m = theta + (v_t - theta) * exp_kdt
        s2 = (
            v_t * xi * xi * exp_kdt * one_minus_exp / kappa
            + theta * xi * xi * one_minus_exp * one_minus_exp / (2.0 * kappa)
        )
        psi = s2 / (m * m + eps)

        z2 = rng.standard_normal(n_paths)
        v_next = np.empty_like(v_t)

        mask = psi <= psi_c
        if np.any(mask):
            psi_m = psi[mask]
            b2 = 2.0 / psi_m - 1.0 + np.sqrt(2.0 / psi_m) * np.sqrt(2.0 / psi_m - 1.0)
            a = m[mask] / (1.0 + b2)
            v_next[mask] = a * (np.sqrt(b2) + z2[mask]) ** 2

        if np.any(~mask):
            psi_h = psi[~mask]
            p = (psi_h - 1.0) / (psi_h + 1.0)
            beta = (1.0 - p) / (m[~mask] + eps)
            u = rng.random(psi_h.size)
            v_next[~mask] = np.where(
                u <= p,
                0.0,
                -np.log((1.0 - p) / (1.0 - u)) / beta,
            )

        # Last-resort guard against tiny negative values from numerical noise.
        v_next = np.maximum(v_next, 0.0)
        V_paths[:, step + 1] = v_next

        z1 = rng.standard_normal(n_paths)
        z_corr = rho * z2 + sqrt_one_minus_rho2 * z1
        v_bar = 0.5 * (v_t + v_next)
        v_bar = np.maximum(v_bar, 0.0)
        S_paths[:, step + 1] = S_paths[:, step] * np.exp(
            (r - 0.5 * v_bar) * dt + np.sqrt(v_bar * dt) * z_corr
        )

    return t_grid, S_paths, V_paths


def _mc_price_from_payoffs(discounted_payoffs: np.ndarray, confidence: float = 0.95) -> MCResult:
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
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + confidence / 2.0))

    ci = (price - z * stderr, price + z * stderr)
    return MCResult(
        price=price,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n,
        confidence=confidence,
        runtime=None,
    )


def price_european_option_heston_mc(
    option_type: Literal["call", "put"],
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    return_ci: bool = True,
) -> MCResult | float:
    """
    Monte Carlo price of a European option under Heston (QE simulation).

    If return_ci is True, returns MCResult with price, variance, stderr, and CI.
    Otherwise returns the price only.
    """
    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if n_paths < 2 and return_ci:
        raise ValueError("n_paths must be >= 2 when return_ci is True.")

    if T <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
        if return_ci:
            return MCResult(
                price=float(intrinsic),
                variance=0.0,
                stderr=0.0,
                ci=(float(intrinsic), float(intrinsic)),
                n_paths=n_paths,
                confidence=0.95,
                runtime=0.0,
            )
        return float(intrinsic)

    _, S_paths, _ = simulate_heston_qe_paths(
        S0=S0,
        v0=v0,
        r=r,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    S_T = S_paths[:, -1]
    payoff_fn = payoff_call if option_type == "call" else payoff_put
    discounted = math.exp(-r * T) * payoff_fn(S_T, K)

    if return_ci:
        return _mc_price_from_payoffs(discounted)
    return float(np.mean(discounted))
