from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from src.gbm import simulate_gbm_terminal
from src.payoffs import payoff_call, payoff_put


@dataclass(frozen=True)
class GreekResult:
    estimate: float
    variance: float
    stderr: float
    ci: tuple[float, float]
    n_paths: int
    confidence: float = 0.95
    runtime: float | None = None


def _validate_variance_reduction(variance_reduction: str) -> str:
    if variance_reduction is None:
        return "none"
    variance_reduction = str(variance_reduction).lower()
    allowed = {"none", "antithetic", "control_variate"}
    if variance_reduction not in allowed:
        raise ValueError(f"variance_reduction must be one of {sorted(allowed)}.")
    return variance_reduction


def _greek_stats(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, tuple[float, float]]:
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 paths to estimate variance/CI reliably.")

    estimate = float(x.mean())
    var = float(x.var(ddof=1))
    stderr = math.sqrt(var / n)
    z = 1.959963984540054 if confidence == 0.95 else float(norm.ppf(0.5 + confidence / 2.0))
    ci = (estimate - z * stderr, estimate + z * stderr)
    return estimate, var, stderr, ci


def _gbm_terminal_from_z(S0: float, r: float, sigma: float, T: float, Z: np.ndarray) -> np.ndarray:
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def _apply_control_variate(values: np.ndarray, control: np.ndarray, control_mean: float) -> np.ndarray:
    control_var = float(control.var(ddof=1))
    if control_var == 0.0:
        return values
    cov = float(np.cov(values, control, ddof=1)[0, 1])
    b_opt = cov / control_var
    return values - b_opt * (control - control_mean)


def _antithetic_values(
    values_pos: np.ndarray, values_neg: np.ndarray, extra: np.ndarray | None
) -> np.ndarray:
    paired = 0.5 * (values_pos + values_neg)
    if extra is None:
        return paired
    return np.concatenate([paired, extra])


def _pathwise_delta_values(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None,
    payoff_kind: str,
    variance_reduction: str,
) -> np.ndarray:
    variance_reduction = _validate_variance_reduction(variance_reduction)

    def delta_from_st(S_T: np.ndarray) -> np.ndarray:
        factor = np.exp(-r * T) * (S_T / S0)
        if payoff_kind == "call":
            return np.where(S_T > K, factor, 0.0)
        return np.where(S_T < K, -factor, 0.0)

    if variance_reduction == "antithetic":
        rng = np.random.default_rng(seed)
        n_pairs = n_paths // 2
        Z = rng.standard_normal(size=n_pairs)
        S_pos = _gbm_terminal_from_z(S0, r, sigma, T, Z)
        S_neg = _gbm_terminal_from_z(S0, r, sigma, T, -Z)
        values_pos = delta_from_st(S_pos)
        values_neg = delta_from_st(S_neg)

        extra = None
        if n_paths % 2 == 1:
            Z_extra = rng.standard_normal(size=1)
            S_extra = _gbm_terminal_from_z(S0, r, sigma, T, Z_extra)
            extra = delta_from_st(S_extra)

        return _antithetic_values(values_pos, values_neg, extra)

    S_T = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=seed)
    values = delta_from_st(S_T)

    if variance_reduction == "control_variate":
        control = np.exp(-r * T) * (S_T / S0)
        values = _apply_control_variate(values, control, 1.0)

    return values


def _fd_greek_values(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None,
    bump: float,
    payoff_fn: Callable[[np.ndarray, float], np.ndarray],
    bump_kind: str,
    variance_reduction: str,
    include_mid: bool,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    variance_reduction = _validate_variance_reduction(variance_reduction)
    rng = np.random.default_rng(seed)

    def payoffs_for_z(S0_loc: float, sigma_loc: float, Z: np.ndarray) -> np.ndarray:
        S_T = _gbm_terminal_from_z(S0_loc, r, sigma_loc, T, Z)
        return np.exp(-r * T) * payoff_fn(S_T, K)

    if variance_reduction == "antithetic":
        n_pairs = n_paths // 2
        Z = rng.standard_normal(size=n_pairs)
        Z_neg = -Z
        Z_extra = rng.standard_normal(size=1) if n_paths % 2 == 1 else None

        if bump_kind == "spot":
            up_pos = payoffs_for_z(S0 + bump, sigma, Z)
            down_pos = payoffs_for_z(S0 - bump, sigma, Z)
            up_neg = payoffs_for_z(S0 + bump, sigma, Z_neg)
            down_neg = payoffs_for_z(S0 - bump, sigma, Z_neg)
            mid_pos = payoffs_for_z(S0, sigma, Z) if include_mid else None
            mid_neg = payoffs_for_z(S0, sigma, Z_neg) if include_mid else None
            up_extra = (
                payoffs_for_z(S0 + bump, sigma, Z_extra) if Z_extra is not None else None
            )
            down_extra = (
                payoffs_for_z(S0 - bump, sigma, Z_extra) if Z_extra is not None else None
            )
            mid_extra = payoffs_for_z(S0, sigma, Z_extra) if include_mid and Z_extra is not None else None
        else:
            up_pos = payoffs_for_z(S0, sigma + bump, Z)
            down_pos = payoffs_for_z(S0, sigma - bump, Z)
            up_neg = payoffs_for_z(S0, sigma + bump, Z_neg)
            down_neg = payoffs_for_z(S0, sigma - bump, Z_neg)
            mid_pos = payoffs_for_z(S0, sigma, Z) if include_mid else None
            mid_neg = payoffs_for_z(S0, sigma, Z_neg) if include_mid else None
            up_extra = (
                payoffs_for_z(S0, sigma + bump, Z_extra) if Z_extra is not None else None
            )
            down_extra = (
                payoffs_for_z(S0, sigma - bump, Z_extra) if Z_extra is not None else None
            )
            mid_extra = payoffs_for_z(S0, sigma, Z_extra) if include_mid and Z_extra is not None else None

        up = _antithetic_values(up_pos, up_neg, up_extra)
        down = _antithetic_values(down_pos, down_neg, down_extra)
        if include_mid:
            mid = _antithetic_values(mid_pos, mid_neg, mid_extra)
            return up, down, mid
        return up, down

    Z = rng.standard_normal(size=n_paths)
    if bump_kind == "spot":
        up = payoffs_for_z(S0 + bump, sigma, Z)
        down = payoffs_for_z(S0 - bump, sigma, Z)
        mid = payoffs_for_z(S0, sigma, Z) if include_mid else None
    else:
        up = payoffs_for_z(S0, sigma + bump, Z)
        down = payoffs_for_z(S0, sigma - bump, Z)
        mid = payoffs_for_z(S0, sigma, Z) if include_mid else None

    if variance_reduction == "control_variate":
        S_T = _gbm_terminal_from_z(S0, r, sigma, T, Z)
        control = np.exp(-r * T) * (S_T / S0)
        up = _apply_control_variate(up, control, 1.0)
        down = _apply_control_variate(down, control, 1.0)
        if include_mid and mid is not None:
            mid = _apply_control_variate(mid, control, 1.0)

    if include_mid:
        return up, down, mid
    return up, down


def _fd_delta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None,
    bump: float,
    payoff_fn: Callable[[np.ndarray, float], np.ndarray],
    variance_reduction: str,
) -> np.ndarray:
    up, down = _fd_greek_values(
        S0,
        K,
        T,
        r,
        sigma,
        n_paths,
        seed,
        bump,
        payoff_fn,
        bump_kind="spot",
        variance_reduction=variance_reduction,
        include_mid=False,
    )
    return (up - down) / (2.0 * bump)


def _fd_vega(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None,
    bump: float,
    payoff_fn: Callable[[np.ndarray, float], np.ndarray],
    variance_reduction: str,
) -> np.ndarray:
    if sigma <= bump:
        raise ValueError("sigma must be greater than bump for vega finite differences.")
    up, down = _fd_greek_values(
        S0,
        K,
        T,
        r,
        sigma,
        n_paths,
        seed,
        bump,
        payoff_fn,
        bump_kind="vol",
        variance_reduction=variance_reduction,
        include_mid=False,
    )
    return (up - down) / (2.0 * bump)


def _fd_gamma(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None,
    bump: float,
    payoff_fn: Callable[[np.ndarray, float], np.ndarray],
    variance_reduction: str,
) -> np.ndarray:
    up, down, mid = _fd_greek_values(
        S0,
        K,
        T,
        r,
        sigma,
        n_paths,
        seed,
        bump,
        payoff_fn,
        bump_kind="spot",
        variance_reduction=variance_reduction,
        include_mid=True,
    )
    return (up - 2.0 * mid + down) / (bump**2)


def bs_delta_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S0 > K else 0.0
    if sigma <= 0:
        return 1.0 if S0 > K * math.exp(-r * T) else 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def bs_delta_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_delta_call(S0, K, T, r, sigma) - 1.0


def bs_gamma(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.pdf(d1) / (S0 * sigma * math.sqrt(T)))


def bs_vega(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S0 * norm.pdf(d1) * math.sqrt(T))


def delta_pathwise_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _pathwise_delta_values(
        S0, K, T, r, sigma, n_paths, seed, "call", variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def delta_pathwise_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _pathwise_delta_values(
        S0, K, T, r, sigma, n_paths, seed, "put", variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def delta_fd_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_delta(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_call, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def delta_fd_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_delta(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_put, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def vega_fd_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_vega(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_call, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def vega_fd_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_vega(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_put, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def gamma_fd_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_gamma(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_call, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate


def gamma_fd_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    seed: int | None = None,
    bump: float = 1e-4,
    confidence: float = 0.95,
    variance_reduction: str = "none",
    return_details: bool = True,
) -> GreekResult | float:
    values = _fd_gamma(
        S0, K, T, r, sigma, n_paths, seed, bump, payoff_put, variance_reduction
    )
    estimate, var, stderr, ci = _greek_stats(values, confidence=confidence)
    result = GreekResult(
        estimate=estimate,
        variance=var,
        stderr=stderr,
        ci=ci,
        n_paths=n_paths,
        confidence=confidence,
        runtime=None,
    )
    return result if return_details else result.estimate
