import numpy as np
import pytest

from src.greeks import (
    bs_delta_call,
    bs_gamma,
    bs_vega,
    delta_fd_call,
    delta_pathwise_call,
    gamma_fd_call,
    vega_fd_call,
)


def test_greeks_deterministic_with_seed():
    res1 = delta_fd_call(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.2,
        n_paths=50_000,
        seed=123,
        return_details=True,
    )
    res2 = delta_fd_call(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.2,
        n_paths=50_000,
        seed=123,
        return_details=True,
    )
    assert res1.estimate == res2.estimate
    assert res1.stderr == res2.stderr


def test_delta_monotonicity_around_atm():
    params = dict(K=100.0, T=1.0, r=0.01, sigma=0.2, n_paths=80_000, seed=7)
    low = delta_pathwise_call(S0=90.0, return_details=False, **params)
    high = delta_pathwise_call(S0=110.0, return_details=False, **params)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_greeks_close_to_black_scholes():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2
    n_paths = 150_000

    bs_delta = bs_delta_call(S0, K, T, r, sigma)
    bs_vega_val = bs_vega(S0, K, T, r, sigma)
    bs_gamma_val = bs_gamma(S0, K, T, r, sigma)

    delta_res = delta_pathwise_call(
        S0, K, T, r, sigma, n_paths, seed=11, return_details=True
    )
    vega_res = vega_fd_call(
        S0, K, T, r, sigma, n_paths, seed=11, return_details=True
    )
    gamma_res = gamma_fd_call(
        S0, K, T, r, sigma, n_paths, seed=11, bump=0.5, return_details=True
    )

    assert abs(delta_res.estimate - bs_delta) <= 3.0 * delta_res.stderr
    assert abs(vega_res.estimate - bs_vega_val) <= 3.0 * vega_res.stderr
    assert abs(gamma_res.estimate - bs_gamma_val) <= 3.0 * gamma_res.stderr


def test_delta_variance_reduction_antithetic():
    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.01
    sigma = 0.25
    n_paths = 20_000

    seeds = range(8)
    plain = []
    anti = []
    for seed in seeds:
        plain.append(
            delta_pathwise_call(
                S0,
                K,
                T,
                r,
                sigma,
                n_paths,
                seed=seed,
                variance_reduction="none",
                return_details=False,
            )
        )
        anti.append(
            delta_pathwise_call(
                S0,
                K,
                T,
                r,
                sigma,
                n_paths,
                seed=seed,
                variance_reduction="antithetic",
                return_details=False,
            )
        )

    var_plain = float(np.var(plain, ddof=1))
    var_anti = float(np.var(anti, ddof=1))

    assert var_anti <= 0.9 * var_plain
