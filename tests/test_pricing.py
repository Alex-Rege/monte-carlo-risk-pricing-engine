import numpy as np
import pytest

from src.options import payoff_call, payoff_put
from src.pricing import (
    black_scholes_call,
    black_scholes_put,
    mc_european_call,
    mc_european_put,
)


def test_payoff_call_vectorized():
    S_T = np.array([50.0, 100.0, 150.0])
    K = 100.0
    out = payoff_call(S_T, K)
    assert np.allclose(out, np.array([0.0, 0.0, 50.0]))


def test_payoff_put_vectorized():
    S_T = np.array([50.0, 100.0, 150.0])
    K = 100.0
    out = payoff_put(S_T, K)
    assert np.allclose(out, np.array([50.0, 0.0, 0.0]))


def test_black_scholes_put_call_parity():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.2

    C = black_scholes_call(S0, K, T, r, sigma)
    P = black_scholes_put(S0, K, T, r, sigma)

    lhs = C - P
    rhs = S0 - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sigma", [0.0, 1e-12])
def test_black_scholes_sigma_zero_limit(sigma):
    # When sigma -> 0, under Q: S_T = S0 e^{rT} deterministically.
    S0 = 100.0
    K = 90.0
    T = 2.0
    r = 0.05
    C = black_scholes_call(S0, K, T, r, sigma)
    P = black_scholes_put(S0, K, T, r, sigma)

    C_det = max(S0 - K * np.exp(-r * T), 0.0)
    P_det = max(K * np.exp(-r * T) - S0, 0.0)

    assert np.isclose(C, C_det, rtol=1e-10, atol=1e-10)
    assert np.isclose(P, P_det, rtol=1e-10, atol=1e-10)


def test_mc_call_bs_within_ci():
    # A robust test: BS price should lie in MC 95% CI with high probability.
    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.02
    sigma = 0.25
    n_steps = 252
    n_paths = 80_000

    bs = black_scholes_call(S0, K, T, r, sigma)
    res = mc_european_call(S0, K, T, r, sigma, n_steps, n_paths, seed=123, return_details=True)

    assert res.variance >= 0.0
    assert res.stderr >= 0.0
    assert res.ci[0] <= res.price <= res.ci[1]
    assert res.ci[0] <= bs <= res.ci[1]

    # Deterministic criterion (tied to estimator's stderr)
    assert abs(res.price - bs) <= 3.0 * res.stderr


def test_mc_put_bs_within_ci():
    S0 = 100.0
    K = 95.0
    T = 0.5
    r = 0.01
    sigma = 0.2
    n_steps = 126
    n_paths = 80_000

    bs = black_scholes_put(S0, K, T, r, sigma)
    res = mc_european_put(S0, K, T, r, sigma, n_steps, n_paths, seed=321, return_details=True)

    assert res.variance >= 0.0
    assert res.stderr >= 0.0
    assert res.ci[0] <= res.price <= res.ci[1]
    assert res.ci[0] <= bs <= res.ci[1]

    # Deterministic criterion (tied to estimator's stderr)
    assert abs(res.price - bs) <= 3.0 * res.stderr


def test_mc_reproducibility_same_seed():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.2
    n_steps = 50
    n_paths = 20_000
    seed = 999

    res1 = mc_european_call(S0, K, T, r, sigma, n_steps, n_paths, seed=seed, return_details=True)
    res2 = mc_european_call(S0, K, T, r, sigma, n_steps, n_paths, seed=seed, return_details=True)

    assert res1.price == res2.price
    assert res1.variance == res2.variance
    assert res1.ci == res2.ci

    assert res1.stderr == res2.stderr
    assert res1.confidence == res2.confidence



def test_mc_raises_on_too_few_paths():
    with pytest.raises(ValueError):
        _ = mc_european_call(
            S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2,
            n_steps=10, n_paths=1, seed=1, return_details=True
        )

def test_mc_runtime_is_reported():
    res = mc_european_call(
        S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2,
        n_steps=10, n_paths=10_000, seed=1, return_details=True
    )
    assert res.runtime is not None
    assert res.runtime >= 0.0
