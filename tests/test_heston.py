import numpy as np

from src.heston import simulate_heston_qe_paths, price_european_option_heston_mc
from src.pricing import black_scholes_call


def test_heston_reproducibility():
    params = dict(
        S0=100.0,
        v0=0.04,
        r=0.01,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.5,
        T=1.0,
        n_steps=12,
        n_paths=256,
        seed=123,
    )
    _, s1, v1 = simulate_heston_qe_paths(**params)
    _, s2, v2 = simulate_heston_qe_paths(**params)
    assert np.array_equal(s1, s2)
    assert np.array_equal(v1, v2)


def test_heston_variance_non_negative_and_finite():
    _, s_paths, v_paths = simulate_heston_qe_paths(
        S0=100.0,
        v0=0.04,
        r=0.0,
        kappa=1.5,
        theta=0.04,
        xi=1.0,
        rho=-0.7,
        T=1.0,
        n_steps=24,
        n_paths=512,
        seed=42,
    )
    assert np.isfinite(s_paths).all()
    assert np.isfinite(v_paths).all()
    assert v_paths.min() >= -1e-12


def test_heston_xi_to_zero_matches_black_scholes():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    sigma = 0.2
    v0 = sigma * sigma
    theta = v0

    price_heston = price_european_option_heston_mc(
        option_type="call",
        S0=S0,
        K=K,
        T=T,
        r=r,
        v0=v0,
        kappa=2.0,
        theta=theta,
        xi=1e-6,
        rho=-0.3,
        n_steps=50,
        n_paths=20000,
        seed=7,
        return_ci=False,
    )
    price_bs = black_scholes_call(S0, K, T, r, sigma)
    assert abs(price_heston - price_bs) <= 0.7
