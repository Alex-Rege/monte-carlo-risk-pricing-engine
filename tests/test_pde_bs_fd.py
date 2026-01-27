import numpy as np
import pytest

from src.pde_bs_fd import price_bs_pde_cn
from src.pricing import black_scholes_call, black_scholes_put


@pytest.mark.parametrize(
    "option_type, bs_fn", [("call", black_scholes_call), ("put", black_scholes_put)]
)
@pytest.mark.parametrize("T", [0.5, 1.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_pde_matches_black_scholes(option_type, bs_fn, T, K):
    S0 = 100.0
    r = 0.03
    sigma = 0.2
    pde = price_bs_pde_cn(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        n_S=250,
        n_t=250,
    )
    bs = bs_fn(S0, K, T, r, sigma)
    assert abs(pde - bs) <= 1.5e-2


def test_pde_monotonicity_call():
    r = 0.02
    sigma = 0.25
    T = 1.0
    n_S = 200
    n_t = 200

    call_low = price_bs_pde_cn(90.0, 100.0, T, r, sigma, "call", n_S, n_t)
    call_high = price_bs_pde_cn(110.0, 100.0, T, r, sigma, "call", n_S, n_t)
    assert call_high > call_low

    call_k_low = price_bs_pde_cn(100.0, 90.0, T, r, sigma, "call", n_S, n_t)
    call_k_high = price_bs_pde_cn(100.0, 110.0, T, r, sigma, "call", n_S, n_t)
    assert call_k_low > call_k_high


def test_pde_monotonicity_put():
    r = 0.02
    sigma = 0.25
    T = 1.0
    n_S = 200
    n_t = 200

    put_low = price_bs_pde_cn(90.0, 100.0, T, r, sigma, "put", n_S, n_t)
    put_high = price_bs_pde_cn(110.0, 100.0, T, r, sigma, "put", n_S, n_t)
    assert put_low > put_high

    put_k_low = price_bs_pde_cn(100.0, 90.0, T, r, sigma, "put", n_S, n_t)
    put_k_high = price_bs_pde_cn(100.0, 110.0, T, r, sigma, "put", n_S, n_t)
    assert put_k_high > put_k_low


def test_pde_deterministic():
    price1 = price_bs_pde_cn(100.0, 100.0, 1.0, 0.03, 0.2, "call", 220, 220)
    price2 = price_bs_pde_cn(100.0, 100.0, 1.0, 0.03, 0.2, "call", 220, 220)
    assert np.isclose(price1, price2)
