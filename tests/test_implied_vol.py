import numpy as np

from src.implied_vol import bs_implied_vol, implied_vol_surface
from src.pricing import black_scholes_call


def test_bs_implied_vol_recovers_sigma():
    S0 = 100.0
    K = 105.0
    T = 0.75
    r = 0.01
    sigma = 0.25
    price = black_scholes_call(S0, K, T, r, sigma)
    iv = bs_implied_vol(price, S0, K, T, r, option_type="call")
    assert abs(iv - sigma) < 1e-4


def test_bs_implied_vol_returns_nan_near_intrinsic_call():
    S0 = 100.0
    K = 120.0
    T = 0.25
    r = 0.01
    price = 0.0
    iv = bs_implied_vol(price, S0, K, T, r, option_type="call")
    assert np.isnan(iv)


def test_implied_vol_surface_preserves_nans():
    prices = np.array([[1.0, 0.0], [2.0, 0.5]])
    strikes = np.array([90.0, 120.0])
    maturities = np.array([0.5, 1.0])
    vols = implied_vol_surface(
        prices,
        strikes=strikes,
        maturities=maturities,
        S0=100.0,
        r=0.01,
        option_type="call",
    )
    assert np.isnan(vols[0, 1])
