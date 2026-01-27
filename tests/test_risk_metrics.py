import numpy as np

from src.risk_metrics import (
    empirical_es,
    empirical_var,
    simulate_portfolio_pnl,
)


def _portfolio_inputs():
    S0 = np.array([100.0, 95.0, 105.0])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([0.2, 0.25, 0.3])
    corr_matrix = np.array(
        [
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    weights = np.array([1.0, 0.8, 1.2])
    return S0, mu, sigma, corr_matrix, weights


def test_var_increases_with_alpha():
    S0, mu, sigma, corr_matrix, weights = _portfolio_inputs()
    pnl = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=1.0,
        n_steps=1,
        n_paths=10_000,
        seed=123,
    )

    var_90 = empirical_var(pnl, alpha=0.90)
    var_99 = empirical_var(pnl, alpha=0.99)

    assert var_99 >= var_90


def test_es_geq_var():
    S0, mu, sigma, corr_matrix, weights = _portfolio_inputs()
    pnl = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=1.0,
        n_steps=1,
        n_paths=12_000,
        seed=7,
    )

    alpha = 0.975
    var = empirical_var(pnl, alpha=alpha)
    es = empirical_es(pnl, alpha=alpha)

    assert es >= var


def test_reproducibility_with_seed():
    S0, mu, sigma, corr_matrix, weights = _portfolio_inputs()
    pnl_1 = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=0.5,
        n_steps=1,
        n_paths=8_000,
        seed=999,
    )
    pnl_2 = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=0.5,
        n_steps=1,
        n_paths=8_000,
        seed=999,
    )

    assert np.array_equal(pnl_1, pnl_2)


def test_portfolio_scaling():
    S0, mu, sigma, corr_matrix, weights = _portfolio_inputs()
    pnl_base = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=weights,
        T=1.0,
        n_steps=1,
        n_paths=10_000,
        seed=2024,
    )
    pnl_scaled = simulate_portfolio_pnl(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        weights=2.0 * weights,
        T=1.0,
        n_steps=1,
        n_paths=10_000,
        seed=2024,
    )

    assert np.allclose(pnl_scaled, 2.0 * pnl_base)
    alpha = 0.95
    var_base = empirical_var(pnl_base, alpha=alpha)
    es_base = empirical_es(pnl_base, alpha=alpha)
    var_scaled = empirical_var(pnl_scaled, alpha=alpha)
    es_scaled = empirical_es(pnl_scaled, alpha=alpha)

    assert np.isclose(var_scaled, 2.0 * var_base)
    assert np.isclose(es_scaled, 2.0 * es_base)
