import numpy as np
from src.gbm import simulate_gbm_paths, simulate_correlated_gbm_paths


def test_gbm_shapes_and_positivity():
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 10
    n_paths = 100

    t_grid, paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=123,
    )

    # Shape checks
    assert t_grid.shape == (n_steps + 1,)
    assert paths.shape == (n_paths, n_steps + 1)

    # Positivity check
    assert np.all(paths > 0.0)


def test_gbm_mean_and_variance_close_to_theory():
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252
    n_paths = 20_000  # use more paths for better accuracy

    t_grid, paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=42,
    )

    S_T = paths[:, -1]
    emp_mean = S_T.mean()
    emp_var = S_T.var(ddof=1)

    theo_mean = S0 * np.exp(mu * T)
    theo_var = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)

    # Tolerances consistent with Monte Carlo error
    assert np.isclose(emp_mean, theo_mean, rtol=0.02)  # 2% relative error
    assert np.isclose(emp_var, theo_var, rtol=0.05)  # 5% relative error


def test_gbm_reproducibility_with_seed():
    """Basic regression check: same seed -> identical paths."""
    S0 = 100.0
    mu = 0.03
    sigma = 0.15
    T = 0.5
    n_steps = 50
    n_paths = 1_000
    seed = 1234

    _, paths_1 = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=seed)
    _, paths_2 = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=seed)

    assert np.array_equal(paths_1, paths_2)


def test_gbm_small_time_drift_and_variance_sanity():
    """
    For small time horizon T, S_T should be close to S0,
    with mean and variance consistent with theory.
    """
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1e-3  # very short horizon
    n_steps = 5
    n_paths = 50_000

    _, paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=7)
    S_T = paths[:, -1]

    emp_mean = S_T.mean()
    emp_var = S_T.var(ddof=1)

    theo_mean = S0 * np.exp(mu * T)
    theo_var = (S0**2) * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1.0)

    # Because T is tiny, everything is close to S0 with tiny variance.
    assert np.isclose(emp_mean, theo_mean, rtol=0.05)
    # variance is tiny; allow generous relative tolerance due to small numbers
    assert np.isclose(emp_var, theo_var, rtol=0.25)


def test_correlated_gbm_shapes_and_positivity():
    S0 = np.array([100.0, 90.0, 120.0])
    mu = np.array([0.05, 0.04, 0.03])
    sigma = np.array([0.2, 0.25, 0.3])
    corr_matrix = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    T = 1.0
    n_steps = 50
    n_paths = 5_000

    t_grid, paths = simulate_correlated_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=456,
    )

    # Shape checks
    assert t_grid.shape == (n_steps + 1,)
    assert paths.shape == (n_paths, n_steps + 1, S0.size)

    # Positivity
    assert np.all(paths > 0.0)


def test_correlated_gbm_empirical_correlation():
    # Two assets with target correlation rho = 0.7
    S0 = np.array([100.0, 100.0])
    mu = np.array([0.05, 0.03])
    sigma = np.array([0.2, 0.25])
    corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])

    T = 1.0
    n_steps = 252
    n_paths = 50_000  # large to get stable correlation estimate

    t_grid, paths = simulate_correlated_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=123,
    )

    # Use simple returns over the whole horizon
    S_T = paths[:, -1, :]  # shape (n_paths, 2)
    R_T = S_T / S0 - 1.0  # shape (n_paths, 2)

    emp_corr = np.corrcoef(R_T.T)  # 2x2 empirical correlation matrix

    # Check diagonal ~ 1 and off-diagonal ~ 0.7 within tolerance
    assert np.isclose(emp_corr[0, 0], 1.0, rtol=1e-3)
    assert np.isclose(emp_corr[1, 1], 1.0, rtol=1e-3)
    assert np.isclose(emp_corr[0, 1], corr_matrix[0, 1], rtol=0.05)
    assert np.isclose(emp_corr[1, 0], corr_matrix[1, 0], rtol=0.05)


def test_correlated_gbm_singular_corr_matrix_runs():
    # Perfectly correlated assets (singular PSD correlation matrix).
    S0 = np.array([100.0, 100.0])
    mu = np.array([0.02, 0.02])
    sigma = np.array([0.1, 0.1])
    corr_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    _, paths = simulate_correlated_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        T=1.0,
        n_steps=252,
        n_paths=30_000,
        seed=7,
    )

    S_T = paths[:, -1, :]
    R_T = S_T / S0 - 1.0
    emp_corr = np.corrcoef(R_T.T)

    assert np.isclose(emp_corr[0, 1], 1.0, rtol=0.02)
    assert np.isclose(emp_corr[1, 0], 1.0, rtol=0.02)


def test_correlated_gbm_rejects_non_psd_corr_matrix():
    S0 = np.array([100.0, 100.0])
    mu = np.array([0.02, 0.02])
    sigma = np.array([0.1, 0.1])
    # Eigenvalues are 3 and -1 -> not PSD.
    corr_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])

    with np.testing.assert_raises(ValueError):
        simulate_correlated_gbm_paths(
            S0=S0,
            mu=mu,
            sigma=sigma,
            corr_matrix=corr_matrix,
            T=1.0,
            n_steps=10,
            n_paths=1_000,
            seed=1,
        )
