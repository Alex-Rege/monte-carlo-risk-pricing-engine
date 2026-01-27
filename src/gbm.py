import numpy as np


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Geometric Brownian Motion (GBM) paths using the exact lognormal discretization.

    S_t satisfies the SDE:
        dS_t = mu * S_t dt + sigma * S_t dW_t.

    The exact solution is:
        S_t = S_0 * exp( (mu - 0.5 * sigma^2) * t + sigma * W_t ).

    We discretize time into n_steps over [0, T] with step dt = T / n_steps,
    and simulate n_paths independent paths.

    Parameters
    ----------
    S0 : float
        Initial asset price S_0.
    mu : float
        Drift parameter (e.g. expected return or risk-free rate under Q).
    sigma : float
        Volatility parameter (standard deviation of returns).
    T : float
        Time horizon (in years).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated GBM paths. Each row is one path, columns correspond to times in t_grid.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # Standard normal increments Z_{i,j} ~ N(0,1)
    Z = rng.standard_normal(size=(n_paths, n_steps))

    # Drift and diffusion terms for log S increments
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Cumulative sum in log-space
    log_S0 = np.log(S0)
    log_increments = drift + diffusion
    log_paths = log_S0 + np.cumsum(log_increments, axis=1)

    # Prepend initial value at time 0
    log_paths = np.concatenate(
        [np.full((n_paths, 1), log_S0), log_paths],
        axis=1,
    )

    paths = np.exp(log_paths)
    return t_grid, paths


def simulate_correlated_gbm_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated Geometric Brownian Motion (GBM) paths for multiple assets.

    For each asset i = 1,...,d:
        dS_t^i = mu_i * S_t^i dt + sigma_i * S_t^i dW_t^i,

    with correlation structure:
        E[dW_t^i dW_t^j] = rho_{ij} dt,

    where rho is the given correlation matrix.

    Time is discretized into n_steps over [0, T] with dt = T / n_steps.
    Correlated Brownian increments are generated via Cholesky factorization of corr_matrix.

    Parameters
    ----------
    S0 : np.ndarray, shape (d,)
        Initial prices for each of the d assets.
    mu : np.ndarray, shape (d,)
        Drift parameters for each asset.
    sigma : np.ndarray, shape (d,)
        Volatility parameters for each asset (non-negative).
    corr_matrix : np.ndarray, shape (d, d)
        Symmetric positive semi-definite correlation matrix with ones on the diagonal.
        PSD is accepted up to tolerance eps.
    T : float
        Time horizon (in years).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    seed : int or None, optional
        Random seed for reproducibility.
    eps : float, optional
        Tolerance for PSD checks. Eigenvalues below -eps are rejected; values in
        [-eps, 0] are clamped to 0. Default is 1e-10.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    paths : np.ndarray, shape (n_paths, n_steps + 1, d)
        Simulated GBM paths. paths[p, k, i] = price of asset i in path p at time index k.
    """
    S0 = np.asarray(S0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    if S0.ndim != 1:
        raise ValueError("S0 must be a 1D array of length d.")
    if mu.shape != S0.shape or sigma.shape != S0.shape:
        raise ValueError("mu, sigma must have the same shape as S0.")
    if corr_matrix.shape != (S0.size, S0.size):
        raise ValueError("corr_matrix must be of shape (d, d) with d = len(S0).")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if np.any(S0 <= 0):
        raise ValueError("All entries of S0 must be positive.")
    if np.any(sigma < 0):
        raise ValueError("All entries of sigma must be non-negative.")

    # Check basic correlation matrix properties
    if not np.allclose(corr_matrix, corr_matrix.T, atol=1e-8):
        raise ValueError("corr_matrix must be symmetric.")
    if not np.allclose(np.diag(corr_matrix), 1.0, atol=1e-8):
        raise ValueError("corr_matrix must have ones on the diagonal.")

    # Eigenvalue-based factorization for PSD correlation matrices
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    if np.any(eigvals < -eps):
        raise ValueError("corr_matrix must be positive semi-definite within eps tolerance.")
    eigvals_clamped = np.where(eigvals < 0.0, 0.0, eigvals)
    L = eigvecs @ np.diag(np.sqrt(eigvals_clamped))

    rng = np.random.default_rng(seed)

    d = S0.size
    dt = T / n_steps
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # Generate independent standard normals: shape (n_paths, n_steps, d)
    Z = rng.standard_normal(size=(n_paths, n_steps, d))

    # Apply correlation: for each path and step, multiply by L
    # Use einsum for vectorized (n_paths, n_steps, d) x (d, d) -> (n_paths, n_steps, d)
    Z_corr = np.einsum("...j,ij->...i", Z, L)

    # Correlated Brownian increments
    dW = Z_corr * np.sqrt(dt)  # shape (n_paths, n_steps, d)

    # Drift term per asset (shape (d,))
    drift = (mu - 0.5 * sigma**2) * dt

    # Prepare arrays in log space
    log_S0 = np.log(S0)
    log_paths = np.empty((n_paths, n_steps + 1, d), dtype=float)
    log_paths[:, 0, :] = log_S0

    # Iterate over time steps in a vectorized way
    for k in range(1, n_steps + 1):
        diffusion_step = sigma * dW[:, k - 1, :]  # shape (n_paths, d)
        log_paths[:, k, :] = log_paths[:, k - 1, :] + drift + diffusion_step

    paths = np.exp(log_paths)
    return t_grid, paths


def simulate_gbm_terminal(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate terminal values S_T of GBM directly (no path storage).
    Exact:
        S_T = S0 * exp((mu - 0.5*sigma^2)T + sigma*sqrt(T)Z)
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    rng = np.random.default_rng(seed)

    if T == 0.0:
        return np.full(n_paths, S0, dtype=float)

    Z = rng.standard_normal(size=n_paths)
    drift = (mu - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)
