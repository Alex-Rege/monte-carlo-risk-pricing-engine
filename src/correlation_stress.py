from __future__ import annotations

import numpy as np


def validate_corr_matrix(corr, eps: float = 1e-10) -> None:
    """
    Validate that corr is a correlation matrix (symmetric, diag=1, PSD within eps).
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square 2D array.")
    if corr.shape[0] < 1:
        raise ValueError("corr must be non-empty.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")
    if not np.allclose(corr, corr.T, atol=1e-10):
        raise ValueError("corr must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
        raise ValueError("corr must have ones on the diagonal.")
    if np.any(np.abs(corr) > 1.0 + 1e-12):
        raise ValueError("corr entries must be in [-1, 1].")

    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals < -eps):
        raise ValueError("corr must be positive semi-definite within eps tolerance.")


def _cholesky_with_jitter(a: np.ndarray, jitter: float = 1e-12, max_tries: int = 8) -> np.ndarray:
    """
    Attempt Cholesky factorization; if it fails, add diagonal jitter and retry.
    Returns the Cholesky factor if successful; raises on failure.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square 2D array.")

    eye = np.eye(a.shape[0], dtype=float)
    j = float(jitter)
    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(a + j * eye)
        except np.linalg.LinAlgError:
            j *= 10.0
    # Final attempt with the last jitter, so the error message is consistent.
    return np.linalg.cholesky(a + j * eye)


def stress_corr_offdiag(corr, target_rho: float) -> np.ndarray:
    """
    Stress all off-diagonal correlations to target_rho, keeping diag=1.

    This is intended for small demo settings (2-5 assets). The resulting matrix
    is validated and (if needed) stabilized via a tiny diagonal jitter for
    downstream Cholesky use.
    """
    corr = np.asarray(corr, dtype=float)
    validate_corr_matrix(corr)

    d = corr.shape[0]
    target_rho = float(target_rho)
    if not -1.0 < target_rho < 1.0:
        raise ValueError("target_rho must be in (-1, 1).")
    if d > 1:
        # Constant-correlation matrix is PSD for rho in [-1/(d-1), 1].
        rho_min = -1.0 / (d - 1)
        if target_rho < rho_min - 1e-12:
            raise ValueError(f"target_rho too negative for dimension d={d} (min {rho_min}).")

    stressed = np.full((d, d), target_rho, dtype=float)
    np.fill_diagonal(stressed, 1.0)

    # Validate PSD (allowing semidefinite) and ensure Cholesky usability via jitter.
    validate_corr_matrix(stressed)
    _ = _cholesky_with_jitter(stressed, jitter=1e-12, max_tries=8)
    return stressed
