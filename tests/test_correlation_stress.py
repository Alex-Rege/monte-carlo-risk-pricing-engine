import numpy as np

from src.correlation_stress import stress_corr_offdiag


def test_stressed_corr_is_symmetric_diag_one():
    base = np.eye(3)
    stressed = stress_corr_offdiag(base, target_rho=0.5)
    assert np.allclose(stressed, stressed.T)
    assert np.allclose(np.diag(stressed), 1.0)


def test_stressed_corr_psd_or_cholesky_passes():
    base = np.eye(4)
    stressed = stress_corr_offdiag(base, target_rho=0.9)
    # stress_corr_offdiag ensures Cholesky usability via small jitter internally;
    # this should succeed without any extra handling here.
    np.linalg.cholesky(stressed + 1e-12 * np.eye(stressed.shape[0]))
