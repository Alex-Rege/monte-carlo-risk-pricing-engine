import numpy as np

from src.uncertainty import bootstrap_statistic, ci_from_samples


def test_bootstrap_reproducible_with_seed():
    x = np.arange(10, dtype=float)
    stat_fn = np.mean

    reps1 = bootstrap_statistic(x, stat_fn, n_boot=200, seed=123)
    reps2 = bootstrap_statistic(x, stat_fn, n_boot=200, seed=123)
    assert np.array_equal(reps1, reps2)


def test_ci_from_samples_percentile():
    samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lo, hi = ci_from_samples(samples, alpha=0.20, method="percentile")
    # Match numpy's quantile convention used by ci_from_samples.
    assert np.isclose(lo, np.quantile(samples, 0.10))
    assert np.isclose(hi, np.quantile(samples, 0.90))


def test_bootstrap_statistic_mean_reasonable():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=1.5, scale=2.0, size=200)
    sample_mean = float(x.mean())

    reps = bootstrap_statistic(x, np.mean, n_boot=500, seed=999)
    assert np.isclose(reps.mean(), sample_mean, atol=0.05)
