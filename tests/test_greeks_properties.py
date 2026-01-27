import numpy as np

from src.greeks import bs_delta_call, delta_fd_call, delta_pathwise_call
from src.pricing import mc_european_call


def test_delta_fd_crn_reduces_variance():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2
    n_steps = 252
    n_paths = 20_000
    bump = 1e-4

    seeds = list(range(20))

    delta_crn = [
        delta_fd_call(
            S0,
            K,
            T,
            r,
            sigma,
            n_paths,
            seed=seed,
            bump=bump,
            variance_reduction="none",
            return_details=False,
        )
        for seed in seeds
    ]

    # No-CRN baseline: price up/down with independent seeds, then central FD.
    delta_no_crn = []
    for seed in seeds:
        seed_up = 1000 + 2 * seed
        seed_down = 1000 + 2 * seed + 1
        c_up = mc_european_call(
            S0 + bump,
            K,
            T,
            r,
            sigma,
            n_steps,
            n_paths,
            seed=seed_up,
            variance_reduction="none",
            return_details=False,
        )
        c_down = mc_european_call(
            S0 - bump,
            K,
            T,
            r,
            sigma,
            n_steps,
            n_paths,
            seed=seed_down,
            variance_reduction="none",
            return_details=False,
        )
        delta_no_crn.append((c_up - c_down) / (2.0 * bump))

    var_crn = float(np.var(delta_crn, ddof=1))
    var_no_crn = float(np.var(delta_no_crn, ddof=1))

    assert var_crn < 0.7 * var_no_crn


def test_delta_pathwise_lower_variance_than_fd():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2
    n_paths = 20_000
    bump = 1e-4

    seeds = list(range(20))

    delta_fd = [
        delta_fd_call(
            S0,
            K,
            T,
            r,
            sigma,
            n_paths,
            seed=seed,
            bump=bump,
            variance_reduction="none",
            return_details=False,
        )
        for seed in seeds
    ]
    delta_pw = [
        delta_pathwise_call(
            S0,
            K,
            T,
            r,
            sigma,
            n_paths,
            seed=seed,
            variance_reduction="none",
            return_details=False,
        )
        for seed in seeds
    ]

    var_fd = float(np.var(delta_fd, ddof=1))
    var_pw = float(np.var(delta_pw, ddof=1))

    assert var_pw <= 1.1 * var_fd


def test_delta_ci_coverage_sanity():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    sigma = 0.2
    n_paths = 50_000
    bump = 1e-4

    bs_delta = bs_delta_call(S0, K, T, r, sigma)

    seeds = list(range(30))
    hits = 0
    for seed in seeds:
        res = delta_fd_call(
            S0,
            K,
            T,
            r,
            sigma,
            n_paths,
            seed=seed,
            bump=bump,
            variance_reduction="none",
            return_details=True,
        )
        if res.ci[0] <= bs_delta <= res.ci[1]:
            hits += 1

    coverage = hits / len(seeds)
    assert 0.85 <= coverage <= 1.0
