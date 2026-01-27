from __future__ import annotations

import numpy as np


def _validate_inputs(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    n_S: int,
    n_t: int,
):
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_S < 3:
        raise ValueError("n_S must be >= 3.")
    if n_t < 1:
        raise ValueError("n_t must be >= 1.")
    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")
    return option_type


def _boundary_values(option_type: str, K: float, r: float, T: float, t: float, S_max: float):
    tau = T - t
    if option_type == "call":
        return 0.0, S_max - K * np.exp(-r * tau)
    return K * np.exp(-r * tau), 0.0


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    if n == 1:
        return np.array([d[0] / b[0]])

    c_prime = np.empty(n - 1, dtype=float)
    d_prime = np.empty(n, dtype=float)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    denom = b[n - 1] - a[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def solve_bs_pde_cn(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_S: int = 400,
    n_t: int = 400,
    S_max: float | None = None,
    return_full_grid: bool = False,
    return_t0_grid: bool = False,
):
    """
    Solve the Black--Scholes PDE using a Crank--Nicolson scheme.

    Returns (S_grid, V_0) by default. If return_full_grid=True, returns
    (S_grid, t_grid, V_grid) with V_grid[k, i] = V(t_k, S_i).
    If return_t0_grid=True, returns (S_grid, V_0) regardless of return_full_grid.
    """
    option_type = _validate_inputs(S0, K, T, r, sigma, option_type, n_S, n_t)
    if T == 0.0:
        S_grid = np.linspace(0.0, float(S_max or max(S0, K) * 5.0), n_S + 1)
        if option_type == "call":
            V0 = np.maximum(S_grid - K, 0.0)
        else:
            V0 = np.maximum(K - S_grid, 0.0)
        return (S_grid, V0) if not return_full_grid else (S_grid, np.array([0.0]), V0[None, :])

    if S_max is None:
        S_max = float(max(S0, K) * 5.0)

    dS = S_max / n_S
    dt = T / n_t

    S_grid = np.linspace(0.0, S_max, n_S + 1)
    t_grid = np.linspace(0.0, T, n_t + 1)

    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)

    if return_full_grid:
        V_grid = np.empty((n_t + 1, n_S + 1), dtype=float)
        V_grid[-1, :] = V

    S_inner = S_grid[1:-1]
    a = 0.5 * sigma**2 * S_inner**2 / dS**2 - 0.5 * r * S_inner / dS
    b = -(sigma**2 * S_inner**2 / dS**2 + r)
    c = 0.5 * sigma**2 * S_inner**2 / dS**2 + 0.5 * r * S_inner / dS

    lower = -0.5 * dt * a
    diag = 1.0 - 0.5 * dt * b
    upper = -0.5 * dt * c

    lower_rhs = 0.5 * dt * a
    diag_rhs = 1.0 + 0.5 * dt * b
    upper_rhs = 0.5 * dt * c

    for n in range(n_t - 1, -1, -1):
        t = t_grid[n]
        t_next = t_grid[n + 1]

        bc_low_next, bc_high_next = _boundary_values(option_type, K, r, T, t_next, S_max)
        V[0] = bc_low_next
        V[-1] = bc_high_next

        rhs = (
            lower_rhs * V[:-2]
            + diag_rhs * V[1:-1]
            + upper_rhs * V[2:]
        )

        bc_low, bc_high = _boundary_values(option_type, K, r, T, t, S_max)
        rhs[0] -= lower[0] * bc_low
        rhs[-1] -= upper[-1] * bc_high

        a_sys = lower[1:]
        b_sys = diag
        c_sys = upper[:-1]
        V_inner = _thomas_solve(a_sys, b_sys, c_sys, rhs)

        V[0] = bc_low
        V[-1] = bc_high
        V[1:-1] = V_inner

        if return_full_grid:
            V_grid[n, :] = V

    if return_t0_grid:
        return S_grid, V
    if return_full_grid:
        return S_grid, t_grid, V_grid
    return S_grid, V


def price_bs_pde_cn(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_S: int = 400,
    n_t: int = 400,
    S_max: float | None = None,
) -> float:
    """
    Price a European option with the Black--Scholes PDE via Crank--Nicolson.
    """
    if T == 0.0:
        return max(S0 - K, 0.0) if str(option_type).lower() == "call" else max(K - S0, 0.0)

    S_grid, V0 = solve_bs_pde_cn(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        n_S=n_S,
        n_t=n_t,
        S_max=S_max,
        return_full_grid=False,
    )
    return float(np.interp(S0, S_grid, V0))
