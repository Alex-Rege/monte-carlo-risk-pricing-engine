import numpy as np


def _validate_cashflows_times(cashflows, times):
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)

    if cashflows.ndim != 1 or times.ndim != 1:
        raise ValueError("cashflows and times must be 1D arrays.")
    if cashflows.size == 0:
        raise ValueError("cashflows must be non-empty.")
    if cashflows.shape != times.shape:
        raise ValueError("cashflows and times must have the same length.")
    return cashflows, times


def _validate_rate(rate: float) -> float:
    rate = float(rate)
    if rate <= -1.0:
        raise ValueError("rate must be greater than -1.")
    return rate


def npv(cashflows, times, rate: float) -> float:
    """
    Compute the net present value (NPV) of a cashflow stream at a flat rate.

    Discounting uses (1 + rate) ** time, with times in years.
    """
    cashflows, times = _validate_cashflows_times(cashflows, times)
    rate = _validate_rate(rate)
    discount_factors = np.power(1.0 + rate, times)
    return float(np.sum(cashflows / discount_factors))


def npv_curve(cashflows, times, rates) -> np.ndarray:
    """
    Compute NPVs across a grid of discount rates.

    Returns an array with the same shape as rates.
    """
    cashflows, times = _validate_cashflows_times(cashflows, times)
    rates = np.asarray(rates, dtype=float)
    if np.any(rates <= -1.0):
        raise ValueError("rates must be greater than -1.")

    discount_factors = np.power(1.0 + rates[..., None], times)
    return np.sum(cashflows / discount_factors, axis=-1)


def dv01(cashflows, times, rate: float, bump: float = 1e-4) -> float:
    """
    Central finite-difference sensitivity of NPV to the discount rate.
    """
    bump = float(bump)
    if bump <= 0.0:
        raise ValueError("bump must be positive.")
    rate = _validate_rate(rate)
    if rate - bump <= -1.0:
        raise ValueError("rate - bump must be greater than -1.")
    npv_up = npv(cashflows, times, rate + bump)
    npv_down = npv(cashflows, times, rate - bump)
    return (npv_up - npv_down) / (2.0 * bump)
