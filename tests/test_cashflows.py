import numpy as np

from src.cashflows import dv01, npv, npv_curve


def test_npv_known_value():
    cashflows = np.array([-100.0, 60.0, 60.0])
    times = np.array([0.0, 1.0, 2.0])
    rate = 0.10

    expected = -100.0 + 60.0 / 1.1 + 60.0 / (1.1**2)
    assert np.isclose(npv(cashflows, times, rate), expected)


def test_npv_monotone_decreasing_in_rate():
    cashflows = np.array([0.0, 0.0, 100.0])
    times = np.array([1.0, 2.0, 3.0])

    npv_low = npv(cashflows, times, rate=0.01)
    npv_high = npv(cashflows, times, rate=0.05)
    assert npv_low > npv_high


def test_npv_curve_matches_loop():
    cashflows = np.array([-50.0, 20.0, 40.0, 60.0])
    times = np.array([0.0, 1.0, 2.0, 3.0])
    rates = np.linspace(0.0, 0.08, 9)

    curve = npv_curve(cashflows, times, rates)
    loop = np.array([npv(cashflows, times, rate) for rate in rates])
    assert np.allclose(curve, loop)


def test_dv01_negative_for_net_positive_future_value():
    cashflows = np.array([0.0, 0.0, 120.0])
    times = np.array([1.0, 2.0, 3.0])
    rate = 0.03

    assert dv01(cashflows, times, rate) < 0.0
