"""Tests for kalman_filter.py — promotes the module's own __main__ sanity checks
into a real regression suite and adds a few properties beyond convergence."""

import numpy as np

from kalman_filter import KalmanFilter1D, KalmanFilter2D


def test_1d_converges_to_true_constant_state():
    rng = np.random.default_rng(0)
    true_price = 100.0
    measurements = true_price + rng.standard_normal(200) * 5.0
    # P0 deliberately large ("completely uncertain" per the class docstring) so
    # variance is guaranteed to shrink toward steady state as evidence accrues —
    # with the default P0=1.0, P can start *below* steady state and rise instead.
    kf = KalmanFilter1D(Q=0.1, R=25.0, x0=measurements[0], P0=1e6)
    estimates, gains, variances = kf.filter(measurements)
    assert abs(estimates[-1] - true_price) < 3.0
    assert 0.0 < gains[-1] < 1.0
    assert variances[-1] < variances[0]


def test_1d_gain_always_between_zero_and_one():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(100)
    kf = KalmanFilter1D(Q=0.5, R=2.0, x0=z[0])
    _, gains, _ = kf.filter(z)
    assert np.all((gains > 0) & (gains < 1))


def test_1d_gain_shrinks_as_filter_gains_confidence():
    """With small, fixed process noise, repeated measurements of a roughly
    stationary series should make the filter progressively more confident
    (gain trending down), not less."""
    rng = np.random.default_rng(3)
    z = 50.0 + rng.standard_normal(300) * 2.0
    kf = KalmanFilter1D(Q=0.01, R=4.0, x0=z[0], P0=10.0)
    _, gains, _ = kf.filter(z)
    assert gains[-1] < gains[10]


def test_2d_recovers_trend_and_price_on_linear_ramp():
    rng = np.random.default_rng(4)
    true_trend = 0.8
    n = 300
    true_prices = np.cumsum(np.full(n, true_trend)) + 20.0
    measurements = true_prices + rng.standard_normal(n) * 2.0
    Q = np.diag([0.3, 0.02])
    kf = KalmanFilter2D(Q=Q, R=6.0, x0=np.array([measurements[0], 0.0]))
    prices, trends, gains = kf.filter(measurements)
    assert abs(prices[-1] - true_prices[-1]) < 5.0
    assert abs(trends[-1] - true_trend) < 0.3
    assert gains.shape == (n, 2)


def test_2d_flat_series_converges_trend_toward_zero():
    rng = np.random.default_rng(5)
    measurements = 75.0 + rng.standard_normal(400) * 1.5
    kf = KalmanFilter2D(Q=np.diag([0.1, 0.001]), R=2.0,
                        x0=np.array([measurements[0], 0.0]))
    _, trends, _ = kf.filter(measurements)
    assert abs(trends[-1]) < 0.2


def test_2d_step_and_filter_agree():
    """step() (single-measurement online interface) and filter() (batch
    interface) must produce identical trajectories for the same inputs."""
    rng = np.random.default_rng(6)
    z = 10.0 + rng.standard_normal(50)

    kf_batch = KalmanFilter2D(Q=np.diag([0.2, 0.01]), R=3.0, x0=np.array([z[0], 0.0]))
    prices_batch, _, _ = kf_batch.filter(z)

    kf_step = KalmanFilter2D(Q=np.diag([0.2, 0.01]), R=3.0, x0=np.array([z[0], 0.0]))
    prices_step = np.array([kf_step.step(zi)[0] for zi in z])

    assert np.allclose(prices_batch, prices_step)
