"""Tests for metrics.py — pure filter-performance metrics."""

import numpy as np
import pytest

import metrics


def test_rmse_zero_for_identical_series():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.rmse(x, x) == 0.0


def test_rmse_known_value():
    true = np.array([0.0, 0.0, 0.0])
    est  = np.array([3.0, 4.0, 0.0])
    assert metrics.rmse(true, est) == pytest.approx(np.sqrt((9 + 16) / 3))


def test_mae_known_value():
    true = np.array([0.0, 0.0])
    est  = np.array([3.0, -3.0])
    assert metrics.mae(true, est) == pytest.approx(3.0)


def test_noise_reduction_db_positive_when_filter_reduces_variance():
    rng = np.random.default_rng(0)
    noisy = rng.standard_normal(1000) * 5.0
    smoothed = np.convolve(noisy, np.ones(20) / 20, mode="same")
    assert metrics.noise_reduction_db(noisy, smoothed) > 0


def test_noise_reduction_db_zero_signal_returns_zero():
    zeros = np.zeros(10)
    assert metrics.noise_reduction_db(zeros, zeros) == 0.0


def test_tracking_error_matches_manual_std():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    est  = np.array([1.5, 2.5, 2.5, 4.5])
    expected = np.std(true - est)
    assert metrics.tracking_error(true, est) == pytest.approx(expected)


def test_filter_lag_recovers_known_delay():
    """Regression test for the sign-convention bug found while building the
    project's out-of-sample backtest: filter_lag used to peak on the wrong
    side of the cross-correlation and silently report ~0 lag regardless of
    the true delay. See metrics.py's filter_lag docstring/comment."""
    t = np.arange(2000)
    signal = np.sin(2 * np.pi * t / 33) + 0.3 * np.sin(2 * np.pi * t / 11)
    lag_true = 9
    filtered = np.roll(signal, lag_true)
    filtered[:lag_true] = 0.0
    assert metrics.filter_lag(signal, filtered, max_lag=50) == lag_true


def test_filter_lag_zero_when_no_delay():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    assert metrics.filter_lag(x, x, max_lag=20) == 0


def test_filter_lag_empty_input_returns_zero():
    assert metrics.filter_lag(np.array([]), np.array([])) == 0


def test_clean_drops_nan_pairs():
    t = np.array([1.0, np.nan, 3.0])
    e = np.array([1.0, 2.0, np.nan])
    tc, ec = metrics._clean(t, e)
    assert len(tc) == 1
    assert tc[0] == 1.0 and ec[0] == 1.0


def test_summarize_returns_all_expected_keys():
    true = np.linspace(0, 10, 50)
    est  = true + 0.1
    row = metrics.summarize("test", true, est)
    assert set(row) == {"Filter", "RMSE", "MAE", "Noise Reduction (dB)",
                        "Tracking Error", "Lag (days)"}
    assert row["Filter"] == "test"
