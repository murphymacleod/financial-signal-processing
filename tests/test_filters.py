"""Tests for filters.py — FIR/IIR filter computations."""

import numpy as np
import pandas as pd

from filters import (compute_bandpass, compute_butter_hp, compute_butter_lp,
                     compute_ema, compute_freq_response, compute_sma,
                     compute_wma)


def test_sma_of_constant_series_is_constant():
    s = pd.Series(np.full(100, 5.0))
    sma = compute_sma(s, 10)
    assert np.allclose(sma.dropna(), 5.0)


def test_sma_has_m_minus_1_leading_nans():
    s = pd.Series(np.arange(50, dtype=float))
    sma = compute_sma(s, 10)
    assert sma.iloc[:9].isna().all()
    assert not sma.iloc[9:].isna().any()


def test_wma_of_constant_series_is_constant():
    s = pd.Series(np.full(60, 3.0))
    wma = compute_wma(s, 15)
    assert np.allclose(wma.dropna(), 3.0)


def test_wma_weights_recent_points_more_than_sma_on_a_ramp():
    """On a rising ramp, a linearly-weighted MA leans on recent (higher)
    values more than an unweighted MA, so it should sit above the SMA."""
    s = pd.Series(np.arange(100, dtype=float))
    m = 20
    sma_val = compute_sma(s, m).iloc[-1]
    wma_val = compute_wma(s, m).iloc[-1]
    assert wma_val > sma_val


def test_ema_of_constant_series_is_constant():
    s = pd.Series(np.full(50, 7.0))
    ema = compute_ema(s, 10)
    assert np.allclose(ema, 7.0)


def test_ema_has_no_leading_nans_unlike_sma():
    s = pd.Series(np.arange(30, dtype=float))
    assert not compute_ema(s, 10).isna().any()


def test_butter_lowpass_reduces_high_frequency_variance():
    t = np.arange(1000, dtype=float)
    rng = np.random.default_rng(0)
    noisy = np.sin(2 * np.pi * t / 200) + rng.standard_normal(1000) * 0.5
    s = pd.Series(noisy)
    filtered = compute_butter_lp(s, cutoff_period=20, order=4)
    # low-pass should smooth out sample-to-sample jumps
    assert np.std(np.diff(filtered)) < np.std(np.diff(noisy))


def test_butter_highpass_of_constant_series_is_near_zero():
    s = pd.Series(np.full(300, 42.0))
    filtered = compute_butter_hp(s, cutoff_period=50, order=4)
    assert np.allclose(filtered, 0.0, atol=1e-6)


def test_bandpass_output_same_length_as_input():
    s = pd.Series(np.random.default_rng(0).standard_normal(300))
    out = compute_bandpass(s, low_period=40, high_period=10, numtaps=51)
    assert len(out) == len(s)


def test_bandpass_strongly_attenuates_dc_on_a_constant_series():
    """A bandpass filter passes no DC component, so a flat input should be
    suppressed to a small fraction of its amplitude (a finite-tap FIR filter
    has finite, not infinite, stopband attenuation, so this won't be exact
    zero — checked away from the filtfilt edge-effect region)."""
    amplitude = 10.0
    s = pd.Series(np.full(400, amplitude))
    out = compute_bandpass(s, low_period=40, high_period=10, numtaps=101)
    assert np.abs(out.iloc[150:250]).max() < 0.01 * amplitude


def test_freq_response_arrays_are_consistent_length():
    b = np.ones(10) / 10
    period, mag_db, phase_deg = compute_freq_response(b, 1, npts=2048)
    assert len(period) == len(mag_db) == len(phase_deg)
    assert len(period) > 0


def test_freq_response_period_is_monotonic():
    b = np.ones(10) / 10
    period, _, _ = compute_freq_response(b, 1, npts=2048)
    assert np.all(np.diff(period) < 0)  # frequency increases -> period decreases
