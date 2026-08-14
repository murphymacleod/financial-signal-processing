"""Tests for fft_analysis.py — synthetic and real-data spectral analysis."""

import numpy as np
import pandas as pd

from fft_analysis import (PERIOD_MAX, PERIOD_MIN, compute_log_returns,
                          compute_spectrum, compute_spectrum_freq,
                          generate_synthetic_signal)


def test_synthetic_signal_shape():
    t, s = generate_synthetic_signal(N=256, f1=0.05, f2=0.15, noise_std=0.1)
    assert len(t) == len(s) == 256


def test_fft_recovers_synthetic_frequencies():
    _, s = generate_synthetic_signal(N=1024, f1=0.05, f2=0.15, noise_std=0.1, seed=0)
    freqs, mag = compute_spectrum_freq(s)
    top2 = freqs[np.argsort(mag)[::-1][:2]]
    assert any(abs(f - 0.05) < 0.01 for f in top2)
    assert any(abs(f - 0.15) < 0.01 for f in top2)


def test_compute_log_returns_drops_first_row():
    price = pd.Series([100.0, 110.0, 121.0])
    returns = compute_log_returns(price)
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], np.log(1.1))


def test_compute_spectrum_period_range_respected():
    s = pd.Series(np.random.default_rng(0).standard_normal(500))
    periods, mag = compute_spectrum(s)
    assert (periods >= PERIOD_MIN).all()
    assert (periods <= PERIOD_MAX).all()
    assert len(periods) == len(mag)


def test_compute_spectrum_windowed_vs_unwindowed_same_length():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(400))
    periods_rect, mag_rect = compute_spectrum(s)
    periods_hann, mag_hann = compute_spectrum(s, np.hanning(len(s)))
    assert len(periods_rect) == len(periods_hann)
    assert not np.allclose(mag_rect, mag_hann)
