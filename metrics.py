"""
metrics.py — Filter performance metrics for the Financial Signal Processing Platform.

Pure functions: no classes, no plotting, no IO.  Accept numpy arrays or pandas Series.
Every function strips NaN before computing so callers do not need to pre-clean data.
"""

import numpy as np


# ── Internal helper ────────────────────────────────────────────────────────────

def _clean(true, estimated) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and drop positions where either is NaN."""
    t = np.asarray(true,      dtype=float)
    e = np.asarray(estimated, dtype=float)
    mask = np.isfinite(t) & np.isfinite(e)
    return t[mask], e[mask]


# ── Metric functions ───────────────────────────────────────────────────────────

def rmse(true, estimated) -> float:
    """Root mean squared error.  Lower = better point accuracy."""
    t, e = _clean(true, estimated)
    return float(np.sqrt(np.mean((t - e) ** 2)))


def mae(true, estimated) -> float:
    """Mean absolute error.  Lower = better; less sensitive to outliers than RMSE."""
    t, e = _clean(true, estimated)
    return float(np.mean(np.abs(t - e)))


def noise_reduction_db(signal, filtered) -> float:
    """
    Noise reduction in decibels.

        NR = 20 · log₁₀(σ_signal / σ_residual)

    where residual = signal − filtered.  Measures how much variance the filter
    removed.  Higher is better.  Returns +inf if the filter is perfect.
    """
    s, f = _clean(signal, filtered)
    sd_in  = np.std(s)
    sd_res = np.std(s - f)
    if sd_in == 0:
        return 0.0
    if sd_res == 0:
        return float("inf")
    return float(20.0 * np.log10(sd_in / sd_res))


def tracking_error(true, estimated) -> float:
    """
    Standard deviation of the estimation error (true − estimated).
    Measures how tightly the filter follows the reference signal.  Lower = better.
    """
    t, e = _clean(true, estimated)
    return float(np.std(t - e))


def filter_lag(signal, filtered, max_lag: int = 100) -> int:
    """
    Estimate filter lag in samples via cross-correlation.

    A causal smoothing filter trails its input; the lag is the sample offset
    (≥ 0) at which the cross-correlation of signal and filtered output is
    maximised.  Returns 0 if the peak is at zero offset or no data.

    Parameters
    ----------
    signal   : reference series (e.g. raw price)
    filtered : filter output to evaluate
    max_lag  : maximum lag to consider (default 100 samples)
    """
    s, f = _clean(signal, filtered)
    if len(s) == 0:
        return 0
    s = s - s.mean()
    f = f - f.mean()
    # mode="full" returns a vector of length 2N-1; zero-lag is at index N-1
    corr   = np.correlate(s, f, mode="full")
    center = len(s) - 1
    end    = min(center + max_lag + 1, len(corr))
    window = corr[center:end]
    return int(np.argmax(window))


def summarize(label: str, true, estimated) -> dict:
    """
    Compute all metrics and return them as a dict.
    Convenient for assembling pandas DataFrame comparison tables.
    """
    t, e = np.asarray(true, dtype=float), np.asarray(estimated, dtype=float)
    return {
        "Filter":               label,
        "RMSE":                 round(rmse(t, e), 4),
        "MAE":                  round(mae(t, e), 4),
        "Noise Reduction (dB)": round(noise_reduction_db(t, e), 2),
        "Tracking Error":       round(tracking_error(t, e), 4),
        "Lag (days)":           filter_lag(t, e),
    }


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N = 500

    true_signal = np.cumsum(rng.standard_normal(N))
    noisy       = true_signal + rng.standard_normal(N) * 3.0
    smoothed    = np.convolve(noisy, np.ones(20) / 20.0, mode="same")

    r   = rmse(true_signal, smoothed)
    m   = mae(true_signal, smoothed)
    nr  = noise_reduction_db(noisy, smoothed)
    te  = tracking_error(true_signal, smoothed)
    lag = filter_lag(noisy, smoothed)

    print(f"RMSE            : {r:.4f}")
    print(f"MAE             : {m:.4f}")
    print(f"Noise reduction : {nr:.2f} dB")
    print(f"Tracking error  : {te:.4f}")
    print(f"Filter lag      : {lag} samples")
    print()
    print(summarize("SMA20", true_signal, smoothed))

    assert r > 0, "RMSE must be positive"
    assert nr > 0, "SMA should reduce noise"
    assert lag >= 0, "Lag must be non-negative"
    print("\nAll sanity checks passed.")
