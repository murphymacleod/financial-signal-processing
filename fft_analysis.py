"""
fft_analysis.py — FFT analysis module for the Financial Signal Processing Platform.

Implements four progressive FFT concepts:
  1. Synthetic signal validation — confirm FFT on a known signal before touching market data
  2. Financial returns spectrum — FFT on log returns with a period-in-days frequency axis
  3. Windowing comparison — spectral leakage demo (Rectangular / Hann / Hamming / Blackman)
  4. Multi-asset spectral comparison — characterise all 5 assets' frequency content
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

ASSETS = {
    "SPY":  "S&P 500 ETF (Equity Index)",
    "QQQ":  "Nasdaq-100 ETF (Technology)",
    "GLD":  "SPDR Gold Shares (Gold)",
    "USO":  "United States Oil Fund (Oil)",
    "CPER": "United States Copper Index Fund (Copper)",
}

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLORS      = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

# Synthetic signal
SYNTH_N     = 512   # samples (~2 years of trading days)
SYNTH_F1    = 0.05  # cycles/day → 20-day period
SYNTH_F2    = 0.15  # cycles/day → ~7-day period
SYNTH_NOISE = 0.5   # noise amplitude relative to unit-amplitude sines

# Period display window (mask everything outside this range on financial spectra)
PERIOD_MIN = 4    # days  — shorter than a trading week is micro-noise
PERIOD_MAX = 500  # days  — longer than this is trend/drift, not a cycle


# ── Asset Selection & Loading ──────────────────────────────────────────────────

def pick_asset() -> tuple[str, str]:
    """Prompt the user to select an asset. Returns (ticker, description)."""
    tickers = list(ASSETS.keys())
    print("\nAvailable assets:")
    for i, (ticker, desc) in enumerate(ASSETS.items(), start=1):
        print(f"  {i}. {ticker:<6} — {desc}")
    print()
    while True:
        raw = input("Enter number or ticker symbol: ").strip().upper()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(tickers):
                ticker = tickers[idx]
                return ticker, ASSETS[ticker]
            print(f"  Please enter a number between 1 and {len(tickers)}.")
        elif raw in ASSETS:
            return raw, ASSETS[raw]
        else:
            print(f"  '{raw}' not recognised. Try a number (1–{len(tickers)}) or a ticker (e.g. SPY).")


def load_asset(ticker: str, data_dir: str) -> pd.DataFrame | None:
    """Load a single asset's CSV. Returns None if the file is missing."""
    path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"  [ERROR] {path} not found. Run download_data.py first.")
        return None
    with open(path) as f:
        f.readline()
        second_line = f.readline()
    skip = [1, 2] if second_line.strip().startswith("Ticker") else []
    return pd.read_csv(path, skiprows=skip, index_col=0,
                       parse_dates=True, date_format="%Y-%m-%d")


# ── Core Computation ───────────────────────────────────────────────────────────

def generate_synthetic_signal(
    N: int, f1: float, f2: float, noise_std: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, signal) for a two-tone sine wave plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    t   = np.arange(N, dtype=float)
    s   = (np.sin(2 * np.pi * f1 * t) +
           np.sin(2 * np.pi * f2 * t) +
           noise_std * rng.standard_normal(N))
    return t, s


def compute_log_returns(price: pd.Series) -> pd.Series:
    """Log returns: r_t = ln(P_t / P_{t-1}). First row is dropped (NaN)."""
    return np.log(price / price.shift(1)).dropna()


def compute_spectrum(
    series: np.ndarray | pd.Series,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided FFT magnitude spectrum, normalised by window sum so windowed
    and non-windowed spectra have comparable amplitudes.

    Returns (periods_days, magnitudes) for the financial frequency range
    [PERIOD_MIN, PERIOD_MAX] days. For synthetic signals pass the raw array
    and interpret the x-axis as cycles/day externally.

    Sampling rate assumed: 1 sample/day (daily data).
    """
    x   = series.values if isinstance(series, pd.Series) else np.asarray(series)
    N   = len(x)
    win = window if window is not None else np.ones(N)
    X   = np.fft.rfft(x * win)
    freqs   = np.fft.rfftfreq(N, d=1.0)   # cycles/day; freqs[0] = 0 (DC)
    mag     = (2.0 / win.sum()) * np.abs(X[1:])   # skip DC
    periods = 1.0 / freqs[1:]
    mask    = (periods >= PERIOD_MIN) & (periods <= PERIOD_MAX)
    return periods[mask], mag[mask]


def compute_spectrum_freq(
    series: np.ndarray | pd.Series,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like compute_spectrum but returns (frequencies_cps, magnitudes) without
    the period mask — used for the synthetic validation plot.
    """
    x   = series.values if isinstance(series, pd.Series) else np.asarray(series)
    N   = len(x)
    win = window if window is not None else np.ones(N)
    X   = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(N, d=1.0)
    mag   = (2.0 / win.sum()) * np.abs(X)
    return freqs, mag


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _apply_date_axis(ax: plt.Axes) -> None:
    """Year-tick x-axis, matching project-wide style."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


def _period_axis(ax: plt.Axes) -> None:
    """Apply log-scale period (days) formatting to x-axis."""
    ax.set_xscale("log")
    ax.set_xlabel("Period (days, log scale)")
    ax.set_xlim(PERIOD_MIN, PERIOD_MAX)
    # Custom ticks at round day values
    ticks = [5, 10, 20, 50, 100, 252, 500]
    ax.set_xticks([t for t in ticks if PERIOD_MIN <= t <= PERIOD_MAX])
    ax.set_xticklabels([str(t) for t in ticks if PERIOD_MIN <= t <= PERIOD_MAX])
    ax.grid(True, alpha=0.3, which="both")


# ── Figure 1: Synthetic validation ────────────────────────────────────────────

def plot_synthetic_validation(
    t: np.ndarray, s: np.ndarray, f1: float, f2: float
) -> plt.Figure:
    """
    Validate the FFT on a synthetic 2-tone signal with known frequencies.
    Panel 1: signal vs sample index.
    Panel 2: FFT magnitude vs frequency (cycles/day) — peaks must align with f1, f2.
    """
    freqs, mag = compute_spectrum_freq(s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # ── Panel 1: signal ────────────────────────────────────────────────────────
    ax1.plot(t, s, color="#2196F3", linewidth=0.8, alpha=0.85)
    ax1.set_title(
        f"Synthetic Signal: sin(2π·{f1}·t) + sin(2π·{f2}·t) + noise  "
        f"(N={len(t)} samples, noise σ={SYNTH_NOISE})",
        fontsize=12, fontweight="bold",
    )
    ax1.set_xlabel("Sample index (days)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: FFT magnitude — validation ───────────────────────────────────
    ax2.plot(freqs, mag, color="#37474F", linewidth=1.0)
    ax2.axvline(f1, color="#FF5722", linewidth=1.2, linestyle="--",
                label=f"f₁ = {f1} c/day  (T = {1/f1:.0f} day)")
    ax2.axvline(f2, color="#4CAF50", linewidth=1.2, linestyle="--",
                label=f"f₂ = {f2} c/day  (T = {1/f2:.0f} day)")
    ax2.set_title("FFT Magnitude Spectrum — peaks must align with f₁ and f₂", fontsize=11)
    ax2.set_xlabel("Frequency (cycles/day)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, 0.5)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ── Figure 2: Financial returns FFT ───────────────────────────────────────────

def plot_returns_fft(
    returns: pd.Series,
    ticker: str,
    description: str,
    color: str,
) -> plt.Figure:
    """
    Panel 1: Log returns vs date (green/red bars).
    Panel 2: FFT magnitude vs period in days (log scale).
    """
    periods, mag = compute_spectrum(returns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ── Panel 1: returns ───────────────────────────────────────────────────────
    pos = returns >= 0
    ax1.bar(returns.index[pos],  returns.values[pos],
            color="#4CAF50", width=1, alpha=0.85)
    ax1.bar(returns.index[~pos], returns.values[~pos],
            color="#F44336", width=1, alpha=0.85)
    ax1.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax1.set_title(f"{ticker} — {description}  |  Daily Log Returns",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Log Return")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax1.grid(True, alpha=0.3, axis="y")
    _apply_date_axis(ax1)

    # ── Panel 2: spectrum ──────────────────────────────────────────────────────
    ax2.plot(periods, mag, color=color, linewidth=1.2)
    ax2.axvline(50,  color="gray", linewidth=0.8, linestyle="--", alpha=0.7,
                label="50-day reference")
    ax2.axvline(252, color="#9E9E9E", linewidth=0.8, linestyle=":",  alpha=0.7,
                label="252-day (annual) reference")
    ax2.set_title("FFT Magnitude Spectrum — Log Returns (no window, rectangular)",
                  fontsize=11)
    ax2.set_ylabel("Magnitude")
    ax2.legend(fontsize=9, framealpha=0.9)
    _period_axis(ax2)

    fig.tight_layout()
    return fig


# ── Figure 3: Windowing comparison ────────────────────────────────────────────

WINDOW_STYLES: dict[str, tuple[str, str]] = {
    "Rectangular (no window)": ("#9E9E9E", "-"),
    "Hann":                    ("#2196F3", "-"),
    "Hamming":                 ("#FF5722", "-"),
    "Blackman":                ("#37474F", "-"),
}


def plot_window_comparison(
    returns: pd.Series, ticker: str, description: str
) -> plt.Figure:
    """
    All four windowed spectra overlaid on one panel.
    Spectral leakage is visible as wide "skirts" around peaks in the rectangular spectrum;
    Hann / Hamming / Blackman progressively suppress it at the cost of frequency resolution.
    """
    N = len(returns)
    windows = {
        "Rectangular (no window)": np.ones(N),
        "Hann":                    np.hanning(N),
        "Hamming":                 np.hamming(N),
        "Blackman":                np.blackman(N),
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    for (name, (color, ls)), win in zip(WINDOW_STYLES.items(), windows.values()):
        periods, mag = compute_spectrum(returns, win)
        ax.plot(periods, mag, color=color, linewidth=1.2, linestyle=ls, label=name, alpha=0.9)

    ax.set_title(
        f"{ticker} — {description}  |  Spectral Leakage & Windowing Comparison\n"
        "Rectangular window shows broad leakage; Blackman has narrowest main lobe",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Normalised Magnitude")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    _period_axis(ax)

    fig.tight_layout()
    return fig


# ── Figure 4: Multi-asset spectral comparison ──────────────────────────────────

def plot_asset_comparison(
    asset_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
) -> plt.Figure:
    """
    Overlay all 5 assets' Hann-windowed spectra, each peak-normalised to [0, 1]
    so amplitude differences don't obscure shape differences.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for (ticker, (periods, mag)), color in zip(asset_spectra.items(), COLORS):
        peak = mag.max() if mag.max() > 0 else 1.0
        ax.plot(periods, mag / peak, color=color, linewidth=1.3,
                label=f"{ticker} — {ASSETS[ticker]}", alpha=0.85)

    ax.set_title(
        "Multi-Asset Spectral Comparison — Hann-Windowed Log Returns (Peak-Normalised)\n"
        "Shape reveals dominant cycle lengths per asset class",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Normalised Magnitude (peak = 1)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    _period_axis(ax)

    fig.tight_layout()
    return fig


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── [1/4] Synthetic validation (no user input) ─────────────────────────────
    print("\n  [1/4] Generating synthetic signal validation...")
    t, s = generate_synthetic_signal(SYNTH_N, SYNTH_F1, SYNTH_F2, SYNTH_NOISE)
    fig1 = plot_synthetic_validation(t, s, SYNTH_F1, SYNTH_F2)
    path1 = os.path.join(FIGURES_DIR, "fft_synthetic_validation.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")

    # ── [2/4] & [3/4] Single-asset analyses ───────────────────────────────────
    ticker, description = pick_asset()
    df = load_asset(ticker, DATA_DIR)
    if df is None:
        return

    color   = COLORS[list(ASSETS.keys()).index(ticker)]
    returns = compute_log_returns(df["Close"])
    print(f"  {ticker}: {len(returns)} return observations  "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    print("  [2/4] Computing returns FFT...")
    fig2  = plot_returns_fft(returns, ticker, description, color)
    path2 = os.path.join(FIGURES_DIR, f"{ticker}_returns_fft.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")

    print("  [3/4] Computing windowing comparison...")
    fig3  = plot_window_comparison(returns, ticker, description)
    path3 = os.path.join(FIGURES_DIR, f"{ticker}_window_comparison.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path3}")

    # ── [4/4] Multi-asset comparison (all 5 assets, Hann window) ──────────────
    print("  [4/4] Computing multi-asset spectral comparison...")
    asset_spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t_name in ASSETS:
        df_t = load_asset(t_name, DATA_DIR)
        if df_t is None:
            print(f"    Skipping {t_name} — file not found.")
            continue
        ret_t  = compute_log_returns(df_t["Close"])
        win_t  = np.hanning(len(ret_t))
        p, m   = compute_spectrum(ret_t, win_t)
        asset_spectra[t_name] = (p, m)

    fig4  = plot_asset_comparison(asset_spectra)
    path4 = os.path.join(FIGURES_DIR, "fft_asset_comparison.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path4}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
