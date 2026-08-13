"""
filters.py — FIR and IIR filter demonstrations for the Financial Signal Processing Platform.

Applies FIR (SMA, WMA, high-pass, bandpass) and IIR (EMA, Butterworth low-pass, high-pass,
bandpass) filters to a user-selected asset's price series and produces three figures:
  1. FIR filter overview (3 stacked panels)
  2. IIR filter overview (3 stacked panels, parallel layout)
  3. FIR vs IIR comparison — time domain + magnitude + phase frequency response
No data is downloaded here; only saved CSVs are consumed.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

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

MA_ORDER       = 50   # M for SMA and WMA (number of points); EMA span; Butterworth LP/HP cutoff
BP_LOW_PERIOD  = 40   # bandpass: reject periods longer than this (days)
BP_HIGH_PERIOD = 10   # bandpass: reject periods shorter than this (days)
BP_NUMTAPS     = 101  # FIR bandpass filter length (must be odd for firwin bandpass)
BUTTER_ORDER   = 4    # Butterworth filter order (4th-order: steep roll-off, numerically stable)
FREQ_RESP_NPTS = 4096 # number of evaluation points for freqz frequency response


# ── Asset Selection ────────────────────────────────────────────────────────────

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


# ── Load ───────────────────────────────────────────────────────────────────────

def load_asset(ticker: str, data_dir: str) -> pd.DataFrame | None:
    """Load a single asset's CSV. Returns None if the file is missing."""
    path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"  [ERROR] {path} not found. Run download_data.py first.")
        return None
    # Newer yfinance writes a 3-row header (Price / Ticker / Date); detect and skip.
    with open(path) as f:
        f.readline()
        second_line = f.readline()
    skip = [1, 2] if second_line.strip().startswith("Ticker") else []
    return pd.read_csv(path, skiprows=skip, index_col=0,
                       parse_dates=True, date_format="%Y-%m-%d")


# ── FIR Filter Computations ────────────────────────────────────────────────────

def compute_sma(series: pd.Series, m: int) -> pd.Series:
    """M-point simple moving average (rectangular-window low-pass FIR)."""
    return series.rolling(m).mean()


def compute_wma(series: pd.Series, m: int) -> pd.Series:
    """M-point weighted moving average (linearly increasing weights, low-pass FIR)."""
    weights = np.arange(1, m + 1, dtype=float)
    return series.rolling(m).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def compute_bandpass(series: pd.Series, low_period: int, high_period: int,
                     numtaps: int) -> pd.Series:
    """
    Bandpass FIR filter passing cycles with periods between high_period and low_period days.
    Uses scipy.signal.firwin + filtfilt (zero-phase, no group delay).

    With daily data: fs = 1 sample/day, Nyquist = 0.5 cycles/day.
      low_cutoff  = 1/low_period  (e.g. 1/40 = 0.025 cycles/day → 0.05 normalized)
      high_cutoff = 1/high_period (e.g. 1/10 = 0.10  cycles/day → 0.20 normalized)
    """
    nyq  = 0.5
    low  = (1.0 / low_period)  / nyq
    high = (1.0 / high_period) / nyq
    b    = signal.firwin(numtaps, [low, high], pass_zero=False)
    out  = signal.filtfilt(b, 1, series.values)
    return pd.Series(out, index=series.index)


# ── IIR Filter Computations ───────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average — the simplest 1st-order IIR filter.
    α = 2/(span+1); each output depends on all past inputs (infinite impulse response).
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_butter_lp(series: pd.Series, cutoff_period: int, order: int) -> pd.Series:
    """Butterworth low-pass IIR filter (zero-phase via filtfilt).
    cutoff_period in days; normalized cutoff = (1/cutoff_period) / Nyquist.
    """
    Wn = (1.0 / cutoff_period) / 0.5
    b, a = signal.butter(order, Wn, btype="low")
    return pd.Series(signal.filtfilt(b, a, series.values), index=series.index)


def compute_butter_hp(series: pd.Series, cutoff_period: int, order: int) -> pd.Series:
    """Butterworth high-pass IIR filter (zero-phase via filtfilt)."""
    Wn = (1.0 / cutoff_period) / 0.5
    b, a = signal.butter(order, Wn, btype="high")
    return pd.Series(signal.filtfilt(b, a, series.values), index=series.index)


def compute_butter_bp(series: pd.Series, low_period: int, high_period: int,
                      order: int) -> pd.Series:
    """Butterworth bandpass IIR filter (zero-phase via filtfilt).
    Passes cycles with periods between high_period and low_period days.
    """
    Wn = [(1.0 / low_period) / 0.5, (1.0 / high_period) / 0.5]
    b, a = signal.butter(order, Wn, btype="band")
    return pd.Series(signal.filtfilt(b, a, series.values), index=series.index)


def compute_freq_response(
    b: np.ndarray, a: np.ndarray | int, npts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (period_days, magnitude_dB, phase_degrees) for a filter (b, a).
    Periods shorter than 4 days or longer than 500 days are masked out.
    """
    w, H  = signal.freqz(b, a, worN=npts)
    freq  = w / (2 * np.pi)                       # cycles/day (0 to 0.5)
    mask  = (freq > 1.0 / 500) & (freq < 1.0 / 4)
    period   = 1.0 / freq[mask]
    mag_db   = 20 * np.log10(np.abs(H[mask]) + 1e-12)
    phase_d  = np.degrees(np.unwrap(np.angle(H[mask])))
    return period, mag_db, phase_d


# ── Plotting ───────────────────────────────────────────────────────────────────

def _apply_date_axis(ax: plt.Axes) -> None:
    """Year-tick x-axis formatting, matching project-wide style."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


def plot_fir_filters(
    price: pd.Series,
    sma: pd.Series,
    wma: pd.Series,
    hp_unweighted: pd.Series,
    hp_weighted: pd.Series,
    bandpass: pd.Series,
    ticker: str,
    description: str,
    color: str,
    m: int,
    bp_low: int,
    bp_high: int,
) -> plt.Figure:
    """
    Three stacked panels (shared x-axis):
      Panel 1 — Price + SMA{m} + WMA{m} on the same axis
      Panel 2 — Unweighted and weighted high-pass residuals on the same axis
      Panel 3 — Bandpass filter output ({bp_high}–{bp_low} day passband) alone
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 1.5]},
    )

    # ── Panel 1: Price + Low-Pass MAs ─────────────────────────────────────────
    ax1.plot(price.index, price,
             color=color,    linewidth=1.5, label="Price",        zorder=4)
    ax1.plot(sma.index,   sma,
             color="#757575", linewidth=1.1, linestyle="--",
             label=f"SMA{m}", zorder=3)
    ax1.plot(wma.index,   wma,
             color="#37474F", linewidth=1.1, linestyle=":",
             label=f"WMA{m}", zorder=3)
    ax1.set_title(f"{ticker} — {description}  |  FIR Low-Pass (M={m})",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Close Price (USD)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: High-Pass Residuals ───────────────────────────────────────────
    ax2.plot(hp_unweighted.index, hp_unweighted,
             color="#757575", linewidth=1.0, label=f"HP Unweighted (Price − SMA{m})")
    ax2.plot(hp_weighted.index,   hp_weighted,
             color="#37474F", linewidth=1.0, label=f"HP Weighted   (Price − WMA{m})")
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("High-Pass Residual (USD)")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Bandpass Output ───────────────────────────────────────────────
    ax3.plot(bandpass.index, bandpass,
             color="#FF9800", linewidth=1.3,
             label=f"Bandpass ({bp_high}–{bp_low} day)")
    ax3.fill_between(bandpass.index, bandpass,
                     color="#FF9800", alpha=0.15)
    ax3.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax3.set_title(f"Bandpass FIR  |  Passband: {bp_high}–{bp_low} day cycles  "
                  f"({BP_NUMTAPS}-tap, zero-phase)",
                  fontsize=10)
    ax3.set_ylabel(f"Bandpass ({bp_high}–{bp_low} day)")
    ax3.grid(True, alpha=0.3)
    _apply_date_axis(ax3)

    fig.tight_layout()
    return fig


def plot_iir_filters(
    price: pd.Series,
    ema: pd.Series,
    butter_lp: pd.Series,
    hp_ema: pd.Series,
    butter_hp: pd.Series,
    butter_bp: pd.Series,
    ticker: str,
    description: str,
    color: str,
    m: int,
    bp_low: int,
    bp_high: int,
) -> plt.Figure:
    """
    Three stacked panels — mirrors plot_fir_filters layout but for IIR filters:
      Panel 1 — Price + EMA{m} + Butterworth LP (same axis)
      Panel 2 — EMA residual HP and Butterworth HP (same axis)
      Panel 3 — Butterworth bandpass alone
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 1.5]},
    )

    # ── Panel 1: Price + IIR Low-Pass filters ─────────────────────────────────
    ax1.plot(price.index, price,
             color=color,     linewidth=1.5, label="Price",                zorder=4)
    ax1.plot(ema.index, ema,
             color="#757575", linewidth=1.1, linestyle="--",
             label=f"EMA{m} (1st-order IIR)",                              zorder=3)
    ax1.plot(butter_lp.index, butter_lp,
             color="#37474F", linewidth=1.1, linestyle=":",
             label=f"Butterworth LP (order {BUTTER_ORDER}, {m}-day cutoff)", zorder=3)
    ax1.set_title(f"{ticker} — {description}  |  IIR Low-Pass (cutoff ≈ {m} day)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Close Price (USD)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: IIR High-Pass signals ────────────────────────────────────────
    ax2.plot(hp_ema.index, hp_ema,
             color="#757575", linewidth=1.0,
             label=f"EMA HP residual (Price − EMA{m})")
    ax2.plot(butter_hp.index, butter_hp,
             color="#37474F", linewidth=1.0,
             label=f"Butterworth HP (order {BUTTER_ORDER})")
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("High-Pass Residual (USD)")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Butterworth Bandpass ──────────────────────────────────────────
    ax3.plot(butter_bp.index, butter_bp,
             color="#FF9800", linewidth=1.3)
    ax3.fill_between(butter_bp.index, butter_bp, color="#FF9800", alpha=0.15)
    ax3.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax3.set_title(
        f"Butterworth Bandpass  |  Passband: {bp_high}–{bp_low} day cycles  "
        f"(order {BUTTER_ORDER}, zero-phase)",
        fontsize=10,
    )
    ax3.set_ylabel(f"Bandpass ({bp_high}–{bp_low} day)")
    ax3.grid(True, alpha=0.3)
    _apply_date_axis(ax3)

    fig.tight_layout()
    return fig


def plot_fir_vs_iir(
    price: pd.Series,
    fir_sma_causal: pd.Series,
    iir_butter_causal: pd.Series,
    iir_ema: pd.Series,
    b_sma: np.ndarray,
    b_bw: np.ndarray,
    a_bw: np.ndarray,
    b_ema: np.ndarray,
    a_ema: np.ndarray,
    ticker: str,
    color: str,
    m: int,
) -> plt.Figure:
    """
    Three panels — time domain (shared date x-axis) then two frequency-domain panels
    (period-in-days log x-axis, NOT shared with panel 1):
      Panel 1 — Price + FIR SMA (causal) + IIR Butter LP (causal) + IIR EMA (causal)
      Panel 2 — Magnitude response |H(f)| in dB vs period
      Panel 3 — Phase response ∠H(f) in degrees vs period
    """
    fig = plt.figure(figsize=(12, 11))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # ── Panel 1: Time-domain — causal filters show real-world phase lag ────────
    ax1.plot(price.index, price,
             color=color,    linewidth=1.3, alpha=0.5, label="Price", zorder=2)
    ax1.plot(fir_sma_causal.index, fir_sma_causal,
             color="#2196F3", linewidth=1.4, label=f"FIR SMA{m} (causal, linear phase)", zorder=4)
    ax1.plot(iir_butter_causal.index, iir_butter_causal,
             color="#FF5722", linewidth=1.4, label=f"IIR Butterworth LP (causal, order {BUTTER_ORDER})", zorder=4)
    ax1.plot(iir_ema.index, iir_ema,
             color="#4CAF50", linewidth=1.4, label=f"IIR EMA{m} (causal, 1st-order)", zorder=4)
    ax1.set_title(
        f"{ticker}  |  FIR vs IIR — Causal Low-Pass Comparison (cutoff ≈ {m} day)",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Close Price (USD)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)
    _apply_date_axis(ax1)

    # ── Frequency responses ───────────────────────────────────────────────────
    p_sma, mag_sma, ph_sma = compute_freq_response(b_sma, 1,     FREQ_RESP_NPTS)
    p_bw,  mag_bw,  ph_bw  = compute_freq_response(b_bw,  a_bw,  FREQ_RESP_NPTS)
    p_ema, mag_ema, ph_ema = compute_freq_response(b_ema, a_ema, FREQ_RESP_NPTS)

    # ── Panel 2: Magnitude response ────────────────────────────────────────────
    ax2.semilogx(p_sma, mag_sma, color="#2196F3", linewidth=1.4, label=f"FIR SMA{m}")
    ax2.semilogx(p_bw,  mag_bw,  color="#FF5722", linewidth=1.4,
                 label=f"IIR Butterworth LP (order {BUTTER_ORDER})")
    ax2.semilogx(p_ema, mag_ema, color="#4CAF50", linewidth=1.4, label=f"IIR EMA{m}")
    ax2.axvline(m, color="gray", linewidth=0.8, linestyle="--", alpha=0.7,
                label=f"{m}-day cutoff")
    ax2.axhline(-3, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
    ax2.set_xlabel("Period (days, log scale)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Frequency Response — Magnitude", fontsize=11)
    ax2.set_xlim(4, 500)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3, which="both")

    # ── Panel 3: Phase response ────────────────────────────────────────────────
    ax3.semilogx(p_sma, ph_sma, color="#2196F3", linewidth=1.4,
                 label=f"FIR SMA{m} (linear phase)")
    ax3.semilogx(p_bw,  ph_bw,  color="#FF5722", linewidth=1.4,
                 label=f"IIR Butterworth LP (nonlinear)")
    ax3.semilogx(p_ema, ph_ema, color="#4CAF50", linewidth=1.4,
                 label=f"IIR EMA{m} (nonlinear)")
    ax3.axvline(m, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.set_xlabel("Period (days, log scale)")
    ax3.set_ylabel("Phase (degrees)")
    ax3.set_title("Frequency Response — Phase", fontsize=11)
    ax3.set_xlim(4, 500)
    ax3.legend(fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    ticker, description = pick_asset()

    df = load_asset(ticker, DATA_DIR)
    if df is None:
        return

    color = COLORS[list(ASSETS.keys()).index(ticker)]
    price = df["Close"]
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Figure 1: FIR filters ──────────────────────────────────────────────────
    print(f"\n  [1/3] Computing FIR filters for {ticker} ({len(df)} trading days)...")
    sma           = compute_sma(price, MA_ORDER)
    wma           = compute_wma(price, MA_ORDER)
    hp_unweighted = price - sma
    hp_weighted   = price - wma
    fir_bandpass  = compute_bandpass(price, BP_LOW_PERIOD, BP_HIGH_PERIOD, BP_NUMTAPS)

    fig1 = plot_fir_filters(
        price, sma, wma, hp_unweighted, hp_weighted, fir_bandpass,
        ticker, description, color, MA_ORDER, BP_LOW_PERIOD, BP_HIGH_PERIOD,
    )
    path1 = os.path.join(FIGURES_DIR, f"{ticker}_fir_filters.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")

    # ── Figure 2: IIR filters ──────────────────────────────────────────────────
    print(f"  [2/3] Computing IIR filters...")
    ema       = compute_ema(price, MA_ORDER)
    butter_lp = compute_butter_lp(price, MA_ORDER, BUTTER_ORDER)
    butter_hp = compute_butter_hp(price, MA_ORDER, BUTTER_ORDER)
    butter_bp = compute_butter_bp(price, BP_LOW_PERIOD, BP_HIGH_PERIOD, BUTTER_ORDER)
    hp_ema    = price - ema

    fig2 = plot_iir_filters(
        price, ema, butter_lp, hp_ema, butter_hp, butter_bp,
        ticker, description, color, MA_ORDER, BP_LOW_PERIOD, BP_HIGH_PERIOD,
    )
    path2 = os.path.join(FIGURES_DIR, f"{ticker}_iir_filters.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")

    # ── Figure 3: FIR vs IIR comparison ───────────────────────────────────────
    print(f"  [3/3] Computing FIR vs IIR comparison...")

    # Causal implementations to expose real phase lag in time domain
    fir_sma_causal = price.rolling(MA_ORDER).mean()
    Wn_lp = (1.0 / MA_ORDER) / 0.5
    sos   = signal.butter(BUTTER_ORDER, Wn_lp, btype="low", output="sos")
    iir_butter_causal = pd.Series(signal.sosfilt(sos, price.values), index=price.index)

    # Filter coefficient arrays for freqz
    b_sma = np.ones(MA_ORDER) / MA_ORDER
    b_bw, a_bw = signal.butter(BUTTER_ORDER, Wn_lp, btype="low")
    alpha = 2.0 / (MA_ORDER + 1)
    b_ema = np.array([alpha])
    a_ema = np.array([1.0, -(1.0 - alpha)])

    fig3 = plot_fir_vs_iir(
        price, fir_sma_causal, iir_butter_causal, ema,
        b_sma, b_bw, a_bw, b_ema, a_ema,
        ticker, color, MA_ORDER,
    )
    path3 = os.path.join(FIGURES_DIR, f"{ticker}_fir_vs_iir.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path3}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
