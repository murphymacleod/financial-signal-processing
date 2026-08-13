"""
compute_metrics.py — Metrics layer for the Financial Signal Processing Platform.

Prompts the user to select an asset, computes daily returns and 30-day rolling
annualized volatility, and produces a 3-panel chart with price + moving averages,
returns, and volatility. No data is downloaded here; only saved CSVs are consumed.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ASSETS, COLORS, DATA_DIR, FIGURES_DIR, apply_date_axis, load_asset, pick_asset

# ── Configuration ──────────────────────────────────────────────────────────────

ROLLING_WINDOW = 30

# Moving-average colors: one neutral hue family so they read as
# "same concept at different timescales" without competing with the price line.
MA_COLORS = {
    "MA10":  "#BDBDBD",  # light gray  — short-term
    "MA50":  "#757575",  # mid gray    — medium-term
    "MA200": "#37474F",  # dark blue-gray — long-term
}


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute moving averages, daily returns, and rolling annualized volatility.
    Annualized vol = rolling std of daily returns × sqrt(252 trading days/year).
    """
    metrics = pd.DataFrame(index=df.index)
    metrics["Close"]      = df["Close"]
    metrics["MA10"]       = df["Close"].rolling(10).mean()
    metrics["MA50"]       = df["Close"].rolling(50).mean()
    metrics["MA200"]      = df["Close"].rolling(200).mean()
    metrics["Returns"]    = df["Close"].pct_change()
    metrics["Volatility"] = metrics["Returns"].rolling(window).std() * np.sqrt(252)
    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_metrics(metrics: pd.DataFrame, ticker: str, description: str,
                 color: str, window: int) -> plt.Figure:
    """
    Three stacked panels sharing the x-axis:
      Panel 1 — Close price + MA10 / MA50 / MA200 (same axis, legend)
      Panel 2 — Daily returns (green/red bars)
      Panel 3 — 30-day rolling annualized volatility (filled line)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1.5]},
    )

    # ── Panel 1: Price + Moving Averages ──────────────────────────────────────
    ax1.plot(metrics.index, metrics["Close"],
             color=color, linewidth=1.5, label="Price", zorder=4)
    for ma_label, ma_color in MA_COLORS.items():
        ax1.plot(metrics.index, metrics[ma_label],
                 color=ma_color, linewidth=1.0, linestyle="--",
                 label=ma_label, alpha=0.85, zorder=3)
    ax1.set_title(f"{ticker} — {description}", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Close Price (USD)")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Daily Returns ─────────────────────────────────────────────────
    pos  = metrics["Returns"] >= 0
    ax2.bar(metrics.index[pos],  metrics["Returns"][pos],
            color="#4CAF50", width=1, alpha=0.85)
    ax2.bar(metrics.index[~pos], metrics["Returns"][~pos],
            color="#F44336", width=1, alpha=0.85)
    ax2.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("Daily Return")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Rolling Volatility ────────────────────────────────────────────
    ax3.plot(metrics.index, metrics["Volatility"],
             color="#FF9800", linewidth=1.3)
    ax3.fill_between(metrics.index, metrics["Volatility"],
                     color="#FF9800", alpha=0.15)
    ax3.set_ylabel(f"{window}-Day Vol (Ann.)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.grid(True, alpha=0.3)
    apply_date_axis(ax3)  # bottom panel only; sharex propagates ticks upward

    fig.tight_layout()
    return fig


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    ticker, description = pick_asset()

    df = load_asset(ticker, DATA_DIR)
    if df is None:
        return

    color = COLORS[list(ASSETS.keys()).index(ticker)]
    metrics = compute_metrics(df, window=ROLLING_WINDOW)

    latest_vol = metrics["Volatility"].iloc[-1]
    avg_vol    = metrics["Volatility"].mean()
    print(f"\n  {ticker}  |  {len(df)} trading days  "
          f"|  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Current {ROLLING_WINDOW}-day vol (ann.): {latest_vol:.1%}   "
          f"|   Historical avg: {avg_vol:.1%}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig  = plot_metrics(metrics, ticker, description, color, ROLLING_WINDOW)
    path = os.path.join(FIGURES_DIR, f"{ticker}_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
