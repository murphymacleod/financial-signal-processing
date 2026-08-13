"""
plot_data.py — Visualization layer for the Financial Signal Processing Platform.
Loads saved CSVs and produces clear price charts. No analysis is performed here.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

ASSETS = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq-100 ETF",
    "GLD":  "SPDR Gold Shares",
    "USO":  "US Oil Fund",
    "CPER": "US Copper Index Fund",
}

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLORS      = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ── Load ───────────────────────────────────────────────────────────────────────

def load_all(data_dir: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load a CSV for each ticker. Missing files are skipped with a warning."""
    data = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            print(f"  [WARN] {ticker}: file not found at {path}, skipping.")
            continue
        # Newer yfinance writes a 3-row header (Price / Ticker / Date);
        # peek at the second line to detect and skip those metadata rows.
        with open(path) as f:
            f.readline()
            second_line = f.readline()
        skip = [1, 2] if second_line.strip().startswith("Ticker") else []
        df = pd.read_csv(path, skiprows=skip, index_col=0,
                         parse_dates=True, date_format="%Y-%m-%d")
        data[ticker] = df
    return data


# ── Plot Functions ─────────────────────────────────────────────────────────────

def _apply_date_axis(ax: plt.Axes) -> None:
    """Apply consistent year-tick formatting to an x-axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)


def plot_normalized_prices(data: dict, colors: list) -> plt.Figure:
    """
    Figure 1 — All assets on one panel, normalized to 100 at their first data point.
    Makes relative performance and correlation visible across different price scales.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for (ticker, df), color in zip(data.items(), colors):
        normalized = df["Close"] / df["Close"].iloc[0] * 100
        ax.plot(df.index, normalized, label=ASSETS[ticker], color=color, linewidth=1.5)

    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Normalized Close Price Performance (Base = 100 at 2020-01-02)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Index (Base = 100)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    _apply_date_axis(ax)
    fig.tight_layout()
    return fig


def plot_individual_prices(data: dict, colors: list) -> plt.Figure:
    """
    Figure 2 — 2-column × 3-row grid; one panel per asset (5 panels, bottom-right hidden).
    Shows absolute price history for each asset independently.
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 10), squeeze=False)

    for i, ((ticker, df), color) in enumerate(zip(data.items(), colors)):
        row, col = divmod(i, 2)
        ax = axes[row][col]
        ax.plot(df.index, df["Close"], color=color, linewidth=1.2)
        ax.set_title(f"{ticker} — {ASSETS[ticker]}", fontsize=11)
        ax.set_ylabel("Close Price (USD)")
        ax.grid(True, alpha=0.3)
        _apply_date_axis(ax)

    # Hide the unused 6th panel
    axes[2][1].set_visible(False)

    fig.suptitle("Individual Close Prices (Daily, 2020 – Present)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    data = load_all(DATA_DIR, list(ASSETS.keys()))
    if not data:
        print("No data found. Run download_data.py first.")
        return

    print(f"Loaded {len(data)} assets. Generating plots...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig1 = plot_normalized_prices(data, COLORS)
    path1 = os.path.join(FIGURES_DIR, "normalized_prices.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path1}")

    fig2 = plot_individual_prices(data, COLORS)
    path2 = os.path.join(FIGURES_DIR, "individual_prices.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
