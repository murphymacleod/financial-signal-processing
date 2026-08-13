"""
common.py — Shared configuration and I/O for the Financial Signal Processing Platform.

Every other module in this project selects from the same 5-asset universe and
loads the same CSVs with the same yfinance header quirk; this is the one place
that logic lives instead of five near-identical copies.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

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

def load_asset(ticker: str, data_dir: str = DATA_DIR) -> pd.DataFrame | None:
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


# ── Plotting ───────────────────────────────────────────────────────────────────

def apply_date_axis(ax: plt.Axes) -> None:
    """Year-tick x-axis formatting, matching project-wide style."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
