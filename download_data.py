"""
download_data.py — Data acquisition layer for the Financial Signal Processing Platform.

Downloads historical OHLCV data for a fixed set of assets from Yahoo Finance and
saves each as a CSV in the local data/ directory. No analysis is performed here.
"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────────

ASSETS = {
    "SPY":  "S&P 500 ETF (Equity Index)",
    "QQQ":  "Nasdaq-100 ETF (Technology)",
    "GLD":  "SPDR Gold Shares (Gold)",
    "USO":  "United States Oil Fund (Oil)",
    "CPER": "United States Copper Index Fund (Copper)",
}

START_DATE = "2020-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
INTERVAL   = "1d"
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Core Functions ─────────────────────────────────────────────────────────────

def download_asset(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    """Download OHLCV data for a single ticker. Returns None if the result is empty."""
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    return df if not df.empty else None


def validate(df: pd.DataFrame, ticker: str) -> bool:
    """Return True if the dataset passes basic quality checks."""
    if df.empty:
        print(f"  [FAIL] {ticker}: empty dataset returned")
        return False

    if not df.index.is_monotonic_increasing:
        print(f"  [FAIL] {ticker}: dates are not in chronological order")
        return False

    nan_pct = df.isnull().mean().max() * 100
    if nan_pct > 5:
        print(f"  [WARN] {ticker}: {nan_pct:.1f}% missing values in at least one column")

    return True


def save(df: pd.DataFrame, ticker: str, out_dir: str) -> str:
    """Save dataframe to CSV and return the file path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}.csv")
    # yfinance >= 0.2 returns a MultiIndex column (Price, Ticker); flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df.to_csv(path)
    return path


def print_summary(ticker: str, description: str, df: pd.DataFrame, path: str) -> None:
    """Print a short download summary for one asset."""
    start = df.index[0].date()
    end   = df.index[-1].date()
    rows  = len(df)
    print(f"  [{ticker}] {description}")
    print(f"    {rows} trading days  |  {start} -> {end}  |  saved to {path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Downloading {len(ASSETS)} assets  [{START_DATE} -> {END_DATE}, interval={INTERVAL}]")
    print("-" * 65)

    success, failed = [], []

    for ticker, description in ASSETS.items():
        df = download_asset(ticker, START_DATE, END_DATE, INTERVAL)

        if df is None or not validate(df, ticker):
            failed.append(ticker)
            continue

        path = save(df, ticker, DATA_DIR)
        print_summary(ticker, description, df, path)
        success.append(ticker)

    print("-" * 65)
    print(f"Complete: {len(success)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()
