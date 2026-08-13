"""
backtest.py — Out-of-sample evaluation of the Kalman-2D trend signal.

Every other module in this project answers "what does the signal look like
after filtering." This one asks the question a quant reader will ask next:
does the filtered signal carry any exploitable information about the *next*
period's return, once you're honest about train/test separation, causality,
and transaction costs?

Design, to keep this honest:
  - Q/R for KalmanFilter2D are derived from TRAIN-period data only
    (2020-01-02 -> 2022-12-31). The evaluation window never touches
    parameter selection.
  - The filter itself runs causally and continuously across the *entire*
    series, exactly as a live system would — nothing resets at the test
    boundary. Only trades occurring in the TEST window count toward
    performance, so the pre-2023 portion acts as legitimate warm-up.
  - Position on day t is decided using the trend estimate as of day t-1
    (i.e. no same-day information), then applied to day t's realized return.
  - A flat 5 bps cost is charged on every position change.

This is not a trading strategy and isn't presented as one — see the
printed caveats in main(). The point is methodological rigor, not alpha.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ASSETS, COLORS, DATA_DIR, FIGURES_DIR, load_asset
from kalman_filter import KalmanFilter2D

TRAIN_END  = "2022-12-31"   # Q/R estimated only on data up to and including this date
TEST_START = "2023-01-01"   # performance is measured only from this date onward
COST_BPS   = 5.0            # flat round-trip-agnostic cost per position change, in bps


# ── Signal construction ──────────────────────────────────────────────────────

def fit_kalman_qr(train_price: pd.Series) -> tuple[np.ndarray, float]:
    """Derive Q (2x2) and R (scalar) from train-period price differences only."""
    dvar = float(np.var(np.diff(train_price.values)))
    Q = np.diag([dvar * 0.5, dvar * 1e-4])
    R = dvar * 100.0
    return Q, R


def compute_trend_signal(price: pd.Series, Q: np.ndarray, R: float) -> pd.Series:
    """
    Run KalmanFilter2D causally over the full series (using Q/R fit on train
    data) and return the trend (velocity) estimate at every day.
    """
    kf = KalmanFilter2D(Q=Q, R=R, x0=np.array([price.iloc[0], 0.0]))
    _, trends, _ = kf.filter(price.values)
    return pd.Series(trends, index=price.index)


# ── Backtest mechanics ─────────────────────────────────────────────────────────

def run_backtest(price: pd.Series, trend: pd.Series, test_start: str,
                 cost_bps: float) -> pd.DataFrame:
    """
    Long/flat rule: hold the asset when yesterday's trend estimate was
    positive, otherwise hold cash. No shorting, no leverage.

    Returns a DataFrame (test window only) with columns:
      asset_return, position, strategy_return, strategy_equity, bh_equity
    """
    asset_return = price.pct_change()

    # Position for day t uses the trend estimate known at the close of t-1 —
    # shift(1) is what prevents same-day lookahead.
    position = (trend.shift(1) > 0).astype(float)

    trade = position.diff().abs().fillna(0.0)
    cost  = trade * (cost_bps / 1e4)

    strategy_return = position * asset_return - cost

    df = pd.DataFrame({
        "asset_return": asset_return,
        "position": position,
        "trade": trade,
        "strategy_return": strategy_return,
    })
    df = df.loc[test_start:].copy()
    df = df.iloc[1:]  # drop the first row of the test window (pct_change NaN edge)

    df["strategy_equity"] = (1.0 + df["strategy_return"]).cumprod()
    df["bh_equity"]       = (1.0 + df["asset_return"]).cumprod()
    return df


# ── Performance metrics ────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio, assuming zero risk-free rate."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from an equity curve starting at 1.0."""
    n_years = len(equity) / periods_per_year
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline in the equity curve (negative number)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def hit_rate(returns: pd.Series, position: pd.Series) -> float:
    """Fraction of in-position days with a positive strategy return."""
    in_position = returns[position > 0]
    if len(in_position) == 0:
        return 0.0
    return float((in_position > 0).mean())


def summarize_backtest(label: str, df: pd.DataFrame) -> dict:
    return {
        "Asset":               label,
        "Strategy Sharpe":     round(sharpe_ratio(df["strategy_return"]), 2),
        "Buy&Hold Sharpe":     round(sharpe_ratio(df["asset_return"]), 2),
        "Strategy CAGR":       round(cagr(df["strategy_equity"]), 4),
        "Buy&Hold CAGR":       round(cagr(df["bh_equity"]), 4),
        "Strategy MaxDD":      round(max_drawdown(df["strategy_equity"]), 4),
        "Buy&Hold MaxDD":      round(max_drawdown(df["bh_equity"]), 4),
        "Hit Rate (in-pos.)":  round(hit_rate(df["strategy_return"], df["position"]), 4),
        "Trades":              int(df["trade"].sum()),
        "Test Days":           len(df),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_equity_curve(df: pd.DataFrame, ticker: str, description: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["strategy_equity"], color=color, linewidth=1.5,
            label="Kalman trend-sign strategy (net of costs)")
    ax.plot(df.index, df["bh_equity"], color="#9E9E9E", linewidth=1.3,
            linestyle="--", label="Buy & hold")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(
        f"{ticker} — {description}  |  Out-of-Sample Backtest ({TEST_START} onward)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sharpe_comparison(results: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, results["Strategy Sharpe"], width, label="Strategy", color="#2196F3")
    ax.bar(x + width/2, results["Buy&Hold Sharpe"], width, label="Buy & Hold", color="#9E9E9E")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(results["Asset"])
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title(
        f"Strategy vs. Buy & Hold — Out-of-Sample Sharpe ({TEST_START} onward)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = []

    print(f"Train (Q/R fit):  start -> {TRAIN_END}")
    print(f"Test  (evaluated): {TEST_START} -> end")
    print(f"Transaction cost:  {COST_BPS} bps per position change\n")

    for i, (ticker, description) in enumerate(ASSETS.items()):
        df_raw = load_asset(ticker, DATA_DIR)
        if df_raw is None:
            continue
        price = df_raw["Close"]

        train_price = price.loc[:TRAIN_END]
        Q, R = fit_kalman_qr(train_price)
        trend = compute_trend_signal(price, Q, R)

        bt = run_backtest(price, trend, TEST_START, COST_BPS)
        row = summarize_backtest(ticker, bt)
        rows.append(row)
        print(f"  [{ticker}] {row}")

        if ticker == "SPY":
            fig = plot_equity_curve(bt, ticker, description, COLORS[i])
            path = os.path.join(FIGURES_DIR, "backtest_spy_equity.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    results = pd.DataFrame(rows)
    print("\n=== Summary (all assets, out-of-sample) ===")
    print(results.to_string(index=False))

    fig2 = plot_sharpe_comparison(results)
    path2 = os.path.join(FIGURES_DIR, "backtest_sharpe_comparison.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path2}")

    plt.show()
    print("\nDone. This is an out-of-sample signal check, not a validated trading")
    print("strategy: single train/test split, single hyperparameter choice, no")
    print("walk-forward re-fitting, no slippage beyond a flat cost assumption.")


if __name__ == "__main__":
    main()
