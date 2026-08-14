"""Tests for backtest.py — trading metrics and the no-lookahead property.

The no-lookahead test is the important one: it's what makes the backtest an
honest out-of-sample check rather than something fitted in hindsight.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import cagr, hit_rate, max_drawdown, run_backtest, sharpe_ratio


def test_sharpe_ratio_zero_std_returns_zero():
    r = pd.Series([0.01, 0.01, 0.01])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_ratio_positive_for_positive_mean_returns():
    rng = np.random.default_rng(0)
    r = pd.Series(0.001 + rng.standard_normal(500) * 0.01)
    assert sharpe_ratio(r) > 0


def test_cagr_doubling_equity_over_one_year():
    equity = pd.Series(np.linspace(1.0, 2.0, 252))
    assert cagr(equity) == pytest.approx(1.0, rel=0.05)


def test_max_drawdown_is_zero_for_monotonic_growth():
    equity = pd.Series(np.linspace(1.0, 2.0, 100))
    assert max_drawdown(equity) == 0.0


def test_max_drawdown_known_value():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])  # peak 1.2, trough 0.9 -> -25%
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_hit_rate_only_counts_in_position_days():
    returns  = pd.Series([0.01, -0.01, 0.02, -0.02])
    position = pd.Series([1, 1, 0, 0])
    assert hit_rate(returns, position) == 0.5


def test_run_backtest_position_has_no_same_day_lookahead():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    price = pd.Series(np.linspace(100, 110, 10), index=dates)
    # trend flips sign exactly at index 5; if position used same-day trend,
    # that day's position would already reflect the flip
    trend = pd.Series([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], index=dates, dtype=float)

    result = run_backtest(price, trend, test_start="2023-01-01", cost_bps=0.0)

    flip_day = dates[5]
    assert flip_day in result.index
    assert result.loc[flip_day, "position"] == 1.0  # still reflects yesterday's (positive) trend


def test_run_backtest_charges_cost_only_on_position_changes():
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    price = pd.Series([100, 101, 102, 103, 104, 105], index=dates, dtype=float)
    trend = pd.Series([1, 1, 1, -1, -1, -1], index=dates, dtype=float)

    result = run_backtest(price, trend, test_start="2023-01-01", cost_bps=10.0)

    # Two trades: entering the market on the first evaluated day (position
    # starts undefined -> 0 before any trend history exists) and the single
    # trend-sign flip partway through. Every other day, no trade.
    assert result["trade"].sum() == 2
    assert (result["trade"].isin([0.0, 1.0])).all()
