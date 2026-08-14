"""Tests for download_data.py's pure validation logic. No network calls —
download_asset() itself (the yfinance call) is intentionally not exercised here."""

import pandas as pd

from download_data import validate


def test_validate_accepts_well_formed_dataframe():
    idx = pd.date_range("2020-01-01", periods=10)
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    assert validate(df, "TEST") is True


def test_validate_rejects_empty_dataframe():
    assert validate(pd.DataFrame(), "TEST") is False


def test_validate_rejects_non_monotonic_dates():
    idx = pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03"])
    df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
    assert validate(df, "TEST") is False


def test_validate_warns_but_still_passes_on_high_nan_pct(capsys):
    idx = pd.date_range("2020-01-01", periods=20)
    df = pd.DataFrame({"Close": [1.0] * 10 + [None] * 10}, index=idx)
    assert validate(df, "TEST") is True
    assert "WARN" in capsys.readouterr().out
