"""Tests for common.py — asset universe, CSV loading, and CLI ticker resolution."""

import pytest

import common


def test_resolve_ticker_by_symbol_case_insensitive():
    ticker, desc = common.resolve_ticker("spy")
    assert ticker == "SPY"
    assert desc == common.ASSETS["SPY"]


def test_resolve_ticker_by_index():
    tickers = list(common.ASSETS)
    ticker, _ = common.resolve_ticker("3")
    assert ticker == tickers[2]


def test_resolve_ticker_invalid_symbol_exits():
    with pytest.raises(SystemExit):
        common.resolve_ticker("NOTATICKER")


def test_resolve_ticker_invalid_index_exits():
    with pytest.raises(SystemExit):
        common.resolve_ticker("99")


def test_resolve_ticker_none_falls_back_to_interactive_picker(monkeypatch):
    monkeypatch.setattr(common, "pick_asset", lambda: ("SPY", common.ASSETS["SPY"]))
    ticker, _ = common.resolve_ticker(None)
    assert ticker == "SPY"


def test_load_asset_reads_real_spy_csv():
    df = common.load_asset("SPY", common.DATA_DIR)
    assert df is not None
    assert "Close" in df.columns
    assert df.index.is_monotonic_increasing
    assert len(df) > 1000


def test_load_asset_missing_file_returns_none(tmp_path):
    assert common.load_asset("NOPE", str(tmp_path)) is None


def test_assets_dict_has_five_entries_with_descriptions():
    assert len(common.ASSETS) == 5
    assert all(isinstance(v, str) and v for v in common.ASSETS.values())


def test_ticker_arg_parser_accepts_optional_positional():
    parser = common.ticker_arg_parser("test")
    args = parser.parse_args([])
    assert args.ticker is None
    args = parser.parse_args(["SPY"])
    assert args.ticker == "SPY"
