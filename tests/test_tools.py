"""
Unit tests for the deterministic data layer (core/tools.py).

These use synthetic data and mocks — no network access or ANTHROPIC_API_KEY
required, so they run anywhere (including CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core import tools


def make_price_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.02 for c in closes],
            "Low": [c * 0.98 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


def test_price_stats_basic():
    df = make_price_df([100, 105, 110, 108, 120])
    stats = tools.price_stats(df)
    assert stats["start_price"] == 100
    assert stats["end_price"] == 120
    assert stats["pct_change"] == 20.0
    assert stats["period_high"] == pytest.approx(120 * 1.02)
    assert stats["period_low"] == pytest.approx(100 * 0.98)


def test_moving_averages_enough_history():
    df = make_price_df([100, 102, 104, 106, 108])
    mas = tools.moving_averages(df, windows=[3])
    expected = (104 + 106 + 108) / 3
    assert mas["sma_3"] == pytest.approx(round(expected, 2))


def test_moving_averages_insufficient_history_returns_none():
    df = make_price_df([100, 102])
    mas = tools.moving_averages(df, windows=[50])
    assert mas["sma_50"] is None


def test_volatility_zero_for_constant_price():
    df = make_price_df([100, 100, 100, 100])
    vol = tools.volatility(df)
    assert vol["daily_std_pct"] == 0.0
    assert vol["annualized_volatility_pct"] == 0.0


def test_volatility_positive_for_moving_price():
    df = make_price_df([100, 110, 95, 120, 90])
    vol = tools.volatility(df)
    assert vol["daily_std_pct"] > 0


def test_fetch_price_history_raises_on_empty():
    with patch("core.tools.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            tools.fetch_price_history("NOTREAL")


def test_recent_news_parses_new_nested_format():
    raw = [
        {
            "content": {
                "title": "Example headline",
                "pubDate": "2026-09-17T12:00:00Z",
                "provider": {"displayName": "Example Wire"},
                "canonicalUrl": {"url": "https://example.com/story"},
            }
        }
    ]
    with patch("core.tools.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = raw
        news = tools.recent_news("AAPL", limit=5)
    assert news == [
        {
            "title": "Example headline",
            "publisher": "Example Wire",
            "published": "2026-09-17T12:00:00Z",
            "url": "https://example.com/story",
        }
    ]


def test_recent_news_parses_old_flat_format():
    raw = [
        {
            "title": "Old-format headline",
            "publisher": "Old Wire",
            "link": "https://example.com/old",
            "providerPublishTime": 1758110400,
        }
    ]
    with patch("core.tools.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = raw
        news = tools.recent_news("AAPL", limit=5)
    assert news[0]["title"] == "Old-format headline"
    assert news[0]["publisher"] == "Old Wire"
    assert news[0]["url"] == "https://example.com/old"


def test_recent_news_respects_limit():
    raw = [{"content": {"title": f"Headline {i}"}} for i in range(20)]
    with patch("core.tools.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = raw
        news = tools.recent_news("AAPL", limit=3)
    assert len(news) == 3


def test_save_price_chart_writes_file(tmp_path):
    df = make_price_df([100, 102, 104, 106, 108, 110, 112])
    out_path = tmp_path / "chart.png"
    result_path = tools.save_price_chart(df, "TEST", str(out_path), ma_windows=[3])
    assert out_path.exists()
    assert result_path == str(out_path)
    assert out_path.stat().st_size > 0
