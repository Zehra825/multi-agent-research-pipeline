"""
Deterministic data-layer functions.

These are plain Python functions with no LLM involved — they do the actual
work of pulling market data and computing statistics. They're used two ways:

1. Directly, by CollectorAgent, to assemble the initial research context.
2. As "tools" the AnalystAgent (an LLM) can call during its reasoning loop —
   see agents/analyst.py for how these get wrapped into Claude tool schemas.

Keeping this layer LLM-free makes it independently testable (see tests/) and
keeps the agents' job purely about reasoning over data, not fetching it.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
import yfinance as yf

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt


def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV daily price history for `ticker` over `period` (e.g. '1mo', '6mo', '1y')."""
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check the symbol.")
    return df


def price_stats(df: pd.DataFrame) -> dict:
    """Summary stats over a price history DataFrame: start/end price, % change, range, volume."""
    start_price = float(df["Close"].iloc[0])
    end_price = float(df["Close"].iloc[-1])
    pct_change = (end_price - start_price) / start_price * 100
    return {
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "pct_change": round(pct_change, 2),
        "period_high": round(float(df["High"].max()), 2),
        "period_low": round(float(df["Low"].min()), 2),
        "avg_volume": int(df["Volume"].mean()),
    }


def moving_averages(df: pd.DataFrame, windows: list[int] | None = None) -> dict:
    """Latest simple moving averages for each window (in trading days)."""
    windows = windows or [20, 50]
    result = {}
    for w in windows:
        if len(df) >= w:
            result[f"sma_{w}"] = round(float(df["Close"].rolling(w).mean().iloc[-1]), 2)
        else:
            result[f"sma_{w}"] = None  # not enough history for this window
    return result


def volatility(df: pd.DataFrame) -> dict:
    """Daily return std dev and its annualized equivalent (252 trading days)."""
    daily_returns = df["Close"].pct_change().dropna()
    daily_std = float(daily_returns.std())
    return {
        "daily_std_pct": round(daily_std * 100, 3),
        "annualized_volatility_pct": round(daily_std * (252 ** 0.5) * 100, 2),
    }


def recent_news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent headlines for `ticker`. Handles both old and new yfinance news payload shapes."""
    raw_items = yf.Ticker(ticker).news or []
    parsed = []
    for item in raw_items[:limit]:
        content = item.get("content", item)  # newer yfinance nests under "content"
        title = content.get("title")
        if not title:
            continue

        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher")
        )

        url = None
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url")
        url = url or item.get("link")

        published = content.get("pubDate") or content.get("displayTime")
        if not published and item.get("providerPublishTime"):
            published = _dt.datetime.utcfromtimestamp(
                item["providerPublishTime"]
            ).isoformat() + "Z"

        parsed.append(
            {
                "title": title,
                "publisher": publisher or "Unknown",
                "published": published or "Unknown",
                "url": url or "",
            }
        )
    return parsed


def save_price_chart(
    df: pd.DataFrame, ticker: str, out_path: str, ma_windows: list[int] | None = None
) -> str:
    """Render a price + moving-average chart to `out_path` (PNG). Returns the path."""
    ma_windows = ma_windows or [20, 50]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Close"], label="Close", color="#1f77b4", linewidth=1.5)
    for w in ma_windows:
        if len(df) >= w:
            ax.plot(
                df.index,
                df["Close"].rolling(w).mean(),
                label=f"SMA {w}",
                linewidth=1,
                alpha=0.8,
            )
    ax.set_title(f"{ticker} — Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
