"""
CollectorAgent — deterministic data-ingestion stage.

Not LLM-backed on purpose: its job is to reliably assemble a clean, typed
context object (price history, computed stats, news) for the reasoning
agents downstream. Mixing an LLM into pure data-fetching would add cost and
non-determinism for no benefit — this is the pipeline's "ETL" stage.
"""

from __future__ import annotations

from core import tools


class CollectorAgent:
    def run(self, ticker: str, period: str = "6mo", news_limit: int = 8) -> dict:
        df = tools.fetch_price_history(ticker, period=period)
        return {
            "ticker": ticker.upper(),
            "period": period,
            "price_df": df,
            "stats": tools.price_stats(df),
            "moving_averages": tools.moving_averages(df),
            "volatility": tools.volatility(df),
            "news": tools.recent_news(ticker, limit=news_limit),
        }
