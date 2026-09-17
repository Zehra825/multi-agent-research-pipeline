"""
AnalystAgent — the agentic reasoning stage.

Unlike CollectorAgent, this one is a real tool-use loop: Claude is handed a
small toolbox (price summary, custom moving averages, volatility, news,
raw closes) bound to the already-collected data, and decides for itself
which tools to call and in what order before producing a structured
verdict. This is the piece of the pipeline that demonstrates agentic
behavior rather than a fixed script — the model can, e.g., pull a 10-day
moving average instead of the default 20/50 if the price action looks like
it needs a closer look.
"""

from __future__ import annotations

from core import tools as data_tools
from core.llm import extract_json, run_agentic_loop

SYSTEM_PROMPT = """You are a senior equity research analyst agent.

You have tools to inspect a stock's price history, volatility, moving
averages, and recent news. Investigate thoroughly — call as many tools as
you need, in whatever order makes sense — before forming a conclusion.
Don't just accept the default stats; if the trend looks noisy, pull a
shorter or longer moving-average window to check.

Once you're confident in your analysis, respond with ONLY a single JSON
object (no markdown fences, no commentary) with exactly these fields:

{
  "trend": "uptrend" | "downtrend" | "sideways",
  "trend_summary": "1-2 sentences on the price trend, citing specific numbers",
  "volatility_assessment": "1-2 sentences on how volatile/stable the stock has been",
  "key_news_themes": ["short theme 1", "short theme 2", ...],
  "risks": ["risk 1", "risk 2", ...],
  "opportunities": ["opportunity 1", "opportunity 2", ...],
  "overall_summary": "3-4 sentence synthesis for a report reader"
}
"""

TOOL_SCHEMAS = [
    {
        "name": "get_price_summary",
        "description": "Get start/end price, % change, period high/low, and average volume.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_moving_averages",
        "description": "Compute simple moving averages for the given windows (in trading days).",
        "input_schema": {
            "type": "object",
            "properties": {
                "windows": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "e.g. [10, 20, 50]",
                }
            },
            "required": ["windows"],
        },
    },
    {
        "name": "get_volatility",
        "description": "Get daily return standard deviation and annualized volatility.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_news",
        "description": "Get recent news headlines for the ticker with publisher and date.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_recent_closes",
        "description": "Get the last N daily closing prices with dates, for a closer look at recent action.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number of most recent trading days"}},
            "required": ["n"],
        },
    },
]


class AnalystAgent:
    def run(self, client, context: dict, on_tool_call=None) -> dict:
        df = context["price_df"]
        news = context["news"]

        tool_impls = {
            "get_price_summary": lambda: data_tools.price_stats(df),
            "get_moving_averages": lambda windows: data_tools.moving_averages(df, windows),
            "get_volatility": lambda: data_tools.volatility(df),
            "get_news": lambda: news,
            "get_recent_closes": lambda n: [
                {"date": str(idx.date()), "close": round(float(val), 2)}
                for idx, val in df["Close"].tail(n).items()
            ],
        }

        user_message = (
            f"Analyze {context['ticker']} using your tools. "
            f"Price history covers {context['stats']['start_date']} to "
            f"{context['stats']['end_date']}."
        )

        final_text = run_agentic_loop(
            client=client,
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tools=TOOL_SCHEMAS,
            tool_impls=tool_impls,
            on_tool_call=on_tool_call,
        )
        return extract_json(final_text)
