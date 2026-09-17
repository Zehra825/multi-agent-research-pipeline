"""
ResearchPipeline — coordinates Collector -> Analyst -> Writer -> Critic.

The orchestrator owns the shared context object and the control flow,
including the one-shot revision loop between Writer and Critic. It doesn't
do any reasoning itself; it's purely plumbing, which is what keeps each
agent easy to test and reason about in isolation.
"""

from __future__ import annotations

from pathlib import Path

from agents.analyst import AnalystAgent
from agents.collector import CollectorAgent
from agents.critic import CriticAgent
from agents.writer import WriterAgent
from core import tools as data_tools
from core.llm import get_client


class ResearchPipeline:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.client = get_client()
        self.collector = CollectorAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(self, ticker: str, output_dir: str = "output", period: str = "6mo") -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"\n[1/4] Collector — fetching {ticker} price history & news ({period})...")
        context = self.collector.run(ticker, period=period)
        self._log(
            f"      {context['stats']['start_date']} -> {context['stats']['end_date']}, "
            f"{context['stats']['pct_change']:+.2f}%, {len(context['news'])} news items"
        )

        chart_filename = f"{context['ticker']}_chart.png"
        chart_path = data_tools.save_price_chart(
            context["price_df"], context["ticker"], str(out_dir / chart_filename)
        )
        self._log(f"      chart saved -> {chart_path}")

        self._log(f"\n[2/4] Analyst — investigating with tool-use loop...")

        def _trace_tool_call(name, tool_input, result):
            arg_str = ", ".join(f"{k}={v}" for k, v in tool_input.items())
            self._log(f"      -> called {name}({arg_str})")

        analysis = self.analyst.run(self.client, context, on_tool_call=_trace_tool_call)
        self._log(f"      trend: {analysis.get('trend')}")

        self._log(f"\n[3/4] Writer — drafting report...")
        draft = self.writer.run(self.client, context, analysis, chart_filename)

        self._log(f"\n[4/4] Critic — reviewing draft against source data...")
        review = self.critic.run(self.client, context, analysis, draft)

        if review.get("approved"):
            self._log("      approved on first pass")
            final_report = draft
        else:
            self._log(f"      revision requested: {review.get('notes')}")
            final_report = self.writer.revise(self.client, draft, review["notes"])
            self._log("      revised draft written")

        report_path = out_dir / f"{context['ticker']}_report.md"
        report_path.write_text(final_report)
        self._log(f"\nDone. Report -> {report_path}")

        return {
            "context": context,
            "analysis": analysis,
            "review": review,
            "report_path": str(report_path),
            "chart_path": chart_path,
            "report_markdown": final_report,
        }
