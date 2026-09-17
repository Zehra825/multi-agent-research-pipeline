"""
WriterAgent — turns the Analyst's structured findings into a polished,
human-readable Markdown report. One-shot generation (no tools needed here —
by this point all the investigation is done; the job is pure synthesis).
"""

from __future__ import annotations

import json

from core.llm import one_shot

SYSTEM_PROMPT = """You are a financial report writer producing a research note
for a general reader. You'll be given raw stats and a research analyst's
structured findings for one stock. Write a clean, well-organized Markdown
report using them.

Requirements:
- Start with a level-1 heading: the ticker and "Research Report".
- Include a short stats table (start/end price, % change, high/low, volume).
- Reference the chart using this exact Markdown image syntax: ![Price Chart]({chart_filename})
- Cover trend, volatility, news themes, risks, and opportunities as clear
  sections, in your own words but faithful to the analyst's findings —
  do not invent numbers or facts that aren't in the provided data.
- End with a one-line italic disclaimer that this is an automated demo
  report, not investment advice.
- Output ONLY the Markdown document, no preamble or commentary.
"""


class WriterAgent:
    def run(self, client, context: dict, analysis: dict, chart_filename: str) -> str:
        system = SYSTEM_PROMPT.format(chart_filename=chart_filename)
        user_message = (
            f"TICKER: {context['ticker']}\n\n"
            f"RAW STATS:\n{json.dumps(context['stats'], indent=2)}\n\n"
            f"MOVING AVERAGES:\n{json.dumps(context['moving_averages'], indent=2)}\n\n"
            f"VOLATILITY:\n{json.dumps(context['volatility'], indent=2)}\n\n"
            f"ANALYST FINDINGS:\n{json.dumps(analysis, indent=2)}\n"
        )
        return one_shot(client, system, user_message)

    def revise(self, client, previous_report: str, revision_notes: str) -> str:
        system = (
            "You are a financial report writer revising a previous Markdown draft "
            "based on editor feedback. Keep the same structure, sections, and the "
            "existing chart image reference untouched; fix only what the feedback "
            "calls out. Output ONLY the revised Markdown document."
        )
        user_message = (
            f"PREVIOUS DRAFT:\n{previous_report}\n\n"
            f"EDITOR FEEDBACK TO ADDRESS:\n{revision_notes}\n"
        )
        return one_shot(client, system, user_message)
