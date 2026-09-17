"""
CriticAgent — a reflection/review stage that closes the loop.

This is what makes the pipeline "multi-agent" rather than just "multi-step":
the Critic reads the Writer's draft against the Analyst's underlying data
and either approves it or sends it back with specific, actionable feedback.
The orchestrator gives the Writer one chance to revise before finalizing,
which is a cheap way to catch numeric hallucinations or unsupported claims
without a full human-in-the-loop review.
"""

from __future__ import annotations

import json

from core.llm import extract_json, one_shot

SYSTEM_PROMPT = """You are a meticulous editor fact-checking a financial
report draft against the underlying data it's supposed to be based on.

Check for:
- Any number in the draft (price, % change, volatility, etc.) that doesn't
  match the source data.
- Claims not supported by the analyst findings or news provided.
- Missing required sections (stats table, chart reference, trend,
  volatility, news themes, risks, opportunities, disclaimer).

Respond with ONLY a JSON object, no markdown fences:
{
  "approved": true | false,
  "notes": "If not approved, specific actionable feedback on what to fix. Empty string if approved."
}
"""


class CriticAgent:
    def run(self, client, context: dict, analysis: dict, draft: str) -> dict:
        user_message = (
            f"SOURCE STATS:\n{json.dumps(context['stats'], indent=2)}\n\n"
            f"SOURCE ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
            f"DRAFT REPORT:\n{draft}\n"
        )
        response_text = one_shot(client, SYSTEM_PROMPT, user_message)
        return extract_json(response_text)
