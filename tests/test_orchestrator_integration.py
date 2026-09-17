"""
Integration test for the full Collector -> Analyst -> Writer -> Critic flow,
with a fake Anthropic client standing in for the real API and yfinance
network calls mocked out with synthetic data.

This exists to catch plumbing bugs (message formatting in the tool-use loop,
JSON extraction, the writer/critic revision loop) without spending real API
credits or needing network access — the kind of thing that's easy to get
subtly wrong in an agentic pipeline and where a live-API test would be slow,
flaky, and non-deterministic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from core.orchestrator import ResearchPipeline


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, tool_input: dict, block_id: str = "tool_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeMessages:
    """Stands in for client.messages — returns canned responses in call order."""

    def __init__(self, canned_responses: list):
        self._responses = list(canned_responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, canned_responses: list):
        self.messages = FakeMessages(canned_responses)


ANALYSIS_JSON = {
    "trend": "uptrend",
    "trend_summary": "Price rose steadily over the period.",
    "volatility_assessment": "Moderate volatility, no sharp swings.",
    "key_news_themes": ["product launch", "analyst upgrade"],
    "risks": ["macro headwinds"],
    "opportunities": ["upcoming earnings beat potential"],
    "overall_summary": "Overall a steady uptrend supported by positive news flow.",
}

DRAFT_REPORT = (
    "# TEST Research Report\n\n"
    "| Metric | Value |\n|---|---|\n| % Change | +9.09% |\n\n"
    "![Price Chart](TEST_chart.png)\n\n"
    "## Trend\nSteady uptrend.\n\n"
    "*This is an automated demo report, not investment advice.*\n"
)


def make_price_df():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    closes = [100, 101, 103, 102, 105, 107, 106, 108, 109, 110]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


def test_full_pipeline_happy_path(tmp_path):
    canned = [
        # Analyst turn 1: requests a tool
        response("tool_use", [tool_use_block("get_price_summary", {})]),
        # Analyst turn 2: final structured answer
        response("end_turn", [text_block(json.dumps(ANALYSIS_JSON))]),
        # Writer: drafts the report
        response("end_turn", [text_block(DRAFT_REPORT)]),
        # Critic: approves on first pass
        response("end_turn", [text_block(json.dumps({"approved": True, "notes": ""}))]),
    ]
    fake_client = FakeClient(canned)

    with patch("core.tools.yf.Ticker") as mock_ticker, patch(
        "core.orchestrator.get_client", return_value=fake_client
    ):
        mock_ticker.return_value.history.return_value = make_price_df()
        mock_ticker.return_value.news = [{"content": {"title": "Test headline"}}]

        pipeline = ResearchPipeline(verbose=False)
        result = pipeline.run("TEST", output_dir=str(tmp_path))

    assert result["analysis"]["trend"] == "uptrend"
    assert result["review"]["approved"] is True
    assert "TEST Research Report" in result["report_markdown"]
    assert (tmp_path / "TEST_report.md").exists()
    assert (tmp_path / "TEST_chart.png").exists()
    # 4 calls: 2 analyst turns + 1 writer + 1 critic, no revision needed
    assert len(fake_client.messages.calls) == 4


def test_pipeline_triggers_revision_when_critic_rejects(tmp_path):
    canned = [
        response("end_turn", [text_block(json.dumps(ANALYSIS_JSON))]),  # analyst, no tools used
        response("end_turn", [text_block(DRAFT_REPORT)]),  # writer draft
        response(
            "end_turn",
            [text_block(json.dumps({"approved": False, "notes": "Fix the % change figure."}))],
        ),  # critic rejects
        response("end_turn", [text_block(DRAFT_REPORT.replace("+9.09%", "+10.00%"))]),  # revision
    ]
    fake_client = FakeClient(canned)

    with patch("core.tools.yf.Ticker") as mock_ticker, patch(
        "core.orchestrator.get_client", return_value=fake_client
    ):
        mock_ticker.return_value.history.return_value = make_price_df()
        mock_ticker.return_value.news = []

        pipeline = ResearchPipeline(verbose=False)
        result = pipeline.run("TEST", output_dir=str(tmp_path))

    assert result["review"]["approved"] is False
    assert "+10.00%" in result["report_markdown"]
    assert len(fake_client.messages.calls) == 4
