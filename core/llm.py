"""
Thin wrapper around the Anthropic SDK shared by all agents.

Two entry points:
  - one_shot(...)        a single system+user call, for agents that just
                          generate text/JSON from a fully-assembled context
                          (WriterAgent, CriticAgent).
  - run_agentic_loop(...) a full tool-use loop: the model can call tools
                          repeatedly, inspecting results before deciding
                          what to do next, until it produces a final answer
                          with no further tool calls (AnalystAgent).
"""

from __future__ import annotations

import json
import os
import re

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model's text response."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))
        raise


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=api_key)


def one_shot(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def run_agentic_loop(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    tools: list[dict],
    tool_impls: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    max_turns: int = 6,
    on_tool_call=None,
) -> str:
    """
    Run a standard Claude tool-use loop until the model stops requesting tools.

    `tools` is a list of Claude tool schemas (name/description/input_schema).
    `tool_impls` maps tool name -> python callable(**input) -> JSON-serializable result.
    `on_tool_call(name, input, result)` is an optional callback for logging/tracing.

    Returns the model's final text response.
    """
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            impl = tool_impls.get(block.name)
            if impl is None:
                result_content = f"Error: no tool named '{block.name}'"
            else:
                try:
                    result = impl(**block.input)
                    result_content = json.dumps(result, default=str)
                except Exception as exc:  # tool errors get fed back to the model, not raised
                    result_content = f"Error running tool '{block.name}': {exc}"

            if on_tool_call:
                on_tool_call(block.name, block.input, result_content)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent did not converge to a final answer within {max_turns} turns.")
