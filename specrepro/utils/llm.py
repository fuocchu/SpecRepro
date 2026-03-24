"""
LLM query utilities for SpecRepro.
Supports: Claude (Anthropic) and OpenAI.
Default backbone: claude-sonnet-4-6 (best quality for spec extraction).
"""

import os
import json
import time
from typing import Optional


# ── Cost tracking ─────────────────────────────────────────────────────────────

_TOKENS_IN: dict[str, int] = {}
_TOKENS_OUT: dict[str, int] = {}

_COST_IN = {
    "claude-sonnet-4-6":   3.00 / 1_000_000,
    "claude-haiku-4-5":    0.80 / 1_000_000,
    "gpt-4o":              2.50 / 1_000_000,
    "gpt-4o-mini":         0.15 / 1_000_000,
}
_COST_OUT = {
    "claude-sonnet-4-6":  15.00 / 1_000_000,
    "claude-haiku-4-5":    4.00 / 1_000_000,
    "gpt-4o":             10.00 / 1_000_000,
    "gpt-4o-mini":         0.60 / 1_000_000,
}


def current_cost() -> float:
    total = 0.0
    for m, n in _TOKENS_IN.items():
        total += _COST_IN.get(m, 0) * n
    for m, n in _TOKENS_OUT.items():
        total += _COST_OUT.get(m, 0) * n
    return total


# ── Main query function ───────────────────────────────────────────────────────

def query_llm(
    prompt: str,
    system_prompt: str = "You are a helpful AI research assistant.",
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.2,
    max_tokens: int = 4096,
    retries: int = 3,
    retry_delay: float = 5.0,
    print_cost: bool = True,
) -> str:
    """
    Send a prompt to the specified LLM and return the response text.

    Supports:
      - Claude models (claude-*): uses ANTHROPIC_API_KEY
      - OpenAI models (gpt-*):    uses OPENAI_API_KEY
    """
    for attempt in range(retries):
        try:
            if model.startswith("claude"):
                response = _query_claude(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
                response = _query_openai(prompt, system_prompt, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown model: {model}. Use a claude-* or gpt-* model.")

            # Approximate token tracking
            in_tokens  = len(prompt.split()) + len(system_prompt.split())
            out_tokens = len(response.split())
            _TOKENS_IN[model]  = _TOKENS_IN.get(model, 0)  + in_tokens
            _TOKENS_OUT[model] = _TOKENS_OUT.get(model, 0) + out_tokens

            if print_cost:
                print(f"  [LLM cost so far: ${current_cost():.4f}]")

            return response

        except Exception as e:
            if attempt < retries - 1:
                print(f"  [LLM error: {e}. Retrying in {retry_delay}s…]")
                time.sleep(retry_delay)
            else:
                raise


def _query_claude(prompt: str, system: str, model: str, temp: float, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install the anthropic package: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature_for_claude(temp, model),
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def temperature_for_claude(temp: float, model: str) -> float:
    """Extended thinking models don't accept temperature."""
    return temp


def _query_openai(prompt: str, system: str, model: str, temp: float, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the openai package: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    # o1/o3 models do not support temperature
    if not (model.startswith("o1") or model.startswith("o3")):
        kwargs["temperature"] = temp

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content
