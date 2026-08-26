"""Per-call cost tracking for chat-completion calls.

Appends one JSONL row to `<run_dir>/costs.jsonl` per successful API call so
the spend of an inline run can be reconstructed without re-reading every
request log. Silently no-ops if the run dir or usage block is missing, or if
the model has no known pricing (e.g. the default local vLLM model), so
unrelated providers / paths are not perturbed with no-signal per-call writes.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


# A trailing OpenAI date-snapshot suffix, e.g. "-2026-05-01".
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


# USD per 1M tokens. Keep in sync with provider pricing pages.
# (input, cached input, output).
_PRICING_PER_M: dict[str, tuple[float, float, float]] = {
    # OpenAI direct, May 2026 pricing.
    "gpt-5.5":      (5.00,  0.50,  30.00),
    "gpt-5.4":      (2.50,  0.25,  15.00),
    "gpt-5.2":      (1.75,  0.175, 14.00),
    "gpt-5.1":      (1.25,  0.125, 10.00),
    "gpt-5":        (1.25,  0.125, 10.00),
    "gpt-5-mini":   (0.25,  0.025, 2.00),
    "gpt-5-nano":   (0.05,  0.005, 0.40),
}


def _pricing_for(model: str) -> tuple[float, float, float] | None:
    if not model:
        return None
    base = str(model).split(":")[0].strip().lower()
    if base in _PRICING_PER_M:
        return _PRICING_PER_M[base]
    # Only fold dated snapshots (e.g. "gpt-5.4-2026-05-01") onto their base
    # rate. Named variants like "-mini"/"-pro" have their own pricing, so
    # return None rather than silently charging the parent rate.
    for key in sorted(_PRICING_PER_M, key=len, reverse=True):
        if base.startswith(key) and _SNAPSHOT_SUFFIX.match(base[len(key):]):
            return _PRICING_PER_M[key]
    return None


def _compute_cost(usage: dict[str, Any], model: str) -> tuple[float | None, int, int, int]:
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    p = _pricing_for(model)
    if p is None:
        return None, inp, cached, out
    fresh = max(0, inp - cached)
    in_rate, cache_rate, out_rate = p
    cost = (fresh * in_rate + cached * cache_rate + out * out_rate) / 1_000_000
    return cost, inp, cached, out


def record_call_cost(
    *,
    run_dir: Path | None,
    provider: str | None,
    model: str | None,
    usage: dict[str, Any] | None,
) -> None:
    if run_dir is None or not isinstance(usage, dict) or not model:
        return
    cost, inp, cached, out = _compute_cost(usage, model)
    if cost is None:
        # No known pricing (e.g. local vLLM) — skip the no-signal write.
        return
    row = {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "prompt_tokens": inp,
        "cached_prompt_tokens": cached,
        "completion_tokens": out,
        "cost_usd": cost,
    }
    # Chat Completions exposes reasoning under completion_tokens_details;
    # Responses uses output_tokens_details.
    reasoning = (
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    )
    if reasoning is not None:
        row["reasoning_tokens"] = int(reasoning)
    try:
        path = Path(run_dir) / "costs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        # Never fail the chat call because cost logging hit a filesystem issue.
        pass
