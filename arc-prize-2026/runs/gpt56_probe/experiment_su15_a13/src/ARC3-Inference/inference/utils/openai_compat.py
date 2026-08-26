"""Helpers for provider-specific OpenAI-compatible requests."""
from __future__ import annotations

from typing import Any


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    if provider in {"openai-api", "oai"}:
        return "openai-api"
    return provider


def build_headers(
    *,
    provider: str,
    api_key: str,
    referer: str = "",
    title: str = "",
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    normalized = normalize_provider(provider)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if normalized == "openai-api":
        # OpenAI reasoning models (gpt-5.x) reject sampling overrides and
        # use max_completion_tokens; top_k/chat_template_kwargs are vLLM-only.
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens
        if seed is not None and seed >= 0:
            payload["seed"] = seed
    else:
        payload["temperature"] = temperature
        payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    if normalized == "vllm":
        if top_k > 0:
            payload["top_k"] = top_k
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if seed is not None and seed >= 0:
            payload["seed"] = seed

    return payload
