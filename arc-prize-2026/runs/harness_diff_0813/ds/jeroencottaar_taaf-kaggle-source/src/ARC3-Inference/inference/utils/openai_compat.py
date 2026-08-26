"""Helpers for provider-specific OpenAI-compatible requests."""
from __future__ import annotations

import json
from typing import Any


# OpenAI reasoning-model families that reject sampling params and need
# max_completion_tokens + extended prompt-cache retention.
_OPENAI_REASONING_PREFIXES: tuple[str, ...] = (
    "gpt-5", "gpt-5.", "o1", "o3", "o4",
)


# Custom key on assistant messages where we stash the raw `output` items
# from a /v1/responses call (reasoning items + message items). When
# build_responses_payload echoes the conversation back, it uses these so
# the model gets its prior chain-of-thought.
RESPONSE_ITEMS_KEY = "_openai_response_items"


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai-compatible", "compat"}:
        return "vllm"
    if provider == "openai":
        return "openai"
    if provider in {"openai-responses", "openai_responses", "responses"}:
        return "openai-responses"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    return provider


def _is_openai_reasoning_model(model: str) -> bool:
    name = str(model or "").lower().lstrip()
    return any(name.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


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
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_provider(provider)

    if normalized == "openai":
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # Reasoning models (gpt-5*, o*) reject temperature/top_p; sampling
        # knobs are replaced by reasoning_effort + verbosity. We leave
        # reasoning_effort unset so the model uses its default (medium for
        # gpt-5.x), which also avoids the chat-completions restriction that
        # forbids reasoning_effort + function tools.
        if _is_openai_reasoning_model(model):
            if max_tokens is not None and max_tokens > 0:
                payload["max_completion_tokens"] = max_tokens
            # gpt-5.5+ rejects the default in_memory cache retention; opt
            # into the extended cache so the duck's long, stable system
            # prompt is reused across turns.
            payload["prompt_cache_retention"] = "24h"
        else:
            if max_tokens is not None and max_tokens > 0:
                payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
            payload["top_p"] = top_p
        if prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        return payload

    # Default OpenAI-compatible payload (also used by openrouter).
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
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


# ---------------------------------------------------------------------------
# Responses API (/v1/responses) support
# ---------------------------------------------------------------------------


def _content_to_response_parts(content: Any, *, is_assistant: bool) -> list[dict[str, Any]] | str:
    """Translate chat-completions message.content into Responses API content parts.

    Chat Completions accepts either a string or a list of `{type, ...}` parts.
    Responses uses `input_text`/`input_image` for user/system and `output_text`
    for assistant turns.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append({"type": "input_text", "text": str(item)})
            continue
        t = item.get("type")
        if t in ("text", "input_text", "output_text"):
            text = str(item.get("text", ""))
            if is_assistant:
                parts.append({"type": "output_text", "text": text})
            else:
                parts.append({"type": "input_text", "text": text})
        elif t in ("image_url", "input_image"):
            # Chat Completions: {"type":"image_url","image_url":{"url":"..."}}
            # Responses:        {"type":"input_image","image_url":"..."}
            url_field = item.get("image_url")
            if isinstance(url_field, dict):
                url = str(url_field.get("url", ""))
            else:
                url = str(url_field or "")
            parts.append({"type": "input_image", "image_url": url})
        else:
            # Pass-through unknown types unchanged.
            parts.append(item)
    return parts


def _tool_calls_to_function_call_items(tool_calls: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function") or {}
        call_id = str(tc.get("id") or "").strip()
        if not call_id:
            continue
        args = function.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args)
        items.append({
            "type": "function_call",
            "call_id": call_id,
            "name": str(function.get("name") or ""),
            "arguments": args,
        })
    return items


def messages_to_response_items(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a chat-completions message list into Responses API input items.

    Reasoning items previously returned by the API are echoed back from the
    custom RESPONSE_ITEMS_KEY field on assistant messages (set by the harness
    after each Responses call). This is how chain-of-thought carries forward
    across turns in stateless mode.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()

        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "").strip()
            content = msg.get("content")
            if isinstance(content, list):
                # Tool results are typically strings; if structured, JSON it.
                output_text = json.dumps(content)
            else:
                output_text = str(content) if content is not None else ""
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_text,
            })
            continue

        if role == "assistant":
            # Replay any reasoning + message items the API returned previously.
            # These carry the encrypted chain-of-thought back to the model.
            prior_items = msg.get(RESPONSE_ITEMS_KEY)
            tool_call_items = _tool_calls_to_function_call_items(msg.get("tool_calls") or [])
            if prior_items:
                # Trust the echoed shape (already canonical Responses items).
                for it in prior_items:
                    items.append(it)
                # function_call items still come from tool_calls (the echoed
                # output items usually contain them too; deduplicate by call_id).
                seen = {it.get("call_id") for it in prior_items if it.get("type") == "function_call"}
                for fc in tool_call_items:
                    if fc["call_id"] not in seen:
                        items.append(fc)
            else:
                content = msg.get("content")
                content_parts = _content_to_response_parts(content, is_assistant=True)
                if isinstance(content_parts, str):
                    if content_parts:
                        items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content_parts}],
                        })
                elif content_parts:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": content_parts,
                    })
                for fc in tool_call_items:
                    items.append(fc)
            continue

        # system / user / developer
        content_parts = _content_to_response_parts(msg.get("content"), is_assistant=False)
        if isinstance(content_parts, str):
            items.append({"role": role, "content": content_parts})
        else:
            items.append({"role": role, "content": content_parts})
    return items


def _tools_to_responses_format(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Chat Completions wraps as {"type":"function","function":{...}}; Responses uses flat."""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and "function" in tool:
            fn = tool["function"] or {}
            out.append({
                "type": "function",
                "name": str(fn.get("name") or ""),
                "description": str(fn.get("description") or ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        else:
            out.append(tool)
    return out


def build_responses_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    prompt_cache_key: str | None = None,
    reasoning_effort: str = "high",
) -> dict[str, Any]:
    """Build a /v1/responses payload using the stateless-echo pattern.

    `store=False` + `include=["reasoning.encrypted_content"]` returns the
    encrypted chain-of-thought blob on the response, which the harness
    stashes on the assistant message and echoes back on the next call so
    the model can build on its prior reasoning.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": messages_to_response_items(messages),
        "reasoning": {"effort": reasoning_effort},
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if max_tokens is not None and max_tokens > 0:
        payload["max_output_tokens"] = max_tokens
    if tools:
        payload["tools"] = _tools_to_responses_format(tools)
        # Responses API tool_choice: "auto" (default), "required", "none",
        # or {"type":"function","name":...}. The harness only uses "auto" so
        # we omit it unless explicitly set to something else.
        if tool_choice and tool_choice not in ("auto", "auto-required"):
            payload["tool_choice"] = tool_choice
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    return payload


def parse_responses_output(resp: dict[str, Any]) -> dict[str, Any]:
    """Reshape a /v1/responses JSON body into a chat-completions-style result.

    Returns a dict with:
      - message: {"role":"assistant","content":..., "tool_calls":[...]} so the
        existing tool-call dispatch loop works unchanged.
      - reasoning_text: concatenated reasoning summary text (often empty when
        summary isn't requested), purely for transcript logging.
      - output_items: raw `output` array, to be stashed on the assistant
        message under RESPONSE_ITEMS_KEY for next-turn carry-over.
      - finish_reason: best-effort string for the existing logging code.
    """
    output = resp.get("output") or []

    content_text = ""
    tool_calls: list[dict[str, Any]] = []
    reasoning_text_parts: list[str] = []

    for item in output:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    content_text += str(part.get("text", ""))
        elif t == "function_call":
            tool_calls.append({
                "id": str(item.get("call_id") or ""),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or ""),
                },
            })
        elif t == "reasoning":
            for s in item.get("summary") or []:
                if isinstance(s, dict) and s.get("type") == "summary_text":
                    reasoning_text_parts.append(str(s.get("text", "")))

    message: dict[str, Any] = {"role": "assistant", "content": content_text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
    # Some responses surface incomplete_details when truncated.
    if resp.get("incomplete_details"):
        finish_reason = str(resp["incomplete_details"].get("reason") or "incomplete")

    return {
        "message": message,
        "reasoning_text": "\n".join(reasoning_text_parts),
        "output_items": output,
        "finish_reason": finish_reason,
        "usage": resp.get("usage"),
    }
