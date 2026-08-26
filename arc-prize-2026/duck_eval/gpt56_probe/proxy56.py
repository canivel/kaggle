"""Local chat-completions -> /v1/responses translation proxy for gpt-5.6-sol.

gpt-5.6-* on /v1/chat/completions rejects function tools while reasoning is
active ("use /v1/responses or set reasoning_effort to 'none'"). The duck
harness speaks chat-completions; disabling reasoning would defeat the probe.
This proxy lets the harness keep its protocol: it listens on localhost,
translates each chat request into a stateless /v1/responses call, and maps
the response (output_text + function_call items) back into chat shape,
including usage token fields (which feed the client-side spend guard).

Run: .venv/Scripts/python.exe duck_eval/gpt56_probe/proxy56.py --port 8056
Env: OPENAI_API_KEY (upstream key), GPT56_REASONING_EFFORT (optional).
"""
from __future__ import annotations

import argparse
import json
import os

import requests
from flask import Flask, Response, jsonify, request

UPSTREAM = "https://api.openai.com/v1"
app = Flask(__name__)


def _key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _normalize_content(content, role: str):
    """Chat content parts -> Responses content parts.

    chat: {"type":"text"|"image_url", ...}; responses input wants
    input_text/input_image (and output_text inside assistant messages).
    """
    if content is None or isinstance(content, str):
        return content or ""
    out = []
    text_type = "output_text" if role == "assistant" else "input_text"
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            out.append({"type": text_type, "text": part.get("text", "")})
        elif ptype == "image_url":
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url", "")
            out.append({"type": "input_image", "image_url": url})
        else:
            out.append(part)
    return out


def _chat_messages_to_input(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _normalize_content(msg.get("content"), role)
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content if isinstance(content, str) else json.dumps(content),
            })
            continue
        if role == "assistant":
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                })
            continue
        items.append({"role": role, "content": content})
    return items


def _chat_tools_to_responses(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


@app.get("/v1/models")
def models() -> Response:
    r = requests.get(f"{UPSTREAM}/models",
                     headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    return Response(r.content, status=r.status_code, content_type="application/json")


@app.post("/v1/chat/completions")
def chat() -> Response:
    payload = request.get_json(force=True)
    body: dict = {
        "model": payload.get("model", "gpt-5.6-sol"),
        "input": _chat_messages_to_input(payload.get("messages", [])),
        "store": False,
    }
    tools = _chat_tools_to_responses(payload.get("tools"))
    if tools:
        body["tools"] = tools
        tc = payload.get("tool_choice")
        if tc in ("auto", "required", "none"):
            body["tool_choice"] = tc
    max_out = payload.get("max_completion_tokens") or payload.get("max_tokens")
    if max_out:
        body["max_output_tokens"] = max_out
    effort = os.environ.get("GPT56_REASONING_EFFORT", "").strip()
    if effort:
        body["reasoning"] = {"effort": effort}

    r = requests.post(
        f"{UPSTREAM}/responses",
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=580,
    )
    if r.status_code >= 400:
        # Pass the upstream error body through verbatim: the harness greps it
        # (e.g. context-length detection) and logs it.
        return Response(r.content, status=r.status_code, content_type="application/json")

    data = r.json()
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in data.get("output", []):
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content_parts.append(part.get("text", ""))
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
        # 'reasoning' items are intentionally dropped.

    finish = "stop"
    if tool_calls:
        finish = "tool_calls"
    elif data.get("status") == "incomplete" and (
        (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    ):
        finish = "length"

    message: dict = {"role": "assistant",
                     "content": "\n".join(content_parts) if content_parts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_in = data.get("usage") or {}
    usage = {
        "prompt_tokens": usage_in.get("input_tokens", 0),
        "completion_tokens": usage_in.get("output_tokens", 0),
        "total_tokens": usage_in.get("total_tokens", 0),
    }
    return jsonify({
        "id": data.get("id", "resp"),
        "object": "chat.completion",
        "model": data.get("model", body["model"]),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8056)
    args = ap.parse_args()
    if not _key():
        raise SystemExit("OPENAI_API_KEY not set")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
