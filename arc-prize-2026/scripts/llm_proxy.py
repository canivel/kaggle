"""Transparent logging proxy in front of the local model server.

WHY
    "Are we saving all the <think>?" should not depend on the harness choosing
    to log it. This proxy sits between the harness and mlx_lm.server and writes
    EVERY request and EVERY response body verbatim into the trace store. On
    mlx_lm.server there is no reasoning parser, so the model's <think> block
    arrives inline in `choices[].message.content` -- capturing the raw response
    body therefore captures the full reasoning trace, with nothing relying on
    the harness's own request logs.

    This is ground truth for the screening rail: what was actually sent, what
    actually came back, how long it took.

LAYOUT
    harness ->  127.0.0.1:1234  (this proxy)  ->  127.0.0.1:1235  (mlx_lm.server)
    traces  ->  runs/traces.db  (SQLite WAL; see scripts/trace_store.py)

    One `call` row per exchange, holding request and response as
    content-addressed compressed blobs plus the measures worth indexing
    (think_chars, n_tool_calls, finish_reason, usage). Because the harness
    resends a near-identical prefix each turn, the blob dedupe makes the
    repetition nearly free to store.

USAGE
    python scripts/llm_proxy.py --listen 1234 --upstream 1235
    (serve_local_model.sh wires this up automatically)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import trace_store  # noqa: E402

_counter = 0
_lock = threading.Lock()
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _next_id() -> str:
    global _counter
    with _lock:
        _counter += 1
        return f"{dt.datetime.now():%H%M%S}-{_counter:05d}"


_con = None
_db_lock = threading.Lock()


def _store():
    """One connection per process; SQLite WAL handles concurrent readers."""
    global _con
    if _con is None:
        _con = trace_store.connect()
    return _con


def _record(trace_id, ts, path, status, elapsed, req_body, resp_body):
    """Never let a logging failure break the proxy."""
    try:
        with _db_lock:
            trace_store.record_call(
                _store(), run_id=None, trace_id=trace_id, ts=ts, path=path,
                status=status, elapsed_s=elapsed,
                request_body=req_body, response_body=resp_body)
    except Exception as exc:  # noqa: BLE001
        print(f"[proxy] trace write failed: {exc}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 1235

    def log_message(self, *_args):  # silence per-request stderr spam
        pass

    def _proxy(self, method: str) -> None:
        rid = _next_id()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        try:
            body = json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            body = None

        url = f"http://127.0.0.1:{self.upstream_port}{self.path}"
        req = urllib.request.Request(url, data=raw or None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)

        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                payload = resp.read()
                status = resp.status
                ctype = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload, status = exc.read(), exc.code
            ctype = "application/json"
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps({"error": str(exc)}).encode()
            status, ctype = 502, "application/json"
        elapsed = time.time() - started

        try:
            rbody = json.loads(payload)
        except Exception:  # noqa: BLE001
            rbody = None

        _record(rid, dt.datetime.now(dt.timezone.utc).isoformat(),
                self.path, status, round(elapsed, 2), body, rbody)

        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802
        self._proxy("POST")

    def do_GET(self):  # noqa: N802
        self._proxy("GET")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=1234)
    ap.add_argument("--upstream", type=int, default=1235)
    args = ap.parse_args()

    Handler.upstream_port = args.upstream
    srv = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    print(f"[proxy] {args.listen} -> {args.upstream}  tracing to "
          f"{trace_store.DB_PATH.relative_to(ROOT)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
