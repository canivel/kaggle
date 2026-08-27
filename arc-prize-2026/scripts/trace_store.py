"""TRACE STORE — every LLM call, every <think>, every harness outcome.

WHY A DATABASE AND NOT JSONL
    The screening loop exists to answer "what is going wrong". Those are
    ANALYTICAL questions across runs -- which games starve the observation
    layer, on which turns did the model ignore an armed affordance, how much of
    each response is reasoning versus answer, did local behaviour diverge from
    Kaggle. JSONL forces a full scan and re-parse for every one of them.

WHY SQLITE
    Already this repo's idiom (bench.db, kaos.db) so there are no new ops.
    Single file, transactional, crash-safe, and in WAL mode a long screen can
    stream rows in while you query from another shell. At this volume (~20
    calls/hour of screening) it is far below any scale where a server database
    would earn its overhead.

WHY CONTENT-ADDRESSED BLOBS -- the design point that matters
    The harness RESENDS a near-identical prompt prefix on every turn; that
    redundancy is exactly why prefill dominates wall-clock (182 s/call). Stored
    naively, a few hundred calls would write the same tens of megabytes over
    and over. Here every payload is keyed by sha256 of its bytes and stored
    ONCE, compressed; calls reference it. The repetition that makes the run
    slow makes the storage nearly free.

LAYOUT
    blob      sha -> compressed payload, deduped
    run       one screening iteration (config, model, git sha, lane)
    call      one LLM request/response, with the measures worth indexing
              pulled out as columns (think_chars, n_tool_calls, ...)
    game_run  per-game outcome from benchmark.json
    finding   diagnosis rows

ANALYSIS
    Plain SQL works. For heavier aggregates DuckDB reads this file directly:
        duckdb -c "ATTACH 'runs/traces.db' AS t (TYPE sqlite); SELECT ..."
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zlib
from pathlib import Path
from typing import Any

try:
    import zstandard as _zstd
    _ZC = _zstd.ZstdCompressor(level=10)
    _ZD = _zstd.ZstdDecompressor()
    _CODEC = "zstd"
except Exception:  # noqa: BLE001
    _CODEC = "zlib"

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "runs" / "traces.db"
_THINK = re.compile(r"<think>(.*?)</think>", re.S)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS blob (
    sha      TEXT PRIMARY KEY,
    codec    TEXT NOT NULL,
    n_bytes  INTEGER NOT NULL,      -- uncompressed size
    z        BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    id        INTEGER PRIMARY KEY,
    label     TEXT NOT NULL,
    started   TEXT NOT NULL,
    finished  TEXT,
    lane      TEXT NOT NULL DEFAULT 'MAC-SCREEN',
    certifies INTEGER NOT NULL DEFAULT 0,   -- ALWAYS 0 here. Kaggle certifies.
    model     TEXT,
    base_url  TEXT,
    git_sha   TEXT,
    config    TEXT
);

CREATE TABLE IF NOT EXISTS call (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER REFERENCES run(id),
    trace_id       TEXT,
    ts             TEXT,
    path           TEXT,
    status         INTEGER,
    elapsed_s      REAL,
    game_id        TEXT,
    action_num     INTEGER,
    analysis_step  INTEGER,
    request_sha    TEXT REFERENCES blob(sha),
    response_sha   TEXT REFERENCES blob(sha),
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    total_tokens       INTEGER,
    finish_reason  TEXT,
    content_chars  INTEGER,
    think_chars    INTEGER,
    has_think      INTEGER,
    reasoning_content_present INTEGER,
    n_tool_calls   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_call_run   ON call(run_id);
CREATE INDEX IF NOT EXISTS ix_call_game  ON call(game_id);
CREATE INDEX IF NOT EXISTS ix_call_think ON call(has_think);
CREATE INDEX IF NOT EXISTS ix_call_tools ON call(n_tool_calls);
CREATE UNIQUE INDEX IF NOT EXISTS ux_call_trace ON call(trace_id);

CREATE TABLE IF NOT EXISTS game_run (
    id               INTEGER PRIMARY KEY,
    run_id           INTEGER REFERENCES run(id),
    game_id          TEXT,
    levels_completed INTEGER,
    number_of_levels INTEGER,
    actions          INTEGER,
    final_score      REAL,
    state            TEXT,
    solver_note      TEXT
);
CREATE INDEX IF NOT EXISTS ix_gamerun_run ON game_run(run_id);

CREATE TABLE IF NOT EXISTS finding (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER REFERENCES run(id),
    severity TEXT,
    category TEXT,
    what     TEXT,
    means    TEXT,
    fix_next TEXT
);
"""


def _compress(raw: bytes) -> bytes:
    return _ZC.compress(raw) if _CODEC == "zstd" else zlib.compress(raw, 6)


def _decompress(z: bytes, codec: str) -> bytes:
    if codec == "zstd":
        return _ZD.decompress(z)
    return zlib.decompress(z)


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    con.executescript(SCHEMA)
    return con


def put_blob(con: sqlite3.Connection, obj: Any) -> str | None:
    """Store a payload once, keyed by content hash. Returns the sha."""
    if obj is None:
        return None
    raw = json.dumps(obj, ensure_ascii=True, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    con.execute(
        "INSERT OR IGNORE INTO blob(sha, codec, n_bytes, z) VALUES (?,?,?,?)",
        (sha, _CODEC, len(raw), _compress(raw)),
    )
    return sha


def get_blob(con: sqlite3.Connection, sha: str) -> Any:
    row = con.execute("SELECT codec, z FROM blob WHERE sha=?", (sha,)).fetchone()
    if not row:
        return None
    return json.loads(_decompress(row[1], row[0]))


def measure_response(body: dict) -> dict:
    """Derive the columns worth indexing, so questions do not need a full scan.

    REASONING ARRIVES ON THREE POSSIBLE CHANNELS and all must be measured:
      * `message.reasoning`          <- what mlx_lm.server actually emits
      * `message.reasoning_content`  <- what vLLM's --reasoning-parser emits
      * inline `<think>...</think>`  <- raw, when no parser splits it out
    The harness's _extract_reasoning_text reads `reasoning` first and falls back
    to `reasoning_content`, so the local server and the Kaggle rail agree more
    closely than a content-only reading would suggest. Measuring only one
    channel is how a full reasoning trace gets recorded as "no thinking".

    Note `content` can be ABSENT entirely when a response is truncated mid-
    reasoning (finish_reason='length'), so nothing here may assume it exists.
    """
    out = {"content_chars": None, "think_chars": None, "has_think": None,
           "reasoning_content_present": None, "n_tool_calls": None,
           "finish_reason": None, "prompt_tokens": None,
           "completion_tokens": None, "total_tokens": None}
    try:
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        inline = sum(len(m) for m in _THINK.findall(content))
        out["content_chars"] = len(content)
        out["think_chars"] = len(reasoning) + inline
        out["has_think"] = int(bool(reasoning) or "<think>" in content)
        out["reasoning_content_present"] = int(
            "reasoning" in msg or "reasoning_content" in msg)
        out["n_tool_calls"] = len(msg.get("tool_calls") or [])
        out["finish_reason"] = ch.get("finish_reason")
        u = body.get("usage") or {}
        out["prompt_tokens"] = u.get("prompt_tokens")
        out["completion_tokens"] = u.get("completion_tokens")
        out["total_tokens"] = u.get("total_tokens")
    except Exception:  # noqa: BLE001
        pass
    return out


def start_run(con: sqlite3.Connection, *, label: str, started: str, model: str,
              base_url: str, git_sha: str | None, config: dict) -> int:
    cur = con.execute(
        "INSERT INTO run(label, started, lane, certifies, model, base_url, git_sha, config)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (label, started, "MAC-SCREEN", 0, model, base_url, git_sha,
         json.dumps(config, ensure_ascii=True)),
    )
    con.commit()
    return int(cur.lastrowid)


def record_call(con: sqlite3.Connection, *, run_id: int | None, trace_id: str,
                ts: str, path: str, status: int, elapsed_s: float,
                request_body: Any, response_body: Any,
                game_id: str | None = None, action_num: int | None = None,
                analysis_step: int | None = None) -> None:
    m = measure_response(response_body) if isinstance(response_body, dict) else {}
    con.execute(
        "INSERT OR REPLACE INTO call(run_id, trace_id, ts, path, status, elapsed_s,"
        " game_id, action_num, analysis_step, request_sha, response_sha,"
        " prompt_tokens, completion_tokens, total_tokens, finish_reason,"
        " content_chars, think_chars, has_think, reasoning_content_present,"
        " n_tool_calls) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, trace_id, ts, path, status, elapsed_s, game_id, action_num,
         analysis_step, put_blob(con, request_body), put_blob(con, response_body),
         m.get("prompt_tokens"), m.get("completion_tokens"), m.get("total_tokens"),
         m.get("finish_reason"), m.get("content_chars"), m.get("think_chars"),
         m.get("has_think"), m.get("reasoning_content_present"),
         m.get("n_tool_calls")),
    )
    con.commit()


def ingest_benchmark(con: sqlite3.Connection, run_id: int, bench: dict) -> int:
    rows = 0
    for r in bench.get("game_runs", []):
        con.execute(
            "INSERT INTO game_run(run_id, game_id, levels_completed, number_of_levels,"
            " actions, final_score, state, solver_note) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, r.get("game_id"), r.get("levels_completed"),
             r.get("number_of_levels"), sum(r.get("actions_per_level") or []),
             r.get("final_score"), r.get("state"), r.get("solver_note")),
        )
        rows += 1
    con.commit()
    return rows


def stats(con: sqlite3.Connection) -> dict:
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    raw = q("SELECT COALESCE(SUM(n_bytes),0) FROM blob")
    comp = q("SELECT COALESCE(SUM(LENGTH(z)),0) FROM blob")
    return {
        "runs": q("SELECT COUNT(*) FROM run"),
        "calls": q("SELECT COUNT(*) FROM call"),
        "game_runs": q("SELECT COUNT(*) FROM game_run"),
        "blobs": q("SELECT COUNT(*) FROM blob"),
        "raw_mb": round(raw / 1048576, 2),
        "stored_mb": round(comp / 1048576, 2),
        "compression": f"{raw / comp:.1f}x" if comp else "n/a",
        "codec": _CODEC,
        "db_mb": round(DB_PATH.stat().st_size / 1048576, 2) if DB_PATH.exists() else 0,
    }


if __name__ == "__main__":
    con = connect()
    print(f"trace store: {DB_PATH.relative_to(ROOT)}")
    for k, v in stats(con).items():
        print(f"  {k:14} {v}")
