"""Ingest campaign artifacts into kaos.db memory rows (Kimi-3 adopt #2, corrected).

The campaign's KAOS memory stopped accruing 2026-05-25 (66 rows) while the war
room accumulated 15+ deep-reads, amendments A8-A13, and panel rounds 10-14 as
prose files. A dream/consolidation pass over a stale corpus is
consolidation-of-nothing (Kimi-3 review, verified 2026-07-18) — so this script
ingests new artifacts FIRST, idempotently.

Expectation (sealed): with no hit-count accrual (consumers read files, not KAOS
memory), dream output is a recency-weighted digest for the panel agenda — a
summarizer, not a knowledge gardener.

Usage: uv run python scripts/kaos_ingest.py [--dry-run]
Then:  cd ../kaos && uv run kaos dream   (weekly, Sundays per protocol)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "kaos.db"

SOURCES = [
    ROOT / "learnings" / "war_room",
    ROOT / "learnings",          # amendments, briefs, preregistrations (top level only)
    ROOT / "runs" / "lb_process_model",
]
PANEL_SUMMARIES = sorted((ROOT / "learnings" / "panel").glob("round*/_summary.json"))
EXCERPT_CHARS = 1200


def iter_artifacts():
    seen = set()
    for src in SOURCES:
        if not src.exists():
            continue
        for p in sorted(src.glob("*.md")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                yield p
    for p in PANEL_SUMMARIES:
        yield p


def row_for(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        d = json.loads(text)
        content = (f"Panel {path.parent.name}: pass={d.get('pass')} "
                   f"accepts={d.get('n_accept')} fatals={d.get('n_fatal_total')} "
                   f"verdicts={ {k: v.get('verdict') for k, v in d.get('reviews', {}).items()} }")
    else:
        content = text[:EXCERPT_CHARS]
    rel = str(path.relative_to(ROOT))
    return {
        "key": f"campaign-doc:{rel}",
        "content": f"[{rel}] {content}",
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # memory.agent_id is NOT NULL FK -> agents; use a dedicated ingest agent.
    AGENT_ID = "campaign-doc-ingest-0001"
    cur.execute(
        "INSERT OR IGNORE INTO agents (agent_id, name, status) "
        "VALUES (?, 'campaign-doc-ingest', 'completed')",
        (AGENT_ID,),
    )

    existing = {r[0]: r[1] for r in cur.execute(
        "SELECT key, content FROM memory WHERE key LIKE 'campaign-doc:%'")}

    inserted = updated = unchanged = 0
    for path in iter_artifacts():
        r = row_for(path)
        prev = existing.get(r["key"])
        if prev == r["content"]:
            unchanged += 1
            continue
        if args.dry_run:
            print(("UPDATE " if prev else "INSERT ") + r["key"])
            inserted += prev is None
            updated += prev is not None
            continue
        if prev is None:
            cur.execute(
                "INSERT INTO memory (agent_id, type, key, content, metadata, created_at) "
                "VALUES (?, 'insight', ?, ?, ?, ?)",
                (AGENT_ID, r["key"], r["content"],
                 json.dumps({"sha": r["sha"], "source": "kaos_ingest"}), r["mtime"]),
            )
            inserted += 1
        else:
            cur.execute(
                "UPDATE memory SET content = ?, created_at = ? WHERE key = ?",
                (r["content"], r["mtime"], r["key"]),
            )
            updated += 1
    if not args.dry_run:
        con.commit()
    total = cur.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
    print(f"inserted={inserted} updated={updated} unchanged={unchanged} total_rows={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
