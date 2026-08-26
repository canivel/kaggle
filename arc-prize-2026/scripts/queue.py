"""Queue helper for the ARC daily submission daemon.

Usage:
  uv run python scripts/queue.py add <kernel> <version> "<message>"
  uv run python scripts/queue.py list
  uv run python scripts/queue.py pop                  # remove front of queue
  uv run python scripts/queue.py clear                # wipe pending (keep history)
  uv run python scripts/queue.py refill               # arm eternal fallback iff empty

Example:
  uv run python scripts/queue.py add canivel/arc3-forge63 1 "v63 = v62 + JEPA n_sims=32"
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

QUEUE = Path(__file__).resolve().parents[1] / "submission_queue.json"


def load() -> dict:
    return json.loads(QUEUE.read_text(encoding="utf-8")) if QUEUE.exists() else {"pending": [], "history": []}


def save(q: dict) -> None:
    QUEUE.write_text(json.dumps(q, indent=2), encoding="utf-8")


def cmd_add(kernel: str, version: str, message: str, file: str = "submission.parquet", upstream: str = "") -> None:
    q = load()
    entry = {
        "kernel": kernel,
        "version": int(version),
        "file": file,
        "message": message,
    }
    # Duck-lineage kernels need trusted-fork preflight (baseline structural
    # checks would false-block them). Auto-tag so a bare 
    # never causes a preflight block again (2026-07-09 incident).
    if upstream:
        # Explicit upstream wins: any byte-identical rebase of a proven public
        # kernel is a trusted fork, whatever its slug is called.
        entry["preflight_mode"] = "trusted-fork"
        entry["upstream"] = upstream
    elif "duck" in kernel:
        entry["preflight_mode"] = "trusted-fork"
        entry["upstream"] = "jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner"
    else:
        # 2026-08-20 incident: the A21 field-arm head (no "duck" in slug, no
        # upstream arg) fell to the strict war-eval structural preflight, which a
        # foreign-lineage notebook can NEVER pass, and the daemon does not fall
        # through to entry 2 -- the draw survives only because the block fired at
        # 00:07Z with the 22:37Z window still ahead. Warn loudly at queue time.
        print("WARNING: no upstream given and slug is not duck-lineage -> this entry "
              "will face the STRICT war-eval structural preflight at submit time. "
              "If this kernel is a rebase of a public artifact, re-add with: "
              "queue.py add <kernel> <version> <message> <upstream>")
    q.setdefault("pending", []).append(entry)
    save(q)
    print(f"queued #{len(q['pending'])}: {kernel} v{version}")


def cmd_list() -> None:
    q = load()
    pending = q.get("pending", [])
    print(f"pending ({len(pending)}):")
    for i, item in enumerate(pending):
        print(f"  {i+1}. {item['kernel']} v{item['version']}  — {item['message'][:90]}")
    hist = q.get("history", [])
    if hist:
        print(f"\nhistory (last 5):")
        for h in hist[-5:]:
            print(f"  - {h.get('submitted_at','?')}  {h['kernel']} v{h['version']}")


def cmd_pop() -> None:
    q = load()
    p = q.get("pending", [])
    if not p:
        print("empty")
        return
    dropped = p.pop(0)
    save(q)
    print(f"popped: {dropped['kernel']} v{dropped['version']}")


def cmd_refill() -> None:
    """Arm the eternal fallback iff pending is empty (manual mirror of the
    daemon's auto-refill). Uses the SAME entry definition — see
    scripts/daily_submit.py:eternal_fallback_entry."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from daily_submit import eternal_fallback_entry  # noqa: E402

    q = load()
    if q.get("pending"):
        print(f"not empty ({len(q['pending'])} pending) — no refill")
        return
    q["pending"] = [eternal_fallback_entry(note="manual")]
    save(q)
    print("refilled: eternal fallback armed")


def cmd_clear() -> None:
    q = load()
    n = len(q.get("pending", []))
    q["pending"] = []
    save(q)
    print(f"cleared {n} entries")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    op = sys.argv[1]
    if op == "add" and len(sys.argv) >= 5:
        cmd_add(sys.argv[2], sys.argv[3], sys.argv[4],
                upstream=(sys.argv[5] if len(sys.argv) >= 6 else ""))
    elif op == "add":
        print("usage: add <kernel> <version> \"<message>\" [<upstream-owner/slug>]")
        return 1
    elif op == "list":
        cmd_list()
    elif op == "pop":
        cmd_pop()
    elif op == "refill":
        cmd_refill()
    elif op == "clear":
        cmd_clear()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
