"""Rewrite teacher note phrasings in an already-generated dataset, in place.

The searcher's voice leaks into the note if you are not careful. "the sequence
my search verified" is something the student model can never truthfully say --
training on it teaches the model to *claim* verification it does not have,
which is a hallucination-inducing pattern, not a policy. This applies the
current `teacher.render_note` phrasing to assistant text everywhere it appears
(targets AND the assistant turns already baked into each example's history), so
a regenerated corpus and a rewritten one agree.

    ../../.venv/Scripts/python.exe rewrite_notes.py --file ../../runs/lora_lane/v0/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REWRITES: list[tuple[str, str]] = [
    (
        " is the sequence my search verified for this state.",
        " -- the shortest sequence consistent with the evidence so far. "
        "I commit it rather than re-testing one action at a time.",
    ),
    (
        "Open questions: the final action of this batch should clear the level; "
        "if it does, I stop and re-read on the next level rather than probing here.",
        "Open questions: whether this batch clears the level. If it does I stop "
        "and re-read on the next level rather than probing here.",
    ),
]


def fix(text: str) -> tuple[str, int]:
    hits = 0
    for old, new in REWRITES:
        if old in text:
            hits += text.count(old)
            text = text.replace(old, new)
    return text, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    path = Path(args.file)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = 0
    for row in rows:
        for message in row["messages"]:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                message["content"], hits = fix(message["content"])
                total += hits
        target = row["target"]
        if isinstance(target.get("content"), str):
            target["content"], hits = fix(target["content"])
            total += hits
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rewrote {total} phrasings across {len(rows)} examples in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
