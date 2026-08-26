#!/usr/bin/env python3
"""O1 -- does the local 25-game screen predict the public board ABOVE the floor?

WHY THIS EXISTS. Competitor report (Kaggle discussion 736578, 2026-08-21): the Tufa
duck harness scored local 2.1% / board 1.4%, and the same author's own harness scored
local 5.0-5.4% / board *still* 1.4%. A ~2.5x local gain bought zero board movement.
Our own screening rail promotes on `trim1`, justified by an R^2 = 0.990 linear fit to
5 LB anchors -- but every one of those anchors sits AT the certified floor. A fit has
no warrant outside the range it was fitted on.

If the mapping is flat above the floor, we are screening on a quantity the board does
not pay for, and the build rail is mis-aimed.

This joins every pulled artifact's LOCAL score to the BOARD draw of the submission that
carried it, using the submission descriptions (which name the kernel/arm) as the join
key. It reports the pairs and refuses to fit anything it cannot support.

Read-only. Writes nothing.

Usage:  uv run python scripts/local_vs_board.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PULLS = REPO / "runs" / "kernel_pulls"


def local_stats(pull_dir: Path):
    bp = pull_dir / "benchmark.json"
    if not bp.is_file():
        return None
    try:
        b = json.loads(bp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    gr = b.get("game_runs")
    if not isinstance(gr, list) or not gr:
        return None
    scores = [float(r.get("final_score") or 0.0) for r in gr]
    lc = sum(int(r.get("levels_completed") or 0) for r in gr)
    mean = sum(scores) / len(scores)
    trim1 = (sum(scores) - max(scores)) / (len(scores) - 1) if len(scores) > 1 else 0.0
    return {
        "n": len(gr), "lc": lc, "mean": mean, "trim1": trim1,
        "solver": str(b.get("solver_label"))[:24],
        "label": str(b.get("label"))[:30],
    }


def main() -> int:
    rows = []
    for d in sorted(PULLS.iterdir()):
        if not d.is_dir():
            continue
        s = local_stats(d)
        if s:
            s["pull"] = d.name
            rows.append(s)

    rows.sort(key=lambda r: -r["lc"])
    print("=" * 92)
    print("LOCAL SCREEN STATISTICS -- every pulled artifact with a benchmark.json")
    print("=" * 92)
    print(f"{'pull':24}{'n':4}{'lc':5}{'mean':9}{'trim1':9}  solver")
    for r in rows:
        print(f"{r['pull']:24}{r['n']:4}{r['lc']:5}{r['mean']:9.3f}{r['trim1']:9.3f}  {r['solver']}")
    print(f"\npulls with a benchmark: {len(rows)}")

    # ---- the join: artifacts whose board draw we actually know ----
    # Hand-maintained because the join key lives in free-text submission
    # descriptions; each entry is traceable to a submission row + a pull dir.
    PAIRS = [
        # (pull dir,           board draw, note)
        ("q38_field_v1",       1.16, "field floor, draw 4 of the config (08-24)"),
        ("execwm_v1",          1.05, "exec-WM v1 first draw (08-26)"),
        ("q38low_v1",          None, "INFRA DEATH -- gate asserted silence, no usable draw"),
        ("p1_notes_v1",        None, "COMPLETED but never submitted (delivery failure read)"),
        ("budget_t3_v1",       None, "screen only, never submitted"),
        ("budget_t05_v1",      None, "screen only, never submitted"),
        ("q38graft_v1",        None, "screen only; Arm 3 read HARM, failed the head gate"),
        ("graft_floor_v1",     None, "screen only"),
        ("graft_confirm_v1",   None, "screen only, lane closed NULL"),
        ("private_base_v1",    None, "private arm, not submitted"),
    ]
    by_pull = {r["pull"]: r for r in rows}

    print()
    print("=" * 92)
    print("THE JOIN -- local screen vs the board draw that artifact actually produced")
    print("=" * 92)
    print(f"{'pull':22}{'lc':5}{'trim1':9}{'mean':9}{'BOARD':8}  note")
    paired = []
    for pull, board, note in PAIRS:
        r = by_pull.get(pull)
        if not r:
            print(f"{pull:22}{'--':>5}{'--':>9}{'--':>9}{'--':>8}  (no benchmark on disk)")
            continue
        b = f"{board:.2f}" if board is not None else "--"
        print(f"{pull:22}{r['lc']:5}{r['trim1']:9.3f}{r['mean']:9.3f}{b:>8}  {note}")
        if board is not None:
            paired.append((r, board))

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    n = len(paired)
    print(f"artifacts with BOTH a local screen and a board draw: {n}")
    if n < 3:
        print()
        print("  REFUSING TO FIT. With n = %d matched pairs there is no regression to run," % n)
        print("  and quoting a slope from them would be inventing evidence. What the pairs")
        print("  can support is a RANGE statement, and only that:")
    lcs = [r["lc"] for r, _ in paired]
    boards = [b for _, b in paired]
    if paired:
        print()
        print("    local lc range observed WITH a board draw : %d .. %d" % (min(lcs), max(lcs)))
        print("    board draws observed                      : %s" % ", ".join(f"{b:.2f}" for b in boards))
    # The decisive question: has any artifact ever screened ABOVE the floor?
    FLOOR_LC = 28
    above = [r for r in rows if r["lc"] > FLOOR_LC]
    print()
    print("  Artifacts screening ABOVE the certified floor (lc > %d):" % FLOOR_LC)
    if not above:
        print("    NONE.")
    for r in above:
        board = dict((p, b) for p, b, _ in PAIRS).get(r["pull"])
        print("    %-22s lc=%-4d trim1=%-7.3f board=%s" % (
            r["pull"], r["lc"], r["trim1"], f"{board:.2f}" if board else "NEVER SUBMITTED"))
    above_and_submitted = [r for r in above
                           if dict((p, b) for p, b, _ in PAIRS).get(r["pull"]) is not None]
    print()
    print("  ==> artifacts that screened above the floor AND produced a board draw: %d"
          % len(above_and_submitted))
    print()
    print("  This is the O1 answer in one line: the local->board mapping is UNCONSTRAINED")
    print("  above the floor because we have never put a point there. Every anchor behind")
    print("  the R^2=0.990 trim1 fit sits AT the floor, so the fit cannot license a")
    print("  promotion decision about a config that screens above it -- which is exactly")
    print("  the decision the screening rail exists to make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
