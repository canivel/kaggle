"""A17 canary v7 — seed-2 CONFIRMATION push (sealed branch B2, 2026-07-29 slot-1).

Authorizing instrument: learnings/a17_threshold_commit_2026-07-28.md §3 branch
B2 — v6 (seed 1, full 7920 s window) read S1 = Sigma N_72B = 5 < 138 with a
VALID read (games_present 4/4, window 7920 s, no drift WARN), so "exactly ONE
confirmation seed fires as tomorrow's kernel slot-1 ... No parameter, prompt,
or harness change between seeds — a change would reset the count."

Seed convention (duck_eval/warpack/build_eval_notebook.py, EVAL_SEED_LINES_A17
comment): "seed N = push N of the identical notebook". A17_CANARY_SEED is a
greppable provenance tag, NOT an RNG input; the draw's independence comes from
LLM sampling. v7 therefore changes EXACTLY two substrings in cell 2 (the tag
"1"->"2" and the banner "seed=1"->"seed=2") and NOTHING else. This script
PROVES that: after the rewrite it asserts every other cell is byte-identical
and that reverse-substituting cell 2 reproduces the original exactly.

NO metadata change (v6 metadata = weights dataset attached, model_sources
removed — pull-back verified 07-28). Idempotence: running twice raises (the
seed=1 anchors are gone after the first run).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "a17-canary" / "arc3-a17-72b-canary.ipynb"
META_PATH = REPO / "notebooks" / "a17-canary" / "kernel-metadata.json"
WEIGHTS_DATASET = "canivel/qwen25-vl-72b-awq"

SEED_TAG_OLD = 'os.environ["A17_CANARY_SEED"] = "1"'
SEED_TAG_NEW = 'os.environ["A17_CANARY_SEED"] = "2"'
BANNER_OLD = '"A17-CANARY seed=1 mode=throughput-canary-v6-dataset-weights '
BANNER_NEW = '"A17-CANARY seed=2 mode=throughput-canary-v6-dataset-weights '


def fail(msg: str) -> None:
    print(f"BUILD-V7 FATAL: {msg}")
    sys.exit(1)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    original = copy.deepcopy(nb)

    cell2 = nb["cells"][2]
    src = "".join(cell2["source"])

    for anchor in (SEED_TAG_OLD, BANNER_OLD):
        n = src.count(anchor)
        if n != 1:
            fail(
                f"anchor must match exactly once in cell 2, found {n}: {anchor!r} "
                "(already built to v7? idempotence guard)"
            )
    for other_cell in (c for i, c in enumerate(nb["cells"]) if i != 2):
        s = "".join(other_cell["source"])
        if "A17_CANARY_SEED" in s and 'os.environ["A17_CANARY_SEED"] = ' in s:
            fail("seed tag found outside cell 2 — layout drifted, refusing")

    new_src = src.replace(SEED_TAG_OLD, SEED_TAG_NEW).replace(BANNER_OLD, BANNER_NEW)
    # Preserve the notebook's source representation (list-of-lines with \n).
    cell2["source"] = new_src.splitlines(keepends=True)

    # ---- PROOF: nothing but the two seed substrings changed ----------------
    if len(nb["cells"]) != len(original["cells"]):
        fail("cell count changed")
    for i, (a, b) in enumerate(zip(original["cells"], nb["cells"])):
        sa, sb = "".join(a["source"]), "".join(b["source"])
        if i == 2:
            reverted = sb.replace(SEED_TAG_NEW, SEED_TAG_OLD).replace(
                BANNER_NEW, BANNER_OLD
            )
            if reverted != sa:
                fail("cell 2 diff is NOT limited to the two seed substrings")
        elif sa != sb or a.get("metadata") != b.get("metadata"):
            fail(f"cell {i} changed — B2 forbids any non-seed change")

    # ---- v6 config still intact (window, mode, metadata) -------------------
    all_src = "".join("".join(c["source"]) for c in nb["cells"])
    for must in (
        "A17_WINDOW_S = 7920.0",
        "mode=throughput-canary-v6-dataset-weights",
        "seed=2",
        SEED_TAG_NEW,
    ):
        if must not in all_src:
            fail(f"required string missing after build: {must!r}")
    if SEED_TAG_OLD in all_src or "seed=1" in all_src:
        fail("a seed-1 remnant survived the rewrite")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if meta.get("model_sources"):
        fail(f"model_sources must stay empty (dead route), got {meta['model_sources']}")
    if not any(WEIGHTS_DATASET in d for d in meta.get("dataset_sources", [])):
        fail(f"weights dataset {WEIGHTS_DATASET} missing from dataset_sources")

    NB_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print("BUILD-V7 OK: seed tag 1->2 and banner seed=1->seed=2; all other cells")
    print("byte-identical; window 7920 intact; metadata untouched (dataset-weights")
    print(f"route, {WEIGHTS_DATASET}). Ready for preflight + push as kernel v7.")


if __name__ == "__main__":
    main()
