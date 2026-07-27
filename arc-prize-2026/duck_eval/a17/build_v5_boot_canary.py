"""A17 canary v5 — DATASET-weights boot canary (2026-07-27).

Why v5 exists: v1/v2/v4 all died because the Kaggle save-kernel API silently
drops `model_sources` at push (root-caused 07-25/07-26; probe in
runs/model_attach_probe/). The pinned route (daily_iterate_prompt.md addendum
2026-07-27) is DATASET-served weights — the same proven pattern the duck
harness uses for the 27B (driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot).

v5 = the staged v4 notebook (fenced-recovery composition, UNTOUCHED serve
config) with exactly three deltas:

  1. metadata: `model_sources` REMOVED; dataset `canivel/qwen25-vl-72b-awq`
     (HF snapshot of Qwen/Qwen2.5-VL-72B-Instruct-AWQ) appended to
     `dataset_sources`. The cell-8 model finder is already mount-path
     agnostic (rglob of /kaggle/input for the Qwen2_5_VLForConditionalGeneration
     + quantization_config + *.safetensors markers), so the notebook needs no
     path change — only the mount mechanism changes.
  2. cell 2: banner rewritten to mode=boot-canary-v5-dataset-weights.
  3. cell 14: A17_WINDOW_S 7920 -> 1500 s and the offline soft_end anchored
     at bm.run entry (datetime.now() + A17_WINDOW_S) instead of the
     budget-derived NOTEBOOK_START_EPOCH rule. Purpose: this push is a SHORT,
     CHEAP serve-config test of the dataset route (vLLM boot + /v1/models
     identity + tool-call round-trip + MM probe already enforced as FAIL-LOUD
     cell-8 boot asserts, plus a ~25-min in-game slice for the fenced-recovery
     adapter) — NOT the sealed bench. Its output is MEASUREMENT ONLY (k=1
     false-NO-GO = 1.0 per learnings/a17_error_model.md — no GO/NO-GO
     interpretation without panel). The full-window bench (7920 s) is queued
     for tomorrow contingent on this canary's PASS.

Side effects of the short window (all benign, pre-noted):
  - zero-action-abort (armed 1800 s..kill-disarm) has an EMPTY arming range
    at window 1500 (disarm boundary = 900 s) -> never fires. Acceptable: the
    window itself is only ~25 min.
  - stall-kill is armed only for the first 900 s of the game window; the boot
    asserts remain the hard gate.
  - window-drift WARN banners compare against 1500 s, not 7920 s.

Every rewrite anchor must match EXACTLY once or this script raises
(same discipline as the cell-8 runtime rewrites). Idempotence: running twice
raises (the v4 anchors are gone after the first run).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "a17-canary" / "arc3-a17-72b-canary.ipynb"
META_PATH = REPO / "notebooks" / "a17-canary" / "kernel-metadata.json"
WEIGHTS_DATASET = "canivel/qwen25-vl-72b-awq"

CELL2_REWRITES: list[tuple[str, str]] = [
    (
        'os.environ["A17_CANARY_SEED"] = "1"  # A17 72B-VL canary (scope v2 sec3 '
        "rho_action denominator; sec5 full 7920s window)",
        'os.environ["A17_CANARY_SEED"] = "1"  # A17 72B-VL v5 BOOT CANARY: '
        "dataset-route serve validation, SHORT 1500s window (NOT the sealed bench)",
    ),
    (
        'print("A17-CANARY seed=1 mode=throughput-canary-v4-fencedrec '
        "games=ft09-0d8bbf25,sb26-7fbdac44,lp85-305b61c3,vc33-5430563c "
        "composition=W0 (duck + (f) continuation, NO warpack); 27B numerator "
        'frozen 480 actions/7920s (w0_eval_s1)")',
        'print("A17-CANARY seed=1 mode=boot-canary-v5-dataset-weights '
        "games=ft09-0d8bbf25,sb26-7fbdac44,lp85-305b61c3,vc33-5430563c "
        "composition=W0+fencedrec; weights=DATASET canivel/qwen25-vl-72b-awq "
        "(model_sources route DEAD, silently dropped at push 07-25/07-26); "
        "window=1500s BOOT TEST, output MEASUREMENT ONLY (a17_error_model.md: "
        'no NO-GO reading at k=1)")',
    ),
]

CELL14_REWRITES: list[tuple[str, str]] = [
    (
        "    A17_WINDOW_S = 7920.0",
        "    A17_WINDOW_S = 1500.0  # v5 BOOT CANARY short window "
        "(dataset-route serve test; sealed bench stays 7920s)",
    ),
    (
        "    if not TRUE_SUBMISSION:\n"
        '        budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)\n'
        "        if budget > 0:\n"
        "            soft_end = min(soft_end, datetime.fromtimestamp(NOTEBOOK_START_EPOCH)"
        " + timedelta(seconds=budget - min(600.0, budget / 2)))\n",
        "    if not TRUE_SUBMISSION:\n"
        "        # v5 BOOT CANARY: fixed short window anchored at bm.run entry, so the\n"
        "        # 72B load time (before this cell) is excluded from the game window.\n"
        "        soft_end = min(soft_end, datetime.now() + timedelta(seconds=A17_WINDOW_S))\n",
    ),
]


def _apply(text: str, rewrites: list[tuple[str, str]], label: str) -> str:
    for old, new in rewrites:
        found = text.count(old)
        if found != 1:
            raise SystemExit(
                f"FATAL {label}: anchor matched {found} times (want 1): {old[:90]!r}"
            )
        text = text.replace(old, new)
    return text


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Precondition: this must be the v4 (fencedrec) composition.
    cell12 = "".join(nb["cells"][12]["source"])
    if "fenced_recovery_patch" not in cell12:
        raise SystemExit("FATAL: cell 12 lacks the fenced-recovery graft — not the v4 notebook")

    for idx, rewrites in ((2, CELL2_REWRITES), (14, CELL14_REWRITES)):
        cell = nb["cells"][idx]
        src = _apply("".join(cell["source"]), rewrites, f"cell {idx}")
        if idx == 14 and "A17_WINDOW_S = 7920.0" in src:
            raise SystemExit("FATAL: 7920.0 window survived the cell-14 rewrite")
        # Compile check (top-level await is legal in the notebook runtime).
        compile(src, f"cell{idx}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        cell["source"] = src.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if "model_sources" not in meta:
        raise SystemExit("FATAL: metadata has no model_sources — already v5? (idempotence guard)")
    dropped = meta.pop("model_sources")
    if WEIGHTS_DATASET in meta["dataset_sources"]:
        raise SystemExit("FATAL: weights dataset already attached (idempotence guard)")
    meta["dataset_sources"].append(WEIGHTS_DATASET)

    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"v5 boot canary written: {NB_PATH}")
    print(f"  metadata: model_sources {dropped} REMOVED; dataset {WEIGHTS_DATASET} attached")
    print(f"  cell 2: {len(CELL2_REWRITES)} rewrites; cell 14: {len(CELL14_REWRITES)} rewrites")


if __name__ == "__main__":
    sys.exit(main())
