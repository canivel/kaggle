"""boristown readiness-gate — 2-seed ENTRY-GATE eval build (2026-07-30).

Discharges entry-gate #1's *live-firing* half (intent boristown_ab_intent_2026-
07-28.md §"Entry gates" #1; prereg boristown_ab_prereg_2026-07-29_DRAFT.md
BLOCKER 3): a FREE, zero-spend Kaggle kernel BUILD of the staged gate canary on
BOTH seeds whose log shows the gate observed firing ("A17-GATE ... GATE armed"
+ "A17-GATE observed-firing vllm_ready_latency_s=... GATE fired", latency ≤180 s,
plus boristown's own "vLLM server ready" line).

WHY A PLAIN BUILD *IS* THE EVAL (no force-offline graft needed):
  Unlike the war/sentinel eval builds (frozen fork of the *duckwar* baseline,
  which needs WARPACK_FORCE_OFFLINE_BENCH=1 to run the offline bench — see
  duck_eval/warpack/build_eval_notebook.py EVAL_LINE), the gate canary forks the
  *frozen duck fork* (notebooks/duckfork/...). That base's run cell (cell 15)
  branches on TRUE_SUBMISSION = KAGGLE_IS_COMPETITION_RERUN: when it is UNSET —
  i.e. any ordinary kernel BUILD — it plays the bundled competition environments
  OFFLINE via _offline_games(), writes a dummy submission.parquet, and is never
  scored. So a plain build of the gate canary already IS the offline eval run
  that fires the gate. No eval-mode flag is grafted (that would fork the audited
  single-cell-graft invariant the A/B rests on).

SEED CONVENTION (duck_eval/warpack/build_eval_notebook.py EVAL_SEED_LINES_*,
duck_eval/a17/build_v7_seed2.py): "seed N = push N of the identical notebook".
The seed tag is a greppable PROVENANCE marker, NOT an RNG input; the draw's
independence comes from LLM sampling. Mirrors the sentinel-eval's
SENTINEL_EVAL_SEED greppable stamp. This script therefore emits seed-1 and
seed-2 notebooks that differ in EXACTLY the seed substrings (the env tag value
and the seed banner value) and NOTHING else, and PROVES it: after building
seed-2 it reverse-substitutes back to seed-1 and asserts byte-identity, cell by
cell (same proof style as build_v7_seed2.py).

FORK-NEVER-BUILD: the seed-1 eval notebook is the staged scored canary
(notebooks/duckgate/arc3-duck-gate.ipynb, built by
duck_eval/a17/build_boristown_gate_canary.py, smoke 47/47) with ONE additive
graft to cell 2 — a greppable DUCK_GATE_EVAL_SEED tag + banner appended after
the existing A17-GATE canary banner. Every other cell (including the gate cell
and the run cell) is byte-identical to the staged canary. The gate/run/solver
surface is untouched, so the eval measures exactly the arm-B mechanism.

METADATA (feedback_kaggle_env_match, feedback_fresh_kernel_slug): fresh eval
slug canivel/arc3-duck-gate-eval (distinct from the SCORED slug
canivel/arc3-duck-gate, exactly as the sentinel used arc3-duck-sentinel-eval
distinct from the scored sentinel — eval builds must never burn the scored
slug's version history nor risk a rerun-mode collision). Every ENV field
(dataset_sources, docker_image, machine_shape, enable_gpu, ...) is byte-
identical to the staged gate canary / frozen fork. NO extra dataset (the gate
uses only `requests`, already in the image — unlike the sentinel-eval, which
added canivel/arc-war-kit for its patch import). NO model_sources.

Idempotence: deterministic-from-staged-canary. If an output already exists the
freshly-built bytes must match it, else raise (the staged canary drifted).

Run:  uv run python duck_eval/a17/build_gate_eval_2seed.py
NO kernel push. NO submission-queue change. $0 cloud. Build-rail only.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGED_NB = REPO / "notebooks" / "duckgate" / "arc3-duck-gate.ipynb"
STAGED_META = REPO / "notebooks" / "duckgate" / "kernel-metadata.json"

EVAL_KERNEL_ID = "canivel/arc3-duck-gate-eval"
EVAL_KERNEL_TITLE = "arc3-duck-gate-eval"
EVAL_NB_NAME = "arc3-duck-gate-eval.ipynb"

OUT = {
    1: REPO / "notebooks" / "duckgate-eval-s1",
    2: REPO / "notebooks" / "duckgate-eval-s2",
}

# Anchor: the staged canary's final cell-2 line (the A17-GATE *canary* banner).
# It has NO trailing newline (last element of the source list), so we append a
# leading "\n" then the seed lines. Must match EXACTLY once in cell 2.
CELL2_ANCHOR = (
    'print("A17-GATE canary mode=readiness-gate-ab-B version=1 '
    "base=canivel/arc3-duck-repro(frozen) "
    "graft=boristown/agi-duck-harness-fast-eval#cell16 (vLLM readiness gate, "
    'sole audited functional diff) : arm B of R22-D2 A/B", flush=True)'
)


def _seed_block(seed: int) -> str:
    """The additive greppable eval-seed provenance stamp for a given seed.

    Mirrors the sentinel-eval SENTINEL_EVAL_SEED stamp + banner. Pure telemetry:
    DUCK_GATE_EVAL_SEED is never read by the harness (seed N = push N of the
    identical notebook; independence comes from LLM sampling).
    """
    return (
        CELL2_ANCHOR
        + "\n"
        + f'os.environ["DUCK_GATE_EVAL_SEED"] = "{seed}"'
        "  # entry-gate eval seed tag (2-seed live-firing; seed N = push N "
        "of the identical notebook)\n"
        f'print("A17-GATE-EVAL seed={seed} mode=readiness-gate-ab-B-eval '
        "base=canivel/arc3-duck-gate(staged) : entry-gate live-firing "
        '(offline bench, NOT scored)", flush=True)'
    )


def _compile(src: str, label: str) -> None:
    compile(src, label, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def fail(msg: str) -> None:
    raise SystemExit(f"BUILD-GATE-EVAL FATAL: {msg}")


def _build_seed(base_nb: dict, seed: int) -> dict:
    nb = copy.deepcopy(base_nb)
    c2 = nb["cells"][2]
    src = "".join(c2["source"])

    n = src.count(CELL2_ANCHOR)
    if n != 1:
        fail(
            f"seed {seed}: cell-2 A17-GATE canary banner anchor matched {n} times "
            "(want 1) — staged canary drifted or already built (idempotence guard)"
        )
    if "DUCK_GATE_EVAL_SEED" in src:
        fail(f"seed {seed}: DUCK_GATE_EVAL_SEED already present in cell 2 (idempotence)")

    new_src = src.replace(CELL2_ANCHOR, _seed_block(seed))
    _compile(new_src, f"cell2_seed{seed}")
    c2["source"] = new_src.splitlines(keepends=True)
    c2["outputs"] = []
    c2["execution_count"] = None
    return nb


def _assert_only_seed_differs(nb1: dict, nb2: dict) -> None:
    """PROOF: seed-1 and seed-2 differ ONLY in the two seed substrings.

    Reverse-substitute seed-2's cell 2 back to seed-1 and require byte-identity,
    cell by cell (build_v7_seed2.py proof style).
    """
    if len(nb1["cells"]) != len(nb2["cells"]):
        fail("cell count differs between seed-1 and seed-2")
    for i, (a, b) in enumerate(zip(nb1["cells"], nb2["cells"])):
        sa, sb = "".join(a["source"]), "".join(b["source"])
        if i == 2:
            reverted = sb.replace(
                'os.environ["DUCK_GATE_EVAL_SEED"] = "2"',
                'os.environ["DUCK_GATE_EVAL_SEED"] = "1"',
            ).replace("A17-GATE-EVAL seed=2", "A17-GATE-EVAL seed=1")
            if reverted != sa:
                fail("cell 2 diff is NOT limited to the seed substrings")
        elif sa != sb or a.get("metadata") != b.get("metadata"):
            fail(f"cell {i} differs between seeds — only seed substrings may differ")


def _assert_nonseed_identical_to_staged(base_nb: dict, seed_nb: dict) -> None:
    """Every cell EXCEPT cell 2 is byte-identical to the staged scored canary."""
    if len(base_nb["cells"]) != len(seed_nb["cells"]):
        fail("cell count changed vs staged canary")
    for i, (a, b) in enumerate(zip(base_nb["cells"], seed_nb["cells"])):
        if i == 2:
            continue
        if "".join(a["source"]) != "".join(b["source"]):
            fail(f"cell {i} changed vs staged canary — only cell 2 may gain the seed stamp")


def _build_meta(base_meta: dict) -> dict:
    meta = copy.deepcopy(base_meta)
    meta["id"] = EVAL_KERNEL_ID
    meta["title"] = EVAL_KERNEL_TITLE
    meta["code_file"] = EVAL_NB_NAME
    # env-match discipline: every field except identity is byte-identical to the
    # staged gate canary (which the smoke already proved == frozen fork family).
    for field in ("dataset_sources", "kernel_sources", "competition_sources",
                  "model_sources", "docker_image", "machine_shape",
                  "enable_gpu", "enable_tpu", "enable_internet", "is_private",
                  "language", "kernel_type", "keywords"):
        if meta.get(field) != base_meta.get(field):
            fail(f"metadata field {field} drifted from staged gate canary")
    if meta.get("model_sources"):
        fail(f"model_sources must stay empty, got {meta['model_sources']}")
    return meta


def _write(out_dir: Path, nb: dict, meta: dict, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_bytes = json.dumps(nb, indent=1)          # match staged canary style (indent=1, no trailing nl)
    meta_bytes = json.dumps(meta, indent=2) + "\n"
    nb_path = out_dir / EVAL_NB_NAME
    meta_path = out_dir / "kernel-metadata.json"

    if nb_path.is_file():
        prev = nb_path.read_text(encoding="utf-8")
        if prev != nb_bytes:
            fail(
                f"seed {seed}: existing eval notebook differs from a fresh build — "
                "the staged canary drifted; re-audit before overwriting"
            )
        print(f"  idempotence: existing seed-{seed} build byte-identical to fresh build (OK)")

    nb_path.write_text(nb_bytes, encoding="utf-8")
    meta_path.write_text(meta_bytes, encoding="utf-8")
    print(f"  seed {seed} written: {nb_path}")


def main() -> int:
    base_nb = json.loads(STAGED_NB.read_text(encoding="utf-8"))
    base_meta = json.loads(STAGED_META.read_text(encoding="utf-8"))

    if len(base_nb["cells"]) != 18:
        fail(f"staged canary has {len(base_nb['cells'])} cells, expected 18 (canary drift)")
    # Refuse to build if the staged canary already carries an eval seed tag.
    if "DUCK_GATE_EVAL_SEED" in "".join("".join(c["source"]) for c in base_nb["cells"]):
        fail("staged canary already carries DUCK_GATE_EVAL_SEED (build from the pristine scored canary)")

    nb_s1 = _build_seed(base_nb, 1)
    nb_s2 = _build_seed(base_nb, 2)

    # PROOF 1: each seed differs from the staged canary in cell 2 only.
    _assert_nonseed_identical_to_staged(base_nb, nb_s1)
    _assert_nonseed_identical_to_staged(base_nb, nb_s2)
    # PROOF 2: the two seeds differ from each other in the seed substrings only.
    _assert_only_seed_differs(nb_s1, nb_s2)

    # Required strings survived; no cross-seed leakage.
    for seed, nb in ((1, nb_s1), (2, nb_s2)):
        allsrc = "".join("".join(c["source"]) for c in nb["cells"])
        for must in (
            f'os.environ["DUCK_GATE_EVAL_SEED"] = "{seed}"',
            f"A17-GATE-EVAL seed={seed}",
            "A17-GATE mode=readiness-gate-ab-B",      # armed banner (gate cell) intact
            ": GATE armed",
            ": GATE fired",
            "vllm_ready_latency_s=",
            "vLLM server ready",                      # boris's own readiness line intact
            "wait_vllm_ready",
        ):
            if must not in allsrc:
                fail(f"seed {seed}: required string missing after build: {must!r}")
        other = 2 if seed == 1 else 1
        if f"A17-GATE-EVAL seed={other}" in allsrc or f'DUCK_GATE_EVAL_SEED"] = "{other}"' in allsrc:
            fail(f"seed {seed}: a seed-{other} remnant leaked into the seed-{seed} notebook")

    meta = _build_meta(base_meta)

    print("gate 2-seed entry-gate eval build:")
    _write(OUT[1], nb_s1, meta, 1)
    _write(OUT[2], nb_s2, meta, 2)
    print(f"  eval slug: {EVAL_KERNEL_ID} (distinct from scored canivel/arc3-duck-gate)")
    print("  env fields byte-identical to staged gate canary / frozen fork; no extra dataset; no model_sources")
    print("  seed-1 and seed-2 differ ONLY in the DUCK_GATE_EVAL_SEED tag + banner (proven)")
    print("BUILD-GATE-EVAL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
