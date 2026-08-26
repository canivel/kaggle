"""boristown vLLM readiness-gate A/B canary — build script (2026-07-28).

Panel R22 directive D2 (5/5 unanimous, carried from R21 #1): schedule the
boristown readiness-gate A/B this week by replacing frozen-filler scored draws.

WHAT THIS BUILDS (fork-never-build discipline):
  Arm B of the A/B. Base = the FROZEN FORK notebook
  (notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb,
  kernel canivel/arc3-duck-repro v3 lineage — the exact bytes we score fillers
  on). The ONLY graft is the single audited functional diff from the boristown
  1.47 artifact (runs/fork_diff_boristown, boris cell 16): a standalone
  ~25-line `wait_vllm_ready()` cell inserted immediately BEFORE the benchmark
  run cell. Everything else is byte-identical to the frozen fork (12/22 boris
  cells are md5-identical to ours, including every load-bearing cell; the fork
  diff memo established boris cell 16 is the sole functional delta).

  This is the sentinel build pattern (duck_eval/sentinel): fork base + minimal
  anchored graft + a banner line identifying mode/version. Unlike the sentinel
  (which grafts a solver-touching patch module into the customization hook cell
  12), the readiness gate is a STANDALONE cell that only polls localhost
  http://127.0.0.1:1234/v1/models before bm.run — zero interaction risk, no
  solver surface, no new packages/datasets/keys (requests is already in the
  image; the frozen fork's own gateway wait uses urllib, this one uses requests
  exactly as boristown ships it).

DELTAS (exactly three, all anchor-exact, all idempotence-guarded):
  1. cell 2: append one banner print identifying mode=readiness-gate-ab-B and
     the gate provenance (boristown/agi-duck-harness-fast-eval cell 16). The
     frozen fork's cell 2 ends with `print(f"taaf.kaggle: TRUE_SUBMISSION=...")`
     — we append immediately after it. No behavioural change; identity/telemetry
     only, so the built kernel self-identifies in its log (feedback: banner line
     identifying mode/version).
  2. NEW cell inserted immediately before the run cell (the cell containing
     `await bm.run(`): the verbatim boristown `wait_vllm_ready()` gate, with a
     one-line GATE-armed banner grafted in so the log shows the gate firing
     (poll observed + vLLM-ready latency) per the A22 entry gate #1.
  3. metadata: fresh id/title (fresh slug per feedback_fresh_kernel_slug), rest
     byte-identical to the frozen-fork family (arc3-duck-repro), which the fork
     diff memo already proved is field-for-field identical to boristown's env.

The gate cell body is the EXACT boristown cell-16 source (verified against
runs/fork_diff_boristown/cells/boris_16_code.txt) plus a banner; if that file
drifts from the constant below the script raises.

Idempotence: the build is deterministic-from-pristine-base — the frozen fork
notebook is never mutated, so re-running regenerates a byte-identical output.
The script asserts this (if an output already exists, the freshly-built bytes
must match it, else it raises: that would mean the base or the graft drifted).
Every rewrite anchor must match EXACTLY once against the pristine base.

Run:  uv run python duck_eval/a17/build_boristown_gate_canary.py
NO kernel push. NO submission-queue change. $0 cloud. Build-rail only.
"""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORK_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
FORK_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
BORIS_CELL16 = REPO / "runs" / "fork_diff_boristown" / "cells" / "boris_16_code.txt"

OUT_DIR = REPO / "notebooks" / "duckgate"
OUT_NB = OUT_DIR / "arc3-duck-gate.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "canivel/arc3-duck-gate"
KERNEL_TITLE = "arc3-duck-gate"

# The gate cell body EXACTLY as boristown ships it (boris cell 16). Verified at
# build time against runs/fork_diff_boristown/cells/boris_16_code.txt.
BORIS_GATE_SRC = '''# Minimal vLLM health check before benchmark execution
import time
import requests

def wait_vllm_ready(timeout=180):
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(
                "http://127.0.0.1:1234/v1/models",
                timeout=5
            )

            if r.status_code == 200:
                print("vLLM server ready")
                return True

        except Exception:
            pass

        time.sleep(5)

    raise RuntimeError(
        "vLLM server is not alive before benchmark start"
    )


wait_vllm_ready()'''

# Our grafted gate cell = boristown's gate + a GATE-armed/observed banner so the
# kernel log shows the gate firing (A22 entry gate #1: "log shows the gate
# observed firing — poll count + vLLM ready latency"). We wrap the boris body:
# the banner strings are additive; the polling logic is byte-identical to boris.
GATE_CELL_SRC = '''# READINESS-GATE A/B — arm B graft (boristown/agi-duck-harness-fast-eval cell 16).
# The SINGLE audited functional diff vs the frozen fork: a vLLM health check that
# runs immediately before the benchmark cell. Closes the startup race where the
# frozen fork waits for the gateway (_wait_for_gateway, 600 s) but never for the
# vLLM server that the setup commands launch async. Only polls localhost; no new
# packages (requests is already in the image). The wait_vllm_ready() body below
# is BYTE-IDENTICAL to boristown cell 16 and is the code that actually executes;
# the A17-GATE banners around it are purely additive telemetry (entry gate #1:
# armed banner + observed-firing latency), and never change the polling logic.
print("A17-GATE mode=readiness-gate-ab-B version=1 "
      "graft=boristown/agi-duck-harness-fast-eval#cell16 "
      "endpoint=http://127.0.0.1:1234/v1/models poll=5s timeout=180s : GATE armed",
      flush=True)

# Minimal vLLM health check before benchmark execution
import time
import requests

def wait_vllm_ready(timeout=180):
    start = time.time()

    while time.time() - start < timeout:
        try:
            r = requests.get(
                "http://127.0.0.1:1234/v1/models",
                timeout=5
            )

            if r.status_code == 200:
                print("vLLM server ready")
                return True

        except Exception:
            pass

        time.sleep(5)

    raise RuntimeError(
        "vLLM server is not alive before benchmark start"
    )


# Observed-firing telemetry (A22 entry gate #1): time the verbatim boris call so
# the log carries the vLLM-ready latency. wait_vllm_ready() is boristown's exact
# function; it raises if vLLM never comes up. No poll-count instrumentation is
# added inside it (that would fork the audited body); latency alone evidences the
# gate fired, and "vLLM server ready" is boris's own greppable readiness line.
_gate_t0 = time.time()
wait_vllm_ready()
print(f"A17-GATE observed-firing vllm_ready_latency_s={time.time() - _gate_t0:.1f} "
      f": GATE fired", flush=True)'''

# cell-2 banner append: anchor is the frozen fork's final line of cell 2.
CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
CELL2_BANNER = (
    'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")\n'
    'print("A17-GATE canary mode=readiness-gate-ab-B version=1 '
    'base=canivel/arc3-duck-repro(frozen) '
    'graft=boristown/agi-duck-harness-fast-eval#cell16 (vLLM readiness gate, '
    'sole audited functional diff) : arm B of R22-D2 A/B", flush=True)'
)

# Anchor that identifies the run cell (the bm.run cell) so we insert the gate
# immediately before it.
RUN_CELL_ANCHOR = "await bm.run("


def _compile(src: str, label: str) -> None:
    compile(src, label, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def main() -> int:
    # 0. Verify the pinned gate body still matches the pulled boris cell 16 byte-for-byte.
    boris_disk = BORIS_CELL16.read_text(encoding="utf-8").rstrip("\n")
    if boris_disk != BORIS_GATE_SRC:
        raise SystemExit(
            "FATAL: pinned BORIS_GATE_SRC no longer matches "
            f"{BORIS_CELL16} — re-audit the fork diff before rebuilding"
        )

    nb = json.loads(FORK_NB.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 17:
        raise SystemExit(f"FATAL: frozen fork has {len(nb['cells'])} cells, expected 17 (fork drift)")

    # Idempotence guard: refuse if already built.
    joined = "".join("".join(c["source"]) for c in nb["cells"])
    if "A17-GATE" in joined:
        raise SystemExit("FATAL: base notebook already carries A17-GATE markers (idempotence guard)")

    # --- delta 1: cell 2 banner append (anchor-exact, once) ---
    c2 = nb["cells"][2]
    c2_src = "".join(c2["source"])
    if c2_src.count(CELL2_ANCHOR) != 1:
        raise SystemExit(
            f"FATAL cell 2: TRUE_SUBMISSION banner anchor matched "
            f"{c2_src.count(CELL2_ANCHOR)} times (want 1)"
        )
    c2_new = c2_src.replace(CELL2_ANCHOR, CELL2_BANNER)
    _compile(c2_new, "cell2")
    c2["source"] = c2_new.splitlines(keepends=True)
    c2["outputs"] = []
    c2["execution_count"] = None

    # --- delta 2: insert gate cell immediately before the run cell ---
    run_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and RUN_CELL_ANCHOR in "".join(c["source"]):
            if run_idx is not None:
                raise SystemExit("FATAL: multiple candidate run cells (bm.run) — ambiguous anchor")
            run_idx = i
    if run_idx is None:
        raise SystemExit(f"FATAL: no run cell containing {RUN_CELL_ANCHOR!r} found")

    _compile(GATE_CELL_SRC, "gate_cell")
    template = copy.deepcopy(nb["cells"][run_idx])
    gate_cell = {
        "cell_type": "code",
        "metadata": template.get("metadata", {}),
        "source": GATE_CELL_SRC.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }
    nb["cells"].insert(run_idx, gate_cell)

    if len(nb["cells"]) != 18:
        raise SystemExit(f"FATAL: post-insert cell count {len(nb['cells'])} != 18")

    # --- delta 3: metadata (fresh slug; env fields byte-identical to frozen fork) ---
    meta = json.loads(FORK_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["code_file"] = OUT_NB.name
    # Assert env-match discipline: every field except identity is unchanged.
    ref = json.loads(FORK_META.read_text(encoding="utf-8"))
    for field in ("dataset_sources", "kernel_sources", "competition_sources",
                  "model_sources", "docker_image", "machine_shape",
                  "enable_gpu", "enable_tpu", "enable_internet", "is_private",
                  "language", "kernel_type", "keywords"):
        if meta.get(field) != ref.get(field):
            raise SystemExit(f"FATAL: metadata field {field} drifted from frozen fork")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_bytes = json.dumps(nb, indent=1)
    meta_bytes = json.dumps(meta, indent=2) + "\n"

    # Deterministic-from-base idempotence: if a prior build exists, the fresh
    # bytes must match it exactly. A mismatch means the pristine base or the
    # pinned graft drifted since the last build — raise rather than silently
    # overwrite (mirrors the anchor-exact discipline of the v5/v6 builders).
    if OUT_NB.is_file():
        prev = OUT_NB.read_text(encoding="utf-8")
        if prev != nb_bytes:
            raise SystemExit(
                "FATAL: existing gate canary differs from a fresh build — "
                "the frozen base or the pinned boris cell 16 graft drifted; "
                "re-audit before overwriting"
            )
        print("idempotence: existing build is byte-identical to fresh build (OK)")

    OUT_NB.write_text(nb_bytes, encoding="utf-8")
    OUT_META.write_text(meta_bytes, encoding="utf-8")

    print(f"gate canary written: {OUT_NB}")
    print(f"  base: {FORK_NB.name} (frozen fork, 17 cells) -> 18 cells")
    print(f"  gate cell inserted at index {run_idx} (immediately before run cell at new index {run_idx + 1})")
    print(f"  cell 2 banner appended; metadata id={KERNEL_ID} (fresh slug, env fields = frozen fork)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
