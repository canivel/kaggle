"""boristown readiness-gate A/B canary smoke — validates the arm-B gated build.

Mirrors a17_v6_smoke.py (PASS/FAIL counters, section-labeled checks). CPU only,
no GPU/LLM, no network. Runtime-tests the STAGED notebook
notebooks/duckgate/arc3-duck-gate.ipynb built by
duck_eval/a17/build_boristown_gate_canary.py (feedback_test_before_submit).

The A/B thesis: arm B = frozen fork + the SINGLE audited functional diff from
the boristown 1.47 artifact (the vLLM readiness gate, boris cell 16), NOTHING
else. So the smoke asserts BOTH that the gate is present/correct AND that every
load-bearing frozen-fork cell is byte-unchanged (the "nothing else" half is what
makes this a clean single-variable causal test).

  S1  notebook JSON loads; 18 cells; gate cell + run cell + banner cell compile
  S2  cell 2 carries the arm-B canary banner; frozen env-detect body intact
  S3  the gate cell exists immediately BEFORE the run cell; carries the verbatim
      boristown wait_vllm_ready() polling logic (endpoint/poll/timeout) + the
      GATE armed/fired telemetry banners; no solver/package surface beyond
      requests+time
  S4  "nothing else" — every load-bearing frozen-fork code cell (install, source
      setup, benchmark restore, customization hook, the run cell itself) is
      byte-identical to the frozen fork base (arm A). Only cell 2 (+banner) and
      the inserted gate cell differ.
  S5  metadata: fresh slug id/title; every env field byte-identical to the
      frozen fork family (kaggle_env_match discipline); no model_sources; no
      new dataset/package/key
  S6  the gate cell's polling body is byte-equal to the pulled boristown cell 16
      (fork-diff provenance: the graft is the audited diff, not a re-derivation)

Run:  uv run python duck_eval/a17/boristown_gate_smoke.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "duckgate" / "arc3-duck-gate.ipynb"
META_PATH = REPO / "notebooks" / "duckgate" / "kernel-metadata.json"
FORK_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
FORK_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
BORIS_CELL16 = REPO / "runs" / "fork_diff_boristown" / "cells" / "boris_16_code.txt"

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not ok else ""))


def cell_src(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def code_cells(nb: dict) -> list[str]:
    return ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    fork = json.loads(FORK_NB.read_text(encoding="utf-8"))

    print("S1 structure + compile")
    check("18 cells", len(nb["cells"]) == 18, f"got {len(nb['cells'])}")
    # locate gate + run cells
    gate_idx = run_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c["source"])
        if "A17-GATE mode=readiness-gate-ab-B" in s and "wait_vllm_ready" in s:
            gate_idx = i
        if "await bm.run(" in s:
            run_idx = i
    check("gate cell found", gate_idx is not None)
    check("run cell found", run_idx is not None)
    for idx, label in ((2, "cell 2"), (gate_idx, "gate cell"), (run_idx, "run cell")):
        if idx is None:
            check(f"{label} compiles", False, "cell not located")
            continue
        try:
            compile(cell_src(nb, idx), label, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            check(f"{label} compiles", True)
        except SyntaxError as exc:
            check(f"{label} compiles", False, repr(exc))

    print("S2 cell 2 banner + env-detect intact")
    c2 = cell_src(nb, 2)
    check("arm-B canary banner", "A17-GATE canary mode=readiness-gate-ab-B version=1" in c2)
    check("banner names the graft provenance",
          "graft=boristown/agi-duck-harness-fast-eval#cell16" in c2)
    check("banner names arm B of R22-D2", "arm B of R22-D2 A/B" in c2)
    check("TRUE_SUBMISSION env-detect body intact",
          'TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN"' in c2)
    check("ONLY_RESET_LEVELS pin intact", 'os.environ["ONLY_RESET_LEVELS"] = "true"' in c2)

    print("S3 gate cell placement + content")
    check("gate is immediately BEFORE the run cell",
          gate_idx is not None and run_idx is not None and run_idx == gate_idx + 1,
          f"gate_idx={gate_idx} run_idx={run_idx}")
    g = cell_src(nb, gate_idx) if gate_idx is not None else ""
    check("gate polls the vLLM /v1/models endpoint",
          "http://127.0.0.1:1234/v1/models" in g)
    check("gate defines wait_vllm_ready", "def wait_vllm_ready(timeout=180)" in g)
    check("gate 5 s poll cadence", "sleep(5)" in g)
    check("gate raises if vLLM never comes up",
          'RuntimeError(' in g and "not alive before benchmark start" in g)
    check("GATE armed banner (entry gate #1)", ": GATE armed" in g)
    check("GATE fired telemetry banner (vLLM-ready latency)",
          ": GATE fired" in g and "vllm_ready_latency_s=" in g)
    check("gate calls the verbatim wait_vllm_ready() (executed path is boris body)",
          "\nwait_vllm_ready()\n" in g or "wait_vllm_ready()\nprint(" in g)
    # no solver/package surface beyond requests + time — scope to executable
    # (non-comment) lines; comments legitimately mention "solver"/"bm.run".
    exec_lines = [ln for ln in g.splitlines() if not ln.lstrip().startswith("#")]
    exec_body = "\n".join(exec_lines)
    check("no solver/bm mutation on any executable line",
          "bm." not in exec_body and "solver" not in exec_body)
    check("no new package beyond requests/time",
          all(pkg not in exec_body for pkg in ("import torch", "import vllm", "litellm", "import numpy")))

    print("S4 'nothing else' — load-bearing frozen cells byte-identical (single-variable test)")
    fork_codes = code_cells(fork)
    gate_codes = code_cells(nb)
    # The gate build inserts ONE new code cell and edits cell 2 only. Every other
    # code cell must be byte-identical to the frozen fork, in order.
    # Remove the inserted gate cell from the gated list, then compare position-wise.
    gate_codes_wo_gate = [s for i, s in enumerate(gate_codes)
                          if "A17-GATE mode=readiness-gate-ab-B" not in s]
    check("exactly one extra code cell inserted",
          len(gate_codes) == len(fork_codes) + 1, f"{len(gate_codes)} vs {len(fork_codes)}+1")
    check("gate list minus gate == frozen code-cell count",
          len(gate_codes_wo_gate) == len(fork_codes),
          f"{len(gate_codes_wo_gate)} vs {len(fork_codes)}")
    diffs = [i for i, (a, b) in enumerate(zip(gate_codes_wo_gate, fork_codes)) if a != b]
    # Only the cell-2 code cell (banner append) may differ.
    check("only ONE frozen code cell differs (cell 2 banner)", len(diffs) == 1, f"diff idxs {diffs}")
    if len(diffs) == 1:
        d = gate_codes_wo_gate[diffs[0]]
        f = fork_codes[diffs[0]]
        check("the one diff is the cell-2 banner append (frozen body preserved)",
              f.rstrip() in d and "A17-GATE canary" in d)
    # spot-check the load-bearing cells verbatim
    fork_run = next((s for s in fork_codes if "await bm.run(" in s), "")
    gate_run = next((s for s in gate_codes if "await bm.run(" in s), "")
    check("run cell (bm.run + submission mechanics) byte-identical to frozen fork",
          gate_run == fork_run and gate_run != "")
    fork_hook = next((s for s in fork_codes if "one-off changes to `bm`" in s), "")
    check("customization-hook cell byte-identical to frozen fork (no solver graft)",
          fork_hook in gate_codes)

    print("S5 metadata (fresh slug, env byte-matched to frozen family)")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    ref = json.loads(FORK_META.read_text(encoding="utf-8"))
    check("fresh slug id", meta.get("id") == "canivel/arc3-duck-gate")
    check("fresh title", meta.get("title") == "arc3-duck-gate")
    check("code_file points at the staged notebook", meta.get("code_file") == NB_PATH.name)
    check("no model_sources", not meta.get("model_sources"))
    for field in ("dataset_sources", "kernel_sources", "competition_sources",
                  "model_sources", "docker_image", "machine_shape",
                  "enable_gpu", "enable_tpu", "enable_internet", "is_private",
                  "language", "kernel_type", "keywords"):
        check(f"metadata field {field} == frozen fork", meta.get(field) == ref.get(field),
              f"{meta.get(field)!r} vs {ref.get(field)!r}")
    check("no new dataset attached vs frozen fork",
          meta.get("dataset_sources") == ref.get("dataset_sources"))

    print("S6 gate polling body == pulled boristown cell 16 (audited-diff provenance)")
    boris = BORIS_CELL16.read_text(encoding="utf-8").rstrip("\n")
    # The pulled boris cell 16 lines must all appear verbatim inside our gate cell
    # (our cell wraps it with banners; the polling function is byte-preserved).
    boris_core = "\n".join(l for l in boris.splitlines()
                           if l.strip() and not l.startswith("#"))
    present = all(line in g for line in boris_core.splitlines())
    check("every boristown cell-16 polling line present verbatim in the gate cell", present)
    check("gate endpoint matches boristown exactly",
          '"http://127.0.0.1:1234/v1/models"' in g and '"http://127.0.0.1:1234/v1/models"' in boris)

    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
