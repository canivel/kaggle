"""2-seed entry-gate eval build smoke — validates the gate-eval seed-1/seed-2
notebooks emitted by duck_eval/a17/build_gate_eval_2seed.py.

Mirrors boristown_gate_smoke.py (PASS/FAIL counters, section-labeled checks).
CPU only, no GPU/LLM/network. Runtime-tests the STAGED eval notebooks
(feedback_test_before_submit). Complements boristown_gate_smoke.py (which
validates the SCORED canary); this asserts the eval derivation is a clean
seed-tag-only graft on top of that scored canary.

  E1  both eval notebooks load; 18 cells; cell 2 + gate cell + run cell compile
  E2  each eval notebook differs from the STAGED SCORED canary in cell 2 ONLY
      (the additive DUCK_GATE_EVAL_SEED tag + banner); the gate cell, the run
      cell and every other cell are byte-identical -> the eval measures exactly
      the arm-B mechanism, nothing grafted onto the solver/gate surface
  E3  seed-1 and seed-2 differ ONLY in the two seed substrings (env-tag value +
      A17-GATE-EVAL banner value); reverse-substitution proves byte-identity
  E4  the gate telemetry the orchestrator greps for survives on BOTH seeds:
      "A17-GATE ... : GATE armed", "A17-GATE observed-firing
      vllm_ready_latency_s=... : GATE fired", boris's own "vLLM server ready"
  E5  metadata: fresh eval slug canivel/arc3-duck-gate-eval (distinct from the
      scored canivel/arc3-duck-gate); every env field byte-identical to the
      staged scored canary (kaggle_env_match); no model_sources; NO extra
      dataset (gate uses only requests) ; code_file points at the eval notebook
  E6  NO eval-mode/force-offline flag was grafted (the frozen-fork base runs the
      offline bench automatically when TRUE_SUBMISSION is unset — a plain BUILD
      IS the eval); assert no WARPACK_FORCE_OFFLINE_BENCH / RUN_HEAVY graft leaked

Run:  uv run python duck_eval/a17/gate_eval_2seed_smoke.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORED_NB = REPO / "notebooks" / "duckgate" / "arc3-duck-gate.ipynb"
SCORED_META = REPO / "notebooks" / "duckgate" / "kernel-metadata.json"
S1_NB = REPO / "notebooks" / "duckgate-eval-s1" / "arc3-duck-gate-eval.ipynb"
S2_NB = REPO / "notebooks" / "duckgate-eval-s2" / "arc3-duck-gate-eval.ipynb"
S1_META = REPO / "notebooks" / "duckgate-eval-s1" / "kernel-metadata.json"
S2_META = REPO / "notebooks" / "duckgate-eval-s2" / "kernel-metadata.json"

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


def code_srcs(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"]]


def find_gate_run(nb: dict):
    gate_idx = run_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        s = "".join(c["source"])
        if "A17-GATE mode=readiness-gate-ab-B" in s and "wait_vllm_ready" in s:
            gate_idx = i
        if "await bm.run(" in s:
            run_idx = i
    return gate_idx, run_idx


def main() -> int:
    scored = json.loads(SCORED_NB.read_text(encoding="utf-8"))
    s1 = json.loads(S1_NB.read_text(encoding="utf-8"))
    s2 = json.loads(S2_NB.read_text(encoding="utf-8"))

    print("E1 structure + compile (both seeds)")
    for label, nb in (("seed1", s1), ("seed2", s2)):
        check(f"{label}: 18 cells", len(nb["cells"]) == 18, f"got {len(nb['cells'])}")
        gate_idx, run_idx = find_gate_run(nb)
        check(f"{label}: gate cell found", gate_idx is not None)
        check(f"{label}: run cell found", run_idx is not None)
        check(f"{label}: gate immediately before run", gate_idx is not None and run_idx == gate_idx + 1,
              f"gate={gate_idx} run={run_idx}")
        for idx, cl in ((2, "cell2"), (gate_idx, "gate"), (run_idx, "run")):
            if idx is None:
                check(f"{label}: {cl} compiles", False, "not located")
                continue
            try:
                compile("".join(nb["cells"][idx]["source"]), cl, "exec",
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                check(f"{label}: {cl} compiles", True)
            except SyntaxError as exc:
                check(f"{label}: {cl} compiles", False, repr(exc))

    print("E2 each eval nb differs from the SCORED canary in cell 2 only")
    sc = code_srcs(scored)
    for label, nb in (("seed1", s1), ("seed2", s2)):
        ec = code_srcs(nb)
        check(f"{label}: same cell count as scored canary", len(ec) == len(sc))
        diffs = [i for i, (a, b) in enumerate(zip(sc, ec)) if a != b]
        check(f"{label}: ONLY cell 2 differs vs scored canary", diffs == [2], f"diffs {diffs}")
        # gate + run cells byte-identical to scored
        g_sc = next(s for s in sc if "wait_vllm_ready" in s and "A17-GATE mode" in s)
        r_sc = next(s for s in sc if "await bm.run(" in s)
        check(f"{label}: gate cell byte-identical to scored canary", g_sc in ec)
        check(f"{label}: run cell byte-identical to scored canary", r_sc in ec)
        # cell 2 is the scored cell 2 + the additive seed graft (scored body preserved)
        check(f"{label}: cell-2 scored body preserved (additive seed graft)",
              sc[2].rstrip() in ec[2] or "arm B of R22-D2 A/B" in ec[2])

    print("E3 seed-1 vs seed-2 differ ONLY in the seed substrings")
    c1 = code_srcs(s1)
    c2 = code_srcs(s2)
    check("same cell count across seeds", len(c1) == len(c2))
    seed_diffs = [i for i, (a, b) in enumerate(zip(c1, c2)) if a != b]
    check("only cell 2 differs across seeds", seed_diffs == [2], f"diffs {seed_diffs}")
    reverted = c2[2].replace('os.environ["DUCK_GATE_EVAL_SEED"] = "2"',
                             'os.environ["DUCK_GATE_EVAL_SEED"] = "1"').replace(
        "A17-GATE-EVAL seed=2", "A17-GATE-EVAL seed=1")
    check("reverse-substitution proves cell-2 diff is seed-only", reverted == c1[2])
    check("seed1 tag present", 'os.environ["DUCK_GATE_EVAL_SEED"] = "1"' in c1[2])
    check("seed2 tag present", 'os.environ["DUCK_GATE_EVAL_SEED"] = "2"' in c2[2])
    check("no seed2 remnant in seed1", "seed=2" not in c1[2] and 'SEED"] = "2"' not in c1[2])
    check("no seed1 remnant in seed2", "seed=1" not in c2[2] and 'SEED"] = "1"' not in c2[2])

    print("E4 gate firing telemetry survives on BOTH seeds (orchestrator grep targets)")
    for label, nb in (("seed1", s1), ("seed2", s2)):
        allsrc = "".join(code_srcs(nb))
        check(f"{label}: 'GATE armed' banner", ": GATE armed" in allsrc)
        check(f"{label}: 'GATE fired' + latency banner",
              ": GATE fired" in allsrc and "vllm_ready_latency_s=" in allsrc)
        check(f"{label}: boris 'vLLM server ready' line", "vLLM server ready" in allsrc)
        check(f"{label}: A17-GATE-EVAL seed banner", "A17-GATE-EVAL seed=" in allsrc)
        check(f"{label}: gate 180 s timeout preserved", "def wait_vllm_ready(timeout=180)" in allsrc)

    print("E5 metadata: fresh eval slug, env byte-matched to scored canary")
    for label, mp in (("seed1", S1_META), ("seed2", S2_META)):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        ref = json.loads(SCORED_META.read_text(encoding="utf-8"))
        check(f"{label}: fresh eval slug id", meta.get("id") == "canivel/arc3-duck-gate-eval")
        check(f"{label}: eval slug != scored slug", meta.get("id") != ref.get("id"))
        check(f"{label}: code_file points at eval nb", meta.get("code_file") == "arc3-duck-gate-eval.ipynb")
        check(f"{label}: no model_sources", not meta.get("model_sources"))
        for field in ("dataset_sources", "kernel_sources", "competition_sources",
                      "model_sources", "docker_image", "machine_shape",
                      "enable_gpu", "enable_tpu", "enable_internet", "is_private",
                      "language", "kernel_type", "keywords"):
            check(f"{label}: env field {field} == scored canary",
                  meta.get(field) == ref.get(field),
                  f"{meta.get(field)!r} vs {ref.get(field)!r}")
        check(f"{label}: NO extra dataset vs scored canary (no arc-war-kit)",
              meta.get("dataset_sources") == ref.get("dataset_sources")
              and not any("arc-war-kit" in d for d in meta.get("dataset_sources", [])))

    print("E6 no eval-mode/force-offline flag grafted (plain BUILD IS the eval)")
    for label, nb in (("seed1", s1), ("seed2", s2)):
        allsrc = "".join(code_srcs(nb))
        check(f"{label}: no WARPACK_FORCE_OFFLINE_BENCH graft",
              "WARPACK_FORCE_OFFLINE_BENCH" not in allsrc)
        check(f"{label}: no RUN_HEAVY graft", "RUN_HEAVY" not in allsrc)
        # the frozen-fork offline branch (TRUE_SUBMISSION False) is byte-preserved
        check(f"{label}: frozen-fork offline branch intact (_offline_games)",
              "_offline_games(" in allsrc and "if TRUE_SUBMISSION:" in allsrc)

    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
