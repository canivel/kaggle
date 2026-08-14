"""B122 verdict scorer — WRITTEN AND TESTED BEFORE THE DATA LANDS (2026-08-13).

`feedback_audit_the_instrument`: audit the scorer before trusting its verdict, and fix it
BEFORE the data arrives. On 2026-08-12 four ARC gates were broken at the moment they were
needed. So this file exists now, with the v2 pre-registration (`brain122b_2026-08-13.md`
§5.2 / §5) hard-coded, and it is unit-tested against synthetic fixtures — including a
fixture that must NOT be allowed to pass.

It emits exactly one of the three sealed verdicts:

  ENVELOPE-PASS   boots, all round-trips OK, and projected actions >= 100.
                  NOT decisive on its own: the projection is an UPPER BOUND, so a pass only
                  licenses the full screen (census steps 6-7).
  ENVELOPE-FAIL   boots but misses the bar, or a format / parser / MM round-trip fails.
                  DECISIVE, and self-certifying (physics, no panel needed) — because even
                  the optimistic instrument missed.
  INFRA DEATH     never reached a measurement. NOT a reading on the brain.

Usage:  python duck_eval/a17/b122_score.py <dir-with-b122_canary.json-and-logs>
        python duck_eval/a17/b122_score.py --selftest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- SEALED CONSTANTS (must match the builder; asserted in the selftest) -----------------
ACTION_BAR = 100.0
TOK_S_27B = 192.0
TOKENS_PER_ACTION = 3168.0
WINDOW_S = 7920.0
STATIC_PRIOR_GB = 17.5          # from model.safetensors.index.json, section 3.3
GB_PER_TOKEN_27B = 27.0
SPEC_HBM_GB_S = 1792.0

PASS, FAIL, INFRA = "ENVELOPE-PASS", "ENVELOPE-FAIL", "INFRA DEATH"

# Round-trips that must all be present for the run to count as "measured at all".
REQUIRED_MARKERS = (
    "B122-CANARY models_endpoint=OK",
    "B122-CANARY tool_call_roundtrip=OK",
    "B122-CANARY mm_image_roundtrip=OK",
)
# Signatures that mean the rail broke, not the brain.
INFRA_SIGNATURES = (
    "libcudart",
    "cannot open shared object file",
    "No space left on device",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Timed out waiting for vLLM server",
    "Missing wheelhouse lock file",
    "Missing attached dataset path",
)


def score(summary: dict | None, log_text: str) -> dict:
    """Apply the sealed rules. `summary` is b122_canary.json, or None if it never got written."""
    notes: list[str] = []

    # 1. No summary => the probe never ran. Distinguish infra from a loud canary refusal.
    if not summary:
        hit = next((s for s in INFRA_SIGNATURES if s in log_text), None)
        if hit:
            return {"verdict": INFRA, "reason": f"no measurement; infra signature {hit!r} in logs",
                    "decisive": False, "notes": notes}
        if "B122-CANARY FATAL" in log_text:
            line = next((ln.strip() for ln in log_text.splitlines()
                         if "B122-CANARY FATAL" in ln), "")
            # A FATAL on the serve contract (parser / MM / template) is a real format failure.
            if any(k in line for k in ("tool-call", "MM boot probe", "served model ids")):
                return {"verdict": FAIL, "reason": f"serve-contract failure: {line[:200]}",
                        "decisive": True, "notes": notes}
            return {"verdict": INFRA, "reason": f"canary refused before measuring: {line[:200]}",
                    "decisive": False, "notes": notes}
        return {"verdict": INFRA, "reason": "no b122_canary.json and no recognised signature",
                "decisive": False, "notes": notes}

    # 2. Summary exists, but the round-trips must all have passed.
    missing = [m for m in REQUIRED_MARKERS if m not in log_text]
    if missing:
        return {"verdict": FAIL,
                "reason": f"measured, but {len(missing)} serve round-trip(s) absent: {missing}",
                "decisive": True, "notes": notes}

    projected = float(summary.get("actions_projected") or 0.0)
    agg = float(summary.get("agg_tok_s") or 0.0)
    single = float(summary.get("single_tok_s") or 0.0)

    # 3. The bar. Projection is an UPPER BOUND => only FAIL is decisive.
    if projected < ACTION_BAR:
        verdict, decisive = FAIL, True
        reason = (f"projected {projected:.1f} actions/{WINDOW_S:.0f}s < bar {ACTION_BAR:.0f} "
                  f"(agg {agg:.1f} tok/s < {ACTION_BAR * TOKENS_PER_ACTION / WINDOW_S:.1f}); "
                  f"DECISIVE because the projection is an upper bound")
    else:
        verdict, decisive = PASS, False
        reason = (f"projected {projected:.1f} actions/{WINDOW_S:.0f}s >= bar {ACTION_BAR:.0f} "
                  f"(agg {agg:.1f} tok/s); NOT decisive — upper bound; licenses the full screen only")

    # 4. Descriptive reads. None of these may change the verdict.
    if agg >= TOK_S_27B:
        notes.append(f"incumbent line CLEARED on the synthetic instrument "
                     f"({agg:.1f} >= {TOK_S_27B:.0f} tok/s) — but the 27B anchor is "
                     f"job-wallclock, so this ratio is INSTRUMENT-MISMATCHED, descriptive only")
    else:
        notes.append(f"incumbent line missed on the synthetic instrument "
                     f"({agg:.1f} < {TOK_S_27B:.0f} tok/s); note this comparison is "
                     f"instrument-mismatched in the direction that FLATTERS the 122B, "
                     f"so missing it is meaningful")

    sweep = summary.get("sweep") or []
    if sweep:
        last = sweep[-1]
        eff = float(last.get("scaling_efficiency") or 0.0)
        notes.append(f"batch scaling efficiency at n={last.get('n')} = {eff:.3f} "
                     f"(1.0 = dense ideal; low = MoE routing tax + BF16 attention path, "
                     f"i.e. the bandwidth edge NOT converting)")

    if single > 0:
        implied_lo = 0.70 * SPEC_HBM_GB_S / single
        implied_hi = SPEC_HBM_GB_S / single
        notes.append(f"implied bytes/token {implied_lo:.1f}-{implied_hi:.1f} GB (DERIVED from "
                     f"batch-1 {single:.2f} tok/s x spec {SPEC_HBM_GB_S:.0f} GB/s, not counted); "
                     f"static prior {STATIC_PRIOR_GB} GB, 27B reference {GB_PER_TOKEN_27B} GB")
        predicted = 0.80 * SPEC_HBM_GB_S / STATIC_PRIOR_GB
        if single >= 0.75 * predicted:
            notes.append(f"batch-1 {single:.1f} tok/s is consistent with the corrected "
                         f"~{STATIC_PRIOR_GB} GB/token prior (predicted ~{predicted:.0f} at 80% of spec)")
        else:
            notes.append(f"batch-1 {single:.1f} tok/s is WELL BELOW the ~{predicted:.0f} tok/s the "
                         f"corrected {STATIC_PRIOR_GB} GB/token prior predicts — the dense BF16 "
                         f"path is costing more than even the correction says")

    rt = summary.get("runtime") or {}
    if rt:
        notes.append(f"runtime: vllm={rt.get('vllm')} transformers={rt.get('transformers')} "
                     f"import vllm.lora={rt.get('vllm.lora')} (for the LoRA lane)")

    return {"verdict": verdict, "reason": reason, "decisive": decisive, "notes": notes,
            "projected_actions": projected, "agg_tok_s": agg, "single_tok_s": single}


def _load(d: Path) -> tuple[dict | None, str]:
    summary = None
    hit = next(iter(sorted(d.rglob("b122_canary.json"))), None)
    if hit:
        try:
            summary = json.loads(hit.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not parse {hit}: {exc}", file=sys.stderr)
    chunks = []
    for pattern in ("*.log", "*.txt", "**/vllm-openai-server.log"):
        for f in sorted(d.rglob(pattern)):
            try:
                chunks.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return summary, "\n".join(chunks)


def _selftest() -> int:
    ok, bad = 0, []

    def expect(name, got, want_verdict, want_decisive=None):
        nonlocal ok
        good = got["verdict"] == want_verdict and (
            want_decisive is None or got["decisive"] == want_decisive)
        if good:
            ok += 1
            print(f"  PASS {name} -> {got['verdict']}")
        else:
            bad.append(name)
            print(f"  FAIL {name} -> {got['verdict']} (want {want_verdict}) :: {got['reason']}")

    markers = "\n".join(REQUIRED_MARKERS)

    # The v1 death, verbatim, must classify as INFRA and never as a verdict on the brain.
    expect("v1 ImportError libcudart.so.13",
           score(None, "ImportError: libcudart.so.13: cannot open shared object file"),
           INFRA, False)
    expect("OOM at load", score(None, "torch.OutOfMemoryError: CUDA out of memory"), INFRA, False)
    expect("boot timeout", score(None, "TimeoutError: Timed out waiting for vLLM server at ..."),
           INFRA, False)
    expect("weights not attached",
           score(None, "RuntimeError: B122-CANARY FATAL: Qwen3.5-122B-A10B-NVFP4 not found under /kaggle/input"),
           INFRA, False)
    # A serve-contract failure IS a real envelope failure, not infra.
    expect("tool-call parser broken",
           score(None, "RuntimeError: B122-CANARY FATAL: tool-call round-trip FAILED under qwen3_coder"),
           FAIL, True)
    expect("vision path broken",
           score(None, "RuntimeError: B122-CANARY FATAL: MM boot probe returned empty content"),
           FAIL, True)

    # Below the bar => decisive FAIL. 30 tok/s => 75 actions.
    slow = {"actions_projected": 30.0 * WINDOW_S / TOKENS_PER_ACTION, "agg_tok_s": 30.0,
            "single_tok_s": 12.0, "sweep": [{"n": 28, "scaling_efficiency": 0.2}]}
    expect("30 tok/s (75 actions) below bar", score(slow, markers), FAIL, True)

    # Above the bar => PASS but explicitly NOT decisive.
    fast = {"actions_projected": 250.0 * WINDOW_S / TOKENS_PER_ACTION, "agg_tok_s": 250.0,
            "single_tok_s": 80.0, "sweep": [{"n": 28, "scaling_efficiency": 0.9}],
            "runtime": {"vllm": "0.24.0", "vllm.lora": "OK"}}
    got = score(fast, markers)
    expect("250 tok/s above bar", got, PASS, False)

    # THE FIXTURE THAT MUST NOT PASS: fast, but a round-trip never happened.
    expect("fast numbers but MM round-trip missing must NOT pass",
           score(fast, "B122-CANARY models_endpoint=OK\nB122-CANARY tool_call_roundtrip=OK"),
           FAIL, True)

    # Exactly at the bar is a pass (>=, as sealed).
    edge = {"actions_projected": 100.0, "agg_tok_s": 40.0, "single_tok_s": 20.0}
    expect("exactly 100 actions is >= bar", score(edge, markers), PASS, False)
    just_under = {"actions_projected": 99.9, "agg_tok_s": 39.9, "single_tok_s": 20.0}
    expect("99.9 actions is below bar", score(just_under, markers), FAIL, True)

    # Constants must match the builder, or the scorer is measuring a different gate.
    builder = (Path(__file__).with_name("build_b122_boot_canary.py")).read_text(encoding="utf-8")
    for name, literal in (("ACTION_BAR", "ACTION_BAR = 100.0"),
                          ("TOK_S_27B", "TOK_S_27B = 192.0"),
                          ("WINDOW_S", "WINDOW_S = 7920.0"),
                          ("SPEC_HBM_GB_S", "SPEC_HBM_GB_S = 1792.0")):
        if literal in builder:
            ok += 1
            print(f"  PASS constant {name} matches the builder")
        else:
            bad.append(f"constant {name}")
            print(f"  FAIL constant {name} does not match the builder")
    if abs(TOK_S_27B * WINDOW_S / 480.0 - TOKENS_PER_ACTION) > 1e-9:
        bad.append("tokens-per-action derivation")
    else:
        ok += 1
        print("  PASS tokens-per-action re-derives from the frozen 27B anchor")

    print(f"\nselftest: {ok} passed / {len(bad)} failed")
    for b in bad:
        print("  FAILED:", b)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        print("B122 SCORER SELFTEST (run before the data exists)")
        return _selftest()
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    summary, log_text = _load(d)
    result = score(summary, log_text)
    print("=" * 78)
    print("B122 VERDICT:", result["verdict"], "(decisive)" if result["decisive"] else "(not decisive)")
    print("reason:", result["reason"])
    for n in result.get("notes", []):
        print("  -", n)
    print("=" * 78)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
