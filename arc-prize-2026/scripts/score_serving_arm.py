"""Sealed scorer for the A24 serving arm.  WRITTEN AND SELFTESTED BEFORE THE DATA LANDED.

Reads the bands from learnings/war_room/serving_arm_prereg_2026-08-31.md (sealed
2026-08-31 08:35 EDT) and applies them mechanically.

feedback_audit_the_instrument, in the order the memory names them:
  * the bands are constants here, fixed pre-data -- changing them after the pull is
    indistinguishable from tuning for the verdict you want;
  * GATE 0 (did the treatment fire?) returns BEFORE the effect size is computed, so a
    queue decision cannot be contaminated by having seen the score;
  * missing input is ERROR, never a silent 0;
  * every branch is exercised against a synthetic fixture by --selftest, including a
    synthetic HEALTHY fixture (the 08-20 arm-mismatch lesson).

Usage:
    uv run python scripts/score_serving_arm.py --pull runs/kernel_pulls/serving_mtp3_v1
    uv run python scripts/score_serving_arm.py --selftest
"""
from __future__ import annotations

import argparse
import json
import io
import re
import sys
import tempfile
from pathlib import Path

# ------------------------------------------------------------------ sealed constants
NULL_MEAN = 266.28   # tok/s, n=3: q38_field_v1 263.61, q38graft_v1 277.66, seed_0829 257.58
NULL_SD = 10.31
FIRES_AT = 297.21    # mean + 3 sd  (+11.6%)
INCONCLUSIVE_AT = 276.60  # mean + 1 sd
HARM_AT = 235.35     # mean - 3 sd
REQUIRED_FLAGS = ["--async-scheduling", "--kv-cache-dtype", "--speculative-config"]
REQUIRED_MAX_LEN = "262144"
EXPECTED_GAMES = 25

NULL_TOKENS = (2119156, 87510)
NULL_ACTIONS = (1446, 194)


def parse_summary(path: Path) -> dict:
    """Parse the taaf summary.txt.  Any missing field is an ERROR, never a zero."""
    if not path.is_file():
        raise FileNotFoundError(f"summary.txt not found at {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = {
        "games": r"^games:\s+(\d+)",
        "mean_score": r"^mean score:\s+([\d.]+)",
        "median_score": r"^median score:\s+([\d.]+)",
        "total_actions": r"^total actions:\s+(\d+)",
        "total_tokens": r"^total tokens:\s+(\d+)",
        "tok_per_s": r"^generated tokens/sec:\s+([\d.]+)",
    }
    out: dict = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text, re.M)
        if match is None:
            raise ValueError(f"summary.txt is missing required field '{key}'")
        raw = match.group(1)
        out[key] = float(raw) if "." in raw else int(raw)
    return out


def gate0(pull: Path) -> dict:
    """Fireability.  Computed and reported BEFORE any effect size is read."""
    record_path = pull / "arc3_serving_arm.json"
    if not record_path.is_file():
        return {"passed": False, "state": "NO_RECORD",
                "detail": "arc3_serving_arm.json absent -- the arm cell did not run; "
                          "no throughput claim may be made from this artifact"}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    state = record.get("state")
    flags = sorted(record.get("flags_present") or [])
    max_len = record.get("max_model_len")
    ok = (state == "ARMED" and flags == REQUIRED_FLAGS and max_len == REQUIRED_MAX_LEN)
    return {
        "passed": ok,
        "state": state,
        "flags_present": flags,
        "max_model_len": max_len,
        "detail": ("armed launch confirmed in the child argv" if ok else
                   f"state={state} flags={flags} max_model_len={max_len} -- the four-flag "
                   "launch did not run; this artifact is a FLOOR draw"),
    }


def primary(tok_per_s: float) -> tuple[str, str]:
    if tok_per_s > FIRES_AT:
        return "FIRES", f"{tok_per_s:.2f} > {FIRES_AT} (mean+3sd)"
    if tok_per_s < HARM_AT:
        return "HARM", f"{tok_per_s:.2f} < {HARM_AT} (mean-3sd)"
    if tok_per_s < INCONCLUSIVE_AT:
        return "REFUTED", f"{tok_per_s:.2f} < {INCONCLUSIVE_AT} (mean+1sd) -- inside the historical range"
    return "INCONCLUSIVE", f"{INCONCLUSIVE_AT} <= {tok_per_s:.2f} <= {FIRES_AT}"


def head_decision(gate: dict, verdict: str | None) -> str:
    """The sealed submit rule, table-for-table from the prereg."""
    if not gate["passed"]:
        return "FIELD FLOOR (canivel/arc3-q38-field-eval)"
    if verdict == "FIRES":
        return "canivel/arc3-serving-mtp3 v1"
    return "FIELD FLOOR (canivel/arc3-q38-field-eval)"


def score(pull: Path) -> dict:
    gate = gate0(pull)
    result: dict = {"pull": str(pull), "gate0": gate}

    summary = parse_summary(pull / "summary.txt")
    result["summary"] = summary

    if summary["games"] != EXPECTED_GAMES:
        result["verdict"] = "VOID"
        result["reason"] = (f"offline eval shape is {summary['games']} games, not "
                            f"{EXPECTED_GAMES}; the sealed null is not comparable")
        result["head"] = "FIELD FLOOR (canivel/arc3-q38-field-eval)"
        return result

    if not gate["passed"]:
        result["verdict"] = "VOID"
        result["reason"] = "GATE 0 failed: " + gate["detail"]
        result["head"] = head_decision(gate, None)
        return result

    verdict, detail = primary(summary["tok_per_s"])
    result["verdict"] = verdict
    result["reason"] = detail
    result["z_tok_per_s"] = round((summary["tok_per_s"] - NULL_MEAN) / NULL_SD, 2)
    result["pct_vs_null"] = round(100.0 * (summary["tok_per_s"] / NULL_MEAN - 1.0), 1)
    result["secondary"] = {
        "total_tokens": summary["total_tokens"],
        "z_total_tokens": round((summary["total_tokens"] - NULL_TOKENS[0]) / NULL_TOKENS[1], 2),
        "total_actions": summary["total_actions"],
        "z_total_actions": round((summary["total_actions"] - NULL_ACTIONS[0]) / NULL_ACTIONS[1], 2),
        "mean_score_NOT_GATED": summary["mean_score"],
    }
    result["head"] = head_decision(gate, verdict)
    return result


# ------------------------------------------------------------------------- selftest
def _fixture(tmp: Path, *, state="ARMED", tok=310.0, games=25, record=True) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    if record:
        (tmp / "arc3_serving_arm.json").write_text(json.dumps({
            "arm": "a24-mtp3-async-fp8kv-262144", "state": state,
            "flags_present": REQUIRED_FLAGS if state == "ARMED" else [],
            "max_model_len": REQUIRED_MAX_LEN if state == "ARMED" else "65536",
        }), encoding="utf-8")
    (tmp / "summary.txt").write_text(
        "benchmark: anim-20260807-anim-25g-p1\nsolver:    duck-harness\n"
        f"games:     {games}\npasses:    1\nruns:      {games} (won: 0)\n"
        "mean score:    5.10\nmedian score:  1.70\n"
        "total actions: 1700\ntotal tokens:  2500000\n"
        f"generated tokens/sec: {tok} (job wallclock)\n", encoding="utf-8")
    return tmp


def selftest() -> int:
    root = Path(tempfile.mkdtemp())
    cases = [
        ("HEALTHY/FIRES", dict(state="ARMED", tok=310.0), "FIRES", "canivel/arc3-serving-mtp3 v1"),
        ("HEALTHY/INCONCLUSIVE", dict(state="ARMED", tok=285.0), "INCONCLUSIVE", "FIELD FLOOR (canivel/arc3-q38-field-eval)"),
        ("HEALTHY/REFUTED", dict(state="ARMED", tok=264.0), "REFUTED", "FIELD FLOOR (canivel/arc3-q38-field-eval)"),
        ("HEALTHY/HARM", dict(state="ARMED", tok=200.0), "HARM", "FIELD FLOOR (canivel/arc3-q38-field-eval)"),
        ("FALLBACK", dict(state="FALLBACK", tok=310.0), "VOID", "FIELD FLOOR (canivel/arc3-q38-field-eval)"),
        ("WRONG SHAPE", dict(state="ARMED", tok=310.0, games=4), "VOID", "FIELD FLOOR (canivel/arc3-q38-field-eval)"),
    ]
    fails = []
    for i, (name, kwargs, want_verdict, want_head) in enumerate(cases):
        res = score(_fixture(root / f"c{i}", **kwargs))
        ok = res["verdict"] == want_verdict and res["head"] == want_head
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: verdict={res['verdict']} head={res['head']}")
        if not ok:
            fails.append(name)

    # a missing record must VOID, and a missing summary must ERROR (never a silent 0)
    res = score(_fixture(root / "c9", record=False))
    ok = res["verdict"] == "VOID" and res["gate0"]["state"] == "NO_RECORD"
    print(f"[{'PASS' if ok else 'FAIL'}] NO RECORD: verdict={res['verdict']}")
    if not ok:
        fails.append("NO RECORD")

    empty = root / "empty"
    empty.mkdir()
    try:
        score(empty)
        print("[FAIL] MISSING SUMMARY: returned instead of raising")
        fails.append("MISSING SUMMARY")
    except FileNotFoundError:
        print("[PASS] MISSING SUMMARY: raises FileNotFoundError")

    print(f"\n{len(cases) + 2 - len(fails)} PASS / {len(fails)} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.pull:
        ap.error("--pull is required unless --selftest")

    result = score(Path(args.pull))
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        io.open(args.out, "w", encoding="utf-8").write(text + "\n")
    print(f"\nGATE 0 : {'PASS' if result['gate0']['passed'] else 'FAIL'} -- {result['gate0']['detail']}")
    print(f"VERDICT: {result['verdict']} -- {result.get('reason')}")
    print(f"HEAD   : {result['head']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
