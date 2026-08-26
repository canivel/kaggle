"""LORA SERVE CANARY verdict scorer — WRITTEN AND TESTED BEFORE THE DATA LANDS (2026-08-14).

`feedback_audit_the_instrument`: audit the scorer BEFORE trusting its verdict, and fix it
BEFORE the data arrives. Three instrument defects in three days (stale ledger, an untested
scorer that would have falsely killed a live arm, preflight BLOCKing a healthy family) — all
silent, all in our favour. So this file exists now, with the design of
`duck_eval/lora/build_lora_serve_canary.py` hard-coded, and it is unit-tested against
synthetic fixtures — including fixtures that MUST NOT be allowed to pass.

THE SEALED TRUTH TABLE (verbatim from the builder's docstring):

    | noop==base | probe!=base | verdict                                                  |
    |------------|-------------|----------------------------------------------------------|
    | yes        | yes         | loaded AND applied.                     SERVE-PASS       |
    | yes        | no          | SILENTLY IGNORED - the exact failure Tufa's own          |
    |            |             | vllm_runtime_lora_guard exists to catch. SERVE-FAIL      |
    | no         | *           | a zero-delta adapter changed the output =>               |
    |            |             | numerically unsound.                     SERVE-FAIL      |

and one verdict that is NOT a reading on LoRA at all:

    INFRA DEATH   the run never reached the differential — CUDA / OOM / disk / boot
                  timeout, or the adapter dataset was not mounted (Kaggle drops an
                  unattachable dataset SILENTLY: `feedback_kaggle_model_attach`).
                  A retry, never a verdict.

PRECEDENCE (sealed; the scorer applies these in order and never reorders them):
  1. An explicit vLLM REFUSAL to serve the adapter (unsupported quantization, rank,
     peft feature) is a DECISIVE SERVE-FAIL: it answers this canary's question directly.
  2. Otherwise, an infra signature with no differential evidence is INFRA DEATH.
  3. Otherwise the differential truth table decides, read from the STRUCTURED line
     (`noop_identical_to_base=... probe_differs_from_base=...`), NOT from the
     `differential=PASS` banner. A banner is a string any code can print; the booleans
     are the evidence. If the two disagree, that is a CONTRADICTION and never a PASS.
  4. A SERVE-PASS additionally requires every serve round-trip marker to be present.
  5. Throughput is scored on a SEPARATE axis and is recomputed from the raw tok/s —
     the notebook's self-reported verdict is never trusted, only cross-checked.

SCOPE WARNING, carried in every verdict: this canary runs on the SCORED wheelhouse,
vLLM 0.19.0. A result here does NOT transfer to vLLM 0.24.0 (the b122 ENVELOPE-PASS
swap). That is why it is sequenced after the 122B verdict.

Usage:  python duck_eval/lora/lora_serve_score.py <dir-with-lora_canary.json-and-logs>
        python duck_eval/lora/lora_serve_score.py --selftest
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --- SEALED CONSTANTS (must match the builder; asserted in the selftest) -----------------
ACTION_BAR = 100.0
TOK_S_27B = 192.0
TOKENS_PER_ACTION = 3168.0
WINDOW_S = 7920.0
LORA_RANK = 16
ADAPTER_DS = "canivel/arc3-lora-probe-adapters"
NOOP_NAME = "arc3-noop"
PROBE_NAME = "arc3-probe"
VLLM_SCOPE = "0.19.0"
# sha256[:16] of what we built and published; the kernel re-verifies these at runtime.
ADAPTER_SHA16 = {"lora-noop": "d777d4c7a7ebec85", "lora-probe": "d7d6918d01ae67f6"}
ADAPTER_BYTES = 41962184

PASS, FAIL, INFRA = "SERVE-PASS", "SERVE-FAIL", "INFRA DEATH"

# Round-trips that must all be present for a PASS to be allowed.
REQUIRED_MARKERS = (
    "LORA-CANARY models_endpoint=OK",
    "LORA-CANARY tool_call_roundtrip=OK",
    "LORA-CANARY mm_image_roundtrip=OK",
)
DIFFERENTIAL_RE = re.compile(
    r"LORA-CANARY differential noop_identical_to_base=(\w+) "
    r"probe_differs_from_base=(\w+) first_divergent_token_index=(\S+)")
DIFFERENTIAL_BANNER = "LORA-CANARY differential=PASS"

# The rail broke, not the LoRA path.
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
# vLLM said, explicitly, that it will not serve this adapter. DECISIVE.
LORA_REFUSAL_SIGNATURES = (
    "LoRA is not supported with",
    "LoRA is not enabled",
    "does not support LoRA",
    "is unsupported with LoRA",
    "not support LoRA yet",
    "max_lora_rank",
    "Unsupported LoRA weight",
    "while LoRA is enabled",
)
# FATAL lines that mean the artifact/staging was wrong, i.e. a retry, not a verdict.
STAGING_FATALS = (
    "adapter dataset not mounted at",
    "adapter_config.json under",
    "the dataset push did not ship what we built",
    "LORA-CANARY FATAL: lora-noop rank",
    "LORA-CANARY FATAL: lora-probe rank",
)
# FATAL lines that ARE a verdict on the serve path.
SERVE_FATALS = (
    ("/v1/models", "vLLM did not register both adapters"),
    ("a ZERO-delta adapter changed the output", "zero-delta adapter perturbed the output"),
    ("a NON-ZERO adapter did not change the output", "adapter SILENTLY IGNORED"),
    ("tool-call round-trip FAILED", "tool-call round-trip failed with LoRA enabled"),
    ("MM probe returned empty content", "vision path does not survive --enable-lora"),
)

_SCOPE = (f"scoped to vLLM {VLLM_SCOPE} (the scored wheelhouse); does NOT transfer to 0.24.0")


def _first_line(log_text: str, needle: str) -> str:
    return next((ln.strip() for ln in log_text.splitlines() if needle in ln), "")


def _parse_differential(log_text: str) -> tuple[bool | None, bool | None, str | None, int]:
    hits = DIFFERENTIAL_RE.findall(log_text)
    if not hits:
        return None, None, None, 0
    noop_s, probe_s, idx = hits[-1]
    to_bool = {"True": True, "False": False}
    return to_bool.get(noop_s), to_bool.get(probe_s), idx, len(hits)


def score(summary: dict | None, log_text: str) -> dict:
    """Apply the sealed rules. `summary` is lora_canary.json, or None if never written
    (the differential raises BEFORE the throughput probe, so a serve failure legitimately
    produces no JSON — the log markers are the primary evidence, by design)."""
    notes: list[str] = [_SCOPE]
    noop_same, probe_diff, div_idx, n_diff_lines = _parse_differential(log_text)
    have_differential = noop_same is not None and probe_diff is not None
    if n_diff_lines > 1:
        notes.append(f"{n_diff_lines} differential lines in the log — the LAST was scored")

    # 1. An explicit vLLM refusal answers the question directly. Decisive.
    if not have_differential:
        refusal = next((s for s in LORA_REFUSAL_SIGNATURES if s in log_text), None)
        if refusal:
            return {"verdict": FAIL, "decisive": True,
                    "reason": f"vLLM REFUSED the adapter ({refusal!r} in the log): "
                              f"{_first_line(log_text, refusal)[:200]}",
                    "notes": notes + ["a refusal IS the answer for this stack; it is not infra"]}

    # 2. Staging / infra deaths, in the absence of any differential evidence.
    if not have_differential:
        staging = next((s for s in STAGING_FATALS if s in log_text), None)
        if staging:
            extra = ("THE 08-13 NEAR-MISS: Kaggle drops an unattachable dataset SILENTLY "
                     "(feedback_kaggle_model_attach). Re-verify the dataset, then re-push."
                     if "not mounted" in staging or "adapter_config.json under" in staging
                     else "the published artifact is not what we built; re-push the dataset")
            return {"verdict": INFRA, "decisive": False,
                    "reason": f"canary refused before measuring: "
                              f"{_first_line(log_text, staging)[:200]}",
                    "notes": notes + [extra]}
        serve_fatal = next(((k, why) for k, why in SERVE_FATALS
                            if k in log_text and "LORA-CANARY FATAL" in log_text), None)
        if serve_fatal:
            return {"verdict": FAIL, "decisive": True,
                    "reason": f"serve-contract failure: {serve_fatal[1]} :: "
                              f"{_first_line(log_text, serve_fatal[0])[:200]}",
                    "notes": notes}
        hit = next((s for s in INFRA_SIGNATURES if s in log_text), None)
        if hit:
            return {"verdict": INFRA, "decisive": False,
                    "reason": f"no measurement; infra signature {hit!r} in the logs",
                    "notes": notes}
        return {"verdict": INFRA, "decisive": False,
                "reason": "no differential line and no recognised signature — the server "
                          "never reached the measurement",
                "notes": notes}

    # 3. The differential truth table, read from the BOOLEANS.
    banner = DIFFERENTIAL_BANNER in log_text
    if not noop_same:
        return {"verdict": FAIL, "decisive": True,
                "reason": f"noop (B=0) is NOT token-identical to the base — a zero-delta "
                          f"adapter changed the output, so the LoRA path is NUMERICALLY "
                          f"UNSOUND, not merely inactive (first divergence at token "
                          f"{div_idx})",
                "notes": notes + (["CONTRADICTION: the log also carries the "
                                   "differential=PASS banner"] if banner else []),
                "noop_identical": noop_same, "probe_differs": probe_diff}
    if not probe_diff:
        return {"verdict": FAIL, "decisive": True,
                "reason": "probe (B~1e-3) did NOT change the output — the adapter is being "
                          "SILENTLY IGNORED. This is the exact failure class Tufa built "
                          "vllm_runtime_lora_guard for, and after a real training run it "
                          "would have read as 'LoRA did not help'",
                "notes": notes + (["CONTRADICTION: the log also carries the "
                                   "differential=PASS banner"] if banner else []),
                "noop_identical": noop_same, "probe_differs": probe_diff}
    if not banner:
        return {"verdict": FAIL, "decisive": True,
                "reason": "the booleans say noop==base and probe!=base, but the canary never "
                          "printed differential=PASS — the run did not complete the assert "
                          "block, so the evidence is INCONSISTENT and is not scored as a pass",
                "notes": notes, "noop_identical": noop_same, "probe_differs": probe_diff}

    # 4. A PASS additionally requires every serve round-trip.
    missing = [m for m in REQUIRED_MARKERS if m not in log_text]
    if missing:
        return {"verdict": FAIL, "decisive": True,
                "reason": f"differential passed, but {len(missing)} serve round-trip(s) never "
                          f"happened: {missing} — the serve contract is not proven",
                "notes": notes, "noop_identical": True, "probe_differs": True}

    result = {"verdict": PASS, "decisive": True,
              "reason": f"{NOOP_NAME} is token-identical to the base AND {PROBE_NAME} differs "
                        f"(first divergence at token {div_idx}); /v1/models, tool-call and MM "
                        f"round-trips all OK on the ADAPTER — the adapter is LOADED and APPLIED",
              "notes": notes, "noop_identical": True, "probe_differs": True}
    if div_idx in (None, "None"):
        result["notes"].append("probe differs but no divergent index was reported — the two "
                               "token lists differ only in LENGTH; descriptive, not scored")

    # 5. Throughput: a SEPARATE axis, recomputed from raw tok/s.
    result.update(_throughput(summary, result["notes"]))
    return result


def _throughput(summary: dict | None, notes: list[str]) -> dict:
    if not summary:
        notes.append("no lora_canary.json — the throughput axis is UNMEASURED (the serve "
                     "verdict above stands on its own)")
        return {"throughput_verdict": "UNMEASURED"}
    tp = summary.get("throughput") or {}
    adapter = tp.get("adapter") or {}
    base = tp.get("base") or {}
    a_tok = float(adapter.get("tok_s") or 0.0)
    b_tok = float(base.get("tok_s") or 0.0)
    if a_tok <= 0:
        notes.append("throughput block present but adapter tok/s is 0 — UNMEASURED")
        return {"throughput_verdict": "UNMEASURED"}
    actions = a_tok * WINDOW_S / TOKENS_PER_ACTION
    tax = 1.0 - a_tok / b_tok if b_tok > 0 else None
    verdict = "THROUGHPUT-PASS" if actions >= ACTION_BAR else "THROUGHPUT-FAIL"
    # Never trust the notebook's own arithmetic: recompute and cross-check.
    claimed = summary.get("verdict")
    if claimed in ("PASS", "FAIL") and claimed != verdict.split("-", 1)[1]:
        notes.append(f"MISMATCH: the kernel self-reported throughput verdict {claimed!r} but "
                     f"the recomputation from raw tok/s says {verdict!r} — the recomputation "
                     f"is authoritative")
    reported = float(adapter.get("actions_per_window") or 0.0)
    if reported and abs(reported - actions) > 1.0:
        notes.append(f"MISMATCH: kernel reported {reported:.0f} actions/window, recomputation "
                     f"gives {actions:.0f}")
    notes.append(f"LoRA throughput tax = {100 * tax:.1f}% (base {b_tok:.1f} -> adapter "
                 f"{a_tok:.1f} tok/s)" if tax is not None else
                 "base throughput missing — the LoRA tax cannot be computed")
    notes.append(f"projected {actions:.0f} actions/{WINDOW_S:.0f}s vs bar {ACTION_BAR:.0f}; "
                 f"this is a SYNTHETIC-load projection using the 27B tokens-per-action "
                 f"constant, so it is an upper bound and only a FAIL is decisive")
    return {"throughput_verdict": verdict, "adapter_tok_s": a_tok, "base_tok_s": b_tok,
            "lora_tax_fraction": tax, "projected_actions": actions,
            "boot_seconds": summary.get("boot_seconds")}


def _load(d: Path) -> tuple[dict | None, str]:
    summary = None
    hit = next(iter(sorted(d.rglob("lora_canary.json"))), None)
    if hit:
        try:
            summary = json.loads(hit.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not parse {hit}: {exc}", file=sys.stderr)
    chunks = []
    for pattern in ("*.log", "*.txt", "*.json", "**/vllm-openai-server.log"):
        for f in sorted(d.rglob(pattern)):
            if f.name == "lora_canary.json":
                continue
            try:
                chunks.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return summary, "\n".join(chunks)


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _diff_line(noop: bool, probe: bool, idx: object = 7) -> str:
    return (f"LORA-CANARY differential noop_identical_to_base={noop} "
            f"probe_differs_from_base={probe} first_divergent_token_index={idx}")


def _good_log(noop: bool = True, probe: bool = True, banner: bool = True,
              markers: tuple[str, ...] = REQUIRED_MARKERS) -> str:
    parts = list(markers) + [_diff_line(noop, probe)]
    if banner:
        parts.append(DIFFERENTIAL_BANNER + " (adapter is loaded AND applied)")
    return "\n".join(parts)


def _summary(a_tok: float, b_tok: float, verdict: str = "PASS") -> dict:
    return {"boot_seconds": 640.0, "lora_tax_fraction": 1.0 - a_tok / b_tok,
            "action_bar": ACTION_BAR, "verdict": verdict,
            "throughput": {"base": {"tok_s": b_tok,
                                    "actions_per_window": b_tok * WINDOW_S / TOKENS_PER_ACTION},
                           "adapter": {"tok_s": a_tok,
                                       "actions_per_window": a_tok * WINDOW_S / TOKENS_PER_ACTION}}}


def _selftest() -> int:
    ok, bad = 0, []

    def expect(name, got, want_verdict, want_decisive=None, want_tp=None):
        nonlocal ok
        good = got["verdict"] == want_verdict
        if want_decisive is not None:
            good = good and got["decisive"] == want_decisive
        if want_tp is not None:
            good = good and got.get("throughput_verdict") == want_tp
        if good:
            ok += 1
            print(f"  PASS {name} -> {got['verdict']}"
                  + (f" / {got.get('throughput_verdict')}" if want_tp else ""))
        else:
            bad.append(name)
            print(f"  FAIL {name} -> {got['verdict']}/{got.get('throughput_verdict')} "
                  f"(want {want_verdict}/{want_tp}) :: {got['reason'][:160]}")

    print("-- the sealed truth table --")
    expect("noop==base & probe!=base & all round-trips => PASS",
           score(_summary(60.0, 70.0), _good_log()), PASS, True, "THROUGHPUT-PASS")
    expect("noop==base & probe==base => SILENTLY IGNORED, decisive FAIL",
           score(None, _good_log(probe=False, banner=False)), FAIL, True)
    expect("noop!=base => numerically unsound, decisive FAIL",
           score(None, _good_log(noop=False, banner=False)), FAIL, True)
    expect("noop!=base AND probe!=base is STILL a FAIL (unsound wins)",
           score(_summary(60.0, 70.0), _good_log(noop=False, probe=True, banner=False)),
           FAIL, True)

    print("-- infra is never a verdict on LoRA --")
    expect("server never booted (CUDA runtime)",
           score(None, "ImportError: libcudart.so.13: cannot open shared object file"),
           INFRA, False)
    expect("OOM with --enable-lora", score(None, "torch.OutOfMemoryError: CUDA out of memory"),
           INFRA, False)
    expect("boot timeout", score(None, "TimeoutError: Timed out waiting for vLLM server"),
           INFRA, False)
    expect("empty log", score(None, ""), INFRA, False)
    expect("THE 08-13 NEAR-MISS: adapter dataset silently dropped by Kaggle",
           score(None, "RuntimeError: LORA-CANARY FATAL: adapter dataset not mounted at "
                       "/kaggle/input/arc3-lora-probe-adapters"), INFRA, False)
    expect("published dataset flattened (no lora-noop/ subdir)",
           score(None, "LORA-CANARY FATAL: no lora-noop/adapter_config.json under "
                       "/kaggle/input/arc3-lora-probe-adapters contents=[...]"), INFRA, False)
    expect("dataset shipped stale weights (sha mismatch)",
           score(None, "LORA-CANARY FATAL: lora-probe sha deadbeefdeadbeef != d7d6918d01ae67f6 "
                       "(the dataset push did not ship what we built)"), INFRA, False)

    print("-- explicit vLLM refusal IS the answer (decisive), not infra --")
    expect("vLLM refuses LoRA on this quantized base",
           score(None, "ValueError: LoRA is not supported with quantization method fp8"),
           FAIL, True)
    expect("rank rejected by the engine",
           score(None, "ValueError: max_lora_rank (16) must be one of ..."), FAIL, True)

    print("-- serve-contract failures are decisive FAILs --")
    expect("/v1/models missing an adapter",
           score(None, "RuntimeError: LORA-CANARY FATAL: /v1/models = ['vrfai/Qwen3.6-27B-FP8'] "
                       "!= [...] -- vLLM did not register both adapters"), FAIL, True)
    expect("tool-call round-trip broken under LoRA",
           score(None, "RuntimeError: LORA-CANARY FATAL: tool-call round-trip FAILED under "
                       "qwen3_coder WITH LoRA enabled"), FAIL, True)
    expect("vision path broken under LoRA",
           score(None, "RuntimeError: LORA-CANARY FATAL: MM probe returned empty content with "
                       "LoRA enabled"), FAIL, True)

    print("-- ADVERSARIAL: fixtures that MUST NOT pass --")
    # (a) The banner is present and the numbers are great, but the BOOLEANS say the
    #     adapter was ignored. A banner is a string; the booleans are the evidence.
    expect("A1 forged/contradictory banner with probe==base MUST NOT pass",
           score(_summary(200.0, 200.0), _good_log(probe=False, banner=True)), FAIL, True)
    # (b) Differential passed but a round-trip never happened.
    expect("A2 differential PASS with the MM round-trip missing MUST NOT pass",
           score(_summary(60.0, 70.0),
                 _good_log(markers=REQUIRED_MARKERS[:2])), FAIL, True)
    # (c) Booleans fine but the assert block never completed (no banner).
    expect("A3 booleans fine but no differential=PASS banner MUST NOT pass",
           score(_summary(60.0, 70.0), _good_log(banner=False)), FAIL, True)
    # (d) A summary JSON claiming success with NO differential evidence at all.
    expect("A4 a summary that claims verdict=PASS with no differential in the log "
           "MUST NOT pass", score(_summary(200.0, 200.0), "\n".join(REQUIRED_MARKERS)),
           INFRA, False)
    # (e) noop==probe==base is the silent-null this canary exists to prevent.
    expect("A5 the silent null (adapter ignored, everything else green) MUST NOT pass",
           score(_summary(190.0, 190.0), _good_log(probe=False, banner=False)), FAIL, True)

    print("-- the throughput axis is separate and recomputed --")
    got = score(_summary(30.0, 70.0), _good_log())
    expect("serve can PASS while throughput FAILs (two axes, not one)",
           got, PASS, True, "THROUGHPUT-FAIL")
    if any("MISMATCH" in n for n in got["notes"]):
        ok += 1
        print("  PASS the kernel's self-reported throughput verdict is cross-checked")
    else:
        bad.append("throughput self-report cross-check")
        print("  FAIL the kernel's self-reported throughput verdict is not cross-checked")
    edge = score(_summary(40.0, 40.0), _good_log())
    expect("exactly 40.0 tok/s == exactly 100 actions is >= the bar",
           edge, PASS, True, "THROUGHPUT-PASS")
    expect("serve PASS with no JSON leaves throughput UNMEASURED, not FAIL",
           score(None, _good_log()), PASS, True, "UNMEASURED")

    print("-- sealed constants must match the builder --")
    builder_path = Path(__file__).with_name("build_lora_serve_canary.py")
    builder = builder_path.read_text(encoding="utf-8")
    for name, literal in (("ACTION_BAR", "ACTION_BAR = 100.0"),
                          ("TOK_S_27B", "TOK_S_27B = 192.0"),
                          ("WINDOW_S", "WINDOW_S = 7920.0"),
                          ("LORA_RANK", "LORA_RANK = 16"),
                          ("ADAPTER_DS", f'ADAPTER_DS = "{ADAPTER_DS}"'),
                          ("NOOP_NAME", f"'{NOOP_NAME}'"),
                          ("PROBE_NAME", f"'{PROBE_NAME}'")):
        if literal in builder:
            ok += 1
            print(f"  PASS constant {name} matches the builder")
        else:
            bad.append(f"constant {name}")
            print(f"  FAIL constant {name} does not match the builder ({literal!r})")
    if abs(TOK_S_27B * WINDOW_S / 480.0 - TOKENS_PER_ACTION) > 1e-9:
        bad.append("tokens-per-action derivation")
        print("  FAIL tokens-per-action does not re-derive from the frozen 27B anchor")
    else:
        ok += 1
        print("  PASS tokens-per-action re-derives from the frozen 27B anchor")

    # The shas we will score against must be the shas we actually built and published.
    manifest_path = (builder_path.parents[2] / "runs" / "lora_lane" / "probe_adapters"
                     / "PROBE_ADAPTERS.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        good = (manifest["noop"]["sha256_16"] == ADAPTER_SHA16["lora-noop"]
                and manifest["probe"]["sha256_16"] == ADAPTER_SHA16["lora-probe"]
                and manifest["noop"]["bytes"] == manifest["probe"]["bytes"] == ADAPTER_BYTES)
    except Exception as exc:  # noqa: BLE001
        good, manifest = False, {"error": repr(exc)}
    if good:
        ok += 1
        print("  PASS sealed adapter shas/bytes match PROBE_ADAPTERS.json")
    else:
        bad.append("adapter sha/bytes seal")
        print(f"  FAIL sealed adapter shas/bytes do not match PROBE_ADAPTERS.json {manifest}")

    # And they must be the shas the BUILT notebook pins at runtime.
    nb = builder_path.parents[2] / "notebooks" / "lora-serve-canary" / "arc3-lora-serve-canary.ipynb"
    if nb.is_file():
        text = nb.read_text(encoding="utf-8")
        if all(s in text for s in ADAPTER_SHA16.values()):
            ok += 1
            print("  PASS the built notebook pins those same shas")
        else:
            bad.append("notebook sha pin")
            print("  FAIL the built notebook does not pin the sealed shas")
    else:
        bad.append("built notebook missing")
        print("  FAIL built notebook not found")

    print(f"\nselftest: {ok} passed / {len(bad)} failed")
    for b in bad:
        print("  FAILED:", b)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        print("LORA SERVE CANARY SCORER SELFTEST (run before the data exists)")
        return _selftest()
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = Path(argv[1])
    summary, log_text = _load(d)
    result = score(summary, log_text)
    print("=" * 78)
    print("LORA SERVE VERDICT:", result["verdict"],
          "(decisive)" if result.get("decisive") else "(not decisive)")
    if result.get("throughput_verdict"):
        print("THROUGHPUT:", result["throughput_verdict"])
    print("reason:", result["reason"])
    for n in result.get("notes", []):
        print("  -", n)
    print("=" * 78)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
