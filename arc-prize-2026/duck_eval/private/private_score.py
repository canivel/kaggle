"""SEALED per-arm certifier + verdict scorer for `canivel/arc3-q38-private-eval`.

Written 2026-08-21 BEFORE any arm of this kernel has run. Constants are SEALED by
`learnings/war_room/private_arm_prereg_2026-08-21.md`; reading data and then adjusting
a number here is the one thing that voids the screen.

THE FROZEN-CERTIFIER LANDMINE, DEFUSED BY CONSTRUCTION
------------------------------------------------------
This kernel ships ARMS: base / edge1 / edge2 / edge12, distinguished ONLY by two flag
constants. The campaign has twice built certifiers whose expected markers were correct
for one arm and fatal for its sibling (q38-low landmine; the 08-20 shared-scorer defect:
"one scorer structurally cannot hold both"). Here the expected values are a PER-ARM
TABLE, the arm is an EXPLICIT required argument, and the selftest includes CROSS-ARM
NEGATIVE CONTROLS: every arm's fixture fed to every OTHER arm's certification MUST
refuse. A certifier that cannot refuse a wrong-arm artifact is not a certifier
(feedback_guard_never_fired).

Usage:
    python duck_eval/private/private_score.py <pulled_kernel_dir> --arm base
    python duck_eval/private/private_score.py <pulled_kernel_dir> --arm edge1 --certify-only
    python duck_eval/private/private_score.py <pulled_kernel_dir> --arm edge1 --base-lc 28 --base-actions 1639
    python duck_eval/private/private_score.py --selftest

`--certify-only` returns BEFORE any effect size is computed so the operational
(queue-head) call stays uncontaminated by the science number.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graft"))
# Reuse the 08-19 log-normalisation fix (CLI 2.2.3 JSON-array logs escape every quote);
# import, never copy (a copy re-introduces the bug the moment the fix improves).
from graft_score import _decode_cli_json_log, _read_log, _infra  # noqa: E402

# ---- SEALED runtime-certification constants (all arms) --------------------
SERVED_MODEL = "Qwen/Qwen3.8-27B-FP8"
EFFORT_TOKEN = "reasoning_effort"          # MUST be absent => template xhigh
BUNDLE_LABEL = "model-20260815-q38-p1"     # 08-15 harness generation, asserted in-notebook
N_GAMES = 25
BANNER_RE = re.compile(
    r"PRIVATE-ARM BANNER: bundle=(?P<bundle>\S+) served=(?P<served>\S+) "
    r"edge1_ctx_raise=(?P<edge1>True|False) analyzer_ctx=(?P<ctx>\d+) "
    r"effective_ctx_budget=(?P<budget>\d+) vllm_max_model_len=(?P<mml>\d+) "
    r"edge2_contract=(?P<edge2>True|False)"
)
CONTRACT_ACTIVE_MARKER = "PRIVATE EDGE 2 (visible-updates contract): ACTIVE (Q38-strengthened)"
CONTRACT_OFF_MARKER = "PRIVATE EDGE 2 (visible-updates contract): OFF - zero-touch"
# Wrong-arm artifact classes (all arms): this kernel NEVER carries grafts.
GRAFT_MARKERS_FORBIDDEN = (
    "TAAF_GRAFTS FEATURES=",
    "[goalkeep] armed",
    "[hudmask] armed",
    "[banking] armed",
    "[clickmap] armed",
    "[searchmap] armed",
)


@dataclass(frozen=True)
class ArmSpec:
    edge1: bool
    edge2: bool
    analyzer_ctx: int
    effective_budget: int
    vllm_mml: int
    serve_flag: str          # must appear in the vLLM serve-command echo
    contract_required: bool  # ACTIVE marker required (True) or FORBIDDEN (False)


# ---- SEALED PER-ARM CERTIFICATION TABLE -----------------------------------
# Expected values FOLLOW THE ARM; the gate logic is invariant (the q38-low lesson,
# named in the 08-21 pre-authorization: "a gate's logic invariant, expected values
# follow the arm").
ARMS: dict[str, ArmSpec] = {
    "base": ArmSpec(
        edge1=False, edge2=False,
        analyzer_ctx=32768, effective_budget=31744, vllm_mml=65536,
        serve_flag="--max-model-len 65536",
        contract_required=False,
    ),
    "edge1": ArmSpec(
        edge1=True, edge2=False,
        analyzer_ctx=65536, effective_budget=64512, vllm_mml=131072,
        serve_flag="--max-model-len 131072",
        contract_required=False,
    ),
    "edge2": ArmSpec(
        edge1=False, edge2=True,
        analyzer_ctx=32768, effective_budget=31744, vllm_mml=65536,
        serve_flag="--max-model-len 65536",
        contract_required=True,
    ),
    "edge12": ArmSpec(
        edge1=True, edge2=True,
        analyzer_ctx=65536, effective_budget=64512, vllm_mml=131072,
        serve_flag="--max-model-len 131072",
        contract_required=True,
    ),
}

# ---- SEALED verdict bands (prereg §3) -------------------------------------
# Comparator: the certified field-floor run q38_field_v1 (08-20): lc_total 28,
# mean_score 6.173, total_actions 1639 — the NEW floor; old 17-21 bands obsolete.
COMPARATOR_LC = 28
COMPARATOR_SCORE = 6.173
COMPARATOR_ACTIONS = 1639
# Absolute bands vs the floor (same ruler as the Arm-3 seal: HARM<=23 / NULL 24-32 /
# SIGNAL>=33). Used when no paired base run exists (base arm itself, or base infra-death).
ABS_HARM_MAX = 23
ABS_SIGNAL_MIN = 33
# Paired bands (edge arm vs the SAME-notebook base arm's measured lc, +-5 lc: the
# single-run-vs-single-run spacing, sqrt(2) x the 3.54-lc pooled sigma, rounded).
PAIRED_DELTA = 5
# EDGE-1 WALLCLOCK KILL (prereg §4): longer context => slower tokens => fewer actions;
# the quadratic punishes it. KILL iff lc_edge <= lc_base AND actions < 60% of base's.
KILL_ACTIONS_RATIO = 0.60


def _fail(msg: str) -> None:
    print(f"INFRA DEATH: {msg}")
    raise SystemExit(2)


def _load_benchmark(run_dir: Path) -> dict:
    candidates = list(run_dir.rglob("benchmark.json"))
    if not candidates:
        _fail("no benchmark.json in pulled kernel output")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _collect_log_text(run_dir: Path) -> str:
    # graft_score._read_log takes a DIRECTORY and concatenates *.log/*.txt/log*/*.out
    # inside it (with the CLI-2.2.3 JSON-array normalisation). Cover subdirectories too.
    dirs = [run_dir] + sorted(d for d in run_dir.rglob("*") if d.is_dir())
    text = "\n".join(filter(None, (_read_log(d) for d in dirs)))
    if not text.strip():
        _fail("no log/txt content found in pulled kernel output")
    return text


def certify(run_dir: Path, arm_name: str) -> dict:
    """Per-arm runtime certification. ANY failure => INFRA DEATH (exit 2), never NULL."""
    spec = ARMS[arm_name]
    log = _collect_log_text(run_dir)

    # C1 completion + game count
    bench = _load_benchmark(run_dir)
    games = bench.get("game_runs") or bench.get("games") or bench.get("results") or []  # INSTRUMENT FIX 2026-08-21: the REAL taaf benchmark key is game_runs; fixtures used games (internal consistency is not correctness, instance 4)
    n = len(games) if isinstance(games, list) else int(bench.get("n_games", 0))
    if n != N_GAMES:
        _fail(f"n_games {n} != {N_GAMES}")

    # C2 served model (feedback_kaggle_model_attach: attach is the silent-drop trap)
    if SERVED_MODEL not in log:
        _fail(f"served model {SERVED_MODEL!r} not found in logs")

    # C3 effort absence => template xhigh ran
    if EFFORT_TOKEN in log:
        _fail(f"{EFFORT_TOKEN!r} present in logs — a pinned-effort config ran, not this arm")

    # C4 bundle generation
    if f"TAAF bundle generation: {BUNDLE_LABEL}" not in log:
        _fail(f"bundle-generation line for {BUNDLE_LABEL!r} missing — wrong harness generation")

    # C5 banner present AND flag states EXACTLY match the arm row
    m = BANNER_RE.search(log)
    if not m:
        _fail("PRIVATE-ARM BANNER missing from logs")
    got = {
        "bundle": m.group("bundle"),
        "served": m.group("served"),
        "edge1": m.group("edge1") == "True",
        "ctx": int(m.group("ctx")),
        "budget": int(m.group("budget")),
        "mml": int(m.group("mml")),
        "edge2": m.group("edge2") == "True",
    }
    want = {
        "bundle": BUNDLE_LABEL,
        "served": SERVED_MODEL,
        "edge1": spec.edge1,
        "ctx": spec.analyzer_ctx,
        "budget": spec.effective_budget,
        "mml": spec.vllm_mml,
        "edge2": spec.edge2,
    }
    if got != want:
        _fail(f"banner/arm mismatch (wrong-arm artifact): got {got}, arm {arm_name!r} expects {want}")

    # C6 the vLLM serve command actually carried the arm's max-model-len
    if spec.serve_flag not in log:
        _fail(f"serve-command flag {spec.serve_flag!r} not found — served ctx != certified ctx")

    # C7 contract marker: REQUIRED for contract arms, FORBIDDEN otherwise
    if spec.contract_required:
        if CONTRACT_ACTIVE_MARKER not in log:
            _fail("contract-ACTIVE marker missing on a contract arm")
        if CONTRACT_OFF_MARKER in log:
            _fail("contract OFF marker present on a contract arm")
    else:
        if CONTRACT_ACTIVE_MARKER in log:
            _fail("contract-ACTIVE marker present on a non-contract arm (wrong-arm artifact)")
        if CONTRACT_OFF_MARKER not in log:
            _fail("contract OFF (zero-touch) marker missing")

    # C8 no grafts, ever, on any arm of this kernel
    for marker in GRAFT_MARKERS_FORBIDDEN:
        if marker in log:
            _fail(f"graft marker {marker!r} present — wrong-arm artifact scored")

    print(f"CERTIFIED: arm={arm_name} bundle={BUNDLE_LABEL} served={SERVED_MODEL} "
          f"ctx={spec.analyzer_ctx} mml={spec.vllm_mml} contract={spec.contract_required}")
    return bench


def _lc_actions_score(bench: dict) -> tuple[int, int, float]:
    games = bench.get("game_runs") or bench.get("games") or bench.get("results") or []  # INSTRUMENT FIX 2026-08-21: the REAL taaf benchmark key is game_runs; fixtures used games (internal consistency is not correctness, instance 4)
    # INSTRUMENT FIX 2026-08-22 (third fixture-key defect in this scorer, disclosed in
    # exp 32/33): the REAL taaf per-run keys are levels_completed / actions_per_level
    # (a per-level LIST) / final_score. Real keys first; fixture fallbacks retained.
    lc = sum(int(g.get("levels_completed", g.get("lc", 0))) for g in games)
    actions = sum(
        sum(g["actions_per_level"]) if isinstance(g.get("actions_per_level"), list)
        else int(g.get("total_actions", g.get("actions", 0)))
        for g in games)
    scores = [float(g.get("final_score", g.get("score", 0.0))) for g in games]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return lc, actions, mean_score


def science(bench: dict, arm_name: str, base_lc: int | None, base_actions: int | None) -> None:
    lc, actions, mean_score = _lc_actions_score(bench)
    print(f"MEASURED: lc_total={lc} total_actions={actions} mean_score={mean_score:.3f}")

    if arm_name == "base" or base_lc is None:
        # Absolute read vs the sealed floor comparator.
        if lc <= ABS_HARM_MAX:
            verdict = "HARM"
        elif lc >= ABS_SIGNAL_MIN:
            verdict = "SIGNAL"
        else:
            verdict = "NULL"
        print(f"VERDICT (absolute, comparator lc {COMPARATOR_LC}): {verdict} "
              f"[HARM<= {ABS_HARM_MAX} | NULL {ABS_HARM_MAX+1}-{ABS_SIGNAL_MIN-1} | SIGNAL>= {ABS_SIGNAL_MIN}]")
        if base_lc is None and arm_name != "base":
            print("NOTE: paired base run unavailable — absolute fallback read per prereg §3; "
                  "the edge attribution is WEAKER on this path and must be labeled as such.")
    else:
        delta = lc - base_lc
        if delta >= PAIRED_DELTA:
            verdict = "SIGNAL"
        elif delta <= -PAIRED_DELTA:
            verdict = "HARM"
        else:
            verdict = "NULL"
        print(f"VERDICT (paired vs base lc {base_lc}): {verdict} (delta {delta:+d}; bands +-{PAIRED_DELTA})")
        # Edge-1 wallclock kill criterion (prereg §4)
        if arm_name in ("edge1", "edge12") and base_actions:
            ratio = actions / base_actions if base_actions else 0.0
            if lc <= base_lc and ratio < KILL_ACTIONS_RATIO:
                print(f"EDGE1 KILL: wallclock trade LOST (lc {lc} <= base {base_lc} AND "
                      f"actions ratio {ratio:.2f} < {KILL_ACTIONS_RATIO}) — the ctx raise slowed "
                      "tokens without buying levels; flag goes OFF and stays OFF.")
            else:
                print(f"EDGE1 wallclock check: actions ratio {ratio:.2f} (kill needs "
                      f"< {KILL_ACTIONS_RATIO} AND lc <= base) — no kill.")
    print(f"RECORDED (non-inferential): mean_score {mean_score:.3f} vs floor {COMPARATOR_SCORE}")


# ---------------------------------------------------------------------------
# Selftest — fixtures per arm + CROSS-ARM NEGATIVE CONTROLS
# ---------------------------------------------------------------------------

def _fixture(arm_name: str, tmp: Path, lc_per_game: int = 1, actions_per_game: int = 66,
             corrupt: str | None = None) -> Path:
    spec = ARMS[arm_name]
    d = tmp / f"fx_{arm_name}_{corrupt or 'ok'}"
    d.mkdir(parents=True, exist_ok=True)
    games = [
        {"game_id": f"g{i:02d}", "levels_completed": lc_per_game,
         "total_actions": actions_per_game, "score": 2.5}
        for i in range(N_GAMES)
    ]
    if corrupt == "n_games":
        games = games[:-1]
    (d / "benchmark.json").write_text(json.dumps({"game_runs": games}), encoding="utf-8")  # real taaf shape (2026-08-21)
    banner = (
        f"PRIVATE-ARM BANNER: bundle={BUNDLE_LABEL} served={SERVED_MODEL} "
        f"edge1_ctx_raise={spec.edge1} analyzer_ctx={spec.analyzer_ctx} "
        f"effective_ctx_budget={spec.effective_budget} vllm_max_model_len={spec.vllm_mml} "
        f"edge2_contract={spec.edge2}"
    )
    lines = [
        f"TAAF bundle generation: {BUNDLE_LABEL}",
        f"Starting vLLM OpenAI server: python -m vllm.entrypoints.openai.api_server "
        f"--model /kaggle/input/models/x --served-model-name {SERVED_MODEL} "
        f"{spec.serve_flag} --reasoning-parser qwen3",
        banner,
        CONTRACT_ACTIVE_MARKER if spec.contract_required else CONTRACT_OFF_MARKER,
    ]
    if corrupt == "effort":
        lines.append('chat_template_kwargs: {"reasoning_effort": "medium"}')
    if corrupt == "graft":
        lines.append("[goalkeep] armed")
    if corrupt == "no_banner":
        lines = [l for l in lines if "PRIVATE-ARM BANNER" not in l]
    if corrupt == "stale_bundle":
        lines = [l.replace(BUNDLE_LABEL, "kaggle-milestone-20260630") for l in lines]
    (d / "run.log").write_text("\n".join(lines), encoding="utf-8")
    return d


def _expect_certify(run_dir: Path, arm: str, should_pass: bool, label: str, results: list) -> None:
    try:
        certify(run_dir, arm)
        ok = should_pass
    except SystemExit:
        ok = not should_pass
    results.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def selftest() -> None:
    results: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # 1) every arm's clean fixture certifies under its OWN arm
        for arm in ARMS:
            _expect_certify(_fixture(arm, tmp), arm, True, f"{arm} fixture certifies as {arm}", results)
        # 2) CROSS-ARM NEGATIVE CONTROLS: every fixture REFUSES under every other arm
        for fx_arm in ARMS:
            fx = _fixture(fx_arm, tmp)
            for cert_arm in ARMS:
                if cert_arm == fx_arm:
                    continue
                _expect_certify(fx, cert_arm, False,
                                f"{fx_arm} fixture REFUSED as {cert_arm}", results)
        # 3) corruption controls (each gate proven able to refuse)
        for corrupt in ("n_games", "effort", "graft", "no_banner", "stale_bundle"):
            _expect_certify(_fixture("base", tmp, corrupt=corrupt), "base", False,
                            f"base fixture with {corrupt} REFUSED", results)
        # 4) verdict arithmetic (bands + kill), pure function checks
        bench_hi = json.loads((_fixture("edge1", tmp, lc_per_game=2) / "benchmark.json").read_text())
        lc, actions, _ = _lc_actions_score(bench_hi)
        band_ok = (lc == 50 and actions == 25 * 66)
        results.append(("lc/actions extraction", band_ok))
        print(f"  [{'PASS' if band_ok else 'FAIL'}] lc/actions extraction")
        checks = [
            (33, "SIGNAL"), (28, "NULL"), (23, "HARM"),  # absolute
        ]
        for lc_val, want in checks:
            got = "HARM" if lc_val <= ABS_HARM_MAX else ("SIGNAL" if lc_val >= ABS_SIGNAL_MIN else "NULL")
            ok = got == want
            results.append((f"absolute band lc={lc_val}->{want}", ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] absolute band lc={lc_val} -> {got}")
        paired = [(28 + 5, "SIGNAL"), (28 + 4, "NULL"), (28 - 4, "NULL"), (28 - 5, "HARM")]
        for lc_val, want in paired:
            d = lc_val - 28
            got = "SIGNAL" if d >= PAIRED_DELTA else ("HARM" if d <= -PAIRED_DELTA else "NULL")
            ok = got == want
            results.append((f"paired band lc={lc_val}->{want}", ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] paired band lc={lc_val} -> {got}")
        kill_cases = [
            (28, 900, 1639, True),    # lc == base, actions 0.55x -> KILL
            (28, 1100, 1639, False),  # ratio 0.67 -> no kill
            (33, 900, 1639, False),   # lc beats base -> never kill
        ]
        for lc_val, act, base_act, want_kill in kill_cases:
            kill = lc_val <= 28 and (act / base_act) < KILL_ACTIONS_RATIO
            ok = kill == want_kill
            results.append((f"kill lc={lc_val} act={act} -> {want_kill}", ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] kill-criterion lc={lc_val} actions={act} -> {kill}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\nSELFTEST: {n_pass}/{len(results)}")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", help="pulled kernel output dir")
    parser.add_argument("--arm", choices=sorted(ARMS), help="which arm this artifact claims to be")
    parser.add_argument("--certify-only", action="store_true")
    parser.add_argument("--base-lc", type=int, default=None,
                        help="paired comparator: the base arm's measured lc_total")
    parser.add_argument("--base-actions", type=int, default=None,
                        help="paired comparator: the base arm's measured total_actions")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        raise SystemExit(0)
    if not args.run_dir or not args.arm:
        parser.error("run_dir and --arm are required (or use --selftest)")
    bench = certify(Path(args.run_dir), args.arm)
    if args.certify_only:
        raise SystemExit(0)
    science(bench, args.arm, args.base_lc, args.base_actions)
