"""Stage-2 scripted-policy non-interference test (protocol step 2) — no LLM.

Runs the three engine-verified scripted policies (sb26/su15/lp85, recorded as
flat action sequences by policies/record_policies.py) through the REAL
`_HarnessGameSession.step_env` -> `_execute_action` path on the local
arcengine, twice per game:

  arm A (baseline): stock harness, ledger graft NOT installed;
  arm B (grafted):  `ledger_patch.install(flags={'ledger','escalation'})`.

Pass gates (intervention_plan.md, protocol step 2):
  N1  all three games still hit levels_completed == 2 under the graft;
  N2  action counts are IDENTICAL between baseline and grafted arms
      (the graft never executes actions);
  N3  the ledger contains the correct action-effect FACTs as a side effect
      (two level-completion facts per game, at the right action numbers);
  N4  all-flags-off install is a no-op (stock semantics preserved).

Run:  f:/kaggle/arc-prize-2026/.venv/Scripts/python.exe duck_eval/ledger/noninterference_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "policies"))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "ledger-noninterference")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class FakeSolver:
    label = "ledger-nonint"
    job_dir = None
    max_actions_per_game = 400
    max_runtime_s_per_game = None

    def soft_time_remaining_seconds(self):
        return None


def make_game(game: str):
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=str(ENV_FILES))
    game_id = next(e.game_id for e in arcade.available_environments
                   if e.game_id.startswith(game))
    api = GameAPI(env_name=game_id,
                  arcade_spec=ArcadeSpec(
                      operation_mode=arc_agi.OperationMode.OFFLINE,
                      environments_dir=str(ENV_FILES)))
    api.start_game()
    return api


def run_sequence(game: str, sequence: list[dict], tmpdir: Path):
    """Drive one recorded policy through the real session step_env path."""
    import inference.framework.solver as solver_mod

    api = make_game(game)
    session = solver_mod._HarnessGameSession(
        solver=FakeSolver(),
        game=api,
        analyzer=SimpleNamespace(generated_tokens=0, _timeout=5.0),
        game_index=0,
        pass_index=0,
        state_path=tmpdir / "runtime_state.json",
        transcript_path=tmpdir / "transcript.txt",
        analysis_html_relpath="solver_analysis/nonint.html",
        stop_event=threading.Event(),
        viewer_data_path=tmpdir / "viewer_data.json",
    )
    last = {}
    for request in sequence:
        last = session.step_env(dict(request))
        assert last.get("executed"), f"{game}: action failed: {request} -> {last}"
    run = api.game_run
    return {
        "levels_completed": int(run.levels_completed),
        "actions": len(run.history),
        "final_state": str(last.get("state")),
    }


def main() -> int:
    print(f"ledger stage-2 non-interference test | repo={REPO}")
    import arcengine

    dist_version = "?"
    try:
        from importlib.metadata import version as _v

        dist_version = _v("arcengine")
    except Exception:  # noqa: BLE001
        pass
    print(f"arcengine dist={dist_version}")

    import record_policies

    print("recording scripted policies on raw engines (verified L2 clears)...")
    sequences = record_policies.record_all()
    for game, sequence in sequences.items():
        print(f"  {game}: {len(sequence)} actions")

    import ledger_patch

    # ---- arm A: baseline, graft not installed -----------------------------
    print("\n[arm A: baseline (stock harness)]")
    baseline = {}
    for game, sequence in sequences.items():
        with tempfile.TemporaryDirectory() as td:
            baseline[game] = run_sequence(game, sequence, Path(td))
        print(f"  {game}: lc={baseline[game]['levels_completed']} "
              f"actions={baseline[game]['actions']}")

    # ---- N4: all-flags-off install is a no-op ------------------------------
    flags_off = ledger_patch.install(None, {})
    check("N4 all-flags-off install is a no-op",
          not any(flags_off.values()), str(flags_off))
    with tempfile.TemporaryDirectory() as td:
        off_result = run_sequence("sb26", sequences["sb26"], Path(td))
        no_ledger_file = not list(Path(td).glob("ledger*.json"))
    check("N4b flags-off run identical to baseline, no ledger file written",
          off_result == baseline["sb26"] and no_ledger_file,
          f"{off_result} vs {baseline['sb26']}")

    # ---- arm B: grafted (ledger + escalation) ------------------------------
    print("\n[arm B: grafted (ledger+escalation)]")
    flags = ledger_patch.install(None, {"ledger": True, "escalation": True})
    assert flags["ledger"] and flags["escalation"], flags
    grafted = {}
    ledgers = {}
    for game, sequence in sequences.items():
        with tempfile.TemporaryDirectory() as td:
            grafted[game] = run_sequence(game, sequence, Path(td))
            # v2: per-game persistence file ledger_<stem>.json
            ledger_files = sorted(Path(td).glob("ledger*.json"))
            ledgers[game] = (json.loads(ledger_files[0].read_text(encoding="utf-8"))
                             if ledger_files else None)
        print(f"  {game}: lc={grafted[game]['levels_completed']} "
              f"actions={grafted[game]['actions']}")

    for game in sequences:
        check(f"N1 {game} still clears L2 through the grafted harness",
              grafted[game]["levels_completed"] == 2,
              f"lc={grafted[game]['levels_completed']}")
        check(f"N2 {game} identical action counts (baseline vs grafted)",
              grafted[game]["actions"] == baseline[game]["actions"]
              and baseline[game]["levels_completed"] == 2,
              f"baseline={baseline[game]['actions']} "
              f"grafted={grafted[game]['actions']}")
        led = ledgers[game]
        level_facts = ([f for f in led["facts"]
                        if f["statement"].startswith("level completed")]
                       if led else [])
        check(f"N3 {game} ledger recorded both level-completion FACTs",
              led is not None and len(level_facts) == 2,
              "; ".join(f["statement"][:48] for f in level_facts) or "none")

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
