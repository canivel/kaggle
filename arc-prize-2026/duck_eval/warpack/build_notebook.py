"""Apply the warpack cell modifications to the vanilla duckfork notebook.

    python duck_eval/warpack/build_notebook.py [--diff-only] [--v2]

Reads notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb,
applies:
  cell 2  : + fast-submit gate (RUN_HEAVY)
  cell 4  : wrapped in `if RUN_HEAVY:` (wheel install)
  cell 6  : wrapped (bundle location)
  cell 8  : wrapped (source import + vLLM setup commands)
  cell 10 : wrapped (benchmark unpickle)
  cell 12 : replaced with the warpack customization hook (hook_cell.py)
  cell 14 : replaced (fast path dummy parquet + soft_end = start + 11h20m)

Writes duck_eval/warpack/duck_warpack.ipynb and prints unified diffs.

--v2 (war-v2, prereg 2026-07-14 §4 design lock: war-v1 + flags {ledger,
escalation} ONLY; observability-only additions per panel R12) additionally:
  cell 2  : + gate detection-signal logging (llm-agents prior #6)
  cell 12 : + ledger graft hook (duck_eval/ledger/hook_cell.py) gated under
            RUN_HEAVY; LEDGER_FLAGS defaults to ledger,escalation
Writes duck_eval/warpack/duck_warpack_v2.ipynb instead.

(f) DEFAULT (prereg amendment 2026-07-23 item 4; W0 screen PASS 49/49): both
modes append the game-over-continuation graft to cell 12 by default. Runtime
kill switch stays (CONTINUATION_DISABLE=1); build opt-out `--no-continuation`
reproduces pre-Jul-23 compositions.

--sentinel (LIVE submission arm, exploration draw #1 per amendment 2026-07-23
A21/C4): vanilla duck + (f) continuation (hygiene default) + budget sentinel
@ SENTINEL_BUDGET=150 ONLY — NO warpack, NO ledger (one-flag discipline).
Cell 2 gains the gate addendum + the LIVE budget stamp (scored regime is
uncapped -> without the export the sentinel is inert, C7 #1 risk); cell 12 =
the sentinel graft + continuation graft (both marker-found in the attached
arc-war-kit dataset; vanilla-duck fallback on any failure). Writes
duck_eval/warpack/duck_sentinel_live.ipynb AND
notebooks/ducksentinel/{arc3-duck-sentinel.ipynb, kernel-metadata.json}
(metadata = duckwar submission metadata with id/title/code_file swapped).
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VANILLA = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
OUT = HERE / "duck_warpack.ipynb"
OUT_V2 = HERE / "duck_warpack_v2.ipynb"
LEDGER_HOOK = REPO / "duck_eval" / "ledger" / "hook_cell.py"

sys.path.insert(0, str(HERE))
import fastsubmit_cells  # noqa: E402

HEAVY_CELLS = (4, 6, 8, 10)

# --sentinel live outputs
OUT_SENTINEL = HERE / "duck_sentinel_live.ipynb"
NB_DIR_SENTINEL = REPO / "notebooks" / "ducksentinel"
KERNEL_ID_SENTINEL = "canivel/arc3-duck-sentinel"
SRC_WAR_META = REPO / "notebooks" / "duckwar" / "kernel-metadata.json"

# LIVE cell-2 stamp for the sentinel arm. The scored regime is UNCAPPED
# (max_actions_per_game=None, confirmed in the W0 log), so the budget MUST be
# exported here or the sentinel is silently inert (C7 #1 pre-seal risk; value
# 150 = C7 AS AMENDED 09:57 EDT 2026-07-23, matching the certified eval seeds).
SENTINEL_LIVE_STAMP = '''
# --- (a) budget-sentinel LIVE stamp (C7-as-amended, prereg 2026-07-23) -------
# Scored regime is uncapped -> export the budget or the sentinel is inert.
os.environ["SENTINEL_BUDGET"] = "150"
print("sentinel-live: SENTINEL_BUDGET=150 game-envelope (C7-as-amended, prereg amendment 2026-07-23)")
print("sentinel-live: arm = duck + (f) continuation + budget sentinel ONLY; NO warpack, NO ledger")
'''

# war-v2 cell-2 addendum: log every raw signal the fast-submit gate keys on so
# a detection misfire (dummy parquet written in a REAL rerun -> 0-score day) is
# diagnosable post-hoc, and hard-fail on the one contradiction that is
# checkable (rerun signal set but gate resolved cold). Panel R11/R12
# llm-agents prior #6.
V2_GATE_SIGNALS = '''
# --- Gate detection-signal record (war-v2; llm-agents #6) --------------------
_gate_signals = {k: os.environ.get(k) for k in (
    "KAGGLE_IS_COMPETITION_RERUN", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")}
print(f"taaf.kaggle: gate signals {_gate_signals} "
      f"TRUE_SUBMISSION={TRUE_SUBMISSION} FORCE_OFFLINE_BENCH={FORCE_OFFLINE_BENCH} "
      f"RUN_HEAVY={RUN_HEAVY}")
assert not (_gate_signals["KAGGLE_IS_COMPETITION_RERUN"] and not RUN_HEAVY), \\
    "fast-submit gate misfire: competition rerun signalled but RUN_HEAVY=False"
'''

V2_LEDGER_HEADER = (
    "\n\n# ==== war-v2 ledger graft (flags {ledger, escalation} ON; prereg §4 "
    "design lock) ====\n"
    "# Gated on RUN_HEAVY: `bm` only exists on the heavy path; the fast-submit\n"
    "# save must stay seconds-fast and ledger-free.\n"
    "if RUN_HEAVY:\n"
)

# war-v2 LEDGER CANARY (panel R12 N6; observability ONLY). Appended INSIDE the
# same `if RUN_HEAVY:` suite, after the ledger hook. Counts, per game:
#   attempts = ledger-patched _build_user_prompt calls
#   digests  = prompts that actually carry the injected digest+protocol block
#   skips    = calls with no ledger bound to the agent (injection skipped)
#   aborts   = ledger bound but the injection hook swallowed an exception
#   escalations = Ledger.escalations_fired (per-game under ledger v2 keying)
# The TOTAL line's stores= count is the keying tripwire: under ledger v2
# (per-game runtime-state-stem keying) stores == number of games; stores=1
# with many games means the shared-store v1 regression is back.
# It wraps the ledger-patched ToolAgent methods AFTER ledger_patch.apply();
# counters only — it never changes prompts, actions, or timing. Kill switches:
# LEDGER_CANARY=0 disarms the canary; LEDGER_FLAGS="" already no-ops the whole
# graft (canary then stays disarmed). The lock below guards ONLY dict ops —
# never call ledger/agent code while holding it (forge_v35 deadlock lesson).
V2_CANARY = '''
# --- war-v2 LEDGER CANARY (panel R12 N6; observability only) -----------------
LEDGER_CANARY_ON = os.environ.get("LEDGER_CANARY", "1").strip().lower() not in {"0", "false"}
_ledger_cfg = dict(getattr(sys.modules.get("ledger_patch"), "_CFG", {}) or {})
if not (LEDGER_CANARY_ON and _ledger_cfg.get("ledger")):
    print(f"ledger canary: DISARMED (canary_on={LEDGER_CANARY_ON} ledger_cfg={_ledger_cfg})")
else:
    try:
        import threading as _canary_threading

        import ledger_patch as _lp
        from inference.agent.tool_agent import ToolAgent as _CanaryToolAgent

        LEDGER_CANARY_STATS = {}
        _CANARY_LOCK = _canary_threading.Lock()
        _CANARY_MARKER = _lp.PROTOCOL_LINES[:48]

        def _canary_bump(game, field):
            # Lock guards ONLY the dict ops below - never call out under it.
            with _CANARY_LOCK:
                st = LEDGER_CANARY_STATS.setdefault(
                    game, {"attempts": 0, "digests": 0, "skips": 0, "aborts": 0})
                st[field] += 1

        _canary_prev_analyze = _CanaryToolAgent.analyze

        def _canary_analyze(self, state_path, action_num, **kwargs):
            try:  # per-game label from the runtime-state filename stem
                self._canary_game = Path(state_path).name.split("_runtime_state")[0]
            except Exception:  # noqa: BLE001 - label is best-effort
                self._canary_game = "unknown"
            return _canary_prev_analyze(self, state_path, action_num, **kwargs)

        _CanaryToolAgent.analyze = _canary_analyze

        _canary_prev_build_prompt = _CanaryToolAgent._build_user_prompt

        def _canary_build_user_prompt(self, action_num, **kwargs):
            game = getattr(self, "_canary_game", "unbound")
            led = getattr(self, "_ledger_state", None)
            _canary_bump(game, "attempts")
            prompt = _canary_prev_build_prompt(self, action_num, **kwargs)
            if led is None:
                _canary_bump(game, "skips")
            elif isinstance(prompt, str) and _CANARY_MARKER in prompt:
                _canary_bump(game, "digests")
            else:
                _canary_bump(game, "aborts")
            return prompt

        _CanaryToolAgent._build_user_prompt = _canary_build_user_prompt

        def ledger_canary_report():
            """Greppable per-game summary; called by the run cell after the bench."""
            with _lp._LEDGERS_LOCK:
                registry = dict(_lp._LEDGERS)
            with _CANARY_LOCK:
                stats = {g: dict(st) for g, st in LEDGER_CANARY_STATS.items()}
            esc_total = sum(int(getattr(led, "escalations_fired", 0))
                            for led in registry.values())
            if not stats:
                print("LEDGER CANARY game=none attempts=0 digests=0 skips=0 "
                      "aborts=0 escalations=0", flush=True)
            for game in sorted(stats):
                st = stats[game]
                print(f"LEDGER CANARY game={game} attempts={st['attempts']} "
                      f"digests={st['digests']} skips={st['skips']} "
                      f"aborts={st['aborts']} escalations={esc_total}", flush=True)
            print(f"LEDGER CANARY TOTAL games={len(stats)} stores={len(registry)} "
                  f"escalations_total={esc_total} "
                  f"attempts={sum(s['attempts'] for s in stats.values())} "
                  f"digests={sum(s['digests'] for s in stats.values())} "
                  f"skips={sum(s['skips'] for s in stats.values())} "
                  f"aborts={sum(s['aborts'] for s in stats.values())}", flush=True)

        print("ledger canary: ARMED (per-game counters; LEDGER_CANARY=0 to disable)")
    except Exception:  # noqa: BLE001 - observability must never fail the run
        traceback.print_exc()
        print("ledger canary: FAILED to arm - continuing WITHOUT canary (graft unaffected)")
'''

# war-v2 cell-14 addendum: print the greppable canary summary after the bench.
# Guarded so a vanilla/killed run prints a diagnosable line instead of raising.
V2_CANARY_SUMMARY = '''

# --- war-v2 LEDGER CANARY summary (greppable; armed in cell 12) --------------
if RUN_HEAVY:
    try:
        ledger_canary_report()  # noqa: F821 - defined in cell 12 when armed
    except NameError:
        print("LEDGER CANARY not armed (ledger off / canary disabled / patch failed)")
    except Exception as exc:  # noqa: BLE001 - observability must never fail the run
        print(f"LEDGER CANARY report failed: {exc!r}")
'''


def _indent(src: str) -> str:
    return "".join(
        ("    " + line if line.strip() else line.rstrip() + "\n")
        for line in src.splitlines(keepends=True)
    )


def _gate_source(src: str) -> str:
    return fastsubmit_cells.HEAVY_GATE_HEADER + _indent(src)


def build(diff_only: bool = False, v2: bool = False, continuation: bool = True,
          sentinel: bool = False) -> None:
    assert not (v2 and sentinel), "--v2 and --sentinel are mutually exclusive"
    import build_eval_notebook  # noqa: PLC0415 - same-dir sibling
    nb = json.loads(VANILLA.read_text(encoding="utf-8"))
    cells = nb["cells"]
    originals = {i: "".join(cells[i]["source"]) for i in (2, 4, 6, 8, 10, 12, 14)}

    cell2 = originals[2] + fastsubmit_cells.CELL2_GATE_ADDENDUM
    if sentinel:
        # one-flag discipline: NO warpack hook at all — cell 12 is the
        # sentinel graft (+ (f) continuation appended below by default).
        cell2 += SENTINEL_LIVE_STAMP
        cell12 = build_eval_notebook.CELL12_SENTINEL
    else:
        cell12 = (HERE / "hook_cell.py").read_text(encoding="utf-8")
    cell14 = fastsubmit_cells.CELL14_SOURCE
    if v2:
        cell2 += V2_GATE_SIGNALS
        cell12 += (V2_LEDGER_HEADER
                   + _indent(LEDGER_HOOK.read_text(encoding="utf-8"))
                   + _indent(V2_CANARY))
        cell14 += V2_CANARY_SUMMARY
    if continuation:
        # (f) default ON in all builds since 2026-07-23 (prereg amendment
        # item 4). Appended last, at top level (order-independent of the
        # warpack/ledger/sentinel monkeypatches: (f) rewrites the
        # PYTHON_ADDENDUM prompt constant). Kill switch CONTINUATION_DISABLE=1
        # stays.
        cell12 += build_eval_notebook.CONTINUATION_DEFAULT_BLOCK
        print("cell-12: (f) continuation graft APPENDED (default ON since "
              "2026-07-23; --no-continuation to omit)")
    else:
        print("cell-12: (f) continuation graft OMITTED (--no-continuation)")

    new_sources: dict[int, str] = {
        2: cell2,
        12: cell12,
        14: cell14,
    }
    for i in HEAVY_CELLS:
        new_sources[i] = _gate_source(originals[i])

    for i, src in new_sources.items():
        cells[i]["source"] = src.splitlines(keepends=True)
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None

    for i in sorted(new_sources):
        diff = difflib.unified_diff(
            originals[i].splitlines(keepends=True),
            new_sources[i].splitlines(keepends=True),
            fromfile=f"vanilla/cell{i}",
            tofile=f"{'warpack-v2' if v2 else 'warpack'}/cell{i}",
        )
        print("".join(diff))

    out = OUT_SENTINEL if sentinel else (OUT_V2 if v2 else OUT)
    if not diff_only:
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {out}")
        if sentinel:
            stem = KERNEL_ID_SENTINEL.split("/", 1)[1]
            meta = json.loads(SRC_WAR_META.read_text(encoding="utf-8"))
            meta["id"] = KERNEL_ID_SENTINEL
            meta["title"] = stem
            meta["code_file"] = f"{stem}.ipynb"
            NB_DIR_SENTINEL.mkdir(exist_ok=True)
            (NB_DIR_SENTINEL / f"{stem}.ipynb").write_text(
                json.dumps(nb, indent=1), encoding="utf-8")
            (NB_DIR_SENTINEL / "kernel-metadata.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {NB_DIR_SENTINEL}/{stem}.ipynb (+metadata, id={KERNEL_ID_SENTINEL})")


if __name__ == "__main__":
    build(diff_only="--diff-only" in sys.argv[1:], v2="--v2" in sys.argv[1:],
          continuation="--no-continuation" not in sys.argv[1:],
          sentinel="--sentinel" in sys.argv[1:])
