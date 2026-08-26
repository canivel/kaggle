"""Phase-1 CPU smoke test — no GPU, no LLM.

Run from the repo root:
    uv run python duck_eval/phase1/smoke_test.py

Covers:
  T1  signature stability / translation sensitivity / noise floor
  T2  dedup archive + frontier scoring + progress tracker
  T3  animation-diff summarizer (synthetic + real sb26 engine frames)
  T4  scripted explore() against local engines (kaggle-data/environment_files)
  T5  patch application on the real taaf-bundle modules + REPL archive
      injection through the real sandbox
  T6  v2 explore gating (mode detector + level-up cooldown + tightened caps;
      spec: learnings/v2_gating_design_2026-07-11.md "Patch spec")
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # f:/kaggle/arc-prize-2026
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "smoke-test-model")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

import phase1_core as core  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def grid_with_rect(base: int, r0: int, c0: int, h: int, w: int, color: int, size: int = 16):
    grid = [[base] * size for _ in range(size)]
    for r in range(r0, min(size, r0 + h)):
        for c in range(c0, min(size, c0 + w)):
            grid[r][c] = color
    return grid


# ---------------------------------------------------------------- T1 signature
def t1_signature() -> None:
    print("T1: segmentation-graph signature")
    g1 = grid_with_rect(0, 2, 2, 3, 3, 5)
    g1b = grid_with_rect(0, 2, 2, 3, 3, 5)
    g2 = grid_with_rect(0, 2, 3, 3, 3, 5)   # translated 1 col
    g3 = grid_with_rect(0, 2, 2, 3, 3, 7)   # recolored
    s1 = core.state_signature(g1)
    check("hash deterministic (same grid twice)", s1 == core.state_signature(g1b))
    check("hash is 16 hex chars", len(s1) == 16 and all(ch in "0123456789abcdef" for ch in s1))
    check("translation changes signature", s1 != core.state_signature(g2))
    check("recolor changes signature", s1 != core.state_signature(g3))
    # noise floor: 1-px sparkle ignored at floor 2, seen at floor 1
    g4 = [row[:] for row in g1]
    g4[10][10] = 9
    check("sparkle changes sig at floor 1", core.state_signature(g1) != core.state_signature(g4))
    check(
        "sparkle ignored at floor 2",
        core.state_signature(g1, noise_floor=2) == core.state_signature(g4, noise_floor=2),
    )
    # shape hash is translation invariant
    check(
        "shape_hash translation-invariant",
        core.shape_hash([(2, 2), (2, 3), (3, 2)]) == core.shape_hash([(7, 5), (7, 6), (8, 5)]),
    )
    # delta description
    delta = core.grid_delta(g1, g3)
    check("grid_delta counts recolored cells", delta["count"] == 9, str(delta))
    check("grid_delta transition labels", delta["transitions"][0]["from"] == "5" and delta["transitions"][0]["to"] == "7")


# ---------------------------------------------------------------- T2 archive
def t2_archive() -> None:
    print("T2: dedup archive + frontier + progress tracker")
    archive = core.DedupArchive()
    g_a = grid_with_rect(0, 1, 1, 2, 2, 3)
    g_b = grid_with_rect(0, 5, 5, 2, 2, 3)
    obs1 = archive.observe(g_a, level=1, step=0, available_actions=["UP", "DOWN", "RESET"])
    obs2 = archive.observe(g_a, level=1, step=1, available_actions=["UP", "DOWN"])
    obs3 = archive.observe(g_b, level=1, step=4, available_actions=["UP", "DOWN", "MOUSE"])
    check("first observation is new", obs1["new"] is True)
    check("repeat observation deduped", obs2["new"] is False and len(archive.states) == 2)
    check("distinct state is new", obs3["new"] is True)
    check("RESET excluded from untried", "RESET" not in archive.states[obs1["sig"]].untried)

    archive.mark_tried(obs1["sig"], "UP")
    check("mark_tried removes from untried", archive.untried_for(obs1["sig"]) == ["DOWN"])
    archive.mark_tried(obs1["sig"], "MOUSE(row=3, col=4)")
    check("mark_tried strips MOUSE coords", "MOUSE" in archive.states[obs1["sig"]].tried)

    # frontier: state at shorter prefix with more untried scores higher
    f = archive.frontier(top_k=5)
    check("frontier non-empty", len(f) >= 2, json.dumps(f))
    check(
        "frontier ordering by novelty/(1+cost)",
        f[0]["score"] >= f[-1]["score"] and f == sorted(f, key=lambda e: -e["score"]),
        json.dumps(f),
    )
    e_a = archive.states[obs1["sig"]]
    check(
        "frontier score formula",
        abs(archive.frontier_score(e_a) - len(e_a.untried) / (1 + e_a.prefix_len)) < 1e-9,
    )

    snap = archive.snapshot(current_sig=obs3["sig"], no_progress_turns=3)
    json.dumps(snap)  # must be JSON-safe
    check("snapshot JSON-safe with expected keys",
          snap["unique_states"] == 2 and snap["no_progress_turns"] == 3 and "frontier" in snap)

    tracker = core.ProgressTracker()
    p1 = tracker.update(state_count=1, level=0)
    p2 = tracker.update(state_count=1, level=0)
    p3 = tracker.update(state_count=1, level=0)
    check("tracker counts no-progress turns", (p1, p2, p3) == (True, False, False) and tracker.turns_without_progress == 2)
    p4 = tracker.update(state_count=2, level=0)
    check("new state resets counter", p4 is True and tracker.turns_without_progress == 0)
    tracker.update(state_count=2, level=0)
    p6 = tracker.update(state_count=2, level=1)
    check("level-up counts as progress", p6 is True and tracker.turns_without_progress == 0)


# ---------------------------------------------------------------- T3 animation
def t3_animation() -> None:
    print("T3: animation-diff summarizer")
    # synthetic: 3px dot moving down 10 rows over 10 intermediate frames
    frames = []
    for i in range(12):
        g = [[0] * 32 for _ in range(32)]
        r = 2 + i
        for dc in range(3):
            g[r][10 + dc] = 4
        frames.append(g)
    text = core.summarize_animation(frames)
    check("synthetic summary non-empty", bool(text), repr(text))
    check("mentions intermediate frame count", "10 intermediate frames" in text, text)
    check("reports drift", "drifted" in text, text)
    check("reports color flips", "0->4" in text or "4->0" in text, text)
    check("within char cap", len(text) <= 240, str(len(text)))
    check("no animation -> empty", core.summarize_animation(frames[:2]) == "")
    check("static frames -> empty", core.summarize_animation([frames[0]] * 5) == "")

    # real engine frames: sb26 ACTION5 emits a long animation
    try:
        import arc_agi

        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_FILES)
        )
        sb26_id = next(e.game_id for e in arcade.available_environments if e.game_id.startswith("sb26"))
        env = arcade.make(sb26_id)
        f0 = env.reset()
        before = [list(row) for row in f0.frame[-1]]
        import arcengine

        f1 = env.step(arcengine.GameAction.from_id(5))
        layers = [[list(r) for r in layer] for layer in f1.frame]
        check("sb26 ACTION5 returns intermediate frames", len(layers) > 1, f"layers={len(layers)}")
        real_text = core.summarize_animation([before, *layers], char_cap=240)
        check("sb26 animation summary non-empty", bool(real_text), repr(real_text))
        check("sb26 summary within cap", len(real_text) <= 240, str(len(real_text)))
        print(f"    sb26 sample: {real_text}")
    except Exception as exc:  # noqa: BLE001
        check("sb26 real-engine animation", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- T4 explore
class _LocalEngineDriver:
    """step_env-style executor over an offline arc_agi environment."""

    def __init__(self, game_prefix: str):
        import arc_agi
        import arcengine

        self._arcengine = arcengine
        self._arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_FILES)
        )
        gid = next(
            e.game_id for e in self._arcade.available_environments if e.game_id.startswith(game_prefix)
        )
        self.env = self._arcade.make(gid)
        self.frame = self.env.reset()
        self.steps = 0
        self.executed_actions: list[str] = []

    def state(self):
        grid = [list(row) for row in self.frame.frame[-1]]
        level = int(self.frame.levels_completed or 0) + 1
        return grid, level, self.steps

    def valid_model_actions(self) -> list[str]:
        from inference.agent.action_names import to_model_actions

        names = []
        for action_id in self.frame.available_actions or []:
            try:
                name = self._arcengine.GameAction.from_id(int(action_id)).name
            except Exception:  # noqa: BLE001
                continue
            if name != "RESET":
                names.append(name)
        return to_model_actions(names)

    def execute(self, action: dict) -> dict:
        from inference.agent.action_names import to_engine_action

        engine_name = to_engine_action(action.get("action"))
        if engine_name is None:
            return {"executed": False, "error": f"unknown action {action}"}
        game_action = self._arcengine.GameAction.from_name(engine_name)
        data = None
        if engine_name == "ACTION6":
            data = {"x": int(action.get("col", 0)), "y": int(action.get("row", 0))}
        before_levels = int(self.frame.levels_completed or 0)
        new_frame = self.env.step(game_action, data=data)
        if new_frame is None:
            return {"executed": False, "error": "engine returned None"}
        self.frame = new_frame
        self.steps += 1
        self.executed_actions.append(engine_name)
        state_name = str(new_frame.state)
        return {
            "executed": True,
            "valid_actions": self.valid_model_actions(),
            "level_completed": int(new_frame.levels_completed or 0) > before_levels,
            "game_over": "GAME_OVER" in state_name,
            "run_complete": "WIN" in state_name,
            "done": "WIN" in state_name,
            "time_remaining_seconds": 9999.0,
        }


def t4_explore() -> None:
    print("T4: scripted explore() against local engines")
    for prefix in ("ls20", "sb26"):
        try:
            driver = _LocalEngineDriver(prefix)
            archive = core.DedupArchive()
            grid, level, step = driver.state()
            valid = driver.valid_model_actions()
            check(f"{prefix}: engine exposes valid actions", bool(valid), str(valid))
            probes = core.run_explore(
                execute=driver.execute,
                get_state=driver.state,
                valid_actions=valid,
                archive=archive,
                budget=8,
                mouse_candidates=3,
                min_time_remaining=300.0,
            )
            executed = [p for p in probes if p.get("executed")]
            check(f"{prefix}: probes executed", len(executed) >= 1, json.dumps(probes)[:300])
            check(f"{prefix}: within budget", len(probes) <= 8, str(len(probes)))
            check(
                f"{prefix}: only valid actions probed",
                all(p["action"].split("(")[0] in set(valid) for p in probes),
                str([p["action"] for p in probes]),
            )
            check(f"{prefix}: archive grew", len(archive.states) >= 1, str(len(archive.states)))
            terminal = any(
                p.get(flag) for p in probes for flag in ("level_completed", "game_over", "run_complete")
            )
            check(
                f"{prefix}: dedup marks tried actions",
                any(archive.states[s].tried for s in archive.states) or terminal,
            )
            summary = core.render_explore_summary(probes, archive, trigger_turns=10)
            check(f"{prefix}: summary <= 1500 chars", 0 < len(summary) <= 1500, str(len(summary)))
            check(
                f"{prefix}: summary format",
                summary.startswith("[HARNESS EXPLORATION REPORT]")
                and "Archive:" in summary
                and "explore_archive" in summary,
            )
            if prefix == "ls20":
                print("    --- ls20 sample summary ---")
                print("    " + summary.replace("\n", "\n    "))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            check(f"{prefix}: explore run", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- T5 patches
def t5_patches() -> None:
    print("T5: patch application on real taaf-bundle modules")
    try:
        import phase1_patch

        cfg = phase1_patch.apply()
        import inference.agent.python_tool_sandbox as sandbox_mod
        import inference.agent.tool_agent as tool_agent_mod

        check(
            "sandbox bootstrap exposes explore_archive",
            'runtime_globals["explore_archive"]' in sandbox_mod._SANDBOX_BOOTSTRAP,
        )
        check(
            "tool description documents explore_archive",
            "explore_archive" in tool_agent_mod._PYTHON_TOOL_DESCRIPTION,
        )
        check("apply is idempotent", phase1_patch.apply() is not None)

        agent = tool_agent_mod.ToolAgent(model="smoke-test-model")

        # hysteresis: build an over-budget message list and confirm one trim
        # call cuts well below budget (not just barely under it)
        filler = "x" * 400  # ~134 tokens each at the /3 estimator
        messages = [{"role": "system", "content": "sys"}]
        for i in range(600):
            messages.append({"role": "user", "content": f"u{i} {filler}"})
            messages.append({"role": "assistant", "content": f"a{i} {filler}"})
        budget = agent._context_budget_tokens
        est_before = agent._estimate_request_input_tokens(messages)
        check("test fixture exceeds budget", est_before > budget, f"{est_before} <= {budget}")
        trimmed = agent._trim_messages_for_context(messages)
        est_after = agent._estimate_request_input_tokens(trimmed)
        check(
            "eviction hysteresis trims to low watermark",
            est_after <= budget * (cfg.evict_low_frac + 0.05),
            f"after={est_after} budget={budget} low_frac={cfg.evict_low_frac}",
        )
        # under budget: no trimming at all
        small = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        check("under-budget messages untouched", agent._trim_messages_for_context(small) == small)

        # REPL injection through the real sandbox
        p1 = phase1_patch.Phase1State(Path("."), cfg)
        p1.archive.observe(grid_with_rect(0, 1, 1, 2, 2, 3), level=1, step=0, available_actions=["UP"])
        p1.current_sig = next(iter(p1.archive.states))
        phase1_patch._TLS.phase1 = p1
        try:
            result = tool_agent_mod.run_sandboxed_python(
                code="result = {'n': explore_archive.get('unique_states'), 'keys': sorted(explore_archive)}",
                timeout_seconds=20,
                initial_state={
                    "current_frame": None,
                    "history": [],
                    "valid_actions": ["UP"],
                    "last_action_result": {},
                },
                action_handler=lambda actions: {"action_result": {}, "state": {}},
            )
        finally:
            phase1_patch._TLS.phase1 = None
        check(
            "sandbox REPL sees explore_archive",
            not result.get("error") and (result.get("result") or {}).get("n") == 1,
            json.dumps(result)[:400],
        )

        # compact action result passes animation_summary through
        compact = agent._compact_action_result(
            {"executed": True, "animation_summary": "ANIMATION (3 intermediate frames): ..."}
        )
        check("compact result keeps animation_summary", "animation_summary" in compact)

        # solver-side patch is installed
        import inference.framework.solver as solver_mod

        check(
            "session._execute_action is patched",
            solver_mod._HarnessGameSession._execute_action.__name__ == "phase1_execute_action",
        )
        check(
            "ToolAgent.analyze is patched",
            tool_agent_mod.ToolAgent.analyze.__name__ == "phase1_analyze",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        check("patch application", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- T6 v2 gating
def t6_v2_gating() -> None:
    print("T6: v2 explore gating (mode detector + level-up cooldown)")
    try:
        import phase1_patch

        cfg = phase1_patch.Phase1Config()
        check("v2 VERSION marker", getattr(phase1_patch, "VERSION", None) == "v2",
              str(getattr(phase1_patch, "VERSION", None)))
        check("v2 default probe budget = 6", cfg.explore_probe_budget == 6,
              str(cfg.explore_probe_budget))
        check("v2 default max explores = 3", cfg.max_explores_per_game == 3,
              str(cfg.max_explores_per_game))
        check("v2 default min level actions = 90", cfg.explore_min_level_actions == 90,
              str(cfg.explore_min_level_actions))
        check("v2 default levelup cooldown = 20", cfg.explore_levelup_cooldown == 20,
              str(cfg.explore_levelup_cooldown))
        check("v2 keeps explore_after_turns = 10", cfg.explore_after_turns == 10,
              str(cfg.explore_after_turns))

        gate = phase1_patch.explore_gate_open

        def fresh_state(streak: int, action_num: int) -> "phase1_patch.Phase1State":
            """Level-0-completed run: one archived state at level 1 (baseline),
            then `streak` no-progress analyzer turns at `action_num` actions."""
            p1 = phase1_patch.Phase1State(Path("."), cfg)
            p1.archive.observe(
                grid_with_rect(0, 1, 1, 2, 2, 3), level=1, step=0,
                available_actions=["UP", "DOWN"],
            )
            p1.progress.update(state_count=1, level=1, action_num=1)  # baseline turn
            for _ in range(streak):
                p1.progress.update(state_count=1, level=1, action_num=action_num)
            return p1

        # (a) does NOT fire before action 90 on level 0, even at streak 10+
        p1 = fresh_state(12, 50)
        check("(a) closed at action 50 / streak 12 / level 0", not gate(cfg, p1, 50),
              f"streak={p1.progress.turns_without_progress}")
        check("(a) baseline level observation is not a level-up",
              p1.progress.levelups == 0, str(p1.progress.levelups))

        # (b) fires at action >= 90 + streak >= 10 + no level-up yet
        p1 = fresh_state(12, 95)
        check("(b) open at action 95 / streak 12 / no level-up", gate(cfg, p1, 95))
        p1 = fresh_state(4, 95)
        check("(b) still closed at streak 4 despite action 95", not gate(cfg, p1, 95),
              f"streak={p1.progress.turns_without_progress}")

        # (c) suppressed within 20 analyzer turns of a level-up
        p1 = fresh_state(0, 100)
        p1.progress.update(state_count=1, level=2, action_num=100)  # real level-up
        check("(c) level-up tracked", p1.progress.levelups == 1
              and p1.progress.turns_since_levelup == 0
              and p1.progress.last_levelup_action == 100,
              f"levelups={p1.progress.levelups} tsl={p1.progress.turns_since_levelup}")
        for _ in range(12):
            p1.progress.update(state_count=1, level=2, action_num=250)
        check("(c) actions_on_current_level from level-up index",
              p1.progress.actions_on_current_level == 150,
              str(p1.progress.actions_on_current_level))
        check("(c) suppressed 12 turns after level-up (cooldown 20)",
              not gate(cfg, p1, 250),
              f"tsl={p1.progress.turns_since_levelup} streak={p1.progress.turns_without_progress}")
        for _ in range(8):
            p1.progress.update(state_count=1, level=2, action_num=250)
        check("(c) opens once cooldown elapses (20 turns)", gate(cfg, p1, 250),
              f"tsl={p1.progress.turns_since_levelup}")
        # cooldown elapsed but <90 actions on the new level -> still closed
        p2 = fresh_state(0, 100)
        p2.progress.update(state_count=1, level=2, action_num=100)
        for _ in range(25):
            p2.progress.update(state_count=1, level=2, action_num=120)
        check("(c) closed with only 20 actions on the new level",
              not gate(cfg, p2, 120),
              f"aocl={p2.progress.actions_on_current_level}")

        # (d) max 3 explores enforced
        p1 = fresh_state(12, 95)
        p1.explores_done = 3
        check("(d) closed at explores_done=3", not gate(cfg, p1, 95))
        p1.explores_done = 2
        check("(d) open at explores_done=2", gate(cfg, p1, 95))

        # kill switch: gates inert when PHASE1_ENABLE_EXPLORE=0
        cfg_off = phase1_patch.Phase1Config(enable_explore=False)
        p1 = fresh_state(12, 95)
        check("(kill) enable_explore=0 closes gate", not gate(cfg_off, p1, 95))

        # (e) probe budget 6: run_explore with the v2 default budget
        try:
            driver = _LocalEngineDriver("ls20")
            archive = core.DedupArchive()
            probes = core.run_explore(
                execute=driver.execute,
                get_state=driver.state,
                valid_actions=driver.valid_model_actions(),
                archive=archive,
                budget=cfg.explore_probe_budget,
                mouse_candidates=cfg.mouse_candidates,
                min_time_remaining=300.0,
            )
            check("(e) probes capped at budget 6", 0 < len(probes) <= 6, str(len(probes)))
        except Exception as exc:  # noqa: BLE001
            check("(e) budget-6 explore run", False, f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        check("v2 gating suite", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"phase1 smoke test | repo={REPO}")
    t1_signature()
    t2_archive()
    t3_animation()
    t4_explore()
    t5_patches()
    t6_v2_gating()
    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
