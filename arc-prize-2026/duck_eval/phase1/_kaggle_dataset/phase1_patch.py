"""Phase-1 patch: graft the exploration substrate onto the Cottaar duck harness.

Applied by calling ``apply()`` from the duckfork notebook's customization-hook
cell (cell 12), after ``bm`` is unpickled and the bundled sources are on
``sys.path``. All patches are class/module-level monkeypatches, so they affect
the already-pickled ``bm.solver`` (methods resolve at call time).

What gets patched (see PATCH_NOTES.md for the full map):
  1. ToolAgent.analyze                    -> no-progress tracking + scripted
                                             harness-side explore() trigger
  2. ToolAgent._build_user_prompt         -> inject curated <=500-token explore
                                             report + 1-line archive status
  3. ToolAgent._compact_action_result     -> pass animation_summary through
  4. ToolAgent._trim_messages_for_context -> eviction hysteresis (evict-to-
                                             low-watermark) for vLLM prefix cache
  5. tool_agent.run_sandboxed_python      -> pre-load `explore_archive` REPL var
  6. tool_agent._PYTHON_TOOL_DESCRIPTION  -> document `explore_archive`
  7. python_tool_sandbox._SANDBOX_BOOTSTRAP -> expose `explore_archive` global
  8. _HarnessGameSession._execute_action  -> per-action archive feed +
                                             animation-diff text summaries

NOT changed (Tufa-proven keeps): no UNDO tool, 4x-upscale single image per
turn, 64k server / 32k analyzer eviction scheme, single `python` tool.
explore() is NOT an LLM-electable tool (their tools-hinder-creativity result).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # package-relative first (dataset layout), then flat (sys.path insert)
    from . import phase1_core as core
except ImportError:  # pragma: no cover
    import phase1_core as core

log = logging.getLogger("phase1")

# Patch-layer version. v2 = gated explore trigger (mode detector + level-up
# cooldown; spec: learnings/v2_gating_design_2026-07-11.md "Patch spec").
VERSION = "v2"

_TLS = threading.local()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class Phase1Config:
    # N no-progress analyzer turns before a scripted explore fires.
    # v2: kept at 10 = p100 of 295 organic streaks (Q4, mining report
    # 2026-07-11); raising it would disable explore entirely.
    explore_after_turns: int = _env_int("PHASE1_EXPLORE_AFTER_TURNS", 10)
    # Real actions per scripted explore (v2: 8 -> 6).
    explore_probe_budget: int = _env_int("PHASE1_EXPLORE_BUDGET", 6)
    # Hard cap on scripted explores per game (v2: 6 -> 3; worst-case
    # contamination 18 actions/run, was 48).
    max_explores_per_game: int = _env_int("PHASE1_MAX_EXPLORES", 3)
    # v2 mode detector (Q1): actions taken on the current level (tracked via
    # level-up action indices) must reach this before explore may fire; on
    # level 0 this is simply action_num >= 90 (p90 of 113 vanilla
    # time-to-first-level runs). Probe cost is ~free past this point.
    explore_min_level_actions: int = _env_int("PHASE1_EXPLORE_MIN_LEVEL_ACTIONS", 90)
    # v2 momentum guard (Q2/Q4): suppress explore within this many analyzer
    # turns of a level-up (level-ups land at streak 0, never after stalls).
    explore_levelup_cooldown: int = _env_int("PHASE1_EXPLORE_LEVELUP_COOLDOWN", 20)
    # MOUSE click candidates per explore (salient-object centroids).
    mouse_candidates: int = _env_int("PHASE1_MOUSE_CANDIDATES", 4)
    # Curated summary cap: 500 tokens ~ 1500 chars at duck's ~3 chars/token.
    summary_char_cap: int = _env_int("PHASE1_SUMMARY_CHAR_CAP", 1500)
    animation_char_cap: int = _env_int("PHASE1_ANIMATION_CHAR_CAP", 240)
    # Skip explore when the per-game clock is nearly spent.
    min_time_remaining: float = _env_float("PHASE1_MIN_TIME_REMAINING", 300.0)
    # v2: only fire scripted explores while the run's best observed level is
    # <= this (frame.level starts at 1, so 1 = "hasn't cleared level 1 yet").
    # Rationale (3-seed v1 gate FAIL, 2026-07-10): explore's real actions
    # deflate RHAE on games duck already progresses in (higher levels carry
    # higher RHAE weight), while every per-game win came from games whose
    # null never clears level 1. Default 99 = v1 behavior.
    explore_max_level: int = _env_int("PHASE1_EXPLORE_MAX_LEVEL", 99)
    # Eviction hysteresis: when context trimming triggers, trim down to this
    # fraction of the budget so the message prefix stays byte-stable (and
    # vLLM-prefix-cached) for many turns between evictions.
    evict_low_frac: float = _env_float("PHASE1_EVICT_LOW_FRAC", 0.5)
    # Signature noise floor (cells); 1 = every component counts.
    noise_floor: int = _env_int("PHASE1_NOISE_FLOOR", 1)
    # Feature switches (for ablations).
    enable_explore: bool = _env_bool("PHASE1_ENABLE_EXPLORE", True)
    enable_animation: bool = _env_bool("PHASE1_ENABLE_ANIMATION", True)
    enable_repl_archive: bool = _env_bool("PHASE1_ENABLE_REPL_ARCHIVE", True)
    enable_evict_hysteresis: bool = _env_bool("PHASE1_ENABLE_EVICT_HYSTERESIS", True)


class Phase1State:
    """Per-game (per ToolAgent session) exploration state."""

    def __init__(self, runtime_dir: Path, config: Phase1Config) -> None:
        self.runtime_dir = runtime_dir
        self.config = config
        self.archive = core.DedupArchive(noise_floor=config.noise_floor)
        self.progress = core.ProgressTracker()
        self.explores_done = 0
        self.pending_summary: str | None = None
        self.last_probes: list[dict[str, Any]] = []
        self.current_sig: str = ""
        self.last_analysis_step: int | None = None
        # RLock: the explore loop holds it while step_env re-enters the
        # per-action hook, which acquires it again on the same thread.
        self.lock = threading.RLock()


def explore_gate_open(cfg: Phase1Config, p1: Phase1State, action_num: int) -> bool:
    """v2 trigger conjunction (pure part; plumbing guards live in the wrapper).

    Fires a scripted explore only when ALL hold (v2_gating_design_2026-07-11):
      1. no-progress streak >= explore_after_turns (10 = p100 organic),
      2. (levels_completed == 0 AND action_num >= explore_min_level_actions)
         OR actions_on_current_level >= explore_min_level_actions,
      3. no level-up within the last explore_levelup_cooldown analyzer turns
         (or no level-up yet),
      4. explores_done < max_explores_per_game and best level within
         explore_max_level.
    Inert when PHASE1_ENABLE_EXPLORE=0.
    """
    if not cfg.enable_explore:
        return False
    prog = p1.progress
    if prog.turns_without_progress < cfg.explore_after_turns:
        return False
    stalled_on_level = (
        prog.levelups == 0 and int(action_num or 0) >= cfg.explore_min_level_actions
    ) or prog.actions_on_current_level >= cfg.explore_min_level_actions
    if not stalled_on_level:
        return False
    if prog.levelups > 0 and prog.turns_since_levelup < cfg.explore_levelup_cooldown:
        return False
    if p1.archive.best_level > cfg.explore_max_level:
        return False
    if p1.explores_done >= cfg.max_explores_per_game:
        return False
    return True


def _get_phase1(agent: Any, state_path: Path, config: Phase1Config) -> Phase1State:
    runtime_dir = state_path.parent
    state = getattr(agent, "_phase1_state", None)
    if state is None or state.runtime_dir != runtime_dir:
        state = Phase1State(runtime_dir, config)
        agent._phase1_state = state
    return state


_APPLIED = False


def apply(bm: Any = None, config: Phase1Config | None = None) -> Phase1Config:
    """Apply all Phase-1 monkeypatches. Idempotent. Returns the active config."""
    global _APPLIED
    cfg = config or Phase1Config()
    if _APPLIED:
        log.warning("phase1 patches already applied; skipping re-apply")
        return cfg

    import inference.agent.python_tool_sandbox as sandbox_mod
    import inference.agent.tool_agent as tool_agent_mod
    import inference.framework.solver as solver_mod
    from inference.agent.runtime_state import load_runtime_state
    from inference.agent.tool_agent import ToolAgent, _normalize_valid_actions

    # ------------------------------------------------------------------
    # (7) sandbox bootstrap: expose `explore_archive` as a REPL global
    # ------------------------------------------------------------------
    if cfg.enable_repl_archive:
        anchor = 'runtime_globals["last_action_result"] = action_result'
        bootstrap = sandbox_mod._SANDBOX_BOOTSTRAP
        if anchor not in bootstrap:
            raise RuntimeError("phase1: sandbox bootstrap anchor not found")
        pos = bootstrap.index(anchor)
        line_start = bootstrap.rfind("\n", 0, pos) + 1
        indent = bootstrap[line_start:pos]
        injection = (
            anchor
            + "\n"
            + indent
            + 'runtime_globals["explore_archive"] = state_payload.get("explore_archive") or {}'
        )
        sandbox_mod._SANDBOX_BOOTSTRAP = bootstrap.replace(anchor, injection, 1)

    # ------------------------------------------------------------------
    # (5) run_sandboxed_python wrapper: inject archive snapshot into the
    #     initial state and into every post-action state refresh
    # ------------------------------------------------------------------
    _orig_run_sandboxed = tool_agent_mod.run_sandboxed_python

    def phase1_run_sandboxed_python(*, code, timeout_seconds, initial_state, action_handler):
        p1: Phase1State | None = getattr(_TLS, "phase1", None)
        if p1 is not None and cfg.enable_repl_archive:
            def _snap() -> dict[str, Any]:
                with p1.lock:
                    return p1.archive.snapshot(
                        top_k=5,
                        current_sig=p1.current_sig,
                        no_progress_turns=p1.progress.turns_without_progress,
                        last_probes=p1.last_probes,
                    )

            initial_state = dict(initial_state)
            initial_state["explore_archive"] = _snap()
            inner_handler = action_handler

            def wrapped_handler(actions):
                payload = inner_handler(actions)
                try:
                    state = payload.get("state")
                    if isinstance(state, dict):
                        state["explore_archive"] = _snap()
                except Exception:  # noqa: BLE001 - never break the tool loop
                    pass
                return payload

            action_handler = wrapped_handler
        return _orig_run_sandboxed(
            code=code,
            timeout_seconds=timeout_seconds,
            initial_state=initial_state,
            action_handler=action_handler,
        )

    tool_agent_mod.run_sandboxed_python = phase1_run_sandboxed_python

    # ------------------------------------------------------------------
    # (6) tool description: document the new REPL variable
    # ------------------------------------------------------------------
    if cfg.enable_repl_archive and "explore_archive" not in tool_agent_mod._PYTHON_TOOL_DESCRIPTION:
        tool_agent_mod._PYTHON_TOOL_DESCRIPTION += (
            " Also available: `explore_archive`, a read-only dict maintained by the harness"
            " (deduped state archive: unique_states, current_state_untried, frontier"
            " with novelty/(1+return_cost) scores, last_explore_probes)."
        )

    # ------------------------------------------------------------------
    # (3) pass animation_summary through the compact action result
    # ------------------------------------------------------------------
    _orig_compact = ToolAgent._compact_action_result

    def phase1_compact_action_result(self, payload):
        compact = _orig_compact(self, payload)
        if payload.get("animation_summary"):
            compact["animation_summary"] = payload["animation_summary"]
        return compact

    ToolAgent._compact_action_result = phase1_compact_action_result

    # ------------------------------------------------------------------
    # (4) eviction hysteresis for vLLM prefix caching
    # ------------------------------------------------------------------
    _orig_trim = ToolAgent._trim_messages_for_context

    def phase1_trim_messages_for_context(
        self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0
    ):
        if not cfg.enable_evict_hysteresis or not messages:
            return _orig_trim(
                self,
                messages,
                tools=tools,
                preserve_recent=preserve_recent,
                extra_safety_tokens=extra_safety_tokens,
            )
        budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        estimate = self._estimate_request_input_tokens(messages, tools=tools)
        if estimate <= budget:
            # Under budget: original is a no-op trim; keep ordering append-only.
            return _orig_trim(
                self,
                messages,
                tools=tools,
                preserve_recent=preserve_recent,
                extra_safety_tokens=extra_safety_tokens,
            )
        # Over budget: evict down to the low watermark in one batch, so the
        # (system + retained history) prefix then stays stable for many turns.
        low_frac = min(0.95, max(0.1, cfg.evict_low_frac))
        hysteresis_tokens = int(self._context_budget_tokens * (1.0 - low_frac))
        return _orig_trim(
            self,
            messages,
            tools=tools,
            preserve_recent=preserve_recent,
            extra_safety_tokens=extra_safety_tokens + hysteresis_tokens,
        )

    ToolAgent._trim_messages_for_context = phase1_trim_messages_for_context

    # ------------------------------------------------------------------
    # (2) user prompt: inject pending explore report + archive status line
    # ------------------------------------------------------------------
    _orig_build_prompt = ToolAgent._build_user_prompt

    def phase1_build_user_prompt(self, action_num, **kwargs):
        prompt = _orig_build_prompt(self, action_num, **kwargs)
        p1: Phase1State | None = getattr(self, "_phase1_state", None)
        if p1 is None:
            return prompt
        extra: list[str] = []
        with p1.lock:
            if p1.pending_summary:
                extra.append(p1.pending_summary)
                p1.pending_summary = None
            if cfg.enable_repl_archive:
                extra.append(
                    f"Exploration archive: {len(p1.archive.states)} unique states,"
                    f" {p1.progress.turns_without_progress} turns since last new state"
                    " (details in `explore_archive` inside python)."
                )
        if extra:
            prompt = prompt + "\n" + "\n".join(extra)
        return prompt

    ToolAgent._build_user_prompt = phase1_build_user_prompt

    # ------------------------------------------------------------------
    # (1) analyze wrapper: progress tracking + scripted explore trigger
    # ------------------------------------------------------------------
    _orig_analyze = ToolAgent.analyze

    def phase1_analyze(
        self,
        state_path,
        action_num,
        valid_actions=None,
        step_env=None,
        **kwargs,
    ):
        try:
            self._ensure_session(state_path)
            p1 = _get_phase1(self, Path(state_path), cfg)
            should_stop = kwargs.get("should_stop")
            current_frame, history = load_runtime_state(Path(state_path))
            model_actions = _normalize_valid_actions(valid_actions)
            analysis_step = kwargs.get("analysis_step")
            if current_frame is not None:
                with p1.lock:
                    obs = p1.archive.observe(
                        current_frame.grid,
                        level=current_frame.level,
                        step=len(history),
                        available_actions=model_actions,
                    )
                    p1.current_sig = obs["sig"]
                    # Count one progress turn per analysis step: retries and
                    # 60s control-yields re-enter analyze with the same step.
                    if analysis_step is None or analysis_step != p1.last_analysis_step:
                        p1.last_analysis_step = analysis_step
                        p1.progress.update(
                            state_count=len(p1.archive.states),
                            level=p1.archive.best_level,
                            action_num=action_num,
                        )
            if (
                step_env is not None
                and current_frame is not None
                and model_actions
                and explore_gate_open(cfg, p1, action_num)
                and not (should_stop is not None and should_stop())
            ):
                _run_scripted_explore(
                    self, p1, Path(state_path), model_actions, step_env, should_stop
                )
        except Exception as exc:  # noqa: BLE001 - never break the turn
            log.warning("phase1 pre-analyze hook failed: %s", exc)

        _TLS.phase1 = getattr(self, "_phase1_state", None)
        try:
            return _orig_analyze(
                self,
                state_path,
                action_num,
                valid_actions=valid_actions,
                step_env=step_env,
                **kwargs,
            )
        finally:
            _TLS.phase1 = None

    def _run_scripted_explore(agent, p1, state_path, model_actions, step_env, should_stop):
        def get_state():
            frame, hist = load_runtime_state(state_path)
            if frame is None:
                return ((), 0, 0)
            return (frame.grid, frame.level, len(hist))

        raw_payloads: list[dict[str, Any]] = []

        def execute(action: dict[str, Any]) -> dict[str, Any]:
            payload = step_env({"actions": [action]})
            if isinstance(payload, dict):
                raw_payloads.append(payload)
                return payload
            return {}

        with p1.lock:
            probes = core.run_explore(
                execute=execute,
                get_state=get_state,
                valid_actions=model_actions,
                archive=p1.archive,
                budget=cfg.explore_probe_budget,
                mouse_candidates=cfg.mouse_candidates,
                min_time_remaining=cfg.min_time_remaining,
                should_stop=should_stop,
            )
            if not probes:
                return
            trigger_turns = p1.progress.turns_without_progress
            p1.pending_summary = core.render_explore_summary(
                probes,
                p1.archive,
                trigger_turns=trigger_turns,
                char_cap=cfg.summary_char_cap,
            )
            p1.last_probes = probes
            p1.explores_done += 1
            p1.progress.reset()
        # Keep the agent's own last-sequence narration coherent with reality.
        try:
            summary = agent._summarize_step_sequence(raw_payloads)
            if summary:
                agent._last_step_summary = summary
                agent._update_summarized_knowledge_from_step_summary()
        except Exception:  # noqa: BLE001
            pass
        log.info(
            "phase1 explore #%d ran %d probes (game dir %s)",
            p1.explores_done,
            len(probes),
            p1.runtime_dir,
        )

    ToolAgent.analyze = phase1_analyze

    # ------------------------------------------------------------------
    # (8) per-action archive feed + animation-diff summaries (harness side)
    # ------------------------------------------------------------------
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def phase1_execute_action(
        self,
        action,
        *,
        batch_index,
        batch_size,
        generated_tokens=None,
        flush_viewer_payload=True,
    ):
        try:
            before_grid = solver_mod._grid_from_state(self.game.current_state)
        except Exception:  # noqa: BLE001
            before_grid = None
        payload = _orig_execute_action(
            self,
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        try:
            state = self.game.current_state
            after_grid = solver_mod._grid_from_state(state)
            if cfg.enable_animation and before_grid is not None:
                anim = state.animation_frames
                if anim:
                    frames = [before_grid]
                    for frame in anim[:64]:
                        data = frame.data
                        frames.append(data.tolist() if hasattr(data, "tolist") else data)
                    frames.append(after_grid)
                    text = core.summarize_animation(frames, char_cap=cfg.animation_char_cap)
                    if text:
                        payload["animation_summary"] = text
            p1: Phase1State | None = getattr(self.analyzer, "_phase1_state", None)
            if p1 is not None and before_grid is not None:
                with p1.lock:
                    sig_before = p1.archive.signature(before_grid)
                    p1.archive.observe(
                        after_grid,
                        level=int(payload.get("level") or 0),
                        step=int(payload.get("action_num") or 0),
                        available_actions=payload.get("valid_actions") or [],
                    )
                    p1.archive.mark_tried(
                        sig_before, str(payload.get("action_display") or "")
                    )
        except Exception as exc:  # noqa: BLE001 - never break the action path
            log.debug("phase1 execute_action hook failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = phase1_execute_action

    _APPLIED = True
    log.info(
        "phase1 %s patches applied: explore_after_turns=%d budget=%d max_explores=%d "
        "min_level_actions=%d levelup_cooldown=%d "
        "animation=%s repl_archive=%s evict_hysteresis=%s(low_frac=%.2f)",
        VERSION,
        cfg.explore_after_turns,
        cfg.explore_probe_budget,
        cfg.max_explores_per_game,
        cfg.explore_min_level_actions,
        cfg.explore_levelup_cooldown,
        cfg.enable_animation,
        cfg.enable_repl_archive,
        cfg.enable_evict_hysteresis,
        cfg.evict_low_frac,
    )
    return cfg
