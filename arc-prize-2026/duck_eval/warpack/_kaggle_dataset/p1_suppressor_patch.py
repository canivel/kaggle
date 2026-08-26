"""P1 - zero-information action suppressor (runner-side).

Spec: ``learnings/war_room/efficiency_diagnosis_2026-08-12.md`` sec5 P1.
Prereg: ``learnings/war_room/p1_prereg_2026-08-12.md`` (SEALED before this file
was pushed to an eval kernel).

THE DEFECT (diagnosis sec4). On a 225-action level the agent runs with
``context_budget_tokens=31744`` / ``history_messages=33``. It therefore FORGETS
what it already tried and re-derives from compacted prose. The harness already
exposes ``history``/``transitions``/``last_transition`` as preloaded Python
globals - the ground truth exists and the agent never queries it. 10.5% of the
actions on cleared levels re-execute a ``(board, action)`` pair already executed
on that level, and 17.6% are fired inside a batch that had already gone dead.

THE FIX (this module), three mechanisms, all runner-side, all game-agnostic:

  A. MEMO DECLINE. Per level, keep ``(board_hash, action_key) -> outcome``.
     When the model asks for a pair the runner has already CONFIRMED, do not
     spend the action: return the memoised outcome plus a one-line note.
  B. BATCH ABORT. Stop the remainder of a multi-action batch the moment an
     action no-ops (or closes a loop inside that same batch). Generalises the
     stock ``stopped_early`` trigger, which fired on only 10/190 batches.
  C. NON-TRUNCATABLE MEMORY BLOCK. A small fixed-schema block appended to the
     user message every turn (the user message is rebuilt each turn and is
     never trimmed): current board fingerprint, states seen this level, untried
     primitives here, known-dead pairs here. This is the load-bearing part -
     the diagnosis root-caused re-exploration to forgetting, not to a missing
     loop detector.

HARD SAFETY CONSTRAINT (diagnosis sec5, non-negotiable): mechanism A must be
disabled on games with latent/ambiguous transitions. Implemented as a GENERAL
ONLINE DETECTOR, never a game-id list (competition legality): if any executed
``(board_hash, action_key)`` ever produces an outcome that differs from the
memoised one, the game is flagged AMBIGUOUS and mechanism A is permanently
disabled for the rest of that game. Verified on our traces to reproduce the
published latent-state set exactly on ``runs/kernel_pulls/animation_v1``
(m0r0 55, re86 19, sk48 11, ka59 10, cd82 8, g50t 4, dc22 3, wa30 2 ambiguous
pairs; zero for the other 17 games) - see ``duck_eval/warpack/p1_smoke.py``.

SECOND SAFETY CONSTRAINT (found by this build, NOT in the diagnosis - see the
prereg sec4). On the recorded traces the level-completing batch of tu93 L1,
sp80 L1 and ar25 L1 OPENS by re-traversing already-visited states (the agent
walks back to a known position and then plays the winning move). Therefore:

  * ``P1_ABORT_REVISIT`` (abort a batch when an action lands on an
    already-visited board) DESTROYS 3 of the 17 cleared levels on
    ``animation_v1``. It ships DEFAULT OFF.
  * ``P1_MEMO_MODE=all`` (decline any repeated pair) declines the opening
    moves of those same batches. It ships DEFAULT OFF; the default is
    ``noop``, which declines only pairs whose CONFIRMED outcome left the board
    byte-identical. Declining a confirmed no-op is board-equivalent to
    executing it, so mechanism A cannot move the agent off its path.

Verified offline (duck_eval/warpack/p1_replay_validate.py) over three recorded
runs: with the shipped defaults, zero level-completing actions are declined or
aborted and zero board-changing actions are declined, on all three runs.

House pattern (mirrors animation_patch / compaction_patch / continuation_patch):
VERSION marker, arm flag + kill switch, blanket-guarded ``apply()``, runtime
banner, ``bm.label`` stamp, greppable events, canary counters, per-session state
(no globals, no locks, no threads), **vanilla fallback on ANY failure**.

Seams (4, all harness-side, zero LLM calls, zero prompt-file edits):
  1. ``solver._HarnessGameSession.play``        -> bind state to the analyzer
  2. ``solver._HarnessGameSession._execute_action`` -> A + B + the record
  3. ``solver._HarnessGameSession.step_env``    -> batch bookkeeping / notes
  4. ``tool_agent.ToolAgent._build_user_prompt``-> C (the memory block)
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

log = logging.getLogger("p1")

VERSION = "v1"
_EVENT_V = "1"

_APPLIED = False


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _flag_on() -> bool:
    return os.environ.get("P1_SUPPRESS", "").strip() == "1"


def _kill_switch() -> bool:
    return os.environ.get("P1_DISABLE", "").strip() == "1"


class Config:
    """Read live on every use so every knob is a runtime kill switch."""

    @property
    def memo(self) -> bool:
        return _env_bool("P1_MEMO", True)

    @property
    def memo_mode(self) -> str:
        mode = os.environ.get("P1_MEMO_MODE", "noop").strip().lower()
        return mode if mode in {"noop", "all"} else "noop"

    @property
    def confirm(self) -> int:
        # Executions of a pair required before it may be declined. MUST be >= 2:
        # at confirm=1 no pair is ever executed twice, so the online ambiguity
        # detector can never fire and the hard safety constraint is void.
        return max(2, _env_int("P1_CONFIRM", 2))

    @property
    def max_declines(self) -> int:
        # A pair may be declined at most this many times per level; the next
        # request executes normally, so no path can ever be permanently blocked.
        return max(1, _env_int("P1_MAX_DECLINES", 1))

    @property
    def abort(self) -> bool:
        return _env_bool("P1_ABORT", True)

    @property
    def abort_noop_streak(self) -> int:
        return max(1, _env_int("P1_ABORT_NOOP_STREAK", 1))

    @property
    def abort_cycle(self) -> bool:
        return _env_bool("P1_ABORT_CYCLE", True)

    @property
    def abort_revisit(self) -> bool:
        # DEFAULT OFF - destroys 3/17 cleared levels on the recorded traces.
        return _env_bool("P1_ABORT_REVISIT", False)

    @property
    def block(self) -> bool:
        return _env_bool("P1_BLOCK", True)

    @property
    def block_max_dead(self) -> int:
        return max(0, _env_int("P1_BLOCK_MAX_DEAD", 8))


CFG = Config()


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def board_fingerprint(grid: Any) -> str:
    """Stable short digest of a normalised board (tuple-of-tuples or list)."""
    try:
        payload = repr([[int(c) for c in row] for row in grid]).encode()
    except Exception:  # noqa: BLE001 - a fingerprint may never raise
        payload = repr(grid).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


class MemoEntry:
    __slots__ = ("out", "n", "noop", "declines")

    def __init__(self, out: str, noop: bool) -> None:
        self.out = out
        self.n = 1
        self.noop = noop
        self.declines = 0


class P1State:
    """Per-``_HarnessGameSession`` state. One game-pass, one thread, no locks."""

    def __init__(self, game: str = "?") -> None:
        self.game = game
        self.level = 0
        self.memo: dict[tuple[str, str], MemoEntry] = {}
        self.visited: set[str] = set()
        self.tried_here: dict[str, dict[str, bool]] = {}   # hash -> act -> noop?
        self.last_hash: str | None = None
        # per-game latent-state detector
        self.ambiguous = False
        self.ambiguity_events = 0
        self.ambiguity_pairs: set[tuple[str, str]] = set()
        # batch bookkeeping
        self.batch_dead = False
        self.batch_states: set[str] = set()
        self.batch_notes: list[str] = []
        self.batch_declined = 0
        self.batch_aborted = 0
        # canary counters
        self.actions_executed = 0
        self.declined = 0
        self.aborted = 0
        self.dup_requests = 0
        self.dup_executed = 0
        self.levels_seen = 0
        self.errors = 0

    # -- level scoping ----------------------------------------------------- #
    def sync_level(self, level: int) -> None:
        if level == self.level:
            return
        self.level = level
        self.levels_seen += 1
        self.memo.clear()
        self.visited.clear()
        self.tried_here.clear()
        self.batch_states.clear()
        self.last_hash = None

    # -- mechanism A ------------------------------------------------------- #
    def should_decline(self, key: tuple[str, str]) -> MemoEntry | None:
        if not CFG.memo or self.ambiguous:
            return None
        ent = self.memo.get(key)
        if ent is None:
            return None
        if ent.n < CFG.confirm:
            return None
        if ent.declines >= CFG.max_declines:
            return None
        if CFG.memo_mode == "noop" and not ent.noop:
            return None
        return ent

    # -- record ------------------------------------------------------------ #
    def record(self, key: tuple[str, str], out: str, noop: bool) -> bool:
        """Record an executed transition. Returns True if it revealed latent
        state (outcome differs from the memoised one)."""
        ent = self.memo.get(key)
        drift = False
        if ent is None:
            self.memo[key] = MemoEntry(out, noop)
        else:
            self.dup_executed += 1
            if ent.out != out:
                drift = True
                self.ambiguity_events += 1
                self.ambiguity_pairs.add(key)
                self.ambiguous = True
            ent.n += 1
            ent.noop = ent.noop and noop
        self.tried_here.setdefault(key[0], {})[key[1]] = noop
        self.visited.add(out)
        self.batch_states.add(out)
        self.last_hash = out
        return drift

    # -- mechanism C ------------------------------------------------------- #
    def memory_block(self, valid_actions: list[str] | None) -> str:
        if not CFG.block:
            return ""
        here = self.last_hash
        if here is None:
            return ""
        tried = self.tried_here.get(here, {})
        lines = [
            "P1 memory (runner ground truth from the full transition record; "
            "never truncated - trust this over your recollection):",
            f"- board fingerprint {here}; {len(self.visited)} distinct board(s) "
            f"seen on this level; {len(self.memo)} (board,action) pair(s) recorded.",
        ]
        if valid_actions:
            untried = [a for a in valid_actions if a not in tried]
            if untried:
                lines.append(f"- NOT YET TRIED from this exact board: {', '.join(untried[:12])}.")
            else:
                lines.append("- every listed primitive has already been tried from this exact board.")
        dead_here = sorted(a for a, noop in tried.items() if noop)
        if dead_here:
            cap = CFG.block_max_dead
            shown = dead_here[:cap]
            more = f" (+{len(dead_here) - cap} more)" if len(dead_here) > cap else ""
            lines.append(
                f"- CONFIRMED NO EFFECT from this exact board: {', '.join(shown)}{more}. "
                f"Re-issuing one of these is not spent and tells you nothing new."
            )
        if self.ambiguous:
            lines.append(
                f"- this game has latent state ({len(self.ambiguity_pairs)} pair(s) gave "
                f"different outcomes from the same board); repeats are NOT suppressed here."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# abort signal
# --------------------------------------------------------------------------- #
class _P1BatchAbort(Exception):
    """Raised from ``_execute_action`` to break the vanilla batch loop.

    Vanilla ``step_env`` catches ``Exception`` inside the loop and breaks with
    ``stop_reason='action_error'`` when at least one action already executed -
    which is always the case here, because the batch can only be marked dead by
    an action that DID execute. ``step_env`` rewrites the reason afterwards.
    """


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def _game_label(session: Any) -> str:
    for getter in (
        lambda: session.game.env.environment_info.game_id,
        lambda: session.game.game_id,
    ):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            continue
    return "?"


def _emit(kind: str, st: P1State, detail: str) -> None:
    print(
        f"P1 v={_EVENT_V} kind={kind} game={st.game} level={st.level} "
        f"declined={st.declined} aborted={st.aborted} "
        f"dup_req={st.dup_requests} dup_exec={st.dup_executed} "
        f"amb={1 if st.ambiguous else 0} amb_pairs={len(st.ambiguity_pairs)} "
        f"{detail}",
        flush=True,
    )


CANARY: dict[str, dict[str, Any]] = {}


def canary_report() -> dict[str, Any]:
    """End-of-run canary. Prints one greppable line and returns the numbers."""
    games = sorted(CANARY)
    tot = {k: sum(CANARY[g].get(k, 0) for g in games)
           for k in ("actions_executed", "declined", "aborted",
                     "dup_requests", "dup_executed", "errors")}
    ambiguous = sorted(g for g in games if CANARY[g].get("ambiguous"))
    report = {
        "version": VERSION,
        "games": len(games),
        "ambiguous_games": ambiguous,
        "ambiguous_pairs": {g: CANARY[g]["ambiguity_pairs"] for g in ambiguous},
        **tot,
    }
    charged = tot["actions_executed"] or 1
    report["dup_rate"] = tot["dup_executed"] / charged
    print(
        f"P1 CANARY v={_EVENT_V} version={VERSION} games={len(games)} "
        f"executed={tot['actions_executed']} declined={tot['declined']} "
        f"aborted={tot['aborted']} dup_exec={tot['dup_executed']} "
        f"dup_rate={report['dup_rate']:.4f} errors={tot['errors']} "
        f"ambiguous_games={','.join(ambiguous) or 'NONE'} "
        f"mode={CFG.memo_mode} confirm={CFG.confirm} "
        f"abort_revisit={int(CFG.abort_revisit)}",
        flush=True,
    )
    return report


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
def _apply_patches() -> int:
    global _APPLIED
    if _APPLIED:
        return 0

    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    patched = 0

    def _state(session: Any) -> P1State:
        st = getattr(session, "_p1_state", None)
        if st is None:
            st = P1State(_game_label(session))
            session._p1_state = st
        return st

    # --- seam 1: bind the state to the analyzer for mechanism C ------------
    _orig_play = solver_mod._HarnessGameSession.play

    def p1_play(self):
        st = _state(self)
        try:
            self.analyzer._p1_state = st
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("p1: analyzer bind failed: %s", exc)
        try:
            return _orig_play(self)
        finally:
            try:
                CANARY[st.game] = {
                    "actions_executed": st.actions_executed,
                    "declined": st.declined,
                    "aborted": st.aborted,
                    "dup_requests": st.dup_requests,
                    "dup_executed": st.dup_executed,
                    "errors": st.errors,
                    "ambiguous": st.ambiguous,
                    "ambiguity_pairs": len(st.ambiguity_pairs),
                }
                _emit("game_end", st, "")
            except Exception:  # noqa: BLE001
                pass

    solver_mod._HarnessGameSession.play = p1_play
    patched += 1

    # --- seam 2: mechanism A + B + the transition record -------------------
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def p1_execute_action(self, action, *, batch_index, batch_size,
                          generated_tokens=None, flush_viewer_payload=True):
        st = _state(self)
        prev_hash = None
        key = None
        try:
            state = self.game.current_state
            st.sync_level(int(state.levels_completed))
            prev_hash = board_fingerprint(solver_mod._grid_from_state(state))
            if st.last_hash is None:
                st.last_hash = prev_hash
                st.visited.add(prev_hash)
                st.batch_states.add(prev_hash)
            key = (prev_hash, solver_mod._format_action_display(
                action.id.name, dict(action.data)))
        except Exception as exc:  # noqa: BLE001 - never break the action path
            st.errors += 1
            log.debug("p1: pre-action bookkeeping failed: %s", exc)

        # --- B: this batch already died -----------------------------------
        if key is not None and batch_size > 1 and st.batch_dead and CFG.abort:
            # The vanilla loop breaks on the first raise, so this fires once per
            # batch; count every action the batch will NOT spend, so the canary
            # is comparable with the offline replay.
            saved = max(1, batch_size - batch_index + 1)
            st.aborted += saved
            st.batch_aborted += saved
            _emit("batch_abort", st,
                  f"action={key[1]} bi={batch_index}/{batch_size} saved={saved}")
            raise _P1BatchAbort(
                "P1: the rest of this batch was not executed because an earlier "
                "action in it produced no board change, so every later action "
                "was fired blind. Re-plan from the board you can now see."
            )

        # --- A: memo decline ----------------------------------------------
        if key is not None:
            ent = None
            if key in st.memo:
                st.dup_requests += 1
                ent = st.should_decline(key)
            if ent is not None:
                ent.declines += 1
                st.declined += 1
                st.batch_declined += 1
                note = (
                    f"P1: '{key[1]}' from this exact board is already recorded as "
                    f"having NO EFFECT ({ent.n} observation(s)). The action was NOT "
                    f"spent and the board is unchanged. Try something else."
                )
                st.batch_notes.append(note)
                _emit("decline", st, f"action={key[1]} n={ent.n}")
                payload = _declined_payload(self, solver_mod, action, note,
                                            batch_index, batch_size)
                return payload

        payload = _orig_execute_action(
            self, action,
            batch_index=batch_index, batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )

        # --- record + B triggers ------------------------------------------
        try:
            st.actions_executed += 1
            out_hash = board_fingerprint(
                solver_mod._grid_from_state(self.game.current_state))
            noop = not bool(payload.get("board_changed"))
            cycle = out_hash in st.batch_states
            revisit = out_hash in st.visited
            if key is not None:
                drift = st.record(key, out_hash, noop)
                if drift:
                    _emit("latent_state", st, f"action={key[1]}")
            if CFG.abort and batch_size > 1 and not st.batch_dead:
                kill = False
                if noop and CFG.abort_noop_streak <= 1:
                    kill = True
                elif noop:
                    st._noop_streak = getattr(st, "_noop_streak", 0) + 1
                    kill = st._noop_streak >= CFG.abort_noop_streak
                else:
                    st._noop_streak = 0
                if not kill and CFG.abort_cycle and cycle and not noop:
                    kill = True
                if not kill and CFG.abort_revisit and revisit:
                    kill = True
                if kill and not payload.get("level_completed") \
                        and not payload.get("run_complete"):
                    st.batch_dead = True
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("p1: post-action bookkeeping failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = p1_execute_action
    patched += 1

    # --- seam 3: batch bookkeeping ----------------------------------------
    _orig_step_env = solver_mod._HarnessGameSession.step_env

    def p1_step_env(self, arguments):
        st = _state(self)
        st.batch_dead = False
        st.batch_states = set()
        st.batch_notes = []
        st.batch_declined = 0
        st.batch_aborted = 0
        st._noop_streak = 0
        if st.last_hash is not None:
            st.batch_states.add(st.last_hash)
        payload = _orig_step_env(self, arguments)
        try:
            if isinstance(payload, dict):
                if st.batch_declined or st.batch_aborted:
                    payload["p1_declined"] = st.batch_declined
                    payload["p1_aborted"] = st.batch_aborted
                    payload["p1_note"] = " ".join(st.batch_notes) or (
                        "P1: the remainder of the batch was not executed - an "
                        "earlier action in it produced no board change, so the "
                        "rest was fired blind."
                    )
                    if st.batch_aborted:
                        payload["stop_reason"] = "p1_batch_aborted"
                        payload["stopped_early"] = True
                        payload.pop("error", None)
                    elif payload.get("stop_reason") == "action_error":
                        payload["stop_reason"] = "p1_suppressed"
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("p1: step_env bookkeeping failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession.step_env = p1_step_env
    patched += 1

    # --- seam 4: mechanism C, the non-truncatable block --------------------
    _orig_build_user_prompt = ToolAgent._build_user_prompt

    def p1_build_user_prompt(self, action_num, *, valid_actions=None, **kwargs):
        text = _orig_build_user_prompt(
            self, action_num, valid_actions=valid_actions, **kwargs)
        try:
            st = getattr(self, "_p1_state", None)
            if isinstance(st, P1State):
                block = st.memory_block(valid_actions)
                if block:
                    return f"{text}\n{block}"
        except Exception as exc:  # noqa: BLE001 - the prompt may never break
            log.debug("p1: memory block failed: %s", exc)
        return text

    ToolAgent._build_user_prompt = p1_build_user_prompt
    patched += 1

    # --- seam 4b: carry the note through the compactor ---------------------
    _orig_compact = ToolAgent._compact_action_result

    def p1_compact_action_result(self, payload):
        compact = _orig_compact(self, payload)
        try:
            if isinstance(payload, dict) and payload.get("p1_note"):
                compact["p1_note"] = payload["p1_note"]
                for k in ("p1_declined", "p1_aborted"):
                    if payload.get(k):
                        compact[k] = payload[k]
        except Exception as exc:  # noqa: BLE001
            log.debug("p1: compact passthrough failed: %s", exc)
        return compact

    ToolAgent._compact_action_result = p1_compact_action_result

    _APPLIED = True
    return patched


def _declined_payload(session, solver_mod, action, note, batch_index, batch_size):
    """Vanilla-shaped payload for an action the runner refused to spend.

    Mechanism A only declines pairs whose CONFIRMED outcome left the board
    byte-identical, so the board reported here is the true current board and
    ``board_changed=False`` is the truth, not a simulation.
    """
    game = session.game
    state = game.current_state
    raw_state = state.raw.state
    display = solver_mod._format_action_display(action.id.name, dict(action.data))
    return {
        "executed": False,
        "suppressed": True,
        "p1_note": note,
        "action_num": session.action_count,
        "level": solver_mod._level_number(game),
        "score": int(state.levels_completed),
        "reward": 0.0,
        "state": raw_state.name,
        "valid_actions": solver_mod.to_model_actions(
            solver_mod._engine_action_names(game)),
        "board_changed": False,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "action_name": action.id.name,
        "action_display": display,
        "batch_index": batch_index,
        "batch_size": batch_size,
        **session.timing_payload(),
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install P1 v1. True on success (or already applied); False on flag-off /
    kill switch / any failure - in which case NOTHING is changed (vanilla duck).

    Arm flag:    P1_SUPPRESS=1
    Kill switch: P1_DISABLE=1
    Sub-flags:   P1_MEMO / P1_MEMO_MODE(noop|all) / P1_CONFIRM / P1_MAX_DECLINES
                 P1_ABORT / P1_ABORT_NOOP_STREAK / P1_ABORT_CYCLE
                 P1_ABORT_REVISIT (default OFF - see the module docstring)
                 P1_BLOCK / P1_BLOCK_MAX_DEAD
    """
    if _kill_switch():
        log.info("p1 %s: P1_DISABLE=1 -> no-op", VERSION)
        return False
    if not _flag_on():
        log.info("p1 %s: P1_SUPPRESS!=1 -> no-op (flag-gated arm)", VERSION)
        return False
    try:
        patched = _apply_patches()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-p1-{VERSION}"
        print(
            f"p1 {VERSION}: ACTIVE ({patched} seams patched) - zero-information "
            f"action suppressor. "
            f"A: memo decline mode={CFG.memo_mode} confirm={CFG.confirm} "
            f"max_declines={CFG.max_declines} "
            f"(online latent-state detector disables A per game on first "
            f"contradictory outcome; NO game-id list). "
            f"B: batch abort on no-op"
            f"{' + intra-batch cycle' if CFG.abort_cycle else ''}"
            f"{' + revisit' if CFG.abort_revisit else ''} "
            f"(revisit is DEFAULT OFF: it cuts the level-completing batch on "
            f"tu93/sp80/ar25 in the recorded traces). "
            f"C: non-truncatable memory block "
            f"{'ON' if CFG.block else 'OFF'} (max_dead={CFG.block_max_dead}). "
            f"Zero LLM calls, no locks, no game-id logic, vanilla fallback.",
            flush=True,
        )
        log.info("p1 %s installed (%d seams)", VERSION, patched)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[p1] apply failed -> stock duck harness (vanilla)")
        return False
