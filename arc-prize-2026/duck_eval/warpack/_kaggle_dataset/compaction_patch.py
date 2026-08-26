"""Compaction v2.1 — PURE region-aware eviction, digest-OFF (the A22 arm).

Design refs:
  - learnings/war_room/a22_compaction_v2_1_prereg_2026-08-06.md (SEALED v2.1
    intent; THE spec for this revision — written before this code. The ONE
    change vs v2: digest injection DISABLED — no digest message is ever
    rendered or injected, no reserve tokens are ever subtracted;
    reserve_applied=0 and digest_tokens=0 on every event. Gated behind
    COMPACTION_DIGEST, default 0; =1 restores v2 behavior. Everything else
    is inherited from v2 UNCHANGED: eviction order + pins, capture-into-store
    at both capture points, RETAIN-OFF default, stuck-suppress K=5, kill
    switch, vanilla fallback, zero LLM calls, NO locks. The anti-self-
    ingestion strip at ingest is KEPT — store hygiene, costless, no
    injection path.)
  - learnings/war_room/a22_compaction_v2_prereg_2026-08-04.md (SEALED v2 intent;
    the parent spec for the mechanism below)
  - learnings/war_room/a22_compaction_prereg_2026-08-01.md (v1 intent; M1/M2/M3
    and K1-K5 inherited verbatim)
  - learnings/sweeps/a22_seed1_screen_2026-08-03.md (v1 seed-1 FAIL evidence:
    toxic digest — hedged/truncated musings promoted to "do NOT re-verify"
    FACTs, refuted list collapsed to "+77 more", digest self-ingestion via the
    model's echoes; retained reasoning -> blind action batching)
  - daily brief 2026-08-03 §3/§4 research ADAPTs: the duck harness already IS
    the rolling-cut recency baseline; the axis is region-aware eviction with
    pinning vs recency (CWL 2606.11213, MemDecay 2607.10582); compaction wins
    are mostly BUDGET RELIEF, not summary prose (LightMem 2607.29104); suppress
    eviction while stuck (SelfCompact 2606.23525); zero LLM calls in the
    eviction path (Zero-Mem 2607.29377).

THE v2 MECHANISM (one flag, COMPACTION=1):

  1. REGION-AWARE EVICTION (primary; replaces digest-of-evicted-span). History
     is parsed into blocks (a user message, or an assistant message + its
     trailing tool results). Under token pressure, eviction selects by class,
     not recency:
       pinned (never evicted except last-resort): system prompt, the most
         recent scientist-note carrier (assistant msg with >=2 scientist-note
         labels), the most recent reasoning block, the preserve_recent tail;
       class 1: STALE action-episode blocks (all episodes but the newest —
         their board effects are already visible in the current frame, which
         rides in the newest user message) — oldest first;
       class 2: older user state-dump blocks (never the current-frame carrier;
         a head user block evicts as the full vanilla turn-cycle unit so the
         head-invariant never silently eats uncaptured/pinned messages);
       class 3: non-pinned assistant reasoning/text blocks;
       class 4: the newest episode block;
       class 5: last resort — pins yield rather than brick the request.
     Everything evicted at BOTH capture points (token trim + 30-turn
     persistence cap) is still folded into the mechanical store.

  2. DIGEST DEMOTED + HYGIENE-GATED — v2.1: OFF BY DEFAULT. The whole
     rendering/injection channel (and the reserve subtraction that funds it)
     runs ONLY under COMPACTION_DIGEST=1 (v2-behavior restore switch; NOT
     used in the v2.1 arm). Description below is the flag=1 behavior
     (render-time; ledger_core is untouched —
     shared library). Renders ONLY the refuted list (NEVER elided — no
     "+N more" collapsing, newest-first, budget-overflow drops oldest lines
     silently), hygiene-gated FACTs (complete declarative sentences only; no
     hedge prefixes "actually/wait/maybe/i think"; no mid-sentence truncation
     tails; no questions), and a small EVICTED/ACTION-EFFECTS/PROGRESS meta
     tail. NO ACTIVE / CONFIRMED hypothesis lines. Header softened ("treat as
     prior, not proof; re-testing is allowed") and marked non-quotable.
     Digest-shaped lines are stripped from evicted assistant text BEFORE
     extraction (breaks the sc25 self-ingestion round trip). If the gate
     empties the digest, NOTHING is injected — budget relief is the win — and
     the token reserve is only subtracted when a non-empty digest would
     actually be injected (kills v1's unconditional 3.2% window shrink).

  3. RETAIN DECOUPLED, OFF BY DEFAULT: COMPACTION_RETAIN defaults to 0 (v1
     defaulted to 1). The reasoning->reasoning_content outbound mirroring is
     installed ONLY when COMPACTION_RETAIN=1. v2 seed-1 runs with it OFF;
     retained_reasoning_msgs=0 on every event is the inverted RETAIN canary.

  4. SUPPRESS-CUT-WHILE-STUCK (deterministic, zero LLM): stuck := the last
     K=5 executed actions visible in the outbound tail all have
     board_changed=false (COMPACTION_STUCK_K override; K=5 sits between p95=3
     and p99=6 of the v1 no-change streak distribution, ~1.8% of action-time).
     While stuck: the 30-turn persistence cap is DEFERRED outright; no reserve
     is subtracted and no digest is injected (compaction never causes an
     eviction vanilla would not have made); budget-forced evictions still
     happen (physics — the request must fit), still region-aware, are
     captured, and emit NO event — counts flush into the next non-stuck event.

  5. KEPT FROM v1: one-flag graft, kill switch COMPACTION_DISABLE=1, blanket-
     guarded apply, runtime banner, vanilla fallback on ANY failure (worst
     case: stock duck), state on the ToolAgent INSTANCE (one thread per game),
     NO locks anywhere (deadlock lesson: forge_v35 scored 0.00), no game-id
     logic (graft rule), zero LLM calls in the eviction path. Events/sidecars
     gain fields: ev_episode / ev_user / ev_reasoning / ev_fallback
     (cumulative eviction-class counts), stuck_suppressed, reserve_applied,
     gated_facts, retain.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import ledger_core
except ImportError:  # local/interactive runs: the module ships next to us
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ledger_core

log = logging.getLogger("compaction")

VERSION = "v2.1"

# Event/sidecar version stamp ("2.1"); the schema itself is UNCHANGED vs v2 —
# digest_tokens/reserve_applied stay in every event and are always 0 with the
# digest channel off (the v2.1 canary greps exactly that).
_EVENT_V = VERSION.lstrip("v")

# Event grep anchor (EVENT_SCHEMA.md convention: one greppable stdout line per
# firing; parsers anchor on the first occurrence of this token, not column 0).
EVENT_ANCHOR = "COMPACTION "

# The digest message is identified (and stripped before re-render) by this
# prefix. It must stay stable across versions — drift would orphan digests.
DIGEST_MARKER = "=== COMPACTED HISTORY"
_DIGEST_HEADER = (
    DIGEST_MARKER
    + " v2 (mechanical digest of turns evicted from your context; previously "
    "observed evidence -- treat as prior, not proof; re-testing is allowed; "
    "internal memo -- do not quote or restate this digest in your reply) ==="
)

# Token budget reserved from a trim call for the injected digest — but ONLY
# when the store would actually inject a non-empty digest (v2; v1 subtracted
# it unconditionally, a pure 3.2% window loss on content-free events).
_DEFAULT_RESERVE_TOKENS = 1000

_DEFAULT_STUCK_K = 5

_APPLIED = False


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def _flag_on() -> bool:
    return os.environ.get("COMPACTION", "").strip() == "1"


def _kill_switch() -> bool:
    return os.environ.get("COMPACTION_DISABLE", "").strip() == "1"


def _retain_enabled() -> bool:
    # v2: RETAIN is decoupled and OFF by default (v1 defaulted ON). The
    # blind-batching harm channel (screen §5.3) is out of the tested mechanism.
    return os.environ.get("COMPACTION_RETAIN", "0").strip() == "1"


def _digest_enabled() -> bool:
    # v2.1: the digest injection channel is OFF by default (pure eviction —
    # the ONE change vs v2, prereg 2026-08-06 §1). COMPACTION_DIGEST=1
    # restores the full v2 behavior (hygiene-gated render + inject + reserve-
    # only-when-earned). With the default 0, no digest is ever rendered or
    # injected and no reserve is ever subtracted: reserve_applied=0 and
    # digest_tokens=0 on every event (the digest-OFF canary).
    return os.environ.get("COMPACTION_DIGEST", "0").strip() == "1"


def _reserve_tokens() -> int:
    raw = os.environ.get("COMPACTION_RESERVE_TOKENS", "").strip()
    try:
        v = int(raw) if raw else _DEFAULT_RESERVE_TOKENS
    except ValueError:
        v = _DEFAULT_RESERVE_TOKENS
    return max(200, v)


def _stuck_k() -> int:
    raw = os.environ.get("COMPACTION_STUCK_K", "").strip()
    try:
        v = int(raw) if raw else _DEFAULT_STUCK_K
    except ValueError:
        v = _DEFAULT_STUCK_K
    return max(2, v)


def _store_key(state_path: Path | None) -> str:
    """Per-game label from the runtime-state filename stem (mirrors
    budget_sentinel_patch._store_key / ledger_patch._ledger_key; used for
    event labels and the sidecar filename only — compaction state itself is
    per-agent-instance, which is already per-game)."""
    if state_path is None:
        return "unknown"
    p = Path(state_path)
    name = p.name
    i = name.find("runtime_state")
    stem = name[:i].rstrip("_.") if i >= 0 else p.stem
    return stem or (p.parent.name or "game")


# --------------------------------------------------------------------------- #
# digest hygiene (v2 — render-time gate; ledger_core extraction is untouched)
# --------------------------------------------------------------------------- #
_HEDGE_PREFIX_RE = re.compile(r"^\W*(?:actually|wait|maybe|i think)\b",
                              re.IGNORECASE)
_TRAILING_JUNK = "\"'”’)]} \t"


def _fact_hygiene_ok(statement: str) -> bool:
    """FACT gate: complete declarative sentences only. Rejects hedge-prefixed
    candidates ("actually", "wait", "maybe", "i think" — case-insensitive),
    mid-sentence truncation tails (no terminal '.'/'!') and questions."""
    s = (statement or "").strip()
    if len(s) < 6:
        return False
    if _HEDGE_PREFIX_RE.match(s):
        return False
    tail = s.rstrip(_TRAILING_JUNK)
    return tail.endswith(".") or tail.endswith("!")


# Digest-shaped lines are stripped from evicted assistant text BEFORE
# extraction, so the model's echo of an injected digest can never be
# re-harvested as a new record (the sc25 FACT-F5-quotes-FACT-F3 round trip).
_DIGEST_ECHO_RE = re.compile(
    r"^\W{0,8}(?:FACT\s+F\d+\s*:|(?:REFUTED|ACTIVE|CONFIRMED)\s+H\d+\b|"
    + re.escape(DIGEST_MARKER) + r")",
    re.IGNORECASE)


def _strip_digest_echoes(text: str) -> str:
    if not text:
        return text
    return "\n".join(line for line in text.split("\n")
                     if not _DIGEST_ECHO_RE.match(line))


# --------------------------------------------------------------------------- #
# stuck rubric (v2 §2.4 — deterministic, parsed from the outbound tail)
# --------------------------------------------------------------------------- #
def _recent_board_flags(messages: list[dict[str, Any]]) -> list[bool]:
    flags: list[bool] = []
    for message in messages:
        if str(message.get("role", "")).strip() != "tool":
            continue
        content = message.get("content", "")
        text = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=True, default=str)
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        for result in _walk_action_results(payload):
            flags.append(bool(result.get("board_changed")))
    return flags


def _is_stuck(messages: list[dict[str, Any]], k: int) -> bool:
    """stuck := the last k executed actions visible in the message tail all
    produced no board change (fewer than k observed results => not stuck)."""
    try:
        flags = _recent_board_flags(messages)
    except Exception as exc:  # noqa: BLE001 - rubric is best-effort
        log.debug("compaction stuck rubric failed: %s", exc)
        return False
    return len(flags) >= k and not any(flags[-k:])


# --------------------------------------------------------------------------- #
# region model (v2 §2.1 — blocks, pins, eviction-class selection)
# --------------------------------------------------------------------------- #
_SCI_LABEL_RE = re.compile(
    r"(?im)^\s*(World model|Goal model|Action model|Recent findings|"
    r"Open questions|Plan|Cross-level notes)\s*:")


def _is_scientist_note(message: dict[str, Any]) -> bool:
    if str(message.get("role", "")).strip() != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return False
    labels = {m.group(1).lower() for m in _SCI_LABEL_RE.finditer(content)}
    return len(labels) >= 2


def _has_reasoning(message: dict[str, Any]) -> bool:
    if str(message.get("role", "")).strip() != "assistant":
        return False
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return isinstance(reasoning, str) and bool(reasoning)


def _parse_blocks(history: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """(lo, hi, kind) spans: kind in {'user', 'episode', 'text', 'other'}.
    An assistant message owns its trailing tool results; 'episode' means the
    assistant message carries tool_calls (an act cycle)."""
    blocks: list[tuple[int, int, str]] = []
    i, n = 0, len(history)
    while i < n:
        role = str(history[i].get("role", "")).strip()
        if role == "assistant":
            j = i + 1
            while j < n and str(history[j].get("role", "")).strip() == "tool":
                j += 1
            kind = "episode" if history[i].get("tool_calls") else "text"
            blocks.append((i, j - 1, kind))
            i = j
        else:
            blocks.append((i, i, "user" if role == "user" else "other"))
            i += 1
    return blocks


def _select_evictable_block(
        history: list[dict[str, Any]],
        preserve_recent: int) -> tuple[str, int, int] | None:
    """Pick the next block to evict (class, lo, hi) or None. Class order per
    the sealed v2 intent §2.1; pins yield only at the class-5 last resort."""
    n = len(history)
    guard = n - max(0, preserve_recent)
    if guard <= 0:
        return None
    blocks = _parse_blocks(history)

    pin_idx: set[int] = set()
    sci = max((i for i, m in enumerate(history) if _is_scientist_note(m)),
              default=None)
    rea = max((i for i, m in enumerate(history) if _has_reasoning(m)),
              default=None)
    for idx in (sci, rea):
        if idx is not None:
            pin_idx.add(idx)

    def span(bi: int) -> tuple[int, int]:
        # A HEAD user block evicts as the full vanilla turn-cycle unit
        # (user + everything up to the next user), so the harness's
        # leading-non-user invariant never silently eats uncaptured messages.
        lo, hi, kind = blocks[bi]
        if kind == "user" and lo == 0:
            j = bi + 1
            while j < len(blocks) and blocks[j][2] != "user":
                hi = blocks[j][1]
                j += 1
        return lo, hi

    def pinned(lo: int, hi: int) -> bool:
        return any(lo <= p <= hi for p in pin_idx)

    episode_bis = [bi for bi, b in enumerate(blocks) if b[2] == "episode"]
    last_episode = episode_bis[-1] if episode_bis else None
    user_bis = [bi for bi, b in enumerate(blocks) if b[2] == "user"]
    last_user = user_bis[-1] if user_bis else None

    # class 1: stale action-episodes (all but the newest) + orphan tool blocks
    for bi, b in enumerate(blocks):
        if b[2] not in ("episode", "other") or bi == last_episode:
            continue
        lo, hi = b[0], b[1]
        if hi < guard and not pinned(lo, hi):
            return ("episode", lo, hi)
    # class 2: older user state-dumps (never the current-frame carrier)
    for bi in user_bis:
        if bi == last_user:
            continue
        lo, hi = span(bi)
        if hi < guard and not pinned(lo, hi):
            return ("user", lo, hi)
    # class 3: non-pinned assistant reasoning/text blocks
    for b in blocks:
        if b[2] != "text":
            continue
        lo, hi = b[0], b[1]
        if hi < guard and not pinned(lo, hi):
            return ("reasoning", lo, hi)
    # class 4: the newest episode
    if last_episode is not None:
        lo, hi = blocks[last_episode][0], blocks[last_episode][1]
        if hi < guard and not pinned(lo, hi):
            return ("episode", lo, hi)
    # class 5: last resort — pins yield rather than brick the request
    for bi, b in enumerate(blocks):
        lo, hi = span(bi) if b[2] == "user" else (b[0], b[1])
        if hi < guard:
            return ("fallback", lo, hi)
    return None


# --------------------------------------------------------------------------- #
# per-agent compaction store
# --------------------------------------------------------------------------- #
class _CompactionStore:
    """Mechanical digester state for ONE ToolAgent session (same lifecycle as
    agent._history_messages). Never touched from more than one thread."""

    __slots__ = (
        "session_dir", "ledger", "extractor", "action_counts", "action_lo",
        "action_hi", "last_action", "max_level", "level_ups", "game_overs",
        "evicted_msgs", "evicted_chars", "episodes", "turn_seq",
        "pending_msgs", "pending_chars", "retained_msgs",
        "evict_episode", "evict_user", "evict_reasoning", "evict_fallback",
        "stuck_suppressed",
    )

    def __init__(self, session_dir: Any) -> None:
        self.session_dir = session_dir
        self.ledger = ledger_core.Ledger()
        self.extractor = ledger_core.HeuristicExtractor(self.ledger)
        # action name -> [seen_count, board_changed_count]
        self.action_counts: dict[str, list[int]] = {}
        self.action_lo: int | None = None
        self.action_hi: int | None = None
        self.last_action = 0
        self.max_level = 1
        self.level_ups = 0
        self.game_overs = 0
        self.evicted_msgs = 0
        self.evicted_chars = 0
        self.episodes = 0
        self.turn_seq = 0
        # per-episode (since last event emission)
        self.pending_msgs = 0
        self.pending_chars = 0
        # RETAIN component counter (for the event line / evidence)
        self.retained_msgs = 0
        # v2: cumulative eviction-class counts + stuck-suppression tally
        self.evict_episode = 0
        self.evict_user = 0
        self.evict_reasoning = 0
        self.evict_fallback = 0
        self.stuck_suppressed = 0

    def has_content(self) -> bool:
        return self.evicted_msgs > 0

    def bump_class(self, cls: str) -> None:
        if cls == "episode":
            self.evict_episode += 1
        elif cls == "user":
            self.evict_user += 1
        elif cls == "reasoning":
            self.evict_reasoning += 1
        else:
            self.evict_fallback += 1

    def gated_fact_count(self) -> int:
        return sum(1 for f in self.ledger.facts
                   if _fact_hygiene_ok(f["statement"]))

    # -- ingestion ----------------------------------------------------------
    def ingest_message(self, message: dict[str, Any]) -> None:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        text = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=True, default=str)
        if role == "user" and text.startswith(DIGEST_MARKER):
            return  # never ingest our own digest (no feedback loop)
        self.evicted_msgs += 1
        self.pending_msgs += 1
        n_chars = len(text)
        if role == "assistant":
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            if isinstance(reasoning, str) and reasoning:
                n_chars += len(reasoning)
                text = f"{text}\n{reasoning}" if text else reasoning
            self.turn_seq += 1
            try:
                # v2: strip digest-shaped echo lines BEFORE extraction so the
                # digest can never round-trip through the model back into the
                # ledger (screen §5.2 observation 3).
                self.extractor.process_turn(
                    _strip_digest_echoes(text), step=self.turn_seq,
                    action=self.last_action)
            except Exception as exc:  # noqa: BLE001 - digester is best-effort
                log.debug("compaction extractor failed: %s", exc)
        elif role == "tool":
            self._ingest_tool_payload(text)
        self.evicted_chars += n_chars
        self.pending_chars += n_chars

    def _ingest_tool_payload(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return
        for result in _walk_action_results(payload):
            try:
                self._fold_action_result(result)
            except Exception as exc:  # noqa: BLE001
                log.debug("compaction action-result fold failed: %s", exc)

    def _fold_action_result(self, result: dict[str, Any]) -> None:
        try:
            action_num = int(result.get("action_num"))
        except (TypeError, ValueError):
            action_num = None
        if action_num is not None and action_num > 0:
            self.last_action = max(self.last_action, action_num)
            self.action_lo = action_num if self.action_lo is None else min(self.action_lo, action_num)
            self.action_hi = action_num if self.action_hi is None else max(self.action_hi, action_num)
        try:
            level = int(result.get("level"))
        except (TypeError, ValueError):
            level = None
        if level is not None and level > self.max_level:
            self.max_level = level
        board_changed = bool(result.get("board_changed"))
        names = result.get("executed_actions")
        if not isinstance(names, list) or not names:
            display = str(result.get("action_display") or "").strip()
            names = [display] if display else []
        for raw in names:
            name = str(raw).strip().split("(", 1)[0].strip() or "UNKNOWN"
            slot = self.action_counts.setdefault(name, [0, 0])
            slot[0] += 1
            if board_changed:
                slot[1] += 1
        if result.get("level_completed"):
            self.level_ups += 1
            try:
                self.extractor.observe_level_up(action_num or self.last_action)
            except Exception as exc:  # noqa: BLE001
                log.debug("compaction level-up observe failed: %s", exc)
        if result.get("game_over"):
            self.game_overs += 1
            try:
                self.extractor.observe_game_over(action_num or self.last_action)
            except Exception as exc:  # noqa: BLE001
                log.debug("compaction game-over observe failed: %s", exc)

    # -- rendering ----------------------------------------------------------
    def render_digest(self) -> str:
        """v2 digest: refuted list (NEVER elided into counts) + hygiene-gated
        FACTs + small meta tail. NO ACTIVE/CONFIRMED lines. Empty gate =>
        return "" and the caller injects NOTHING (budget relief is the win)."""
        if not self.has_content():
            return ""
        refuted = [h for h in self.ledger.hypotheses
                   if h["status"] == "refuted"]
        gated_facts = [f for f in self.ledger.facts
                       if _fact_hygiene_ok(f["statement"])]
        if not refuted and not gated_facts:
            return ""
        cap_chars = max(300, (_reserve_tokens() - 40) * ledger_core.CHARS_PER_TOKEN)
        lines = [_DIGEST_HEADER]
        budget = cap_chars - len(_DIGEST_HEADER) - 1

        def emit(line: str) -> bool:
            nonlocal budget
            if len(line) + 1 > budget:
                return False
            lines.append(line)
            budget -= len(line) + 1
            return True

        # REFUTED first (the M3 payload): one line per record, never a count.
        # Newest-first so a budget overflow drops the OLDEST lines silently —
        # residual truncation, never "+N more" collapsing (v2 intent §2.2:
        # stop at the first overflow so the drop is exactly the oldest tail).
        for hyp in reversed(refuted):
            ev = f" | ev: {hyp['evidence'][0][:60]}" if hyp["evidence"] else ""
            if not emit(f"REFUTED {hyp['id']} [{hyp['family']}]: "
                        f"{hyp['statement'][:90]}{ev}"):
                break
        # FACTs second, hygiene-gated, newest-first (same §2.2 render rule:
        # overflow drops the OLDEST fact lines silently).
        for fact in reversed(gated_facts):
            if not emit(f"FACT {fact['id']}: {fact['statement'][:110]}"):
                break
        # Meta tail third.
        span = "?"
        if self.action_lo is not None and self.action_hi is not None:
            span = (f"{self.action_lo}" if self.action_lo == self.action_hi
                    else f"{self.action_lo}-{self.action_hi}")
        emit(f"EVICTED: {self.evicted_msgs} messages (~{self.evicted_chars} "
             f"chars) spanning actions {span}")
        if self.action_counts:
            top = sorted(self.action_counts.items(),
                         key=lambda kv: kv[1][0], reverse=True)[:8]
            rendered = ", ".join(
                f"{name} x{count} (changed board {changed}/{count})"
                for name, (count, changed) in top)
            emit(f"ACTION-EFFECTS (evicted turns): {rendered}")
        emit(f"PROGRESS: max_level_seen={self.max_level} "
             f"level_ups={self.level_ups} game_overs={self.game_overs}")
        return "\n".join(lines)


def _walk_action_results(node: Any) -> list[dict[str, Any]]:
    """Collect every compact action-result dict (has 'executed' + 'action_num')
    anywhere in a parsed tool payload — matches the shapes _run_python_tool
    emits (result.action_result, result.last_action_result, action_calls...)."""
    out: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "executed" in node and "action_num" in node:
            out.append(node)
        for value in node.values():
            out.extend(_walk_action_results(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_walk_action_results(item))
    return out


def _get_store(agent: Any) -> _CompactionStore:
    session_dir = getattr(agent, "_session_runtime_dir", None)
    store = getattr(agent, "_compaction_store", None)
    if store is None or store.session_dir != session_dir:
        store = _CompactionStore(session_dir)
        agent._compaction_store = store
    return store


def _digest_message(digest: str) -> dict[str, Any]:
    return {"role": "user", "content": digest}


def _is_digest_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (str(message.get("role", "")).strip() == "user"
            and isinstance(content, str)
            and content.startswith(DIGEST_MARKER))


# --------------------------------------------------------------------------- #
# RETAIN: reasoning -> reasoning_content mirroring (outbound only; v2 sub-arm,
# installed ONLY when COMPACTION_RETAIN=1 — OFF by default)
# --------------------------------------------------------------------------- #
def _mirror_reasoning(messages: list[dict[str, Any]]) -> int:
    """Mirror assistant `reasoning` into `reasoning_content` in place so the
    server-side chat template (preserve_thinking) can actually see it.
    Returns the number of messages carrying retained reasoning."""
    mirrored = 0
    for message in messages:
        if str(message.get("role", "")).strip() != "assistant":
            continue
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            if not message.get("reasoning_content"):
                message["reasoning_content"] = reasoning
            mirrored += 1
    return mirrored


# --------------------------------------------------------------------------- #
# event emission
# --------------------------------------------------------------------------- #
def _emit_event(agent: Any, store: _CompactionStore, digest_tokens: int,
                reserve_applied: bool) -> None:
    """ONE countable event per (non-stuck) trim that has pending evictions.
    Greppable stdout line + best-effort per-game jsonl sidecar."""
    state_path = getattr(agent, "_compaction_state_path", None)
    game = _store_key(state_path)
    gated = store.gated_fact_count()
    retain = 1 if _retain_enabled() else 0
    print(
        f"COMPACTION v={_EVENT_V} kind=evict_compact game={game} "
        f"evicted_msgs={store.pending_msgs} evicted_chars={store.pending_chars} "
        f"total_evicted_msgs={store.evicted_msgs} episodes={store.episodes} "
        f"digest_tokens={digest_tokens} facts={len(store.ledger.facts)} "
        f"gated_facts={gated} refuted={store.ledger.refuted_count()} "
        f"ev_episode={store.evict_episode} ev_user={store.evict_user} "
        f"ev_reasoning={store.evict_reasoning} ev_fallback={store.evict_fallback} "
        f"stuck_suppressed={store.stuck_suppressed} "
        f"reserve_applied={1 if reserve_applied else 0} retain={retain} "
        f"retained_reasoning_msgs={store.retained_msgs}",
        flush=True,
    )
    if state_path is not None:
        try:
            rec = {
                "kind": "evict_compact",
                "v": _EVENT_V,
                "game": game,
                "evicted_msgs": store.pending_msgs,
                "evicted_chars": store.pending_chars,
                "total_evicted_msgs": store.evicted_msgs,
                "episodes": store.episodes,
                "digest_tokens": digest_tokens,
                "facts": len(store.ledger.facts),
                "gated_facts": gated,
                "refuted": store.ledger.refuted_count(),
                "ev_episode": store.evict_episode,
                "ev_user": store.evict_user,
                "ev_reasoning": store.evict_reasoning,
                "ev_fallback": store.evict_fallback,
                "stuck_suppressed": store.stuck_suppressed,
                "reserve_applied": 1 if reserve_applied else 0,
                "retain": retain,
                "retained_reasoning_msgs": store.retained_msgs,
            }
            path = Path(state_path).parent / f"{game}_compaction_events.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
            log.debug("compaction sidecar write failed: %s", exc)
    store.pending_msgs = 0
    store.pending_chars = 0


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
def _apply_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from inference.agent.tool_agent import ToolAgent

    retain = _retain_enabled()

    # (1) analyze: bind state_path for event labels/sidecars -----------------
    _orig_analyze = ToolAgent.analyze

    def compaction_analyze(self, state_path, action_num, valid_actions=None,
                           step_env=None, **kwargs):
        try:
            self._compaction_state_path = Path(state_path)
        except Exception as exc:  # noqa: BLE001 - never break the turn
            log.debug("compaction pre-analyze hook failed: %s", exc)
        return _orig_analyze(self, state_path, action_num,
                             valid_actions=valid_actions, step_env=step_env,
                             **kwargs)

    ToolAgent.analyze = compaction_analyze

    # (2) capture-only wrap of the vanilla drop point (used by the fallback
    # path and _force_reduce_messages; the v2 region-aware trim below does its
    # own selection + capture and never calls this) -------------------------
    _orig_drop = ToolAgent._drop_oldest_history_block

    def compaction_drop_oldest(self, history, *, preserve_recent):
        before = list(history)
        dropped = _orig_drop(self, history, preserve_recent=preserve_recent)
        if dropped:
            try:
                store = _get_store(self)
                kept_ids = {id(m) for m in history}
                for message in before:
                    if id(message) not in kept_ids:
                        store.ingest_message(message)
                store.bump_class("fallback")
            except Exception as exc:  # noqa: BLE001 - never break the trim
                log.debug("compaction drop-capture failed: %s", exc)
        return dropped

    ToolAgent._drop_oldest_history_block = compaction_drop_oldest

    # (3) capture point B: the 30-assistant-turn persistence cap — DEFERRED
    # outright while stuck (v2 §2.4: the only fully discretionary cut) -------
    _orig_keep = ToolAgent._keep_recent_history_turns

    def compaction_keep_recent(self, messages, *, max_turns):
        try:
            if _is_stuck(messages, _stuck_k()):
                kept = _orig_keep(self, messages, max_turns=max_turns)
                if len(kept) < len(messages):
                    store = _get_store(self)
                    store.stuck_suppressed += 1
                    return list(messages)  # defer the cap: no cut while stuck
                return kept
        except Exception as exc:  # noqa: BLE001 - never break persistence
            log.debug("compaction stuck-defer failed: %s", exc)
        kept = _orig_keep(self, messages, max_turns=max_turns)
        try:
            store = _get_store(self)
            kept_ids = {id(m) for m in kept}
            for message in messages:
                if id(message) not in kept_ids:
                    store.ingest_message(message)
        except Exception as exc:  # noqa: BLE001 - never break persistence
            log.debug("compaction keep-capture failed: %s", exc)
        return kept

    ToolAgent._keep_recent_history_turns = compaction_keep_recent

    # (4) the v2 region-aware trim (replaces v1's capture-only wrap) ---------
    _orig_trim = ToolAgent._trim_messages_for_context

    def compaction_trim(self, messages, *, tools=None, preserve_recent=1,
                        extra_safety_tokens=0):
        try:
            if not messages:
                return _orig_trim(self, messages, tools=tools,
                                  preserve_recent=preserve_recent,
                                  extra_safety_tokens=extra_safety_tokens)
            store = _get_store(self)
            system_message = messages[0]
            # Strip stale digests: re-rendered fresh below; never compounds.
            history = [m for m in messages[1:] if not _is_digest_message(m)]
            preserve = max(0, preserve_recent)
            stuck = _is_stuck(history, _stuck_k())
            reserve = _reserve_tokens()
            # v2.1: the digest channel exists ONLY under COMPACTION_DIGEST=1
            # (default 0 — pure eviction; nothing rendered, no reserve ever
            # subtracted, trim budget exactly vanilla). Under =1 this is the
            # v2 rule verbatim: reserve subtracted ONLY when a non-empty
            # digest would actually be injected (and never while stuck).
            reserve_applied = (_digest_enabled() and (not stuck)
                               and bool(store.render_digest()))
            budget_extra = max(0, extra_safety_tokens) + (
                reserve if reserve_applied else 0)
            budget_tokens = max(1, self._context_budget_tokens - budget_extra)
            evicted_this_call = False
            while history and self._estimate_request_input_tokens(
                    [system_message, *history], tools=tools) > budget_tokens:
                selected = _select_evictable_block(history, preserve)
                if selected is None:
                    break
                cls, lo, hi = selected
                for message in history[lo:hi + 1]:
                    store.ingest_message(message)
                del history[lo:hi + 1]
                store.bump_class(cls)
                evicted_this_call = True
            # harness head invariant: history must start with a user message
            # (vanilla _drop_until_first_user_message — but captured here).
            while history and str(history[0].get("role", "")).strip() != "user":
                store.ingest_message(history.pop(0))
                store.bump_class("fallback")
                evicted_this_call = True
            if evicted_this_call:
                store.episodes += 1
            digest = store.render_digest() if reserve_applied else ""
            digest_tokens = ledger_core.estimate_tokens(digest) if digest else 0
            if store.pending_msgs > 0 and not stuck:
                _emit_event(self, store, digest_tokens, reserve_applied)
            elif stuck and (evicted_this_call or store.pending_msgs > 0):
                # budget-forced cut OR suppressed event emission while stuck:
                # no event (v2 §2.4); stuck_suppressed counts every suppressed
                # cut/emission opportunity (§2.5) and the pending counters
                # flush into the next non-stuck event.
                store.stuck_suppressed += 1
            if digest:
                return [system_message, _digest_message(digest), *history]
            return [system_message, *history]
        except Exception as exc:  # noqa: BLE001 - fall back to stock behavior
            log.debug("compaction trim wrapper failed: %s", exc)
            return _orig_trim(self, messages, tools=tools,
                              preserve_recent=preserve_recent,
                              extra_safety_tokens=extra_safety_tokens)

    ToolAgent._trim_messages_for_context = compaction_trim

    # (5) RETAIN sub-arm: mirror reasoning -> reasoning_content outbound.
    # v2: OFF by default; installed ONLY when COMPACTION_RETAIN=1.
    if retain:
        _orig_chat = ToolAgent._chat_completion

        def compaction_chat_completion(self, messages, **kwargs):
            try:
                mirrored = _mirror_reasoning(messages)
                store = _get_store(self)
                store.retained_msgs = mirrored
            except Exception as exc:  # noqa: BLE001 - never break the request
                log.debug("compaction retain hook failed: %s", exc)
            return _orig_chat(self, messages, **kwargs)

        ToolAgent._chat_completion = compaction_chat_completion

    _APPLIED = True


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install compaction v2.1 (pure region-aware eviction, digest-OFF).
    Returns True on success (or already applied), False on flag-off / kill
    switch / any failure — in which case NOTHING is changed (vanilla
    fallback: stock duck harness).

    Flag gate: requires COMPACTION=1 (the A22 arm flag).
    Kill switch: COMPACTION_DISABLE=1 -> no-op, returns False.
    Sub-arm: COMPACTION_DIGEST=1 -> restore the v2 digest channel (v2.1
    default OFF: nothing rendered/injected, no reserve subtracted).
    Sub-arm: COMPACTION_RETAIN=1 -> ALSO mirror reasoning (v2 default OFF).
    """
    if _kill_switch():
        log.info("compaction %s: COMPACTION_DISABLE=1 -> no-op", VERSION)
        return False
    if not _flag_on():
        log.info("compaction %s: COMPACTION!=1 -> no-op (flag-gated arm)", VERSION)
        return False
    try:
        _apply_patches()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-compaction-{VERSION}"
        # Runtime banner on stdout: the build log is the only proof of which
        # dataset version actually ran (feedback_kaggle_dataset_code_sync).
        digest_state = (
            f"ON (COMPACTION_DIGEST=1 restores v2: hygiene-gated, "
            f"empty-allowed, refuted never elided; reserve={_reserve_tokens()} "
            f"tok, applied only when non-empty)"
            if _digest_enabled() else
            "OFF (v2.1 default: pure eviction, no injection, no reserve; "
            "reserve_applied=0 digest_tokens=0 on every event)")
        print(
            f"compaction {VERSION}: ACTIVE "
            f"(region-aware eviction: pin system+scientist-note+latest-reasoning, "
            f"stale action-episodes evicted first; "
            f"digest={digest_state}; "
            f"stuck-suppress K={_stuck_k()}; "
            f"retained-reasoning mirroring={'ON' if _retain_enabled() else 'OFF (v2 default)'} "
            f"[reasoning->reasoning_content, feeds preserve_thinking]; "
            f"ledger_core digester aboard; zero LLM calls in the eviction path)",
            flush=True,
        )
        log.info("compaction %s installed", VERSION)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[compaction] apply failed -> stock duck harness (vanilla)")
        return False
