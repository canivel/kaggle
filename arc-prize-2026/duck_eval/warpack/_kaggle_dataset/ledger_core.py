"""Hypothesis Ledger core — pure logic, no harness imports (R2, intervention_plan.md).

Two record kinds, persisted per game OUTSIDE the 14-message window:
  HYPOTHESIS(id, statement, family, status: untested/executing/refuted/confirmed,
             evidence[], born_step, born_action, refuted_action)
  FACT(id, statement, born_step, born_action)   # action-effect observations

Plus:
  * GOAL:/RESULT: (and FACT:) prompt-field regex extraction (new prompt contract),
  * a legacy HeuristicExtractor for replaying recorded transcripts that predate
    the contract (stage-1 replay test),
  * a <=600-token digest renderer injected every turn (survives GAME_OVER
    restarts and level transitions because the store lives outside messages),
  * goal-family escalation: after N=3 same-family fully-executed refutations,
    arm a ONE-SHOT forced enumeration of 4 mechanically distinct goal families.

No game-id logic anywhere (graft rule).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DIGEST_TOKEN_CAP = 600
# duck's analyzer averages ~3 chars/token on this prompt mix (phase1 measured);
# use chars/3 as a conservative over-estimate so 600 "tokens" is a hard cap.
CHARS_PER_TOKEN = 3
DIGEST_CHAR_CAP = DIGEST_TOKEN_CAP * CHARS_PER_TOKEN

ESCALATION_N = 3

# The 4 mechanically distinct goal families (intervention_plan.md R2).
FAMILIES = ("ordering", "transfer", "merge", "alignment")

_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ordering", re.compile(
        r"(?i)\b(order(?:ing)?|sequence|revers\w*|arrange\w*|arrangement|permutat\w*"
        r"|left.to.right|right.to.left|fill (?:the |them |all )?(?:\w+ )?slots?"
        r"|execution|program|subroutine|swap\w*|in that order|correct order)\b")),
    ("transfer", re.compile(
        r"(?i)\b(move (?:\w+ ){0,3}(?:in)?to|transfer|reach(?:es|ing)?|bring|carry"
        r"|deliver|put (?:\w+ ){0,3}in(?:to|side)?|drag (?:\w+ ){0,3}(?:in)?to"
        r"|get (?:\w+ ){0,3}(?:in)?to|collect)\b")),
    ("merge", re.compile(
        r"(?i)\b(merge|combine|collide|absorb\w*|physics|gravity|stack(?:ing)?"
        r"|join|fuse|grow)\b")),
    ("alignment", re.compile(
        r"(?i)\b(align\w*|match(?:ing|es)? (?:the |a |with )?(?:\w+ ){0,3}"
        r"(?:bar|pattern|position|row|column|color|top)|mirror|symmetr\w*|overlap"
        r"|same (?:position|column|row)|diagonal line|form a)\b")),
]

ESCALATION_PROMPT_TEMPLATE = (
    "ESCALATION -- GOAL-FAMILY CHECK (one-shot). {count} goal hypotheses from the "
    "SAME family ('{family}') have now been fully executed and refuted (see the "
    "REFUTED ledger above). Stop proposing variants of that family. Before your "
    "next action: list 4 mechanically distinct goal families for this game -- "
    "(1) execution-order/program: the order operations RUN in, including "
    "subroutine/call-style jumps between structures, may differ from visual "
    "reading order; (2) transfer-between-structures: items must be moved from "
    "one structure into another; (3) merge/physics: objects combine, collide or "
    "absorb each other; (4) spatial-alignment: objects must match a pattern, "
    "bar or position. Then pick the family your refuted set LEAST resembles and "
    "state a new GOAL: line from that family."
)


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def classify_family(text: str) -> str:
    best, best_hits = "other", 0
    for family, pattern in _FAMILY_PATTERNS:
        hits = len(pattern.findall(text or ""))
        if hits > best_hits:
            best, best_hits = family, hits
    return best


def _norm_key(statement: str) -> str:
    return re.sub(r"\W+", " ", (statement or "").lower()).strip()[:160]


_WORD = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    "that this with then them they there maybe goal goals what when have been "
    "should would could might need needs think about level game square squares "
    "click clicking clicked moving".split())


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


# --------------------------------------------------------------------------
# Prompt-field extraction (new prompt contract): GOAL: / RESULT: / FACT:
# --------------------------------------------------------------------------
GOAL_RE = re.compile(
    r"(?im)^\s*(?:[-*>#`\s]*)GOAL:\s*(?:\[(?P<fam1>[a-z/-]+)\]\s*|family\s*=\s*(?P<fam2>[a-z/-]+)\s+)?(?P<statement>.+?)\s*$")
RESULT_RE = re.compile(
    r"(?im)^\s*(?:[-*>#`\s]*)RESULT:\s*(?P<verdict>confirmed|refuted|unclear|ok)\b[\s:.-]*(?P<evidence>.*?)\s*$")
FACT_RE = re.compile(r"(?im)^\s*(?:[-*>#`\s]*)FACT:\s*(?P<statement>.+?)\s*$")


def extract_goal_result(text: str) -> list[dict[str, str]]:
    """Extract GOAL:/RESULT:/FACT: prompt fields from an assistant message.

    Returns records in document order:
      {'kind': 'goal',   'statement': ..., 'family': ...}
      {'kind': 'result', 'verdict': confirmed|refuted|unclear, 'evidence': ...}
      {'kind': 'fact',   'statement': ...}
    """
    if not text:
        return []
    hits: list[tuple[int, dict[str, str]]] = []
    for m in GOAL_RE.finditer(text):
        family = (m.group("fam1") or m.group("fam2") or "").strip().lower()
        if family not in FAMILIES:
            family = classify_family(m.group("statement"))
        hits.append((m.start(), {
            "kind": "goal", "statement": m.group("statement").strip(),
            "family": family}))
    for m in RESULT_RE.finditer(text):
        verdict = m.group("verdict").lower()
        if verdict == "ok":
            verdict = "confirmed"
        hits.append((m.start(), {
            "kind": "result", "verdict": verdict,
            "evidence": m.group("evidence").strip()}))
    for m in FACT_RE.finditer(text):
        hits.append((m.start(), {"kind": "fact",
                                 "statement": m.group("statement").strip()}))
    return [record for _, record in sorted(hits, key=lambda item: item[0])]


# --------------------------------------------------------------------------
# The Ledger
# --------------------------------------------------------------------------
class Ledger:
    """Persistent per-game hypothesis/fact store (lives outside the message
    window; save()/load() give it disk persistence across process events)."""

    def __init__(self) -> None:
        self.hypotheses: list[dict[str, Any]] = []
        self.facts: list[dict[str, Any]] = []
        self._hyp_keys: dict[str, int] = {}
        self._fact_keys: set[str] = set()
        # escalation state: one-shot per trigger
        self.escalation_armed: bool = False
        self.escalation_family: str = ""
        self.escalation_trigger_action: int | None = None
        self.escalations_fired: int = 0
        self._family_fired_at: dict[str, int] = {}  # refuted-count consumed per family
        self.game_overs: int = 0
        self.levels_seen: int = 0

    # -- record management -------------------------------------------------
    def add_hypothesis(self, statement: str, family: str | None = None, *,
                       step: int = 0, action: int = 0,
                       status: str = "executing") -> dict[str, Any] | None:
        statement = (statement or "").strip()
        if len(statement) < 8:
            return None
        key = _norm_key(statement)
        if key in self._hyp_keys:
            return None  # duplicate restatement, not a new hypothesis
        hyp = {
            "id": f"H{len(self.hypotheses) + 1}",
            "kind": "HYPOTHESIS",
            "statement": statement[:220],
            "family": family if family in FAMILIES or family == "other"
            else classify_family(statement),
            "status": status,
            "evidence": [],
            "born_step": int(step),
            "born_action": int(action),
            "refuted_action": None,
        }
        self._hyp_keys[key] = len(self.hypotheses)
        self.hypotheses.append(hyp)
        return hyp

    def add_fact(self, statement: str, *, step: int = 0,
                 action: int = 0) -> dict[str, Any] | None:
        statement = (statement or "").strip()
        if len(statement) < 6:
            return None
        key = _norm_key(statement)
        if key in self._fact_keys:
            return None
        fact = {
            "id": f"F{len(self.facts) + 1}",
            "kind": "FACT",
            "statement": statement[:180],
            "born_step": int(step),
            "born_action": int(action),
        }
        self._fact_keys.add(key)
        self.facts.append(fact)
        return fact

    def active_hypotheses(self) -> list[dict[str, Any]]:
        return [h for h in self.hypotheses
                if h["status"] in ("untested", "executing")]

    def refute(self, hyp: dict[str, Any], evidence: str, *,
               action: int = 0) -> None:
        if hyp["status"] != "refuted":
            hyp["status"] = "refuted"
            hyp["refuted_action"] = int(action)
        self.append_evidence(hyp, evidence)
        self._maybe_arm_escalation(hyp["family"], action=action)

    def append_evidence(self, hyp: dict[str, Any], evidence: str) -> None:
        evidence = (evidence or "").strip()[:160]
        if evidence and evidence not in hyp["evidence"] and len(hyp["evidence"]) < 4:
            hyp["evidence"].append(evidence)

    def refuted_count(self, family: str | None = None) -> int:
        return sum(1 for h in self.hypotheses if h["status"] == "refuted"
                   and (family is None or h["family"] == family))

    # -- escalation ----------------------------------------------------------
    def _maybe_arm_escalation(self, family: str, *, action: int = 0) -> None:
        if self.escalation_armed or family == "other":
            return
        count = self.refuted_count(family)
        if count - self._family_fired_at.get(family, 0) >= ESCALATION_N:
            self.escalation_armed = True
            self.escalation_family = family
            if self.escalation_trigger_action is None:
                self.escalation_trigger_action = int(action)

    def consume_escalation(self) -> str | None:
        """One-shot: return the enumeration prompt once per trigger."""
        if not self.escalation_armed:
            return None
        family = self.escalation_family
        text = ESCALATION_PROMPT_TEMPLATE.format(
            count=self.refuted_count(family), family=family)
        self._family_fired_at[family] = self.refuted_count(family)
        self.escalation_armed = False
        self.escalation_family = ""
        self.escalations_fired += 1
        return text

    # -- ingestion of extracted prompt fields --------------------------------
    def ingest(self, records: list[dict[str, str]], *, step: int = 0,
               action: int = 0) -> None:
        for record in records:
            kind = record.get("kind")
            if kind == "goal":
                self.add_hypothesis(record.get("statement", ""),
                                    record.get("family"),
                                    step=step, action=action)
            elif kind == "fact":
                self.add_fact(record.get("statement", ""), step=step,
                              action=action)
            elif kind == "result":
                verdict = record.get("verdict")
                evidence = record.get("evidence", "")
                target = self._match_hypothesis(evidence)
                if target is None:
                    continue
                if verdict == "refuted":
                    self.refute(target, evidence, action=action)
                elif verdict == "confirmed":
                    target["status"] = "confirmed"
                    self.append_evidence(target, evidence)
                else:
                    self.append_evidence(target, evidence)

    def _match_hypothesis(self, evidence: str) -> dict[str, Any] | None:
        """Newest active hypothesis, preferring content-word overlap."""
        active = self.active_hypotheses()
        if not active:
            return None
        ev_words = _content_words(evidence)
        best, best_overlap = None, 0
        for hyp in active:
            overlap = len(ev_words & _content_words(hyp["statement"]))
            if overlap >= best_overlap:  # >= so newest wins ties
                best, best_overlap = hyp, overlap
        return best

    # -- digest ---------------------------------------------------------------
    def render_digest(self) -> str:
        """Compact ledger digest, <= DIGEST_TOKEN_CAP tokens, injected every
        turn. Refuted list newest-first; older entries aggregate per family."""
        refuted = [h for h in self.hypotheses if h["status"] == "refuted"]
        active = self.active_hypotheses()
        confirmed = [h for h in self.hypotheses if h["status"] == "confirmed"]

        header = ("=== HYPOTHESIS LEDGER (persistent; survives GAME_OVER "
                  "restarts and level changes; do NOT re-test refuted "
                  "entries) ===")
        lines: list[str] = [header]
        budget = DIGEST_CHAR_CAP - len(header) - 1

        def emit(line: str) -> bool:
            nonlocal budget
            if len(line) + 1 > budget:
                return False
            lines.append(line)
            budget -= len(line) + 1
            return True

        # facts: earliest-learned action-effects are the most durable (e.g.
        # "SPACE only decrements the timer" from level 1) — keep the oldest 6
        # AND the newest 6 so early facts survive noisy late accumulation.
        selected_facts = self.facts[:6]
        for fact in self.facts[-6:]:
            if fact not in selected_facts:
                selected_facts.append(fact)
        for fact in selected_facts:
            if not emit(f"FACT {fact['id']}: {fact['statement'][:110]}"):
                break
        for hyp in list(reversed(active))[:2]:
            emit(f"ACTIVE {hyp['id']} [{hyp['family']}]: "
                 f"{hyp['statement'][:100]}")
        for hyp in list(reversed(confirmed))[:2]:
            emit(f"CONFIRMED {hyp['id']} [{hyp['family']}]: "
                 f"{hyp['statement'][:90]}")

        shown = 0
        omitted: dict[str, int] = {}
        for hyp in reversed(refuted):
            ev = f" | ev: {hyp['evidence'][0][:70]}" if hyp["evidence"] else ""
            line = (f"REFUTED {hyp['id']} [{hyp['family']}]: "
                    f"{hyp['statement'][:80]}{ev}")
            if shown < 12 and emit(line):
                shown += 1
            else:
                omitted[hyp["family"]] = omitted.get(hyp["family"], 0) + 1
        if omitted:
            summary = "; ".join(f"{n} more {fam}" for fam, n in omitted.items())
            tail = f"(+{summary} refuted variants -- all failed; do not retry)"
            if not emit(tail):
                lines.append(tail[:max(0, budget)])
        digest = "\n".join(lines)
        if estimate_tokens(digest) > DIGEST_TOKEN_CAP:  # belt and braces
            digest = digest[:DIGEST_CHAR_CAP]
        return digest

    # -- persistence -----------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "hypotheses": self.hypotheses,
            "facts": self.facts,
            "escalation_armed": self.escalation_armed,
            "escalation_family": self.escalation_family,
            "escalation_trigger_action": self.escalation_trigger_action,
            "escalations_fired": self.escalations_fired,
            "family_fired_at": self._family_fired_at,
            "game_overs": self.game_overs,
            "levels_seen": self.levels_seen,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Ledger":
        led = cls()
        led.hypotheses = list(payload.get("hypotheses") or [])
        led.facts = list(payload.get("facts") or [])
        led._hyp_keys = {_norm_key(h["statement"]): i
                         for i, h in enumerate(led.hypotheses)}
        led._fact_keys = {_norm_key(f["statement"]) for f in led.facts}
        led.escalation_armed = bool(payload.get("escalation_armed"))
        led.escalation_family = str(payload.get("escalation_family") or "")
        led.escalation_trigger_action = payload.get("escalation_trigger_action")
        led.escalations_fired = int(payload.get("escalations_fired") or 0)
        led._family_fired_at = dict(payload.get("family_fired_at") or {})
        led.game_overs = int(payload.get("game_overs") or 0)
        led.levels_seen = int(payload.get("levels_seen") or 0)
        return led

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        path = Path(path)
        if path.is_file():
            try:
                return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - corrupt file -> fresh ledger
                pass
        return cls()


# --------------------------------------------------------------------------
# Legacy heuristic extractor (stage-1 replay over pre-contract transcripts)
# --------------------------------------------------------------------------
_GOAL_LINE = re.compile(
    r"(?i)(the goal (?:is|might|may|could|seems|in level \d+ is)"
    r"|maybe the goal|perhaps the goal|i think the goal|goal hypothesis"
    r"|maybe the (?:correct |right )?(?:order|mapping|arrangement|pattern|issue) is"
    r"|maybe (?:the top|i need to|the green|the red|the mapping|it should)"
    r"|what if the|let me try (?:placing|a simpler|the opposite|reversing)"
    r"|the correct (?:order|sequence|pattern|arrangement) (?:is|should|might))")

_REFUTE_LINE = re.compile(
    r"(?i)(can'?t reach|cannot reach|impossible|off the board|off.?board"
    r"|didn'?t (?:work|complete)|isn'?t completing|not complet\w+"
    r"|nothing (?:changed|worked|happened)|no way to|would be negative"
    r"|goal is (?:probably )?not|suggests? the goal is not"
    r"|doesn'?t seem right|=\s*-\d|col\s*=?\s*-\d|that'?s what i have)")

_ARITHMETIC = re.compile(r"-\d|\d+\s*[-+]\s*\d+\s*=|off the board|off.?board")

_FACT_LINE = re.compile(
    r"(?i)\b(SPACE|ACTION\s?\d|MOUSE|clicking (?:on )?(?:a |the )?\w+"
    r"|each click|clicks?)\b.{0,70}?"
    r"\b(just|only|decrement\w*|no effect|nothing|doesn'?t change"
    r"|didn'?t change|without changing|moves? (?:it|them|the \w+)? ?"
    r"(?:up|down|left|right)|creates?|converts?|decreas\w+)\b")


class HeuristicExtractor:
    """Turn-by-turn extractor for recorded (pre-contract) duck transcripts.

    Feeds a Ledger exactly as the live GOAL:/RESULT: tap would have:
      * goal-like sentences -> HYPOTHESIS records (deduped, family-classified),
      * self-disproof / negative-evidence sentences -> refute the best-matching
        active hypothesis (content-word overlap, newest wins ties),
      * supersession: when a new hypothesis arrives and older active ones were
        already executed (env actions advanced past their birth) without a
        level completion, the older ones are marked refuted ("fully executed,
        superseded"),
      * action-effect sentences -> FACT records.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.last_level_up_action = 0

    def observe_level_up(self, action: int) -> None:
        self.last_level_up_action = int(action)
        self.ledger.levels_seen += 1
        # A level up confirms whatever was being executed; archive actives.
        for hyp in self.ledger.active_hypotheses():
            hyp["status"] = "confirmed"
            self.ledger.append_evidence(hyp, f"level completed at action {action}")

    def observe_game_over(self, action: int) -> None:
        self.ledger.game_overs += 1
        self.ledger.add_fact(f"GAME_OVER occurred at action {action}; "
                             "ledger persists across the restart",
                             action=action)

    def process_turn(self, text: str, *, step: int, action: int) -> None:
        led = self.ledger
        for raw_line in (text or "").split("\n"):
            line = raw_line.strip()
            if len(line) < 25:
                continue
            is_refute = bool(_REFUTE_LINE.search(line))
            if is_refute:
                target = led._match_hypothesis(line)
                if target is not None and target["born_action"] < action:
                    led.refute(target, line[:160], action=action)
                elif target is not None:
                    led.append_evidence(target, line[:160])
                else:
                    # arithmetic evidence lines that arrive just after a
                    # refutation: attach to the newest refuted hypothesis
                    refuted = [h for h in led.hypotheses
                               if h["status"] == "refuted"]
                    if refuted and _ARITHMETIC.search(line):
                        led.append_evidence(refuted[-1], line[:160])
            if _GOAL_LINE.search(line) and not is_refute:
                newly = led.add_hypothesis(line[:220], None, step=step,
                                           action=action)
                if newly is not None:
                    self._supersede(newly, action=action)
            m = _FACT_LINE.search(line)
            if m and not is_refute and len(line) < 200:
                led.add_fact(line[:180], step=step, action=action)

    def _supersede(self, new_hyp: dict[str, Any], *, action: int) -> None:
        """Older executed actives of the SAME family are refuted when a new
        same-family variant supersedes them without a level completion."""
        for hyp in self.ledger.active_hypotheses():
            if hyp is new_hyp:
                continue
            if (hyp["family"] == new_hyp["family"]
                    and hyp["born_action"] < action
                    and hyp["born_action"] >= self.last_level_up_action):
                self.ledger.refute(
                    hyp,
                    f"fully executed, superseded at action {action}; "
                    "level did not complete",
                    action=action)
