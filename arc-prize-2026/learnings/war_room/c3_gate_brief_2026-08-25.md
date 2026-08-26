# TASK — C3 GATE: discard-vs-truncation verification (ZERO GPU, forensic)

You are a forensic agent on the ARC-AGI-3 campaign. Repo root: F:\kaggle\arc-prize-2026.
Work ONLY from files on disk. Do NOT push anything to Kaggle. Do NOT spend GPU.

## BACKGROUND
Arm "C3" was provisionally sealed on 2026-08-22 with this mechanism claim:

  "C3 = LOCAL_ANALYZER_YIELD_SECONDS raised to ~150s (one constant,
   subtraction-of-waste class, xhigh preserved) -- it converts dead
   invocations into acting turns at ZERO extra wallclock and without
   touching thinking quality... it cuts NO thinking, it only stops
   DISCARDING COMPLETED GENERATIONS."

The coordinator ordered a hard gate BEFORE any build:
  "For invocations ending `turn_time_budget`, determine whether a completed
   generation existed and was DROPPED, or whether generation was TRUNCATED
   mid-stream. Cite the code path in the SHIPPED harness, not `inference`."

The conditional is explicit: "If discard is confirmed, seal C3 and make it slot 1."
If discard is NOT confirmed, C3's premise fails and it must NOT take a slot.

## THE SHIPPED HARNESS (this exact tree, the 08-07 anim bundle the field floor runs)
runs/harness_diff_0813/ds/jakobbrggen_taaf-kaggle-source-anim-20260807-anim/src/ARC3-Inference/

## MY PRELIMINARY READ -- YOUR JOB IS TO REFUTE IT, NOT CONFIRM IT
I read `inference/agent/tool_agent.py` and concluded: **NEITHER discard NOR
truncation. The yield is WORK-PRESERVING.** Evidence I used:

- L154  `_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)`
- L1085 `self._yield_seconds = None if <=0 else float(...)`
- `framework/kaggle.py:114` defaults the env to **"60"** on the Kaggle rail (so the yield IS live)
- `framework/kaggle.py` also defaults `LOCAL_ANALYZER_TOOL_STEPS` to **"0"** -> `_tool_steps=None`
  (L1083), i.e. UNLIMITED tool steps -- so the 60s yield is the only thing bounding an invocation
- L2153-2163 `control_yield_reason()` returns "turn_time_budget" purely on elapsed wallclock
- L2168-2170: the check runs at the TOP of the `while` loop, BEFORE `turn_count += 1` and
  BEFORE any request is issued -> no generation is in flight -> nothing can be truncated
- L2288, L2353: the other two check sites are AFTER a response is fully received / after
  `_dispatch_tool` returns -> again post-completion, not mid-stream
- L2139 `preserve_history = True`; it is set False ONLY on exceptions (L2365/2383),
  or when the yield lands MID-BATCH of a multi-tool-call response (L2350-2357,
  guarded by `tool_index < len(tool_calls) - 1`)
- L2400-2403 `finally: if preserve_history: self._history_messages =
  self._persistent_history_messages(messages, ...)` -- i.e. the completed generation's
  assistant message + tool results ARE COMMITTED to history and carried to the next invocation
- L2037-2054 `_persistent_history_messages` keeps the most recent
  `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` (L173) assistant turns, so the just-completed
  turn survives by construction (subject only to context-budget trimming)

## WHAT TO DELIVER
1. **ADVERSARIAL CODE VERDICT.** Try hard to REFUTE the above. Default to
   "discard IS happening" if you find any path where completed generated tokens are
   thrown away on a `turn_time_budget` yield. Specifically hunt for:
   (a) any path where `preserve_history=False` co-occurs with a turn_time_budget yield
       and how often that mid-batch condition can realistically hold;
   (b) whether `_trim_messages_for_context` inside `_persistent_history_messages` can
       evict the just-completed turn (context budget = ANALYZER_CONTEXT_WINDOW 32768);
   (c) whether the SOLVER side (search `inference/` + the bundle's solver/framework code
       for the caller of the analyzer, e.g. AnalyzerTurnResult consumers, `yielded_control`)
       discards or re-does the yielded turn's work;
   (d) whether the reported `generated_tokens` accounting attributes yielded-turn tokens
       to any environment action (see duck_eval/cadence/cadence_instrument.py docstring:
       `tail_tokens_no_action`).
   Give a one-word verdict: DISCARD / TRUNCATION / WORK-PRESERVING, plus the file:line cites.

2. **ARTIFACT LEG (quantify it).** In `runs/kernel_pulls/` there are retained pulls
   (q38_field_v1 is the certified field floor; also q38_v1, q38_v2, q38graft_v1, q38low_v1,
   private_base_v1, etc.). Look for analyzer transcripts / logs / benchmark.json.
   Measure, on the field-floor pull above all:
   - how many analyzer invocations ended with `Yielded control to solver: turn_time_budget`
     (the status string is written at tool_agent.py:2409-2410) vs "Step executed." vs
     "No action(...) call was captured."
   - of the yielded ones, what fraction eventually produced an action on a LATER invocation
   - the token cost sitting in yielded invocations (tie to cadence_instrument.py's
     `tail_tokens_no_action` if you can run it: 
     `uv run --no-project python duck_eval/cadence/cadence_instrument.py --validate` then on the pull)
   If the transcripts were not retained, SAY SO PLAINLY -- do not estimate. Report exactly
   which files you checked and what was absent.

3. **THE HONEST RESIDUAL.** If the yield is work-preserving, C3's stated mechanism is dead,
   but a WEAKER version may survive: raising 60->150 means fewer control-returns, so fewer
   prompt rebuilds (system prompt + user prompt + frame re-sent each invocation) and fewer
   context re-trims. Price that residual from the artifacts: how many extra invocations does
   the 60s yield actually cause per game, and what do they cost? State whether that residual
   is worth ONE OF TWO daily Kaggle slots against the certified comparator
   (field floor lc 28 + Arm A base lc 30 => mean 29.0, pooled seed sd 2.80).
   Be willing to say "NOT WORTH A SLOT."

4. Write your findings to: `learnings/war_room/c3_gate_read_2026-08-25.md`
   Structure: VERDICT (one word) / CODE EVIDENCE (file:line) / ARTIFACT EVIDENCE (or
   "NOT RETAINED") / RESIDUAL PRICING / SLOT RECOMMENDATION.
   Mark every claim [V]=verified-from-file or [INF]=inferred. No unmarked claims.

Return a compact summary as your final message.
