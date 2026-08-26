# Daily brief — 2026-07-26 (Sunday)

## §1a Result deep-dive

### Scored window (00:07Z): frozen-fork filler = 0.84 — in-band, frozen n=12

**Draw:** `canivel/arc3-duck-repro` v3 scored **0.84**. Pre-registered
expectation: frozen-control draw from band 0.82–1.33 — met (lower half;
t ≈ −0.9 under the declared t-predictive, unremarkable). Enters the frozen
stratum: n=11 → **n=12: chronological {0.82, 0.89, 0.93, 1.02, 0.95, 1.33,
0.92, 0.93, 1.14, 0.82, 1.05, 0.84}, mean ≈ 0.970, s ≈ 0.148** (exact
recompute belongs to the ratified machinery — amendment still DRAFT, see
incident below). No trend claim; 07-24 MK/CUSUM no-trend verdict stands.

### A17 72B canary v3: COMPLETE, 0.00/0-actions — root-caused to TOOL-CALL
### FORMAT ADHERENCE; envelope CLEARS; 99.5% offline-recoverable

Full forensics: `runs/kernel_pulls/a17_canary_v3/analysis.md`. Headlines:

- **The /2 model pin WORKED** (v1's attach defect fixed): weights mounted,
  GPU asserts passed, vLLM served the 72B-VL healthily for 2h12m —
  **0 stalls, 0 restarts, gen_tps mean 34.3 aggregate over 4 concurrent
  games** (~67k generated tokens/game). A23 envelope question: **FITS.**
- All boot asserts passed **including the hermes tool-call round-trip** —
  but in-game, under the real ~31k-token context, the model emitted its
  python as **markdown-fenced ```python blocks instead of hermes
  `<tool_call>` markup on 1200/1200 LLM responses** → `tool_call_count: 0`,
  `step_executed: False` everywhere → 0 actions, all 4 games `gave_up`,
  score 0.00. The short boot probe passes because a minimal context elicits
  the trained tool-call format; the long duck prompt (which itself contains
  fenced-code examples) reverts the model to markdown. The harness's
  existing markup-recovery path covers tool-call markup in text, NOT fenced
  code, so it never fired.
- **Offline recovery replay (deterministic, $0, `runs/a17_recovery_replay/`):**
  extracting the fenced block as a python tool call recovers **434/436
  analysis turns (99.5%)** — every recovered turn is exactly one
  ast-valid block referencing `action()`/`current_frame`. Cadence datapoint:
  436 turns across the 4 screen games/window vs the 27B numerator's ~480
  actions — **ratio ≈ 1.1x** (turn≈action caveat on record), far inside the
  3.5x envelope NO-GO penalty.
- **Interpretation (C3 discipline): no capability claim either way** — the
  action channel was severed upstream of the games; the sealed §9.1 gate
  boolean was never evaluated. Three canary defects in sequence, each
  narrower, each forensically closed (v1 attach → v3 format); all hard
  physical risks (fit, serve health, vision path, throughput) now cleared.
  **Candidate v4 = v3 + fenced-python recovery adapter** (harness-side only,
  validated at 99.5% on real recorded traffic, new banner
  `fenced-recovery=on hits=<n>`).

### INCIDENT: R20 never ran on 07-25

`learnings/panel/round20/` contains only the three reviewer prompts — the
07-25 session died after prompt-write, before any reviewer launched.
Consequence: **amendment 2026-07-24 is still DRAFT** (t-predictive model,
§(a)–(i), model-pin /1→/2 deviation) and the 07-25 open questions are
unanswered. Same wedge class as 07-21/07-22. R20 relaunches TODAY on this
brief (3 reviewers, prior=R19). Queue was also EMPTY at session start —
refilled 08:30 EDT with frozen-fork filler (head swappable by panel ruling
until ~18:00).

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-26.md)

- Feed QUIET — zero new threads. #728278 "Is 100% Realistic" gained
  comments: **ADAPT (intel)** — community independently converges on our A17
  thesis (single-27B on ~100GB is the binding cap; a larger model is the
  scaling axis). The public ~70%/~36% numbers are public-games/paid-API, not
  the Kaggle sandbox. **No community intel exists on our actual blocker**
  (Qwen2.5-VL tool-call format under vLLM) — it is ours to solve.
- #728934 Opus-5 30%: IGNORE for config (new comment confirms off-harness).
  #684625 vLLM silent-hang: open, unchanged; concurrency<8 + watchdog ADAPT
  holds. boristown 1.47: score unchanged, upvotes 151→165; defensive-diff
  posture stands.
- **LB:** KOJIMA 1.86 #1; top compressing — four teams ≥1.56 (Yuchen20 →#5
  at 1.58); gold cutoff ≈1.49 (top-13). boristown #15→#16. Our 1.33 eroded
  to ~#50+ (neighbors at 1.34). Gold wall +0.16 above us. Strategic read
  unchanged: only a depth event moves us; A17 is that lane.

## §1c Research sweep (learnings/war_room/research_2026-07-26.md)

- **PRIORITY topic resolved with citations: the 1200/1200 fenced-code failure
  is a KNOWN Qwen tool-format pathology, not primarily our prompt length.**
  Community evidence: bare hermes parser with Qwen collapses to ~60% fenced
  code + 40% plain JSON (~0% hermes markup) under verbose prompts.
- **ADOPT (v4 primary): `tool_choice="required"` + vLLM structured outputs
  (xgrammar)** — server-side FSM forces schema-valid tool JSON, bypassing the
  fenced-code channel entirely. Pre-registered caveats: "Failed to advance
  FSM" failures at low temp/complex schemas (vLLM #16321); must validate the
  ACTION6 schema against xgrammar locally; restate schema in prompt.
- **ADAPT-low (v4 fallback, no server change): few-shot `<tool_call>`
  exemplars in the system prompt** — plugin evidence shows 100% compliance
  holding under ~90-line verbose prompts.
- **Two cheap confounder checks for v4:** (1) non-streaming requests (open
  hermes streaming raw-text bug vLLM #31871, repro'd on 0.13.0 — plausibly
  OUR exact defect); (2) dump the served 72B-AWQ chat template and confirm
  tool support present (Qwen2.5-VL AWQ has a template-strip history,
  issue #1093).
- Standing sweep: no new in-window papers; Opus-5 30.2% unchanged (PARK);
  2511.15703 PARK as a VL-prompt design note ("naively rendering ARC grids
  as images hurts precise rule execution" — keep textual grid encoding
  alongside frames); schema replication still ZERO at 50 upvotes.

## §1w Weekly fingerprint table (Sunday)

16 incidents, 8 recurring families, **no NEW incidents this week** (latest
family member still 07-08; the A17 canary ERRORs/zero-action runs are eval
kernels, tracked in the lane's own forensic chain, not the submission
fingerprint store):

| family | n | first | last |
|---|---|---|---|
| class:ERROR:none | 7 | 05-26 | 06-28 |
| provenance:scratch-built | 5 | 05-26 | 06-28 |
| slug:arc3-final | 4 | 05-26 | 06-10 |
| class:COMPLETE:0.00 | 3 | 03-29 | 06-10 |
| slug:arc3-forge35 | 3 | 04-24 | 06-22 |
| slug:arc3-pilot-eval | 3 | 07-07 | 07-08 |
| t1:07d0f524 | 3 | 07-07 | 07-08 |
| class:COMPLETE:null-band | 2 | 06-01 | 06-08 |

Weekly KAOS: ingest +40 rows (184 total); dream digest
`Dreams/2026-07-26-123133.md` — recency-only, skills_scored=0 (matches the
sealed expectation).

## §2 Today's plan

1. **Panel R20 (relaunch, 3 reviewers: rl-planning, methodology, systems;
   prior=R19):** ratify amendment 2026-07-24 §(a)–(i) + model-pin deviation;
   rule on canary v4 authorization; boristown adoption timing; tonight's
   queue head.
2. **On v4 authorization:** build canary v4 (fenced-python recovery adapter
   in the A17 setup-rewrite; boot asserts unchanged; new recovery banner),
   smoke incl. transcript replay, push (0/2 pipeline pushes used today).
3. **Queue:** filler head armed at 08:30; swap only on explicit panel ruling.
4. Weekly items: DONE (above).

## Open questions (for R20)

1. Ratify amendment 2026-07-24 §(a)–(i) (unchanged ask from the 07-25 brief;
   all numbers from `runs/r19_hygiene/`)?
2. Accept the /1→/2 model-pin deviation (48/48 weight shards size-identical;
   argument: size+name+card identity suffices for a canary whose GO is
   re-certified at promotion)?
3. **Authorize canary v4, and pick its composition.** Available layers, all
   $0 to stage: (i) fenced-python recovery adapter (harness-side, validated
   99.5% on our real recorded traffic); (ii) `tool_choice="required"` +
   xgrammar structured outputs (server-side forcing; standard mechanism but
   FSM-failure caveats, needs local schema validation); (iii) few-shot
   `<tool_call>` exemplars in the system prompt; (iv) non-streaming requests
   (kills the open hermes streaming bug as a confounder); (v) chat-template
   tool-support verification at boot (template-strip history on the AWQ
   repo). **Confounder checks (iv)/(v) resolved MOOT this morning:** the
   harness is non-streaming (`openai_compat.py` `"stream": False`) so the
   hermes streaming bug cannot be our defect, and the passing boot tool-call
   probe already proves the served template carries tool support. Author
   recommendation: **(i) alone as v4** — the only layer validated against
   our own traffic (staged + smoked TODAY: `duck_eval/a17/
   fenced_recovery_patch.py`, smoke 18/18 incl. 434/434 replay through the
   real patched module); (ii)/(iii) alter what the model is asked to emit
   and ride v5 only if v4's on-node recovery disappoints. One push,
   ~2.5 GPU-h. If authorized: does turn≈action suffice for canary-stage
   ρ_action, exact parity deferred to promotion?
4. boristown §(i) monitored-continuation filler replacement: schedule now or
   still hold for A17 outcome?
5. Exploration draw 2/12: entry bar §(c) — no arm currently clears it
   (sentinel shelved, war-v4 waits on A17). Confirm filler rides.
