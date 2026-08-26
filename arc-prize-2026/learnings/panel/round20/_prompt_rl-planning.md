You are Professor of Reinforcement Learning and Planning (MCTS, model-based RL, exploration theory; 20 years; famously skeptical of under-specified search claims).

You are reviewer #1 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026).

GROUND-TRUTH LEADERBOARD STATE (refreshed 2026-07-25 from the live Kaggle API; the
draw-by-draw submission ledger is at runs/lb_ground_truth.md; treat THESE numbers as canonical —
any different numbers you may remember from earlier rounds were a stale briefing):
# LB ground truth — refreshed 2026-07-25 by the daily loop (live Kaggle API)

Account: canivel (Danilo Canivel, d.canivel@gmail.com). Competition:
arc-prize-2026-arc-agi-3. Verification command:
`uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`.

- OUR BEST (public LB): **1.33** (frozen-fork filler draw, 2026-07-18). Current rank
  ~#50–53 (slid out of the loaded top-50 overnight; 1.33–1.34 is a crowded floor).
- LEADER: YUTO KOJIMA **1.86**. #2 Tecnod8.AI 1.61, #3 DhanaLakshmiMalla 1.60,
  #4 ippeiogawa 1.58. Gold cutoff ≈ **1.49** (top 13; #14 = 1.48).
  Dense band 1.44–1.61; 7+ teams at 1.46–1.47 (boristown's public 1.47 seeding).
- External context: Claude Opus 5 posted 30.2% on the ARC-AGI-3 benchmark (arcprize.org,
  Jul 24) via API at High reasoning effort — different regime (unconstrained API vs
  Kaggle quantized/time-limited local), no artifact to lift; directional support for
  capability-over-harness.
- The "best 0.43 / leader 1.56" figures in pre-R19 briefings were a STALE HARDCODED
  TEMPLATE (May-era), root-caused and fixed 2026-07-24 (panel_round.py now reads this
  file). Reconciliation: 0.43 was the team's best in early May (forge-era agents);
  the frozen duck fork lifted the floor to the 0.82–1.33 band from 2026-07-05 on.

## Draw-by-draw scored ledger (all API-verified)

Frozen-fork control (n=11): 0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
1.05 → mean 0.982, s ≈ 0.150. War arm (n=5, CLOSED per A9): 0.91, 1.08, 0.88, 1.05, 0.76.
Sentinel exploration arm (n=1, HARM-PAUSED 07-24, SHELVED by disposition memo): 0.71.

Recent tail (newest first): 1.05 filler (07-25) · 0.71 sentinel (07-24) · 0.82 filler
(07-23) · 1.14 filler (07-22) · 0.93 filler manual (07-21) · 0.92 filler (07-20).

External anchors: byte-identical public forks of the same duck artifact family have
drawn 1.39 (zoli800) and 1.47 (boristown agi-duck-harness-fast-eval, whose only real
functional diff is a vLLM readiness gate — see
learnings/war_room/fork_diff_boristown_2026-07-24.md). Artifact tail ≥ 1.47 confirmed.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

**PRIOR-OBJECTION RESOLUTION STATUS (all verified before new comments):**

**[RESOLVED] Prior FATAL: R5/RC4 — scored-regime bottleneck.** Confirmed still resolved and now operationally validated: exploration draw 1/12 fired, scored, and was dispositioned under the pre-registered rule. No relapse.

**[UNRESOLVED → ESCALATED, see FATAL below] Prior MAJOR: cited LB state conflicts with panel briefing.** Second consecutive round with zero reconciliation. The brief now cites best = 1.33 (#45→#49), leader KOJIMA 1.86, gold ≈ 1.49; the panel briefing says best 0.43, leader 1.56. No submission logs, no account identification, no acknowledgment the discrepancy exists. Escalated below.

**[PARTIALLY-RESOLVED] Prior MAJOR: §6 falsification disjunction.** Mooted by events, not fixed by text: the scored-regime item (first exploration draw) actually fired, so the pathological validation path I described cannot occur *this cycle*. The rule text was still not amended to make the scored-regime item load-bearing. One sentence. Write it.

**[UNRESOLVED] Prior MAJOR: tail-model inconsistency.** The brief re-uses the same Gaussian machinery (z = −1.70/−1.75 vs σ̂ = 0.144) and prices the spent window at "~0.001–0.002 E[max]-equiv" with no GPD/mixture fit, no empirical exceedance count at 1.33, and no iid/stationarity statement. The demanded artifact — one fitted tail model from which *both* the "filler-only is losing" claim and the "exploration is nearly free" claim are derived — does not exist. All published prices in this brief remain unaudited.

**[UNRESOLVED] Prior MAJOR: promotion gate unreachable; harm-pause miscalibrated.** The harm-pause fired exactly as I modeled — n=1, threshold 0.80, no false-pause probability stated, no 2-of-3 requirement adopted. In this instance the pause is probably correct (three aligned negatives), but the rule got the right answer for reasons outside its own denomination, which is luck, not calibration. The promotion side is untouched: no arm can still ever be credibly promoted at +0.06 mean-lift with 11 remaining windows, and promotion remains priced in mean-currency. This must be fixed before exploration draw 2 — otherwise the remaining 11 windows are drawing toward a gate that cannot mathematically open.

**[PARTIALLY-RESOLVED] Prior MAJOR: asymmetric stopping rule.** "NO-GO at modest lift is the designed outcome" and the ≥+4-to-+5-level detector framing partially pre-registers the capability threshold, and the loud-fail-on-serve policy is good hygiene. But the canary pushes *today* and the quantified false-GO probability and the exact GO/NO-GO/CONTINUE numeric boundaries are still not written down pre-run. The amendment remains one paragraph and remains unwritten.

**[UNRESOLVED] Prior MINOR: seal-termination downgrade logging.** Not mentioned in this brief. Carried.

**[PARTIALLY-RESOLVED] Prior MINOR: build-rail mutation accounting.** The ρ_action-poisoning analysis (silent 27B fallback would corrupt the frozen numerator) is exactly the mutation-awareness I asked for, applied to one push. The general column in the GPU-hour table — which rail state each push mutates — is still missing for the other queue items.

**[UNRESOLVED] Prior MINOR: A21 allocation policy.** Q3 asks the panel to improvise pacing instead of presenting a policy. Acknowledgment is not a fix; one paragraph (max concurrent arms, priority order, reallocation-from-paused rule, front-load vs. spread) is still owed and is now urgent since window 2 is imminent.

**NEW OBJECTIONS:**

**[FATAL] LB ground truth unverified for a second round — all pricing floats on an unaudited number.** Escalated from the prior MAJOR per my stated stakes: if the briefing's 0.43/1.56 is correct, the control ledger (n=15, mean 0.962), the harm-pause threshold, the erosion narrative, and the fork-EV math are all fiction; if 1.33/1.86 is correct, the briefing is stale and the panel is reviewing under false premises. This is a five-minute fix — attach draw-by-draw submission logs with the LB account named and screenshot-date the leaderboard — and its two-round survival is itself evidence of a process problem. I will not price-audit any further number until this is closed. Until then, every E[max] figure in the document carries an asterisk.

**[MAJOR] The entry bar admitted an arm with known-negative sealed evidence — window 1/12 was spent at negative expected VOI.** By the brief's own account, *before* the draw fired: the screen showed Δlc ≈ −0.05/game with direction flipping across seeds, the eval rail was negative on BOTH certified seeds (s1 −0.315 at p=0.997), and the sealed W1 mechanism story said "fires, doesn't pay" with the team's own doctrine holding that efficiency observables have no depth channel under the completion-weighted scorer. A "canary PASS + non-harm screen" bar that admits an arm with two aligned negative rails and a doctrinally-priced-at-zero upside is not an entry bar; it is a formality. With 11 windows left, fix the bar to require positive right-tail evidence (e.g., eval-rail point estimate > 0 on ≥1 seed, or a mechanism story with an identified depth channel), not mere non-harm — exploration budget should buy exceedance probability, not confirm sealed nulls on the scored rail.

**[MAJOR] The boristown 1.47 fork is the highest-EV move in the document and it is filed as question 5 instead of action item 1 — and its downstream accounting consequences are unexamined.** A near-free +0.14 floor-raise dominates everything else on today's slate (compare: one frontier depth event prices +0.19–0.29 at real risk; the sentinel W2 prices +0.00). Nobody has done the second-order math: adopting a 1.47 floor (a) obsoletes the entire control ledger — harm-pause at 0.80 and the n=15 pooled control are denominated on the old filler and need re-basing with fresh draws before any A21 rule is meaningful again; (b) *strengthens* the exploration case, since P(old-filler draw > 1.47) ≈ 0 even under their own Gaussian (z ≈ 3.5), making exploration windows genuinely near-free post-fork — the one place their "exploration is cheap" claim would actually become true. Answer to their Q5(iii) from exploration theory: a public fork is a filler-replacement, not an arm — no window cost, but it mandates a control re-baseline (n ≥ 5) before harm-pause thresholds re-arm. Fork-diff today; this should not have waited for a panel question.

**[MINOR] Slot-2 adjudication: the W2 confirmatory null is zero-VOI and should lose to tr87 by inspection.** The sentinel arm is paused, has three aligned negatives, and by the brief's own note "cannot re-enter LB regardless without a new A21 entry case" — so W2's outcome changes no live decision, which is the definition of zero value-of-information. Confirming a kill on a dead arm is ritual, not inference. tr87 is the only non-A17 depth-targeting line and depth is where all the priced upside lives; the concentration-risk argument in their own Q2 answers their own question. Shelve the sentinel on W1 + harm-pause; log it as "closed on eval evidence, scored draw consistent."

**[MINOR] Target denomination is stale: all lift-pricing is against today's wall, but the decision horizon is Nov 2.** Gold moved 1.44 → 1.47 → 1.49 in days with 20 teams ≥ 1.45 compressing upward; a +0.14 fork lands the team *at today's* cutoff, which will not be November's cutoff. The d4 reprice (+0.19–0.29) and all GO/NO-GO capability thresholds should be denominated against a forecast Nov-2 gold band (even a crude linear/logistic extrapolation of the cutoff time series, with a sensitivity band), not the current snapshot. This is a one-cell spreadsheet and it changes what "enough lift" means for every gate.


=====================================================================

THE PROPOSAL (sha256 of the full document: d018f3ed6c94f989; full length 8942 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
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
   repo). Author recommendation: **(i) + (iv) + (v) as v4** — (i) is the
   only layer validated against our own traffic and is model-agnostic;
   (ii)/(iii) alter what the model is asked to emit and can ride v5 if v4's
   recovery rate on-node disappoints. One push, ~2.5 GPU-h. If authorized:
   does turn≈action suffice for canary-stage ρ_action, exact parity deferred
   to promotion?
4. boristown §(i) monitored-continuation filler replacement: schedule now or
   still hold for A17 outcome?
5. Exploration draw 2/12: entry bar §(c) — no arm currently clears it
   (sentinel shelved, war-v4 waits on A17). Confirm filler rides.

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
