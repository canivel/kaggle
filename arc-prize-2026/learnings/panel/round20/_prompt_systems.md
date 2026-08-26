You are Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, quota economics; kills plans that don't fit the compute envelope).

You are reviewer #3 on a 5-person adversarial review panel evaluating a competition
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

**[Prior FATAL 1 — quota is not free] PARTIALLY-RESOLVED (no change; precondition violated).** The SKU half improved indirectly: two independent external sources (#684625 Scott Le Grand reproducing the vLLM hang on RTX Pro 6000 with the duck notebook; lastloop-ai's vllm-blackwell-guide for sm_120) now corroborate the Blackwell rail. But the demanded first-party artifact (`nvidia-smi` / `get_device_properties` output from the team's own kernel log, with path) is still unattached, and the quota-exemption footnote for scored windows has **zero** new evidence — and Q5(iii) ("is a public fork a 'filler' (no window cost) or an arm?") reveals the team itself does not know the answer. The A17 canary is scheduled for slot 1 today *before* this evidence exists, which inverts the pre-registration ordering I required.

**[Prior FATAL 2 — A17 envelope] PARTIALLY-RESOLVED.** The parity prong is now denominated on measured end-to-end action throughput (ρ_action with 27B numerator frozen at 480 actions/7920s), which is exactly the fix I asked for, and NO-GO-at-modest-lift is stated as the designed outcome. But the numeric GO/NO-GO threshold on ρ_action and the derived turns-per-level floor live in unattached references (C3, `a17_error_model`) — this brief does not let me verify the arithmetic. See also the new concurrency-confound objection below, which attacks the ρ_action construction itself.

**[Prior MAJOR — watchdog kills the bench] RESOLVED.** Remains resolved; the canary's log-heartbeat observable implements my cold-start note. No regression.

**[FATAL, escalated from Prior MAJOR — headline numbers contradict the briefing] UNRESOLVED, third consecutive round.** The brief asserts best 1.33 / KOJIMA 1.86 / gold ≈1.49 / rank #49; the panel briefing states best 0.43 / leader 1.56. Not one line reconciles them, despite this being an explicit condition of approval last round. Every economic quantity in today's brief — the +0.14 fork delta, the "4 ranks/day bleed," the +0.19–0.29 depth-event repricing, the E[max] currency of the A21 ledger — is denominated in the unreconciled metric. I escalate to FATAL: a strategy document whose objective function cannot be tied to the ground-truth leaderboard is unactionable, and repeated silence suggests either a stale artifact or a normalization the team has not audited. One paragraph with the formula or the artifact resolves this; its continued absence is itself the finding.

**[Prior MAJOR — RC4/R5 pricing contradiction] RESOLVED.** Better than resolved: the 0.71 draw is the mechanism's first live firing — pause executed pre-loop, no n=1 inference claimed, cost booked against the 12-window ledger. This is the strongest section of the brief.

**[Prior MINOR — no tail model] PARTIALLY-RESOLVED (no change).** The 0.71 analysis reuses the implicit-Gaussian z-machinery (z=−1.70/−1.75, one-sided p≈0.04) on n=10/15 with no stated fit family or CI. Not load-bearing today (three aligned negatives across two rails is a robust qualitative signal), but the requested fit statement and CI have now been ignored twice.

**[Prior MAJOR — single-point hardware claim] PARTIALLY-RESOLVED.** Merged with FATAL 1's SKU half above: external corroboration is real progress and the canary's "self-certifying envelope" will settle the fit half empirically. But the pre-registration demand — attach the scored-rail and bench-rail device artifacts *before* quota is spent — is being violated by today's slot-1 push. This costs ten minutes; do it before the push, not after.

**[Prior MINOR — contingency line] UNRESOLVED (not addressed).** This brief contains no budget table, so I cannot verify the re-lining (contingency ≥ 1× largest run, setup itemized). Carried forward; note the "full-window" canary description still does not itemize the 43GB download/load/warmup, which directly affects whether the denominator window is 7920s of serving or 7920s minus 45–90 min of setup.

**[MAJOR — new] ρ_action is confounded by concurrency regime; the bias direction produces false GOs.** The 27B numerator (480 actions/7920s) was measured from certified 25-game runs; the 72B denominator will be measured at 4-game concurrency. Under vLLM continuous batching, per-game action latency *improves* at lower concurrency (less queueing, more KV headroom per stream), so the 72B's per-game action rate at 4-way is flattered relative to its rate at deployment concurrency — ρ_action will overestimate 72B parity, and the screen's parity prong can pass a model that is out of envelope at real load. Fix before interpreting the canary: either (a) add a 27B 4-game control leg on the same rail to renormalize the numerator at matched concurrency (cheap — ~1 window or bench-rail), or (b) pre-register a stated correction model with the C3 threshold tightened accordingly.

**[MAJOR — new] The hang-risk mitigation is diagnostic, not preventive, and the "safe" concurrency margin is not transferable to the 72B.** The reported hang threshold (≥8 concurrent sessions, 15–20 min) was observed on the *27B duck* stack; hang thresholds of this class are typically memory-pressure dependent, and 72B-AWQ at ~43GB weights leaves materially less KV headroom, so 4-game concurrency is not demonstrably below the danger zone for *this* config. As written, a silent hang burns the full scored window and slot 1, and is only diagnosed post-run via the heartbeat log. Fix: add an in-run liveness gate (no completed action in N minutes → one server restart, then loud-fail) and/or a 20–30 min bench-rail smoke at 4-way before committing the scored window; watch #684625 for root cause is not a mitigation.

**[MINOR — new] Q5's accounting question must be answered by evidence, not definition.** Whether a boristown fork submission is "filler (no window cost)" is precisely the FATAL-1 quota-exemption question wearing a new hat; declaring it filler by fiat would launder the unevidenced footnote into policy. The fork-diff-first ordering ($0, today, byte-matched metadata) is correct and I endorse it; the *classification* waits on the quota ledger.


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
