You are Professor of Empirical ML Methodology and Statistics (experimental design, multiple-comparisons, noise-band inference; rejects any plan that draws conclusions from single noisy samples).

You are reviewer #2 on a 5-person adversarial review panel evaluating a competition
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

**[PRIOR — UNRESOLVED] Provisional and conflicting numbers.** The brief again proceeds on 1.33/#49 and now cites KOJIMA at 1.86, while the panel briefing still carries 0.43/1.56. Three rounds in, the one-sentence reconciliation with an artifact path in `runs/verify_2026-07-21/report.md` has still not been written. This costs one sentence; its continued absence means every panel round reviews under a numbers conflict the team could have closed in July.

**[PRIOR — PARTIALLY-RESOLVED] Single-draw regime evidence / per-arm pre-registration.** The sentinel arm did have a filed template (`sentinel_w2_preregistration.md`, z-rule vs w0_s1=1.731, two-seed KILL α≈0.02), and the brief correctly refuses to feed the scored 0.71 into an eval-denominated rule — that is the discipline I asked for, operating. But the *entry* expectation ("2-seed canary PASS + non-harm screen") is not a regime-observable pre-registration, and the brief concedes no mechanism observable was pullable from the scored rerun ("logs hidden") — i.e., the window produced a score and nothing else, exactly the failure mode I predicted. Q3's "next arm" (war-v4) must not enter A21 without named non-score observables and a null criterion filed first.

**[PRIOR — UNRESOLVED] Retrofitted seal rule (A25).** The brief mentions "Panel R19 (routine, reduced bench)" but nowhere states that qualifying rounds for the seal begin at R18+ and that R16/R17 do not count. Silence persists; a "reduced bench" round drifting toward seal-qualification without the prospectivity sentence is precisely the risk. One sentence, still owed.

**[PRIOR — UNRESOLVED] Falsification disjunction / self-satisfiable endpoint.** The first draw has now been fired *and* analyzed, which incidentally satisfies my proposed replacement endpoint — but §6 itself has not been amended, no primary endpoints are designated, and "draw fired" presumably still counts on its own. The reset remains structurally unfalsifiable until the amendment is filed.

**[PRIOR — PARTIALLY-RESOLVED] Multiplicity control on the build-rail funnel.** The W2 rule is now calibrated (α≈0.02, named control) — progress. But the funnel objection just materialized as an outcome: the sentinel entered A21 via a canary PASS plus a non-harm screen whose pooled Δlc was *negative* (−0.05/game) with seed-flipped direction, while the eval rail had *both certified seeds negative* (s1 −0.315, s2 −0.166) and a sealed "fires, doesn't pay" verdict. The entry gate ignored the strongest pre-existing evidence and spent 1/12 windows confirming it. Fix: the A21 entry case must aggregate *all* prior evidence on the composition (eval seeds, screens, mechanism verdicts) into a single stated prior, and an arm with net-negative prior evidence needs an explicit "rail-transfer test" justification pre-registered — which, had it existed here, would have made the 0.71 a *confirmed prediction* and Q1 moot.

**[PRIOR — PARTIALLY-RESOLVED] E[max] derivation / wall trajectory.** The wall trajectory I demanded is now qualitatively confirmed — 1.44 "fully submerged," 20 teams ≥1.45, gold ≈1.49, ~4 ranks/day bleed — vindicating the flag, and `d4_provisional_reprice` exists. Still missing: the stated tail model, σ̂'s CI propagated into P(touch), and the 4-week linear fit (one overnight #45→#49 observation is not a trajectory estimate; see new objection below). The old P(touch 1.44)≈0.18 is now doubly moot and must be formally retired, not silently superseded.

**[PRIOR — UNRESOLVED, MINOR] Rule-of-three bounds on 29/29 and 49/49.** Not mentioned; sentinel n not extended. Carried.

**[PRIOR — PARTIALLY-RESOLVED] Pooled-posterior stationarity.** Reporting the draw against both frozen (n=10) and pooled (n=15) controls is a genuine robustness gesture, and the explicit exclusion rule ("0.71 does NOT enter — different composition") is the kind of prospective pooling statement I asked for, applied once. But the time-ordered trend/changepoint check on the 15 draws still does not exist, and the general pooling rule is still not written down. Ten minutes of work; do it before draw 2/12 is priced.

**[PRIOR — PARTIALLY-RESOLVED] Harm-pause error rates.** The brief's framing — pause as exposure control, not inference ("no inference is claimed from n=1") — is the correct defense and I accept it as far as it goes. But the error rates remain unstated: under the team's own pooled posterior, a healthy arm draws <0.80 with probability Φ(−1.125) ≈ 13%, so this pause is individually consistent with pure noise; it is the *corroborating eval-rail evidence*, not the 0.71, that makes shelving defensible here. A21 must state the 13%/power numbers and define the resume path (I proposed two-draw), or the next false pause will burn a window *and* an arm with no procedure to recover it.

**[NEW — MAJOR] Tail-model shopping: normal tails when they help, t-predictive when it helps.** The 0.71 significance is computed with a plain normal z (p≈0.044 vs frozen); the honest small-sample calculation is a t-predictive — t = −0.265/(0.156·√(1+1/10)) ≈ −1.62 on 9 df, one-sided p ≈ 0.07 — while the earlier P(touch 1.44)≈0.18 was only recoverable under a *fat-tailed* t-predictive. Fat tails were used where they inflated opportunity, thin tails where they inflate the negative signal; both choices flatter the narrative. Fix: declare one predictive model (t with ν = n−1 and √(1+1/n) inflation) in the ledger and recompute both; the sentinel conclusion survives (via the eval seeds), so this costs nothing except the false precision.

**[NEW — MAJOR] "Three independent negative signals" is an unpre-registered, independence-unverified aggregation.** The two eval seeds share the same build, harness, and composition — they are replicates within one rail, not independent signals — and no combination rule (Fisher, Stouffer, anything) was pre-registered before the phrase "the prior is now poor" was written. Meanwhile the brief correctly refuses to let 0.71 enter the W2 KILL rule formally, then immediately uses it informally to argue shelving — selective formality. Fix: either run the $0 pre-registered W2 instrument and close the line by the rule that exists, or write a one-paragraph disposition memo stating the evidence weights and their dependence structure before shelving. (This is my answer to Q1: the calibrated instrument exists and is $0 in build — use it if a slot is spare; do not set the precedent that sealed lines close by informal aggregation.)

**[NEW — MAJOR] Q5 (boristown 1.47 fork) silently invalidates the entire ledger calibration, and the brief doesn't notice.** Every A21 quantity — pooled μ=0.962/σ̂=0.144, the 0.80 pause threshold, filler-window pricing, E[max] — is denominated on the current fork band. Adopting a 1.47-based filler resets the control distribution to n=0: the pause threshold is meaningless against the new band, the exploration-window price changes (P(filler > 1.33) is no longer ≈0), and any arm composition drawn atop the new fork has no control to be compared against. Fix before adoption: a prospective re-baselining plan — first k fork draws (state k, e.g., 8–10) form the new control band with no arms riding on them, a new pause threshold derived from the new (μ, σ̂), and a written rule for whether old-band draws are retired or kept as a separate stratum. Answer to Q5(iii): it is neither filler nor arm — it is a *baseline change*, a third category the accounting must define.

**[NEW — MINOR] "~4 ranks/day bleed" is a single-overnight-delta trend claim.** One #45→#49 observation is being extrapolated into a strategic backdrop ("continue to erode") that feeds the depth-lane urgency argument. Rank deltas near a dense band are high-variance; fit the last 7–14 days of rank/score history (the data exists in the daily briefs) before "bleed rate" enters any allocation decision.


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
