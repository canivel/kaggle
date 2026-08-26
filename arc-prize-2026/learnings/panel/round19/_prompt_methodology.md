You are Professor of Empirical ML Methodology and Statistics (experimental design, multiple-comparisons, noise-band inference; rejects any plan that draws conclusions from single noisy samples).

You are reviewer #2 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026). The proposing team has a
best score of 0.43; the leader is at 1.56; the winning Milestone-1 notebook is public.

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

**[PRIOR — PARTIALLY-RESOLVED] Provisional and conflicting numbers.** The verification run now exists (`runs/verify_2026-07-21/report.md`, "4 discrepancies found and adopted"), which is what I asked for. But the explicit reconciliation I demanded — panel briefing 0.43/1.56 vs. proposal 1.33/1.44+ — appears nowhere; the revision simply proceeds on 1.33 as if the briefing conflict never existed. The entire A21 opportunity-cost calculation conditions on "current best 1.33"; state in one sentence, with the artifact path in the verify report, which number is stale and why the briefing carried it.

**[PRIOR — PARTIALLY-RESOLVED] Single-draw regime evidence.** The pricing half is genuinely fixed: under max-currency, displacing one of ~102 filler draws costs ≈ P(draw > best)·E[exceedance], and I can reproduce their ~0.001–0.002 per window and ΔE[max] ≈ −0.01 to −0.02 for 12 windows. But the evidence half of my objection is untouched: A21 still does not pre-register *which non-score observables* an exploration draw is expected to produce per arm and what constitutes a null result. "No 2nd window before the 1st is analyzed" is sequencing, not pre-registration; without a per-arm analysis template filed before the draw, the first exploration result will be interpreted post hoc. Fix: the one-page intent (A22) must include, for any arm entering A21, the named regime observables and the null criterion.

**[PRIOR — PARTIALLY-RESOLVED] Retrofitted seal rule (A25).** The revision no longer asserts "R16 already qualifies," which removes the most egregious double-dip, but it does not state that both qualifying rounds must occur after A21–A25 ratification, and it did not adopt the "0 FATALs and no *new* MAJORs" pairing. Silence is not prospectivity. One sentence fixes this: "the two qualifying rounds begin with R18 or later; R16/R17 do not count."

**[PRIOR — UNRESOLVED] Falsification is still a disjunction — and now contains a self-satisfiable endpoint.** §6 is "unchanged in spirit": five heterogeneous endpoints, refutation only if *none* lands. Worse, one endpoint — "first exploration draw fired" — is satisfied by the team's own act of submitting, independent of any outcome, so the reset is now *structurally unfalsifiable*: the team can always fire a draw by Aug 6. Fix as previously specified: designate one or two primary endpoints (A17 capability/parity/envelope numbers; sentinel verdict at adequate n) whose joint failure refutes the reset, and strike "draw fired" as an endpoint or replace it with "draw fired *and analyzed against its pre-registered template*."

**[PRIOR — UNRESOLVED] No multiplicity control on the build-rail funnel.** A22's intent template still lists only "metric, canary, kill rule" — no pre-specified n, no held-out confirmation, no sibling-count/shrinkage disclosure at the canary+screen bar that now gates entry to A21's 12 windows. The low priced cost of exploration windows mitigates the *consequence* of winner's curse but does not remove the *selection* problem: 12 windows allocated to the luckiest of many free experiments is still a funnel. And the sentinel remains a 2-seed verdict (§2 table) with no statement of what effect size 2 seeds resolve.

**[PRIOR — PARTIALLY-RESOLVED] E[max] derivation.** Inputs are now shown (n=15, mean 0.962, σ̂ 0.144, ~102 remaining), and E[max] ≈ 1.35 is in the right band. But P(touch 1.44) ≈ 0.18 is *inconsistent with a normal tail*: z = (1.44−0.962)/0.144 ≈ 3.32 gives per-draw p ≈ 4.5×10⁻⁴ and 1−(1−p)^102 ≈ 0.045, a 4× discrepancy. The number is recoverable under a t-predictive with ~14 df (per-draw p ≈ 0.0025 → ≈ 0.23), which suggests they used parameter-uncertainty-aware tails — but the model is nowhere stated, and the strategy hinges on this figure ("down 40% from the stale 0.29"). Also still missing: any wall-trajectory estimate — 1.44 is again treated as static for 102 days with a public winning notebook, which I explicitly flagged. Fix: state the tail model, show σ̂'s CI propagated into P(touch) (with n=15, the 95% CI on σ spans roughly 0.10–0.23, moving P(touch) by an order of magnitude), and attach the 4-week leaderboard linear fit.

**[PRIOR — UNRESOLVED, MINOR] Rule-of-three bounds on 29/29 and 49/49.** Not addressed; sentinel n was not extended despite the free build-rail track existing precisely for this.

**[NEW — MAJOR] The pooled 15-draw posterior assumes i.i.d. stationarity across regimes and dates, untested.** §0 pools 12 fork-band draws with 3 others into one (μ, σ̂) that prices everything in §1. These draws span weeks of config churn, two killed live mechanisms, and 8/11 days of infra incidents — a textbook non-stationary mixture. A single drifting regime inflates σ̂ or biases μ and silently reprices A21. Fix: show the 15 draws time-ordered with a trend/changepoint check (even a runs test or a split-half mean comparison suffices at this n), and state the pooling rule prospectively so future draws can't be selectively included.

**[NEW — MAJOR] The harm-pause rule "arm draw < 0.80" is a single-noisy-sample decision with ~13% false-pause and near-zero power.** Under the team's own posterior, P(a *perfectly fine* arm draws < 0.80) = Φ((0.80−0.962)/0.144) ≈ 0.13 — so roughly 1–2 of the 12 exploration windows will pause a healthy arm on pure noise; conversely, a genuinely harmful arm (say true mean 0.85) escapes the pause ~2/3 of the time. This is the exact single-sample inference pattern the panel exists to reject, now embedded in the reset's flagship mechanism. Fix: either state and accept these error rates explicitly in A21, or make the pause a two-draw rule (both < 0.80, false-pause ≈ 1.7%) given that pausing—resuming is cheap.


=====================================================================

THE PROPOSAL (sha256 of the full document: 8c1df549836e36b1; full length 9225 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# Daily brief — 2026-07-24

## §1a Result deep-dive

### Exploration draw 1/12: sentinel arm = 0.71 — campaign low; HARM-PAUSE executed

**Draw:** `canivel/arc3-duck-sentinel` v1 (vanilla duck + (f) continuation
hygiene-default + budget sentinel v2 @ SENTINEL_BUDGET=150; NO warpack, NO
ledger) scored **0.71** at the 00:07Z fire — the lowest scored draw of the
campaign (previous low: war draw #5 = 0.76; frozen-fork low 0.82).

**Pre-registered expectation vs outcome:** entry was via 2-seed canary PASS +
non-harm screen (pooled Δlc ≈ −0.05/game, direction flipped across seeds).
The sealed A21/C2 rule — draw < 0.80 pauses the arm — **triggered; the pause
was executed pre-loop** by the early interactive session. This is the rule
operating exactly as designed: no inference is claimed from n=1, but exposure
stops immediately.

**Statistics (validated this morning):**
- vs frozen control n=10 (mean 0.975, σ̂ 0.156): **z = −1.70**, one-sided
  p ≈ 0.044.
- vs pooled control n=15 (mean 0.962, σ̂ 0.144): z = −1.75, p ≈ 0.041.
- A single draw at p ≈ 0.04 is suggestive, not conclusive — but it is the
  THIRD independent negative signal for this composition: eval-rail
  Δlog1p(RHAE) was negative on BOTH certified seeds (s1 −0.315 p=0.997; s2
  −0.166 p=0.90). Three aligned negatives across two rails = the prior for
  the sentinel composition is now poor.

**Mechanism evidence:** none pullable from the scored rerun (logs hidden);
the composition's mechanism story is already sealed from the eval rail (W1:
mechanism PASS / behavior "fires, doesn't pay" — warnings changed nothing,
21/22 fired games kept grinding). The 0.71 is consistent with the sealed
story: the sentinel adds FACT lines that at best do nothing and at worst
perturb the policy slightly negative, and under the completion-weighted
scorer an efficiency observable has no depth channel to pay through.

**Cost accounting:** one exploration window of 12 (A21); priced ~0.001–0.002
E[max]-equiv in max currency. 11 windows remain. LB best UNCHANGED at 1.33.

**Disposition question for panel (Q1 below):** the calibrated W2 rule
(sentinel_w2_preregistration.md amendment: z-rule vs legal control w0_s1 =
1.731; two-seed KILL α ≈ 0.02) is denominated on the EVAL rail — the scored
0.71 does not formally enter it. Is the $0 W2 eval build still worth a push
slot to close the line cleanly, or does the harm-pause + sealed W1 evidence
suffice to shelve the sentinel as "certified telemetry, no lift channel"
without W2?

### Ledger

Frozen control unchanged (0.71 was a different composition — it does NOT
enter): n=10 mean 0.975 σ̂ 0.156, band 0.82–1.33. Pooled n=15 mean 0.962 σ̂
0.144.

### LB context

Field compression continues: **20 teams ≥ 1.45** this morning (the old 1.44
wall is fully submerged; yesterday gold ≈ 1.47). Top: KOJIMA 1.86, Tecnod8.AI
1.61, then a dense 1.45–1.60 band. We hold 1.33 (~#45) and continue to erode
without a depth-channel gain. This is the strategic backdrop for the A17
screen and the depth-budget lane: efficiency channels price ≈ 0; a single
frontier depth event prices +0.19 to +0.29 rail (d4_provisional_reprice).

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-24.md)

- **ADAPT (ops-critical, TODAY) — #684625 Scott Le Grand comment:** vLLM
  silently hangs on RTX Pro 6000 with the duck notebook after 15–20 min at
  ≥8–25 concurrent sessions (reproduced off-Kaggle). Directly relevant to
  today's A17 canary (same GPU + vLLM stack). Mitigation posture: the canary
  runs 4-game concurrency (below the reported ≥8 threshold), and our
  certified 27B 25-game runs completed clean at 2h12m — but the canary gains
  a log-heartbeat observable so a silent hang is diagnosable post-run; A24
  watchdog wording already covers the class. Watch thread for root cause.
- **ADOPT (upgraded) — boristown `duck-harness-fast-eval` public 1.47 gold,
  144 upvotes.** The top public artifact is 1.47, not 1.39. Defensive
  fork-diff target for our frozen fork upgrades accordingly — potential
  low-risk +0.14 (fork-never-build discipline; byte-matched metadata
  mandatory; fork-diff analysis is $0 and precedes any decision). New Q5.
- **WATCH:** unscored "Deep Reasoning Agent (179/183 levels)" code-tab entry
  updated 1h ago — hype-smelling; becomes ADOPT-diff only if it scores
  >1.47. Re-check tomorrow.
- **IGNORE:** #728210 (beginner), trailing drive-by comments on 07-23
  threads, #697720 grumbles (dropdown-check ops-ADAPT already carried).
- **LB (live top-49):** gold cutoff ≈ **1.49** (top 13); wall band 1.44–1.60
  dense; **we slid #45 → #49 overnight at 1.33** (~4 ranks/day bleed), now
  tied with Jack Cole. Two low-entry high scores (1.60@6, 1.49@5) exceed all
  public artifacts → real methods, not clones; the public 1.47 is itself
  seeding the 1.44–1.47 band. No arc_agi version-bump news.

## §1c Research sweep (learnings/war_room/research_2026-07-24.md)

- **Zero ADOPT, three ADAPT (all wording/design, no build), two PARK.**
- **ADAPT (low) — arXiv 2607.20972** cue-anchored working memory (Jul 23):
  agents made 0 voluntary memory ops in 114 turns; harness-injected
  deterministic cues worked with zero false alarms. Application: the
  depth-budget lane harness re-injects EWM contract + certificate state on
  deterministic cues (level change / revise-loop entry), never
  agent-volitional.
- **ADAPT (low-med) — arXiv 2607.07196** world-model admissibility ladder:
  action-following robustness ranks ABOVE visual fidelity; one line into EWM
  v1.1 certification wording (admissibility-before-verdict).
- **PARK — 2607.08964** Long-Horizon-Terminal-Bench (dense partial credit,
  best model 15.2%): external support for depth ≫ efficiency. **PARK —
  2601.22129** SWE-Replay banked-prefix replay (+3.8% at −17.4% cost):
  cleanest external statement of banking, but blocked by our latent-state
  audit (branching presumes faithful state restoration; state-aliasing
  breaks it).
- **Schema replication watch: still ZERO** (HF traces frozen at 50
  trajectories; no reproduction anywhere). OPINE-World follow-ups: none.
- **A17 serving:** no direct 72B-VL-AWQ-on-96GB datapoint anywhere;
  vllm-blackwell-guide (lastloop-ai) confirms quantized vLLM serves clean on
  RTX PRO 6000 sm_120 at 27B/35B — **toolchain half de-risked; the
  weights+vision-tower+KV fit half is settled empirically by today's
  canary** (as A23 anticipated: envelope is self-certifying).

## §2 Today's plan

1. **A17 canary push (slot 1, ratified; all sign-offs sealed via scope-v2 §9).**
   4-game full-window Qwen2.5-VL-72B-AWQ canary: measures ρ_action
   denominator (27B numerator frozen 480 actions/7920s), serve-config smoke
   (hermes parser, no thinking flags, MM path), envelope check. Build agent
   running (starts from proven eval builder; loud-fail policy if 72B fails to
   serve — a silent 27B fallback would poison ρ_action). §7 preconditions
   walked in order before push; interpretation discipline per C3
   (a17_error_model): the screen is a ≥+4-to-+5-level capability-existence
   detector; NO-GO at modest lift is the designed outcome.
2. **Panel R19 (routine, reduced bench)** on this brief. Then implement
   directives.
3. **Slot 2 contested — panel rules:** W2 sentinel confirmatory null vs
   tr87 fresh-stream depth-lane confirmation (held from the OBJ-H
   ratification: "rides a later quota slot after the panel sees this
   ratification" — the panel is seeing it today).
4. Queue: frozen-fork filler head (verified pending); daemon 18:37/20:07.

## Open questions (for R19)

1. **Sentinel disposition after harm-pause:** shelve on sealed W1 evidence +
   harm-pause (no W2, save the push), or spend a slot on the $0 W2
   confirmatory null to close the calibrated two-seed KILL cleanly? Note the
   arm cannot re-enter LB regardless without a new A21 entry case.
2. **Slot-2 allocation today:** tr87 fresh-stream confirmation (depth lane,
   OBJ-H ratified GO, kill-switch = held-out null10) vs W2 vs hold. The
   depth lane is the only non-A17 depth-targeting line; concentration risk
   says fund it.
3. **Exploration budget pacing:** 1/12 spent on a paused arm. Should the
   next exploration draw wait for the A17 screen outcome (GO → war-v4
   composition is the obvious next arm) or is there a cheaper
   depth-composition candidate first?
4. **A17 canary contingency:** if the canary shows serve-fit failure at
   32768 (envelope NO-GO per A23 self-certification), does the FP8 fallback
   TBD-(i) search start today or wait for the formal AWQ NO-GO?
5. **boristown 1.47 public fork (NEW, from §1b):** the top public artifact
   now scores 1.47 — above our 1.33 best and at the gold cutoff. A
   byte-matched fork of a public 1.47 is the exact move that produced our
   frozen duck (ashvin 0.42 era) and is low-risk under fork-never-build.
   Questions for panel: (i) does adopting it violate any sealed discipline
   (it is a PUBLIC artifact — generalization-first memory says avoid
   luck-chasing, but this is a floor-raise, not a variance lottery)? (ii)
   fork-diff first ($0, today) then queue as filler-replacement? (iii) how
   does it interact with the exploration-budget accounting — is a public
   fork a "filler" (no window cost) or an arm?

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
