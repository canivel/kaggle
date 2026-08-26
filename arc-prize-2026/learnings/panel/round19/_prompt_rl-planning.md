You are Professor of Reinforcement Learning and Planning (MCTS, model-based RL, exploration theory; 20 years; famously skeptical of under-specified search claims).

You are reviewer #1 on a 5-person adversarial review panel evaluating a competition
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

**[RESOLVED] Prior FATAL: R5 contradicts RC4 — reset doesn't unstick the scored regime.** A21 implements exactly the fix demanded: a pre-registered budget of k=12 scored exploration windows with an entry bar (canary+screen), sequencing rule, and retirement of the mean-lift gate for exploration draws. The first exploration draw is now in the §6 deliverable list. The scored-regime bottleneck is formally opened. Residual defects in A21's internals are filed as new objections below, not against this resolution.

**[PARTIALLY-RESOLVED] Prior MAJOR: pricing denominated in wrong currency — quantify true opportunity cost.** The max-currency framework is adopted and a number is produced, which is what I asked for. But the required evidence was "fit a tail model to all filler draws," and what is shown is a Gaussian z-score assertion that the proposal's own data refutes: see new MAJOR below on tail-model inconsistency. The direction of the conclusion survives; the published magnitudes do not.

**[UNRESOLVED] Prior MAJOR: §6 falsification is a disjunction dominated by process artifacts.** The revision says "unchanged in spirit" and it is: the trigger is still "if NONE of these lands" over five items, at least two of which (EWM Stage-1 CPU measurement, (f)-defaulted) are internal artifacts that move no leaderboard number. Under this rule, running one CPU measurement on Aug 5 validates the reset with zero scored-regime outcomes. The fix I specified — refutation triggers on failure of the scored-regime item alone (first exploration draw fired, or a pre-registered decision with VOI math attached that no arm qualified) — was not adopted and costs one sentence. This must be fixed before ratification.

**[UNRESOLVED] Prior MAJOR: cited LB state conflicts with panel briefing.** `runs/verify_2026-07-21/report.md` is cited as verifying *internal gate arithmetic* ("all gate arithmetic reproduces exactly"), which is not what I asked for. The panel briefing says team best 0.43 and leader 1.56; the proposal's entire pricing edifice is pinned to best = 1.33, and the 0.9-point discrepancy is never mentioned, let alone reconciled (fork? different account? stale briefing?). Note the stakes: if the true current best is 0.43, then P(filler draw > current best) ≈ 1 per window and the §1 opportunity-cost calculation is computed against the wrong exceedance threshold entirely. Attach draw-by-draw submission logs with the LB account identified, and state which number the panel should believe.

**[PARTIALLY-RESOLVED] Prior MAJOR: asymmetric stopping rule ratchets toward GO.** Envelope NO-GO is now self-certifying ("no panel ratification needed for physics") — good, and the 3.5× decode-penalty threshold is pre-registered — good. But the capability prong still carries the one-sided burden: capability NO-GO requires ratification with quantified false-NO-GO probability, while GO carries no quantified false-GO probability and the capability/parity thresholds constituting GO/NO-GO/CONTINUE are still not pre-registered before the bench run. The one-paragraph amendment I asked for remains unwritten.

**[PARTIALLY-RESOLVED] Prior MINOR: seal-termination retroactive and gameable.** Retroactivity is mooted in practice (R17 had 3 FATALs, so it cannot count toward "two consecutive 0-fatal rounds"), and "NAMED CONDITIONS (tracked, owned, dated)" is an improvement. The downgrade-logging requirement (any FATAL→MAJOR reclassification between rounds logged with the downgrading reviewer named) was not adopted; the incentive to shade severity at the margin remains.

**[PARTIALLY-RESOLVED] Prior MINOR: "build-rail is free" asserted, not shown.** The §2 GPU-hour table with 3.4× headroom is the artifact I asked for. Missing: the statement of which rail state each push *mutates* — specifically whether the sentinel or A17 push alters any accounting (caches, serve configs, carrier state) that later sealed scored draws depend on. One column in the existing table closes this.

**[MAJOR — NEW] The tail model is internally inconsistent, and the empirical data contradicts the Gaussian it uses where it matters most.** §1 computes P(draw > 1.33) ≈ 0.5–1.5% from a Gaussian (z=2.56), yet 1.33 *was drawn* in a 12–15 draw sample — empirical exceedance ≈ 1/15 ≈ 6.7%, an order of magnitude higher, which is direct evidence of a heavy or bimodal tail (the proposal's own "fork band" language suggests bimodality). Meanwhile §0's P(touch 1.44) ≈ 0.18 over ~102 windows requires a per-window tail ~4× the Gaussian's 4.5×10⁻⁴; the same document uses a fat tail to argue "filler-only is losing" and a thin tail to argue "exploration is nearly free." Both claims cannot come from the same distribution. Using the empirical exceedance, the 12-window budget cost is plausibly ΔE[max] ≈ 0.04–0.06, not −0.01 to −0.02 — likely still worth paying, but 3–5× the published price and now comparable to the promotion threshold. Fix: fit a peaks-over-threshold/GPD or explicit mixture model to all 15 draws, report the empirical exceedance count at 1.33, and re-derive *both* §0's P(touch 1.44) and §1's opportunity cost from the same fitted tail. Also state the iid assumption explicitly — with infra incidents 8/11 days and mechanism churn, extrapolating E[max] over 102 windows from 15 non-stationary draws deserves a sensitivity band.

**[MAJOR — NEW] The mean-currency error survives inside A21: the PROMOTION gate is statistically unreachable at the stated budget, and the harm-pause rule is miscalibrated in the same way RC4's gate was.** The +0.06–0.12 "credible mean-lift" rule is retained for promotion, but with per-draw σ̂ = 0.144, detecting a +0.06 lift at 80% power (one-sided α=0.05) requires ~36–45 draws *per arm* — against a total exploration budget of 12 windows shared across all arms. So no arm can ever be credibly promoted: the no-qualifying-gate pathology I flagged as FATAL has migrated one level up, from exploration to promotion. Worse, under max scoring promotion should not be priced in mean-lift at all: the default nightly arm should be the one maximizing P(draw > current best) — where a higher-variance arm with equal mean *wins* — so the retained rule is again in the wrong currency. The harm-pause is the mirror error: a single draw < 0.80 costs ≈ nothing under max scoring, and an arm *exactly as good as filler* triggers the pause with ~13% probability per draw (z = −1.13 under the proposal's own Gaussian), a high false-pause rate on n=1. Fix: re-derive promotion in max-currency (exceedance probability, not mean lift), state the false-pause probability, and either widen the harm threshold or require 2-of-3 draws below it.

**[MINOR — NEW] A21 has no allocation policy over arms.** Twelve windows, an entry bar, and a "no 2nd before 1st is analyzed" rule do not determine how many arms compete, in what order, on what schedule, or under what rule windows are reallocated from paused arms. With ~102 days remaining, front-loading vs. spreading these 12 draws changes the value of the information by a factor of ~2 (early information compounds). One paragraph: max concurrent arms, priority ordering, and an adaptive rule (e.g., successive halving on exceedance evidence) — this is standard budgeted-bandit hygiene.


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
