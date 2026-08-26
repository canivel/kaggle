You are Professor of Program Synthesis and Neurosymbolic AI (inductive program synthesis, world models as code, verification; insists on falsifiable synthesis-quality metrics).

You are reviewer #3 on a 5-person adversarial review panel evaluating a competition
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

**Prior-round resolution audit (required first):**

1. **[FATAL] Wall-closer unspecified/ungated — RESOLVED (as asked), with residual transferred to new objection N2.** The three demanded items — mechanism class, pre-registered gate with prongs and α, expected per-draw Δ with basis — are all delivered, and the honest answer ("full frontier-style cracking is NOT available at Qwen tier") is exactly the falsifiable answer the question was designed to force. The strategic consequence (the only wall-closer is now v4, itself thin) is a new, distinct objection below, not a carry-forward.

2. **[MAJOR] Banking falsification regime-invalid — RESOLVED (in visible text).** Banking is reclassified UNTESTED-IN-REGIME per A9 (my fix option b), enters last, requires full-panel sign-off, and must pass an A10 trigger-firing bench with the exact counters I asked for (replay_attempted > 0, replay_succeeded = replay_attempted, ≥5 games). Cannot verify the Q1(e) line's disposition in Part 2 (see N1).

3. **[MAJOR] Process grades, not falsifiable predictions — RESOLVED.** §2 is precisely the events-per-run × max-value-per-event exercise I demanded, done with the real scorer (I verified the marginal-value table, the ft09 L1 arithmetic, and the E[max-of-2] banking table independently; all reproduce). The MDE/2 rule is applied as stated: (c) at +0.014 correctly forfeits its standalone window.

4. **[MAJOR] Q1(d) deferred without the free offline metric — PARTIALLY-RESOLVED.** (d) is promoted to the first flagged window with a counting bound — good. But the specific zero-cost measurement I asked for — prediction accuracy (fraction of PREDICT lines contradicted by RESULT) computed offline on the existing N5 deterministic replay traces, with a pre-registered threshold — is still not reported. The gate counts re-probes and PREDICT-scored events; it never measures whether the predictions are better than chance, which is the synthesis-quality signal that would (i) kill (d) cheaply if predictions are noise and (ii) supply the per-game trigger signal the allocation question still needs. Run it before W1 seals; it costs nothing.

5. **[MAJOR] su15 info-theoretic wall claim — RESOLVED.** Verdict downgraded to "suspended (A13)," su15 excluded from component evaluations, re-probe scheduled after (f) ships — exactly the demanded downgrade-and-re-evaluate path.

6. **[MINOR] Ledger −0.128 single-seed — RESOLVED (in visible text).** war-v2-eval is now cited only for trigger-rate = 0 (1552 digests, 0 escalations); the −0.128 figure does not reappear in Part 1.

7. **[MINOR] Q1(g) frozen-list classifier — UNVERIFIABLE.** (g) does not appear anywhere in Part 1. If it was dropped, say so explicitly in Part 2; if it survives, my prior objection stands in full. Note the same frozen-list pathology reappears in banking's retry-target selection (see N4).

**New objections:**

**[MAJOR] N1: I was given Part 1 of 2, and even Part 1 is cut mid-sentence.** The document self-describes as "Part 1 of 2"; the visible text ends "above the build-rail's detection floor only cumulatively, below [TRUNCATED]" — the final verdict sentence and all of Part 2 (which presumably carries the Q1(e)/(g) dispositions, instrument updates, and whatever the sha-covered 30,355 chars contain beyond this) are unreviewable. Per panel rules I file this formally: my RESOLVED grades on objections 2 and 6 are provisional on Part 2 containing no contradicting material, and objection 7 cannot be graded at all. Fix: circulate the complete document before the Jul 21–22 W1 sealed look.

**[MAJOR] N2: war-v4, now the sole registered wall-closer, rests on an unvalidated cross-model extrapolation and an underspecified gate.** The +150-pts / ~+6-per-draw figure is a GPT-5.6 demonstration; "capturing even 10%" of it via Qwen3.6-72B-4bit is asserted with no evidence that any of the NOT-distillable capabilities (recursive abstraction, representation invention, model-fitting over transitions) exist at 72B — the same probe decomposition that killed prompt-tier fixes could equally kill the 72B tier, and 4-bit quantization degrades exactly the long-horizon reasoning at issue. The throughput guard is also internally implausible: with 2.5–3× slower decode, "total actions within 10% of 27B baseline" will essentially never hold under a wall-clock budget, and the else-branch ("Δlc must beat the throughput-adjusted null") is undefined — no formula for the adjusted null is given. Fix, cheap and decisive: before the Aug 1 registration, run 72B-4bit offline on the four probe games (ft09/sb26/lp85/vc33) under the identical harness within the 30 GPU-h/wk allowance, and pre-register a capture-rate go/no-go (e.g., ≥2 levels beyond the 27B baseline across the four games at full budget); simultaneously publish the throughput-adjusted-null formula. If 72B replicates 27B's ~1-level-each profile, the campaign has no wall-closer and the panel needs to know that in July, not September.

**[MAJOR] N3: §3's per-window gate contradicts §2's own power analysis and is structurally biased toward killing true-but-small components.** §2 states single components (+0.01–0.05 rail) sit "far below the ~0.2 Δ-run-mean 3-seed MDE — hence the cumulative-stack gate design in §3," but §3's design is still a per-window, component-ON vs prior-stack contrast with PASS requiring the score prongs at α = 0.0125 — each look is exactly the underpowered single-component test §2 says cannot succeed. Under the stated rule ("FAIL on score prongs with mechanism firing → flag OFF"), a component that fires its trigger and delivers its true +0.02 will be parked as "mechanism fires, doesn't pay" with near-certainty, and the stack whose value was argued to be cumulative gets dismantled window by window. Fix: pre-register the binding decision as the cumulative contrast (final stack, all components ON, vs the W0 baseline — 3 seeds, same prongs), and demote per-window looks to (i) mechanism-prong verification and (ii) the −0.10 non-inferiority guard only; a below-MDE component with a firing trigger and non-inferior score should stay ON pending the cumulative look, not be turned OFF.

**[MINOR] N4: banking's "feasible ≤ +0.15" ceiling embeds a post-hoc frozen retry list.** The 2–3-game retry set (ft09 + sc25 + re86, "the top variance carriers") is selected from the same 3 seeds whose per-game deltas the earlier gate showed to be LOO-fragile — the identical pathology I flagged for Q1(g). Since a live agent cannot know which games are high-variance for this attempt, the retry-selection rule must be pre-registered as an online, game-agnostic policy (e.g., retry the cleared-something games with lowest first-attempt completion fraction per action spent), and the feasible ceiling recomputed under that rule rather than under oracle selection.

**[MINOR] N5: the rail→LB conversion factor (0.56×) is a ratio of null means asserted as a multiplicative transfer law.** Nothing shown establishes that a rail effect scales by the population ratio rather than, e.g., transferring additively or vanishing on the LB game mix; every headline LB number in §2's "Honest sum" inherits this assumption. Label it as an assumption with a one-line sensitivity range (0.4×–0.8×) rather than a derived constant.


=====================================================================

THE PROPOSAL (sha256 of the full document: a4f53dc1133ff1ce; full length 56675 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# R15 Circulation — 2026-07-19

Untruncated single-part delivery via @file (A20 compliant; <=2 parts required, this is 1). Per-part sha256 below; each part ends with an explicit END-OF-PART line. If any part lacks its END line, your copy is truncated — file that as a FATAL objection.


## Manifest
- PART A — Grinder-Cracking Design Doc (R13 FATAL discharge; reviewed by R14): `learnings/war_room/grinder_cracking_design.md` sha256:61b1b4a705692fd7 len:29442
- PART B — Preregistration Amendment 2026-07-18 (A8-A13): `learnings/preregistration_amendment_2026-07-18.md` sha256:4ae0c9e8c23fa1ba len:4186
- PART C — Preregistration Amendment 2026-07-18b (A14-A20; gate recalibration THIS ROUND SEALS): `learnings/preregistration_amendment_2026-07-18b.md` sha256:5aac3957379a5c95 len:9165
- PART D — State of the War synthesis 2026-07-18: `learnings/war_room/state_of_the_war_2026-07-18.md` sha256:8a293d986c28d7f0 len:5305
- PART E — Daily Brief 2026-07-19 (draw #5 A5/A8 FAIL, W0 screen, sweeps, open questions Q1-Q6): `learnings/daily_brief_2026-07-19.md` sha256:d8ff906f71979f3d len:6786



---

# PART A — Grinder-Cracking Design Doc (R13 FATAL discharge; reviewed by R14)
(sha256:61b1b4a705692fd7 len:29442)

# Grinder-cracking design — war-v3 conversion stack + v4 model line

Filed 2026-07-18, before the A6 build deadline (NLT Jul 20). Discharges panel R13's
two [FATAL] items (`learnings/panel/round13/prog-synthesis.md` obj. 1,
`learnings/panel/round13/rl-planning.md` obj. 1): (i) mechanism class, (ii)
pre-registered offline gate with prongs and α, (iii) expected per-draw Δ with basis.
All numbers below are counted from local artifacts; every table cites its source.
Constraint priors applied: simplicity-wins, prompt-is-noise, one-flag-per-window
(A12), game-agnostic legality, zero cloud spend.

Evidence base: `runs/gpt56_probe/experiment_full/` (GPT-5.6-sol, same harness/context
as scored runs), `runs/kernel_pulls/war_eval_v{1,2,3}/` (Qwen3.6-27B, 3 certified
seed replicates), `runs/war_eval_v1/determinism_audit_25.{json,md}` (N5),
`learnings/war_room/gpt56_distill_{grinders,ft09_su15}.md`,
`learnings/war_room/transcript_forensics.md`, `learnings/preregistration_2026-07-14.md`,
`learnings/preregistration_amendment_2026-07-18.md` (A8–A13).

---

## 1. Mechanism class — the honest answer

**R13's sharp question ("what does cracking consist of without a model change, given
the probe decomposition says grinders = model gap?") gets a direct answer: it doesn't.
Full frontier-style cracking is NOT available at Qwen tier.** The distill's
NOT-distillable lists are explicit and evidence-backed
(`gpt56_distill_grinders.md` §"NOT distillable", `gpt56_distill_ft09_su15.md` §A.4):

- one-shot recursive abstraction (sb26 L3–L6 solved in single batches);
- inventing representations (lp85 aliased-cell crossing model — Qwen *stated* "the
  loops share tiles" and never used it);
- model-fitting over recorded transitions (lp85 L4 permutation scoring);
- one-shot correspondence induction from 3 examples (ft09 legend rule);
- terse load-bearing world models; in-head enumeration convergence.

The controlling negative result: the shared system prompt already commands BFS
verbatim; **GPT-5.6 wrote BFS 6× in lp85 alone, Qwen wrote it 0× across six grinder
transcripts.** Prompt exhortation is inert (consistent with
`feedback_prompt_is_noise`). Nothing shipped below is advice.

**The available mechanism class is "conversion-first scaffolding"** — it does not
teach Qwen the missing concepts; it stops Qwen from mechanically destroying the value
of the concepts it does reach:

- **(A) Eliminate the three proven mechanical death modes:**
  1. *Unseen budget deaths* — every Qwen grinder death was a budget death it never
     saw coming (lp85 GAME_OVERs at actions 68 and 131–133 vs the 60-click budget;
     sb26 move-limit GAME_OVER at 140; ft09 2 exhaustion GAME_OVERs; distill pivot L1:
     "budget never identified in any seed"). GPT read the budget on turn 2.
  2. *Verbatim re-submission loops* — sb26 v2 re-submitted literally identical
     arrangements after window eviction (steps 13→20); forensics: same slot coords
     re-clicked 16–32×/seed, SPACE re-tested 8–20×/run, ACTION7 re-probed 8–20×/run.
  3. *Game-over deadlock* — the "stop acting immediately" prompt language cost
     GPT-5.6 56 of 60 minutes on su15 (correct restart computed every turn, guarded
     behind an unreachable `elif`). Our duck prompt carries similar language; Qwen
     currently survives by disobedience. This is a latent harness bug, fixed in (f).
- **(B) Mechanize the one cross-cutting discipline:** the per-action diff summarizer.
  Per-action diff attribution is the substrate of every GPT win (pivots S1, L2) and
  the thing Qwen provably cannot sustain ("I'm not tracking the positions correctly";
  lp85 re-derives the layout from ascii every turn, 8 consecutive analysis-only
  turns). Scripted, information-only, capped block.
- **(C) Bank solved levels across attempts.** N5 falsified R12's randomization
  premise: **0/25 games divergent, all 25 frame-deterministic, recorded lc reproduced
  on second plays; all 8 Δlc-positive games bankable**
  (`determinism_audit_25.md`). The sc25/m0r0 aborts were a `prune_trace` bug
  (leading `board_changed=False` actions that mutate hidden state are dropped →
  replay desyncs at step 0; `runs/war_eval_v1/prune_replay_diag.json`). Fix =
  unpruned or trailing-only-pruned replay. **A9 caveat:** banking belongs to the
  warpack line, which A9 parked as UNTESTED-IN-REGIME with a double-lock reopen
  rule. Banking-fixed therefore enters this stack LAST, only with full-panel
  sign-off, and only through an A10-compliant trigger-firing bench (§3).

**Model swap — named explicitly as the ceiling-raiser, and explicitly NOT part of
v3.0.** The probe decomposition says the grinder gap is model tier: GPT-5.6 under
the *identical* harness/context/tools scores ft09 5/6, sb26 5/8, lp85 4/8, vc33 3/7
in ≤100 actions where Qwen gets ~1 level each. In rail score units the frontier
demonstration on those four games alone is worth on the order of +150 game points
(~+6 per-draw; different game-version baselines, so order-of-magnitude only) —
capturing even 10% of it exceeds the entire v3 scaffolding ceiling computed in §2.
The only Kaggle-legal implementation is a stronger vLLM-servable open model:
primary candidate **Qwen3.6-72B-tier at 4-bit (AWQ/W4A16)** — weights ~40 GB, fits
the RTX PRO 6000 (96 GB) build rail with KV headroom, evaluated inside the free
30 GPU-h/wk (zero cloud spend per standing constraint); secondary candidate 27B at
higher precision (cheap, small expected delta). Cost/risk: 72B decode throughput is
~2.5–3× lower → fewer actions per scored budget; the net can be NEGATIVE if action
throughput binds, so the line carries a throughput guard in its gate. **This is a
separately registered, separately gated line (war-v4-model, §5). It is the wall
closer candidate; v3.0 is not.**

---

## 2. Counting-bound Δ estimates per component (R13's required arithmetic)

Method: events-per-run counted from the actual transcripts × maximum value-per-event
under the exact pooled-single-run scorer
(`scripts/phase1_gate.py::rhae_score`, validated 0e+00 vs Tufa's 500 runs):
level score = min(115, (base/actions)² · 100) if completed else 0, with level k
(1-based) carrying weight k, so level k of an n-level game contributes
`k/Σ(1..n) · min(115,(b/a)²·100)` game points; per-draw = game points / 25.

**The governing fact of this scorer: reclaimed actions on an UNCOMPLETED level are
worth exactly zero.** Uncompleted levels score 0 regardless of actions spent, so
"saving X actions" converts to score only via (i) a level that then clears inside
budget, or (ii) a cleared level cleared faster. Every bound below applies this.

Reference marginal values (computed with the real scorer, war_eval baselines):

| clear event | at base acts | at 2× base | at 4× base | at 8× base |
|---|---|---|---|---|
| sb26 L2 (w 2/36, base 28) | +5.56 pts (+0.222/draw) | +1.39 (+0.056) | +0.35 (+0.014) | +0.09 (+0.004) |
| lp85 L2 (w 2/36, base 38) | +5.56 (+0.222) | +1.39 (+0.056) | +0.35 (+0.014) | +0.09 (+0.004) |
| ft09 L2 (w 2/21, base 12) | +9.52 (+0.381) | +2.38 (+0.095) | +0.60 (+0.024) | +0.15 (+0.006) |

Qwen clears grinder levels at 4–8× base when it clears at all (v2 sb26 L2 attempt
died at 236 acts vs base 28), so the 4×–8× columns are the operative ones.

### (f) Game-over-continuation fix — counting bound ≈ 0.00, ships anyway

Events in Qwen transcripts: full deadlocks observed = **0** (Qwen plays through
game-overs naturally — it did so twice in ft09 v1). The 56-minute deadlock is a
GPT-only observation; for Qwen this is latent-risk removal plus the A13 su15
re-probe prerequisite. **Expected Δ: 0.00 measurable.** Ships FIRST as the A12
standalone hygiene window precisely because it claims nothing.

### (a) Budget sentinel — ceiling +0.06/draw (rail), expectation +0.01–0.03

Events: unseen budget deaths ≈ 1–2 per run on each of {lp85, sb26, ft09, tu93}
(lp85 GAME_OVERs at 68, 131–133; sb26 at 140; ft09 ×2; tu93 v3 burned 301 acts on
L1 vs base 19). Value per event: a budget death wastes the level attempt in
progress. But the transcripts show Qwen was **never on a correct plan when it
died** — lp85 was executing rotations that provably return to start; sb26 was
inside the wrong arrangement family. The sentinel cannot supply the missing
concept, so the countable value is: at most one otherwise-just-out-of-budget clear
per run panel-wide, at Qwen's demonstrated ~4× base efficiency → ~+1.4 game pts →
**ceiling +0.06/draw; honest expectation +0.01–0.03** (Δlc ceiling +0.04, i.e. 1
level/25 games).

### (c) Submission-fingerprint refutation — ceiling +0.02/draw direct; enabling value uncounted

Events: verbatim re-submits — sb26 v2: ~10 arrangements over ~240 actions, ≥2
byte-identical; forensics adds 16–32 same-coord re-clicks/seed. Reclaimable: ~50–70
of sb26's 240 wasted actions, similar on 1–2 other games. Conversion: reclaimed
actions pay only if a NEW hypothesis family is generated, and the CONCEPT lock
(never leaves family #1 in 13 forensic runs) is exactly what Qwen doesn't do. Even
granting an sb26 L2 clear at 4× base: +0.35 pts → **direct ceiling +0.014/draw ≈
+0.02**. This is **below MDE/2 on every instrument** — per R13's own rule it must
NOT consume a window alone. It ships inside the same flag as (d) (one subsystem:
mechanical refutation records; §3), where its real value is giving the escalation
machinery a prose-free trigger (war-v2-eval: 1552 digests, **0 escalations** —
the trigger never fired on prose extraction).

### (d) PREDICT→RESULT wiring + no-effect FACTs — ceiling +0.08/draw, expectation +0.02–0.04

Events: dead-target re-probes — ACTION7 8–20×/run, SPACE 8–20×/run, ft09
source-grid/corner clicks 3× each, su15 ACTION7 "submit" presses ~12; ft09 v1
burned 134 actions on L1 (~30 env-tested hypotheses, each refutable offline; 2
GAME_OVERs) vs base 43. Countable channel that actually pays: **faster first-clear
on levels Qwen already clears** (efficiency on completed levels is the one place
reclaimed actions score). Exact scorer arithmetic on ft09 L1 (base 43): 134 acts →
0.49 pts; 94 acts → 1.00 (+0.02/draw); 60 acts → 2.45 (**+0.078/draw = the
ceiling**); plus reliability value: ft09 v3 cleared 0 levels — mechanical
refutation records raise the floor toward the v1/v2 outcome. **Ceiling +0.08/draw;
honest expectation +0.02–0.04.** The distill's own grade stands: this converts a
~30-guess grind into a ~30-guess grind *without repeats and without budget
suicide* — it does not produce the rule induction that clears ft09 L2+.

### (b) Probe-diff summarizer — ceiling ≈ +0.06/draw, expectation +0.01–0.03, token-cost risk

Events: every action (Qwen loses object tracking within levels; lp85 seed1 spent
75 min in 8 consecutive analysis-only turns re-deriving the same layout). Value:
reclaims wall-clock/turns rather than actions; converts to score only through the
same clear-faster/clear-at-all channels bounded above, so its independent ceiling
overlaps (a)/(d): ~+0.06/draw, expectation +0.01–0.03. Unlike (a)/(c)/(d) it adds
~120 tokens to EVERY action result — the only stack member with a per-turn cost —
so it gates strictly after the refutation stack, non-inferiority guarded.

### Banking-fixed (N5 prune fix) — ceiling +0.29/draw (budget-infeasible), feasible ≤ ~+0.15, expectation +0.03–0.08

Basis: all 25 games deterministic; recorded lc reproduces on a second play; card
score = MAX over plays → a banked clear makes further attempts downside-free. The
harvestable quantity is **across-attempt sampling variance**, measured directly
from the 3 certified Qwen seeds on the 8 bankable games (E[max of 2 attempts] −
E[single attempt], exact scorer):

| game | seed scores (v1/v2/v3) | seed lc | Δscore(max2) | Δlc(max2) |
|---|---|---|---|---|
| ft09 | 0.49 / 14.29 / 0.00 | 1/2/0 | +3.17 | +0.44 |
| ka59 | 0.47 / 3.57 / 3.57 | 1/1/1 | +0.69 | +0.00 |
| re86 | 0.46 / 6.09 / 1.53 | 1/2/1 | +1.25 | +0.22 |
| sc25 | 2.14 / 3.50 / 0.00 | 2/2/0 | +0.78 | +0.44 |
| tu93 | 0.96 / 2.99 / 0.01 | 2/2/1 | +0.66 | +0.22 |
| sb26 | 2.78 / 2.25 / 2.78 | 1/1/1 | +0.12 | +0.00 |
| su15 | 2.22 / 0.37 / 0.00 | 1/1/0 | +0.49 | +0.22 |
| lp85 | 2.78 / 2.78 / 2.78 | 1/1/1 | +0.00 | +0.00 |
| **TOTAL** | | | **+7.16 pts → +0.286/draw** | **+1.54 lc → Δlc +0.062/draw** |

That +0.29 assumes a full second attempt on all 8 games inside one scored budget —
infeasible (a retry costs 45–300 actions/game). A realistic 2–3-game retry budget
(ft09 + sc25 + re86, the top variance carriers) keeps ≤ ~+0.15 ceiling;
**expectation +0.03–0.08.** Governance: this is warpack-line machinery; per A9 it
is parked and does not reopen on any n=5 statistic. It enters only as the LAST
window, with full-panel sign-off, through an A10 bench that demonstrates the replay
trigger firing ≥1/run on ≥5 games (the audit's caveat also applies: 15/25 local
engine versions differ from the Kaggle build; behavioral parity is suggested by lc
reproduction, not proven).

### Honest sum

Units: build-rail per-draw score points. LB conversion: the rails differ in
population (rail null run-mean 1.636 vs LB control 0.922, prereg §7) → LB effect ≈
0.56× rail effect. Wall gap: 1.44 − 0.980 (war ledger n=4 mean) = **0.46 LB units**.

| component | ceiling (rail) | expectation (rail) |
|---|---|---|
| (f) continuation | 0.00 | 0.00 |
| (d)+(c) mechanical refutation | +0.10 | +0.02–0.05 |
| (a) budget sentinel | +0.06 | +0.01–0.03 |
| (b) diff summarizer | +0.06 | +0.01–0.03 |
| banking-fixed (feasible) | +0.15 | +0.03–0.08 |
| **stack** | **≈ +0.31 rail ≈ +0.17 LB** | **≈ +0.07–0.19 rail ≈ +0.04–0.10 LB** |

(Ceilings are not strictly additive — (a)/(b)/(d) partially reclaim the same wasted
actions — so +0.31 is itself generous.)

**Against the 0.14 LB MDE:** the stack's summed *ceiling* (+0.17 LB) exceeds it
marginally; the honest *expectation* (+0.04–0.10 LB) does not — no LB draw count
this campaign can run will ever confirm the stack (consistent with A8's
accumulation-only status). The build rail is the only instrument, and even there
single components (+0.01–0.05) sit far below the ~0.2 Δ-run-mean 3-seed MDE — hence
the cumulative-stack gate design in §3.

**Against the 0.46 wall gap:** ceiling closes ~37%, expectation closes ~10–20%.
**The conclusion R13 pre-authorized: the conversion stack is a floor/mid raiser at
≈ +0.1–0.2 rail (+0.05–0.10 LB) per-draw mean; the wall needs more. The
model-swap line (war-v4) is therefore the only registered wall-closer and must be
scoped as v4 with its own gate (§5) — it is not smuggled into v3.0.**

---

## 3. Pre-registered offline gate (per A10/A12; seals on filing of this doc)

**Bench (budget-faithful, A10):** compressed budgets — per-game action caps scaled
to ~40% of the Qwen observed median per-game use so that each window's trigger
fires ≥1/run on ≥5 games. **Canary run verifies trigger counts BEFORE the gate
seals** (the ledger-canary precedent). A component whose trigger cannot be made to
fire on the rail gets LB-accumulation status only — never a build-rail kill (the A9
lesson, conceded in R13).

**Design per window:** 3 certified seed-only-diff replicates of the cumulative
stack (component-ON) vs the prior stack (component-OFF), 17/17-cell sha
certification as in war_eval; sealed look after seed 3; no interim looks.

**Prongs (compound rule, unchanged from A1):**
1. **Primary:** pooled per-game Δlevels_completed (paired, 3 seeds pooled), exact
   sign-flip test, **α = 0.0125**, one-sided. Tie convention (R13 minor, fixed
   now): exact zeros are dropped before the flip; W/L counts reported alongside.
2. **Secondary:** mean Δlog1p(RHAE) across seeds ≥ 0.
3. **Mechanism prong (necessary, not sufficient):** the window's trigger counter
   ≥1/run on ≥5 games PLUS the component observable —
   (f): 0 post-game-over idle turns; (d)+(c): dead-target re-probes ≤2/run
   (P4 threshold) and verbatim-resubmit count = 0; (a): unseen budget deaths
   halved vs the 3 control seeds; (b): re-derivation paragraph recurrence −70%
   (P3 machinery); banking: replay_attempted>0, replay_succeeded=replay_attempted,
   0 frame-divergence aborts on ≥5 games.
- PASS = all three prongs → flag stays in the cumulative stack.
- FAIL on score prongs with mechanism prong firing → flag OFF, line parked with an
  honest "mechanism fires, doesn't pay" label (no regime excuse available: A10
  guarantees the trigger fired).
- Non-inferiority guard: pooled Δlc ≤ −0.10 at any look → flag OFF immediately.

**Ordering rules:** (f) hygiene first, standalone, unflagged (A12). **su15 is
excluded from every post-(f) component evaluation** (A12's pre-registered
exclusion; A13's re-probe is a separate frontier-tier experiment). One flag per
window, no exceptions without full-panel sign-off. Single declared exception,
registered here: **(c) and (d) constitute ONE flag** — "mechanical refutation
records" is a single code path (harness diff engine writing FACT/RESULT records);
attribution inside a pass is pre-registered via the separate counters
(fingerprint-block count vs no-effect-FACT count vs PREDICT-scored count), so a
pass decomposes mechanically, not by anecdote. (b) is NOT part of that flag (it
adds per-turn context cost; separate window). Banking-fixed requires full-panel
sign-off before its window opens (A9 adjacency, stated in §2).

---

## 4. Per-game conversion targets (the 8 Δlc-positive / bankable games)

Failure classes from `transcript_forensics.md` (CONCEPT/MEMORY/PERCEPTION) +
war_eval transcripts; expected clears delta = integer levels per run, honest.

| game (levels) | what kills Qwen there (evidence) | component | expected Δclears |
|---|---|---|---|
| ft09 (6) | CONCEPT: ARC-1 transformation prior; ~30 hypotheses env-tested at 1–8 acts each, 2 GAME_OVERs, L1 won by luck at 134 acts; v3 cleared 0 | (d)+(c) no-effect FACTs kill dead-target re-probes; PREDICT gate kills retries | **0–1** (L1/L2 reliability + speed; L2+ rule induction = model gap) |
| ka59 (7) | stuck-game grind: L2 (base 109) consumed 40–168 acts, never converts; only game positive in all 3 warpack seeds (recovery anecdote) | banking-fixed (protect L1, free retries); (a) | **0–1** |
| re86 (8) | long-level budget grind: v1 died on L2 at 232 acts (base 42); v2 cleared L2+L3 — high seed variance | banking-fixed (harvest variance); (a) | **0–1** |
| sc25 (6) | fragility: 2 levels with warpack recovery in v1/v2, 0 in v3 (102 acts on L1); prune-bug abort game | banking-fixed (floor protection) | **0–2** (variance harvest, not new capability) |
| tu93 (9) | budget indiscipline: v3 spent 301 acts on L1 (base 19); v1/v2 clear 2 | (a) sentinel; (d) | **0–1** |
| sb26 (8) | CONCEPT lock (one arrangement family, ~240 acts, 10 arrangements) + MEMORY (verbatim resubmits, restart amnesia) | (c) hard-blocks resubmits, feeds N=3 escalation | **0** honest (L2 connector semantics = NOT-distillable, pivots S2/S4) |
| lp85 (8) | budget deaths (68, 131–133) + PERCEPTION re-parsing + MEMORY; the L2 crossing mechanism is NOT-distillable | (a) sentinel; (b) diff summarizer | **0** honest (survival ≠ solution) |
| su15 (9) | wall verdict suspended (A13); mechanics near-unobservable in budget | (f) only; **excluded from all evaluations per A12** | **0** |

Honest panel-wide sum: **1–4 extra clears per run best case, mostly variance
harvest** → Δlc/draw ≈ +0.04–0.16, consistent with §2's score arithmetic. The two
canonical "grinders" (sb26, lp85) — the games the phrase "grinder cracking" was
coined for — carry expected Δclears of **zero** at Qwen tier. That is the model-gap
finding, stated without cosmetics.

---

## 5. Timeline ((f) first, then counting-bound order; build NLT Jul 20 held)

| date | window | content |
|---|---|---|
| **Jul 18** | — | this doc filed (gate seals on filing); (f) built + runtime-tested (`feedback_test_before_submit`), pushed as standalone unflagged hygiene |
| **Jul 19** | W0 (f) | (f) quick screen (0 idle-turn observable); build (d)+(c) mechanical-refutation flag; A10 canary for its triggers; seed 1 push |
| **Jul 20** | W1 (d)+(c) | **build work complete — A6 deadline met**; seeds 2–3 pushed |
| Jul 21–22 | W1 | sealed 3-seed gate look (d)+(c) |
| Jul 22–24 | W2 (a) | compressed-budget canary (budget triggers ≥1/run on ≥5 games) → 3 seeds → sealed look |
| Jul 25–27 | W3 (b) | diff summarizer window, non-inferiority guarded (token cost) |
| Jul 28–30 | W4 banking-fixed | ONLY with full-panel sign-off (A9 adjacency): trailing-only-prune replay + soft-time fix, A10 trigger bench, 3 seeds, sealed look |
| Aug 1 | v4 scoping | war-v4-model registration: 72B-tier 4-bit throughput bench on the 30 GPU-h/wk rail; gate = compound rule at budget-faithful FULL budgets + throughput guard (total actions within 10% of 27B baseline, else Δlc must beat the throughput-adjusted null); enters build only by its own gate, never by fiat |

Push budget: 2/day; 3-seed windows therefore span ~2 days + look. Any window whose
canary fails to fire triggers forfeits its slot to the next window rather than
running an unpowered gate (A10).

---

## Final statement (the three sentences R13 asked for, plus the verdict)

**Mechanism class (2 sentences):** Grinder cracking without a model change does not
exist — the probe decomposition and the NOT-distillable lists show the sb26/lp85
concept gap is model tier, and prompt exhortation is proven inert; what is
available at Qwen tier is conversion-first scaffolding: mechanical elimination of
the three proven death modes (unseen budget deaths, verbatim re-submission,
game-over deadlock), a scripted per-action diff substrate, and deterministic
level banking. The model swap (72B-tier open model under vLLM on the free GPU
rail) is the only true ceiling-raiser and is registered as a separately gated v4
line, not part of v3.0.

**Summed counting bound:** stack ceiling ≈ **+0.31 rail ≈ +0.17 LB** per-draw
(non-additive, generous); honest expectation **+0.07–0.19 rail ≈ +0.04–0.10 LB** —
above the build-rail's detection floor only cumulatively, below the 0.14 LB MDE in
expectation, closing **~10–20% (expectation) to ~37% (ceiling) of the 0.46 gap to
the 1.44 wall**.

**Verdict on the FATAL:** this document supplies all three demanded items —
mechanism class (§1), pre-registered gate with prongs and α (§3), per-draw Δ with
counted basis (§2, §4) — and the Jul 20 build deadline holds for the v3.0 stack
(W1 build completes Jul 20). **The FATAL is discharged, but its premise is
corrected rather than satisfied: A6's label of grinder cracking as "wall-closer"
is retracted — the v3 stack is a floor/mid raiser, and the panel should re-scope
A6 so that the wall-closer designation transfers to the gated war-v4 model line.
If the panel reads that re-scoping as a slip of A6's substance, R13 explicitly
pre-authorized that conclusion; the build date itself does not slip.**

---

## Addendum (2026-07-18, filed for R15): EWM-execute event-shaped canary — schema, aggregator, Stage-0 dry-run

*Appended after the Final statement; no prior section altered. Corrected adopt #1
from the Kimi-3 review cycle. Supports the Stage-1 EWM-execute line
(`opine_world_deepread.md` §4, `state_of_the_war_2026-07-18.md` priority 1).*

### A. Why the canary must be event-shaped

`scripts/predict_metric.py` (R14 artifact) killed component (d) with
totals-shaped counters — sufficient for a pooled-accuracy kill, structurally
insufficient for the EWM Stage-0 mechanism prong ("plans executed >=1/run on
>=5 games, mismatch-aborts logged, 0 post-abort deadlocks"). Totals cannot
express per-game plan-step rates, post-abort survival vs deadlock (an ordering
property), or plan-length/abort-step distributions (where in a plan divergence
lands decides whether a sim is shippable). The fix is one greppable stdout line
per contract event:

```
EWMEVT v=1 kind=plan_step game=ls20 plan=0 step=3 act=A6:31,22 pred=a1b2c3d4 obs=a1b2c3d4 match=1 lvl=1 t=413.0
```

Schema sealed in `duck_eval/ewm_exec/EVENT_SCHEMA.md` (5 kinds: plan_start /
plan_step / mismatch_abort / plan_done / fallback, + trunc volume guard;
aggregator semantics sealed there too so the gate cannot be argued post hoc).
Aggregator: `scripts/ewm_events.py` — per-game table, abort-step distribution,
survival window (N=25 events), deadlock detection, A10 canary verdict line, and
the GSME activation prong-0 verdict (arXiv:2607.13683 — the anti-0/1552-ledger
check). **Selftests: 17/17 pass** (clean plan, mid-plan abort w/ recovery,
abort-then-deadlock, mid-line-truncated log, zero-event run, 5-game canary
PASS with prefix-stamped lines, planned-but-never-executed does not fire).

**Log volume vs the 10 MB Docker cap** (discussions_2026-07-18 infra
constants): measured on the dry-run below, 97 B/line avg, max line 111 chars;
2,840-4,065 lines per 25-game run = **0.28-0.39 MB/run measured**. Normative
hard caps in the schema (2,000 lines/game, 25,000/run, then `trunc` +
1-in-10 step sampling; abort/done/start/fallback never sampled) bound the
adversarial worst case at **5.0 MB**, leaving half the cap for the rest of the
harness stdout.

### B. Stage-0 gate dry-run on real traces (BEFORE any build)

The executor does not exist yet, so `scripts/ewm_replay_dryrun.py` replays the
recorded action streams (war_eval_v1/v2/v3 Kaggle pulls + gpt56 probe
experiment_full) through the 12 saturated exec_wm sims: recorded batches stand
in for plans, each recorded action is simulated from the recorded pre-action
settled frame (teacher-forced), predicted-vs-settled hash compared, first
mismatch per plan emits abort+fallback exactly as the contract would. Output:
`runs/ewm_dryrun/{*.log,report.md,raw.json}`. Honest scope: this measures sim
fidelity on real Kaggle trajectories and that the gate is expressible/computable
from the stream — NOT BFS plan quality (plans are the recorded agent's, not
sim-derived).

**Verdicts (all 4 sources):** canary PASS — 12/12 games fired on each war_eval
seed, 5/5 on gpt56; activation ACTIVE; **0 deadlocks, 0 malformed lines,
0 selfdiff** (double-run: two independent sim instances agreed on all 11,747
lines — trajectory-level determinism holds even for the module-state sims
tr87/sp80), 0 sim crashes; done-flag agreement 98-100%.

**Per-game on-trace step accuracy (shadow = all steps verified), vs held-out:**

| game | held-out sat% | v1 / v2 / v3 / gpt56 | aborts landing at step 0 (pooled war) |
|---|---:|---|---:|
| ft09 | 100.0 | 0.98 / 0.56 / 1.00 / **0.07** | 5/7 |
| lf52 | 100.0 | 0.30 / 0.50 / 0.75 / - | 139/139 |
| lp85 | 100.0 | 0.11 / 0.46 / 0.09 / 0.07 | 197/198 |
| ls20 | 100.0 | 0.64 / 0.92 / 0.80 / - | 38/43 |
| s5i5 | 99.5 | 0.26 / 0.30 / 0.13 / - | 99/99 |
| sb26 | 100.0 | 0.16 / 0.11 / 0.16 / 0.03 | 242/244 |
| sp80 | 100.0 | 0.03 / **0.88** / 0.07 / - | 154/157 |
| su15 | 99.5 | 0.31 / 0.15 / 0.81 / 0.17 | 144/145 |
| tn36 | 100.0 | 0.53 / 1.00 / 0.98 / - | 11/11 |
| tr87 | 100.0 | 0.82 / 0.77 / 0.82 / - | 100/105 |
| tu93 | 100.0 | 0.73 / 0.78 / 1.00 / - | 30/42 |
| vc33 | 99.5 | 0.24 / 0.67 / 0.37 / 0.31 | 172/172 |

### C. Findings (material for the Stage-1 gate and its Δ estimate)

1. **Held-out saturation does NOT transfer to on-trajectory prediction.**
   Sims validated at 99.5-100% state-exact on local held-out tuples score
   0.03-1.00 on real Kaggle streams, varying wildly across seeds of the SAME
   game (sp80: 0.03/0.88/0.07). Drivers visible in the event stream: aborts
   land overwhelmingly at step 0 (first verification) with small median
   diff-cell counts (1-56 cells — timer rows / hidden-counter phase
   misalignment, engine-version drift), plus depth-transfer failure (gpt56
   ft09 = 0.07 because GPT reaches level-2+ states the sim never saw; war_eval
   ft09 = 0.98-1.00 on level-1 states). This is precisely the information a
   totals counter destroys — and it is a **material discount on the Stage-1
   counting bound**: the +0.10-0.30 rail expectation assumed the 12 sims stay
   saturated in-regime; the dry-run says the reliable carriers at Qwen-trace
   regime are **tn36, tr87, tu93, ls20, ft09(L1)** (0.77-1.00 on >=2 seeds),
   while vc33/s5i5 (two of the five named Stage-1 target games) sit at
   0.13-0.67 and would abort at step 0 on most plans.
2. **The fail-closed contract prices this drift at ~1 wasted action per
   aborted plan** (abort-at-step-0 dominance), which is exactly the safety
   property claimed in the OPINE deep-read legality note — now demonstrated
   with real numbers instead of asserted. A per-game live-fidelity breaker
   (stop planning after k step-0 aborts; emit `fallback reason=budget`) should
   be part of the executor spec.
3. **The Stage-0 gate is fully computable from the stream** on day one: canary
   and activation verdicts, survival, deadlocks all fell out of recorded data
   with 0 malformed lines — no new instrumentation needed at build time
   beyond printing EWMEVT lines.
4. **Double-run rejection is free and passed everywhere** (0 selfdiff /
   11,747 lines), so adopting OPINE refinement (ii) costs nothing and stays.
5. Caveat for R15: dry-run "plans" are the recorded agent's action batches
   (mostly length 1-3 on click games), not sim-derived BFS paths; step
   accuracy conditions on the recorded agent's state distribution. A
   sim-guided executor visits different states — the numbers above are the
   best available prior, not a measurement of the executor.

Artifacts: `duck_eval/ewm_exec/EVENT_SCHEMA.md`, `scripts/ewm_events.py`
(`--selftest`), `scripts/ewm_replay_dryrun.py`, `runs/ewm_dryrun/`.


## END OF PART A ##


---

# PART B — Preregistration Amendment 2026-07-18 (A8-A13)
(sha256:4ae0c9e8c23fa1ba len:4186)

# Pre-registration amendment — 2026-07-18 (panel R13 seals, filed BEFORE draw #5 observation)

Responds to panel round 13 (5× MAJOR-REVISION, 2 FATAL; `learnings/panel/round13/`).
Filed 2026-07-18 ~08:00 EDT. Draw #5 fires 20:07 EDT tonight and scores ~04:00
EDT Jul 19; nothing in this document may be revised after that observation.

## A8 — A5 variance-gate fail consequence (R13 methodology major, sealed pre-observation)

Panel arithmetic (verified): with ledger {0.91, 1.08, 0.88, 1.05}, A5
(χ²-CI-hi(σ) < 0.25 at df=4) passes only if draw #5 ∈ [0.955, 1.005]. The 0.25
threshold as written is therefore near-unpassable — a calibration error in A5,
conceded. Sealed consequences:

- **If A5 FAILS at n=5 (expected):** LB windows for the war-v1 arm remain
  **accumulation-only** (which they already are under A4) and additionally
  lose eligibility as an A/B *readout* arm at any future n. No mechanism
  line may cite war-arm LB deltas as evidence, positive or negative, until a
  re-registered gate with a recalibrated threshold (set from the frozen-fork
  control's own CI-hi at equal n, not an absolute constant) passes.
- **If A5 PASSES:** no new licenses are granted beyond what A4 already
  permits; a pass at a miscalibrated threshold is not evidence of low σ.
- The A5 threshold for future arms is re-based: CI-hi(arm) < 1.5 ×
  CI-hi(frozen control at same n), df-matched — relative, not absolute.

## A9 — Warpack reopening rule (R13 regime-transfer major, sealed pre-observation)

The A1 gate closed the warpack build-rail line on an offline bench that
suppresses budget-pressure firing conditions (R13: "refuted a composition,
not components" — conceded, and consistent with the banking trigger counter
never being observed live). Sealed now, before draw #5:

- The warpack line is reclassified from REFUTED to **UNTESTED-IN-REGIME,
  parked**. It does NOT reopen on any LB statistic at n=5. Reopening requires
  BOTH: (i) war-arm LB ledger at n≥8 with one-sided Welch p < 0.05 vs the
  frozen control ledger at equal-or-greater n, AND (ii) a budget-faithful
  build-rail bench (A10) demonstrating the banking/recovery trigger counters
  fire at ≥1 event/run on ≥5 games. Neither alone reopens. No other
  statistic, draw, or eyeballing reopens it.
- war-arm draws beyond tonight's #5 are NOT scheduled; the frozen fork
  resumes as default filler (its 1.33 demonstrates the order-stats floor;
  war accumulation past n=5 has no sealed purpose).

## A10 — Budget-faithful bench (R13 rl-planning/systems major)

Before any budget-regime mechanism (budget sentinel (a), per-game
re-allocation (g), banking soft-time (e)) enters a sealed gate: the build
rail must run a **compressed-budget bench** — per-game action/wall-clock
budgets scaled so the mechanism's trigger counter fires ≥1 time/run on ≥5
games (verified by canary before the gate seals, as the ledger canary did).
Mechanisms whose triggers cannot be made to fire on the rail get LB-ledger
accumulation status only, never a build-rail kill.

## A11 — Ledger conclusion relabeled (R13 methodology major, conceded)

"REFUTED" is withdrawn. Sealed label: **"trigger never fires as built
(mechanistic, certain: 0/1552); effect size unmeasured (n=1 screen,
p=0.86)."** The −0.128/−0.314 point estimates may not be cited in ranking
or retirement arguments. Ledger-as-built still does not enter scored
windows (no benefit channel); its firing-trigger upgrades compete in Q1 on
their own counting bounds.

## A12 — Unbundling (R13 unanimous major, conceded)

The (a)+(f) single-window lean is withdrawn. (f) game-over-continuation
ships FIRST as a standalone hygiene window with its own quick screen and a
pre-registered su15 exclusion from any later (a) evaluation. One flag per
window, no exceptions without full-panel sign-off.

## A13 — su15 wall verdict suspended (R13 prog-synthesis/llm-agents major, conceded)

"Accept-the-loss" is suspended pending one disambiguating experiment: after
the (f) fix lands in the local rig, re-probe su15 once with GPT-5.6-sol
(covered by the existing API credit; single game, 60-min/100-action caps,
$10 spend ceiling). Wall verdict re-affirmed or retracted on that evidence.


## END OF PART B ##


---

# PART C — Preregistration Amendment 2026-07-18b (A14-A20; gate recalibration THIS ROUND SEALS)
(sha256:5aac3957379a5c95 len:9165)

# Pre-registration amendment — 2026-07-18b (post-R14)

Filed 2026-07-18 ~13:30 EDT, interactive daily-loop session, in response to panel
round 14 (`learnings/panel/round14/`: 5× MAJOR-REVISION, 0 accepts; new fatal-class
objection raised independently by methodology, llm-agents, and rl-planning).
Supersedes §3 of `learnings/war_room/grinder_cracking_design.md` (the "seals on
filing" clause). LB draw #5 (war-v1 final accumulation) has NOT been observed at
filing time (submits ~00:07Z Jul 19); nothing in this amendment conditions on it.

## A14 — §3 gate seal VOID; recalibrated gate design (discharges the R14 FATAL)

**Void declaration.** The §3 gate never validly sealed: (i) the panel reviewed a
truncated circulation ("Part 1 of 2", cut mid-sentence — argv budget defect in
`scripts/panel_round.py`), and a gate cannot seal on a document the panel has not
seen in full; (ii) the sealed primary prong was arithmetically unpassable — an
exact one-sided sign-flip at α = 0.0125 requires ≥7 uncontradicted nonzero wins
(2⁻⁷ ≈ 0.0078), while the doc's own §4 expectation is 1–4 nonzero improvements per
window. As written, the FAIL rule ("flag OFF") would deterministically park every
component the doc itself expects to help (methodology FATAL; llm-agents N1;
rl-planning independently). Both defects concede the panel's point in full.

**Recalibrated design (seals on R15 full-document circulation, before the first
flagged-window look):**

1. **Pooling unit (defined, per methodology Q2):** the paired unit is the GAME —
   per-game mean Δlevels_completed across the 3 certified seeds (n = 24 game-level
   pairs; su15 excluded per A12; exact zeros dropped before the flip; W/L reported).
   Not 75 game×seed pairs — cross-seed consistency is not demanded of variance-
   harvest components.
2. **Binding score decision = ONE cumulative sealed look:** final v3 stack (all
   mechanism-retained flags ON) vs the W0 baseline ((f)-only), 3 certified
   seed-only-diff replicates, compound prongs unchanged in form (pooled Δlc
   sign-flip α = 0.0125 one-sided; mean Δlog1p(RHAE) across seeds ≥ 0). The summed
   stack expectation (+0.07–0.19 rail) is the alternative this look is powered
   against; single components are not score-gated (they sit below the 3-seed MDE,
   as §2 itself states).
3. **Per-window looks DEMOTED to:** (i) the mechanism prong (trigger counter
   ≥1/run on ≥5 games + the component observable, incl. the A19 counter below) and
   (ii) the non-inferiority guard: pooled Δlc ≤ −0.10 → flag OFF. The guard is
   evaluated ONLY at the window's sealed look after seed 3 — never per-seed; there
   are no interim looks (resolves the "at any look" contradiction, methodology
   minor). Per-window score statistics are reported as descriptive monitoring,
   never as ON/OFF criteria.
4. **Retention rule:** a component stays in the cumulative stack iff mechanism
   prong PASS and non-inferiority PASS. "Mechanism fires, doesn't pay" is no longer
   a per-window kill — it is the question the cumulative look answers.
5. **P(pass | §4 expectations), published (methodology fix ii):** under the §4
   table (expected nonzero positive game-level means ≈ 4–8 at the full stack, ~0–1
   negatives), a binomial sketch gives P(pass) ≈ 0.05 (4 positives, clean) to
   ≈ 0.6 (7–8 positives, ≤0 negatives); point estimate ≈ 0.2–0.4. Honest reading:
   the cumulative look is a real test with real failure probability, not a rubber
   stamp — and not a false-negative machine.
6. **Cumulative-FAIL consequence (sealed now, per methodology Q5):** on FAIL, the
   stack is NOT dismantled (components already passed mechanism + non-inferiority);
   it ships with the honest label "mechanisms verified, score effect unconfirmed at
   rail MDE" and LB-accumulation status per A8. On PASS it is labeled a confirmed
   floor/mid raiser. Either way war-v4 remains the only registered wall-closer.

## A15 — Compressed-bench transfer rule (3× MAJOR)

A compressed-budget (40%-cap) window pass grants **provisional inclusion only**.
Before the cumulative look can claim score credit — and before any component
enters a scored-stack LB kernel — one FULL-budget certified confirmation replicate
of the accepted stack vs the W0 baseline must run on the rail. Per-component
trigger frequencies at full budget (from existing war_eval seeds 1–3 transcripts)
are published alongside every compressed-bench count, so the compression factor is
an explicit checkable assumption. All §2 ceilings measured at compressed budgets
are relabeled compressed-regime quantities.

## A16 — Banking retry de-bias (methodology MAJOR; rl-planning/llm-agents/prog-synthesis minors)

The frozen retry-target list (ft09/sc25/re86) is RETIRED — it was selected on the
same 3 seeds that estimated E[max-of-2] (winner's curse). Replacement, pre-registered
as an online game-agnostic policy: **retry a game iff its current-attempt outcome is
below its banked record AND the remaining soft-time budget covers the banked trace's
replay cost; order retries by (banked − current) descending.** The feasible ceiling
(≤ +0.15) must be recomputed under this policy with a permutation-calibrated
shrinkage haircut before the banking window opens; the banking mechanism prong
counts KAGGLE-side replay successes, not local ones (15/25 local engine versions
differ). Full-panel sign-off requirement (A9 adjacency) unchanged.

## A17 — war-v4 capability screen (3× MAJOR; pre-Aug-1, blocking)

Before the Aug 1 v4 registration: run Qwen3.6-72B-tier 4-bit under the IDENTICAL
harness on ft09/sb26/lp85/vc33 on the free Kaggle GPU build rail (30 GPU-h/wk),
with a measured tokens/s bench. Pre-registered go/no-go: **GO iff ≥2 levels beyond
the 27B baseline summed across the 4 games at full per-game budgets AND measured
throughput sustains ≥90% of the 27B action count under the binding budget** (the
binding budget is wall-clock on the scored rail; the expected 72B action count must
be computed from the measured tokens/s before registration). Throughput-adjusted
null (formula, closing the undefined else-branch): for each game, null_adj = the
levels the 27B baseline had completed by action N₇₂B, where N₇₂B = measured 72B
actions achievable in the wall-clock budget; 72B must beat Σ null_adj. NO-GO
finding ("72B replicates the ~1-level grinder profile") goes to the panel
immediately — the campaign would then have no registered wall-closer, and the
panel decides in July, not September.

## A18 — (d) offline prediction-accuracy metric (prog-synthesis MAJOR; before W1 seals)

The no-effect-FACT recurrence accuracy is being computed today on the N5
deterministic replay traces (`scripts/predict_metric.py` → `runs/predict_metric/`).
Pre-registered threshold, sealed before results are observed: (d)'s window
proceeds iff **recurrence accuracy P(no-effect again | prior no-effect observed) >
majority-class baseline** AND **trigger opportunities ≥1/run on ≥5 games**.
Otherwise (d) is killed cheaply now and W1 becomes (a)'s window.

**RESULT (observed ~14:30 EDT, after seal): (d) KILLED.** Pooled over 175
game-runs / 29,487 actions across 7 pulls (board_changed label integrity-verified
by independent frame hashing, 0/29,487 disagreements): recurrence accuracy 0.465
(Wilson 95% [0.436, 0.494]) at state_action granularity vs majority baseline
0.903 — decisive fail on the accuracy prong (trigger coverage passed, 68/175
runs). A recurring "no-effect" (state, action) pair actually changes the board
~54% of the time on these near-deterministic engines: the FACT rule would be
actively wrong most times it fired. Per the sealed rule, W1 becomes (a)'s window;
(c)'s disposition (formerly one flag with (d); forfeits standalone window under
the MDE/2 rule) goes to R15. Artifacts: `runs/predict_metric/{report.md,raw.json}`,
`scripts/predict_metric.py`.

## A19 — (c) mechanism prong upgrade (llm-agents minor N4)

"Verbatim-resubmit count = 0" is demoted to a code-shipped check (it is guaranteed
by the block's presence). The value-bearing observable added to (c)'s mechanism
prong: **post-block novel-family rate** — fraction of hard-block events followed
within 10 actions by an action outside the blocked family. Prong threshold: rate
> 0 on ≥3 games (the CONCEPT-lock finding predicts 0; this is the falsifiable bet).

## A20 — Declarations and process fixes

- **(g) per-game budget re-allocation is DEAD** (explicit resolution-by-removal;
  rl-planning/prog-synthesis demanded the declaration).
- **0.56× rail→LB conversion is an ASSUMPTION**, not a derived constant: all
  LB-unit claims carry a 0.4×–0.8× sensitivity band (llm-agents N5, methodology
  minor). §2's LB figures are so relabeled.
- **Circulation rule:** panel-facing documents must fit the reviewer argv budget;
  the full design doc + this amendment go to R15 in ≤2 parts with per-part sha256
  and untruncated END lines, before the recalibrated gate seals.
- **Timestamps (methodology Q3):** amendment 2026-07-18 (A8–A13) filed ~08:00 EDT
  Jul 18; this amendment ~13:30 EDT Jul 18; LB draw #5 submits ~20:07 EDT Jul 18
  (00:07Z Jul 19) and scores ~04:00 EDT Jul 19. Both amendments precede the draw's
  observation.


## END OF PART C ##


---

# PART D — State of the War synthesis 2026-07-18
(sha256:8a293d986c28d7f0 len:5305)

# State of the War — 2026-07-18 deep-understanding synthesis

Inputs: `lb_process_model/report.md`, `winners_deepread_2026-07-18.md`,
`opine_world_deepread.md`, `grinder_cracking_design.md`, panel R13,
amendments A8–A13, GPT-5.6 probe + distill corpus. Panel target: R14/R15.

## What we now KNOW (high confidence, instrumented or replicated)

1. **The harness transfers; the analyzer is the gap.** GPT-5.6-sol through our
   unmodified scaffolding: ft09 5/6, sb26 5/8, lp85 4/8 vs Qwen's ~1 level.
   (Probe, 2026-07-16.)
2. **All 25 games are frame-deterministic** (N5 audit 0/25 divergent), and the
   published 20/25 OPINE-World result is built on exactly that property.
3. **The LB draw distribution is generatively explained by our own bench** —
   no hidden deep-play regime. A 1.33 night needs the measured *common-night
   correlation* (shared server/sampling luck across the 110 slots), and is
   44% ft09-level-2. σ̂=0.074 was a lucky-tight n=5 sample of a σ≈0.13–0.17
   process. (lb_process_model, 20k-night sims, exact scorer.)
4. **Honest window pricing:** E[max@107 remaining] ≈ 1.39 central (pooled-10
   posterior predictive); P(touch 1.44) ≈ 0.29; P(reach 1.86) ≈ 0.01.
   **Filler is a lottery ticket, not a plan.** Break-even for spending a
   window on an experiment: credible official-set lift ≥ +0.06–0.12. The
   existing +0.12 gate thresholds already price this correctly.
5. **Mechanical no-effect refutation + verify-before-act are THE convergent
   scaffolding primitives** — independently arrived at by Reki (dead-signature
   veto, his 0.64→0.86), the 3rd-place build, OPINE's counterexample loop,
   and our GPT-5.6 distillation; and their absence is exactly why our ledger
   idled (1552 digests / 0 escalations, prose triggers never fire).
6. **The 1.44+ band is not explained by any public artifact.** Winner-tier
   public code tops out at 0.86–1.21; Tufa credits "multimodality + better
   base models"; ~14 wall-breakers share nothing. No public competitor shows
   OPINE-style executable world models.
7. **Our fork is drift-free** vs the public duck notebook (retires the Jul-16
   re-fork question).
8. **The v3 micro-stack cannot close the wall**: real-scorer counting bounds
   give ceiling +0.31 rail / +0.17 LB, expectation +0.04–0.10 LB
   (grinder_cracking_design.md). Reclaimed actions on uncompleted levels
   score zero.

## What we BELIEVE (medium confidence, one good source or inference)

- **su15 is NOT an information-theoretic wall** — OPINE solved it 9/9. Our
  A13 re-probe is now expected-retraction (the wall verdict rested on a
  deadlock-confounded probe). ka59/sk48/lf52/bp35/s5i5 (OPINE's failures)
  are *search-budget* walls, not world-model walls — and our exec_wm sims
  are already saturated on lf52 (100%) and s5i5 (99.5%).
- The 1.86 leader is most plausibly an OPINE-family system or a
  frontier-tier analyzer; either way, per-draw mean — not draw volume — is
  what they have.
- Night-level correlation (shared vLLM/sampling temperature luck) is real
  and material to LB variance; deterministic build-rail RNG (3rd-place
  trick) buys gate power without sacrificing LB order-stats value.

## What we DON'T KNOW

- Whether the plan-execute-verify contract survives contact with the scored
  regime's budgets (A10 bench must fire its triggers first).
- Whether a 72B-tier AWQ analyzer nets positive after the 2.5–3× throughput
  penalty (war-v4's central risk; INT4-vs-FP8 reasoning quality on grinders
  unbenchmarked).
- What the sealed A5/A8 look will say tonight at n=5 (no peeking).

## The strategy stack this implies (for R14/R15 to ratify)

| priority | line | basis | ceiling/expectation | cost |
|---|---|---|---|---|
| 1 | **EWM-execute: OPINE plan-execute-verify contract on our 12 saturated exec_wm sims** (harness-side BFS, one action/step, hash-verify vs settled frame, fail-closed) | KNOW#2, BELIEVE#1, uncontested edge (KNOW#6) | ceiling +0.5/draw rail — exceeds entire v3 stack; expect +0.10–0.30 | 2–3 days, 0 LLM tokens, own gated window |
| 2 | **(c)+(d) flag with Reki dead-signature as the click component** | KNOW#5 (quadruple convergence) | ceiling ~+0.10 rail | 1–2 days, already spec'd |
| 3 | **war-v4 model swap scoping** (72B AWQ, free rail bench first) | KNOW#1, KNOW#6 | the only proven wall-sized lever | scoping Aug 1, gated |
| 3b | mixed-tier routing (27B grinds / 72B consulted) — bench row ONLY inside v4 scoping | Kimi-3 review 07-18 | contingent on throughput binding | costs incl. FRONT-LOADED penalty: two model loads before action 1 + split KV cache, not just per-decode; dual-serve in 9h unproven; simplicity-wins prior against |
| 4 | su15 GPT-5.6 re-probe post-(f) (A13; expected retraction) | BELIEVE#1 | epistemic repair | ~$10, capped |
| 5 | Filler in every window no experiment credibly beats (+0.06–0.12 rule) | KNOW#4 | lottery: ~29% touch 1.44 over 107 | free |

Retired/parked: warpack (A9 double-lock), ledger-as-built (A11), prompt-line
transfer (distill-proven inert), r11l as a variance story (contributes ~0).

## Window discipline (unchanged, now priced)

One flag per window (A12). A window goes to an experiment only when its
pre-registered expected lift ≥ +0.12 official-set equivalent; otherwise
filler. Tonight: war draw #5 (committed, completes n=5 → sealed look).


## END OF PART D ##


---

# PART E — Daily Brief 2026-07-19 (draw #5 A5/A8 FAIL, W0 screen, sweeps, open questions Q1-Q6)
(sha256:d8ff906f71979f3d len:6786)

# Daily Brief — 2026-07-19 (Sunday)

## §1a Result deep-dive

### Draw #5 = 0.76 → sealed A5/A8 look = FAIL → war accumulation ENDS
War draw #5 scored **0.76 — campaign low** (observed LB range now 0.76–1.33). The sealed A5/A8 look was executed pre-loop this morning (`runs/a5_a8_look_2026-07-19.json`, deterministic arithmetic, consequences sealed 07-18 before observation): war ledger n=5 {0.91, 1.08, 0.88, 1.05, 0.76}, mean 0.936, σ̂ 0.1309, **χ²-CI-hi 0.376 ≥ 0.25 → FAIL**. Sealed consequence (A8): the war arm is accumulation-only permanently — ineligible as an A/B readout arm at any n; no mechanism may cite war-arm LB deltas as evidence either way. Per A9, war accumulation ends (no draw #6); the frozen fork resumes as filler (already queue head).

Was the pre-registered expectation met? Yes in the meta sense: R13 predicted the gate was near-unpassable (the draw needed [0.955, 1.005] to pass), and the A8 amendment sealed the fail-consequence before observation — the machinery worked exactly as designed. **Validation bonus:** pooled n=11 across both arms → mean 0.9655, σ̂ **0.154** — dead-center in the LB process model's predicted 0.13–0.17 bracket. The σ=0.074 era is formally closed. Window pricing stands: filler E[max@~106] ≈ 1.39; experiments must credibly claim ≥ +0.06–0.12 to price a window.

### W0 continuation eval — mechanism PASS, non-inferior, (f) hygiene CONFIRMED
`canivel/arc3-duck-w0-continuation-eval` v1 completed (2h12m, 25 games). Full screen: `runs/kernel_pulls/w0_eval_s1/screen_report.md`.
- **Tripwires:** all 3 banners present; zero warpack/ledger lines (the only greps are the "NO warpack" banners themselves).
- **Mechanism (pre-registered expectation: 0 idle turns): PASS — 49 GAME_OVER episodes across 12 games, 0 idle post-game-over actions.** Every game over recovered on the immediately following action (bp35 alone had 13 episodes).
- **Non-inferiority (descriptive, no score gate per the 0.00 counting bound):** 16 total levels — inside the ledger-OFF seed band {13, 15, 22}; mean 1.73 vs seeds {1.16, 1.58, 1.62}. No game below the 3-seed floor.
- **Interpretation:** (f) is pure hygiene and it works perfectly. Its value is eliminating idle-turn tail risk, not adding levels (Qwen recovers from game over anyway; the graft makes it deterministic). Author recommendation to R15: adopt continuation as a **default layer in all future builds**; seed-2 unnecessary (mechanism is deterministic, 49/49; there is no score gate a second seed would feed).

### Leaderboard
Leader Yuto Kojima 1.86 (resubmitted 00:02 today). Wall unchanged: 16th place = 1.44; ~14 teams in the 1.44–1.61 band; Tufa Labs 1.45. Our best 1.33 (frozen-fork right tail). No structural shift.

## §1b Discussions sweep (since 07-18)
Three threads active; no adopt-worthy LB technique.
- **#727505 "Constraint Before Control" (Yakunin, new):** verifier-holds-sole-action-authority architecture; author's own result 0.17 and only after disabling nearly everything. **ADAPT (low): conceptual convergence with our EWM plan-execute-verify contract; no validated technique to lift.**
- **#727119 host "500 submissions" thread, new comments:** Yakunin reports **reset-logic fragility** — a 5-resets-per-level cap turned a 9-min working agent into a 1-hour 0-score run. **ADAPT: concrete landmine for the war-v4 72B screen and any experimental window — reset-path changes must be A/B'd against the frozen fork before trusting.** Model-stack chatter (Qwen-3.6-27B/Gemma-4-31B opinions): IGNORE (unvalidated). Zejun_ team-up drama: IGNORE.
- **#726367 AGI-timeline thread:** IGNORE (speculation; mild confirmation the public field is stuck near 1.86).

## §1c Research sweep (since 07-16; back-filled early July)
Two ADAPTs land directly on our EWM step-0 abort problem:
- **OCM — arXiv:2607.02846 (ADAPT, high priority):** coupled object-knowledge + procedure-knowledge codebases; **procedures verified against the refined object model before execution**. Candidate fix for our step-0 aborts: validate the planned procedure against the model pre-execution instead of discovering phase misalignment at live step 0 — decouples verification from timer/hidden-counter phase.
- **World-model collapse phase transition — arXiv:2606.31399 (ADAPT):** "world-state fidelity fails before action validity." Diagnostic frame: our step-0 aborts are likely world-state-fidelity (phase) failures, not planning bugs → argues for an explicit **re-observe/resync step before declaring abort** rather than fail-closed on first mismatch.
- AgentLTL arXiv:2607.02599 (ADAPT low, park): online prefix-gating formalism; verifies ordering not state equality.
- Agentic TTT arXiv:2607.03441: IGNORE (needs training data; Kaggle no-weight-update regime).
- 72B AWQ on 96GB: **no external throughput anchor exists** — memory fit confirmed comfortable (~35–40GB weights), throughput must come from our own A17 bench. We are the reference.

## Weekly fingerprint table (Sunday step ii)
16 incidents, 8 recurring families. Top families: `class:ERROR:none` n=7 (05-26→06-28), `provenance:scratch-built` n=5 (the arc3-final/forge35/jepa/execwm scratch-drift cluster — now blocked by preflight), `slug:canivel/arc3-final` n=4, `class:COMPLETE:0.00` n=3, `slug:arc3-pilot-eval` n=3 (07-07/08), `class:COMPLETE:null-band` n=2. No NEW incidents this week — preflight recurrence WARN + fork-never-build holding.

## Open questions → R15 (full 5-reviewer, circulation per A20 via untruncated @file delivery)
1. **A14 recalibrated gate seal** — this circulation is the sealing one: cumulative stack-vs-W0 look as THE binding score decision; per-window looks = mechanism prong + non-inferiority only. Confirm or object now; seals on this round.
2. **EWM Stage-1 re-pricing** — Stage-0 dry-run showed held-out saturation does NOT transfer on-trajectory (vc33 + s5i5 = 2 of 5 targets abort at step 0). Discount the +0.10–0.30 expectation; and rule on the two candidate step-0 fixes from today's sweep (OCM pre-execution procedure validation; resync-before-abort per 2606.31399).
3. **W0 disposition** — adopt continuation as default hygiene layer in all future builds? Seed-2 needed? (Author: yes-default, no-seed-2.)
4. **W1 owner = (a) budget sentinel** (A6 deadline Jul 20) — confirm scope + A10 compressed-budget canary design; (c) Reki-signature suppression disposition (build order after (a)?).
5. **A17 72B screen scope** — weights dataset + vLLM bench kernel on the free rail; reset-fragility caution from §1b applies; go/no-go = ≥2 levels beyond 27B on ft09/sb26/lp85/vc33 AND throughput-adjusted null formula. Pre-Aug-1 blocking.
6. **state_of_the_war priority ratification** — EWM-execute line vs design-doc ordering, given the Stage-0 discount (Q2) and that EWM remains the only uncontested-edge line.


## END OF PART E ##
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
