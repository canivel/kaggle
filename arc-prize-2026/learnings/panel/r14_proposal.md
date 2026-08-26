# R14 PROPOSAL — Part 1 of 2: grinder-cracking design doc (R13 FATAL discharge; PRIMARY review target)

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

# R14 PROPOSAL — Part 2 of 2: daily brief 2026-07-18 (variance flip + sweeps + Q-A..Q-E)

# Daily Brief — 2026-07-18

Author: daily loop (08:23). Inputs: overnight LB, sweeps (`war_room/discussions_2026-07-18.md`, `war_room/research_2026-07-18.md`), gate-look aftermath (ITERATION_LOG 2026-07-17/18), `preregistration_amendment_2026-07-18.md` (A8–A13, filed pre-observation ~08:00 EDT), `war_room/grinder_cracking_design.md` (R13 FATAL discharge — R14's primary review target).

## 1a. Result deep-dive: filler draw = 1.33, new campaign best — and a variance bombshell

**What fired:** the Jul-17 loop died at its 80-turn cap with panel R13 zombied and the queue head unset, so the **frozen-fork filler** (`canivel/arc3-duck-repro` v3) fired at 00:07Z and scored **1.33** — the campaign's best single draw. No intervention was aboard; this is a vanilla draw.

**What it means (validated interpretation, not the raw number):**

1. **The pre-registered control σ̂ = 0.074 is now empirically refuted as a point estimate.** Under the frozen control ledger {0.82, 0.89, 0.93, 1.02, 0.95} (mean 0.922, σ̂ 0.074), a 1.33 draw is z ≈ +5.5 — effectively impossible. The χ² 95% CI on σ [0.044, 0.213] always warned the n=5 estimate was loose; 1.33 is the right tail announcing itself. **Pooled recompute treating 1.33 as frozen-fork draw #6 (descriptive; formal ledger adoption needs an amendment ruling — R14 question Q-A):** mean 0.990, σ̂ **0.179**, χ² 95% CI on σ (df 5) **[0.112, 0.440]**. LOO caveat: the entire update is carried by the single 1.33 draw — but that is how right tails work at n=6; the draw is real and scored.
2. **The order-stats conclusion flips at the revised σ.** The 07-14 reconciliation ruled order statistics "a floor-raiser, never a wall-breaker" at σ=0.074 (E[max] 1.11@k=110). At σ̂=0.179 and mean 0.990: E[max of k draws] ≈ **1.36@k=30, ≈1.44@k=110**. With ~107 windows left, *nightly resubmission of the frozen fork alone* has an expected maximum at the 1.44 resubmission wall — the current top-15/20 cutoff band. The "only per-draw mean gains reach the wall" premise weakens: draw volume is now a live wall-path. Implication if it survives panel scrutiny: **the filler is not just a fallback, it is a strategy**; every window burned on a null-EV experimental draw has a real opportunity cost, and every window must clear "beats a vanilla lottery ticket" — while conversely, harm-risky interventions look worse.
3. **War-v1 context:** ledger n=4 {0.91, 1.08, 0.88, 1.05}, mean 0.980, σ̂ 0.0997. Draw #5 fires tonight (00:07Z Jul 19) → n=5, then the **sealed A5/A8 variance look** (χ²-CI-hi < 0.25 at df ≥ 4). Note the tension to resolve: war σ̂ (n=4, CI-hi 0.372) vs revised control σ̂ 0.179 — under A8, future thresholds are relative to control CI-hi, which just widened.
4. **Tonight's window is already committed** to war draw #5 (final accumulation, licensed by prereg §3 + A8). The variance flip does NOT change tonight; it changes how Jul-19+ windows are priced.

## 1b. Discussions sweep (2 new; details in `war_room/discussions_2026-07-18.md`)

- **#727119 host post "500 Submissions Analyzed" (Greg Kamradt) — ADAPT.** ~1/3 of failed subs "just get stuck" (no traceable error = our 0.00 infra-death class); ~20% GPU-code-without-GPU-flag. Host confirms organizers cannot see notebooks until open-sourced → the 1.86 leader stays opaque; no one can inspect us either. Action: preflight asserts GPU flag for the war-v4 rail; wall-clock deadline on any watchdog/summarizer thread.
- **#724841 host reply on rerun limits — ADOPT (infra constants).** SIGSEGV=139 surfaced but core dumps hidden; Docker logs silently truncate at 10 MB; /kaggle/working quota 20 GB with ~60 GB scratch outside it; memory = cgroup-enforced 30 GB physical; no RLIMIT_NPROC/AS. Actions: cap probe-diff summarizer logging < 10 MB; banking/replay traces to scratch, not /working; budget war-v4 72B thread stacks against 30 GB.

## 1c. Research sweep (details in `war_room/research_2026-07-18.md`)

- **OPINE-World (arXiv:2607.01531 v2) — ADAPT, top priority.** Published ARC-AGI-3 result: **20/25 games, 160/183 levels, no per-game training** via object-centric programmatic world model (Python + CEGIS), deterministic-transition assumption, and a replay-check against settled state. Independently ratifies frame-determinism (our N5), executable rule banks, and PREDICT→RESULT verification. Cheapest legal extraction: the **replay-check contract** — hash the predicted next frame vs actual, log mismatch as a refutation FACT — which gives the (d)/(c) ledger records a *mechanical firing trigger*. Also a sobering capability benchmark: 20/25 exists in public literature while we clear ~8–10 games' worth of levels; plausibly the 1.86 leader's family. Feeds war-v3 (d)+(c) and the war-v4 case.
- **GSME (arXiv:2607.13683) — ADOPT (methodology, now).** Gated semantic quality-diversity for harness evolution; names our exact 07-17 failure: credit a patch only when its **mechanism actually fired** (activation gate BEFORE the significance gate). Action: formalize the activation gate as prong 0 in the war-v3 build protocol (A10 already requires trigger-firing benches — this makes it first-class), and bucket the (a)–(g) backlog by failure pathology (budget-death, verbatim-resubmit, game-over-deadlock, stuck-loop) rather than mechanism name.
- Weak/park: task-count replay analysis (2607.12338, methods cite); π-Distill (needs training loop — parked with war-v4); no new 72B-AWQ serving result (INT4-for-reasoning caution stands — bench INT4 vs FP8 on the grinder set specifically).

## 1d. Today's plan + open questions for R14

**Plan (per A12/design-doc timeline):** (f) game-over-continuation fix ships FIRST, standalone, unflagged — implemented today as a guarded runtime patch (phase1/ledger pattern; `prompts.py:100` "stop acting immediately" → continuation semantics), smoke-tested, then A13 su15 re-probe (GPT-5.6, local, $10 ceiling) once (f) lands in the local rig. Queue: war draw #5 tonight (committed). Pushes 0/2 used.

**Questions for panel R14** (alongside its primary charge: adversarial review of `grinder_cracking_design.md`):

- **Q-A (variance flip):** Adopt 1.33 into the frozen control ledger formally (amendment), or hold it as a descriptive 6th draw? Either way, does the revised σ̂ (0.179, CI [0.11, 0.44]) overturn the 07-14 "order stats are never a wall-breaker" ruling — and if yes, what is the pre-registered window-pricing rule (which nights are experimental draws still worth burning)?
- **Q-B (frozen-vs-war allocation):** war mean 0.980 (n=4) vs frozen mean 0.990 (n=6, descriptive): after tonight's sealed n=5 look, is there any remaining case for war draws over frozen draws in scored windows, given A9 parked warpack as UNTESTED-IN-REGIME?
- **Q-C (activation gate):** ratify GSME-style activation gate as prong 0 of the A10 compound gate (mechanism-fired before Δlc significance is even computed)?
- **Q-D (replay-check contract):** fold OPINE-World's predicted-frame-hash refutation trigger into the (d)+(c) flag spec (single code path, separate counters)?
- **Q-E (reach-probe line):** the su15 distill's "measure reach before paying per-step costs" prompt line — include in (f)'s window (it is also pure hygiene), sequence it later, or drop under `feedback_prompt_is_noise`? Author lean: exclude from (f) (keep (f) claim-free), let A13's re-probe run continuation-only per the amendment's letter.

## END OF PROPOSAL ##
