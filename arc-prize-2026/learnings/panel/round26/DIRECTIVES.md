# ROUND 26 — DIRECTIVES (mined, not adjudicated)

**Status of the verdict: NOT A BLOCKER.** All 5 returned MAJOR-REVISION / 0 fatal (5, 5, 6, 6, 6).
R10–R26 have produced **0 accepts**; MAJOR-REVISION cannot terminate by design (principal's addendum
2026-07-27) and the panel is **ADVISORY ONLY**. No further round is recommended or required. This file
extracts the *directives* and discards the *disposition*.

Reviewers: `rl-planning` (RL), `llm-agents` (LA), `prog-synthesis` (PS), `methodology` (ME), `systems` (SY).

---

## 1. CONSENSUS (3+ reviewers, independently)

### C1 — Publish the Q38 arm's operating characteristics BEFORE the read is pulled. **5/5.**
Every reviewer raised this as MAJOR. The demand is identical: P(CONFIRM), P(REFUTE), P(INDETERMINATE)
as a function of true effect δ, written down *before* `kernels output` touches v2.

> "Without those numbers the pre-commitment is theater: the panel would be choosing a label, not a
> decision rule." — **ME**

> "if the disposition is not fixed before the result is pulled, the sealed prereg's whole purpose is
> defeated by an interpretive degree of freedom held in reserve." — **LA**

> "deciding without a power calculation is not deciding, it is postponing with extra steps." — **RL**

**Sub-consensus (4/5: LA, PS, ME, SY) — the units do not connect.** The re-anchored prior is in LB
points; the sealed read is in levels-completed. No mapping exists.

> "The re-anchored prior (+0.17 LB) and the sealed read (levels: ≥32 / ≤25) are in **incommensurable
> units**, so open question 3 is ill-posed as written." — **LA**

⚠ See §5 F8: the panel's "nobody computed the power" premise is **partly false** — the MDE *is* sealed
in the prereg. Only the LB↔levels conversion is genuinely missing.

### C2 — The "+0.17-class" re-anchor is not a valid estimator. **4/5** (RL, LA-minor, PS, ME).
Two independent defects: **selective exclusion** of aRc's +0.74 (same ΔSub==1 dating rule) and
**max-censoring** in both directions.

> "Re-anchoring to '+0.17-class' silently drops the largest observation from the identical dateability
> criterion; the honest statement is **'dateable effects span 0.00 to +0.74, n=3, heterogeneity
> dominates the estimate.'**" — **ME**

> "FOYSAL's 1.61 → 1.61 is *right-censored* — his post-swap draw could have been 0.9 or 1.60 and the
> board shows the same nothing, so 'FOYSAL drew +0.00' is not a data point, it is an unobserved value
> with an upper bound. … exactly the best-of-N pathology your own adopted paper (2608.13087) warns
> about." — **RL**

### C3 — Demote "the control arm did not move" from load-bearing. **3/5** (RL, PS, ME).
Two teams × one draw each, against incumbent bests that are themselves high order statistics.

> "Calling this 'the opposite of what a shared commodity-engine story predicts' converts an
> **underpowered null into affirmative evidence** — the exact error class of 2608.13087, which the brief
> itself ADOPTs as a lens." — **ME**

> "Compute the actual likelihood ratio P(flat | swap, δ=+0.17-class) vs P(flat | no swap) before calling
> this line load-bearing; **my back-of-envelope says it is close to 1**." — **PS**

Quoted detection probabilities: PS "~40–60%", ME "0.4–0.7 per team", RL "well under half the time".

### C4 — OQ-5 is the campaign's center of gravity and it arrived empty. **4/5** (RL, LA, PS, SY).
Nothing on the rail has an *expected* — or in SY's stronger form, even an *upside* — case reaching the
gold line.

> "not yet a plan; it is **an excellent lab notebook attached to a strategy vacuum**." — **RL**

> "the rail contains: a frozen filler (0.94 mean), Q38 (+0.17-class), a dead LoRA canary, and a CPU-only
> diagnostic. Nothing on the rail has a mechanism story that reaches even the *current* gold line. This
> is a **portfolio-allocation defect, not a measurement defect**." — **SY**

> "As written this is a well-instrumented description of standing still." — **LA**
> "a well-instrumented plan to measure standing still." — **PS** (independent, near-identical phrasing)

RL closes the escape hatch: cstl's **1.59 → 2.70 on our own artifact family** "proves the family ceiling
permits it, so 'no such artifact exists' is not available as an excuse." (**VERIFIED** — see §5.)

### C5 — OQ-1's probe is endorsed in direction and blocked in form. **4/5 conditions** (RL, LA, PS, ME); SY endorses unconditionally.
No reviewer disputes the reframing. Four demand it be made falsifiable *before* it becomes a rail item.

> "'is the agent's action-selection criterion monotone in what scores?' is currently **a slogan**." — **PS**

> "a confirmed non-monotonicity must name the intervention it licenses or it is **a diagnostic with no
> consumer, the very pathology the papers describe**." — **PS**

> "no branch: what artifact gets built if the criterion is found non-monotone, given that prompt-side
> intervention is exactly what 1c-2 says fails … (e.g., **decode-side action reranking or scorer-in-the-loop
> selection, which *are* available in this harness**)" — **LA**

> "retrospective on trajectories already on disk and therefore **HARKing-prone** … define the statistic and
> its null band on a nominated *subset*, then score the held-out remainder." — **ME**

### C6 — Today's instrument fixes are the wrong *class* of fix. **3/5** (LA, PS, SY).
All three say the same thing from different angles: the remediation improved **detection after death**, not
**prevention**, and the fixes themselves were never verified.

> "the remediation list (backfill-before-report, staleness banner, `--dry-run`) improves *detection after
> death*. The prevention fix is cheap and specific: (a) a lint rule rejecting any heredoc body that
> references a name defined in notebook cells, or (b) require the build check to *execute* the heredoc in
> an isolated interpreter." — **SY**

> "the fixed suite should also be **reordered load-bearing-first** (engine-attachment check before cosmetic
> byte checks), not just patched; the brief adopts the lesson **rhetorically but not structurally**." — **PS**

> "'we edited the checks until the push passed' is **observationally identical to the failure mode the
> checks exist to prevent** … a rule that any edit *loosening* a check voids the push. The claim that fix 2
> is 'strictly stronger' is asserted, not demonstrated." — **LA**

---

## 2. DISSENT (real contradictions — do not average)

**D-A — Is the +0.17 re-anchor good discipline or a statistical error?**
- **SY**: it is *evidence of quality* — "the +0.17-class re-anchoring of the Q38 prior … the best in this
  campaign" — and SY **builds its own MAJOR on top of it** (the ~64% bar-clearance figure assumes δ=+0.17
  is the operative prior).
- **RL / PS / ME**: it is *the brief's central inferential defect*; RL goes furthest — "**drop the re-anchor
  and admit the prior is uninformative**."
- **LA** sits between: directionally right (down from 2×) but must be stated as "≤ small, n=2, censored."

*What the disagreement tells you:* the evidence is thin enough that a competent reader can take it as either
a sound update or an artifact. **Do not plan against +0.17 as a point value.** The only defensible objects are
the interval [0.00, +0.74] over n=3, or an explicit "unidentified from LB data."

**D-B — Does a pre-data power audit license touching a sealed arm?**
- **PS**: yes — "If that calculation shows REFUTE-2× is the modal outcome even under δ=+0.17, **the arm should
  be re-scoped before slot 2 is spent, sealed read or not** — sealing protects constants from post-hoc motion,
  not from a pre-data power audit."
- **RL / LA / SY**: no — annex/label only. RL: "say so *in the prereg annex before the kernel completes*."
  SY: "pre-commit now that REFUTE-2× is logged as **UNDERPOWERED-AT-PRIOR**" — a labelling change, not a re-scope.

*Operational consequence:* PS would spend slot 2 differently; the other three would spend it as sealed and
change only the write-up. **Recommendation: take the RL/LA/SY side** — the campaign's whole seal discipline
(`feedback_audit_the_instrument`) is worth more than one arm's efficiency, and §5 F8 shows the power question
is answerable *without* moving anything.

**D-C — Is OQ-1 adoptable as-is?**
- **SY**: yes, unconditionally — "the CPU-only reframing in OQ1 **fits the compute envelope exactly**."
- **LA / PS**: no, MAJOR — not without statistic + threshold + named intervention branch.
- **ME**: no, MINOR — not without a mini-prereg and a held-out split.

*Reading:* SY is judging **cost**; LA/PS/ME are judging **inferential validity**. Both are right about
different questions. The brief's APPENDIX settled cost; it did not settle validity.

**D-D — Is the engine artifact vetted?**
- **LA / SY**: MAJOR — unverified anonymous weights, serving-compat gate not discharged.
- **RL / PS / ME**: explicitly park it in "What I cannot judge."

*This dissent is void:* both objecting reviewers are working from a false premise (§5 F1, F3).

---

## 3. TOP ACTIONABLE DIRECTIVES (ranked by impact ÷ cost)

**All six are FREE / CPU-only and runnable TODAY at zero cost.** That is itself the panel's headline: not one
of the five reviewers demanded a GPU-spend or a kernel slot to fix anything. Follow-through on D2 and D3
eventually consumes slots; the directives themselves do not.

| # | Directive | Reviewers | Cost class | First step |
|---|---|---|---|---|
| **1** | **Append a disposition table to the Q38 prereg BEFORE any `kernels output` pull on v2** — P(CONFIRM)/P(REFUTE)/P(INDET) over a δ grid, plus the pre-committed label for each landing zone. Touch **no constant**. | **5/5** | **FREE / CPU-only** ⏱ *time-critical — window closes when the kernel completes* | Append §13 to `learnings/war_room/q38_engine_swap_prereg_2026-08-15.md` using the already-sealed σ̂=0.141740 and SE(Δ)=0.163667 lc/game; the table is in §4 of this file, ready to paste. Add one line: **"the LB-Δscore→Δlc conversion is NOT DERIVABLE from the record"** (see F7). |
| **2** | **Answer OQ-5 in writing with ≥1 named mechanism whose UPSIDE ≥ 1.65, and attach a slot budget** — or formally re-scope the campaign objective away from gold and say so. | RL, LA, PS, SY (**4/5**) | **FREE to draft** (CPU-only); follow-through is **kernel-slot**) | Start from the standing existence proof: cstl went **1.59 → 2.70 on the duck artifact family we also run** (`runs/lb_ground_truth.md` 08-14 correction, lines 120–126) with zero disclosed method. Enumerate what a band team can change with **no solver-code edit**, and state what fraction of the ~156 remaining slots goes to ≥1.65-upside lanes vs. instrument work (RL's Q5/Q7). |
| **3** | **Mini-prereg the OQ-1 monotonicity probe — estimand, statistic, null band, held-out split, and the named intervention each branch triggers — THEN run it.** | RL, LA, PS, ME (**4/5**); SY endorses unconditionally | **FREE / CPU-only** (feasibility already closed by the brief's APPENDIX) | Write `learnings/war_room/monotonicity_prereg_2026-08-17.md`: statistic = Kendall τ between the per-step action-selection signal and realized Δ`levels_completed`; sample = a nominated 12 of the 25 games in `runs/a22_v2_seed1/intermediate_states.pkl`, scored on the held-out 13; null = permutation. **Branch, stated before running:** non-monotone ⇒ **decode-side action reranking / scorer-in-the-loop** (LA's named lanes) — **never prompt-side**, which 1c-2 predicts nulls. Env frictions already documented in the APPENDIX. |
| **4** | **Re-derive the Q38 prior over ALL THREE dateable points {aRc +0.74, Ya Xu +0.17, FOYSAL 0.00} with the censoring direction explicit — or withdraw the re-anchor entirely.** | RL, LA, PS, ME (**4/5**) | **FREE / CPU-only** | One paragraph + one calculation: P(Δbest = 0 \| mean shift δ) using s=0.1563 from `runs/ledger.json`, per team, given where their banked max sits. Then replace "+0.17-class" with the interval or with "unidentified." **Do not let +0.17 enter any downstream plan as a point value (see D-A).** |
| **5** | **Convert instrument audit into three PREVENTION rails, not another detection report** (this is the OQ-6 answer the panel gave in substance). | LA, PS, SY (**3/5**) | **FREE / CPU-only** | (a) **Reorder** the gate suite load-bearing-first in `scripts/preflight.py` — engine/dataset-attachment before byte-cosmetic checks (PS); (b) add a **heredoc lint or isolated-exec build check** so `"$PYTHON" - <<'PYSETUP'` bodies referencing notebook-cell names fail at build (SY) — this surface has now killed 2 kernels 20 days apart; (c) **verifier change-control**: pre/post sha256 + written rationale, and a standing rule that **any edit that loosens a check voids the push** (LA). |
| **6** | **Pre-register the OQ-4 per-game secondary read as strictly non-inferential, with a multiplicity correction.** | ME only (**1/5**) | **FREE / CPU-only** | State in the prereg annex that the per-game split across ~25 games is descriptive, carries no verdict, and cannot contaminate the primary read — mirroring §4.4's existing treatment of the score-based reading. |

---

## 4. RULINGS ON THE BRIEF'S 6 OPEN QUESTIONS

### OQ-1 — retire "did transitions arrive?" → ask "is the criterion monotone in what scores?"
**RULING: ADOPT — with blocking pre-registration conditions. 5/5 endorse the direction; 4/5 attach conditions.**

The reframing itself drew **zero objections**. PS, whose field this is, is explicit: *"I endorse the reframing
direction (2608.12959/12321's 'information present, consumer's criterion doesn't use it' is exactly the right
diagnostic class, and it is squarely in my field)."* RL: *"I support the CPU-only diagnostic direction — the two
ADAPTed papers earn it."*

**The APPENDIX's feasibility claim is uncontested and was independently affirmed on cost grounds** — SY: *"the
CPU-only reframing in OQ1 fits the compute envelope exactly."* No reviewer questioned 0 GPU / 0 slots / 0 dollars.

What is blocked is the *form*, not the *strategy*. Conditions, all four demanded before rail entry:
1. **Estimand + statistic** — "monotone in *what*? (logit of chosen action vs. counterfactual score-delta? rank
   correlation over the action set at each step?)" (RL)
2. **Trajectory set + selection rule** — "on *which* trajectories (successful only? the selection is confounded)" (RL)
3. **Pre-registered threshold and null** — "Kendall τ with a CI excluding some null band" (PS); permutation null (ME)
4. **Anti-HARKing split** — nominate a subset, score the held-out remainder (ME)
5. **★ A named intervention branch** — the hardest condition, from LA and PS: the papers' *fix* does not transfer
   (no explicit CEM cost to swap), so a confirmed non-monotonicity must name what gets built. LA supplies the two
   candidates: **decode-side action reranking** or **scorer-in-the-loop selection**, both available in this harness.
   **Prompt-side is excluded by our own adopted 1c-2.**

**Net: run it, but write the mini-prereg first. Cost of the prereg: one file. Cost of ignoring the condition: an
unfalsifiable "the objective is broken" narrative (RL's exact words).**

### OQ-2 — pre-register that the 31,744-ceiling fix will null
**RULING: PARTIALLY ADDRESSED (2/5 — PS, ME). Both support it; both say it is currently one-sided and therefore
not yet science.**

> "as stated it is unfalsifiable-in-practice for the routing hypothesis: a null confirms it, and a positive can be
> attributed to power or confounds. **State now what observed effect size at the raised ceiling would count as
> *refuting* the routing reading** of 2608.12321 for our substrate." — **PS**

ME independently: *"must specify the two-sided decision rule … or it becomes an unfalsifiable told-you-so."*
RL, LA, SY: silent. **Action: cheap — add the refutation threshold to the same annex as directive 1.**

### OQ-3 — if v2 lands REFUTE at a +0.17-class true effect, is that a refutation of the claim or of our power?
**RULING: THE PANEL DEMANDED THE ARITHMETIC AND — CONTRARY TO ITS OWN PREMISE — MOST OF IT ALREADY EXISTS SEALED.**

**Who actually gave numbers (the task asks specifically):**
- **`methodology`, the reviewer who should own this, gave NO power or MDE arithmetic for the Q38 read.** ME's
  entire OQ-3 objection is a *demand*: *"no one has stated P(REFUTE | true effect = +0.17), P(REFUTE | δ = 0), or
  the variance of the level-count endpoint under the sealed thresholds."* ME's only numbers — *"P(one new draw
  fails to beat a high incumbent max | true +0.17 mean shift) is plausibly 0.4–0.7 per team"* — are about the
  **MindsAI/Tufa control arm**, not the Q38 read. ME's prescription: publish P(CONFIRM), P(REFUTE), P(neither)
  under **δ ∈ {0, +0.17, +0.74}** before the kernel completes.
- **`systems` is the only reviewer who produced explicit arithmetic**, reproduced exactly:
  > "a true +0.17 lift shifts the draw mean to ≈1.11; a mean-of-4 at s=0.1563 gives **σ/√4≈0.078**, so even a
  > *real* +0.17 effect clears the 1.0826 bar only **~64% of the time**" — **SY** (independently verified correct:
  > (1.1124−1.0826)/0.078 = 0.38 ⇒ 65%). SY's prescription: *"state the minimum detectable effect of the sealed 2×
  > design at α of choice. If MDE > +0.17-class, pre-commit now that REFUTE-2× is logged as
  > **UNDERPOWERED-AT-PRIOR**, not as evidence against the engine."*
- **PS** demands the same table over δ ∈ {0, +0.17, +0.74} and adds the one-line reframe that resolves D-B:
  *"sealing protects constants from post-hoc motion, not from a pre-data power audit."*
- **RL** frames it as the decision to be made: *"If that probability is below ~0.5 … a REFUTE is nearly
  uninformative about the engine claim and you must say so in the prereg annex before the kernel completes, or
  **the sealed read becomes a coin-flip ritual**."*

**★ CORRECTION TO THE PANEL (all four): the MDE is already sealed.** `q38_engine_swap_prereg_2026-08-15.md` §4.3,
"POWER HONESTY (SCREEN_PROTOCOL §4.6)", pre-registered before the push:
- σ̂ = **0.141740** lc/game (df 6, m=3 baseline: lc 18 / 19 / 21)
- SE(Δ) at k=1, m=3 = σ̂·√(1+1/3) = **0.163667 lc/game = 4.09 levels over 25**
- CONFIRM-2× power vs a true doubling = **95.3%**; false-positive under the null = **0.11%**
- P(REFUTE-2× | true doubling) = Φ(−3.20) = **0.07%**
- **80%-power MDE floor = (2.02 + 0.8416)·0.14174·√(1+1/3) = 0.468 lc/game = 11.7 levels**
- and the design **already says so out loud**: *"The lc read is well powered *against a doubling* and **badly
  powered against small effects**."*

**The disposition table the panel asked for, derivable today with no unsealing** (thresholds: REFUTE Δlc ≤ +0.250,
CONFIRM Δlc ≥ +0.500, SE 0.163667; the δ=0.773 row reproduces the prereg's own 95.3% / 0.07%, which validates the
arithmetic):

| true Δlc (lc/game) | ≈ levels over 25 | P(REFUTE-2×) | P(INDETERMINATE) | P(CONFIRM-2×) |
|---|---|---|---|---|
| 0.000 (null) | 19.3 | **93.7%** | 6.2% | 0.11% |
| 0.100 | 21.8 | **82.0%** | 17.3% | 0.7% |
| 0.250 | 25.6 | 50.0% | 43.7% | 6.3% |
| 0.468 (MDE) | 31.0 | 9.1% | 48.6% | 42.3% |
| 0.773 (a true 2×) | 38.6 | 0.07% | 4.7% | **95.3%** |

**THE RULING — answer it this way, before the pull, and it costs nothing:**
Unless the true effect is ≥ ~+0.25 lc/game, **REFUTE-2× is the modal outcome BY PRE-REGISTERED DESIGN**. That is
neither a surprise nor a power failure — the arm was sealed as a test of a **doubling**, and the verdict is literally
named `REFUTE-2×`. So:

> **A REFUTE-2× reading kills the "2× on the local 25" claim on our harness at better than 3σ, and says NOTHING
> about the engine's marginal value at a +0.17-class effect, which sits below this design's 11.7-level MDE and is
> therefore UNMEASURED by this arm.**

Pre-commit that sentence now. It refutes *the claim*, not *the engine*; and it is not a refutation of our power
either, because the power was honestly declared before the push. **The one genuinely missing object is the
LB-Δscore → Δlc conversion — and it is NOT DERIVABLE** (F7): the baseline's own within-null `mean_score` spread is
1.427 / 1.939 / 3.420 (sd 1.033 on n=3, a 2.4× spread *inside the null*), which is why §4.4 already ruled the
score-based reading carries no verdict. Say "not derivable" in the annex rather than inventing a ratio.

### OQ-4 — should the Q38 read carry a per-game secondary read (Le Grand's public/private split risk)?
**RULING: ADDRESSED BY 1/5 (methodology). Conditional yes.**
> "will it be pre-registered with a **multiplicity correction**, given per-game splits multiply comparisons across
> ~25 games and **the primary read must not be contaminated** by a 'significant' secondary?" — **ME**

RL, LA, PS, SY: **NOT ADDRESSED.** Treat as: add it, mark it strictly descriptive, correct for 25 comparisons.

### OQ-5 — what is the next artifact that could clear 1.0826, and is anything on the rail aimed at it?
**RULING: 4/5, and this is where the panel's real weight fell.** See C4 and directive 2. Three distinct escalations:
- **RL** — this is a *fatal* strategic gap filed as a discussion item: *"I require the revision to include at least
  one concrete artifact hypothesis with a mechanism-level reason it could clear ~1.5 (not 1.08)."* Note the target
  RL sets: **1.5, not the internal bar** — because 1.0826 "measures 'better than our own frozen fork,' not
  'competitive'."
- **SY** — attach a slot budget: *"what fraction of remaining GPU slots goes to lanes whose upside case ≥ 1.65 —
  or formally re-scope the campaign objective away from gold."*
- **PS** — the honest fallback is legitimate but must be *stated*: *"or state explicitly that the campaign strategy
  is now 'await Q38 + instrument hygiene,' which the panel should then evaluate as such."*

### OQ-6 — should "instrument audit" become a standing rail item?
**RULING: NOT ADDRESSED AS POSED — no reviewer answers the rail/no-rail question. But 3/5 independently prescribe
the CONTENT such a rail item would have (C6), and 2/5 warn about its BUDGET.**

The content is directive 5 (reorder gates load-bearing-first; heredoc lint or isolated-exec; verifier change-control).

**The countervailing signal matters more than the endorsement.** RL asks, in the same breath as OQ-5:
> "what fraction of remaining slots (≈78 days × 2) goes to **searching for one vs. auditing instruments**?" — **RL**

and SY lists "a CPU-only diagnostic" among the rail items that reach nothing. **So: yes, make the three prevention
rules standing — they are one-time and free — but do NOT let "instrument audit" become a recurring daily rail item
that consumes the capability lane.** The panel's implicit position is that instrument hygiene is a *precondition*,
not a *program*.

---

## 5. WHAT THE PANEL GOT FACTUALLY WRONG

Do not let any of these propagate into tomorrow's rail.

**F1 [LA MAJOR, SY MAJOR] — "`saltb0x/qwen3-8-27b-fp8` has no stated provenance chain / appears nowhere in the
record before this brief." FALSE.**
- `learnings/qwen38_scout_2026-08-15.md` L157 catalogs it: **25,346,275,232 B, 2026-08-14 22:55:20 UTC, 30 downloads.**
- Prereg §1.1 is an explicit **three-mirror file-level hash comparison** (saltb0x / mustangliu / johnlussier).
- `ITERATION_LOG.md` 08-15: *"All three mirrors' config/tokenizer/template files pulled and hashed: **byte-identical
  across all three, nothing missing** … `text_config` identical in **all 33 fields**."* Selection was made on
  **operational risk only** (most downloads; Apache-2.0 vs mustangliu's `unknown`) and that reason is stated in the
  prereg as *"a weak reason honestly stated."*
- **Residual valid fragment, and it is narrow:** no shard-level hash against the *official HF release* was done —
  only cross-mirror agreement and config/tokenizer identity. Worth one line; not a MAJOR.

**F2 [LA] — "the 08-15 ground truth records exactly three community uploads." FALSE.** The scout records at least
five (saltb0x, mustangliu, johnlussier, trailblazeranemo, overseer66). **⚠ Our own file caused this:**
`runs/lb_ground_truth.md` (08-15 entry, L76–78) still says *"No FP8 Kaggle artifact exists; two anonymous community
uploads made 08-14 do"* — stale and wrong against the same day's scout. **Fix `lb_ground_truth.md` or this error
recurs every round.**

**F3 [SY MAJOR + MINOR] — "the serving-compat gate … has never been observed" / "v1 died at t=425 s, **before
load**" / "no wall-clock budget for the 25 GB engine exists anywhere in the record." FALSE ON ALL THREE.**
The v1 log (prereg §9; `ITERATION_LOG.md` 08-15) is second-by-second:
- **394.8 s — vLLM READY**, serving `Qwen/Qwen3.8-27B-FP8`, `max_model_len 65536`; **25.3 GB load + boot = 295 s**
- **417.3 s — stock smoke PASSED**: `Generated: 2 + 2 equals 4.`
- **~420 s / ~423 s — tool-call OK, mode=FORCED and mode=AUTO** (`parser=qwen3_coder`), `{"action":"ACTION6","x":3,"y":7}`
- **425.5 s — `RuntimeError: Q38-EVAL FATAL: MM boot probe returned empty content`** ⇒ ERROR. Total GPU cost **7 min 6 s**.

**The engine LOADED, SERVED, and EMITTED TOOL CALLS. The `vision_config` native-VLM risk — SY's own "#1 declared
failure risk" — is EMPIRICALLY DISCHARGED.** The load-time budget SY says is missing is measured at **295 s**. SY's
requested "load-only smoke" has already been run, in production, and passed.

**F4 [LA MINOR] — "Q38 v1's death at t=425 s is … never root-caused" / "if v1 died on the vision-tower load path."**
Root-caused on 08-15 and committed (`8894862`: *"INFRA DEATH (not decisive) — the ENGINE WORKED, my own MM boot probe
killed it"*). **Fair residual:** the 08-16 brief did not restate it, so the reviewer could not see it — a *briefing*
defect, not a campaign gap. Restate v1's cause in every brief while the arm is live.

**F5 [SY Q3] — "What GPU did Q38 v2 allocate, and does that SKU have hardware FP8 support?"** Answered on record:
`machine_shape=NvidiaRtxPro6000`, and v1's own GPU check at 8.1 s reported **RTX PRO 6000 Blackwell** — which has
hardware FP8. No dequant-fallback throughput risk.

**F6 [RL MINOR] — "The Q38 read has an **undefined dead zone** at 26–31 levels … the ambiguity will be resolved
post-hoc." FALSE.** Prereg §4.2 pre-registers 26–31 as **INDETERMINATE**, and §4.3 pre-commits the disposition:
*"An INDETERMINATE result therefore means exactly what it says — one seed cannot separate a +0.25…+0.50 lift from
noise — and **may not be reported as either a confirmation or a refutation**."* The thing RL asks us to prevent was
prevented before the push. (RL's Q3 — does INDETERMINATE consume slot 2? — **is** open and worth answering.)

**F7 [RL MAJOR / Q1–Q2] — "computable today … from the frozen fork's level-count distribution across **your 33 filler
draws** (you have the per-draw level counts on disk)." FALSE ON BOTH LIMBS.**
- (a) `runs/ledger.json` n=33 is the **whole-campaign draw record**, not 33 filler draws. Only the last **five** days
  are AUTO-REFILL filler; the 1.33 max (07-18) is a non-filler artifact.
- (b) The ledger carries **LB scores, not level counts**. There is no per-draw level count on disk for LB draws.
  The level-count baseline that *does* exist is m=3 (`gate_eval_v1` 18, `gate_eval_v2` 19, `tmp_pullback_duckgate_v1post`
  21; σ̂=0.141740) — already consumed to seal the read.
- ⇒ RL's Q2 ("what does the score→level mapping look like on our own 33 filler draws") **has no answer and cannot be
  made to have one.** Answer it as **NOT DERIVABLE**, which is also the honest answer to the 4/5 units complaint (C1).

**F8 [RL, LA, PS, ME — the framing "no power analysis exists / nobody computed it"] — PARTLY FALSE.** Prereg §4.3
"POWER HONESTY" pre-registered σ̂, SE(Δ), 95.3% CONFIRM power vs a doubling, 0.11% false-positive, 0.07% REFUTE-under-
doubling, **and an explicit 80%-power MDE floor of 0.468 lc/game = 11.7 levels**, together with the sentence *"badly
powered against small effects."* The panel's demand is therefore **already ~80% satisfied on the record**; what is
missing is only the LB↔lc conversion (F7, not derivable) and the δ-grid presentation (directive 1, free). **Do not
accept "we never did a power analysis" into the log — it is not true and it would misattribute the defect.**

---

## 6. VERIFIED-CORRECT (so it is not re-litigated next round)

- **cstl 1.59 → 2.70 on our own artifact family** (RL, PS) — **CORRECT.** `runs/lb_ground_truth.md` 08-14 correction,
  L120–126: cstl sat at 1.59 (08-04→08-09) inside the duck band, then 1.59 → 2.52 (08-11) → 2.70 (08-12). Family
  ceiling **≥ 2.70** stands; it refutes the ≈1.26–1.36 efficiency ceiling as a property of the *family*.
- **78 days to deadline / Nov 2** (RL, SY) — **CORRECT.** Deadline 2026-11-02.
- **"the prize is the private twin of two selected submissions"** (RL) — **CORRECT.** `state_of_campaign_2026-08-09.md` §A.
- **Ledger arithmetic** — mean 0.9424, s 0.1563, max 1.33 ≈ +2.5σ (RL), bar 1.0826, 29 days since 07-18 (PS) — all **CORRECT**.
- **SY's ~64% bar-clearance figure** — **CORRECT** as arithmetic (0.38σ ⇒ 65%), though it inherits the disputed
  δ=+0.17 premise (D-A).
- **aRc +0.74** (PS, ME) — **CORRECT** (1.17 → 1.91, ΔSub == 1).
- **ME's "s rose 0.1533 → 0.1563 is arithmetic, not a finding"** — **CORRECT and accepted**; strike the causal framing.
