# R17 Directives Synthesis (2026-07-23) — the SEAL-BY-DISCHARGE ruling

**Verdicts: 0 ACCEPT, 5× MAJOR-REVISION, 0 FATAL. All scores 7.0.** Second
consecutive sealing round with the identical verdict shape (R16 was also 0/5/0,
scores 6–7). Every reviewer calls R17 "the strongest circulation of the
campaign" / "the best artifact this process has produced" and — critically —
**every one of the R16 nine-item checklist items is marked RESOLVED (several
"exemplary" / "exceeded")** by the reviewer who raised it. The held-out resolver
collapse (10/11 DROP) was executed exactly as specified and the destructive
answer accepted; the §5.4 config-diff rule fired against the authors' own
preferred n=4 band; A17″ adopted every remedy verbatim.

**The single structural fact that governs this synthesis:** *every new MAJOR in
R17 attacks the §12 ADDENDUM (filed 2026-07-23, the morning after §1–§11), not
the sealed body.* Four of five reviewers explicitly write that §1–§11 may seal
and they "do not ask to re-open it" (methodology: "the B− branch may stand
sealed"; prog-synthesis: "N2's resolution alone would justify sealing the B−
mechanism claims"; rl-planning marks all six priors RESOLVED; llm-agents marks
all four RESOLVED). The addendum introduced its own new evidence (W1 sentinel
"fires-doesn't-pay"; verified depth-weighted scorer) and then proposed authorial
rulings on it — and the panel is objecting to the *rulings*, not the body. This
is not a stalled seal; it is a seal whose body is done and whose addendum
overreached.

---

## Part 1 — Per-reviewer objection ledger (deduplicated, tagged)

Tags: **[SEALED-threshold]** = attacks a threshold already sealed pre-measurement
(illegitimate to relitigate absent new evidence); **[WORDING]** = wording /
completeness / aggregation-rule gap, dischargeable same-day at $0; **[NEW-defect]**
= genuinely new, needs a computation/measurement/dated-decision.

### Prior-objection resolution accounting (all five reviewers concur)

| R16 prior objection | R16 checklist # | R17 status (by raising reviewer) |
|---|---|---|
| Binding-look composition unsealed / EWM contradiction (rl MAJOR#1) | 1 | **RESOLVED** (rl: "the fix as specified") |
| Resolver verdicts in-sample / selection-inflated (rl MAJOR#2, prog N2) | 2 | **RESOLVED, exemplary** (both) |
| Engine-drift precondition (methodology R2) | 3 | **RESOLVED** ("in the sealing text, beyond my ask") |
| B+ rows pre-audit flag / A16 replay cost (prog N3, methodology) | 4 | **RESOLVED** (prog: "real, not cosmetic") |
| A17′ dead-code boolean / false-NO-GO / ρ / cost (C1–C3, systems 13/15/16/17) | 5 | **RESOLVED-by-pointer** (all; verbatim-quote caveat, see D1) |
| EWM Stage-1 config / tr87 / phase-augment (C5, llm N12, prog N6) | 6 | **RESOLVED** (llm: "honest collapse taken without flinching") |
| Sentinel per-attempt/per-game (C4, llm N11, prog N5) | 7 | **RESOLVED, exceeded** (re-keyed to v2 game-envelope) |
| Hash/timestamp commitment (methodology N6) | 8 | **PARTIALLY-RESOLVED, accepted** (mtime hole = D2, minor) |
| Sensitivity annexes (methodology R3–R5) | 9 | **RESOLVED**, every cell re-derived to the digit |

**No R16 prior objection is marked UNRESOLVED by any reviewer.** The checklist is
fully discharged. This is the load-bearing convergence fact (see Part 3).

### New objections, deduplicated

**OBJ-A — Sentinel W1 is behaviorally dead / possibly net-negative, and the
proposed ruling re-labels it rather than running its guard.**
Raised by: rl-planning [MAJOR], llm-agents [MAJOR O1], prog-synthesis [MAJOR R1],
systems [MAJOR #19]. (4/5 — the strongest convergence in the round.)
- **Descends from:** no R16 prior — this is genuinely NEW evidence (W1 ran
  2026-07-22 12:47Z, after the R16 panel). The *keying* defect it rode in on
  (C4/N11) is RESOLVED; the behavioral deadness is a distinct successor finding.
- **Tag: [NEW-defect] (mechanism) + [WORDING] (remedy).** The finding is real
  (1/22 fired games advanced post-warning; +618 total actions; wa30 ground 560
  actions through all 3 warnings; paired-seed Δ −0.72). But it does **not** attack
  a sealed threshold — the sentinel's sealed function was always warn-only
  telemetry; "lift" was *window pricing, never a gate premise* (authors, conceded
  by rl-planning). What the reviewers demand is a **sealed-text sequencing rule**,
  which is dischargeable same-day: state whether W1/W2 ON seeds feed the §8R16
  per-window guard and what (a)'s default state is when the guard is unevaluable
  at look time. systems #19 gives the exact fix sentence; llm-agents O1(iii) and
  rl-planning offer option (iii) = demote (a) to sidecar-logging-only. This is
  the **top sealing blocker** and it is $0.

**OBJ-B — Verified depth-weighted scorer (§12.2) re-denominates every sealed
channel price; nothing re-propagated.**
Raised by: rl-planning [MAJOR], llm-agents [MAJOR O2], prog-synthesis [MAJOR R2],
methodology [MINOR N3]. (4/5.)
- **Descends from:** no direct R16 prior; llm-agents/prog frame it as "the O1/stale-
  calibration defect in its third costume" (the *pattern* recurs, the *content* is new).
- **Tag: [NEW-defect] (a $0 recompute).** The community scorecard dissection
  (verified to 1e-9 per authors) establishes per-level efficiency caps at 115%,
  overshoot decays quadratically, unreached levels cost double. The surviving EWM
  channels (tu93/ls20 "L1 speed", ft09-L1 reliability) and the (a)/(b) +0.06 rows
  are *efficiency/trimming*-denominated — precisely what the true scorer discounts.
  **Crucially, methodology (the statistics remit) rules this does NOT move the
  seal:** "The binding look's sign test is on Δlc (depth events) and is unaffected,
  so no seal must move; but the branch pair-range inputs and the EWM/filler
  prioritization derive from the old pricing." So OBJ-B is a **PROVISIONAL-flag +
  $0 oracle recompute**, not a seal-blocker for the sign test. The authors already
  hold a free deterministic offline scoring oracle (adopted in §12.2).

**OBJ-C — W1/W2 comparison anchored to the ILLEGAL warpack control (methodology
N1) + W2 pre-registration is statistically uncalibrated (methodology N2).**
Raised by: methodology [MAJOR N1, MAJOR N2]; overlaps rl-planning obj#1, systems #19,
llm-agents O1 tail ("mean inside 1.16–1.73 … cannot fail").
- **Descends from:** N1 is a direct *internal-consistency* consequence of methodology's
  own R16-R2 (engine-drift) as sealed in §5.4 — the same warpack diff that voided the
  n=4 band also confounds the W1 −0.60 (sentinel effect vs warpack-vs-none conflated).
  N2 is new.
- **Tag: [NEW-defect] (both).** N1 is a genuine live contradiction the authors cannot
  hold both sides of, and it is $0 to fix: re-anchor W1/W2 to the legal control
  (w0_s1 now; {w0_s1..s3} once fallback seeds land), publish the warpack comparison as
  diagnostic only. N2 is load-bearing and correct: the W2 band [1.16,1.73] is the
  min–max of n=3 draws → a true-null draw lands inside with prob (n−1)/(n+1)=0.5 (a
  coin-flip acceptance region), "positive" is numerically undefined, and the band was
  chosen *after* seeing W1's 0.855. methodology also flags a real internal
  inconsistency: the authors calibrate every guard at σ̂=0.189 yet dismiss the −0.60
  draw by appealing to a seed spread implying σ≈0.29 — the −0.60 is z≈−2.7 (a ~0.4%
  event) under the frozen σ̂. **This directly governs the W2 pre-registration file**
  (`sentinel_w2_preregistration.md`), which as drafted encodes exactly the uncalibrated
  band methodology rejects — see Ruling R-W2 below.

**OBJ-D1 — A17″ single boolean sealed by pointer, not quoted verbatim.**
Raised by: prog-synthesis [MAJOR N1 residual], methodology [Q1], llm-agents/systems
[caveat on all A17 sign-offs], rl-planning [Q6].
- **Descends from:** R16 C1 (dead-code boolean), marked RESOLVED-on-the-fix by all.
- **Tag: [WORDING].** Every reviewer says everything they *can* see is consistent with
  resolution; they withhold only the verbatim check because `a17_72b_screen_scope_v2.md`
  is sealed by hash+pointer and absent from their packet. Dischargeable in one sentence:
  quote the single gate boolean verbatim in the sealing text. **A gate that last round
  sealed with two published forms cannot this round seal by reference** (prog-synthesis) —
  this is correct and cheap.

**OBJ-D2 — null_adj / ρ_action concurrency confound (systems #20).**
Raised by: systems [MAJOR #20] (sole competent reviewer on prefill/decode/concurrency).
- **Descends from:** R16 C3/systems#16 (ρ measures wrong ratio), marked RESOLVED-as-
  directed (ρ:=ρ_action adopted). This is a NEW second-order defect *in the adopted fix*.
- **Tag: [NEW-defect], but $0 and A17-scoped (not A14-scoped).** The 27B numerator (480
  actions) was measured at concurrency=28 across 25 games; the 72B screen serves ~4 games
  — a different contention regime, so the pooled actions/s ratio measures serving-stack
  utilization, not model speed. Direction is conservative (biases toward NO-GO) but
  mismeasured. Fix is *cheaper than the defect*: seal "null_adj evaluated at the 72B's
  realized per-game action counts from the pull; ρ_action demoted to pre-run planning
  diagnostic." Zero GPU-h, removes the last free parameter from the binding boolean.
  **Must land in the A17″ amendment before the screen runs — not an A14-seal blocker.**

**OBJ-E — Condition-4 ±15% aggregation rule adjudicated post-hoc (per-game vs mean).**
Raised by: rl-planning [MINOR], llm-agents [MINOR O3], methodology [MINOR N4],
systems [MINOR #22]. (4/5, all MINOR.)
- **Descends from:** R16 C4/methodology-R6 (scored-envelope check), RESOLVED as
  architecture; the aggregation ambiguity is the "new wrinkle" (methodology's word).
- **Tag: [WORDING].** Sealed sentence says "the tokens/game grep must land within ±15%";
  §12.1 reports mean 64.3k, 23/25 in band, declares PASS. Per-game reading → 2 games fail
  and the frozen re-derivation ("no other discretion") should fire; mean reading → pass.
  Choosing after seeing the data is the discretion the seal was meant to remove. Fix:
  quote the machine-readable predicate from `r17_thresholds.json` and state the outlier
  consequence. $0.

**OBJ-F — Document hygiene: duplicate §12; addendum seal-scope undefined; truncated
sha prefix.**
Raised by: rl-planning [MINOR], llm-agents [MINOR O4], prog-synthesis [MINOR R5],
systems [MINOR #23].
- **Tag: [WORDING], trivial.** Renumber addendum → §13; state in one sentence "A14 seals
  on §1–§11 + A17″; §13.1–13.3 are ruling requests that, if granted, become numbered
  amendments with thresholds in r17_thresholds.json"; publish full 64-char digests.

**OBJ-G — tn36 admitted as EWM carrier despite failing its own certificate (methodology
N5).**
Raised by: methodology [MINOR N5], rl-planning [MINOR record-hygiene] (su15-in-carrier-
list variant).
- **Tag: [WORDING].** tn36 (Wilson LB 0.890 < 0.95) prices zero but still contributes a
  game to the paired sign test / a channel for a spurious pair. Clean move: exclude it,
  or state its pair is excluded from the look exactly as su15's is. $0 one-liner.

**OBJ-H — Portfolio concentration: everything prices ≈0, the one depth mechanism
(Schema revise-loop) has no registration date (rl-planning [MAJOR]).**
Raised by: rl-planning [MAJOR] (unique).
- **Tag: [NEW-defect] requiring a DATED DECISION, not compute.** After the holdout
  collapse + W1 null + depth scorer: B− is a sealed near-certain FAIL (0.04), EWM is +0.04
  central with no new-clear channel, banking's best carrier sc25 retains 36–43 live actions
  post-replay (which a depth-dominated scorer cannot plausibly convert to a level). The
  only depth-targeting line is A17″. The Schema-class revise loop (283–412 actions/game,
  3–8× B=150, closed g50t/m0r0/sk48 first-line) is deferred with no date. rl-planning:
  "as a plan to reach 1.44+ by Nov 2 it currently rests on A17″ alone." Fix = file the
  contract-v1.1 / budget-regime GO-or-KILL decision on a **dated line concurrent with the
  A14 look**, with the B=150-vs-revise-loop tension resolved not tabled. This is the one
  new MAJOR that is strategy, not hygiene — and it is right.

**OBJ-I — Schema fixed-resolver verification is a $0 experiment wrongly deferred
(prog-synthesis [MINOR R3]).**
- **Tag: [NEW-defect], $0, non-seal-blocking.** A *fixed* external hypothesis (Schema's
  wa30 mod-rate, ka59 parity-inverted, tr87 ⌊n/2⌋) verified against all 8 streams needs no
  train/test split and consumes no selection budget — legitimate under the authors' own C6
  logic. Deferring it leaves tr87's struck channel hostage to a free test. Schedule it into
  the sealed re-entry path. Good cheap upside; not a blocker.

---

## Part 2 — Rulings per open question

**R-SENTINEL-SEAL (does the sentinel mechanism-half seal with score prong NULL?
§12.1 / addendum 12.1 proposal): APPROVED-WITH-CONDITIONS.**
YES — seal the sentinel as a certified **observable**: mechanism half sealed
(condition 4 discharged, cumulative-envelope keying proven live, ≤3 events/game
held, v1→v2 re-key confirmed on carriers), score prong recorded NULL with the
"fires-doesn't-pay" label. This is uncontested by the panel *as a factual
record*. **But the seal cannot pass with (a) still able to default ON at the
binding look un-adjudicated (OBJ-A/OBJ-C).** Two conditions, both $0 same-day:
(i) seal one sentence fixing (a)'s guard schedule and its default state when the
guard is unevaluable — adopt **systems #19's exact fix**: *"if the §8R16 guard is
unevaluable at look time AND the (a)-arm 2-seed mean sits below baseline − 0.28,
(a) defaults OFF and the branch is re-labeled; otherwise the look is postponed
until n=3 exists."* This closes rl-planning obj#1, llm-agents O1, prog R1, systems
#19 simultaneously. (ii) re-anchor the W1 comparison to the legal control w0_s1
(methodology N1). The "sentinel-as-observable" ruling itself is ratified; the
disarmable-guard loophole is not.

**R-W2 (is W2-as-confirmatory-null ratified? `sentinel_w2_preregistration.md`):
NOT RATIFIED AS DRAFTED — ratify only after re-writing the decision rule.**
The *concept* (a $0 confirmatory-null free build, byte-identical config, seed 2)
is approved by all. The *decision rule as filed is rejected by methodology N2 and
is correct to reject*: the band [1.16, 1.73] is the min–max of the n=3 prior
draws, so a true-null draw lands inside with P=0.5 (coin-flip acceptance),
"positive" carries no number, and the band was chosen after observing W1. Before
W2 pushes, amend the pre-registration file (new dated section — it is immutable
per its own header, so this is an append not an edit) to seal: (a) the W1 z under
the frozen σ̂=0.189 against the **legal control w0_s1** (not the warpack baseline);
(b) a **numeric** W2 rule as a z-band with stated false-alarm and miss rates under
the frozen σ̂; (c) the pre-committed consequence that a **replicated deficit at or
beyond the −0.28 guard boundary kills (a)** via the existing §8 machinery (not a
new ad-hoc adjudication); (d) make the **behavioral prong** (post-first-warning
strategy-switch rate vs baseline, computable from existing W1 transcripts at $0)
the registered W2 statistic, not the uninformative mean (llm-agents O1). W2 may
push *today* once these four amendments land — they are all writing, no compute.

**R-DEPTH-REPOINT (does the §12.2 depth≫efficiency re-pointing get adopted into
the sealed objectives?): ADOPTED AS A PROVISIONAL RE-PRICING, sign test unchanged.**
methodology (the statistics remit) rules the binding sign test on Δlc is
unaffected, so **no sealed threshold moves**. Adopt the offline scoring oracle
(`duck_eval/scoring_oracle.py`, validated `runs/atlas_oracle/validation.md`,
reproduces all 25 sentinel-run harness scores to **0.00e+00** — verified on disk
2026-07-23) as the sealed deterministic local scoring authority (zero cloud
spend). The oracle confirms the §12.2 dissection concretely: the real aggregate
is a **level-number-weighted mean (Σ score_i·i / Σ i)** — late levels weigh most,
which *sharpens* the depth≫efficiency finding beyond §12.2's framing. Mark every
efficiency-denominated price (EWM §7 tu93/ls20/ft09 channels, (a)/(b) +0.06 rows,
§2 EWM-in pair increments) **PRE-SCORECARD/PROVISIONAL exactly as the B+ rows are
marked pre-A16** (llm-agents O2's explicit alternative), and republish them
through the oracle before any window-priority decision cites the old numbers.
This is a $0 recompute + a labeling amendment; it does not require a panel look
because it moves no seal.

**R-DRIFT-LIVE (the engine-drift precondition, §5.1 — is it hypothetical?):
NO — IT WOULD TRIP TODAY ON 20/25 GAMES. The sealed drift rule is validated as
load-bearing and must run per-game at eval time, not be assumed.**
The atlas-oracle validation (verified on disk) surfaces that **baseline/game-
version drift is REAL and CURRENT**: for 20 of 25 games the run baselines
(benchmark.json base_actions_per_level the harness actually used) differ from the
local `environment_files/` atlas baselines (`atlas==run_base = False` for
20/25 in Table A; using atlas baselines mis-scores 7 completed-level games by up
to 0.70). This directly ratifies methodology R16-R2 / §5.1's sealed precondition
(versioned-game-id equality, drop-neither-side rule, `fallback_trigger_drops=3`):
**the identity check is not a theoretical guard — it would fire on 20/25 games
today if a control run and the binding look straddled the version bump.** This
also independently *re-confirms* the §5.4 warpack-fallback decision to run fresh
W0 seeds on the then-current engines (a stale pooled band would be scored against
drifted baselines). Two consequences for the sealing text, both $0: (i) the §5.1
drift check must be sealed as a **per-game runtime check at eval time** (compare
each look game's benchmark.json baselines to its control counterpart's; drop on
mismatch) — NOT a one-time pre-check assumed to hold; (ii) all local re-scoring
must use `score_run` / `load_baselines_from_benchmark()` (reads each run's own
benchmark.json baselines — mitigation already shipped and validated), so
cross-version re-scoring is provably safe. This is not a new defect against the
seal; it is *confirming evidence that the sealed drift rule is correct and
necessary* — fold it into the discharge memo as ratification of methodology's
R2, and add the per-game-runtime-check wording.

**R-ADAPTS (do the two research ADAPTs — 2607.12227 held-out-transfer, 2606.24842
transition-local certification — get adopted into the sealed objectives?):
ADOPTED AS WORDING, not thresholds.** No reviewer objected to either; both
*strengthen* the existing gate discipline. (i) 2607.12227 (held-out beat-null10,
never beat-baseline-on-tuning-games) is the external charter for the gate
discipline already sealed — adopt as a cited rationale, no numeric change.
(ii) 2606.24842 (certification is transition-local, not model-global) tightens
EWM v1.1 wording ("BFS-in-sim sound only over transitions carrying a live local
certificate") and reframes §1's holdout collapse as the *expected* outcome — adopt
as the proposed **wording** amendment to the EWM measurement config (authors' own
framing). The Kamradt A17-boundary note (per-game score must never re-enter agent
context) is recorded in the A17″ amendment text. All three are $0 text.

**R-A14-SEAL (does A14 seal at an R18, and what does R18 require?):
A14 SEALS BY DISCHARGE MEMO — NO R18 FULL ROUND.** See Part 3 for the reasoning.
A14 seals when the R17 discharge memo (below) is circulated to the objecting
reviewer personas showing every R17 MAJOR either discharged with a $0
artifact/sealed-sentence or ruled out-of-scope with a stated reason. **No new
full adversarial round is warranted** — the R16 checklist is 9/9 discharged, no
prior objection is UNRESOLVED, and every R17 MAJOR is either a §13-addendum
ruling (not a body defect), a $0 wording gap, or a dated-decision request. The
one exception requiring genuine escalation is OBJ-H (portfolio concentration),
which needs a **dated GO/KILL decision** but not a review round.

---

## Part 3 — Convergence check (honest)

**The majors are NOT genuinely blocking new body defects. The seal is being held
by an addendum the authors themselves bolted on, plus the adversarial prompt's
structural bar on ACCEPT.** Here is the evidence, stated plainly:

1. **The R16 checklist is 9/9 discharged with zero UNRESOLVED priors.** Every
   reviewer marked every one of their own R16 objections RESOLVED (several
   "exemplary/exceeded"). In a normal review process this is an ACCEPT.

2. **Every R17 MAJOR lands on the §12/§13 ADDENDUM or on A17 (a separate gate),
   not on the §1–§11 body being sealed.** Four of five reviewers explicitly write
   that the body may seal and they do not ask to re-open it (methodology and
   prog-synthesis in nearly identical language). The addendum was filed
   2026-07-23 — *after* the sealing body — and introduced its own fresh evidence
   (W1, scorer) plus authorial rulings on it. The panel is objecting to the
   authors' handling of their own new evidence, which is a good-faith objection
   but is **not the body failing review**.

3. **R17 majors vs R16 majors — is this new load-bearing issues or ratcheting?**
   Mixed, leaning ratchet-plus-one:
   - **New & load-bearing (genuine):** OBJ-A (sentinel dead behavior — real new
     measurement), OBJ-C/N2 (W2 mis-calibration — real statistical defect),
     OBJ-H (portfolio concentration — real strategic finding), OBJ-D2 (ρ_action
     concurrency confound — real second-order defect in the R16 fix).
   - **Ratchet / wording (the same defect re-costumed or a completeness gap on an
     already-resolved item):** OBJ-B is explicitly called "the stale-calibration
     defect in its THIRD costume" by two reviewers — i.e. the *pattern* recurs but
     methodology rules it moves no seal; OBJ-D1 (verbatim-quote the boolean —
     pure completeness); OBJ-E (aggregation wording); OBJ-F (renumber §12);
     OBJ-G (tn36 wording). These are dischargeable in an afternoon of editing and
     are **not blocking defects in substance** — they are the adversarial prompt's
     "a clean pass is a review failure" reflex finding the last reachable seams.
   - **Net:** ~2 genuinely new load-bearing statistical/strategic issues (OBJ-C/N2,
     OBJ-H), ~2 new measurements the authors themselves surfaced and mishandled
     (OBJ-A, OBJ-B), and ~5 wording/completeness ratchets. The MAJOR *count* (14
     from systems, 5 from prog/llm) is inflated by prior-resolution bookkeeping —
     the NEW blocking substance is 2 sentences of sealed text (OBJ-A + OBJ-C-N1)
     plus 1 dated decision (OBJ-H).

   - **New CONFIRMING evidence (strengthens the seal, not a blocker):** the atlas-
     oracle validation (verified on disk 2026-07-23) proves the sealed §5.1 engine-drift
     precondition is live and correct — it would trip 20/25 games today — and validates
     the §12.2 depth-scorer adoption with an exact (0.00e+00) level-number-weighted
     oracle. This *ratifies* two reviewer-demanded seals (methodology R2, the §12.2
     re-pointing) rather than opening a new one.

4. **Is the adversarial prompt structurally preventing ACCEPT?** Partly, yes. Two
   consecutive rounds of uniform 7.0/MAJOR-REVISION/0-FATAL with a fully-discharged
   checklist is the signature of a rubric where "clean pass = reviewer failure"
   floors every honest reviewer at MAJOR-REVISION regardless of substance. The
   reviewers are behaving correctly *within that rubric* — they each found a real
   seam — but the seams are now in an addendum the seal need not even contain.

**RECOMMENDATION — adopt a seal-by-discharge rule (and this synthesis invokes it):**
A14 seals when every R17 MAJOR is either (a) discharged with an artifact or a
sealed sentence, or (b) ruled out-of-scope with a stated reason — circulated as a
**written discharge memo to the objecting reviewer personas**, with **no R18 full
round**. A third uniform-7.0 round would almost certainly re-discover the same
class of seam (some new same-day evidence, mishandled, re-costumed) and burn
another cycle while the LB wall erodes (1.44 → dense 1.44–1.60; our 1.33 slid
#44→#45). The discharge memo must: quote the A17″ boolean verbatim (OBJ-D1); seal
the two sentences (OBJ-A guard-default + OBJ-C-N1 legal-control re-anchor);
append the re-written W2 rule (R-W2); mark the depth-scorer re-pricing PROVISIONAL
(OBJ-B); fix the four wording items (OBJ-E/F/G + OBJ-D2 into A17″); and carry
OBJ-H as a dated GO/KILL decision line concurrent with the A14 look.

---

## Part 4 — Prioritized directives for TODAY ($0-first; max 2 kernel pushes, one → W2)

Ordered by what unblocks the seal.

**D1 [$0, ~1h, UNBLOCKS SEAL] — Write the two sealed sentences + the discharge memo.**
   (i) (a)-guard-default sentence (systems #19 verbatim): (a) defaults OFF if the
   guard is unevaluable at look time and the 2-seed mean < baseline − 0.28, else the
   look postpones to n=3. (ii) Re-anchor W1/W2 to the legal control w0_s1 (methodology
   N1); publish the warpack comparison as diagnostic only. Mirror both into
   `runs/sealed/r17_thresholds.json` as amendments. This is the top blocker.

**D2 [$0, ~30min, UNBLOCKS A17] — Quote the A17″ gate boolean verbatim (OBJ-D1) and
   seal null_adj-at-realized-72B-actions (OBJ-D2/systems #20) in `a17_72b_screen_scope_v2.md`.**
   Both are one-sentence amendments; the boolean quote is the last thing between four
   reviewers and an A17 sign-off, and #20 removes the last free parameter from the
   binding boolean before the pre-Aug-1 screen.

**D3 [$0, ~1h, UNBLOCKS W2 PUSH] — Re-write `sentinel_w2_preregistration.md` (append a
   dated section) with a calibrated numeric rule (R-W2):** W1 z under frozen σ̂ vs w0_s1;
   a z-band W2 rule with stated false-alarm/miss rates; the replicated-deficit-kills-(a)
   consequence; and the **behavioral prong** (post-warning strategy-switch rate from
   existing W1 transcripts) as the registered statistic. W2 cannot push until this lands.

**D4 [$0, ~1h] — Run the VALIDATED offline scoring oracle (`duck_eval/scoring_oracle.py`,
   0.00e+00 reproduction) over the priced channels (OBJ-B) and republish the §7 EWM
   per-subset prices + §2 (a)/(b)/EWM-in pair rows through the verified level-number-
   weighted (Σ score_i·i / Σ i) scorer, marked PRE-SCORECARD/PROVISIONAL.** No seal
   moves (methodology confirms the sign test is on Δlc); labeling + recompute pass. The
   level-number weighting sharpens depth≫efficiency — efficiency channels on early
   L1s are worth even less than §12.2 implied.

**D4b [$0, ~30min, RATIFIES METHODOLOGY R2 / §5.1] — Seal the engine-drift check as a
   per-game RUNTIME check + verify the LB-rerun version.** Fold the atlas-oracle finding
   into the discharge memo as confirmation the §5.1 drift precondition is live (would trip
   20/25 games today). Amend §5.1 wording: the versioned-id + baseline identity check runs
   per-game at eval time (each look game's benchmark.json baselines vs its control
   counterpart's; drop on mismatch), NOT assumed once. Seal that all local/control
   re-scoring uses `load_baselines_from_benchmark()` (per-run baselines — shipped +
   validated). ACTION ITEM outside the memo: **verify which game version the actual LB
   rerun plays** (environment_files/ currently holds different guids, e.g. ar25-e3c63847,
   than the 07-22 run's ar25-0c556536) — re-pull environment_files or pin per-run
   baselines so the binding look is scored against the version the LB will actually use.

**D5 [$0, ~15min each] — Wording sweep:** renumber addendum §13; state seal scope
   (§1–§11 + A17″; §13 = ruling requests); full 64-char digests; condition-4 aggregation
   rule quoted from thresholds JSON with outlier consequence (OBJ-E); tn36 pair
   excluded-from-look like su15 (OBJ-G). Note for the record: `runs/sealed/r17_thresholds.json`
   mtime is 2026-07-22 08:50 — **before** the W1 kernel start (12:47Z), which independently
   confirms the seal-before-measure claim for condition 4 (closes methodology R4/OBJ-D2-timestamp).

**D6 [DATED DECISION, not compute — do TODAY] — File the contract-v1.1 / budget-regime
   GO-or-KILL line (OBJ-H)** concurrent with the A14 look, resolving the B=150-vs-Schema-
   revise-loop tension rather than tabling it. This is the one new MAJOR that is strategy;
   rl-planning is right that the plan-to-1.44 currently rests on A17″ alone.

**Kernel pushes (2 available):**
- **Push 1 → W2** (seed-2 confirmatory-null free build) — ONLY after D3 lands the
  calibrated rule; otherwise the push burns a slot on an uninterpretable band.
- **Push 2 → HOLD** for the Schema fixed-resolver verification (OBJ-I, prog R3) if D1–D6
  finish with time left; it is a $0-value-upside free build that could re-admit tr87's
  struck channel via the sealed re-entry path. Do NOT spend it on anything A14-seal-
  related — the seal is discharged by memo, not by compute.

**Commit the discharge memo + amended thresholds JSON + W2 re-registration together**
so the §9 hash-commitment chain gets its external timestamp (methodology Q2, systems #23).

---

## Bottom line

**A14 seals by discharge memo, not R18.** The R16 checklist is 9/9 discharged with
zero UNRESOLVED priors and four of five reviewers say the §1–§11 body may seal. Every
R17 MAJOR lands on the self-added §13 addendum, on A17 (a separate gate), or on wording
— the genuinely new blocking substance is **two sealed sentences** (the (a)-guard-default
rule and the legal-control re-anchor) plus **one calibrated W2 rule** and **one dated
GO/KILL portfolio decision (OBJ-H)**. Two consecutive uniform-7.0/0-fatal rounds on a
fully-discharged checklist is the adversarial rubric flooring honest reviewers at
MAJOR-REVISION, not the body failing review; a third full round would re-discover the
same class of seam while the LB wall erodes. Discharge the four items above by memo today,
push W2 only after its rule is calibrated, and hold the second push for the free
Schema fixed-resolver verification.
