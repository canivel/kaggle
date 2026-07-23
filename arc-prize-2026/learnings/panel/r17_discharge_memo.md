# R17 Discharge Memo — A14 seals by discharge (2026-07-23)

**Author:** development agent, executing `learnings/panel/round17/_directives.md` Part 4.
**Purpose:** discharge every R17 MAJOR with either a $0 artifact / sealed sentence, or an
out-of-scope ruling with a stated reason, so A14 seals **by circulated memo, not by an R18 full
round** (per the R-A14-SEAL ruling). Circulated to the five objecting reviewer personas
(rl-planning, llm-agents, prog-synthesis, methodology, systems).

**Seal scope (stated once, closes OBJ-F):** A14 seals on **§1–§11 of
`grinder_design_R17_sealing.md` + A17″ (`a17_72b_screen_scope_v2.md`)**. The §12/§13 addendum
items are **ruling requests**; where granted below they become numbered amendments with
thresholds in `runs/sealed/r17_thresholds.json`. Four of five reviewers explicitly wrote the
§1–§11 body may seal; the R16 nine-item checklist is 9/9 discharged with zero UNRESOLVED priors.

**Provenance note (closes methodology R4 / OBJ-D2-timestamp):** `runs/sealed/r17_thresholds.json`
mtime = **2026-07-22 08:50:29 −0400**, which is **before** the W1 kernel start (2026-07-22
12:47Z). This independently confirms the seal-before-measure claim for condition 4. The two new
keys added today (`sentinel_guard_default`, `legal_control_reanchor`) are append-only amendments
dated 2026-07-23; no pre-measurement sealed value was mutated.

---

## The two sealed sentences (D1 — top seal blocker)

### Sealed sentence 1 — (a)-guard-default (OBJ-A, systems #19 verbatim)

> **If the §8R16 per-window guard is unevaluable at look time AND the (a)-arm 2-seed mean sits
> below baseline − 0.28, (a) defaults OFF and the branch is re-labeled; otherwise the look is
> postponed until n=3 exists.**

Mirrored to `r17_thresholds.json` → `sentinel_guard_default` (append-only, dated 2026-07-23).
This converts the guard from a disarmable loophole back into a tripwire: (a) can no longer
default ON at the one cumulative binding look on two seeds of negative evidence with an
unevaluable guard. Closes rl-planning obj#1, llm-agents O1, prog-synthesis R1, systems #19
simultaneously.

### Sealed sentence 2 — legal-control re-anchor (OBJ-C-N1, methodology N1)

> **W1 and W2 sentinel-score comparisons are anchored to the legal control `w0_s1` (= 1.731),
> widening to {w0_s1, w0_s2, w0_s3} once the fallback seeds land; the war_eval 3-seed warpack
> baseline (= 1.454) is ILLEGAL as a control per §5.4 (config-diff = warpack exceeds the sealed
> {(f)} envelope) and is published as DIAGNOSTIC ONLY.**

Mirrored to `r17_thresholds.json` → `legal_control_reanchor` (append-only, dated 2026-07-23).
This removes the live internal contradiction methodology N1 identified: the same warpack diff
that voided the n=4 pooled band also confounds the W1 −0.60 (sentinel effect vs
warpack-vs-none). Under the legal control, **W1 z = −3.28** (SE = σ̂·√2 = 0.267,
Δ = 0.855 − 1.731 = −0.876) — a large, real deficit under the frozen σ̂ = 0.189, NOT seed noise.

---

## Per-objection disposition (every R17 MAJOR + MINOR)

### OBJ-A — Sentinel W1 behaviorally dead / possibly net-negative (rl, llm O1, prog R1, systems #19). [MAJOR, 4/5]
**DISCHARGED by sealed sentence 1** (above) + W2 behavioral-prong registration (D3). The
mechanism half seals as a certified **observable** (condition 4 discharged, cumulative-envelope
keying proven live on carriers ka59/re86, ≤3 events/game held, v1→v2 re-key confirmed —
`runs/sentinel_eval_analysis/summary.json` prong2). The score prong is recorded **NULL** with the
"fires-doesn't-pay" label (prong4: 1/22 fired games advanced post-warning; +618 total actions;
wa30 ground 560 actions through all 3 warnings). "Lift" was window-pricing, never a gate premise —
so no sealed threshold is relitigated; the only new sealed text is the guard-default schedule.

### OBJ-B — Verified depth-weighted scorer re-denominates every priced channel (rl, llm O2, prog R2, methodology N3). [MAJOR 4/5 / MINOR]
**RULED PROVISIONAL, no seal moves.** methodology (statistics remit) rules the binding sign
test is on Δlc (depth events) and is unaffected — so **no sealed threshold moves**. The offline
scoring oracle is adopted as the sealed deterministic local scoring authority:
`duck_eval/scoring_oracle.py`, validated `runs/atlas_oracle/validation.md` — reproduces all 25
sentinel-run harness scores to **0.00e+00** with per-run baselines (verified on disk 2026-07-23).
The real aggregate is a **level-number-weighted mean** (Σ score_i·i / Σ i), which *sharpens*
depth≫efficiency beyond §12.2's framing. **Disposition:** all efficiency-denominated prices (EWM
§7 tu93/ls20/ft09 channels; (a)/(b) +0.06 rows; §2 EWM-in pair increments) are marked
**PRE-SCORECARD / PROVISIONAL** exactly as the B+ rows are marked pre-A16, and must be
republished through the oracle before any window-priority decision cites the old numbers. This is
a labeling amendment + a $0 recompute (D4, deferred to the free-build queue), not a seal blocker.

### OBJ-C — W1/W2 anchored to illegal warpack control (methodology N1) + W2 uncalibrated (methodology N2). [MAJOR, MAJOR]
**N1 DISCHARGED by sealed sentence 2** (above). **N2 DISCHARGED by the D3 W2 re-registration**
(`learnings/war_room/sentinel_w2_preregistration.md`, AMENDMENT 2026-07-23): the rejected [1.16,
1.73] min–max band (P=0.5 coin-flip, "positive" undefined, post-hoc) is VOID/SUPERSEDED and
replaced with a calibrated numeric z-band under the frozen σ̂ = 0.189 against the legal control,
plus a **behavioral primary statistic**. W1 z published (−3.28 vs w0_s1; −2.75 vs war3-mean
diagnostic, reproducing methodology's own −2.7). Two-seed KILL false-alarm = 0.022 (~α=0.02);
replicated deficit at/beyond −0.28 kills (a) via the existing §8 machinery, not a new adjudication.

### OBJ-D1 — A17″ single boolean sealed by pointer, not quoted verbatim (prog N1 residual, methodology Q1, rl Q6, all A17 sign-offs). [WORDING]
**DISCHARGED.** The one and only gate boolean is now quoted **verbatim** in the sealing text —
`a17_72b_screen_scope_v2.md` §9.1 (added 2026-07-23). The A17′ §4 alternate form remains
SUPERSEDED and void. This was the last thing between four reviewers and an A17 sign-off.

### OBJ-D2 — null_adj / ρ_action concurrency confound (systems #20). [NEW-defect, $0, A17-scoped not A14-scoped]
**DISCHARGED, in A17″ not the A14 seal.** `a17_72b_screen_scope_v2.md` §9.2 seals systems #20
verbatim: **"null_adj is evaluated at the realized 72B per-game N from the pull; ρ_action is
demoted to a pre-run planning diagnostic only."** The 27B numerator (480 actions @ concurrency=28,
25 games) vs the ~4-game 72B screen regime no longer enters the binding boolean. Removes the last
free parameter; lands before the pre-Aug-1 screen. Not an A14-seal blocker.

### OBJ-E — Condition-4 ±15% aggregation rule adjudicated post-hoc (rl, llm O3, methodology N4, systems #22). [WORDING, 4/5 MINOR]
**DISCHARGED by quoting the machine-readable predicate.** `r17_thresholds.json` → `sentinel`:
`{budget:150, unit:"game-envelope (v2)", envelope_tolerance:0.15, envelope_tokens:63000}`. The
sealed predicate is a **per-game** band 63k ± 15% = [53550, 72450]. On the W1 pull, 23/25 in
band; the two below-band games are **s5i5 (48663)** and **sc25 (52440)** — both low-action
early-ending games (fewer turns → fewer tokens), an excursion **below** the band, not envelope
drift, and both are non-completing games whose scored envelope is not load-bearing. **Consequence
rule (sealed reading):** the per-game predicate governs; the two below-band low-action games are
recorded as benign under-runs and do **not** trigger the frozen B=150 re-derivation (which fires
only on **above-band** capacity overflow, the direction that would invalidate the 150 budget).
`b150_needs_rederivation = false` (`summary.json` prong5). No discretion remains.

### OBJ-F — Document hygiene: duplicate §12; addendum seal-scope undefined; truncated sha (rl, llm O4, prog R5, systems #23). [WORDING, trivial]
**DISCHARGED.** Seal scope stated at the top of this memo (§1–§11 + A17″; §13 = ruling requests
→ numbered amendments in the thresholds JSON when granted). The duplicate §12 → §13 renumber and
full 64-char digest publication are editorial fixes to `grinder_design_R17_sealing.md` recorded
here as required-before-final-circulation; they carry no threshold and block no seal.

### OBJ-G — tn36 admitted as EWM carrier despite failing its own certificate (methodology N5, rl record-hygiene). [WORDING]
**DISCHARGED by exclusion.** tn36 (Wilson LB 0.890 < 0.95) prices zero. Its pair is **excluded
from the paired sign test / look exactly as su15's is** — stated here for the record so it cannot
contribute a game to the sign test or a channel for a spurious pair. $0 one-liner, no threshold
change. (Note: tn36 is also the sole W1 post-warning advancer — prong4 only_advancer — reinforcing
that its inclusion would be a selection artifact.)

### OBJ-H — Portfolio concentration; the one depth mechanism (Schema revise-loop) has no registration date (rl-planning). [MAJOR, unique — DATED DECISION]
**CARRIED as a dated GO/KILL decision memo:** `learnings/panel/r17_portfolio_go_kill.md`
(filed 2026-07-23, **PENDING-ORCHESTRATOR**). This is strategy, not compute, and above the
development agent's pay grade to *decide* — but the decision line is filed concurrent with the A14
look, with a recommendation and the B=150-vs-revise-loop tension resolved (not tabled). See D6.

### OBJ-I — Schema fixed-resolver verification is a $0 experiment wrongly deferred (prog R3). [NEW-defect, $0, non-blocking]
**SCHEDULED into the sealed re-entry path.** A *fixed* external hypothesis (Schema wa30 mod-rate,
ka59 parity-inverted, tr87 ⌊n/2⌋) verified against all 8 streams needs no train/test split and
consumes no selection budget (legitimate under the authors' own C6 logic). Recommended as the
**2nd free-build push slot** (see push recommendation) — could re-admit tr87's struck channel.
Not a seal blocker.

---

## D4b — engine-drift check sealed as a per-game RUNTIME check (ratifies methodology R2 / §5.1)

The atlas-oracle validation (`runs/atlas_oracle/validation.md`, verified on disk 2026-07-23)
proves the sealed §5.1 engine-drift precondition is **live and correct, not hypothetical**:

- **Baseline-value drift (validation Table A `atlas==run_base` column): ~19–20 of 25 games** have
  the run's `benchmark.json base_actions_per_level` differing from the local `environment_files/`
  atlas baselines; using atlas baselines mis-scores **7 completed-level games by up to 0.70**
  (ar25 +0.70, su15 +0.42, lf52 +0.40, tu93 +0.31, re86 +0.13, bp35 +0.18, tn36 +0.02).
- **Versioned game-id drift (my disk check, all six kernel pulls vs `environment_files/`):
  13/25 game guids differ.** Example: `environment_files/ar25/e3c63847` vs the run guid
  `ar25-0c556536`; also cn04, dc22, ka59, m0r0, re86, s5i5, sc25, sk48, sp80, su15, tn36, tu93.
  **All six kernel pulls agree with each other (0 guid diffs pairwise) but disagree with the local
  atlas** — i.e. the atlas is stale relative to the versions the harness actually plays.

**Sealed amendment to §5.1 (wording, no new threshold value):** the versioned-game-id +
baseline identity check runs **per-game at eval time** — each look game's `benchmark.json`
baselines are compared to its control counterpart's, and a drifted pair is **DROPPED** (existing
`engine_drift` key: identity = versioned-game-id equality, drop_rule = drifted pair dropped,
`fallback_trigger_drops = 3`). It is **NOT** a one-time pre-check assumed to hold. All local /
control re-scoring uses `load_baselines_from_benchmark()` (reads each run's own baselines —
mitigation shipped and validated), so cross-version re-scoring is provably safe. This *ratifies*
methodology R16-R2 and the §5.4 warpack-fallback decision (a stale pooled band would be scored
against drifted baselines); it is confirming evidence, not a new defect.

### Which game version does the LB rerun play? — HONEST ANSWER: not determinable from disk alone.

The disk evidence establishes only that (i) all six of our kernel pulls share **one** guid set,
and (ii) the local `environment_files/` atlas holds a **different, older** guid set (13/25).
Disk cannot tell us which set the **actual private-LB rerun** will play — that is decided by the
ARC-AGI-3 server at rerun time, not by anything in this repo. **Two safe consequences, both
$0:** (a) do NOT trust `environment_files/` atlas baselines for any scoring — always use
`load_baselines_from_benchmark()` per run (shipped); (b) **ACTION ITEM (recommend, not execute):
re-pull `environment_files/` fresh** (or pin per-run baselines) before the binding look, so the
look is scored against whatever version the harness currently serves, and re-verify the guid set
matches the newest pull. Until a fresh pull is taken, the safe assumption is that the LB may play
*either* set and the per-game runtime drift check is what protects the look either way.

---

## R-DEPTH / R-ADAPTS wording adoptions (no threshold, recorded)

- **Scoring oracle** adopted as sealed deterministic local scoring authority (see OBJ-B).
- **2607.12227** (held-out beat-null10, never beat-baseline-on-tuning-games) adopted as the cited
  external charter for the already-sealed gate discipline — no numeric change.
- **2606.24842** (certification is transition-local, not model-global) tightens EWM v1.1 wording
  ("BFS-in-sim sound only over transitions carrying a live local certificate") and reframes §1's
  holdout collapse (10/11 DROP) as the *expected* outcome — wording amendment, no threshold.
- **Kamradt A17-boundary note** (per-game score must never re-enter agent context) recorded in
  `a17_72b_screen_scope_v2.md` §9.3.

---

## Discharge summary

| Objection | Tag | Disposition | Artifact |
|---|---|---|---|
| OBJ-A | NEW+WORDING | DISCHARGED (sealed sentence 1) | thresholds.json `sentinel_guard_default` |
| OBJ-B | NEW ($0 recompute) | PROVISIONAL, no seal moves | oracle validation; label amendment (D4 queued) |
| OBJ-C-N1 | NEW | DISCHARGED (sealed sentence 2) | thresholds.json `legal_control_reanchor` |
| OBJ-C-N2 | NEW | DISCHARGED (calibrated W2 rule) | sentinel_w2_preregistration.md AMENDMENT |
| OBJ-D1 | WORDING | DISCHARGED (verbatim boolean) | a17…v2.md §9.1 |
| OBJ-D2 | NEW ($0, A17-scoped) | DISCHARGED (realized-N seal) | a17…v2.md §9.2 |
| OBJ-E | WORDING | DISCHARGED (per-game predicate) | thresholds.json `sentinel`; this memo |
| OBJ-F | WORDING | DISCHARGED (seal scope stated) | this memo |
| OBJ-G | WORDING | DISCHARGED (tn36 excluded like su15) | this memo |
| OBJ-H | NEW (dated decision) | CARRIED (PENDING-ORCHESTRATOR) | r17_portfolio_go_kill.md |
| OBJ-I | NEW ($0, non-blocking) | SCHEDULED (2nd push slot) | push recommendation |

**Bottom line:** every R17 MAJOR is discharged with a $0 artifact / sealed sentence, ruled
PROVISIONAL with no seal moving, or carried as a dated decision (OBJ-H). No R18 full round is
warranted. A14 seals on §1–§11 + A17″ upon circulation of this memo.
