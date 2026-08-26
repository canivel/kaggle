# Daily Brief — 2026-08-05 (Wednesday; weekday cadence, no panel)

## 1. Results deep-dive

### 1a. Overnight scored draw + state reconciliation
- **1.21** (frozen-fork filler, 08-05 00:07Z, API COMPLETE). **Highest draw since
  the 07-18 record 1.33** (z ≈ +1.76 vs n=21 stats) — interior, band 0.82–1.33
  holds, no watch-rule (rules watch the low side). Ledger → **n=22, mean 0.9564,
  s 0.1582**. Fourth consecutive interior draw; stationarity verdict keeps
  holding. Interpretation: nothing changed in the artifact — this is the frozen
  fork's own tail, a live reminder that single-draw deltas of ±0.25 are noise
  (sigma discipline; the draw's z would "fire" any naive one-sided uplift rule).
- LB 08-05: KOJIMA 1.86 frozen; **NEW #3 Lord Han Solo 1.65**; **gold cutoff
  (top-13) HOLDS at 1.56 — first non-rising day since 07-28**; top-10 cutoff
  1.58 with a dense 4-way pack at 1.58 (shared-public-artifact signature; the
  effective published ceiling may now sit above the 1.47 boristown anchor).
  Sweep cross-check: top-5 cutoff 1.61; our 1.33 now below #49.
- Reconciliation: the 08-04 session was terminated mid-wait on the v2 build
  agent. Patch was WRITTEN (08-04 08:41) but smoke/regression/pushes never ran;
  arc-war-kit was still at its 08-02 version and the eval kernel's last run was
  v1. Recovered today (§4). Second consecutive session truncated by agent
  waits → handoff rule: fire pushes the same session the build lands, treat
  next-day recovery as the fallback, not the plan.

### 1b. No new scored A22 evidence
v2 is a free build-time eval, measurement-only; the frozen filler fires tonight
(queued 12:38 with n=22 stats). Nothing can pass promotion gates today.

## 2. Discussions sweep (`learnings/sweeps/discussions_sweep_2026-08-05.md`)
**Quiet window BROKEN at 3 days** — 4 new threads + 2 comment-level. 2 ADAPT, 0
ADOPT:
- **Reki "Hotstack" local-eval screenshot**: anchor 1.6441 on the 25 public
  games; component ablations show "typed causal memory" +0.098 and "verified
  causal loop" +0.082. Gold-band numbers are now public at the *local-eval*
  level (Reki sits ~#365 on LB, so the letter of the absence-signal — no LB
  claim above the 1.17 duck — still holds). **Validates the A22 memory lane
  direction from an independent builder**; coverage-shaped payoff.
- **borro1980 500-run variance study**: 84.7% of score variance is binary
  level-clears; 2 games carry 65% of variance; ~4 duck passes max in 9h on the
  RTX 6000. Adopted as planning facts (variance budget, pass-count ceiling).
  Their merger offer = strategic option for the principal, not actioned.
- IGNOREs: score-poll, open-source ad, errors complaint, deleted cross-post.
- Cadence: every-other-day NOT supported — keep daily through 08-07.
- Bonus: arcprize.org/leaderboard readable via browser route now (Opus 5 High
  30.2% still the V3 frontier; no drift).

## 3. Research sweep (`learnings/sweeps/research_sweep_2026-08-05.md`)
16 relevant items (arXiv window 08-03T13Z..08-04; 08-05 not yet announced —
next sweep must cover it; API is now HTTPS-only). **0 ADOPT, 5 ADAPT — all R24
v3 candidates per sealed-spec discipline; nothing contradicts the sealed v2
spec**:
- LeanMem 2608.03463 (write-once immutable memory records), ParEvalLayer
  2608.02444 (four-state partial-eval gate readouts), ScrambleToolBench
  2608.02358 (utilization-gap forensics — agents ignore their own map),
  Screenshots-or-Tools 2608.03327 (**confirms v2's board-visibility staleness
  proxy**), ContinualSkillBench 2608.03874 (in-context ≈ skill library).
- PRO-LONG's GitHub is public — log schema recovered (typed [PLAN] blocks +
  `Action N | Level L | ...` headers), legal to imitate; → R24 with the v3 pile.
- Absences: ARC blog quiet 4th sweep (latest 07-06); no ARC-AGI-3 arXiv since
  Tycho (07-30, 6 days); banking/replay quiet 3rd sweep.

## 4. Today's development: A22 v2 SHIPPED (verify → push → RUNNING)
- **Verify gate** (`duck_eval/warpack/a22_v2_verify_report_2026-08-05.md`):
  VERDICT **GO**. Patch audited against every §2 prereg point; 2 documented
  into-compliance edits (digest newest-first + overflow break; stuck_suppressed
  counts emission opportunities). Final sha `5d8579ad…e1804f`. New
  `compaction_smoke_v2.py` **142/142 PASS**; builder regression byte-identical;
  ledger_core untouched.
- **Pushes** (`duck_eval/warpack/a22_push_report_2026-08-05.md`): arc-war-kit
  dataset version (byte-audit PASS, served == staged) → kernel
  `arc3-duck-compaction-eval` **version 2** (pull-back PASS: 8/8 cells, zero
  metadata drift) → status **RUNNING** ~12:47. ETA ~2.2 GPU-h.
- **Tonight/tomorrow read** (prereg §3): `compaction v2: ACTIVE` banner (a v1
  banner = stale dataset = VOID), COMPACTION=1, ≥1 COMPACTION event,
  **retained_reasoning_msgs=0 everywhere** (RETAIN-OFF canary, inverted vs v1).
  Then the seed-1 M1 screen vs war-eval (≥ −0.128 mean / ≥ −1.0 worst). K3 FAIL
  here ⇒ v2 paused and the A22 lane is one FAIL from DEAD.

## 5. Open questions
- (THE question) Does region-aware eviction alone recover the war-eval seed-1
  baseline? v2 build reads today/tomorrow.
- R24 (Sunday) agenda accumulating: v3 riders (pin-refuted-list, status-tagged
  claims, post-batch verifier, LeanMem write-once records, ParEvalLayer
  readouts, PRO-LONG log schema); Reki/borro1980 variance facts → does the
  2-games-carry-65% result change which games the eval weights?
- Does the 1.58 four-way pack resolve into a identifiable public artifact? If a
  ≥1.5 artifact is public, the fork-baseline question reopens (vs
  generalization-first discipline).
- Sweep cadence: hold daily through 08-07 (quiet window broke today).
