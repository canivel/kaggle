# AMENDMENT 2 to `bench_prereg_2026-08-30.md`
**SEALED 2026-08-30, BEFORE ANY BENCH DATA LANDED** (both kernels RUNNING, ~9 h from return).
**Source: panel round 29 (Sunday strategic review), reviewer `rl-planning`, severity FATAL,
plus three MAJORs adopted verbatim.** This amendment only TIGHTENS the read.

## A2.0 THE PANEL'S FATAL, ACCEPTED IN FULL

> *"The 2v2 bench cannot license a kill decision, and the proposal's own variance data proves
> it. The minimum achievable p-value for a permutation test with n=2 per arm is 1/C(4,2) =
> 1/6 ≈ 0.167 (one-sided) — even the maximally separated outcome {A both low, B both high}
> occurs by chance under exchangeability one time in six. … There is no symmetric 'do not let
> it kill anything at n=2' guard, and killing is exactly what the handoff invites."*

**This is correct and it supersedes §3 of the base prereg.** The base prereg guarded against
the arm being over-read as a PROMOTION and left the KILL direction unguarded — an asymmetry
that is exactly the error `feedback_screen_calibration_range` and `feedback_seed_vs_own_config`
were written about, committed in the opposite direction. Sealing the fix now.

## A2.1 NEITHER BENCH SESSION MAY LICENSE A STOP/CONTINUE DECISION

**Superseding rule:** a single bench session (n=2/arm) licenses **NO decision in either
direction** — not promote, not kill, not "stop investing". Both of tonight's runs are
**descriptive only**. Their sole admissible outputs are (a) the infra certification of §4,
and (b) two (lc, final_score) pairs per arm banked toward the pooled read below.

**Numeric gate, pre-registered now** (answering the panel's *"never states the gap, the
statistic, or the error rate it accepts"*):

- **Statistic:** `levels_completed` per replica, arm B minus arm A. `final_score` is reported
  but is NOT the decision statistic (it is cap-bound and zero-inflated; see A2.4).
- **Accumulation:** no read until **n ≥ 6 per arm** on the SAME game and SAME bundle census —
  reachable in 3 zero-draw sessions.
- **KILL gate at n≥6/arm:** exact one-sided Wilcoxon rank-sum on B > A with **α = 0.05**
  (at n=6/arm the minimum attainable p is 1/C(12,6) = 1/924, so the test can actually fire),
  **AND** median(B) − median(A) ≥ 1.0 levels. Both conditions, or no kill.
- **Any other outcome is NO-VERDICT**, including a large gap that misses α.
- **A > B remains FORBIDDEN from promoting anything** at every n (base prereg §3 stands).

**If the bundle is republished mid-accumulation, the pool RESETS** — replicates from different
`composite.py` bytes are not exchangeable and may not be pooled (base prereg §5).

## A2.2 PER-ARM VALIDITY GATE — no banner in arm A ⇒ VOID, not null
Adopted verbatim from the panel's MAJOR:

> *"if `install()` silently falls back in arm A, the instrument compares placebo to placebo
> and returns a clean null that will be read as 'the stack does nothing.' … no banner in A →
> run is VOID, not a null."*

**Gate:** for EVERY arm-A replica, the artifact must carry
`TAAF_GRAFTS FEATURES={...} API_VERSION=1` with all 13 keys true, AND no
`graft install failed` / fallback line. **A missing or degraded banner on any A replica VOIDS
that session** — it is not evidence of a null and may not be banked. Symmetrically, arm B
must show the suppression actually took effect in the prompt text; if B's prompts are
indistinguishable from A's, the placebo did not fire and the session is VOID.

*(This gate is now cheap to run: the same check on the 1.62 artifact returned all 13 keys true
with no fallback, so the assertion is known to be evaluable on this artifact class.)*

## A2.3 ORDER / BUDGET CONFOUND — logged, and read as a limitation
Adopted from the panel's MAJOR. Four replicas against a 7920 s per-game wall-clock is
~8.8 h — essentially the whole session — so if `A0/B0/A1/B1` execute sequentially, **arm is
confounded with session position**, and under a binding token budget (our own
`feedback_decision_budget_binding`) suppressing graft text in arm B *frees tokens*.

**Required at readout, before any number is quoted:** per-replica execution order and
termination cause (wall-clock vs completion) extracted from the artifact paths and logs.
**Treatment definition, stated now so a null is not misread:** arm B is *"no graft information
AND the token refund from not carrying it"*; a null therefore means **"the information is
worth approximately its own token cost,"** not "the information is worthless."
Counterbalancing is not available within one session (the tag order is fixed by the rig), so
across sessions the tag order will be alternated `A0/B0/A1/B1` ↔ `B0/A0/B1/A1` and order
carried as a covariate.

## A2.4 EXTRAPOLATION RULE — stated before the run
Adopted from the panel's MAJOR (*"one game cannot kill a stack scored over multiple games"*).

**Sealed:** a result on `sb26` licenses a claim about **`sb26` only**. It may NOT be read as a
verdict on the TV28 arm, the 13-graft stack, or the config's board score. The most it can ever
support is *"on the single game carrying 50.4% of the field floor's mean_score, arm B ≥ arm A
at n≥6"* — which would be a reason to **schedule a multi-game bench**, not to stop.
The panel's point about selection is also recorded: **`m0r0` was chosen by Tennant, not by us,
and his selection criteria are UNKNOWN** — which is an independent reason the m0r0 kernel is
demoted to a rig replication (AMENDMENT 1).

## A2.5 CARRIED FORWARD, NOT ACTIONED TONIGHT (logged so they are not lost)
- **[MAJOR, panel] Zero-inflation in the draw distribution.** Byte-identical 1.82/0.00 plus our
  own 0.41 suggests a mixture of a competence distribution and a failure mode. If so,
  **config mean conflates capability with infra reliability**, and the right statistics are
  failure rate and conditional mean *separately*. This bears directly on
  `project_arc_final_selection_rule` (mean-based selection on a possibly bimodal
  distribution). **Next action: classify every near-zero draw by cause from run logs** — this
  is free, uses artifacts we already hold, and is the top candidate for tomorrow.
- **[MAJOR, panel] No exploration policy, only measurement policy.** *"A plan that can only
  lose more slowly is a plan to lose."* Every action in today's handoff is defensive. The gap
  to #10 widened 0.75 → 0.93 while we banked a draw. **This is the standing strategic gap and
  it now has a panel FATAL-adjacent flag on it; it belongs at the top of the next Sunday
  agenda, with the never-built stagnation supervisor (`feedback_arc_supervision_gap`) and
  "fork a SCORING lineage rather than a non-scoring one" as the two named candidates.**
- **[MINOR, panel] Inconsistent inference standard on single draws.** Our +2.01σ is labelled
  variance; Nader's +1.58 and Liao's +0.92 are labelled capability steps. Without their config
  variances these are not distinguishable. **Adopted: downgrade both to UNKNOWN** in the
  top-3 pattern tracking until a variance model is stated.
