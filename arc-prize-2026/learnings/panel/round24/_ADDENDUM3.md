# R24 PACKET ADDENDUM 3 — S1/S1b runnability check, 2026-08-09

Not part of the sha256-hashed proposal body. Landed while the panel was in session.
Full plan: `duck_eval/r24_prep/s1_s1b_execution_plan_2026-08-09.md`. Two new runners were
written and dry-run/smoke-verified — `s1_threaded_replay.py`, `s1b_bank_refire_noprune.py`.
**No experiment was run, nothing was pushed, no existing file was modified.**

---

## C1. Verdict: RUNNABLE

Every asset the proposal sequences was verified by direct read/import:

| Asset | Status |
|---|---|
| exec-wm simulators (`exec_wm/sims/`) | **25 present** — not 24; bp35 has a sim but was never scale-validated |
| recorded traces (`runs/kernel_pulls/war_eval_v1/artifacts/`) | 25/25 present; `benchmark.json` holds 25 histories / 3,638 actions |
| `runs/kernel_pulls/war_eval_v1/` baseline | exists |
| `prune_trace` | `warpack_patch.py` L164 |
| 11-game prefix-splice-safe set | `grinder_design_R17_sealing.md` L319–324 |
| local engines, `arcengine` / `arc_agi` | import cleanly |

Cost: **CPU-only, <10 min total, $0, 0 pushes.** S1 ≈1–3 min; S1b ≈1–2 min (calibrated against
`determinism_audit_25.py`, which does 25 games in 13.5 s).

## C2. Four things that must be SEALED before either run, or the result is post-hoc

These are pre-registration defects in the proposal as written, not implementation problems.

1. **"Coverage" is not literally measurable on our sims.** The simulators have **no abstention
   channel** — `simulate()` always returns a concrete grid; there is no `UNKNOWN`/`-1`. The
   Tycho protocol's whole point is that a sim that knows what it doesn't know scores *low
   coverage* rather than *high-confidence-wrong* — and we cannot express that natively. The
   runner therefore emits three channels (`coverage_strict`, `coverage` via an
   identity-on-change proxy, and `accepted_match` over committed steps). **R24 must seal which
   channel the S1 gate reads, before the run.** Unsealed, the analyst picks after seeing three
   numbers.
2. **"Carrier" has no numeric definition anywhere in the proposal.** The runner defaults to the
   repo's own precedent (`accepted_match ≥ 0.92`, `coverage ≥ 0.50`, excluding
   ALIASED-UNRESOLVED) and hard-labels it **PROPOSED / NOT SEALED**. Until R24 fixes a number,
   *"the carrier set must expand beyond ~4 games"* is **unfalsifiable** — "~4" is a tilde, not
   a threshold, and `r16` names **3** carriers, not 4.
3. **The old harness had a real bug, not merely the wrong protocol.**
   `scripts/ewm_replay_dryrun.py` never reset simulator module state, so **g50t / re86 / tr87**
   (`reset_state`, `reset_phase`, `reset_step_parity`) were measured with desynced hidden
   counters. Those three games are the only ones where S1 could legitimately score *higher*
   than the historical read — which means S1's headline result is partly a **bug-fix effect,
   not a protocol effect**, and the minutes must attribute it that way.
4. **§6.4 misdescribes our own prior evidence.** "91.7% held-out `state_exact`" is **22/24
   games in Class A — a game-count, not a match rate** (true per-game range 23.0–100%). The
   underlying objection (held-out fidelity collapsed on-trajectory) **survives intact**; only
   the sentence is wrong and should be corrected in the sealed text.

## C3. S1b: gate-set ambiguity, and an honest de-rating of its value

- **Gate set conflict:** the proposal says **11 games**; the **R17 SEALED banking rule says 10**
  (tn36 excluded by its flag). The runner emits **both** verdicts; the minutes must state which
  one binds. All 11 have `lc ≥ 1` in `war_eval_v1`, so the gate is fully evaluable either way.
- **Value de-rating (stated plainly):** `determinism_audit_25.py` probe A **already ran full
  unpruned replay on 25/25 with zero divergence.** S1b's marginal contribution is running it
  *through the `_bank` fire path* with the pruned arm paired in-session. The scoping smoke
  reproduced the 2026-07-15 signature exactly (pruned aborted at step 0; unpruned survived).
  **Price S1b as cheap confirmation, not as an open question** — the proposal oversells it as a
  falsifier.

## END OF ADDENDUM 3 ##
