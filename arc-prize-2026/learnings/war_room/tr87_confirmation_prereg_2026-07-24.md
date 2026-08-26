# tr87 fresh-stream confirmation — pre-registration (filed 2026-07-24, BEFORE push)

Per the OBJ-H ratification (learnings/panel/r17_portfolio_go_kill.md, push
discipline note: "a fresh-stream tr87 confirmation build rides a later quota
slot after the panel sees this ratification") and R19 rl-planning's slot-2
ruling (tr87 over W2 by inspection). Immutable once the push fires; amendments
by dated append only.

## Vehicle

Re-push of the EXISTING certified staging `notebooks/duckw0-eval/`
(arc3-duck-w0-continuation-eval) = **W0 seed 2**. No code changes of any kind;
w0_eval_smoke re-run 2026-07-24: **20/20 PASS**. One push (2/day budget: this +
the A17 canary = 2/2). Rationale for the vehicle: any certified eval run
produces fresh per-game event streams; reusing the proven W0 composition adds
zero build risk (fork-never-build discipline) and doubles as the W0 control-arm
seed-2 the R16 round left open.

## Pre-registered expectations (sealed before observation)

1. **PRIMARY — fixed-law transfer to fresh data:** the fresh tr87 stream's
   augmented determinism under the FIXED Schema law (phase key = n % K,
   K=2 at levels 0–4, K=4 at level 5; n = actions since level start) is
   **1.000 on all fresh visits** (the law admits no exceptions; any violated
   visit is a law break, not noise).
2. **PRIMARY — pooled certificate strengthens:** pooled augmented determinism
   (existing 9 streams incl. sentinel_eval_v1 appendix + fresh stream) stays
   ≥ 0.99 AND Wilson 95% LB ≥ 0.95 (baseline: 181 visits, det 1.0000,
   LB 0.9792 — added support can only tighten LB if det holds).
3. **SECONDARY (free, W0-hygiene replication):** 0 idle post-game-over actions
   (replicating seed-1's 49/49); descriptive levels-completed within the
   certified band (no gate — (f) counting bound stays 0.00).
4. **Mechanism observables named (per R19 methodology Q6 discipline):** the
   confirmation is evaluated on the {tr87}_p0_events.jsonl stream via the
   sealed extraction in scripts/latent_state_audit.py machinery, reproducer
   runs/schema_fixed_resolver/verify_fixed_resolvers.py — a non-score
   observable; the run's lc plays no role in the tr87 verdict.

## Decision rule (sealed)

- Expectations 1 AND 2 met → tr87 fixed-hypothesis re-entry is CONFIRMED on
  fresh data; the sealed re-entry filing (EWM channel + banking key
  re-admission) goes to the next panel round for seal.
- Expectation 1 fails (any fresh-visit law violation) → re-entry WITHDRAWN;
  tr87 returns to ALIASED-UNRESOLVED; the OBJ-H depth-lane proceeds without
  tr87 (g50t/m0r0/sk48 scope unaffected).
- Expectation 2 fails with 1 passing (possible only via pooled-set anomaly) →
  bug investigation before any claim; flag to panel.
- No promotion, no LB submission, no window: this is a build-rail
  confirmation; the depth lane's LB kill-switch (beat null10 on held-out
  games) remains sealed and untouched by this run.

## Cost

1 kernel push (2nd of 2 today), ~2.2h free Kaggle GPU build, $0 cloud.
Post-run: pull with `uvx --from kaggle==2.0.0 kaggle kernels output
canivel/arc3-duck-w0-continuation-eval -p runs/kernel_pulls/w0_eval_s2`, grep
the three W0 banners + negative warpack/ledger check, then run
verify_fixed_resolvers.py with the fresh stream added.
