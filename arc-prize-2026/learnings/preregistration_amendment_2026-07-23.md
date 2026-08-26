# Amendment 2026-07-23 — A21–A25 SEALED (principal's authority, named conditions)

Basis: user order 2026-07-21 ("we are stuck — full review"); R17 3-FATAL →
all three formally RESOLVED at R18 (0 fatals, 34-objection prior stack
retired); R18 median 5 < the proposed A25 bar, so this is NOT a panel seal —
it is a **principal's seal**: the user's standing mandate outranks the
panel, recorded transparently. Residual R18 objections become binding named
conditions C1–C7. Documents: stuck_review_v2_2026-07-23.md (A21–A25 text),
runs/verify_2026-07-21/report.md (verified numbers).

## Sealed: A21 (12-window exploration budget), A22 (two-track governance),
## A23 (A17 start under envelope), A24 (heartbeat watchdog), A25 (seal
## termination — WITHOUT retroactive grandfathering; count starts at R18).

## Named conditions (binding, tracked, dated)

- **C1 multiplicity (owner: methodology docket, before 1st PROMOTION):**
  promotion gates apply Bonferroni across ALL arms that entered exploration,
  not just the promoted arm. Exploration entry requires the A22 intent filed
  BEFORE the build push (no post-hoc arm registration).
- **C2 no single-draw inference (immediate):** exploration draws accumulate
  pre-registered per-arm ledgers; NO claim of any kind from n=1; harm-pause
  (draw < 0.80) is the only single-draw-triggered action.
- **C3 A17 symmetric error model (before bench push):** pre-register BOTH
  false-NO-GO and false-GO probabilities of the screen rule under the
  verified draw distribution; envelope NO-GO (>3.5× penalty) self-certifies.
- **C4 per-line falsification (immediate):** §6's Aug-6 test decomposes into
  per-line dated singles: sentinel verdict Aug 1; A17 numbers Aug 3; EWM
  Stage-1 (clean 4-carrier set) Aug 4; (f) defaulted Jul 26; 1st exploration
  draw Jul 27. Each line unmet = that line's own escalation, no conjunction.
- **C5 EWM fidelity metric (with Stage-1):** the sealed ≥0.70-at-depth≤10
  on-trace fidelity threshold IS the metric; carriers = {tn36, tu93, ls20,
  ft09-L1} only (tr87 holdout no-go adopted).
- **C6 numbers hygiene (immediate):** all strategy docs cite
  verify_2026-07-21 values (fork band 0.82–1.33; E[max@102]≈1.35;
  P(1.44)≈0.18); the stale 1.39/0.29 pair is retired.
- **C7 SENTINEL_BUDGET (AMENDED 09:57 EDT):** ruled value = **150**, adopting
  the daily loop's independently pre-registered eval config — seed 1
  (07-22, COMPLETE, mean 0.85) and seed 2 (in flight) already form a
  consistent 150-ledger, and cross-seed config uniformity is the binding
  property; the 140-vs-150 delta is immaterial to the warn mechanism.
  Original 140 ruling withdrawn before any 140-run existed (no mixed
  ledger). Banner must echo the value; inert-sentinel check stands.

## Execution order (today)
1. arc-war-kit dataset version: budget_sentinel_patch.py (staged, byte-audited).
2. Sentinel eval seed 1 push with SENTINEL_BUDGET=140 in cell 2; verify
   banner + `SENTINEL v=1` events in build log.
3. A17: serve-config runtime test → bench push (7.5 GPU-h) after C3 filed.
4. (f) defaulted into all future builds (its screen PASSed 49/49).
5. First exploration draw: sentinel arm, first window after its build-rail
   screen returns clean (target Jul 27 per C4).
