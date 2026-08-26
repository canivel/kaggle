# Author response — Round 6 (v2 → v3)

**Document:** `learnings/path_forward_v3_2026-07-13.md` (161 lines, complete — the round-6 review copy's mid-§R1 truncation was a distribution artifact; v3 section inventory: change-log table, Thesis, Evidence base, §E2, Instruments, R0–R3, Windows/Quota ledgers, Kill criteria, Risks). New analysis script: `panel/round6/_e2_analysis.py`, run against `runs/null10/merged_null_benchmark.json`.

**Headline:** we ran every free analysis the panel demanded before re-arguing anything. Two of the panel's suspicions were *confirmed against us* (the restart EV does not survive measured discounts under last-attempt scoring; the R2 p<0.01 was pooled and is retracted), one was *refuted by the data* (the version-drift confound: 0/25 games version-unstable within null10, 16/16 flips intra-version). v3 restructures R1 around the single harness fact the EV actually turns on, and adds the two-track promotion machinery four reviewers independently demanded.

---

## The shared MAJOR: gate/EV incoherence (RL-B, PS-N1, ME-NEW-1, SY-N1)

**Accepted, and the fix is stronger than a threshold tweak.** v3 pre-registers a two-track promotion authority (§Instruments): Track A (aggregate +0.12) for candidates whose pre-registered EV ≥ +0.12 official; Track B (mechanism/event-log primary evidence + aggregate regression guard, kill at Δ̂<0) for small effects. Track B promotions are always provisional, excluded from the control class, and must re-confirm in the stack gate. A pre-registered *gate-consistency check* prints P(promote | works-as-designed) for every candidate before submission; track assignment is fixed at pre-registration. The Sep-30 projection is now P(promote)-weighted (§Targets): P-weighted draw mean ≈ 1.08–1.25, top-100 contested only in the upper quartile of outcomes — stated plainly.

Note the fix interacts with §E2: under best-across-attempts scoring the scheduler's EV is +0.13 official (not +0.055), and its Track B mechanism statistic (restart-recovered clears, predicted ≈1.5/draw, 10.1% per-event attribution error) gives P(promote | works) ≈ 0.8. Under last-attempt scoring the lever is killed before any window is spent.

## The shared MAJOR: restart EV provenance (RL-A, LA-N2, ME-NEW-2, PS-N4)

**Accepted; all four demanded analyses were run (§E2), and the headline number did not survive.**

1. **Version confound (ME-NEW-2.1): refuted.** All 25 games carry one version suffix across all 10 seeds; all 16 flips are intra-version. Flip evidence stands.
2. **Depth discount (RL-A.i, LA-Q2, ME-Q2, PS-Q6): measured**, exactly as RL specified — budget-truncated value curves of good-mode runs. disc(90)=0.365 (the assumed 0.4 was close), disc(120)=0.254, disc(150)=0.159, disc(180)=0.079. The v2 derivation's real errors: second-restart discount (0.079, not 0.16) and no FP charge.
3. **Full re-derivation with sweep (RL-C):** under **last-attempt** scoring, net EV ≈ 0 at every (trigger, cap) — FP loss (−0.24 at t=90) cancels the stuck-run gain (+0.185). **v2's +0.10 ± 0.05 is retracted.** Under **best-across-attempts** scoring, FP restarts cost nothing and EV = +0.24 local at (90, cap 2), dominating the sweep. R1 is therefore gated on a one-day, free, binary scoring-semantics fact-check (R0.4) recorded in `ITERATION_LOG.md` before any R1 work.
4. **RESET semantics (LA-N2.i, RL-Q1):** specified — fresh episode + fresh per-game analyzer context (no carried scratchpad/summaries), enforced in harness code, verified in transcript logs before the gate is scored. Our own pollution results are cited as the reason retained context would produce a correlated, degraded draw.
5. **Exchangeability (ME-NEW-2, PS-N4):** cannot be settled by replay (no restarts in null10 — LA is right that this is unmeasurable from the corpus); pre-registered live check: first scheduler windows report restart-attempt good-mode rate vs cross-seed p on the same games; failure kills the lever.

## Per-reviewer dispositions

### RL-planning (7/10)
- **A (MAJOR)** — Accepted, resolved as above; discount measured, EV retracted/conditioned, RESET specified.
- **B (MAJOR)** — Accepted; two-track fix, P(promote) printed, projections reweighted.
- **C (MINOR)** — Accepted; sweep table published (§E2.3); the EV replay covers all 250 transcripts including the 16 flip games; per-transcript simulation ships with the build.
- **M2 residual** — Accepted; p<0.01 retracted; honest worst-case ≤0.12 stated; selectivity explicitly reassigned to the conjunction (screen ∧ mechanism prediction ∧ r11l ∧ Track A window confirmation).
- **M3 residual** — On schedule; the Aug 3 BFS one-pager will include a *measured* dry-run tokens/action estimate on logged null10 stall segments (your Q6: yes).
- **M5 residual** — §Risks is present in the complete v3 (truncation artifact); the failure consequence: audit finding the top band predominantly game-agnostic → hypothesis fails → contingency windows convert to wholesale porting and the Sep objective is restated without assuming Nov recovery.
- **Questions:** (1) RESET clears the analyzer context entirely — retained context predicts p′ ≪ p per ar25/su15, which is why fresh-context is a verified build requirement. (2) 0.4 was assumed; measured curve now in §E2.2. (3) Under best-across semantics the EV is +0.13 official and the Track B mechanism statistic is the promotion path (P≈0.8); under last-attempt the lever is not submitted at all. (4) We can't justify q<0.058 from 10 seeds — we no longer claim it; honest bound 0.12. (5) See M5 above. (6) Yes, measured.

### LLM-agents (6/10)
- **N1 (MAJOR)** — Accepted; R0.2 gates the unmodified vanilla duck (2 windows) before any port; vanilla ≥ substrate → revert, re-base, re-derive all EVs. The design also separates drift (vanilla-now vs 1.21) from fork damage (vanilla-now vs substrate-now, same windows).
- **N2 (MAJOR)** — Accepted; see shared-MAJOR section. Your (ii) is adapted: the null10 corpus cannot contain restarts, so the second-attempt rate check is pre-registered on the first live scheduler windows (a dedicated pre-gate local seed would cost quota without new information beyond what window 1 yields; the lever is killed if the check fails at window 1).
- **N3 (MINOR)** — Accepted; tokens/action kill rule + joint non-inferiority apply to all R0 ports; R0 audit step 1 verifies the 1.56 leader is public/Milestone-eligible, else the target restates to the highest audited fork.
- **N4 (MINOR)** — Accepted; complete document filed with section inventory above.
- **M2 residual** — Accepted; decision table committed (hash recorded in `ITERATION_LOG.md`) *before* the first forensic transcript is read; exec-WM metric fully defined (object-level, state-changing transitions only, +15-pt margin over copy-last-frame, seed-held-out, n≥200).
- **Questions:** (1) Yes — fully cleared; enforced in harness, verified in transcripts. (2) Assumed; measured 0.365 at t=90; the revision trigger was exactly this measurement (and it flipped the conclusion via the FP term, not the discount itself). (3) Unknown among drift/regression/config — R0.2 is the experiment that separates them. (4) Yes, before first read; hash will be in `ITERATION_LOG.md`. (5) Object-level transitions, state-changing only, ≥200 held-out-by-seed transitions, +15-pt margin over identity. (6) To be verified as audit step 1; the porting target restates if not.

### Prog-synthesis (6/10)
- **N1 (MAJOR)** — Accepted; two-track fix (shared section).
- **N2 (MAJOR)** — Accepted in full: identity-baseline margin (+15 pts), state-changing transitions only, seed-held-out split, r11l transfer report, ≥50 consecutive-frame pairs for segmentation with a binomial-CI floor (≥90% point, ≥75% lower bound).
- **N3 (MAJOR)** — Accepted verbatim: only confirmed promotions enter the control class; provisional-build draws logged in a separate excluded class.
- **N4 (MINOR)** — Accepted; sensitivity table published; measured curve authoritative; live exchangeability check pre-registered.
- **Questions:** (1) The identity baseline will be computed on the same held-out set; committed margin +15 pts. (2) Held out by seed, never random within-game. (3) No — provisional promotes are excluded from the control class until stack-gate re-confirmation (your ratchet is exactly why). (4) 14% under the old gate — which is why the scheduler is no longer gated that way; Track B gives ≈0.8, and the whole check is now a pre-registered process rule for every candidate. (5) Daily: every scored run logs game_id+version suffix; blind-interval gate decisions are retroactively voided. (6) Assumed; now measured (0.365/0.254/0.159/0.079). (7) Verified at audit step 1; fallback = highest audited fork. (8) Filed complete.

### Methodology (6/10)
- **NEW-1 (MAJOR)** — Accepted; two-track fix; Sep-30 arithmetic reweighted; if the scheduler is killed (unfavorable semantics), P-weighted mean ≈ 1.08–1.18 and this is stated in §Risks-3.
- **NEW-2 (MAJOR)** — Accepted and run: (1) version check — 0/25 unstable, 16/16 intra-version flips (your confound is excluded *for this corpus*; the 15/24 July figure was a longer observation window); (2) sensitivity published; (3) no natural mid-run resets exist in the logs, so the discount is measured by budget-truncation (RL's construction) and exchangeability moves to a pre-registered live check.
- **NEW-3 (MAJOR)** — Accepted, option (a): control resets to post-promotion draws only; widened SE printed; refill via default redraws with the latency carried in the window ledger.
- **NEW-4 (MINOR)** — Accepted; era-local σ̂/df; sign-flip rule consumes the era-local CI.
- **M4 residual** — Accepted; pooled rule-of-three retracted; honest worst-case 0.12 (your 0.08–0.13 bracket confirmed at the simultaneous-UB corner); the gate is re-labeled a screen with selectivity from the conjunction.
- **Questions:** (1) Yes — identical suffixes on all 10 seeds for all 25 games; flip count unchanged at 16 on version-matched rows. (2) Assumed; measured curve §E2.2; EV at 0.2/0.4/0.6 published, and the decision now keys on scoring semantics, not the discount. (3) 14% — conceded; the scheduler no longer uses that instrument; if killed, arithmetic in §Risks-3. (4) 0.12 worst-case; the r11l holdout + mechanism prediction + Track A confirmation carry the rest; we no longer print a single-number "p" for the conjunction. (5) Under the old design ~3–6 weeks of mixed control; under v3 the control resets at promotion, so the transition-window false-promote spike you identified cannot occur (cost: temporarily wider SE, printed). (6) Finalized; filed complete.

### Systems (6/10)
- **N1 (MAJOR)** — Accepted; two-track + gate-consistency check + P(promote)-weighted projections (shared section).
- **N2 (MAJOR)** — Accepted: vLLM is in-kernel on Kaggle GPU (no external API); per-game async wall caps in-harness; top-level watchdog hard-kills and checkpoints at 10.5 h; failed-commit week → cap 1 new build, contingency cut first; R0-exit evidence includes usage-page hours for one *treated* build.
- **n1 (MINOR)** — Accepted; joint non-inferiority on both metrics + composite clears-per-wall predictor validated against null10 throughput.
- **n2 (MINOR)** — Accepted; era-local df (see ME-NEW-4).
- **M1 residual** — Calibration seed is seed 1 of 3 (total unchanged, $14–28); de-scoping table printed verbatim in §R2.
- **M2 residual** — Itemized: 2×12 h commits + 2×1.5 h smokes = 27 h; failure protocol above.
- **Questions:** (1) Seed 1 of 3; table now verbatim in §R2. (2) Itemized above; failed commit → cap 1/wk that week, contingency candidates cut first, gate schedule re-checked. (3) In-kernel vLLM; wall cap enforced by harness-side async timeouts; the 10.5 h watchdog bounds the commit even if a game hangs. (4) Track B, P≈0.8 conditional on semantics; yes — projections are now P(promote)-weighted and the unfavorable-semantics case is priced in §Risks-3. (5) null10 ran 25 games concurrently (28-way with retries) on one A40-class GPU; stall-scoped BFS replaces analyzer turns inside the same per-game wall cap, so it changes tokens/action (lower during bursts) and actions-per-wall, not concurrency or dollar cost.

## What did *not* change
Hard constraints; the R2 primary gate structure (two distinct games, ≥2/3 seeds, r11l blocking holdout); the decision table's content; reserve-unlock conditions; the priority rule and redraw policy; kill criteria carried from v2; the no-game-ID rule; fork-never-build env discipline.
