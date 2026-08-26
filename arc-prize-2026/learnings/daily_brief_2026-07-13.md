# Daily brief — 2026-07-13

**Author:** claude-fable-5 (daily loop). **Binding plan:** `path_forward_v3_2026-07-13.md` (panel round 7: 3× MAJOR-REVISION, 0 fatals, scores 6/7/7 — NOT yet approved; `winning_solution_FINAL.md` remains the fallback authority).
**Position:** draws {0.82, 0.89, 0.93, 1.02, 0.95} + sched-v1 0.90. Control mean 0.922, σ̂ 0.074. LB top: 1.61 (Tecnod8.AI); #20 = 1.38; Sep-30 top-100 projection 1.35–1.5.

---

## 1. Yesterday's results (facts)

### 1a. LB: duck+attempt-scheduler v1 draw #1 = **0.90**
Submitted 2026-07-13 02:19Z (manual, post-panel). Dead-center in the control band (z ≈ −0.3 vs frozen-duck mean 0.922). No aggregate lift visible; consistent with the panel's "EV ≈ 0" prediction. **Queue tonight already holds sched draw #2 of 2** with the pre-registered gate: promote if 2-draw mean ≥ 1.042 (requires draw2 ≥ 1.184 — above every control draw ever seen); kill if mean < 0.922 (draw2 < 0.944).

### 1b. Scheduler mechanism evidence (Kaggle build rail, 1 seed, `runs/kernel_pulls/sched_v1/`)
- Banner verified: `sched v1 ... restart_at=90 max_restarts=2 NO context injection`.
- 18 restarts on 12 games; 4 parks (sk48, tr87, m0r0, wa30) at exactly 272 actions — mechanism fired precisely as designed.
- **4 restart-recovered L1 clears** (bp35, ls20, tu93, ft09) — the mechanism *works behaviorally*.
- Offline score (validated formula, null10 baselines): sched 1.314 vs null 1.636, paired Δ −0.32, 9W/10L. Recovered clears scored ≈ 0: ls20 0.069 (null 0.62), ft09 0.54 (null 10.2). Only tu93 netted positive (+1.36) — because it cleared **L2 cleanly after the recovery**.

### 1c. THE DECISIVE FACT-CHECK IS ANSWERED — from code, not documentation (panel R7 objections D / NEW-1 / N7 resolved)
Read the actual competition scoring path (`arc_agi-0.9.6` wheel shipped in the kernel, + taaf):
1. `arc_agi/api.py:417` — competition mode **blocks a second environment per game_id** (`has_environment` → refuse). Tufa's own comment (`competition_arcade.py:66`): "competition scorecards can only create one run per game ID."
2. Therefore `EnvironmentScoreList.score = max(run.score for run in runs)` (`scorecard.py:192`) — the "best-across-attempts" reading cited in last night's submission — **is unreachable: there is exactly one run per game.**
3. Within the single run, actions **pool**: `Card.actions` is cumulative (`scorecard.py:655,673`), and `level_actions = cumulative diff at level-up` (`scorecard.py:430`). A RESET is a level_reset (taaf sets `ONLY_RESET_LEVELS=true`, `game_api.py:221`) charging +1 action, never a new run.
4. **Consequence:** attempt-1's ~90 wasted actions land in the first-cleared level's action bucket → L1-recoveries are RHAE-crushed ((base/(90k+a))²). **But levels ≥2 after a recovery count cleanly** (the diff accounting only taxes the first cleared level) — tu93 empirically confirms.

So the true semantics are neither of the two branches the panel debated (last-attempt vs best-across): they are **pooled-single-run with first-clear tax**. The v3 EV table is invalid in both columns. The lever survives only as: P(recover) × (clean L2+ value + crushed L1 value) − wall cost. Re-derivable from null10 today, free.

### 1d. Phase-1 v2: local 3-seed gate = **FAIL** — but the Kaggle screen disagrees on mean
- `runs/phase1_v2/gate_report_FINAL.md` (seeds 201–203): mean paired Δ **−0.54**, p = 0.92. Clean FAIL.
- Kernel v5 (true v2, banner verified `phase1 v2 ... min_level_actions=90 levelup_cooldown=20`): offline score **+0.19 mean but 8W/11L** (sign-negative; mean carried by outliers). 1 seed.
- Yesterday's pre-registration said: "retry look #2 spent only if screen is positive → 3 seeds." The screen is positive on mean, negative on signs, and the 3-seed rail gate already failed. **Panel must adjudicate: is the phase-1 line closed, or does the sign-vs-mean conflict + rail discrepancy (pod/local vs Kaggle) justify anything further?** Author position: closed — a 1-seed mean cannot overturn a 3-seed pre-registered FAIL; the +0.19 is exactly the variance-domination failure mode we pre-registered against.

### 1e. Housekeeping
- ITERATION_LOG was not updated by last night's session (backfilled today). Deployment-bug loop from Jul 12 formally closed: v5 banner proves true-v2 shipped.
- Kernel pushes today: 0/2 so far. GPU reserve untouched (~$68). Submission window: tonight 20:00 EDT (queue: sched draw #2).

## 2. Today's open questions (for the panel)

**Q1 (window, decide by 18:00):** Keep sched draw #2 in tonight's queue (honors pre-registration; near-certain "kill" readout; costs the window's information otherwise) — or replace with frozen-duck σ-draw #6 (control class, tightens σ̂ to df 5) given that the code fact-check (1c) already kills the v1 scheduler *as implemented* independent of any draw? v3's own language for a killed-by-fact-check lever was "zero windows spent."

**Q2 (lever redesign, free):** Under pooled-single-run semantics, is a redesigned scheduler (restart-at-90, park) still EV-positive via the clean-L2+ path? Concrete work item: replay null10 (250 runs) — among runs whose first L1 clear is late/never, what fraction of *fresh-start* clears progress to L2+ (using cross-seed exchangeability, which §E2 validated)? Also: park-only variant (no restart, just bound dead games at 90 actions) — parks cost nothing under pooled scoring and free wall/throughput; is park-only the actual surviving lever?

**Q3 (R7 majors → v4):** File path_forward_v4: (D/NEW-1/N7) answered by 1c — inline the code citations; (E/NEW-2) λ₀ calibration for Track B from null10; (NEW-4) seed-split EV for the chosen config; (N6) porting-gate track assignment; (NEW-5/H) fix panel distribution truncation (checksum). Round 8 today or tomorrow?

**Q4 (R0 leg):** v3's first free leg — vanilla-duck base gate + 1.28–1.56 fork-band audit — is still unstarted and is the only leg with no unverified assumptions. Start today in parallel?

## 3. Author's proposed plan for today
1. Backfill ITERATION_LOG (done with this brief).
2. Panel round 8 on this brief (routine: 3 reviewers — rl-planning, methodology, llm-agents).
3. DEVELOP per verdicts, default: (a) Q2 replay analysis (free, local, ~2h); (b) v4 filing with 1c code-forensics inlined; (c) Q4 fork audit start (free, public kernels).
4. Queue decision per Q1 before 18:00; ARCDailySubmit fires 20:07 EDT.
