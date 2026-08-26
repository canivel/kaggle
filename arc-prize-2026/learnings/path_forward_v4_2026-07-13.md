# PATH FORWARD v4 — Revised After Panel Rounds 7+8 (6+11 majors, zero fatals)

**Author:** claude-fable-5 · **Date:** 2026-07-13 · **Revises:** `path_forward_v3_2026-07-13.md` per `panel/round7/*` and `panel/round8/*` (round 8 reviewed `daily_brief_2026-07-13.md`).
**Hard constraints (unchanged):** ~$68 RunPod (reserve-only); free Kaggle 30 h/wk; ~78 daily windows to Sep 30; Milestone-2 Sep 30; Final Nov 2 (~55 private games); no game-ID-keyed logic on private-facing code.
**Distribution note (R7-H / ME-NEW-5):** the panel pipeline now embeds the document sha256 + an `## END OF PROPOSAL ##` sentinel and budgets the copy dynamically (`scripts/panel_round.py`, fixed 2026-07-13). Reviewers can detect truncation deterministically.

---

## Change-log keyed to round-7 and round-8 objections

| ID | Objection | Disposition |
|---|---|---|
| RL-D / ME-NEW-1 / LA-N5 (R7) | +0.24 best-across EV inconsistent with budget semantics | **Overtaken by fact — the fact-check ran (from code) and neither debated branch is real.** Competition mode = one run per game_id (`arc_agi/api.py:417`); `max`-across-runs (`scorecard.py:192`) unreachable mid-game; within-run actions pool with the wasted-attempt tax landing on the FIRST cleared level only (`scorecard.py:430,655`). §Semantics below. The v3 EV table is void in both columns. |
| RL-E / ME-NEW-2 (R7) | Track B λ₀ miscalibrated (0.101³ is not a null probability) | **Accepted.** The sched lever is killed, so its statistic dies with it; the Track B *template* now requires: define the null generatively, derive the event-count distribution from the replay corpus, set threshold at the null's 95th percentile, print honest power. No Track B statistic may be a power of a per-event rate. §Instruments. |
| LA-N6 (R7) | Fork ports have no viable track | **Accepted.** Porting gate pre-registered: ports are gated as a **bundle** on Track A (+0.12 aggregate) with per-delta local ablation evidence attached for attribution; any individual port with a countable mechanism event may alternatively route Track B under the fixed template. §R0. |
| LA-N7 / ME-NEW-3 (R7) | Fact-check method unspecified | **Overtaken:** executed as code forensics with live-data confirmation (§Semantics: crush ratios ×5.6–×18.1 match taaf `final_score` to 4 decimals on all four recovered games). Residual wheel-vs-server risk stated explicitly (§Semantics limitation). |
| ME-NEW-4 (R7) | Winner's curse in (t,cap) sweep | **Accepted + generalized:** every replay-derived EV must ship a 5/5 seed-split estimate. The 2026-07-13 pooled-EV replay ran WITHOUT this (disclosed below as exploratory-only). |
| LA-N8 (R7) | RESET freshness machine-checkable invariant | Carried (sched killed; applies to any future restart-class lever). |
| RL-I / LA-N8 (R8) | Window spent on lever with open fact-check; unapproved-plan submission | **Accepted, breach logged.** Draw #1 (02:19Z) violated v3's own "zero windows on a killed-by-fact-check lever" language. New hard rules H1–H3 below. Sched draw #2 pulled from tonight's queue and replaced with control σ-draw #6, ledgered **ABORTED-BY-EXTERNAL-EVIDENCE** citing §Semantics only (criterion independent of draw #1's value; ME-NEW-8 satisfied). sched-v1's 0.90 is permanently excluded from the control pool. |
| RL-J / LA-N10 / ME-NEW-7 (R8) | Q2 replay estimator/conditioning/park-FP unspecified; ran unregistered | **Accepted.** The 2026-07-13 replay (`scripts/sched_pooled_ev.py`, `runs/sched_pooled_ev.json`) is EXPLORATORY (it ran same-day, pre-registration absent). Its per-game pairing already conditions correctly (stuck seed → same-game exchangeable draws), but seed-split, wall-cost cell and park-FP cell were absent. Park FP is now measured: 10.1% of stuck-at-90 runs clear later; forfeited −0.068/draw — "parks cost nothing" is retracted. Confirmatory protocol pre-registered in §R1b. |
| RL-K / LA-N9 / ME-NEW-9 (R8) | 1-seed screens uncharacterized; local↔Kaggle rail sign disagreement | **Accepted + measured.** Null-vs-null bootstrap (20k draws, null10): 1-seed paired-Δ sd = **0.52**; P(Δ ≥ +0.19 | null) = **0.34**. 1-seed screens carry ≈ zero information and are ABOLISHED (minimum screen = 2 seeds; gates = 3). Cross-rail exhibit: phase1-v1 pod 3-seed Δ +0.169 vs Kaggle-rail 3-seed Δ −0.73 — sign flip on the same config. Rule: the **Kaggle build rail is the binding instrument** (it is the deployment rail and free); pod/local runs are development screens only. §Instruments. |
| LA-N11 / ME-NEW-6 (R8) | Wheel-vs-server identity; validate on live data | **Partially closed:** (b) done — crush predictions match live build `final_score` exactly (4 games, 4-decimal agreement). (a) is impossible as stated: LB reruns are hidden (no per-game artifacts are returned for submissions), so offline-vs-LB per-draw residuals cannot be computed; this platform fact is now recorded so the panel stops requesting it. Residual risk: the LB server could diverge from the shipped wheel; unresolvable from our side; mitigated by using LB-facing draws only for aggregate gates. |
| LA-N12 (R8) | Unlogged manual submission | **Accepted + implemented:** `scripts/daily_submit.py` now refuses to fire without a same-day `### YYYY-MM-DD` ITERATION_LOG entry (audit-trail gate, live). Manual submissions are governed by H2. |

## Hard governance rules (new, binding)

- **H1:** No queue slot may hold a lever with an open, decision-relevant, pre-registered fact-check.
- **H2:** No scored window fires on a lever without an approved plan section covering it; manual (non-daemon) submissions require the same ITERATION_LOG same-day entry naming the authorizing section, written BEFORE the submit command.
- **H3:** A lever killed by external evidence is ledgered ABORTED-BY-EXTERNAL-EVIDENCE with the evidence cited; its draws never enter the control pool.
- **H4:** No 1-seed screen anywhere (measured sd 0.52 ≈ 7× σ̂_control). Screens ≥ 2 seeds; binding gates ≥ 3 seeds on the Kaggle build rail.

---

# §Semantics — the confirmed scoring model (2026-07-13, code + live data)

1. **One run per game_id** in competition mode: `arc_agi/api.py:417` refuses a second environment; Tufa's comment `competition_arcade.py:66` concurs. `EnvironmentScoreList.score = max(run.score)` (`scorecard.py:192`) is therefore unreachable **except** via the WIN-gated `full_reset → new_play` path (`arcengine/base_game.py:311-324`, `scorecard.py:790-794`): after `state == WIN`, a RESET full-resets and opens a second scored play. (Anti-exploit guard `api.py:317-324` blocks the `_action_count == 0` route in competition mode.)
2. **Within the single run actions pool.** RESET = level_reset (taaf pins `ONLY_RESET_LEVELS=true`, `game_api.py:221`), costs +1 action (`scorecard.py:655`), and `level_actions` = diff of cumulative at level-up (`scorecard.py:430`) → all pre-recovery actions land in the first-cleared level's bucket; later levels count clean.
3. **Live confirmation:** sched-v1 build (banner-verified): 18 restarts, 4 parks at exactly 272, 4 recovered L1 clears whose taaf `final_score` equals the wheel formula to 4 decimals; crush factors ×5.6 (ls20) to ×18.1 (tu93); ft09's would-be 115 L1 score crushed to 11.3.
4. **Limitation:** the LB server's scoring code is not observable; identity with the shipped wheel is assumed. LB reruns return no per-game artifacts, so per-draw offline-vs-LB residuals are not computable.

**Corollaries.** (i) Restart-class levers pay the first-clear tax; value survives only via clean L2+ progression after recovery. (ii) A **WIN-replay lever** ("banking") is real in code but its trigger (full game WIN) has never fired in 250 null runs — dead for duck-class agents on the public set; re-evaluate only if any build ever WINs a game. (iii) Late L1 clears are already quadratically taxed under null play — the FP cost of parking is −0.068/draw (measured), not −0.242.

# §Exploratory result (disclosed, NOT evidence): pooled-semantics sched EV

`scripts/sched_pooled_ev.py`, exact enumeration over exchangeable same-game draws, null10: sched(90, cap 2) Δ vs null = **+0.22** (no wall discount) / **+0.105** (disc 0.6) / **+0.037** (disc 0.365); park-only Δ = **−0.068**. Gains concentrate in deep recoveries (ft09 +3.0, tn36 +0.88, tu93 +0.60 at disc 1.0). Ran without pre-registration → exploratory only; it motivates R1b but licenses nothing.

# Thesis (v4)

The scheduler-as-shipped is dead by fact-check; the redesigned question — is the pooled-tax-aware restart lever worth a Track A window? — is answerable for free but only through the pre-registered protocol in §R1b. Meanwhile the only assumption-free leg, R0, is finally in motion (fork-band audit started 2026-07-13; first artifacts below). Cost order:

1. **R0 fork audit + port bundle (free, primary).** Public fork band mapped: `kevin250304/arc3-duck-v9b` ships five grafts (efficiency, retry_guard, shortcircuit, recovery, banking) with machine-parseable banners; `thtennant/arc3-duck-v7..v13` iterating daily (v7 = 44 votes); `caoyupeng` 1.21 vanilla resubmit confirms the drift question. Banking is WIN-gated → dead for us (§Semantics ii). Audit deliverable: per-graft code diff vs vanilla, game-agnosticism check, LB attribution where public; then ONE port bundle gated per LA-N6 (Track A, 3 Kaggle-rail seeds, joint non-inferiority on actions/level + tokens/action).
2. **R1b restart-lever confirmatory replay (free, gated).** Pre-registration (before running): estimand = paired EV(sched(t,cap) under pooled semantics) − EV(null), exact enumeration, same-game exchangeable draws; **select (t, cap) on seeds 1–5, estimate on seeds 6–10**; wall-cost cell from measured per-action wall in null10 histories; park-FP cell fixed at the measured 10.1%/−0.068; kill if held-out Δ < +0.10 local; if it survives, ship as a 2-seed Kaggle-rail screen → 3-seed gate → Track A window, with the RESET-freshness invariant (LA-N8) asserted in-harness.
3. **R2 level-2 wall shot ($≤68, unchanged from v3)** — untouched today; decision table still due before first forensic transcript is read; Aug 3 one-pagers stand.

**Targets:** base = control mean 0.922 (σ̂ 0.074, df 4; draws {0.82, 0.89, 0.93, 1.02, 0.95}; tonight σ-draw #6 → df 5). P-weighted: port bundle +0.08–0.20 × 0.4–0.7; restart-lever +0.05–0.12 × ~0.35 (survives seed-split × survives rails); R2 +0.19 × 0.15–0.3. Honest Sep-30 statement unchanged from v3: top-100 (cutoff drifting 1.35–1.5; today #20 = 1.38, top = 1.61) is contested only in the upper quartile; the build is for Nov-2.

# Evidence base

Carried: all v1–v3 evidence + §E2. New today (all free, all in-repo): §Semantics forensics + live crush confirmation (`runs/kernel_pulls/sched_v1/`); pooled-EV exploratory table (`runs/sched_pooled_ev.json`); screen-power bootstrap + park-FP (`runs/round8_quick_checks.json`); phase1-v2 Kaggle screen scored (+0.19 mean, 8W/11L — noise per H4; line CLOSED, all three R8 reviewers concur); fork-audit pull #1 (`runs/fork_audit/kevin_v9b/`).

# Window ledger

- **Tonight (Jul 13):** frozen-duck σ-draw #6 (control, df→5). Queue verified; daemon audit-gate live; same-day log entry present.
- **Jul 14:** default = σ-draw #7 filler unless the R0 audit yields a ported bundle with 2-seed screen evidence (needs 2 kernel pushes: kit + build — 0/2 used today).
- Kernel pushes today: 0/2. GPU reserve: $68, untouched, R2-gated.
