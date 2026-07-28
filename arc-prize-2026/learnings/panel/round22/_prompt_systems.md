You are Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, quota economics; kills plans that don't fit the compute envelope).

You are reviewer #5 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026).

GROUND-TRUTH LEADERBOARD STATE (refreshed 2026-07-28 from the live Kaggle API; the
draw-by-draw submission ledger is at runs/lb_ground_truth.md; treat THESE numbers as canonical —
any different numbers you may remember from earlier rounds were a stale briefing):
# LB ground truth — refreshed 2026-07-28 (live Kaggle API: submissions + full leaderboard CSV)

Account: canivel (Danilo Canivel, d.canivel@gmail.com). Competition:
arc-prize-2026-arc-agi-3. Verification command:
`uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`.

- OUR BEST (public LB): **1.33** (frozen-fork filler draw, 2026-07-18). Current rank
  **#51** (leaderboard CSV pull 07-28: 47 teams strictly above, 6 tied at 1.33 spanning
  ranks 48–53).
- LEADER: YUTO KOJIMA **1.86**. #2 Tecnod8.AI 1.61, #3 DhanaLakshmiMalla 1.60,
  #4 ippeiogawa / Yuchen20 1.58. Gold cutoff ≈ **1.49** (top 13; #13–14 both 1.49,
  #15 = 1.48). Dense band 1.46–1.61 unchanged (boristown's public 1.47 seeding).
- External context: Claude Opus 5 posted 30.2% on the ARC-AGI-3 benchmark (arcprize.org,
  Jul 24) via API at High reasoning effort — different regime (unconstrained API vs
  Kaggle quantized/time-limited local), no artifact to lift; directional support for
  capability-over-harness.
- The "best 0.43 / leader 1.56" figures in pre-R19 briefings were a STALE HARDCODED
  TEMPLATE (May-era), root-caused and fixed 2026-07-24 (panel_round.py now reads this
  file). Reconciliation: 0.43 was the team's best in early May (forge-era agents);
  the frozen duck fork lifted the floor to the 0.82–1.33 band from 2026-07-05 on.

## Draw-by-draw scored ledger (all API-verified)

Frozen-fork control (n=14): 0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
1.05, 0.84, 1.02, 0.90 → mean 0.9686, s ≈ 0.1384. War arm (n=5, CLOSED per A9): 0.91,
1.08, 0.88, 1.05, 0.76. Sentinel exploration arm (n=1, HARM-PAUSED 07-24, SHELVED by
disposition memo; R21 directive #4 asks for a pre-registered un-shelve rule): 0.71.

Recent tail (newest first): 0.90 filler (07-28) · 1.02 filler (07-27) · 0.84 filler
(07-26) · 1.05 filler (07-25) · 0.71 sentinel (07-24) · 0.82 filler (07-23) ·
1.14 filler (07-22).

Refresh 2026-07-28 (live API `competitions submissions` + full leaderboard CSV
2026-07-28T11:24Z): incorporated the 07-28 00:07Z frozen draw **0.90** (API status
COMPLETE, description "frozen-fork filler ... n=13 after 07-27 draw 1.02"). Stats
recomputed numerically: n=14, mean 0.9686, s 0.1384 (was n=13 / 0.974 / 0.143 —
mean −0.005, s tightened). 0.90 is interior (z ≈ −0.53 vs prior stats): no band
change, no drift signal, no trigger. Leaderboard cross-check: our best 1.33 rank #51
(47 strictly above, 6 tied at 1.33), leader KOJIMA 1.86, gold cutoff 1.49 (#13–14 at
1.49, #15 = 1.48) — all unchanged from 07-27.

Refresh 2026-07-27 (live API `competitions submissions` + full leaderboard CSV):
incorporated the 07-26 (0.84) and 07-27 (1.02) frozen draws that previously existed
only in briefs (stale-at-n=11 flagged by panel R21 directive #3). Both cross-checked
against runs/submission_log.jsonl (ok=true, arc3-duck-repro v3, trusted-fork
preflight). Recomputed stats agree exactly with
learnings/artifacts/result_deepdive_2026-07-27.md (n=13, mean ≈ 0.974, s ≈ 0.143) —
no discrepancy.

External anchors: byte-identical public forks of the same duck artifact family have
drawn 1.39 (zoli800) and 1.47 (boristown agi-duck-harness-fast-eval, whose only real
functional diff is a vLLM readiness gate — see
learnings/war_room/fork_diff_boristown_2026-07-24.md). Artifact tail ≥ 1.47 confirmed.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

[MAJOR] The boristown readiness-gate diff is the highest-EV systems fix on the table and the brief allocates it zero action — Your own fork-diff (`learnings/war_room/fork_diff_boristown_2026-07-24.md`) says a byte-identical artifact whose *only functional diff is a vLLM readiness gate* drew 1.47 — above gold cutoff — while our fork's 13-draw max is 1.33 and mean 0.974. The mechanistic story is obvious to anyone who runs vLLM: without a readiness gate, the harness starts issuing actions while the server is still loading weights / compiling CUDA graphs, and the failed/timed-out early actions eat the scoring window — this also plausibly explains the wide 0.82–1.33 band as cold-start variance, not "noise." Freezing the fork is a defensible discipline against scratch-building, but it should not immunize a ~10-line serving-side change with a confirmed external anchor ≥1.47. Actionable: pre-register a small arm (n=3–5 draws) of frozen-fork + readiness-gate; if the band shifts up, the entire daily-filler regime should switch to it.

[MAJOR] A17 72B feasibility is asserted, never computed — Qwen2.5-VL-72B AWQ is ~40 GB of weights before KV cache; that does not fit Kaggle's 2×T4 (32 GB, and sm_75 lacks fast AWQ kernels anyway), so you are implicitly betting on the L4×4 (96 GB) tier with TP=4, where vLLM AWQ throughput for a 72B VLM is realistically single-digit-to-low-teens tok/s decode plus vision-encoder latency per frame. With `A17_WINDOW_S=7920` (2.2 h) and, say, 500–1500 tokens per action turn, the action budget may be only tens of actions per game — possibly below the floor needed to score at all. The brief must show the arithmetic: GPUs targeted, measured tok/s from the v5 canary, tokens/action, and the implied ρ_action *before* v6 fires. "~2.5 GPU-h" for v6 is likewise unsubstantiated.

[MAJOR] No pre-registered PASS/FAIL thresholds for the v5/v6 canaries — v6 "push fires only on v5 PASS," but PASS is undefined in this document in throughput terms. A boot canary that merely proves the model loads is not evidence the plan fits the window; a 72B that boots in 25 minutes and decodes at 4 tok/s "passes boot" and still fails the competition regime. Actionable: publish numeric gates now (e.g., load+warmup ≤ X s, decode ≥ Y tok/s, ≥ Z actions/hr projected) so the Sunday-only panel structure (open question 2) doesn't let an underpowered config auto-promote on a weekday via "sealed arithmetic gates" that were never sealed with the right arithmetic.

[MAJOR] Quota and slot economics are entirely absent — Kaggle GPU quota (~30 h/week) must cover: daily filler submissions, the running v5 canary, v6 (~2.5 GPU-h claimed), retry slots, and any exploration draws — yet no weekly GPU-hour ledger exists anywhere in the brief. Simultaneously, daily filler draws at mean 0.974 have near-zero probability of beating the standing 1.33 (13 draws, empirical max 1.33; each additional draw's P(new max) is shrinking), so their marginal LB value is ≈0 while their quota cost is nonzero. Actionable: add a weekly compute ledger (hours per arm) and an explicit statement of what each filler draw costs in GPU-hours and submission slots, then justify the cadence against the A17 build's needs.

[MINOR] Default trajectory cannot reach gold and the brief doesn't say so — Frozen band ceiling is 1.33; gold ≈1.49 with a dense wall at 1.44–1.61. Everything except A17 (and the shelved exploration arm) is treadmill. The brief should state plainly that A17 (or the readiness-gate arm above) is the only live path to the gold line, so the panel prices the v5-FAIL branch correctly — "retry slot reserved, 6 days + slack" is not a plan B, it's a countdown.

[MINOR] Weekday auto-fire of v6 lacks a resource cap — Under the Sundays-only restructure, v6 pushes automatically on v5 PASS with no human in the loop until the following Sunday. Require the sealed gate to include a hard wall-clock/GPU-hour abort (e.g., kill at 3 GPU-h) so a hung vLLM boot or TP deadlock on the dataset-weights route can't silently burn a week's quota.


=====================================================================

THE PROPOSAL (sha256 of the full document: e8801f7a411c663f; full length 6779 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# Daily Brief — 2026-07-28 (user-ordered full evaluation cycle)

Panel note: yesterday's round ran as **round 21** (learnings/panel/round21/,
5/5 MAJOR-REVISION, directives synthesized 07-27). Today's cycle therefore runs
as **round 22** with round 21 as prior — re-running "--round 21" would have
overwritten the existing round-21 record.

## 1. A17 v5 boot canary — VERDICT: PASS (dataset-weights route ALIVE)

Pull: `runs/kernel_pulls/a17_v5/` (kernel `canivel/arc3-a17-72b-canary`,
status COMPLETE). All greps from the 07-27 build memo hit, evidence lines
(kernel log timestamps in seconds):

- `A17-CANARY model_path=/kaggle/input/qwen25-vl-72b-awq` (t=8.09/8.30) — THE
  dataset-route verdict line: weights found under the DATASET mount.
- `A17-CANARY gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition` (t=8.12).
- `A17-CANARY setup-commands rewrite OK (10 anchors replaced; loud-fail mode,
  no 27B fallback)` (t=7.71).
- `A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ` (t=344.9) — 72B actually
  served; boot-to-serve ≈ 337 s on the dataset mount.
- `A17-CANARY tool-call-roundtrip=OK parser=hermes name=submit_action
  args={"action": "submit_action", "x": 3, "y": 7}` (t=345.8) — risk D
  discharged on the dataset route.
- `A17-CANARY mm-image-roundtrip=OK reply='Yellow'` (t=345.9) — risk E boot
  probe green; vision path intact.
- `fenced-recovery v1: graft applied` (t=347.0); `fenced-recovery v1 hits=1`
  (t=354.8) — adapter live, ≥1 markup-fenced tool call recovered.
- `A17-CANARY games=ft09-0d8bbf25,lp85-305b61c3,sb26-7fbdac44,vc33-5430563c
  (n=4 of 4)`; 12 HEARTBEAT lines; no runtime `A17-CANARY FATAL` (the FATAL
  strings in the log are source-code echoes at t≈7.7 s, not raises).
- 1500 s slice raw counts (MEASUREMENT ONLY, no interpretation at k=1 per
  `learnings/a17_error_model.md`): N(ft09)=2, N(sb26)=1, N(lp85)=0, N(vc33)=2,
  lc=0 across all four.

## 2. A17 v6 full-window bench — PUSHED (today's push slot 1 of 2)

- Built by `duck_eval/a17/build_v6_full_window.py` (notebook was already staged
  v6 locally; idempotence guard confirmed). Smoke
  `duck_eval/a17/a17_v6_smoke.py`: **56/56 PASS** (window 7920 restored,
  budget-derived soft_end restored verbatim, cell-8 serve graft untouched,
  fenced-recovery composition intact, metadata = duckwar family + weights
  dataset, model-finder replay vs the real 43.0 GB / 11-shard local snapshot:
  exactly 1 hit, 27B decoy refused).
- Pushed: **kernel version 6** of `canivel/arc3-a17-72b-canary`. Pull-back
  verification: served `kernel-metadata.json` has `model_sources: []` and
  `dataset_sources` includes `canivel/qwen25-vl-72b-awq`; pulled notebook
  carries `mode=throughput-canary-v6-dataset-weights` + `A17_WINDOW_S = 7920`.
- Regime: free Kaggle kernel BUILD (zero-budget rule) — consumes NO scored
  submission slot. Runs under the 07-26 v4 prereg (G1 recovery ≥ 0.95, G2
  ≥ 100 executed actions, G3 cadence measurement, G4 no capability
  interpretation) and delivers the ρ_action denominator (480 / Σ N₇₂B).
  Output is MEASUREMENT ONLY; numbers go to the sealed walk + Sunday panel.
- R21 named-condition status, stated plainly: v6 fired mid-week on a boot-PASS
  under an explicit user order (the panel is advisory under the 07-27
  restructure). NC-3 (hard resource abort): the kernel session cap + stall-kill
  + zero-action-abort machinery bound the burn to one session (~2.5 GPU-h
  planned). NC-4 (≥200-replay offline parse study) and NC-5 (numeric ρ_action
  kill threshold) are NOT yet discharged — flagged as open question 3 below;
  no scored draw is being consumed by v6, which is the asset NC-4 protects.

## 3. Week's scored results vs verified posterior (ledger refreshed TODAY)

`runs/lb_ground_truth.md` refreshed 2026-07-28 from live API (submissions +
full leaderboard CSV 11:24Z). All numbers below come from that file.

- New overnight draw: **0.90** frozen filler (07-28 00:07Z, COMPLETE).
- Frozen ledger now **n=14, mean 0.9686, s ≈ 0.1384** (was 13/0.974/0.143).
  Week's draws vs posterior (z vs prior n=13 stats): 1.05 (07-25, z ≈ +0.53) ·
  0.84 (07-26, z ≈ −0.94) · 1.02 (07-27, z ≈ +0.32) · 0.90 (07-28, z ≈ −0.53)
  — all interior to band 0.82–1.33, alternating around the mean; no monotone
  run, no low-streak, MK/CUSUM no-trend verdict (07-24) stands. s has
  tightened 3 refreshes in a row (0.148 → 0.143 → 0.138).
- Sentinel arm: n=1 (0.71, 07-24, harm-paused). Under the honest t-predictive
  that draw is one-sided p ≈ 0.07 — suggestive, not significant; C2 forbids
  any single-draw claim and none is made. Status: SHELVED by
  `learnings/war_room/sentinel_disposition_2026-07-24.md` ("certified
  observable, no lift channel"; eval-rail evidence load-bearing, scored draw
  consistent-with only). R21 directive #4 (3/5) calls the n=1 shelving
  statistically indefensible and wants a pre-registered un-shelve rule
  (date for draw #2, n ≥ 4–5 per disposition, sequential stopping boundary).
  Disposition is OPEN — question 2 below.
- LB context (same refresh): our best 1.33, rank #51 (47 strictly above, 6
  tied); leader KOJIMA 1.86; gold cutoff 1.49 (#13–14 = 1.49, #15 = 1.48);
  boristown fork anchor 1.47 unchanged. P(single frozen draw ≥ 1.49) remains
  ≈ 2×10⁻⁴ — filler holds rank, it does not climb (R21 directive #5 stands).

## 4. Open questions for round 22

1. **A17 v6 result handling:** v6 is in flight (full 7920 s window, seed 1).
   Before its numbers are read: does the panel ratify a numeric ρ_action kill
   threshold NOW (R21 NC-5 wants Y stated pre-observation: ρ_action < Y ⇒ 72B
   route dead, slots revert to frozen/gated-A/B)? Propose the panel name Y and
   the expected-LB mapping (R21 directive #2 arithmetic) before the sealed
   walk reads v6.
2. **Sentinel arm disposition:** shelved-at-n=1 vs R21 directive #4. Options:
   (a) pre-register un-shelve rule + date for draw #2 (e.g., after v6 lands),
   n ≥ 4–5, sequential stop (re-shelve only on 2 consecutive < 0.80 or mean of
   first 3 < 0.80); (b) uphold the disposition memo (eval-rail evidence
   load-bearing, doctrinal zero-upside) and spend those slots on the
   boristown readiness-gate A/B instead. Panel to pick one, with error rates.
3. **Boristown vLLM readiness-gate A/B (R21 directive #1, 5/5):** still
   unscheduled. If ratified, it competes with sentinel draw #2 for the same
   filler slots — panel to rank the two and set the pre-registration (n=3–5
   gated vs frozen, one-sided at the 1.47-anchor-implied effect size).
4. **EWM Stage-1 (due Aug 4, r16 §9 window Jul 28–Aug 3):** still BLOCKED by
   the latent-state audit (r16 §10, 0.99 bar sealed) + §9.2 cheap measurement.
   Six days remain. Does the panel re-affirm the gate-then-window sequence, or
   re-price given the A17 rail now occupies the build slots through ~Aug 3?

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
