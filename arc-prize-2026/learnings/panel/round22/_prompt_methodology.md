You are Professor of Empirical ML Methodology and Statistics (experimental design, multiple-comparisons, noise-band inference; rejects any plan that draws conclusions from single noisy samples).

You are reviewer #4 on a 5-person adversarial review panel evaluating a competition
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

**[MAJOR] Ledger provenance gap: the n=13 statistics include two draws not in the canonical API-verified ledger.** The canonical file (refreshed 07-25) shows frozen-fork n=11, mean 0.982, s≈0.150. The brief's n=13, mean 0.974, s≈0.143 is arithmetically correct *only if* a 0.84 draw (07-26, sourced from "the 07-26 brief") and today's 1.02 are real — neither is in the API-verified file, which was "deliberately not edited." This recreates in miniature the exact failure mode of the stale-template incident: statistics computed from a chain of briefs rather than from the verified source. Fix: run the stated verification command (`uvx --from kaggle==2.0.0 kaggle competitions submissions ...`) for the 07-26 and 07-27 draws and append them to the canonical ledger via the refresh path *before* any n=13 statistic is cited in a decision.

**[MAJOR] No draws-to-cutoff power analysis; the filler strategy cannot reach gold under its own distribution, and the one experiment that could explain the 1.47 anchor is not scheduled.** Under the frozen empirical distribution (mean≈0.974, s≈0.143), P(single draw ≥ 1.49) ≈ P(z ≥ 3.6) < 3×10⁻⁴ — order-statistics of daily fillers through Nov 2 (~100 draws) give expected max ≈ 1.33–1.38, nowhere near the 1.49 gold cutoff. Yet byte-identical forks drew 1.39 and 1.47, a 2.7–3.3σ discrepancy versus your own draws, which means either the draws are not exchangeable with the external anchors or the boristown vLLM readiness gate is causal. That gate is a single-variable, directly testable intervention on the frozen fork; it is the highest-information experiment available and appears nowhere in the brief's build queue. Fix: pre-register a readiness-gate A/B on the frozen fork (k≥4 draws per arm, one-sided test at the anchor-implied effect size) and present a draws-to-cutoff calculation for whatever arm survives.

**[MAJOR] Exploration-arm disposition and cadence are being decided from n=1, with a per-draw threshold whose null false-trigger rate is ~11%.** The sentinel arm was harm-paused and shelved on a single draw of 0.71 — under the frozen distribution that is z ≈ −1.8, not significant, and the *closed war arm itself* contains a 0.76. Moreover the 0.80 harm-pause threshold fires with probability ≈ 0.11 per draw even if the exploration arm's true mean equals the frozen mean (z = (0.80−0.974)/0.143 ≈ −1.22). Open question 4 must be answered with a pre-registered exploration sample size (n≥4–5 minimum) and a sequential stopping rule with stated error rates (e.g., a group-sequential boundary or SPRT), not draw-by-draw thresholding. As written, the exploration program is structurally guaranteed to be killed by noise.

**[MAJOR] "Sealed arithmetic gates" for weekday promotion have no stated operating characteristics — this is a condition on acknowledging the restructure (open question 2).** Any threshold rule applied to single daily draws with s≈0.14 will trigger falsely at a substantial and *computable* rate; the brief proposes to remove human review on weekdays without quoting that rate. Before the panel acknowledges Sundays-only review, each gate must publish: (a) its false-trigger probability per draw under the frozen empirical distribution, (b) its family-wise trigger rate over a 6-day unreviewed week, and (c) which triggers escalate to an unscheduled panel. Without (a)–(c) the restructure trades review latency for uncontrolled error rates.

**[MINOR] The "band met" check and the MK/CUSUM "no drift" verdict carry almost no evidential weight as stated.** The 0.82–1.33 band is the empirical min–max of the sample itself; under stationarity a new draw falls inside it with probability ≈ (n−1)/(n+1) ≈ 86%, so "met, comfortably interior" is close to vacuous, and the endpoints are data-dependent rather than fixed at pre-registration. MK/CUSUM at n=13 with s=0.14 has negligible power against any drift smaller than ~0.1/week; "no-trend verdict stands" should be reported as "insufficient power to detect trend," and the CUSUM control limits/reference value should be stated so repeated daily testing has a defined ARL.

**[MINOR] Untracked multiplicity across the daily trigger family.** Each day the loop evaluates band membership, harm-pause, W2, OBJ-H null10, and drift tests; over months this is hundreds of implicit hypothesis tests with no alpha-spending or family-wise accounting, so both the false-alarm and the missed-alarm rates of the overall monitoring system are unknown. A one-page specification of the full trigger family with per-trigger and family-wise error rates under the frozen null would fix this.


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
