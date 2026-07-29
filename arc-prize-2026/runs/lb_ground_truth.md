# LB ground truth — refreshed 2026-07-29 (live Kaggle API: submissions + leaderboard head)

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

Frozen-fork control (n=15): 0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
1.05, 0.84, 1.02, 0.90, 1.03 → mean 0.9727, s ≈ 0.1343. War arm (n=5, CLOSED per A9):
0.91, 1.08, 0.88, 1.05, 0.76. Sentinel exploration arm (n=1, HARM-PAUSED 07-24, SHELVED
by disposition memo; R22 disposition: pre-registered un-shelve rule adopted, queued
BEHIND the boristown readiness-gate A/B): 0.71.

Recent tail (newest first): 1.03 filler (07-29) · 0.90 filler (07-28) · 1.02 filler
(07-27) · 0.84 filler (07-26) · 1.05 filler (07-25) · 0.71 sentinel (07-24) · 0.82
filler (07-23) · 1.14 filler (07-22).

Refresh 2026-07-29 (live API `competitions submissions` + leaderboard head): incorporated
the 07-29 00:07Z frozen draw **1.03** (API status COMPLETE, description "...n=14 after
07-28 draw 0.90..."). Stats recomputed numerically: n=15, mean 0.9727, s 0.1343 (was
n=14 / 0.9686 / 0.1384). 1.03 is interior (z ≈ +0.44 vs prior stats): no band change, no
drift signal, no trigger. Leaderboard head cross-check: leader KOJIMA 1.86, #2 Tecnod8.AI
1.61, #3 DhanaLakshmiMalla 1.60, gold cutoff 1.49 (#13–14 at 1.49, #15 = 1.48) — all
unchanged from 07-28.

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
