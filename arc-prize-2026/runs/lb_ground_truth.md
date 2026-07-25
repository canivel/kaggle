# LB ground truth — refreshed 2026-07-25 by the daily loop (live Kaggle API)

Account: canivel (Danilo Canivel, d.canivel@gmail.com). Competition:
arc-prize-2026-arc-agi-3. Verification command:
`uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`.

- OUR BEST (public LB): **1.33** (frozen-fork filler draw, 2026-07-18). Current rank
  ~#50–53 (slid out of the loaded top-50 overnight; 1.33–1.34 is a crowded floor).
- LEADER: YUTO KOJIMA **1.86**. #2 Tecnod8.AI 1.61, #3 DhanaLakshmiMalla 1.60,
  #4 ippeiogawa 1.58. Gold cutoff ≈ **1.49** (top 13; #14 = 1.48).
  Dense band 1.44–1.61; 7+ teams at 1.46–1.47 (boristown's public 1.47 seeding).
- External context: Claude Opus 5 posted 30.2% on the ARC-AGI-3 benchmark (arcprize.org,
  Jul 24) via API at High reasoning effort — different regime (unconstrained API vs
  Kaggle quantized/time-limited local), no artifact to lift; directional support for
  capability-over-harness.
- The "best 0.43 / leader 1.56" figures in pre-R19 briefings were a STALE HARDCODED
  TEMPLATE (May-era), root-caused and fixed 2026-07-24 (panel_round.py now reads this
  file). Reconciliation: 0.43 was the team's best in early May (forge-era agents);
  the frozen duck fork lifted the floor to the 0.82–1.33 band from 2026-07-05 on.

## Draw-by-draw scored ledger (all API-verified)

Frozen-fork control (n=11): 0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82,
1.05 → mean 0.982, s ≈ 0.150. War arm (n=5, CLOSED per A9): 0.91, 1.08, 0.88, 1.05, 0.76.
Sentinel exploration arm (n=1, HARM-PAUSED 07-24, SHELVED by disposition memo): 0.71.

Recent tail (newest first): 1.05 filler (07-25) · 0.71 sentinel (07-24) · 0.82 filler
(07-23) · 1.14 filler (07-22) · 0.93 filler manual (07-21) · 0.92 filler (07-20).

External anchors: byte-identical public forks of the same duck artifact family have
drawn 1.39 (zoli800) and 1.47 (boristown agi-duck-harness-fast-eval, whose only real
functional diff is a vLLM readiness gate — see
learnings/war_room/fork_diff_boristown_2026-07-24.md). Artifact tail ≥ 1.47 confirmed.
