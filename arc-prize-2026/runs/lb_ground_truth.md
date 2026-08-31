# LB Ground Truth (refreshed by ARCMorningCheck)

**Last refresh: 2026-08-31 06:00 local (pull_utc 2026-08-31T10:00:47Z), full download, 2651 rows, SubmissionCount present.**
Source: `runs/lb_daily/lb_full_2026-08-31.csv` (heartbeat sha b017f7a69b3d, `--check` exit 0, `HEARTBEAT OK`, `PRIOR-DAY OK`).

## Us
- **Canivel — Rank 173 of 2651, Score 2.05, SubmissionCount 127, LastSubmissionDate 2026-08-31 00:07:09.**
- Yesterday: #153 of 2624 at 2.05 → **-20 ranks on a FLAT score in 24h.** The day before: #147/2616 → #153/2624 (-6). **Cumulative -26 ranks in 48h while holding the campaign-best 2.05.** Holding a score is losing rank on this board, and the rate of loss is accelerating.
- One submission landed in this window (126 → 127): the 08-31 00:07 `arc3-tv28-fork` **draw 2 = 1.36** (COMPLETE).
- **TV28 fork config now n=2: 1.62, 1.36 → mean 1.49.** That is **BELOW the field-floor config mean (1.5413, n=8)**. The +0.08 edge draw 1 appeared to have is gone; the fork is not a gain. No verdict at n=2, but the direction is not favourable.
- Public `Score` is a **max over submissions**; the banked **2.05 is a max of eight** field-floor draws (1.59 / 1.58 / 1.63 / 1.16 / 1.92 / 1.14 / 1.26 / 2.05 → **config mean 1.5413, sd 0.3376, n=8**), not a level. `project_arc_final_selection_rule` selects the final two by **config mean**, so 1.5413 is the number that matters and 2.05 is the number that does not.

## Score lines (this pull)

| line | score | prev (08-30) | Δ |
|---|---|---|---|
| top-5 (prize) | **4.05** | 3.37 | **+0.68** |
| **top-10** | **3.17** | 2.98 | **+0.19** |
| top-13 (gold) | 2.98 | 2.78 | +0.20 |
| top-50 | 2.45 | 2.37 | +0.08 |
| top-100 | 2.21 | 2.16 | +0.05 |
| top-250 | 1.88 | 1.84 | +0.04 |
| us (#173) | 2.05 | 2.05 (#153) | +0.00 |

**The top-10 line moved for the THIRD consecutive day (2.80 → 2.94 → 2.98 → 3.17)** and this is the largest single-day step of the three. The prize line jumped +0.68 in one day. Against our floor config (mean 1.5413, sd 0.3376) the top-10 line is now **+4.82σ**.

## Top 15

| rank | team | score | subs | last sub |
|---|---|---|---|---|
| 1 | **cstl** | **7.51** | 41 | 2026-08-30 19:46:19 |
| 2 | Lord Han Solo | 4.99 | 48 | 2026-08-30 22:22:54 |
| 3 | Tufa Labs | 4.71 | 123 | 2026-08-30 04:50:22 |
| 4 | Tong Hui Kang | 4.27 | 58 | 2026-08-30 22:05:58 |
| 5 | **Daniel Franzen** | **4.05** | 57 | 2026-08-30 20:04:36 |
| 6 | Son Pham & Mark Barney | 3.58 | 33 | 2026-08-30 17:33:24 |
| 7 | Kyutai | 3.37 | 34 | 2026-08-30 14:54:25 |
| 8 | **Dawid Kopiczko** | **3.31** | 42 | 2026-08-30 23:28:10 |
| 9 | Youssef Nader @ Aurelic | 3.19 | 13 | 2026-08-30 05:53:08 |
| 10 | Tony G | 3.17 | 14 | 2026-08-27 06:36:21 |
| 11 | Liao Zixu | 3.13 | 15 | 2026-08-29 13:07:35 |
| 12 | sawada | 3.10 | 32 | 2026-08-31 00:05:19 |
| 13 | OzanM. | 2.98 | 101 | 2026-08-30 07:53:03 |
| 14 | Abstraction Lab & MindsAI (Jack Cole) | 2.94 | 133 | 2026-08-30 17:44:51 |
| 15 | Tony Li | 2.80 | 17 | 2026-08-31 00:54:17 |

**cstl 5.99 → 7.51 (+1.52 on ONE draw).** #1 is now **3.7× our banked max** and pulling away. Daniel Franzen +0.90 on one draw into #5. Method **UNKNOWN** for both — this instrument measures score, not method.

## Control arm (08-30 → 08-31)

| team | score | Δscore | subs | Δsubs | **Δ/draw** | flag |
|---|---|---|---|---|---|---|
| Jack Cole — Abstraction Lab & MindsAI | 2.94 | +0.00 | 133 | +1 | **0.0000** | DREW-NO-GAIN |
| Tufa Labs | 4.71 | **+0.04** | 123 | +1 | **0.0400** | DRIFT |

Jack Cole bought a draw and gained nothing for the **second consecutive day** (2.94 flat, now -3 ranks). Tufa Labs moved **+0.04 on one draw** — a drift, not a step. The shared-regime story **survives but is not confirmed**: a +0.04 move is within what any single re-draw of an unchanged config buys. Evidence class **UNKNOWN**.

## The 1.55–1.65 band (08-30 → 08-31)

| | old (08-30) | new (08-31) | delta |
|---|---|---|---|
| teams in band | 53 | 48 | **-5** |
| median score | 1.60 | 1.60 | **+0.00** |
| median subs | 9.0 | 9.0 | **+0.0** |

2 entered, 7 left. **Second consecutive day of a flat median on a shrinking band.** A drop-in engine swap would lift the band as a body; it has not. Teams transit individually. Evidence class **UNKNOWN**.

## Best-of-N confound (measured, this window)

- 308 teams submitted; **only 85 (27.6%) gained anything** on 349 new submissions.
- Median Δscore/Δsub among *gainers*: **0.2100**. Max: **2.3000** (SuperVisor, 2 lifetime subs).
- **Kyutai bought 18 draws and gained 0.00.** The single clearest demonstration on this board that draw-count is not score.
- ~72% of teams who paid for a draw got nothing. Every single-draw "improvement" — including our own 2.05 — must be read against that base rate.
