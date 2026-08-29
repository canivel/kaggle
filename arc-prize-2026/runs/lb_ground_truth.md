# LB Ground Truth (refreshed by ARCMorningCheck)

**Last refresh: 2026-08-29 06:01 local (pull_utc 2026-08-29T10:01:01Z), full download, 2603 rows, SubmissionCount present.**
Source: `runs/lb_daily/lb_full_2026-08-29.csv` (heartbeat sha a420868632f2, `--check` exit 0, PRIOR-DAY OK).

## Us
- **Canivel — Rank 203 of 2603, Score 1.92, SubmissionCount 124, LastSubmissionDate 2026-08-28 00:11:01.**
- Yesterday: #193 of 2589 at 1.92 → **-10 ranks on a FLAT score**, fifth consecutive day of pure board drift under us. Cumulative #146 → #159 → #182 → #193 → #203 over five pulls, score unchanged throughout.
- **NO SUBMISSION WAS MADE IN THE 08-28 NIGHT WINDOW** — `SubmissionCount` is unchanged at 124 and `LastSubmissionDate` is still the 08-28 00:11 draw. A daily draw was LOST (see ITERATION_LOG 2026-08-29).
- Public `Score` is a **max over submissions**; the banked **1.92 is a max of seven** field-floor draws (1.59 / 1.58 / 1.63 / 1.16 / 1.92 / 1.14 / 1.26 → config mean 1.4686, sd 0.2897, n=7), not a level.

## Score lines (this pull)

| line | score | prev (08-28) |
|---|---|---|
| top-5 (prize) | 3.37 | 3.37 |
| top-13 (gold) | 2.72 | 2.72 |
| top-50 | 2.32 | 2.32 |
| top-100 | 2.15 | 2.14 |
| top-250 | 1.82 | 1.81 |
| us (#203) | 1.92 | 1.92 (#193) |

## Top 15

| rank | team | score | subs | last sub |
|---|---|---|---|---|
| 1 | cstl | 5.99 | 39 | 2026-08-28 17:45:24 |
| 2 | Lord Han Solo | 4.99 | 46 | 2026-08-28 22:27:40 |
| 3 | Tufa Labs | 4.67 | 121 | 2026-08-28 10:55:35 |
| 4 | Tong Hui Kang | 4.27 | 56 | 2026-08-28 01:38:34 |
| 5 | rfbr | 3.37 | 15 | 2026-08-28 14:50:23 |
| 6 | Tony G | 3.17 | 14 | 2026-08-27 06:36:21 |
| 7 | Daniel Franzen | 3.15 | 55 | 2026-08-28 23:25:25 |
| 8 | OzanM. | 2.98 | 99 | 2026-08-28 06:11:04 |
| 9 | Abstraction Lab & MindsAI | 2.94 | 131 | 2026-08-27 14:54:08 |
| 10 | Tony Li | 2.80 | 14 | 2026-08-28 12:40:04 |
| 11 | Tatu Helander | 2.78 | 74 | 2026-08-28 07:03:58 |
| 12 | Akhil Tolani | 2.73 | 79 | 2026-08-28 18:14:47 |
| 13 | AbeLincoln1865 | 2.72 | 20 | 2026-08-28 16:30:29 |
| 14 | wking edewd | 2.70 | 3 | 2026-08-22 02:47:50 |
| 15 | Diya Sharma | 2.69 | 1 | 2026-08-24 11:20:25 |

## Control arm (08-28 → 08-29)

| team | score | prev | dScore | subs | dSubs | d/draw |
|---|---|---|---|---|---|---|
| Jack Cole (MindsAI) | 2.94 | 2.94 | +0.00 | 131 | +0 | — (IDLE) |
| Tufa Labs | 4.67 | 4.67 | +0.00 | 121 | +1 | 0.0000 |

**Neither control team moved.** Cole did not submit at all; Tufa bought a draw and gained nothing → the commodity-engine / shared-regime story is **WEAK** on this evidence, for a second consecutive window.

## 1.55–1.65 band (08-28 → 08-29)

| | old | new | delta |
|---|---|---|---|
| teams in band | 51 | 57 | +6 |
| median score | 1.60 | 1.60 | +0.00 |
| median subs | 12.0 | 14.0 | +2.0 |

Band grew by 6 teams (7 entered, 1 left) on a **flat median score** and +2 median draws — arrivals and extra draws, not a lift. Not the shape of a drop-in engine swap.

## Draws bought vs gain (this window)
- 278 teams submitted; **only 53 (19.1%) gained anything**; 301 new submissions total.
- Median Δscore/Δsub among gainers **0.2000**; max **1.4500** (nithar1609, 1 draw, 0.15 → 1.60).
- 13 of the 15 teams that bought 2 draws gained **exactly 0.00**.

## Evidence discipline
This instrument measures **Score, SubmissionCount, Rank, TeamName, LastSubmissionDate** — nothing else. **Do not infer method from movement.** `LastSubmissionDate` is the team's MOST RECENT submission while `Score` is their BEST; they need not be the same submission, so this file **cannot date a scoring run**. Every method claim carries an evidence class (DISCLOSED / INFERRED / UNKNOWN) per `learnings/top6_evidence_audit_2026-08-15.md`. All movement above is **UNKNOWN**. Today's tally: 0 DISCLOSED / 1 INFERRED / 7 UNKNOWN.
