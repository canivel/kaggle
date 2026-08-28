# LB Ground Truth (refreshed by ARCMorningCheck)

**Last refresh: 2026-08-28 06:15 local (pull_utc 2026-08-28T10:15:05Z), full download, 2589 rows, SubmissionCount present.**
Source: `runs/lb_daily/lb_full_2026-08-28.csv` (heartbeat sha 19d7f21f1cd7, `--check` exit 0, PRIOR-DAY OK).

## Us
- **Canivel — Rank 193 of 2589, Score 1.92, SubmissionCount 124, LastSubmissionDate 2026-08-28 00:11:01.**
- Yesterday: #182 of 2564 at 1.92 → **-11 ranks on a FLAT score**, fourth consecutive day of pure board drift under us. Cumulative #146 → #159 → #182 → #193 over four pulls, score unchanged throughout.
- Public `Score` is a **max over submissions**; last night's field-floor draw of **1.26** is below our 1.92 max, so `Score` is unchanged and the 1.92 is still the 08-25 filler draw.
- Certified Q38 field-floor config draws: **1.59 / 1.58 / 1.63 / 1.16 / 1.92 / 1.14 / 1.26 → config mean 1.4686, sd 0.2897 (n=7)**. The banked **1.92 is a max of seven**, not a level; the config mean has now FALLEN on three consecutive draws (1.5760 → 1.5033 → 1.4686).

## Score lines (this pull)

| line | score | prev (08-27) |
|---|---|---|
| top-5 (prize) | 3.37 | 3.37 |
| top-13 (gold) | 2.72 | 2.70 |
| top-50 | 2.32 | 2.30 |
| top-100 | 2.14 | 2.12 |
| top-250 | 1.81 | 1.79 |
| us (#193) | 1.92 | 1.92 (#182) |

## Top 15

| rank | team | score | subs | last sub |
|---|---|---|---|---|
| 1 | cstl | 5.99 | 37 | 2026-08-26 15:48:54 |
| 2 | Lord Han Solo | 4.99 | 45 | 2026-08-27 03:49:47 |
| 3 | Tufa Labs | 4.67 | 120 | 2026-08-27 07:10:25 |
| 4 | Tong Hui Kang | 3.88 | 55 | 2026-08-27 01:11:30 |
| 5 | rfbr | 3.37 | 14 | 2026-08-27 14:48:23 |
| 6 | Tony G | 3.17 | 14 | 2026-08-27 06:36:21 |
| 7 | Daniel Franzen | 3.04 | 54 | 2026-08-27 16:39:13 |
| 8 | OzanM. | 2.98 | 98 | 2026-08-27 05:53:12 |
| 9 | Abstraction Lab & MindsAI | 2.94 | 131 | 2026-08-27 14:54:08 |
| 10 | Tony Li | 2.80 | 13 | 2026-08-27 23:13:34 |
| 11 | Tatu Helander | 2.78 | 73 | 2026-08-27 19:14:02 |
| 12 | Akhil Tolani | 2.73 | 78 | 2026-08-27 00:32:10 |
| 13 | AbeLincoln1865 | 2.72 | 19 | 2026-08-27 15:31:42 |
| 14 | wking edewd | 2.70 | 3 | 2026-08-22 02:47:50 |
| 15 | Diya Sharma | 2.69 | 1 | 2026-08-24 11:20:25 |

## Control arm (08-27 → 08-28)

| team | score | prev | dScore | subs | dSubs | d/draw |
|---|---|---|---|---|---|---|
| Jack Cole (MindsAI) | 2.94 | 2.94 | +0.00 | 131 | +1 | 0.0000 |
| Tufa Labs | 4.67 | 4.67 | +0.00 | 120 | +1 | 0.0000 |

**Neither control team moved.** Both bought a draw and gained nothing → the commodity-engine / shared-regime story is WEAK on this evidence.

## 1.55–1.65 band (08-27 → 08-28)

| | old | new | delta |
|---|---|---|---|
| teams in band | 48 | 51 | +3 |
| median score | 1.60 | 1.60 | +0.00 |
| median subs | 12.5 | 12.0 | -0.5 |

## Draws bought vs gain (this window)
- 284 teams submitted; **only 52 (18.3%) gained anything**; 308 new submissions total.
- Median Δscore/Δsub among gainers **0.1850**; max **2.1600** (EndeavourRyo, 1 draw).

## Evidence discipline
This instrument measures **Score, SubmissionCount, Rank, TeamName, LastSubmissionDate** — nothing else. **Do not infer method from movement.** `LastSubmissionDate` is the team's MOST RECENT submission while `Score` is their BEST; they need not be the same submission, so this file **cannot date a scoring run**. Every method claim carries an evidence class (DISCLOSED / INFERRED / UNKNOWN) per `learnings/top6_evidence_audit_2026-08-15.md`. All movement above is **UNKNOWN**.
