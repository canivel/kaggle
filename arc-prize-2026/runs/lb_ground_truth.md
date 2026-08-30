# LB Ground Truth (refreshed by ARCMorningCheck)

**Last refresh: 2026-08-30 06:00 local (pull_utc 2026-08-30T10:00:47Z), full download, 2624 rows, SubmissionCount present.**
Source: `runs/lb_daily/lb_full_2026-08-30.csv` (heartbeat sha 578857f57a8f, `--check` exit 0, `HEARTBEAT OK`, `PRIOR-DAY OK`).

## Us
- **Canivel — Rank 153 of 2624, Score 2.05, SubmissionCount 126, LastSubmissionDate 2026-08-30 00:19:33.**
- Yesterday: #203 of 2603 at 1.92 → **+50 ranks**, bought entirely by the **2.05 draw of 2026-08-29 11:41**, which was a **+2.01σ variance draw of an UNCHANGED kernel version** (same slug, same version as the 1.14 and 1.26 draws), not an improvement. See ITERATION_LOG 2026-08-29 (night).
- **Since that draw the rank has already decayed: #147/2616 last night → #153/2624 this morning, -6 on a flat score in under 12 hours.** Holding a score is losing rank on this board.
- Two submissions landed in this window (124 → 126): the 08-29 11:41 field-floor filler (**2.05**) and the 08-30 00:19 `arc3-tv28-fork` draw 1 (**1.62**, COMPLETE, n=1 on a fresh config — no verdict).
- Public `Score` is a **max over submissions**; the banked **2.05 is a max of eight** field-floor draws (1.59 / 1.58 / 1.63 / 1.16 / 1.92 / 1.14 / 1.26 / 2.05 → **config mean 1.5413, sd 0.3376, n=8**), not a level. `project_arc_final_selection_rule` selects the final two by **config mean**, so 1.5413 is the number that matters and 2.05 is the number that does not.

## Score lines (this pull)

| line | score | prev (08-29) |
|---|---|---|
| top-5 (prize) | 3.37 | 3.37 |
| **top-10** | **2.98** | **2.94** |
| top-13 (gold) | 2.78 | 2.72 |
| top-50 | 2.37 | 2.32 |
| top-100 | 2.16 | 2.15 |
| top-250 | 1.84 | 1.82 |
| us (#153) | 2.05 | 1.92 (#203) |

**The top-10 line moved for the second consecutive day (2.80 → 2.94 → 2.98).** Every line at or below top-13 also drifted up. Against our floor config (mean 1.5413, sd 0.3376) the top-10 line is **+4.27σ**.

## Top 15

| rank | team | score | subs | last sub |
|---|---|---|---|---|
| 1 | cstl | 5.99 | 40 | 2026-08-29 19:25:36 |
| 2 | Lord Han Solo | 4.99 | 47 | 2026-08-29 22:18:06 |
| 3 | Tufa Labs | 4.67 | 122 | 2026-08-29 17:42:40 |
| 4 | Tong Hui Kang | 4.27 | 57 | 2026-08-29 23:54:16 |
| 5 | rfbr | 3.37 | 16 | 2026-08-29 14:52:26 |
| 6 | **Youssef Nader @ Aurelic** | **3.19** | **12** | 2026-08-29 20:03:18 |
| 7 | Tony G | 3.17 | 14 | 2026-08-27 06:36:21 |
| 8 | Daniel Franzen | 3.15 | 56 | 2026-08-29 21:20:45 |
| 9 | **Liao Zixu** | **3.13** | **15** | 2026-08-29 13:07:35 |
| 10 | OzanM. | 2.98 | 100 | 2026-08-29 07:02:05 |
| 11 | Abstraction Lab & MindsAI (Jack Cole) | 2.94 | 132 | 2026-08-29 19:56:56 |
| 12 | Tony Li | 2.80 | 15 | 2026-08-29 03:44:50 |
| 13 | Tatu Helander | 2.78 | 75 | 2026-08-29 07:53:32 |
| 14 | Akhil Tolani | 2.73 | 79 | 2026-08-28 18:14:47 |
| 15 | AbeLincoln1865 | 2.72 | 20 | 2026-08-28 16:30:29 |

Two new entrants to the top 10 this window, **both on a single draw and both from low lifetime submission counts** — Youssef Nader @ Aurelic (+1.58 on 1 draw, 12 subs) and Liao Zixu (+0.92 on 1 draw, 15 subs). Neither gain is explicable as draw-count grinding. Method **UNKNOWN** for both.

## Control arm (08-29 → 08-30)

| team | score | Δscore | subs | Δsubs | **Δ/draw** |
|---|---|---|---|---|---|
| Jack Cole — Abstraction Lab & MindsAI | 2.94 | +0.00 | 132 | +1 | **0.0000** |
| Tufa Labs | 4.67 | +0.00 | 122 | +1 | **0.0000** |

**Both control teams bought a draw and gained exactly nothing.** The authors of the TTT literature and of the harness we fork did not move. On this evidence the commodity-engine / shared-regime story is **weak**. This is a measurement about scores, not about method.

## The 1.55–1.65 band (this window)

| | old (08-29) | new (08-30) | delta |
|---|---|---|---|
| teams in band | 57 | 53 | **-4** |
| median score | 1.60 | 1.60 | +0.00 |
| median subs | 14.0 | 9.0 | **-5.0** |

Band **shrank** by 4 (4 entered, 8 left) on a **flat median score** and a sharply lower median draw count. A drop-in engine swap would lift the band as a body; this is individual transit through it. Not the shape of an engine swap.

## Draws bought vs gain (this window)
- 292 teams submitted; **only 59 (20.2%) gained anything**; 317 new submissions total.
- Median Δscore/Δsub among gainers **0.2300**; max **1.5800** (Youssef Nader @ Aurelic, 1 draw, 1.61 → 3.19).
- **~80% of teams who paid for a draw gained 0.00.** This is the base rate against which every single-draw "improvement" — including our own 2.05 and our 1.62 — must be read.

## Evidence discipline
This instrument measures **Score, SubmissionCount, Rank, TeamName, LastSubmissionDate** — nothing else. **Do not infer method from movement.** `LastSubmissionDate` is the team's MOST RECENT submission while `Score` is their BEST; they need not be the same submission, so this file **cannot date a scoring run**. Every method claim carries an evidence class (DISCLOSED / INFERRED / UNKNOWN) per `learnings/top6_evidence_audit_2026-08-15.md`. All movement above is **UNKNOWN**. Today's tally: 0 DISCLOSED / 0 INFERRED / 9 UNKNOWN.
