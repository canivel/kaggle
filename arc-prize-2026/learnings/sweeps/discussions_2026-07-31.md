# ARC-AGI-3 Discussion Sweep — 2026-07-31

Scope: threads posted or with comment activity NEW since 2026-07-30 (~24h window).
Sorts checked: `?sort=new` (Recently Posted) and `?sort=recent-comments`. Page 1 of each (rendered via chrome-devtools snapshot; WebFetch returns empty JS shell).
Plan context for verdicts: frozen-fork duck harness (ledger n=15, μ0.973 s0.134, best 1.33; LB head KOJIMA 1.86, gold 1.49); A17 72B route DEAD (format livelock, ΣN=5); active build lane = boristown readiness-gate A/B (1.47 anchor, prereg draft, Sunday R23 ratifies); hard constraints: 9h rerun wall, no-final-rerun (host-confirmed), zero cloud budget, 2 kernel pushes/day.

## New posts / activity

| # | Title | Author | Posted / Activity | Gist | Verdict | Reason (vs plan) |
|---|-------|--------|-------------------|------|---------|------------------|
| 1 | [Question About Dual Notebook Executions During Competition Submission](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/731290) | Alex Paul (alexazander, 178th) | Posted 8h ago (07-31); 1 comment 7h ago by Quanyi Li (428th) | Asks why submitting spawns TWO notebook runs (the scoring rerun + a save-version run he didn't request); does the second eat GPU quota, is it safe to cancel. Quanyi Li's answer: yes it consumes GPU quota, and canceling it does NOT affect the submission/score. 0 votes. | IGNORE | Known Kaggle code-comp mechanic (save-version run vs. hidden scoring rerun). No effect on our lane: our BUILD runs are load-bearing (results JSON = evidence artifacts under zero-cloud policy), so we don't cancel them; quota-trim tip noted but changes nothing. |

## Notes on boundary items (no new content in window)

- Pinned "Clarification on deadline for milestone prizes" — last comment still 3d ago (KostasMouratidis); same item as 07-30 sweep. IGNORE.
- "Three clarifications on final scoring mechanics" (729985) — last comment still 3d ago (Hendrik Nowak); already covered 07-30. IGNORE.
- "Tufa Labs' Winning Solution for ARC-AGI-3 Milestone 1" (717133) — last comment still 3d ago (Mustang Liu chatter); covered 07-30. IGNORE.
- Yesterday's two new posts unchanged: "Hello! I'm NK!" (730528) now 6 votes, still no technical content; MDL self-promo (730225) slid to -6 votes, still no level-completion evidence. Verdicts stand (IGNORE / IGNORE).

## Verdict summary
- New posts in window: 1 (Alex Paul dual-notebook-execution Q&A, 07-31).
- Bumped older threads with <24h comments: 1 (same Alex Paul thread — Quanyi Li reply). No other thread on either sort shows activity newer than 3d.
- Non-IGNORE verdicts: NONE.
- Plan unchanged: boristown readiness-gate A/B remains the active build lane; frozen duck fork filler nightly continues; no discussion-driven course change.
