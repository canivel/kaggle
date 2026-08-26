# Discussions Sweep — 2026-07-27

Baseline for dedup: `learnings/war_room/discussions_2026-07-26.md` (07-26 sweep) +
`learnings/daily_brief_2026-07-26.md` §1b.
Fetch method: chrome-devtools MCP (anonymous session), feed sorted recent-comments, page 1;
thread #728278 pulled via Kaggle internal API
(`/api/i/discussions.DiscussionsService/GetForumTopicById`) after the DOM refused to render the
newest comment. LB via `kaggle==2.0.0` CLI.

---

## NEW POSTS since 2026-07-26

**Zero new threads.** The only feed movement is on the already-tracked #728278.

| Thread | Author | Gist | Verdict |
|---|---|---|---|
| #728278 "Is 100% Accuracy Realistic" — comment #3504145 (posted ~1h before sweep, listed "by Doruk Doğrular") | Doruk Doğrular | **DELETED** — feed still links it (11 comments) but the topic API payload contains only 10 comments, all ≤07-25; content unrecoverable anonymously | IGNORE — no content exists to evaluate; re-check tomorrow in case of repost |
| #728278 — comment 07-25 17:58Z (posted before yesterday's sweep but not itemized in it) | Scott Le Grand | Disputes the "one RTX Pro 6000 ≈ 1.5x B200" equivalence claim (consumer-class Blackwell vs datacenter part) | IGNORE — hardware-comparison opinion; no harness/config content; doesn't touch our A17 blocker |

All other threads (#728934 Opus-5 30%, #727629 schema-harness 99%, #727505 Yakunin, #728350,
#728299 Busya scorer, #728220 arc_agi 0.9.9, #728210, #727119 500-submissions, #684625 pinned
vLLM-hang, #697720 accelerators) show last-comment timestamps ≥2d — unchanged since the 07-26
sweep; all prior verdicts stand. Host threads: NO new announcements. Still ZERO community intel on
Qwen2.5-VL tool-call format under vLLM — the A17 blocker remains ours alone to solve (v4
fenced-python recovery adapter already staged + smoked per 07-26 brief).

## LEADERBOARD (CLI top-20, 2026-07-27)

No cutoff movement vs 07-26:
- #1 KOJIMA **1.86** (last sub 07-26 00:11) — unchanged.
- #2 Tecnod8.AI 1.61 · #3 DhanaLakshmiMalla 1.60 · #4 ippeiogawa 1.58 · #5 Yuchen20 1.58 ·
  #6–8 @1.56 (Mathurin, anngle, **NoOneAhead — fresh sub 07-27 03:10, score unchanged**) ·
  #9–10 @1.54 · #11–12 @1.50.
- **Gold cutoff still ≈1.49 (top-13, Yan Zhang — fresh sub 07-27 00:35, score unchanged).**
- #16 暗黑AGI/boristown 1.47 (sub 07-26) — public ceiling unchanged.
- Wall (top-25 tail) still ~1.44; us @1.33 ~#50+, gap to gold +0.16.

Two teams at/near the gold line resubmitted overnight without score change — churn without
movement.

## PLAN IMPACT

None. The feed is fully quiet (the single "new" item was a comment that its author deleted before
we could read it; the only other unlogged comment is off-topic hardware griping), the gold cutoff
(1.49), wall (1.44), public ceiling (boristown 1.47) and #1 (1.86) are all exactly where they were
yesterday, and no community intel emerged on the Qwen2.5-VL tool-call format defect that gates
A17. Today's plan (per 07-26 brief: R20 relaunch → ratify amendment + authorize canary v4 with the
fenced-python recovery adapter; frozen-fork filler heads the queue) proceeds unmodified — no
ADOPT/ADAPT items from this sweep, defensive boristown-1.47 diff posture and vLLM-hang
concurrency<8 + watchdog ADAPT both carry unchanged.
