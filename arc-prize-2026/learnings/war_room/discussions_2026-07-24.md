# Discussion Sweep — 2026-07-24

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline for dedup: discussions_2026-07-23.md. Yesterday's state: #728299 Busya scorer
dissection (ADOPT, depth >> efficiency, 1.15x per-level cap), #727505 Yakunin ADAPT,
#728220 arc_agi 0.9.9 bump watch, LB compressed into 1.44–1.60 band, us #45 @ 1.33.

Fetch method: chrome-devtools MCP (new_page → navigate → a11y snapshot + innerText
evaluate) — worked. Read recent feed front page, threads #728210 / #684625 / #697720,
Code tab (Hotness + sorted-by-score), and live leaderboard top 49.

---

## HEADLINE — Scott Le Grand: vLLM SILENTLY HANGS on RTX Pro 6000 with the duck notebook

### #684625 (pinned "How to get started"), comment by Scott Le Grand, 1d ago
Verbatim: "VLLM silently hangs on RTX Pro 6000 instances with the duck notebook after
15-20 minutes with at least 8 or 25 concurrent sessions. I can reproduce this locally on
my own RTX Pro 6000 and machine. This seems like doomsday for any submission to the test
set and I am trying to diagnose the root cause." (Also his 4d-ago comment: no telemetry
for failed submissions — known pain, no news.)

**Verdict: ADAPT (ops-critical, today).** Directly threatens both live lanes:
- **A17 72B-VL screen (pushing today):** vLLM on the exact same GPU. If the hang is
  concurrency-triggered (≥8 concurrent sessions), our screen must pin low concurrency
  (or serialize game sessions) and add a liveness watchdog + per-request timeout so a
  silent hang degrades to partial score instead of ERROR/0.00.
- **Depth-budget lane (283–412 actions/game on g50t/m0r0/sk48, tr87):** long runtimes ×
  many actions = maximum exposure to a 15-20-min-in hang. Same mitigation: cap concurrent
  envs, heartbeat log, hard per-level wall-clock budget with graceful skip.
- Our frozen 1.33 filler is duck-harness family — verify its session concurrency is
  below the reported trigger (it has scored reliably 89 times, so likely <8, but assert
  it in preflight rather than assume).
- Reproducible locally on his own RTX Pro 6000 ⇒ genuine vLLM/driver-level issue, not
  Kaggle flakiness. Watch for a root-cause follow-up from him; if he posts a fix
  (e.g. vLLM version pin, disable prefix caching, NCCL/env var), adopt immediately.

---

## NEW since 2026-07-23 — Discussion feed (rest): essentially QUIET

### #728210 "A clarification for the input that enters the agent, FOR THE SAKE OF BETTER SCORE" — Maren Sajdaras (2d, 0 comments, -2 votes)
Beginner confusion about how the 64x64 frame maps to on-screen movement. Missed in
yesterday's sweep (posted late 07-22). **Verdict: IGNORE.** No content.

### No other new threads.
#727629 (schema-harness 99%), #727505 (Yakunin), #728278 (100% realistic?) each picked up
a trailing comment from Doruk Doğrular (2d ago) — drive-by remarks, no new substance;
verdicts from 07-23 stand (IGNORE / ADAPT-validation / IGNORE). #728299 (Busya scorer)
and #728350 unchanged, 0 comments.

### #697720 (pinned "Update on accelerators") — two 3d-ago comments, host-silent
- Scott Le Grand: complaint that the only available HW caps you at quantized <100B
  models. **IGNORE** — restates our A17 read (72B-AWQ is the practical ceiling).
- S. Brodehl: re-reports the RTX-6000-Pro accelerator dropdown missing (can only select
  TPU). Still no staff reply. **Carry the ops-ADAPT:** keep the preflight GPU-flag assert
  and verify the accelerator dropdown before today's A17 build.

---

## Code tab — public ceiling is now firmly 1.47; one hype title to watch

Sorted by score, top public artifacts:
- **boristown 【暗黑AGI】duck-harness-fast-eval — 1.47, gold, 144 upvotes (was 138),
  updated 2d ago.** The top public artifact. This, not the 1.39, is now the reference for
  the defensive diff: **upgrade yesterday's ADOPT — diff our frozen fork vs the 1.47
  config** (potential low-risk +0.14, would move us ~#49 → ~#15-equivalent band).
- zoli800 taaf-duck resubmission — 1.39 (updated 1d); cottaar original resub 1.34;
  zoli800 "just resubmission" 1.33 (= our score, public).
- "Tufa Labs duck harness [June 30 milestone winner]" re-ran 14h ago (scored copy = 1.25);
  no new score posted. Not the feared full milestone-code drop.
- **"ARC-AGI-3 Deep Reasoning Agent (179/183 levels)" — updated 1h ago, UNSCORED,
  1 upvote.** Title claims 179/183 levels. **WATCH (IGNORE until scored)** — smells like
  hype/local-claim; if it posts a score >1.47 it becomes an immediate ADOPT-diff item.
- Rest is ≤1.25 duck derivatives + retrieval spam. Nothing else new above 1.39.

---

## Leaderboard (live top-49 snapshot, 2026-07-24)

- #1 YUTO KOJIMA **1.86** (48 subs, active 12h) — unchanged, opaque.
- #2 Tecnod8.AI 1.61 · #3 DhanaLakshmiMalla **1.60 with only 6 entries** · #4 ippeiogawa
  1.58 · #5 Mathurin Ache 1.56 (idle 8d) · #6 anngle 1.56 · #7 NoOneAhead 1.56 ·
  #8 paul 1.54 · #9 Seok 1.54 · #10-11 @1.50 · #12 Yan Zhang 1.49 · #13 "Artificial
  General illusion" **1.49 with only 5 entries**. Gold cutoff ≈ **1.49** (top 13).
- #14 hiranorm 1.48 · **#15 boristown 1.47** (= his public notebook) · #16-18 @1.46 ·
  #19 Yuchen20 1.45 · **#20 Tufa Labs 1.45** · #21-24 @1.44 (incl. "Figuring out ARC
  AGI") · #25 1.43 · #28-31 @1.40 · #32 1.39 · … the 1.34-1.40 band now holds ranks
  28-45 (thickening below the wall too).
- **US (Canivel) rank 49, 1.33** (89 entries, last sub 12h ago). **Slid #45 → #49
  overnight at unchanged score.** Tied at 1.33 with #46 Samrish B, **#47 Jack Cole
  (MindsAI, 89 entries)**, #48 "today" (118 entries). Rank bleed ≈ 4 places/day now.
- **Clone watch:** the two low-entry high scores (1.60 @ 6 entries, 1.49 @ 5 entries)
  exceed every public artifact (max public = 1.47), so they are *not* pure clones of the
  duck notebooks — either light mods of boristown-1.47 or independent. No new entries
  above 1.39 that look like pure copies of the 1.39/1.47 public artifacts; instead the
  public 1.47 itself is seeding the 1.44-1.47 band.

**Read:** field keeps compressing from below AND above us. Standing on 1.33 loses ~4
ranks/day; silver floor is approaching. The single highest-EV defensive move remains a
config diff of our frozen fork vs the public boristown 1.47 (same duck family), while
the offensive EV stays with the depth-budget lane per the #728299 scorer math.

---

## Watch-item status
- **vLLM silent hang on RTX Pro 6000 (#684625, Scott Le Grand): NEW, OPEN.** Concurrency
  ≥8 sessions, 15-20 min in, reproducible off-Kaggle. Mitigate in today's A17 push (low
  concurrency + watchdog + per-level wall-clock); watch thread for root cause.
- **arc_agi 0.9.9 version bump (#728220):** unchanged, host-silent — keep version assert.
- **RTX-6000-Pro accelerator dropdown (#697720, Brodehl):** re-reported 3d ago, still
  host-silent — keep GPU-flag assert + manual dropdown check pre-build.
- **1.15x efficiency cap:** remains RESOLVED (07-23, via #728299 code read).
- **"Deep Reasoning Agent 179/183" notebook:** unscored — re-check tomorrow.

## Net verdict for the daily brief
Discussion feed is quiet (only real new thread = IGNORE beginner post), but two items
matter: (1) **Scott Le Grand's reproducible vLLM silent-hang on RTX Pro 6000 under the
duck harness at ≥8 concurrent sessions — ADAPT into today's A17 72B-VL screen and the
depth-budget lane as concurrency caps + watchdog + wall-clock budgets**; (2) leaderboard
erosion accelerated — we fell #45 → #49 at unchanged 1.33 (now tied with Jack Cole/
MindsAI), gold cutoff ≈1.49, and the top public artifact is boristown's 1.47 gold
notebook (144 upvotes) — the defensive diff target upgrades from the 1.39 to the 1.47.
No milestone-code drop, no new public artifact above 1.47, no pure clones above 1.39.
