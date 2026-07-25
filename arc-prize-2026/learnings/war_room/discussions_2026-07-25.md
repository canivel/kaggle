# Discussion Sweep — 2026-07-25

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline for dedup: discussions_2026-07-24.md. Yesterday's state: HEADLINE = Scott Le
Grand vLLM silent-hang on RTX Pro 6000 (#684625, ADAPT); public ceiling firmly 1.47
(boristown duck-harness-fast-eval, gold); "Deep Reasoning Agent 179/183" UNSCORED (WATCH);
we slid #45→#49 @1.33; gold cutoff ~1.49.

Fetch method: chrome-devtools MCP (had to kill 8 stale chrome-devtools-mcp instances
first, then new_page → navigate → evaluate_script innerText). Read recent feed front page,
threads #728934 / #684625, Code tab (score-sorted + hotness), and live leaderboard top-50.

---

## HEADLINE — Only one genuinely NEW thread: Claude Opus 5 / 30% claim (off-harness)

### #728934 "Claude Opus 5 achieves 30% on ARC-AGI-3" — Geremie Yeo (17h ago, 3 votes, 2 comments)
Body: "Today Anthropic released Claude Opus 5 and it was reported to have a score of 30%
on the ARC AGI 3 benchmark, >3x more than the next best. Very interesting. The top score
is still <2% here." Links an Anthropic/@claudeai X post.
Comments:
- Doruk Doğrular (10h): "I think it was using another machine, not g4 standard 48. So they
  calculate cost not time. I am not sure. Does anyone know?"
- Van-Phuc Huynh (32nd, 10h): "I hope we'll see a score above 10% here within the next two
  weeks."

**Verdict: IGNORE (for our lanes) / note-worthy context.** The 30% is Anthropic's OWN
ARC-AGI-3 benchmark eval via API (unconstrained compute, cost-based), NOT this Kaggle
harness (g4-standard-48, quantized <100B ceiling, time-limited — where raw top is still
<2% / 1.86 scaled). It does not change our config: we cannot run Opus 5 on-node (API calls
are disallowed in the Kaggle sandbox; ceiling is a local quantized VLM ~72B-AWQ). It IS a
useful data-point that a frontier model with heavy inference-time reasoning crushes the
task — mild reinforcement that the depth-budget/reasoning direction (our tr87 lane) is the
right axis, but nothing directly adoptable. Doruk's "cost not time" note is correct and
reinforces that the public LB's constrained-compute regime is a different game.

---

## STANDING ITEMS — verified, NO change since 07-24

### 1. #684625 vLLM silent-hang (Scott Le Grand) — UNCHANGED, still OPEN, no root cause
Scott's hang comment is the SAME 2d-ago post (verbatim identical to yesterday). No
follow-up, no root-cause, no fix posted. Thread's only other new-ish activity is old
1.15x-cap Q&A (already resolved 07-23). **Carry yesterday's ADAPT verdict unchanged:**
concurrency <8 sessions + liveness watchdog + per-level wall-clock budget in the A17 72B-VL
screen and depth-budget lane. Keep watching for his root-cause post.

### 2. boristown 【暗黑AGI】duck-harness-fast-eval — 1.47, UNCHANGED score, upvotes 144→151
Still the top public artifact, still 1.47 gold, updated 3d ago (no new version since
yesterday). **Defensive-diff target stands (ADOPT the diff, not the whole notebook):**
config-diff our frozen 1.33 fork vs this 1.47 (same duck family) for a low-risk delta.
No public artifact anywhere above 1.47.

### 3. "Deep Reasoning Agent (179/183 levels)" — STILL UNSCORED (updated 1d ago, 1 upvote)
Absent from the score-sorted Code tab (which only lists scored notebooks); present only on
the hotness view with no score field. **WATCH / IGNORE until it posts a score >1.47.**
Re-check tomorrow.

### 4. Host announcements — NONE new. Pinned host threads unchanged (Update on accelerators
last host activity is old; "3 MONTHS TO GO" banner = ~early-Nov final deadline, consistent
with ~Nov 2). arc_agi 0.9.9 bump (#728220) still host-silent — keep version assert.
RTX-6000-Pro accelerator dropdown issue — no host reply — keep GPU-flag assert + manual
dropdown check pre-build.

### 5. Anything else new? Feed is otherwise QUIET.
- #728278 "Is 100% Accuracy Realistic With the Available Compute?" picked up a trailing
  comment (hwe owe, 3h ago) — drive-by, no substance. IGNORE (unchanged verdict).
- All other threads (#727629 schema-harness 99%, #727505 Yakunin, #728299 Busya scorer,
  #728350, #728210, #727119 500-submissions-analyzed) unchanged since 07-24 — no new
  substantive comments. Verdicts from prior sweeps stand.

---

## Code tab — public ceiling still 1.47 (unchanged), no new artifact above it

Score-sorted top public artifacts (2026-07-25):
- **boristown 【暗黑AGI】duck-harness-fast-eval — 1.47, gold, 151 upvotes (was 144),
  updated 3d ago.** Top public artifact. Defensive-diff target.
- taaf-duck-harness resubmission (from 暗黑AGI) — 1.39 (updated 2d).
- cottaar taaf-duck resubmission — 1.34 (gold, 131 up).
- "just resubmission, working on experiments" — 1.33 (= our public-equivalent score).
- taaf-duck-harness-kaggle — 1.30; Tufa Labs milestone winner — 1.25; rest ≤1.25 duck
  derivatives. **No milestone-code full drop, no artifact >1.47.**
- "Deep Reasoning Agent (179/183 levels)" — UNSCORED (hotness view only).

---

## Leaderboard (live PUBLIC top-50 snapshot, 2026-07-25)

- #1 YUTO KOJIMA **1.86** (48 subs, 2d) — unchanged, opaque.
- #2 Tecnod8.AI 1.61 · #3 DhanaLakshmiMalla **1.60 (7 entries)** · #4 ippeiogawa 1.58 ·
  #5 Mathurin Ache 1.56 (idle 9d) · #6 anngle 1.56 · #7 NoOneAhead 1.56 · #8 paul 1.54 ·
  #9 Seok 1.54 · #10 Dinesh kumar 1.50 · #11 Mohammad Saadati 1.50.
- **Gold cutoff ≈ 1.49 (top 13):** #12 Yan Zhang 1.49 (16 entries) · #13 "Artificial
  General illusion" **1.49 (5 entries)** · #14 hiranorm 1.48 (first below gold).
- **#15 暗黑AGI/boristown 1.47** (24 entries, active 10h) = his public notebook.
  #16-19 @1.46 · #20 Yuchen20 1.45 · #21 Tufa Labs 1.45 · #22-25 @1.44 · #26-27 @1.42-43 ·
  #28 @1.41 · #29-32 @1.40 · #33 1.39 · #34-35 1.38 · #36-37 1.37 · #38 1.36 · #39-41 1.35.
- **1.33-1.34 band now holds ~#42-49:** #42-46 @1.34 (SCU, Surya, Rokaiya Somapti,
  "blatant warrior", Peter) · **#47 Samrish B 1.33 · #48 Jack Cole (MindsAI, 89 entries)
  1.33 · #49 "today" (119 entries) 1.33.**
- **US (Canivel) @ 1.33** now sits just BELOW the visible #47-49 tied cluster — i.e.
  ~#50-53 (fell out of the loaded top-50; couldn't resolve exact rank, LB API 400'd).
  Continued erosion at unchanged score, ~4 ranks/day trend intact. We are now tied-or-just-
  behind Jack Cole/MindsAI at the very bottom of the 1.33 shelf.

**Read:** field keeps compressing; 1.33 is now a crowded floor (~8 teams at 1.33-1.34) and
we've slipped under the visible cutoff. Gold wall is 1.49 (top-13), a full +0.16 above us.
Highest-EV defensive move remains the boristown-1.47 config diff (same duck family,
low-risk ~+0.14 toward the 1.44-1.47 pack); offensive EV stays with the tr87 depth-budget
lane (reinforced today by the Opus-5 heavy-reasoning 30% datapoint — the reasoning-depth
axis is the one that scales).

---

## Watch-item status
- **Claude Opus 5 30% (#728934): NEW — IGNORE for config** (off-harness API eval, not
  runnable on-node), but note as directional support for the depth/reasoning axis.
- **vLLM silent hang on RTX Pro 6000 (#684625): OPEN, no change** — carry ADAPT
  (concurrency<8 + watchdog + wall-clock) into A17 build; watch for root cause.
- **boristown 1.47:** unchanged (upvotes 144→151); defensive-diff target stands.
- **"Deep Reasoning Agent 179/183":** still UNSCORED — re-check tomorrow.
- **arc_agi 0.9.9 bump (#728220):** host-silent — keep version assert.
- **RTX-6000-Pro accelerator dropdown:** host-silent — keep GPU-flag assert + dropdown
  check pre-build.

## Net verdict for the daily brief
Feed is quiet: ONE new thread (#728934 Claude Opus 5 30% on ARC-AGI-3) — IGNORE for our
config (it's Anthropic's own unconstrained API eval, not the Kaggle quantized/time-limited
harness; top here still <2% raw / 1.86 scaled), but it directionally validates the
reasoning-depth axis behind our tr87 lane. No new standing-item movement: Scott Le Grand's
vLLM hang unchanged (still no root cause, ADAPT holds), public ceiling still 1.47 (boristown,
now 151 up, defensive-diff target), "Deep Reasoning 179/183" still unscored (WATCH), no host
announcements, no artifact >1.47. LB erosion continues — we're now just below the visible
#47-49 1.33 cluster (~#50-53), gold cutoff 1.49 (top-13), #1 YUTO KOJIMA 1.86.
