# Discussion Sweep — 2026-07-26

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline for dedup: discussions_2026-07-25.md. Yesterday's state: ONE new thread (Opus 5 30%,
IGNORE-for-config); Scott Le Grand vLLM silent-hang (#684625, ADAPT, no root cause); public
ceiling firmly 1.47 (boristown duck-harness-fast-eval, gold); "Deep Reasoning 179/183" UNSCORED
(WATCH); we slid ~#50-53 @1.33; gold cutoff ~1.49.

Fetch method: chrome-devtools MCP. Had to kill 8 stale chrome-devtools-mcp chrome processes
(profile lock on C:\Users\dcani\.cache\chrome-devtools-mcp\chrome-profile) via CIM CommandLine
filter, then new_page → navigate → evaluate_script innerText. Read recent feed front page, threads
#728934 / #728278 / #684625, Code tab (score-sorted), and live leaderboard top-49. NOTE: this
browser context is NOT signed in, so the "my rank" marker and Raw-Data CSV are gated — exact rank
below the visible #49 could not be resolved (same wall as yesterday). LB table renders only through
#49; the 50-1925 block stays collapsed and un-expandable in an anonymous session.

---

## HEADLINE — Feed is QUIET. No genuinely new thread. Two threads gained comments; both off-harness.

### #728278 "Is 100% Accuracy Realistic With the Available Compute?" — NOW HIGH RELEVANCE to A17
Thread grew to 10 comments; new tail comments (18h ago). This is the most on-topic thread for our
active A17 lane (27B-VL vs 72B-class swap). Substance worth logging:
- **OP (OverfitOracle, 170th):** explicitly frames the exact regime we're in — "single RTX Pro 6000
  Blackwell … limited to models around the Qwen 3.6 27B class if we want sufficient memory for long
  contexts, environment state, multiple rollouts." Argues a 27B may be capability-capped and that
  frontier-tier (Kimi K3 / GLM 5.2 / DeepSeek V4 Pro) on 200-250+GB VRAM is what unlocks high acc.
- **OverfitOracle (later):** claims **Kimi K3 hit ~70% on the public data** with a harness "specified
  for your model," but concedes it "required a lot of tokens" and "a lot of money to test" — i.e.
  off-node paid API, NOT the Kaggle sandbox.
- **Scott Le Grand (18h):** "I'm sitting at about 36% on the public games after 2 weeks of effort so
  that doesn't seem unreachable." (public-games / local-eval framing, not scaled LB.)
- **Ivan Martin Valle:** "The competition does not assume the use of LLMs." (contrarian; ignore.)
- **hwe owe:** trying `airllm` to run Kimi K3 (VRAM-offload hack; impractical latency on-node).

**Verdict: ADAPT (intel), not adopt.** Reinforces the strategic thesis behind A17 — that the 27B is
the binding capability constraint and a larger model is the axis that scales. BUT the 70%/36% numbers
are public-games / paid-API regime, not the g4-standard / 9h / single-RTX-Pro-6000 quantized sandbox
where the scaled top is still 1.86. No adoptable config here, and it does NOT resolve our actual A17
blocker (tool-call format: 72B emits markdown-fenced python vs hermes `<tool_call>`). Directionally
it says: getting the 72B-AWQ working is worth the fight; capability headroom is real. Still zero
community intel on Qwen2.5-VL tool-calling / vLLM tool-parser — the one thing we most need.

### #728934 "Claude Opus 5 achieves 30% on ARC-AGI-3" — Scott Le Grand comment added (21h ago)
New comment from Scott Le Grand: *"But this is the 25 game private validation set, no? … Neither of
these Frontier lab models can be run under the constraints of 9 hours on a single RTX Pro 6000 so
apples and oranges, no?"* — an independent confirmation of yesterday's read.
**Verdict: IGNORE for config (UNCHANGED).** Scott's comment nails it: the 30% is an off-harness
Anthropic API eval; frontier models can't run under the 9h/single-RTX-Pro-6000 sandbox constraint.
Still directional support for the reasoning-depth axis (tr87 lane). Nothing adoptable.

---

## STANDING ITEMS — verified

### 1. #684625 vLLM silent-hang (Scott Le Grand) — UNCHANGED, still OPEN, no root cause
Scott's hang comment is the SAME 3-day-ago post, verbatim: *"VLLM silently hangs on RTX Pro 6000
instances with the duck notebook after 15-20 minutes with at least 8 or 25 concurrent sessions. I can
reproduce this locally … This seems like doomsday for any submission to the test set."* No follow-up,
no root-cause, no fix. **Carry ADAPT unchanged:** concurrency <8 sessions + liveness watchdog +
per-level wall-clock budget in the A17 72B-VL screen and depth-budget lane. The "8 or 25 concurrent
sessions" detail directly supports the concurrency-cap mitigation. Keep watching for his root cause.

### 2. boristown 【暗黑AGI】duck-harness-fast-eval — 1.47, UNCHANGED score, upvotes 151→165
Still the top public artifact, still 1.47 gold, updated 4d ago (no new version). **Defensive-diff
target stands (ADOPT the diff, not the whole notebook):** config-diff our frozen 1.33 fork vs this
1.47 (same duck family) for a low-risk delta. No public artifact anywhere above 1.47.

### 3. "Deep Reasoning Agent (179/183 levels)" — STILL UNSCORED
Absent from the score-sorted Code tab (only scored notebooks listed). **WATCH / IGNORE until it posts
a score >1.47.** Re-check tomorrow.

### 4. Host announcements — NONE new.
Pinned host threads unchanged. "3 MONTHS TO GO" banner = ~early-Nov final deadline (consistent with
~Nov 2). "Update on accelerators" last host activity old (5d, Scott Le Grand comment). arc_agi 0.9.9
availability (#728220 Imed Magroune, 4d) still host-silent — keep version assert. RTX-6000-Pro
accelerator questions (#697944) host-silent — keep GPU-flag assert + dropdown check pre-build.

### 5. Anything else new? Feed otherwise QUIET.
All other threads (#727629 schema-harness 99%, #727505 Yakunin, #728299 Busya scorer, #728350,
#728210, #727119 500-submissions, #728220 arc_agi 0.9.9) unchanged since 07-25 — no new substantive
comments. Prior verdicts stand.

---

## Code tab — public ceiling still 1.47 (unchanged), no new artifact above it

Score-sorted top public artifacts (2026-07-26):
- **暗黑AGI/boristown 【暗黑AGI】duck-harness-fast-eval — 1.47, gold, 165 upvotes (was 151),
  updated 4d ago.** Top public artifact. Defensive-diff target.
- taaf-duck-harness resubmission (from 暗黑AGI) — 1.39 (updated 3d, bronze).
- cottaar taaf-duck resubmission — 1.34 (gold, 133 up).
- "just resubmission, rn working on experiments" — 1.33 (= our public-equivalent).
- taaf-duck-harness-kaggle — 1.30; Tufa Labs milestone winner — 1.25; rest ≤1.25 duck derivatives.
  **No milestone-code full drop, no artifact >1.47.**
- "Deep Reasoning Agent (179/183 levels)" — UNSCORED (not in score-sorted view).

---

## Leaderboard (live PUBLIC top-49 snapshot, 2026-07-26; anonymous session, rank>49 gated)

- #1 YUTO KOJIMA **1.86** (50 subs, 12h) — unchanged, opaque, still runaway leader.
- #2 Tecnod8.AI 1.61 · #3 DhanaLakshmiMalla **1.60 (8 entries)** · #4 ippeiogawa 1.58 ·
  **#5 Yuchen20 1.58 (14 entries) — up from ~1.45, +0.13 jump** · #6 Mathurin Ache 1.56 (idle 10d) ·
  #7 anngle 1.56 · #8 NoOneAhead 1.56 · #9 paul 1.54 · #10 Seok Jeongeum 1.54.
- #11 Dinesh kumar 1.50 · #12 Mohammad Saadati 1.50.
- **Gold cutoff ≈ 1.49 (top 13):** #13 Yan Zhang 1.49 (17 entries) · #14 "Artificial General
  illusion" **1.49 (5 entries)** (just outside, tied) · #15 hiranorm 1.48.
- **#16 暗黑AGI/boristown 1.47** (24 entries) = his public notebook (was #15 yesterday).
  #17-20 @1.46 (Biubiu, MLRush, Arunodhayan, Kochi Loki) · #21 Tufa Labs 1.45 · #22-26 @1.44 ·
  #27 1.43 · #28 1.42 · #29 1.41 · #30-33 @1.40 · #34 1.39 · #35-36 1.38 · #37-38 1.37 · #39 1.36 ·
  #40-42 @1.35 · #43-47 @1.34 (SCU, Surya, **Rokaiya Somapti**, blatant warrior, Peter).
- **1.33 shelf THINNED to two visible rows:** #48 Samrish B 1.33 · **#49 Jack Cole (MindsAI, 90
  entries) 1.33** — last row before the collapsed "50 - 1925" block.
- **US (Canivel) @ 1.33** sits inside the collapsed 50-1925 block → **~#50+** (exact rank not
  resolvable anonymously; consistent with yesterday's ~#50-53 estimate). The 1.34 band that was
  #42-46 yesterday firmed up; several 1.33 teams from yesterday climbed to 1.34, pushing the 1.33
  floor (and us) down further.

**Read:** field keeps compressing at the top (Yuchen20 +0.13 into #5; four teams now ≥1.56) while the
1.33 floor thins as neighbors climb to 1.34 — net erosion for us at unchanged score. Gold wall now
1.49 (top-13), a full +0.16 above our 1.33. Highest-EV defensive move remains the boristown-1.47
config diff (same duck family, low-risk ~+0.14 toward the 1.44-1.47 pack). Offensive EV stays with
the A17 72B-VL swap + tr87 depth-budget lane, reinforced today by #728278 (27B is the binding
capability cap; larger model is the axis that scales) and the Opus-5 heavy-reasoning 30% datapoint.

---

## Watch-item status
- **#728278 "100% Realistic" (10 comments): NEW SIGNAL — ADAPT (intel).** Validates A17 thesis
  (27B capability-capped; frontier/larger model scales); Kimi-K3-70% and Scott-36% are off-node/
  public-games regime, not adoptable; NO Qwen2.5-VL tool-call intel (our real A17 blocker).
- **Claude Opus 5 30% (#728934): IGNORE for config (UNCHANGED)** — Scott Le Grand's new comment
  independently confirms off-harness (can't run frontier under 9h/single-RTX-Pro-6000); directional
  support for reasoning-depth axis only.
- **vLLM silent hang (#684625): OPEN, no change** — carry ADAPT (concurrency<8 + watchdog +
  wall-clock) into A17 build; "8 or 25 concurrent sessions" detail supports the cap. Watch for root cause.
- **boristown 1.47:** unchanged score (upvotes 151→165); defensive-diff target stands.
- **"Deep Reasoning Agent 179/183":** still UNSCORED — re-check tomorrow.
- **arc_agi 0.9.9 (#728220):** host-silent — keep version assert.
- **RTX-6000-Pro accelerator dropdown:** host-silent — keep GPU-flag assert + dropdown check pre-build.

## Net verdict for the daily brief
Quiet feed, zero genuinely new threads — two existing threads gained comments, both off-harness.
Most relevant is #728278 (100% realistic): ADAPT-as-intel — it strongly validates the A17 thesis that
the single-27B on ~100GB is the binding capability cap and that a larger model is the scaling axis
(OP explicitly names the Qwen-27B regime; Kimi-K3 reportedly ~70% on public data, Scott ~36% — both
off-node/paid, NOT the Kaggle sandbox), but it gives ZERO Qwen2.5-VL tool-calling / vLLM tool-parser
intel, which remains our actual A17 blocker (72B emits markdown-fenced python vs hermes <tool_call>).
Opus-5 30% thread: Scott Le Grand's new comment independently confirms IGNORE-for-config (off-harness).
vLLM hang unchanged (ADAPT holds). Public ceiling still 1.47 (boristown, 165 up, defensive-diff
target); no artifact >1.47; "Deep Reasoning 179/183" still unscored. LB: top compresses (Yuchen20
jumps to #5 @1.58; four teams ≥1.56), 1.33 floor thins as neighbors climb to 1.34 — we erode to ~#50+
at unchanged 1.33; gold cutoff 1.49 (top-13); #1 KOJIMA 1.86.
