# Discussion Sweep — 2026-07-21

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline for dedup: discussions_2026-07-18.md + daily_brief_2026-07-20.md §1b.
Threads already triaged in prior sweeps (IGNORE/covered): #727629 (schema-harness 99%),
#727505 (Constraint Before Control), #727119 (500 Submissions Analyzed), #726367 (AGI
timeline), #724841 (RLIMIT_AS), #724890 (RTX6000), #716295 (parallelism), #726903
(x(-1)/week), #725002 (Milestone#1), #726552 (run-to-run variance), #726340 (GPT-5.6 Sol
7.8%), #723792 (Kaggle Error after submission), #716711 (Akhil Gemma-4-31B 0.79).

Fetch method: WebFetch returns only the page title (Kaggle discussion is client-rendered).
WebSearch returned only competition landing pages, no thread bodies. **chrome-devtools MCP
(new_page + navigate + a11y take_snapshot) successfully read the full recent-comments list
page and individual thread comment trees** — this is the working method, same as 07-18.

---

## NEW since 2026-07-20

The recent-comments front page shows only two items with comment activity inside the
07-20→07-21 window. **No new standalone threads. No new public notebook >0.79. No YUTO
KOJIMA (LB #1, 1.86) footprint anywhere.** Both new comments are on old pinned/getting-
started threads, not new OPs.

### 1. #684625 "How to get started + Official Discord" — NEW comment: Scott Le Grand (~19h ago)
Full text: *"Really wish there were more telemetry about failed submissions. It's next to
impossible to debug when it's a black box."* One-line lament, no new info.
**Verdict: IGNORE.** Pure sentiment; the silent-failure black box is already fully covered
by our 07-18 triage of #727119 (500-submissions error taxonomy) and #724841 (Greg's rerun
death-mode constants: exit 139=SIGSEGV, 10MB log cap, 20GB /working, 30GB cgroup). Nothing
actionable beyond preflight items already logged. Reinforces prior: the ~1/3 "silent stuck"
failure bucket is the community's #1 pain — our watchdog wall-clock-deadline + GPU-flag
asserts remain the right response.

### 1b. (surfaced while reading #684625 — OLDER comments, not previously logged, scoring-relevant)
On the same thread, an UNRESOLVED host-tagged question about the scoring cap:
- Sirish Somanchi + Hendrik Nowak flag a **discrepancy in the per-level efficiency cap**:
  docs.arcprize.org/methodology says **1.15x** (new; also changes human baseline from
  second-best to (upper) median), while the Kaggle **/data "Scoring Method"** page AND
  arXiv:2603.24621 say **1.0x**. Nowak: "Since it hasn't been announced on Kaggle, I'm
  assuming it hasn't been implemented here yet." No host has answered (@gregkamradt /
  @macruzbar / @inversion all pinged, silent).
**Verdict: ADAPT (watch-item, scoring-model).** Directly governs how our efficiency score
maps to LB: if Kaggle is still on **1.0x**, per-level over-efficiency is NOT rewarded beyond
parity, so chasing sub-baseline step counts on solved levels yields no LB upside — pure
level-COMPLETION (breadth) dominates, consistent with our generalization-first prior. If/when
Kaggle flips to 1.15x + median baseline, fast-solve efficiency becomes worth up to +15%/level
and the calculus shifts toward speed on already-solvable games. ACTION: keep an eye on host
answer; treat current LB math as 1.0x (completion-weighted) until Kaggle confirms otherwise.
No code change today. (This is an old unresolved item newly relevant because the sweep read
the full thread; not a 07-20→07-21 post.)

### 2. #697720 "Update on accelerators" (pinned, María Cruz) — NEW comment: S. Brodehl (~3h ago)
Thread is the RTX-6000-Pro / H100-stockout hardware-logistics pin; visible comment body
(Brodehl's new one) did not render in the a11y snapshot (Hotness sort collapses newest;
older visible comments are the g2→g4-standard-48 machine-type correction, 2mo old).
**Verdict: IGNORE.** Pinned hardware-logistics housekeeping. The RTX PRO 6000 / g4-standard-48
rail is already the confirmed environment we target (war-v4 72B-AWQ screen assumes it). No
new constraint. If Brodehl's comment turns out to raise a new hardware limit, it will resurface
on tomorrow's recent-comments scan.

---

## Standing watch (unchanged from 07-20)
- No YUTO KOJIMA methods leak; 1.86 leader stays fully opaque (Greg confirmed 07-18: hosts
  cannot read notebooks pre-open-source). No public notebook above Akhil's 0.79 (#716711).
- No host answer yet on #726552 env-seeding / run-to-run variance.
- schema-harness 99% (#727629) still IGNORE — closed-source frontier-API + keep-best-rerun,
  no code; re-triage only if code drops.
- GPT-5.6 Sol SOTA 7.8% (#726340) — external headline, not a Kaggle-runtime method; the
  substance is captured by arXiv:2607.15439 (Rodionov EWM+verification), already ADOPT-directional.

## Net verdict for the daily brief
Quiet window. Zero new threads, zero new notebooks, zero leader leak. Only two new comments,
both non-substantive (one submission-telemetry lament → IGNORE; one hardware-pin housekeeping
→ IGNORE). The one item worth carrying forward is an OLD-but-unresolved **1.15x vs 1.0x
efficiency-cap ambiguity** (ADAPT/watch): assume Kaggle LB math is still 1.0x (completion-
weighted, favors breadth/generalization — matches our prior) until a host confirms the 1.15x
switch; no code change today.
