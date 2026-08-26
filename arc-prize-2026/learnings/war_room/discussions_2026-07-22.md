# Discussion Sweep — 2026-07-22

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline for dedup: discussions_2026-07-21.md (quiet window; watch-item = 1.15x-vs-1.0x
efficiency-cap discrepancy, host-unanswered).

Fetch method: chrome-devtools MCP (new_page → navigate → a11y take_snapshot / wait_for) —
worked again; read the recent-comments front page, thread #728210, pinned #697720 comment
tree, the Code tab (Recently Created), and the public leaderboard.

---

## NEW since 2026-07-21 — Discussion feed

### 1. #728210 "A clarification for the input that enters the agent, FOR THE SAKE OF BETTER SCORE" — Maren Sajdaras (NEW thread, 31m ago, 0 comments)
Beginner question: how does the 64x64 frame map to the visually-rendered game (big square
moves, not a pixel); claims answering it would raise the whole leaderboard.
**Verdict: IGNORE.** Frame semantics (integer grid, upscaled blocks in the web renderer) are
long-solved in our stack (frozen Cottaar harness + EWM both consume raw frames); nothing new.

### 2. #697720 "Update on accelerators" (pinned) — NEW comment: Scott Le Grand (~1d ago)
*"So this benchmark is supposed to be an evaluation of whether frontier models can hit AGI
performance, but the only HW available … can at best run quantized models with <100B params."*
**Verdict: IGNORE.** Sentiment, no new constraint. If anything it confirms our A17 choice of
Qwen2.5-VL-72B-AWQ as the practical ceiling model on the RTX PRO 6000 rail.

### 3. #697720 — S. Brodehl comment now readable (was hidden in yesterday's snapshot)
*"There seems to be an issue with the RTX 6000 Pro instances. Currently, I do not have access
to them, but can select TPU instead. Any official update on usable accelerators?"* No host
reply yet.
**Verdict: ADAPT (ops-check, 1 minute).** If the RTX-6000-Pro selector is flaky platform-wide,
our free-Kaggle GPU builds (30 GPU-h/wk evidence runs) and the A17 72B-VL screen stall.
ACTION: before the next GPU build, verify the accelerator dropdown on our kernel actually
offers RTX 6000 Pro; keep the existing GPU-flag assert in preflight. Likely a per-account or
transient UI glitch (nobody else has echoed it), so watch, don't panic.

---

## NEW — Code tab / leaderboard-relevant (the real news today)

### 4. PUBLIC notebook at 1.39: zoli800 "taaf-duck-harness-kaggle-share (Resubmission 573a60…)" — Score 1.39, updated 1d ago
Fork of boristown's 【暗黑AGI】duck-harness-fast-eval, but internally it is the **Cottaar TAAF
lineage**: source bundle `jeroencottaar/taaf-kaggle-source-share` + `driessmit1/arc3-vllm-
h100-wheelhouse-v3` + model snapshot `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`
(Qwen3.6-27B FP8, vLLM). Author Chekhlov Dmitrii is rank 22 (team "Figuring out ARC AGI",
1.44 — exactly our wall). Parent author boristown is rank 13 at 1.47; a second public copy
(Sim) scored 0.81, and Danny Wu's "celestia duck literal" scored 0.66 — so the public
duck-harness family now spans 0.66→1.39 depending on config.
**Verdict: ADOPT (highest-priority item today).** Two reasons:
(a) **Direct upgrade path for our filler**: we run a frozen Cottaar duck-harness fork (LB
1.33, rank 44). Same lineage is publicly demonstrated at **1.39** — above us — with a fully
visible config (TAAF source version + 27B-FP8 vLLM model + wheelhouse). Diff their
DATASET_SOURCES/pins against our frozen fork; if the delta is just model snapshot + TAAF
version, that is a low-risk +0.06 that clears several ranks without touching our active lines.
Respect feedback_kaggle_env_match: replicate kernel-metadata (GPU flag, dataset sources,
docker image) exactly, and runtime-test before push.
(b) **Moat erosion**: anyone can now fork to ~1.39, so our 1.33 will be swallowed by a fork
wave within days. Standing still means sliding out of silver. This raises urgency on the
active lines (EWM, latent-state audit, A17) but the immediate defensive move is (a).

### 5. mbmmurad "arc-agi-3-lb-0-86-3rd-place-candidate-milestone" (surfaced via a 2d-old fork)
Public notebook advertising LB 0.86 — first public artifact above Akhil's 0.79, apparently a
Milestone-#1 3rd-place candidate (milestone rules require sharing).
**Verdict: IGNORE (superseded).** At 0.86 it is far below both our 1.33 and item 4's 1.39;
only note is that Milestone-#1 code-sharing is now materializing — expect more/better public
milestone notebooks (the 1.39 resubmission is plausibly part of the same wave). Skim only if
its method differs from duck-harness lineage.

### 6. Misc new kernels (BioVLM-New 43m, SDFT_training 25m, Graph Exploration Agent 5h, Fine-Tuning Qwen3-8b 3h, offline atlas 2d, verified world-model CPU commit 2d)
No scores or scores ≤0.92; several are off-topic spam (YOLO pose, ColBERT/BEIR).
**Verdict: IGNORE.** Nothing beats what we already run; "ARC3 Duck verified world-model CPU"
(obirdy, 0.92 baseline variant) is the only thematically adjacent one and is below our filler.

---

## Watch-item status: 1.15x vs 1.0x efficiency cap (#684625) — STILL UNANSWERED
Thread's last comment remains Scott Le Grand's 2d-old telemetry lament; no host reply to the
Somanchi/Nowak cap question. **Keep treating LB math as 1.0x (completion-weighted, breadth
dominates)** per yesterday's ADAPT. No change.

## Leaderboard glance (public, 2026-07-22)
- #1 YUTO KOJIMA **1.86** (46 entries, last sub 12h ago — still active, still opaque).
- #2 Tecnod8.AI 1.61; gold cutoff ~1.47 (rank 13, boristown); wall band 1.44 = ranks 19–22.
- Us (Canivel) **1.33, rank 44** (87 entries, last 12h). Tufa Labs (driessmit1, whose model
  snapshot powers the 1.39 notebook) rank 18 at 1.45.

## Net verdict for the daily brief
Discussion feed itself is near-quiet (1 trivial new thread, 2 pinned-thread comments). The
action is on the **Code tab**: a public Cottaar-TAAF duck-harness resubmission at **1.39**
(ADOPT — diff vs our frozen 1.33 fork; also a fork-wave threat to our rank), plus first
evidence of Milestone-#1 code-sharing (0.86, superseded). One ops ADAPT: verify RTX-6000-Pro
is selectable before the next GPU build (Brodehl access report, host-silent). Efficiency-cap
watch-item unchanged: host still silent, keep assuming 1.0x.
