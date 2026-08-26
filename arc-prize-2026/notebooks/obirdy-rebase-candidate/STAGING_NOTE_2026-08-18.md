# STAGED: obirdy re-base candidate — 2026-08-18 (graft lane, coordinator-authorized byte-audit + staging; NO push, NO re-base tonight)

**Source:** `obirdy/arc3-duck-qwen-3-8-visible-memory-candidate` (public, last run 2026-08-15 20:50:50; pulled 08-17, re-audited 08-18). Files here are the pulled bytes, unmodified — kernel-metadata.json still carries obirdy's slug on purpose; re-basing requires a deliberate retarget + `feedback_kaggle_env_match` pass.

## Byte-audit result: CLEAN
- **Structure:** same duck family as our frozen fork — 17 cells, 8 code. Diffs vs `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb`: code cells **[6, 8, 12]** (code ordinals [2,3,5]) + markdown [0, 11, 13] (documentation only).
- **Cell 6:** `DATASET_SOURCES` engine swap only — `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` → `saltb0x/qwen3-8-27b-fp8`.
- **Cell 8:** setup-command rewrite to serve Qwen3.8 (`saltb0x/Qwen3.8-27B-FP8`), **fail-loud**: raises if expected literals are absent or the qwen3_coder/--reasoning-parser flags are lost; asserts the dataset mount exists.
- **Cell 12:** prompt-only "visible world-model update contract" — appends a mandatory visible-update instruction to `ToolAgent`'s system prompt (an independent implementation of forum 734843's hidden-reasoning memory-loss fix), preflight-asserted, audit JSON written to working dir. Its own audit declares `request_or_response_transport_modified: false`.
- **Footguns: NONE.** Token parity with the frozen fork is exact (`TRUE_SUBMISSION` 11=11, `reset` 1=1); `banking`/`transfer`/`scorecard`/`external_game_id` occur ZERO times. No env-wrapper driving, no scorecard mechanics, no dup-gates.
- **Metadata:** identical env to the frozen fork (docker sha `57e612b…`, `NvidiaRtxPro6000`, GPU on, internet off, competition attached); only the engine dataset differs.

## Attribution and expectation (per coordinator ruling, 08-18)
- Attribution **[INF]**: author (Manan Gupta #2, teamId 16567629) posted **1.48 on 2026-08-16 17:19** — the day after this kernel's run; this is the ONLY public duck-lineage kernel that *precedes* its author's score. Not proven to be the scoring artifact.
- Expectation if re-based: **+0.15** over our public max 1.33. Modest, honestly stated.
- **Tension to resolve before any re-base:** our sealed Q38 engine-swap eval was REFUTE-2× on levels; this kernel = Q38 engine + the visible-updates contract. If it outperforms, the interesting variable is the CONTRACT, not the engine.
- **Decision = coordinator's.** Queue-head candidacy is for a LATER window, after tonight's graft-floor decision; it does not compete with it.

## PAIR-ARM PRE-AUTHORIZATION (coordinator, 2026-08-18 — carry into the pair prereg verbatim)

**If the 08-19 graft-floor read lands NULL-with-positive-drift (or SIGNAL on a subset suggesting the memory defect is the binding half): the pair arm — v19 flags + THIS notebook's cell-12 visible-updates contract — is PRE-AUTHORIZED as the immediate follow-up build.** Seal its prereg same-day; push next available slot; the single variable is the PAIR, documented as such. If 08-19 SIGNALs outright, the pair arm becomes the v2 iteration after the scored exploration draw. This byte-audited copy is the implementation source.

**The connection that must appear prominently in the pair prereg and its journal row:**
exp_id 17 (ACCEPT, 2026-08-18): the duck harness captures carried world-model state ONLY from the model's VISIBLE message — 97.64% of generated content routes to the HIDDEN channel under a tool-calling grammar; `goalkeep` patches the WIPE path only, never the CAPTURE path. Our own ft09 record shows the same defect from the other side: 64 turns, `content_chars=0` on every one — two hours of thinking that never became visible context. **The agent may not have been ignoring its context; much of its context never existed on the channel the harness reads.** The two published fixes are therefore complementary halves of one defect — goalkeep (wipe) + visible-updates contract (capture) — and the pair arm tests them as the pair exp 17's own conclusion prescribes. This reframes part of the runner-supplied-context null (exp_id 9) and belongs restated in the pair arm's row when it reports.
