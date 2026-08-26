# Discussion Sweep — 2026-07-23

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent).
Baseline for dedup: discussions_2026-07-22.md. Yesterday's state: zoli800 public
Cottaar-TAAF resubmission at 1.39 (ADOPT flagged), Milestone-#1 sharing starting (0.86),
RTX-6000-Pro access wobble (Brodehl, host-silent), efficiency-cap watch-item unanswered.

Fetch method: chrome-devtools MCP (new_page → navigate → a11y take_snapshot) — worked. Read
recent feed front page, threads #727629 / #728299 / #727505, the Code tab (Recently Run), and
the full public leaderboard (downloaded CSV, 1877 teams).

---

## HEADLINE — the efficiency-cap watch-item is effectively ANSWERED (from code, not host)

### #728299 "Reading the score exactly: finishing 4 of 6 levels scores 47.6, not 66.7" — Busya PRIME (NEW, 17h, 0 comments, verified to 1e-9)
A rigorous dissection of the *actual* shipped scorer (`arc_agi/scorecard.py`,
`EnvironmentScoreCalculator`), reproduced offline against 3 cases to 1e-9. The formula:

- **Per completed level:** `score = min((baseline_actions / actions_taken)**2 * 100, 115)`.
  Incomplete level = 0.
- **Per game:** `weighted = sum(level_score[i]*i)/sum(i)`, then `cap = sum(i over completed
  levels)/sum(i) * 100`, then `game = min(weighted, cap)`. Levels weighted by level number.
- **LB number** = plain mean of per-game scores.

Two load-bearing consequences, both confirmed:
1. **Depth >> efficiency.** An unreached level costs its weight *twice* (once in the weighted
   mean, once via the completion cap), and later levels carry the heavy weights. Example
   (game cd82, 6 levels): clearing 4/6 perfectly = **47.62**, not 66.7 — because weights
   1+2+3+4=10 of 21. Getting to level 5 moves the score far more than shaving actions off
   level 1.
2. **Overshoot is quadratic.** 2x baseline actions = **25%** of the level score, not 50%.
   Matching the human baseline = 100; per-level value is **capped at 115 (=1.15x)**, so
   beating the baseline is worth a little but capped.

**Verdict: ADOPT (highest-signal item today).**
- This **resolves the 1.15x-vs-1.0x watch-item** without a host reply: the 115 cap is the
  *per-level* ceiling (1.15x of matching-baseline), and the *game/LB* aggregate is
  completion-weighted with a completion cap. Both our prior readings were half-right — treat
  LB as completion-weighted (breadth/depth dominates) AND respect a 1.15x per-level headroom.
- **Directly re-prioritizes EWM/A17:** our objective must be *reach deeper levels*, not trim
  actions on shallow ones. Efficiency work below ~1.0x baseline is near-worthless; the payoff
  is level depth (the heavy-weighted late levels) and never over-shooting (quadratic penalty).
  This sharpens the EWM contract v1.1 objective for R17 sealing.
- Author shipped a no-API-key offline reproduction + a map of all 25 bundled games:
  kaggle.com/code/busyaprime/arc-agi-3-offline-atlas-and-scoring. **ACTION:** pull this atlas
  into our local harness as a free, deterministic scoring oracle (zero cloud spend) — lets us
  score EWM candidates offline against the exact shipped formula before any Kaggle build.

---

## NEW since 2026-07-22 — Discussion feed (rest)

### #727505 "Constraint Before Control: A Semantic Architecture for ARC-AGI-3" — Vladimir Yakunin (NEW-ish, 4d, 2 comments)
A Paper-Track architecture published early. Topology: (1) World research → (2) Semantic
reasoning (local LLM proposes hypotheses) → (3) Grounding/operationalization → (4)
Verification & control (only the verifier authorizes an official action). Adds "ternary logic
with irrelevance" (required / forbidden / irrelevant, plus unresolved) and externalized
structured memory for recursive correction. Falsifiable claims: imagined states must not
become official facts; visible change ≠ confirmed meaning; every semantic commitment stays
revisable by official evidence.
**Verdict: ADAPT (validation, no code).** This is *strikingly parallel to our EWM contract +
certification-as-resync + verifier* design — independent convergence on "predicted state is
not a fact; only verified official transitions update the model." Good outside confirmation
that our R17 topology is sound. BUT author's own results are **weak: 0.17, and only after
gutting the architecture** to "minimal research → LLM hypothesis → verify → execute"; every
richer version failed on runtime-env compatibility (echoes our own feedback_arc_kernel_
structural_drift pain). No adoptable artifact; the value is the design cross-check and the
reminder that env-integration, not concept, is the killer. Steal one framing: the **ternary
required/forbidden/irrelevant** tag for hypotheses could tighten our EWM memory (remember
irrelevant routes without turning them into universal prohibitions).

### #727629 "https://schema-harness.github.io/ pub 99%" — CreateAMind (3d, 3 comments)
A third-party site claiming ~99% on public games. Community reaction (keithtyser, 81st):
**not open source, and its "fallback rule reruns weak games with stronger models and keeps
the better score" — likely overfit to the public games**, unknown on semi-private. Yakunin
links arXiv:2602.02710 as prior art.
**Verdict: IGNORE (unverifiable + overfit-smell).** A 99% public claim with a keep-the-best
rerun fallback is exactly the public-LB overfitting our plan avoids (feedback_arc_
generalization_first). Not open, nothing to adopt. Watch only to see if it ever posts a
semi-private number.

### #728278 "Is 100% Accuracy Realistic With the Available Compute?" — OverfitOracle (12h)
Sentiment/compute-ceiling debate. **IGNORE.** No new constraint; consistent with our A17 read
that 72B-AWQ is the practical ceiling on the RTX PRO 6000 rail.

### #728220 "When will arc-agi 0.9.9 be available in Competition Notebooks?" — Imed Magroune (1d)
Ops question about the harness package version bump. **IGNORE (watch).** If the competition
runtime bumps `arc_agi` to 0.9.9 it could change the scorer/env — but no host reply yet and
no evidence of a change. Note for preflight: keep asserting the installed `arc_agi` version
in our kernel so a silent bump doesn't invalidate a build.

### #728350 "Level pass animation - snake halucination" / #728299-adjacent misc (Doruk Doğrular, 12h)
Beginner/observation posts about render artifacts. **IGNORE.**

---

## Code tab (Recently Run) — no new threat above 1.39

- **zoli800 "taaf-duck-harness-kaggle-share (Resubmission 573a60…)" — 1.39** (updated 8h ago):
  same artifact flagged yesterday; still the top *public* notebook. No new clones of *our*
  artifact appeared. Parent boristown 【暗黑AGI】duck-harness-fast-eval = **1.47, gold, 138
  upvotes** (updated 1d).
- Everything else new is ≤0.17 or off-topic: Yakunin "LCLD Qwen V9" 0.14, Pranshu "xCaliber
  Gemma-4-26B NVFP4" 0.07, David Martin "BFS Pipeline v16" 0.17, plus a wave of RVQ/ColBERT/
  BEIR retrieval spam and generic "Fine-Tuning Qwen 3 8b / SDFT_training" experiments (no
  scores). Nothing beats our filler.
**Verdict: IGNORE the noise; the 1.39 remains the ADOPT/defensive item carried from 07-22.**

---

## Leaderboard (public, full CSV 2026-07-23, 1877 teams)

The field COMPRESSED upward — the wall eroded into a dense 1.44–1.60 band and pushed us down
one rank.

- #1 YUTO KOJIMA **1.86** (47 subs, active) — unchanged, still #1, still opaque.
- New/risen into the 1.5s: Tecnod8.AI 1.61, DhanaLakshmiMalla 1.60, ippeiogawa 1.58,
  Mathurin Ache / anngle / NoOneAhead 1.56, paul / Seok 1.54. Gold-ish cutoff drifting up.
- **boristown (暗黑AGI) rank 13, 1.47** (matches the public gold notebook).
- **Tufa Labs (driessmit1/cottaar et al.) rank 18, 1.45** — the model-snapshot source behind
  the 1.39 public notebook.
- **"Figuring out ARC AGI" (incl. zoli800/Chekhlov) rank 22, 1.44** — the old "wall," now the
  *bottom* of the compressed band.
- **US (Canivel) rank 45, 1.33** (88 entries, last sub today 00:07). Down 1 rank from #44 —
  pure field-compression erosion, our score unchanged. A "Tufa Labs fan" clone sits at 1.21
  (#115), so the duck-harness family keeps spawning copies below the leaders.

**Read:** our 1.33 frozen filler is sliding as the 1.44+ band thickens. The defensive ADOPT
from yesterday (diff our frozen fork vs the public 1.39 config for a low-risk +0.06) is now
*more* urgent — standing still = continued rank bleed out of/below silver.

---

## Watch-item status
- **1.15x-vs-1.0x efficiency cap (#684625): RESOLVED via #728299 code read.** 115 = per-level
  cap (1.15x); game/LB = completion-weighted with completion cap. Optimize for DEPTH, never
  overshoot (quadratic). Host still hasn't replied in-thread, but we no longer need them to.
- **arc_agi 0.9.9 version bump (#728220):** open, host-silent — keep the version assert in
  preflight.
- **RTX-6000-Pro access wobble (07-22, Brodehl):** no new host reply; no fresh reports today,
  so likely transient/per-account. Keep the GPU-flag assert; verify the accelerator dropdown
  before the next build (unchanged ops-ADAPT).

## Net verdict for the daily brief
Real signal today is the **exact scorer breakdown (#728299)** — depth-over-efficiency +
quadratic overshoot penalty + 1.15x per-level cap, verified to 1e-9, with a free offline
scoring atlas we should wire into the local harness. Second: independent architectural
validation of our EWM/verifier topology (#727505, "Constraint Before Control") — right idea,
weak results, no code, but a borrowable ternary required/forbidden/irrelevant memory tag.
Leaderboard compressed upward; we slid #44→#45 at unchanged 1.33, making yesterday's 1.39-diff
ADOPT more urgent. No new clones of our artifact; the "99%" schema-harness site is
closed/overfit-smell → IGNORE.
