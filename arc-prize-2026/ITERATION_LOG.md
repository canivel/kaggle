# ARC-AGI-3 Iteration Storyline (for cross-AI evaluation)

A chronological narrative of every iteration we've attempted on the
**arc-prize-2026-arc-agi-3** Kaggle competition. Read top-to-bottom to follow
the decision tree, regressions, and current open questions.

---

## 0. Project context

- **Competition**: Kaggle "ARC Prize 2026 — ARC-AGI-3" (interactive reasoning benchmark)
- **Deadline**: 2026-11-02
- **Scoring metric**: **RHAE** (Random Human-Action Efficiency) — per level it's `(human_actions / agent_actions)^2`, summed/averaged across 25 public games. Displayed score is a **percentage** (so "0.43" on the LB = 0.43%, "1.30" = 1.30%).
- **Game shape**: Agent controls a player via ACTION1-7 (4 directional + 1 interact + 1 click `(x,y)` + 1 undo) on 64×64 grids with 16 discrete colors. 25 public games × N levels each.
- **Kaggle environment**: Code-competition. Notebook reruns inside Kaggle: ALL 25 games run in parallel threads sharing CPU. ~8 hours per game wall-clock. **GPU OOMs with 25 parallel CNNs**, so production env is effectively CPU-only.
- **Repo location**: `f:/kaggle/arc-prize-2026/`
- **Notebook slug**: `canivel/arc3-final` (each "vNN" below = kernel version #)
- **Best score so far**: 0.43% (single lucky draw; mean ~0.29 on our champion code)
- **Current #1 on LB**: Tufa Labs at **1.30%** (3-4× above next tier)

---

## 1. Pre-eval-harness era (Apr-May 2026)

We submitted >15 daily variants of a BFS + CNN agent (the **FORGE / v35 / v39 family**), scoring all over the 0.17-0.33 range with no clear pattern. The key insight that took us months to internalize:

> **Public LB single-submission noise is ≥0.21 wide on IDENTICAL code.** v31=0.43 and v32=0.22 on the literal same kernel proved it.

Highlights from this era:

| Date | Kernel | Code description | Kaggle |
|------|--------|------------------|--------|
| 2026-05-06 | v22 | BFS + CLTI demos, `sel.set_data` fix for ACTION6 | **0.30** |
| 2026-05-12 | v27 | v22 + Blind-Squirrel back-label win-distance | 0.25 |
| 2026-05-13 | v28 | v22 + AXIOM-lite object-centric verb dict | 0.19 |
| 2026-05-15 | v30 | v22 + offline-pretrained ForgeNet (507 BFS demos) | 0.22 |
| 2026-05-16 | v15 | **v35** = per-level BFS→GraphExplorer hybrid (memory's "breakthrough") | local sweep showed 39 levels vs v22=36 |
| 2026-05-19 | v17/19 | v39 = same engine + correct env (`enable_gpu:true` + BYOD docker `kaggle-private-byod@sha256:00377cd1...`) | 0.24 |
| 2026-05-24 | v23 | v44 = v39 + TTT-step-1 (CNN persists across levels in a game) | 0.20 |
| 2026-05-26 | v27 | "v46" = v39 baseline revert (after v45 error) | **0.33** |
| 2026-05-27 | v28 | v46 variance check, SAME code | 0.24 |

The 0.33 → 0.24 swing on identical code (one-day apart) was the moment we admitted public-LB iteration is statistically useless.

---

## 2. Critical recurring bugs we hit and fixed

### 2a. ACTION6 `set_data` bug (memorialized)

`GameAction.ACTION6` defaults to `(x=0, y=0)` when sent via the gateway unless you call `sel.set_data({"x":x, "y":y})`. Multiple versions stored coords in `self._last_action_data` but never invoked `set_data`. **Every CNN-driven click went to (0,0)**. Confirmed via `arcengine` source. v22 was the first version to fix it cleanly.

### 2b. Poisoned pretrained ForgeNet weights

Discovered 2026-05-31: the `canivel/forge-pretrained-weights` dataset was trained on data collected BEFORE the set_data fix existed. The CNN learned click coordinates that "only work" when clipped to (0,0). Once we fixed the agent to actually click at the predicted (x,y), the pretrained CNN's coordinates were garbage → catastrophic regression (**v30 → 0.04**). Fix: **don't load pretrained ForgeNet** (random init, learn online). Result: v31=0.43 (lucky draw), v32=0.22 (same code variance).

### 2c. `scored.sort(reverse=True)` dict-comparison bug

`v39_agent.py` line 1538 had `scored.append((score, act_id, data))` followed by `scored.sort(reverse=True)`. When `score` ties, Python tries to compare the `data` dicts → `TypeError`. The outer `try/except` in `choose_action` catches it → random fallback action. Silent degradation on every novelty-sampling call. Fixed: `scored.sort(key=lambda t: (t[0], t[1]), reverse=True)`.

### 2d. PYTHONHASHSEED salt

Python's builtin `hash(str)` is PYTHONHASHSEED-salted per process. Earlier versions used `hash(game_id)` for the per-game seed — every Kaggle submission was a fresh random draw. v37 fixed it with `int(hashlib.md5(str(game_id).encode())[:8], 16)`. After this fix, local results became fully reproducible.

---

## 3. The JEPA-WM detour (2026-05-24 to 2026-05-26)

User asked for a "Tufa-class" world model. We built the full **JEPA (Joint-Embedding Predictive Architecture)** stack in a single session:

- `jepa_wm/models/jepa.py` — ViT-S encoder (12L d=256), EMA target encoder, action-conditioned predictor. **25.64M params, ~98MB ckpt.**
- `jepa_wm/inference/mcts.py` — Latent-space PUCT MCTS with batched dynamics.
- `jepa_wm/data/gen_trajectories.py` — BFS replay across 25 games + 400 random actions/game → **10,188 (s_t, a, s_t+1, r, done) tuples**, saved as `jepa_wm/data/trajectories.npz`.
- `jepa_wm/training/train.py` — EMA target encoder + L1 latent loss + reward/done heads + VICReg variance reg.
- `jepa_wm/inference/agent_hooks.py` — `JEPAHook.pick_action(frame, available_actions, click_candidates)` for v45 integration.

**Training**: 20k steps, batch 32 × grad_accum 4, AMP. Final r-loss 0.175, latent_var 1.63 (no collapse). ~2.5h on RTX 3080.

**Integration v45 = v39 + JEPAHook for BFS-failed levels.**

### Submitted as kernel v26 (2026-05-26 00:15 UTC). **ERRORED on rerun.**

Diagnosis (later, via local load-test of 25 parallel processes):

- **9.9GB RAM** just for 25 × 400MB JEPA worker copies (Kaggle's parallel env has ~13GB total)
- **41 seconds per MCTS pick** at the original budget (32 sims, depth 8) — Kaggle agent gateway has ~10s per-action timeout
- JEPA at 25M params just won't fit Kaggle's CPU+parallel rerun

Concluded: **need a ViT-XXS distillation (~3-4M params)** before retrying JEPA. Deferred indefinitely.

---

## 4. Building the local eval harness (the game-changer, 2026-05-28)

Public LB noise made A/B impossible. So we built `eval_harness.py`:

- For each of 25 games: instantiate the game class directly via `importlib`, bypass the HTTP gateway, run `agent.choose_action()` loop bounded by `--budget` actions and `--bfs-s` BFS-time cap.
- Records: `levels_completed`, `actions_per_level`, terminal state (WIN / GAME_OVER / BUDGET), wall time.
- Supports A/B compare mode (two agent .py files side-by-side).
- BFSSolver monkey-patch enforces `bfs_timeout` cap regardless of agent's internal adaptive logic.

**Determinism verified**: ran v39 baseline twice (replicate 1 vs replicate 2), 0 game-level differences. The hashlib seed fix makes everything reproducible.

### Baseline results (budget=200, bfs-s=15):

| Agent | total_levels | games_with_progress | Notes |
|---|---|---|---|
| **v39 baseline** (BFS + CNN) | **12** | 11/25 | dict-sort fix applied |
| v44 (CNN persist across levels) | 12 | 11/25 | IDENTICAL to v39 — CNN-persist invisible at 200-action budget |
| **v35 (per-level BFS→GE hybrid)** | **19** | 13/25 | **+7 vs v39.** Gains: bp35 +1, cd82 +2, r11l +1, tu93 +4. Loss: ar25 -1. |
| **v35 @ bfs-s=30** (more BFS time) | **25** | 13/25, 1 WIN | **+13 vs v39.** Also unlocked sk48 (0→1), tu93 (5→9). First terminal WIN. |
| v37 (v35 + per-level wall-clock deadline) | 20 | 11/25 | LESS than v35 |
| v47 (PURE SG port, no BFS) | **2** | 2/25 | Catastrophic. Needs 8h/game for online CNN to actually train. |
| v48 (v39 BFS + SG-CNN cold replacing GE) | **11** | 11/25 | WORSE than v39. SG-CNN cold-start can't replace GE at 200-action budget. |

Probed the 14 zero-solve games at budget=400, bfs-s=60 → **ALL stayed at 0**. These games (g50t, ka59, lf52, re86, sb26, sc25, su15, tn36, tr87, wa30, etc.) are structurally beyond BFS+GE. They have arm/unlock mechanics where actions produce zero visible frame-change → search has nothing to traverse.

**The 25-level local ceiling = our BFS+GE architecture's hard ceiling.** Breaking past it requires:
- Learned dynamics (JEPA / world models) — deferred (too heavy for Kaggle rerun)
- Object-centric (AXIOM) — tested in v28 standalone, got 0% on these games
- LLM-driven code synthesis (Rodionov-style) — out of scope for 1-2 day builds

---

## 5. The "1.30 score mystery" research arc (2026-06-03 to 2026-06-10)

User reported: "someone scored 1.30 — investigate".

### 5a. Deep-research dive

Spawned multi-agent workflow (98 sub-agents, 2.6M tokens, 50min) to find the technique:

- **Tufa Labs / StochasticGoose (Dries Smit)** = current #1 on main LB at 1.30. Advised by **Jack Cole (MindsAI)** — MindsAI is the team famous for TTT (test-time training) on ARC-AGI-1 and 2.
- Earlier Preview Competition: Tufa scored 12.58% RHAE (which displays as 12.58% on Preview LB, or some smaller number on main LB).
- Public Tufa code: **github.com/DriesSmit/ARC3-solution/custom_agents/action.py** (489 lines).
- Architecture: **16-channel one-hot 64×64 input → 4-layer CNN backbone (32→64→128→256) → action head (5 logits ACTION1-5) + FULLY-CONVOLUTIONAL 64×64 coord head (4096 logits ACTION6)** → unified 4101-logit output. BCE loss on `(state, action) → frame_changed`. 200K hash-dedup buffer. Reset model + buffer on level-up. NO BFS, NO MCTS, NO symbolic search. ~100 lines of PyTorch.
- **Key architectural insight**: BFS/GraphExplorer can't discriminate which ACTION6 click coords are even LEGAL on a 64×64 grid. SG's fully-conv coord head learns exactly that.
- Other notable verified techniques:
  - **Rodionov "Executable World Models"** (arXiv 2605.05138, 32.58% mean RHAE) — Codex CLI + GPT-5.4 synthesizes a refactorable Python world model. Best technique overall but needs LLM Code-CLI integration, not 1-2 day work.
  - **Rudakov graph-explorer** (arXiv 2512.24156, 3rd in Preview) — same architecture family as our v35. Confirms v35 lineage is sound.
  - **Duke LRM harness** — 97.1% on TR87 but 0% on BP35. Doesn't generalize. Out.

### 5b. Pure-SG port → kernel v34

- File: `notebooks/forge_agent/v47_agent.py` (~250 lines, faithful port of the SG repo)
- Local at budget=200, bfs-s=15: **2 levels** — confirmed SG architecture needs 8h game wall-clock for online CNN to train.
- **Shipped anyway** (Tufa's 1.30 validates the architecture, just needs the bigger budget).
- **Kaggle v34 = 0.18.** Within v39 noise floor but no breakthrough.
- Interpretation: my pure-SG matches the public Preview-era SG, but Tufa's main-comp 1.30 has more than the public repo.

### 5c. SG + pretrained CNN → kernel v36 (DISASTER)

- Hypothesis: SG's random-init CNN wastes most of the 8h on cold learning. Pretraining on our existing 10k BFS trajectories should warm it up.
- Built `train_sg_pretrain.py`: loads `jepa_wm/data/trajectories.npz`, computes `frame_changed` labels, trains SG `ActionModel` for 20k steps with BCE loss.
- **Final val acc 0.95.** Shipped weights as Kaggle dataset `canivel/sg-pretrained-weights` (131MB).
- Patched v47 → v49 to load weights at level-reset.
- **Kaggle v36 = 0.01.** CATASTROPHIC.

**Root cause**: The 10k trajectories are **78.8% positive labels** (`frame_changed=1`) because they come from BFS solutions = sequences of productive actions. CNN learned to predict `1` for almost everything → overconfident in wrong directions → worse than random init.

### 5d. Class-balanced SG retrain → kernel v37

- Added `pos_weight = (1-pos_frac)/pos_frac = 0.27` to the BCE loss to counter the bias.
- Re-trained 20k steps. Final acc 0.947 (vs 0.953 unbalanced — small drop, expected).
- Uploaded as v2 of `canivel/sg-pretrained-weights`.
- **Kaggle v37 = 0.22.** Recovered to v35-baseline range. Class balancing did fix the catastrophe. But still nowhere near Tufa's 1.30.

### 5e. BFS + balanced-pretrained-SG hybrid → kernel v38 (PENDING)

- File: `notebooks/forge_agent/v50_agent.py` = v48 (BFS + SG hybrid) + the v49 pretrained-weight loading code path.
- Logic: BFS first for any BFS-solvable level → SG-CNN with warm pretrained weights for BFS-failed levels.
- Just submitted 2026-06-10 13:56 UTC. Awaiting score.

---

## 6. Current state (2026-06-10)

### Recent Kaggle scores

| Kernel | Date (UTC) | Code | Score |
|--------|-----------|------|-------|
| v29 | 2026-05-31 | v35 (BFS + GE + CNN, set_data fix) | 0.31 |
| v30 | 2026-06-01 | v35 + load poisoned ForgeNet pretrained | 0.04 |
| v31 | 2026-06-02 | v35 + skip pretrained CNN (random init) | **0.43** (lucky) |
| v32 | 2026-06-04 | SAME as v31 | 0.22 |
| v33 | 2026-06-05 | v37 (per-level deadline) | 0.14 |
| v34 | 2026-06-06 | v47 pure-SG port (random init) | 0.18 |
| v35 | 2026-06-07 | v35 4th sample | 0.19 |
| v36 | 2026-06-08 | SG + unbalanced pretrain | **0.01** disaster |
| v37 | 2026-06-09 | SG + class-balanced pretrain | 0.22 |
| v38 | 2026-06-10 | BFS + class-balanced pretrained SG hybrid | **PENDING** |

### Kaggle LB top 10 (as of 2026-06-08)

| Rank | Team | Score | Date |
|---|---|---|---|
| 1 | **Tufa Labs** | **1.30** | 2026-06-08 |
| 2 | Redfield Rentals | 0.68 | 2026-04-17 |
| 3 | Barada Sahu | 0.66 | 2026-06-07 |
| 4 | Kevin E R MILLE | 0.66 | 2026-06-07 |
| 5 | SVG | 0.65 | 2026-04-16 |
| 6 | Matthew Philip Poetker | 0.64 | 2026-05-20 |
| ... | ... | ... | ... |
| **N/A** | **Us** | **0.43 (peak)** | **2026-06-02** |

Gap from us to mid-pack (0.50-0.66): ~0.1-0.2.
Gap from mid-pack to Tufa: ~0.7.

### v35 distribution (4 samples on identical "best" code)

`{0.43, 0.31, 0.22, 0.19}` — mean ≈ 0.29, std ≈ 0.10. So we're sampling from a distribution whose mean is around mid-pack-floor but our peak shots can hit mid-pack.

---

## 7. Architecture inventory (local-eval scoreboard at budget=200, bfs-s=15)

| Agent | total_levels | What it does |
|---|---|---|
| v47 pure-SG | 2 | 16-ch one-hot → 4-conv → 5+4096 logits, BCE on frame_changed, random init |
| v48 BFS + cold-SG | 11 | v39 BFS infra + SG-CNN (random init) as BFS-failed fallback |
| v39 baseline | 12 | BFS + CNN + CLTI demos + AEM memory + novelty-guided exploration |
| v44 CNN persist | 12 | v39 + CNN/buffer survive level changes within a game |
| v37 per-level deadline | 20 | v35 + hard wall-clock cap per level on BFS |
| **v35 (best, bfs-s=15)** | **19** | per-level **BFS → GraphExplorer** primary policy. Full GE port from arXiv 2512.24156. |
| **v35 (best, bfs-s=30)** | **25** | same, more BFS time |

---

## 8. Things we KNOW work for someone but we can't (yet) implement

- **Test-Time Training (TTT) on a pretrained transformer** — MindsAI's signature. Tufa's likely secret. Requires a pretrained foundation model that adapts during the 8h game window via gradient updates on rolling experience.
- **Executable World Models** (Rodionov) — Codex CLI synthesizes a Python game model at test time. 32.58% mean RHAE, 7/25 games fully solved. Out of our 1-2 day reach.
- **LLM-driven Python history-retrieval harnesses** (Duke) — 97.1% on TR87 but 0% on BP35. Wins specific games, doesn't generalize. Tactical option.

---

## 9. Open questions for the next AI to weigh in on

1. **The bias correction was the easy fix; what about the data itself?** Our `trajectories.npz` is BFS solutions + 400 random actions/game. The class-balanced pretrain gets us back to baseline but not above. Is the data fundamentally wrong for SG-style supervised learning, or do we need MORE data, or a different label signal?

2. **What's Tufa's actual extra ingredient?** Public SG (Preview-era) ≈ our v34 (0.18). Tufa main-comp 1.30 = 7× our number. The gap is too big for "they trained longer". What architectural addition gets 7× from public SG?

3. **Should we attempt the Rodionov LLM-code-synthesis path?** That's literally proven to work (32.58% mean RHAE) but requires Codex CLI inside a Kaggle code-comp notebook. Feasible?

4. **The JEPA dead-end revisit**: we shelved it because 25M params × 25 parallel agents = 9.9GB RAM, too tight. If we distill to a ViT-XXS (~3-4M params), it fits. Worth revisiting now that we've spent a week on SG?

5. **What does the v38 hybrid result (pending) tell us?** If v38 > 0.30, hybrid stacks; ~0.25 means they're redundant; <0.15 means SG-CNN hurts the BFS path.

6. **The 14 unreachable zero-solve games**: g50t, ka59, lf52, re86, sb26, sc25, su15, tn36, tr87, wa30 (and a few others). What single mechanism cracks them? Memory/state we're not tracking? Hidden game-rule discovery?

7. **Sample efficiency on Kaggle vs local**: our local 25 → Kaggle 0.31 mapping suggests 1 local level ≈ 0.012 Kaggle RHAE points. Tufa's 1.30 = ~108 "local levels equivalent". Where do those come from?

---

## 10. Where to look in the code

- Best baseline agent: `notebooks/forge_agent/v35_agent.py` (3596 lines)
- v39 baseline (cleaner, smaller): `notebooks/forge_agent/v39_agent.py` (1838 lines)
- SG port: `notebooks/forge_agent/v47_agent.py` (244 lines)
- SG + pretrain (v36/v37/v49): `notebooks/forge_agent/v49_agent.py`
- BFS + SG hybrid (v38/v50): `notebooks/forge_agent/v50_agent.py`
- Local eval harness: `eval_harness.py`
- SG pretrain script: `train_sg_pretrain.py`
- 25-game data: `kaggle-data/environment_files/<gid>/<guid>/<gid>.py`
- ARC-AGI-3 agent framework: `kaggle-data/ARC-AGI-3-Agents/`
- JEPA stack (dormant): `jepa_wm/`
- Trained JEPA ckpt: `jepa_wm/checkpoints/jepa_wm_final.pt` (98MB)
- Pretrained SG weights: `runs/sg_pretrain/sg_action_model_balanced.pt` (131MB)
- Notebook source: `notebooks/arc3-final/arc3-final.ipynb` + `kernel-metadata.json`
- Submission history: `kaggle competitions submissions arc-prize-2026-arc-agi-3`

---

*Generated 2026-06-10 by Claude Code as a handoff document.*

---

## Daily loop entries (winning_solution_FINAL.md campaign, Jul 2026–)

### 2026-07-09 — Phase 0 (day 3): σ-draw #1 = 0.89; phase1 eval seed 1 null-consistent
- **Score:** frozen duck fork σ-draw #1 = **0.89** (Jul 8 repro = 0.82). Both in Tufa band 0.77–1.30, above P0 kill line (<0.8). Phase-0a exit gate (≥0.9 in 2 scored attempts) missed by 0.01 → +1 reserved duck retry engaged; σ-draw #2 queued (`canivel/arc3-duck-repro` v3, trusted-fork).
- **Builds:** `arc3-duck-phase1` v1 COMPLETE — Phase-1 substrate patches confirmed applied; 25 games × 1 pass on Kaggle RTX PRO 6000, mean **1.22** (duck band). Evidence saved: `runs/duck_eval/phase1_seed1.json`.
- **Gate eval (non-decisive, 1 seed):** vs `runs/tufa_example_run/variance_summary.json` null — mean per-game Δ = −0.38, game-level sign-flip p = 0.47, run-level z = −0.83. No regression, no lift; need ≥3 seeds/arm.
- **Watch across seeds:** tr87 cleared level 1 (Tufa 0/20 ever — possible explore() win); lp85 = 0.27 **below** Tufa's 20-pass min (2.01) — possible substrate harm on a reliable game; sp80 = 0 (low but in-support).
- **Iterate:** pushed `arc3-duck-phase1` v2 = phase1 eval seed 2 (1 of 2 daily pushes; ~2.4 GPU-h/build, quota healthy). Next: seed 3 tomorrow → 3-seed evidence artifact → phase1 becomes queue-eligible experimental build.

### 2026-07-10 — Phase-0a EXIT GATE MET (0.93); Phase-1 gate look #1 = FAIL (backfilled 2026-07-11)
- **Score:** σ-draw #2 = **0.93** → Phase-0a exit gate (≥0.9) **PASSED** on the reserved retry. Draws {0.82, 0.89, 0.93}.
- **Builds/gate:** pushed `arc3-duck-phase1` v3 (Kaggle phase1 seed 3, mean 0.87) AND provisioned an H100 pod ($2.99/hr, ~11h) that ran the full pod-side 3-seed A/B (`runs/phase1_ab/`, seeds pulled 13:43/17:27/19:40 local).
- **GATE (look 1 of 2): FAIL.** `runs/phase1_ab/gate_report_final.md` — mean paired Δ **+0.169**, exact game-level sign-flip p = **0.308** vs α = 0.0125 (scorer validated to 0e+00 against Tufa's 500 runs). Secondary lc Δ +0.13, p = 0.041. Positive but variance-dominated.
- **Per-game map:** wins concentrate on games whose null never clears L1 (sc25 +3.1, cn04 +1.4, cd82 +0.8, sk48 +0.6; tn36 +5.2 = the animation-diff target); losses on games duck already progresses in (ar25 −1.6, sp80 −1.7, vc33 −1.5, tu93 −1.2, ft09 −0.8).
- **Queue:** σ-draw #4 submitted; σ-draw #5 left as filler pending v2 gate.

### 2026-07-11 — σ-draw #4 = 1.02 (band high); phase1-v2 (level-gated explore) seed 1 pushed
- **Score:** frozen duck fork σ-draw #4 = **1.02**. Draws {0.82, 0.89, 0.93, 1.02}: mean 0.915, σ̂ = 0.084, χ²-CI (df 3) [0.047, 0.311]. All in Tufa band 0.77–1.30; kill line never touched. **Watch: 4 draws strictly monotone increasing** (p ≈ 0.042 under exchangeability) — possible LB/game-version drift; check at draw #5.
- **Builds:** `arc3-duck-phase1` v3 COMPLETE = Kaggle-rail phase1 seed 3, mean **0.87**; vs Tufa null Δ = −0.73, sign-flip p = 0.966, z = −1.63. Evidence saved `runs/duck_eval/phase1_seed3.json`. Kaggle rail 3-seed mean 1.34 vs null 1.60 — agrees with pod gate: **v1 substrate = no lift, leaning harmful.**
- **Diagnosis:** explore()'s real actions deflate RHAE precisely on progressing games (higher levels carry higher RHAE weight); all wins came from null-lc≤0.25 games (stuck split +0.54 vs working −0.07 in pod data).
- **Iterate:** built substrate **v2 = `PHASE1_EXPLORE_MAX_LEVEL=1`** (explore fires only before the first level clear; animation/archive/hysteresis unchanged). Smoke 58/58 + gate unit test pass. Dataset `arc-phase1-kit` v2 + kernel v4 pushed = **v2 screen seed 1** (screens are 1-seed, not gate looks; retry look #2 spent only if screen is positive → 3 seeds).
- **Queue:** pending = σ-draw #5 (`arc3-duck-repro` v3) — correct filler while v2 screens; not empty. Pushes today: 1/2.

### 2026-07-12 — σ-draw #5 = 0.95; v4 "v2" screen = null AND arm was mis-deployed; true v2 shipped on kernel v5
- **Score:** frozen duck fork σ-draw #5 = **0.95**. Draws {0.82, 0.89, 0.93, 1.02, 0.95}: mean 0.922, σ̂ = **0.074**, χ²-CI (df 4) [0.044, 0.213]. Monotone-increase watch from draw #4 **cleared** (1.02 → 0.95 breaks it; no drift signal). All 5 draws in Tufa band 0.77–1.30.
- **Builds/screen:** `arc3-duck-phase1` v4 COMPLETE (mean 1.60 = exactly null-scale). Offline gate scorer (validated 0e+00 vs Tufa's 500 runs): mean paired Δ **+0.0090**, exact sign-flip p = **0.489** → screen NEGATIVE, retry look NOT spent (screens are 1-seed, no look consumed). Report: `runs/phase1_v2_screen/gate_report_screen_v2.md`.
- **Deployment bug found:** kernel v4 did NOT run the designed v2 arm. Its log banner shows v1 defaults (budget=8, max_explores=6, no min_level_actions/cooldown) and `duck_eval/phase1/_kaggle_dataset/` still held the v1 patch (0 hits for `min_level_actions`) — the full-v2 patch (design: `learnings/v2_gating_design_2026-07-11.md`) existed only locally. v4's arm was at best "v2-lite" (`MAX_LEVEL=1` env on v1 code); its null screen does not condemn the designed v2.
- **Fix + iterate:** synced all 5 kit files to staging, smoke 77/77 PASS, pushed `arc-phase1-kit` new version (files verified live 12:29Z), replaced notebook cell 12 with the self-verifying v2 hook (banner prints `phase1 v2` + budget=6 max_explores=3 min_level_actions=90 levelup_cooldown=20), pushed **kernel v5 = true v2 screen seed 1** (1/2 daily pushes). Tomorrow: banner check is decisive before scoring.
- **Queue:** pending = σ-draw #6 (`arc3-duck-repro` v3) — correct filler; not empty. No gate/kill decision today; retry look still in hand.

### 2026-07-12 (evening session, backfilled 2026-07-13) — position/headroom/failure analyses; path_forward v1→v3; panel R5–R7; sched-v1 built + submitted; phase1-v2 local 3-seed FAIL
- **Planning cycle:** wrote `position_analysis`, `headroom_analysis`, `failure_analysis` + `path_forward_v1/v2/v3_2026-07-13.md`; panel rounds 5–7 (`learnings/panel/round5-7/`). Round 6: 5× MAJOR-REVISION (scores 6–7, 0 fatal). Round 7 (on v3, 3 reviewers): 3× MAJOR-REVISION, 0 fatals, 6 majors — core: (D/NEW-1/N5) best-across +0.24 EV inconsistent with budget semantics; (E/NEW-2) Track B λ₀ miscalibrated; (N7/NEW-3) fact-check method unspecified. **v3 NOT approved.**
- **Phase-1 true-v2:** local 3-seed gate (seeds 201–203, `runs/phase1_v2/gate_report_FINAL.md`): mean paired Δ **−0.54**, p = **0.923** → **FAIL**. Kernel v5 banner verified `phase1 v2` (deployment bug of v4 fixed; loop closed).
- **Sched-v1 built + submitted:** attempt scheduler (restart at 90 if lc=0, cap 2, park; NO context injection; `duck_eval/scheduler/`). Smoke 41/41; kernel `canivel/arc3-duck-sched` v1; submitted 02:19Z = **draw #1 = 0.90** (control band; no lift). Queue holds draw #2 of 2 w/ pre-registered gate (promote ≥ baseline+0.12; kill < baseline).

### 2026-07-13 — sched-v1 0.90; scoring-semantics fact-check ANSWERED from code (pooled-single-run); phase1 line adjudication
- **Score:** sched-v1 draw #1 = **0.90** (z ≈ −0.3 vs control 0.922/σ̂ 0.074). Mechanism verified on build rail: 18 restarts/12 games, 4 parks at exactly 272 actions, **4 restart-recovered L1 clears** (bp35, ls20, tu93, ft09) — but offline Δ −0.32 vs null: recovered L1s RHAE-crushed by pooled action accounting; only tu93 (+1.36, clean L2 after recovery) netted positive.
- **Fact-check (panel R7 D/NEW-1/N7) resolved by code forensics:** competition mode = **one run per game_id** (`arc_agi/api.py:417`); `max`-across-runs (`scorecard.py:192`) unreachable; within-run actions **pool** with the wasted-attempt tax landing on the *first* cleared level only (`scorecard.py:430,655`). True semantics = pooled-single-run with first-clear tax — invalidates BOTH columns of v3's EV table; surviving path is P(recover) × clean-L2+ value. Full detail: `learnings/daily_brief_2026-07-13.md` §1c.
- **Phase-1 v2 screen (kernel v5, 1 seed):** offline +0.19 mean but **8W/11L** — sign-negative, outlier-carried; author position: line stays closed per 3-seed FAIL.

### 2026-07-14 — WAR BUILD v1 = 0.91 (warpack draw #1, null-consistent); new LB top 1.86
- **Score:** war v1 (duck + warpack: banking/recovery/retry_guard/shortcircuit/soft_end 11h20m/fast-submit; ledger aboard flags OFF) = **0.91**. vs control mean 0.922/σ̂ 0.074 → z ≈ −0.16: no lift, no harm in draw #1. Warpack's EV is order-stats over draws + banking upside — neither observable in one draw; scored rerun is hidden (fast-submit gate verified live: dummy parquet at 0.6s, RUN_HEAVY=False), so no per-game artifacts (settles R9 ME-NEW-12 identity question: pulled outputs ≠ scored run).
- **LB:** new #1 **YUTO KOJIMA 1.86** (overnight, zero public footprint) > Tecnod8.AI 1.61; 1.44 resubmission wall intact. Our best 1.02.
- **Brief:** `learnings/daily_brief_2026-07-14.md`. Open Qs: tonight's arm (ledger A/B window 1 vs another war draw), warpack control accounting, which R9 majors block (variance reconciliation = blocking + free).
- **Panel R10** (daily brief; methodology/llm-agents/rl-planning): 3× MAJOR-REVISION, 0 fatal — unanimous: (1) do NOT flip ledger tonight (Q1 answer = option b); variance reconciliation is blocking and must precede R2 window 1; (2) war-v1 needs its own control ledger (≥3 draws before contrast, ≥5 before standardized effects, t not z); (3) order-stats arithmetic refutes "wall mechanism adopted" — publish E[max]-vs-k; (4) P1–P5 must be inline with observables; (5) banking needs an integrity canary. Reviews: `learnings/panel/round10/`.
- **Variance reconciliation DONE** (`runs/variance_reconcile/report.md` + `raw.json`, scorer 0e+00-validated): build-rail 1-seed run-mean sd **0.572** (paired-Δ 0.780) — worse than the 0.52 bootstrap; **ft09 = 54.3% of run-mean variance** (ft09+vc33 = 71.5%); LB σ 0.074 is a different population (7.5× ratio explained — both numbers right for their own instrument). **RHAE-mean retired as build-rail gate statistic (3-seed power 0.02 vs +0.10). New primary = paired Δ levels_completed (sd 0.086, power 0.73 vs +0.20 lc); secondary = Δ log1p(RHAE) (power 0.45).** E[max over k LB draws]: 1.07@30 / 1.11@110 at σ=0.074 (1.46@110 only at χ²-CI-hi σ=0.213) → order stats are a floor-raiser (~+0.15 total), never a wall-breaker at the point estimate.
- **Pre-registrations filed** (`learnings/preregistration_2026-07-14.md`): H4 amendment (gate statistic), warpack control ledger (war-v1 draw #1 = 0.91; n≥3/n≥5 rules), R2 A/B launch conditions (banking held identical in both arms; alternate-nightly; stopping rule CI half-width <0.10 or 6 windows/arm), P1–P5 verbatim + observables, ME-NEW-12 disposition (scored rerun = hidden separate execution; cross-rail tripwire instead of infeasible identity check), sched-v1 KILLED by its own pre-registered gate (0.90 < 0.922; draw #2 removed from queue).
- **Builds:** pushed `canivel/arc3-duck-war-eval` v1 (1/2 pushes; one-line diff = `WARPACK_FORCE_OFFLINE_BENCH=1` via `duck_eval/warpack/build_eval_notebook.py`; smoke 48/48) — first-ever heavy warpack execution on Kaggle hardware; delivers banking replay canary counts + war-v1 build-rail lc baseline. Status at EOD: RUNNING (~2.4 GPU-h expected).
- **Queue (verified end-to-end):** head = war-v1 σ-draw #2 (`arc3-duck-war` v1, byte-identical; preflight dry-run ALLOW with self-referential upstream — NOTE: Cottaar upstream blocks on the intended 7-cell warpack diff; queue entry fixed accordingly), filler = frozen fork v3. Daemon fires 18:37, window opens 20:00 EDT.
- **Handoff → 2026-07-15:** Tonight's LB draw = war-v1 #2 (append to warpack ledger, NOT frozen control). First actions tomorrow: (1) pull `arc3-duck-war-eval` output (`uvx --from kaggle==2.0.0 kaggle kernels output canivel/arc3-duck-war-eval -p runs/kernel_pulls/war_eval_v1`) → grep "warpack banking: replayed" for the canary + record build-rail lc baseline in the prereg §7 monitor; (2) log war draw #2, recompute war-v1 ledger stats; (3) if eval shows banking replay divergence or lc regression vs null10, convene panel on warpack composition BEFORE draw #3; else queue war-v1 draw #3 (completes n≥3 → R2 window 1 eligible 07-16 with ledger flags ON per prereg §4, provided P1–P5 thresholds stand under the new gate statistic). Pushes remaining today: 1. GPU reserve untouched. Retry look (phase-1) unspent; phase-1 stays CLOSED.

### 2026-07-15 — war-eval screen: Δlc +0.272 (p=0.0074) / RHAE flat; draw #2 fired midday (daemon quota-day hole found)
- **Score:** none overnight — daemon 22:37Z Jul 14 correctly skipped (UTC-day quota consumed by draw #1 at 00:13Z Jul 14); one LB window went unused. war-v1 ledger still n=1 {0.91}.
- **war-eval v1 (build rail, first heavy warpack run on Kaggle HW):** budgets byte-comparable to null10 (146 vs 140 actions/run). Screen (`runs/war_eval_v1/screen_report.md`, validated scorer 0e+00): **PRIMARY paired Δlc +0.272/game, 12W/5L, exact sign-flip p=0.0074** (lc 22 vs 15.2) — first positive primary-statistic screen of the campaign; **secondary Δlog1p(RHAE) −0.036, p=0.61 — flat** (RHAE 1.579 vs null 1.636). Warpack clears ~45% more levels at full action cost (pooled-single-run tax). Banking canary: ZERO replay events (vacuous with 1 pass/0 wins, not divergent — mechanism still never observed live).
- (entry continues below after panel R11 + submission)
- **Panel R11** (daily brief; methodology 6/10, llm-agents 5/10, rl-planning 4/10 — 3× MAJOR-REVISION, 0 fatal; `learnings/panel/round11/`): unanimous Goodhart alarm on Δlc-only gating + vacuous banking green-light. Directives executed same-day: (1) **prereg amendment filed** (`learnings/preregistration_amendment_2026-07-15.md`) — compound gate rule (Δlc p<0.0125 AND Δlog1p(RHAE)≥0), pass/fail licenses, A/B MDE published (0.17 @ 3v3 — LB A/B unpowered, decision endpoint = P1–P5 mechanism observables), handoff green-light rewritten (banking clause struck; war-v2 scored windows BLOCKED until engineered replay fires); (2) LOO sensitivity appended to screen report (survives: worst-case p 0.029; magnitude-free sign test p 0.14); (3) daemon fixed zero-code — second Task trigger 20:07 EDT added (18:37 kept as safety net), window-day logic untouched per rl-planning.
- **Banking-fire validation PASS** (`runs/war_eval_v1/bank_fire_validation.json`, amendment A2): replayed war-eval's own recorded actions on local engines — all 4 games reproduce eval lc (engines deterministic). Banking FIRED verbatim + score-invariant on ar25 (47 actions, 2 plays) & s5i5 (13 actions); **sc25/m0r0 abort `frame_divergence` step 0** (per-play randomization → strict-frame guard refuses, by design); **starved case reproduces the eval zero exactly** (`bank_skip time 30.0`). Conclusion: banking fires ONLY with ≥120s soft time left AND non-randomized frames → plausibly ~inert in scored LB runs (games exhaust budgets). Arms stay identical per design lock; banking fix = war-v3 material, NOT the A/B.
- **Actions:** war draw #2 SUBMITTED 12:33Z manually via daemon (UTC quota was free; score ETA ~20:30Z). war-eval **seed 2 pushed** (v2, RUNNING, push 1/2). Queue head = war draw #3 (fires 00:07Z Jul 16 via new trigger; ledger → n=3). GPU reserve untouched.

**Handoff → 2026-07-16:** Overnight brings THREE artifacts: war draw #2 score (~20:30Z), war draw #3 score (~08:00Z fire+8h), war-eval seed 2 output. First actions: (1) pull seed 2 (`uvx --from kaggle==2.0.0 kaggle kernels output canivel/arc3-duck-war-eval -p runs/kernel_pulls/war_eval_v2`), run `scripts/war_eval_screen.py` variant on it; (2) log draws #2/#3, compute war-v1 ledger n=3 stats (σ̂ recompute per amendment A3 — if σ̂>0.15 LB windows downgrade to monitoring-only); (3) push seed 3 (push #1) AND build+smoke war-v2 (ledger flags ON + canary counting attempts/skips/aborts) — push #2 ONLY if smoke is green with slack; **priority if conflict: seed 3 > war-v2** (gate look Jul 17 feeds everything; war-v2 window can slip). (4) Brief MUST carry: panel-objections disposition section, order-stats curve + per-mechanism reach table (dodged twice — rl-planning escalates), P1–P5 verbatim, LB screenshot artifact. Compound gate look Jul 17 (seeds 1–3): PASS = Δlc p<0.0125 AND mean Δlog1p(RHAE) ≥ 0. Phase-1 CLOSED; retry look unspent; pushes used today 1/2.

### 2026-07-16 — seed-3 screen NEGATIVE (war-v1 3-seed evidence now {+0.272, −0.008, −0.088}); panel R12 3× MAJOR-REVISION; GPT-5.6-sol probe launched; headless loop hit turn cap
- **Scores:** overnight draws #2 = 1.08 (campaign single-draw max) and #3 = 0.88 → war-v1 ledger n=3 {0.91, 1.08, 0.88}, mean 0.957, σ̂ 0.108. Accumulation-only per prereg §3, no gate consumed.
- **Morning loop (08:23, hit 80-turn cap 08:46):** wrote `learnings/daily_brief_2026-07-16.md` (merging deep-review 1a–1c: draw3_analysis, discussions, research sweeps), ran **panel R12** (rl-planning/llm-agents/methodology: 3× MAJOR-REVISION, 0 fatal, 9 majors), pushed war-eval **seed 3** (push 1/2), fixed queue head (war draw #4, accumulation), seed audit (#726552) **PASS** — no unseeded policy RNG → no ledger reset (A7). Loop died before ITERATION_LOG/development; this entry + remainder executed by the interactive session.
- **Seed 3 pulled + screened** (`runs/war_eval_v3/screen_report.md`, validated scorer 0e+00): **Δlc −0.088 (6W/12L, p=0.796)**, Δlog1p(RHAE) −0.202 (p=0.87). Seed-only-diff CERTIFIED (17/17 cells sha-identical to the deterministic local build; only the offline-bench force line prepended; no seed constant exists on this rail — replicates are sampling draws). Three-seed picture: seed 1's +0.272 did NOT recur (−0.008, −0.088); RHAE secondary negative in all three. **Pooled compound gate look stays SEALED for Jul 17 per R12 (option (c) refused).**
- **Panel R12 responses filed** (`learnings/preregistration_amendment_2026-07-16.md`): **A4** R2 A/B power published (n/arm 14–111 across σ candidates → LB inference infeasible; R2 gated on build-rail only; LB = accumulation + non-inferiority harm check at n=5, margin 0.15); **A5** variance gate restated on χ²-CI-hi < 0.25 at df ≥ 4 (old A3 at df=2 passed σ=0.20 with p≈0.43 — could not fail informatively); **A6** R3–R5 grinder cracking UNCONDITIONAL, scoping began today, build work NLT Jul 20; **A7** seed-audit PASS → ledgers valid.
- **N5 (banking determinism audit, all 25 games) + N6 (war-v2 ledger-ON eval build+smoke)** delegated to agents, running at time of writing; results appended below when landed.
- **GPT-5.6-sol probe (user-provided OpenAI key, R3 scoping instrument):** duck harness runs LOCALLY with gpt-5.6-sol (the model that publicly solved ft09 at 87%) as analyzer vs the bundled engines — ft09 + sb26/su15/lp85 + vc33 control, same 32k context as scored runs, hard spend cap ($3 smoke → $40 full) enforced by a sticky cross-process guard in the local rig (`duck_eval/gpt56_probe/`, local-only patches to `openai_compat.py`/`tool_agent.py`). Purpose: capability-vs-harness decomposition per grinder → distillation targets for R3/war-v3. Kaggle-legal: local development only; only game-agnostic harness changes we author ship.
- **Fix:** `scripts/war_eval_screen.py` stdout now utf-8-wrapped (cp1252 UnicodeEncodeError on Δ; files were unaffected).
- **Queue:** head = war draw #4 (accumulation toward n=5 per R12 M2), filler = frozen fork. Pushes used 1/2 (seed 3); push #2 reserved for N6 war-v2-eval pending review.
- **N5 RESULT (determinism audit, all 25 games): divergent fraction 0/25 — every game frame-deterministic across plays.** R12's premise falsified. The sc25/m0r0 `frame_divergence` aborts were a **`prune_trace` bug** (drops leading `board_changed=False` actions that mutate hidden state → pruned replay desyncs at step 0; `runs/war_eval_v1/prune_replay_diag.json`). Full unpruned histories replay frame-identically on all 25; recorded lc reproduced on second plays. All 8 Δlc-positive games are bankable. Banking's binding constraints = prune bug + R11 soft-time skip — both engineering fixes → **war-v3 backlog: replay unpruned (or trailing-only prune) + scored-budget-compatible soft-time threshold**. Caveat: 15/25 local engine versions differ from Kaggle build (behavioral parity suggested by lc reproduction). Artifacts: `runs/war_eval_v1/determinism_audit_25.{json,md}`.
- **N6 RESULT (ledger-ON efficacy screen): `canivel/arc3-duck-war-v2-eval` v1 PUSHED (push 2/2)** after fixing a real bug the build agent found: ledger store was keyed by the SHARED artifacts dir → one cross-contaminated ledger for all 25 concurrent games (stage-1/2 tests missed it via per-game tmpdirs). Fix = ledger v2 per-game keying (`ledger_<stem>.json`, dict-ops-only lock, runtime banner prints VERSION+keying). Smokes: replay 9/9, noninterference 11/11, war_v2_eval_smoke 39/39 incl. two-games-concurrent zero-contamination + canary `stores=2`. arc-war-kit dataset v2 pushed + verified live (ledger_patch.py 13443B); **warpack byte-pinned at live v1** so the A/B contrast is ledger-flags-only (local warpack v2 drift NOT shipped). Ledger-OFF pair = war-eval seed 1. Post-build checks: grep log for `ledger v2: store keying = per-game:runtime-state-stem` (stale-dataset tripwire) + `LEDGER CANARY TOTAL ... stores≈25`.
- **GPT-5.6 probe blocker found+fixed:** gpt-5.6-sol rejects function tools on /v1/chat/completions while reasoning is active → built local translation proxy (`duck_eval/gpt56_probe/proxy56.py`: chat→/v1/responses, stateless store:false, tool-call+usage mapping preserved so the client-side spend guard keeps working). Also: private re-arc-3 repo unreachable → local `re_arc` shim (kaggle-data/re_arc) over the bundled 25-game engine snapshot; probe rig = repo .venv, no cloud.
- **GPT-5.6 probe SMOKE (ft09, 10-action cap): L1 cleared in 4 actions vs baseline 17** (better-than-baseline RHAE efficiency), mid-L2 at cap; 4 API calls, $0.05. Pipeline validated end-to-end (proxy multimodal-parts fix required: chat `text`/`image_url` → responses `input_text`/`input_image`). FULL probe launched: ft09+sb26+su15+lp85+vc33, 100 actions/60min per game, $40 hard cap. Early smoke signal: the harness is NOT the ft09 bottleneck at L1 — a frontier analyzer converts immediately, consistent with the capability-gap hypothesis (distillation targets, not harness rewrites, close ft09).
- **GPT-5.6 FULL PROBE COMPLETE ($14.67 of $40 cap, 100-action/60-min caps):** ft09 5/6 levels (score 71.4; every level ≤~baseline actions), sb26 5/8 (40.2), lp85 4/8 (27.8), vc33 3/7 (16.0), su15 1/9 cancelled at runtime cap (1.6; 103k output tokens burned on L2 — hard even for frontier). Same harness/context/tools as Qwen (which gets ~1 level on the grinders) → **capability-vs-harness DECIDED: harness supports deep runs; grinders = model gap; su15 = genuinely hard**. Artifacts: runs/gpt56_probe/experiment_full/ (transcripts, prompts, events, per-level action counts). Distillation (transcript divergence → game-agnostic war-v3 scaffolding) launched via Fable agents → learnings/war_room/gpt56_distill_*.md.
- **Distill (grinders sb26/lp85, learnings/war_room/gpt56_distill_grinders.md):** decisive negative on prompt-based transfer — system prompt already demands BFS; GPT wrote it 6x in lp85, Qwen 0x across six transcripts (prompt exhortation inert; consistent with feedback_prompt_is_noise). Qwen's gap = concept CONVERSION not perception (saw the sb26 connector turn 11, held it 2h, never derived semantics; stated lp85 shared-tile insight and ignored it). Ranked transferable deltas: (1) scripted auto probe-diff summarizer (~120-token fixed block per action result), (2) budget sentinel (shrinking-bar detector → ledger FACT; every Qwen grinder death = unseen budget death), (3) submission-fingerprint refutation (hash layout at submit; blocks verbatim re-submits, feeds N=3 escalation). NOT-distillable: recursive abstraction, representation invention, model-fitting, in-head enumeration. BFS helper library = highest-ceiling/lowest-certainty, A/B-only.
- **Distill (ft09/su15, learnings/war_room/gpt56_distill_ft09_su15.md):** ft09's separating behavior = offline hypothesis falsification against the board's own self-labeling legends + predict-next-frame before repeating an action family (Qwen env-tests ~30 hypotheses at 1-8 actions each, 2 GAME_OVERs, wins L1 by luck at action 135). Deltas: PREDICT→RESULT ledger wiring + harness-injected board_changed=False no-effect FACTs (reliable); legend-extraction assist (third). su15 VERDICT CORRECTED: GPT played only 4 min (L2 model half-right, 32-step budget dead at action 54); remaining 56 min = game_over-compliance deadlock (correct restart computed every turn, never executed) + upstream quota death 13:47 (251× 429, pre-top-up). su15 mechanics (vacuum radius, irreversible 3-way merge loss, exact-count goal) are near-unobservable within level budget → **information-theoretic wall; accept-the-loss for Qwen-tier**; free hygiene: game-over-continuation prompt fix + measure-reach-before-paying line.

**Handoff → 2026-07-17:** Tonight 20:07: war draw #4 fires (accumulation; audit gate satisfied). Overnight: arc3-duck-war-v2-eval (ledger-ON, per-game keying) completes — pull it, grep `ledger v2: store keying = per-game:runtime-state-stem` + `LEDGER CANARY TOTAL ... stores≈25` (stale-dataset tripwire), screen vs war-eval seed 1 (ledger-OFF pair) with scripts/war_eval_screen.py. THEN the **sealed 3-seed compound gate look** (seeds {+0.272, −0.008, −0.088}; PASS = pooled Δlc p<0.0125 AND mean Δlog1p(RHAE) ≥ 0 — expect FAIL on both prongs; on FAIL, per A4/A6 the war-v1 composition closes and war-v3 takes the build queue). Panel R13 on the day's brief must rank the war-v3 backlog (all game-agnostic): banking unpruned-replay + scored-budget soft-time (N5), probe-diff summarizer, budget sentinel, submission-fingerprint refutation (grinder distill), PREDICT→RESULT ledger wiring + no-effect FACTs, game-over-continuation fix (ft09/su15 distill). One flag per window; simplicity-wins prior applies; BFS helper library A/B-only. GPT-5.6 probe assets: runs/gpt56_probe/ ($14.67 spent, rig reusable — user topped up +$10; su15 re-probe optional at frontier tier). Pushes used 2/2 today; GPU reserve untouched.
- **LATE ADD (20:30 EDT): war draw #4 SUBMITTED 00:07Z (PENDING, ~04:00 EDT score). war-v2-eval (ledger-ON seed 1) COMPLETE + pulled + screened same night:** tripwires PASS (`ledger v2: store keying = per-game:runtime-state-stem`, canary `games=25 stores=25 attempts=1552 digests=1552 skips=0 aborts=0`) — but **efficacy screen NEGATIVE: Δlc −0.128 (p=0.86), Δlog1p(RHAE) −0.314, RHAE 0.893 vs null 1.636**, below all three ledger-OFF seeds. With 1552 digests and **0 escalations**, the ledger is a constant context tax whose key trigger never fires — phase-1's always-on-injection lesson repeated. R13 input: ledger-as-built does NOT enter scored windows; its distill upgrades (budget-sentinel FACTs, submission-fingerprint refutation, PREDICT→RESULT) are the fixes that give it firing triggers. Artifacts: runs/kernel_pulls/war_v2_eval_s1/, runs/war_v2_eval_s1/screen_report.md.

### 2026-07-17 — GATE LOOK: FAIL both prongs → warpack build-rail line CLOSED; draw #4 = 1.05 (war ledger n=4, mean 0.980)
- **Sealed 3-seed compound gate look executed** (rule A1 verbatim, no discretion; `runs/war_gate_look_2026-07-17.json`): prong (i) pooled Δlc +0.059, 10W/10L, sign-flip p = 0.225 (need <0.0125) FAIL; prong (ii) mean Δlog1p(RHAE) across seeds = −0.132 (all three negative) FAIL. **FAIL ON BOTH → warpack build-rail line closes; LB ledger continues to n=5; R2 A/B decision escalates to full 5-reviewer panel (= today's R13).** LOO p range [0.14, 0.42] — no single game rescues. Only ka59 positive across all 3 seeds. Seed 1's +0.272/p=0.0074 confirmed as a 1-seed draw.
- **Score:** war draw #4 = **1.05**. Ledger n=4 {0.91, 1.08, 0.88, 1.05}: mean 0.980, σ̂ 0.0997, χ²-CI σ (df 3) [0.056, 0.372]. A3: σ̂ < 0.15 → LB windows live. Δ vs frozen control +0.058 (Welch t 0.97, unpowered as designed). A5 variance gate evaluable at n=5 (tonight). Build-rail-vs-LB tension noted in brief §1a (offline-bench regime ≠ 8h LB regime; build rail governs build decisions per A1).
- **Sweeps:** discussions — 1 ADAPT (#716295 new comments: swarm.py per-game budget allocation + per-agent LLM routing overridable → per-game budget re-allocation is a legal war-v3 lever); research — 1 ADOPT (arXiv:2607.08716 proactive memory sidecar: selective-injection-beats-always-on ablation externally confirms our war-v2-eval 0-escalations tax finding), 2 ADAPT (MemCon 2607.13591 bandit-gated retrieval; 2607.09493 persistence taxonomy).
- **Brief:** `learnings/daily_brief_2026-07-17.md`. Panel R13 launched at FULL 5-reviewer strength (A1 escalation): ranks war-v3 backlog (a)–(g), owns R2 A/B disposition, confirms tonight's window (author lean: war draw #5 → n=5).
- **RECOVERY NOTE (2026-07-18 07:35, interactive session):** Jul 17 loop hit the 80-turn cap with panel R13 mid-flight; the five KAOS reviewer agents zombied (parents killed, no `result` rows) → R13 never reviewed, queue head never set → **frozen-fork filler fired 00:07Z and scored 1.33 = NEW CAMPAIGN LB BEST** (right-tail draw of the vanilla band 0.77–1.30; order-stats floor-raiser working exactly as modeled — E[max] climbs with k; note this is the FROZEN fork, not warpack). R13 relaunched interactively 07:35 on the same 07-17 brief (full 5-reviewer per A1 escalation). Loop-robustness fix needed: panel collection must survive parent death (persist agent IDs to disk at spawn; collect on next session) — added to backlog.

### 2026-07-18 (early, interactive session; loop fires 08:23)
- **Filler draw = 1.33 = NEW LB BEST** (frozen fork right-tail; band now observed 0.77-1.33). Fired because Jul 17 loop died at turn cap with R13 zombied (recovery note above).
- **Panel R13 RERUN complete** (`learnings/panel/round13/`): **0 accepts, 5x MAJOR-REVISION, 2 FATAL** — both fatals = grinder-cracking spec missing (the only wall-closer has no design/gate/expected-Δ; "the panel is rubber-stamping a strategy that converges to ~1.11"). Majors: budget-regime mechanisms untestable on current rail (→A10); (a)+(f) bundling (→A12); su15 wall confounded by (f) bug (→A13 re-probe); ledger REFUTED overstated (→A11 relabel); A5 near-unpassable + reopening rule needed pre-observation (→A8/A9); (g) targeting = noise-level signals, needs FDR rule; Q1 needs counting upper bounds.
- **`preregistration_amendment_2026-07-18.md` FILED ~08:00 EDT, before draw #5 observation**: A8 A5-fail consequence sealed (accumulation-only, no readout arm; future thresholds relative to control CI-hi); A9 warpack = UNTESTED-IN-REGIME parked, reopens ONLY on n≥8 Welch p<0.05 AND budget-faithful trigger-firing bench, neither alone; A10 compressed-budget bench requirement; A11 ledger relabel; A12 (f)-first unbundling + su15 exclusion; A13 su15 wall suspended pending post-fix re-probe.
- **Queue**: head = war draw #5 (final war accumulation → n=5), filler = frozen fork (re-added; was consumed last night). Grinder-cracking design doc (FATAL discharge) in flight via Fable agent → today's loop + R14.
- **Grinder-cracking design doc FILED (`learnings/war_room/grinder_cracking_design.md`, R13 FATAL discharged 2 days early)** — by CORRECTING the premise: real-scorer counting bounds show the v3 conversion stack = **ceiling +0.31 rail / +0.17 LB, expectation +0.04–0.10 LB** — a floor/mid raiser, NOT a wall-closer (reclaimed actions on uncompleted levels score zero; sb26/lp85 expected Δclears = 0 at Qwen tier; (f)≈0.00 — Qwen never deadlocks, that was GPT). **The wall-closer label transfers to the gated war-v4 MODEL line: Qwen-72B-tier AWQ on the free Kaggle RTX PRO 6000 96GB rail** (costed incl. 2.5-3x throughput penalty risk), scoping Aug 1. Timeline: (f) hygiene Jul 18-19 → (d)+(c) Jul 20 (A6 met) → (a) → (b) → banking (A9 double-lock, full-panel sign-off). Gate = A10 compressed-budget bench, canary-verified triggers, 3 seeds, compound rule + mechanism prong, non-inferiority −0.10. R14 reviews this doc.

### 2026-07-18 (daily loop, 08:23) — filler 1.33 refutes control sigma 0.074 (recompute: 0.179, E[max@110]~1.44 = the wall); (f) hygiene day; R14 running
- **Score deep-dive (1a):** 1.33 filler draw pooled with frozen control {0.82,0.89,0.93,1.02,0.95} -> n=6 mean 0.990, sigma-hat **0.179**, chi2 CI [0.112, 0.440]. Old sigma 0.074 refuted as point estimate (1.33 was z~+5.5 under it). **Order-stats conclusion flips:** E[max of k draws] ~1.36@k=30, **~1.44@k=110** at revised sigma — nightly frozen-fork resubmission alone now has an expected max AT the 1.44 resubmission wall over the remaining ~107 windows. Window-pricing question (Q-A/Q-B) handed to R14. Tonight stays committed: war draw #5 (final accumulation -> n=5 -> sealed A5/A8 look). Preflight dry-run on head: ALLOW (T1-T4 OK).
- **Sweeps:** discussions 2 new (host "500 submissions" post ADAPT -> preflight GPU-flag assert + thread deadlines; host infra constants ADOPT -> 10MB log cap, scratch-not-working, 30GB cgroup); research: **OPINE-World arXiv:2607.01531 ADAPT top-priority (published 20/25 games / 160/183 levels on ARC-AGI-3**, no per-game training; replay-check contract = mechanical firing trigger for (d)+(c); plausibly the 1.86 leader's family), GSME 2607.13683 ADOPT (activation gate before significance gate = first-class prong 0). Files: war_room/{discussions,research}_2026-07-18.md.
- **Brief:** learnings/daily_brief_2026-07-18.md (Q-A..Q-E for R14). **Panel R14 launched** (full 5, prior-dir round13, proposal = grinder design doc + brief; agent IDs persisted at spawn — zombie-proof).
- **(f) build:** continuation_patch delegated (phase1/ledger pattern; patches prompts+tool_agent PYTHON_ADDENDUM bindings; kill switch; 6-test smoke; wired into gpt56 probe rig for A13). Result appended below when landed.
- **R14 COLLECTED (recovery: all 5 reviewers had completed in KAOS; morning loop died before collecting 2)**: **5x MAJOR-REVISION, 0 accepts** (`learnings/panel/round14/`). Load-bearing finding, derived independently by methodology (FATAL), llm-agents, rl-planning: **the design doc's per-window gate was arithmetically unpassable** — exact sign-flip at alpha=0.0125 needs >=7 uncontradicted nonzero wins (2^-7=0.0078) vs the doc's own expectation of 1–4 → the sealed FAIL rule would deterministically park every true-but-small component ("a pre-registered machine for discarding true positives"). Second unanimous defect: panel reviewed a TRUNCATED circulation (Part 1 of 2, argv budget in panel_round.py) — a gate cannot seal on an unseen document. Also 3x MAJOR: war-v4 (sole wall-closer) evidence-free at 72B tier, needs a cheap capability screen pre-Aug-1; compressed-budget bench measures a trigger-rich regime and needs a transfer rule; banking retry list = winner's curse on the same 3 seeds.
- **Amendment `preregistration_amendment_2026-07-18b.md` (A14–A20) FILED ~13:30 EDT, before draw #5 observation:** A14 gate seal VOID + recalibrated (pooling unit = 24 game-level pairs; binding score decision = ONE cumulative stack-vs-W0 look; per-window looks demoted to mechanism prong + non-inferiority at the sealed look only; P(pass|expectations)~0.2–0.4 published; cumulative-FAIL consequence sealed = honest label, no dismantling); A15 compressed-bench pass = provisional, full-budget confirmation required; A16 banking frozen retry list RETIRED → online policy (retry iff current < banked record) + shrinkage recompute; A17 war-v4 72B go/no-go screen (>=2 levels beyond 27B on ft09/sb26/lp85/vc33 AND throughput-adjusted null formula defined) pre-Aug-1 blocking; A18 (d) offline kill-test threshold sealed; A19 (c) prong += post-block novel-family rate; A20 (g) DEAD, 0.56x = assumption w/ 0.4–0.8x band, R15 circulation in <=2 sha-stamped parts.
- **(d) KILLED by sealed A18 threshold (`runs/predict_metric/`):** recurrence accuracy of the no-effect-FACT rule = 0.465 [0.436,0.494] vs majority baseline 0.903, on 175 game-runs / 29,487 actions (board_changed label verified by independent frame hashing, 0 disagreements). A recurring "no-effect" pair changes the board ~54% of the time — the rule is actively wrong when it fires. Trigger coverage passed (68/175); accuracy is binding. **W1 becomes (a) budget sentinel's window; (c) disposition → R15.** Cost: $0, CPU-only, before any build spend — the panel's cheap-kill discipline paying out exactly as designed.
- **(f) W0 SHIPPED (push 1/2):** continuation smoke 12/12 → `canivel/arc3-duck-w0-continuation-eval` v1 pushed, RUNNING. Built via new `--w0` mode in the proven war-eval builder (15/17 cells byte-identical to war-eval; diff = eval-force+W0 stamp cell + warpack graft replaced by continuation-only graft, no warpack/ledger imports — verified by new 20/20 w0_eval_smoke against the real harness). arc-war-kit dataset version pushed FIRST + verified live (continuation_patch.py 6073B ready). Banner greps for tomorrow: `continuation v1: game-over-continuation ACTIVE (2 modules patched)`, `w0-continuation-eval: SEED=1`, negative: no `warpack`/`LEDGER` lines. Idle-turn observable: count post-`game_over` non-progress lines in `{game}_p0_events.jsonl`.
- **Queue verified:** head = war draw #5 (preflight re-run: ALLOW, T1–T4 OK), filler = frozen fork. Daemon 18:37/20:07. Pushes 1/2. GPU reserve untouched. $0 cloud spend.

**Handoff → 2026-07-19:** Overnight artifacts: (1) war draw #5 score (~04:00 EDT) → ledger n=5 → run the sealed A5/A8 look (variance gate on chi2-CI-hi < 0.25 at df>=4; A8 fail-consequence already sealed: accumulation-only, no readout arm); (2) W0 eval completes → pull (`uvx --from kaggle==2.0.0 kaggle kernels output canivel/arc3-duck-w0-continuation-eval -p runs/kernel_pulls/w0_eval_s1`), grep the three banners + negative check, then W0 quick screen: idle-turn observable (expect 0 post-game-over idle turns) + descriptive non-inferiority vs war-eval seeds (NO score gate — (f) counting bound is 0.00). First actions after that: (3) **R15 circulation** — full grinder design doc + amendments 18/18b in <=2 parts, per-part sha256, untruncated END lines; the recalibrated A14 gate seals on this circulation, so R15 must run before the first flagged-window look; (4) **build (a) budget sentinel** (W1 owner now, per A18 kill of (d)) + its A10 compressed-budget canary — A6 build deadline Jul 20 transfers to (a); (5) scope A17 72B screen (weights dataset + vLLM bench kernel on the free rail) — blocking for Aug 1. Push budget: seeds for (a) will need both pushes tomorrow if the canary is same-day; W0 seed-2 decision belongs to R15 (W0 is unflagged hygiene — 1 seed may suffice for a 0-idle-turn mechanism check). Retry look unspent; phase-1 CLOSED; no cloud spend.
- **OPINE-World deep-read (`learnings/war_room/opine_world_deepread.md`):** (1) **su15 SOLVED 9/9 by OPINE (334 acts)** — our A13 "information-theoretic wall" verdict is likely WRONG (re-probe now expected-retraction; systematic program-synthesis + replay-verify cracks it). (2) OPINE fails ka59/sk48/lf52/bp35/s5i5 on planner branching/budget — and **our exec_wm sims are SATURATED on two of their failures (lf52 100%, s5i5 99.5%)**: search depth is the shared wall, and we already own accurate world models there. (3) Architecture = 2 Opus-4.8 agents + Python game_engine admitted only by exact double-run replay; per-step predict-vs-settled verify, first mismatch aborts plan → counterexample; NOT kernel-runnable as published (closed API, no code release, zero ablations) but the contracts are pure Python. (4) **Highest-value adaptation: plan-execute-verify contract on our 12 saturated exec_wm sims** — harness-side BFS, one live action/step, hash-compare vs settled frame, fail-closed. Zero LLM tokens, ~2-3 build days, counting-bound ceiling ≈ +0.5/draw rail (EXCEEDS the entire v3 stack ceiling), expectation +0.10-0.30. Needs panel sign-off + own gated window. Settled-state comparison also kills the N5 prune_trace bug class by construction and gives (d)/(c) a mechanical firing trigger (fixes the 0/1552 activation failure).
- **Winners deep-read (`learnings/war_room/winners_deepread_2026-07-18.md`):** (1) fork drift CLEAN — public duck notebook cell-identical to ours minus intentional grafts; Jul-16 action retired. (2) **Reki's dead-signature suppression (his 0.64→0.86) = our (c) one abstraction up**: structural component signature (color,size,is_rect,twins), 2 inert observations suppress class per level, HARD VETO of LLM clicks — adopt as the click component inside the (c)+(d) flag, not a new window. (3) 3rd place = Reki's family with everything OFF (simplicity-wins externally confirmed again); adopt build-rail-only deterministic RNG for gate power, keep LB stochastic for order-stats. (4) **Both runner-ups independently converged on mechanical (not prose) no-effect refutation** — exactly what our 0/1552 ledger lacked; (c)+(d) as first flag now triply validated. (5) **1.44+ band unexplained by ANY public artifact** (winner tier tops at 0.86-1.21; wall-breakers ~14 teams share nothing; Tufa attributes their own edge to "multimodality + better base models") → war-v4 model-swap corroborated as wall-closer; **no OPINE-style executable world models anywhere public = uncontested edge if we build it**. (6) Hygiene adds: first-action RESET pre-boot, sticky vLLM-failure fallback, request-timeout=min(remaining), commit-mode smoke; fork-trap warning: Reki's public metadata now has enable_gpu:false.
- **LB process model (`runs/lb_process_model/`):** 1.33 is generatively explained by our OWN bench + measured common-night correlation (shared server/sampling luck across 110 slots) — NO hidden deep-play regime (R13's regime-transfer worry dissolves for the vanilla harness); a ≥1.30 night is 44% ft09-L2. σ̂ 0.074 was a lucky-tight n=5 sample of σ≈0.13-0.17. Honest E[max@107] ≈ 1.39 central; P(touch 1.44)=0.29; P(1.86)≈0.01 → **filler = lottery not plan; window break-even = credible ≥+0.06-0.12 lift; existing +0.12 gates price correctly**. Deterministic build-rail RNG adopted for gate power (3rd-place trick), LB stays stochastic.
- **STATE OF THE WAR synthesis FILED (`learnings/war_room/state_of_the_war_2026-07-18.md`)** — 8 KNOWs / 3 BELIEVEs / 3 UNKNOWNs + ratification-ready strategy stack: (1) EWM-execute line (OPINE contract on our 12 saturated sims, ceiling +0.5/draw, uncontested edge, 2-3 days), (2) (c)+(d)+Reki-signature flag, (3) war-v4 model scoping Aug 1, (4) su15 re-probe (expected retraction), (5) filler for all unpriced windows. R14/R15 to ratify priority 1 vs the design doc's ordering.
- **Kimi-3 review cycle adopt-set IMPLEMENTED (corrected form):** (#2) `scripts/kaos_ingest.py` built + run — 76 campaign artifacts (war-room deep-reads, amendments, panel summaries R5-R14) ingested into kaos.db (66→142 rows; memory had been frozen since 05-25); dream cycle validated end-to-end (digest at Dreams/2026-07-18-190924.md; skills_scored=0 confirms the sealed "summarizer not gardener" expectation); weekly Sunday step added to daily protocol. (#4) mixed-tier routing bench row added to state_of_the_war with the front-loaded cost (double model load + split KV). (#3 two-tier fingerprints + #1 event-shaped EWM canary) building via agents with historical-backfill / Stage-0-dry-run validations — results appended when they land.
- **Adopt #1 DONE + Stage-0 dry-run (runs/ewm_dryrun/, 41 streams, 11747 events, 17/17 selftests):** canary PASS all sources, 0 deadlocks, 0 selfdiff, fail-closed cost ≈1 action/aborted plan. **LOAD-BEARING FINDING the totals-shaped counters could not see: held-out sim saturation does NOT transfer on-trajectory** — aborts land overwhelmingly at step 0 (timer/hidden-counter phase misalignment + engine drift); gpt56 ft09 collapses to 0.07 step-acc (GPT reaches L2+ states the sims never saw). Reliable Qwen-regime carriers = tn36/tr87/tu93/ls20/ft09-L1; **vc33 + s5i5 (2 of 5 Stage-1 targets) abort at step 0 on most plans → material discount on the +0.10-0.30 Stage-1 expectation. Flagged for R15.** Schema sealed in duck_eval/ewm_exec/EVENT_SCHEMA.md (log volume ≤5.0MB vs 10MB cap); aggregator scripts/ewm_events.py; design-doc addendum appended.
- **Adopt #3 DONE + historical backfill (runs/failure_fingerprints_backfill.md, store runs/failure_fingerprints.json, 15/15 tests):** 16 incidents → 13 fingerprints → 8 recurring families. **Counterfactual: 5 scored LB windows burned that would have carried a pre-submit WARN** (v36, v38 via arc3-final slug family at death #2; v63/v64/v65 via scratch-built-drift family at death #2 — root cause was found manually only after death #5). preflight.py recurrence WARN live (new optional `recurrence` key, contract untouched — verified vs daily_submit parsing + dry-run on tonight's actual queue head: ALLOW exit 0). fingerprint_report.py --brief wired into weekly protocol. Transient 403 on submissions API during agent run — verified recovered.

### 2026-07-19 (early, interactive; loop fires 08:23)
- **Draw #5 = 0.76 (campaign-low; observed range now 0.76-1.33). SEALED A5/A8 LOOK EXECUTED** (`runs/a5_a8_look_2026-07-19.json`, deterministic arithmetic, consequences sealed pre-observation 07-18): war ledger n=5 {0.91,1.08,0.88,1.05,0.76} mean 0.936, σ̂ 0.1309, χ²-CI-hi **0.376 ≥ 0.25 → FAIL** (as R13 predicted: the draw needed [0.955,1.005]). Sealed consequence (A8): war arm = accumulation-only permanently, ineligible as A/B readout arm at any n; no mechanism may cite war-arm LB deltas as evidence either way. Per A9 war accumulation ENDS (no draw #6); frozen fork resumes as filler (already queue head).
- **LB model validation bonus:** pooled n=11 across both arms → mean 0.9655, σ̂ **0.154** — dead-center in the model's predicted 0.13-0.17 bracket (the 0.074 era is formally over). Window pricing from state_of_the_war stands: filler E[max@~106] ≈ 1.39, experiments must credibly claim ≥ +0.06-0.12.
- Today (loop): R15 on grinder design doc + EWM Stage-0 discount re-pricing; Sunday weekly steps (kaos dream + fingerprint brief table); (d)+(c) build due Jul 20 per A6.

### 2026-07-19 (daily loop, Sunday) — W0 screen PASS (49 GO episodes, 0 idle turns); panel argv bug root-caused + fixed; R15 running untruncated
- **W0 quick screen (runs/kernel_pulls/w0_eval_s1/screen_report.md):** all 3 banners present; negative check clean (only "NO warpack" banner lines match). **Mechanism PASS: 49 GAME_OVER episodes across 12 games, 0 idle post-game-over actions — the continuation graft recovers on the immediately-following action every single time.** Non-inferiority (descriptive): 16 total levels inside ledger-OFF seed band {13,15,22}, mean 1.73 vs {1.16,1.58,1.62}. (f) confirmed pure hygiene at zero cost; author rec to R15: default layer in all future builds, no seed-2.
- **Panel truncation defect ROOT-CAUSED (R14's unanimous complaint):** KAOS embeds the full task INTO the agent system prompt (kaos/ccr/prompts.py build_system_prompt), and the Agent SDK passes a plain-string system prompt as a literal `--system-prompt` argv — Windows CreateProcess ~32K cap. panel_round.py's old 30K argv budget was accidentally masking this; my first untruncated R15 launch (57K circulation) failed all 5 reviewers with a bogus "Claude Code not found" (WinError 206 → FileNotFoundError → SDK mislabels as CLINotFoundError). **Fixes:** (1) kaos CLI `run` now accepts `@file` task delivery; (2) panel_round.py writes reviewer prompts to files (truncation code deleted; A20 satisfied by construction — single part, sha-stamped, END-line tripwire); (3) AgentSDKProvider now delivers system prompts via `--system-prompt-file` (temp file, cleaned up). Smoke-tested end-to-end (PONG). R15 relaunched with the full 56.7K circulation (grinder doc + A8-A13 + A14-A20 + state_of_the_war + today's brief, per-part sha256).
- **Sweeps:** discussions — 1 ADAPT (reset-logic fragility from host thread: a 5-resets/level cap turned a 9-min agent into a 1-hour 0.00 run; binding caution for war-v4 screen + any harness change), 2 IGNORE; research — 2 ADAPT on the EWM step-0 abort problem (OCM arXiv:2607.02846 pre-execution procedure-vs-model verification; arXiv:2606.31399 world-state-fidelity-fails-first → re-observe/resync before abort), 1 park (AgentLTL), 1 IGNORE (agentic TTT). 72B AWQ: no external throughput anchor exists; we are the reference (A17 bench must self-measure).
- **Queue:** head = frozen fork (war accumulation ENDED per A9 after draw #5 = 0.76, sealed A5/A8 FAIL executed pre-loop). Preflight re-verified interactively: **ALLOW, T1-T4 OK, 0 warns, recurrence clean.** Daemon 18:37/20:07 armed. Weekly fingerprint table: no NEW incidents this week; 8 known families stable.
- Builds in flight (background agents): (a) budget sentinel + A10 compressed-budget canary (A6 deadline Jul 20); A17 72B screen scoping doc. Results appended below when landed.
- **Panel R15 COLLECTED (first fully-untruncated circulation, 5/5 END-line confirmed): 0 ACCEPT, 5x MAJOR-REVISION, 0 new fatals.** THE convergent directive (5/5 reviewers): **A14 does NOT seal this round** — A18's (d)-kill was never propagated into §2/§4/P(pass)/Part D; the gate seals on a REPUBLISHED circulation (R16). Other binding items: α re-derivation (0.0125 has lost its Bonferroni rationale with one binding look; 0.05 one-sided ~doubles P(pass)); cumulative dismantle branch (pooled Δlc ≤ −0.10 → (f)-only); guard false-kill calibration; **state-aliasing = ONE root cause across predict-metric 0.465 / EWM step-0 aborts / N5 prune bug → latent-state audit = blocking prereq for EWM Stage-1 AND banking**; (c)+Reki = unregistered 3-way bundle, needs own flag+bound or dies; EWM re-price is the AUTHORS' job (reliable carriers only, fidelity^L depth bound, Stage-1 gate must exist, resync-before-abort = contract change with own bound). A18's kill process called "exemplary". Synthesis: learnings/panel/round15/_directives.md.
- **A17 scope FILED (learnings/war_room/a17_72b_screen_scope.md), R15 gate repairs incorporated.** LOAD-BEARING: **the duck harness is MULTIMODAL** (grid rendered as 4x image; 27B baseline is a VL model) → **the 72B swap must be Qwen2.5-VL-72B-Instruct-AWQ** (43GB, Kaggle Model qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1); text-only 72B would silently drop the visual channel and confound the screen. SKU verified from kernel logs: RTX PRO 6000 Blackwell x1 sm_120, both rails identical. Gate boolean: CAPABILITY AND (ACTION-PARITY OR beats Σ null_adj + 1-level margin); comparator = per-game MAX over 3 certified seeds; Σ null_adj = 4 (ρ≤2.5) / 3 (ρ≤3.0). ~7.5 GPU-h full screen. Top risk: VL serve-config (hermes parser, NO qwen3 thinking flags) — tool-call round-trip must be runtime-tested pre-push.
- **(a) budget sentinel BUILT + validated locally (A6 Jul-20 deadline met): smoke 29/29; A10 canary PASS 3/3 seeds (23-25 games fire, ≥5 required); R15-O5 deterministic predicate PASS (49 budget deaths, 0 unwarned; negative path validated).** Zero-token-tax design (FACT only on crossing turns). NOT pushed — held for tomorrow: (1) the eval regime is UNCAPPED (max_actions_per_game=None confirmed in W0 log) so cell 2 must export SENTINEL_BUDGET=<value> or the sentinel is silently inert; the live budget VALUE is an unsealed design decision → R16 agenda item; (2) handoff plan already scheduled (a) seeds for the Jul-20 pushes. Build report: learnings/war_room/sentinel_build_2026-07-19.md (exact banners for post-run verification).
- **Weekly (Sunday):** kaos_ingest +2 rows (144 total); dream digest Dreams/2026-07-19-124559.md (recency-only, skills_scored=0 as sealed expectation predicts — R16 agenda); fingerprint table in brief (no NEW incidents; 8 known families stable).
- **KAOS/panel infra hardened:** kaos run @file tasks; AgentSDKProvider --system-prompt-file (argv 32K root cause); panel_round.py file-based prompts, truncation code deleted. Bogus "Claude Code not found" = WinError 206 command-line-too-long, documented in memory.

**Handoff → 2026-07-20:** Tonight 00:07Z: frozen-fork filler fires (preflight ALLOW verified, T1-T4 OK; war accumulation ENDED per A9 — no war draws ever again). Tomorrow's critical path, in order: (1) **R16 republication FIRST** — republish §2 sum/§4 table/P(pass) with (d) removed + banking branches, α re-derivation, dismantle branch, guard calibration, (c) disposition, SENTINEL_BUDGET value, W0-control-arm seed count for the cumulative look, dream digest review; circulate untruncated (infrastructure now proven); **A14 seals on R16's approval**. (2) Push arc-war-kit dataset version (budget_sentinel_patch.py staged, byte-identical copy verified) THEN sentinel eval seed 1 with the R16-ruled SENTINEL_BUDGET exported in cell 2 — verify banner + `SENTINEL v=1` events in build log (inert-sentinel check is the #1 pre-seal risk); seed 2 = push 2/2 if quota allows. (3) Register the latent-state audit (offline, N5 traces, $0) — blocking for EWM Stage-1 and banking; delegable to an agent same-day. (4) A17: draft the amendment text with the repaired boolean + comparator for panel sign-off; weights-dataset attach test can wait for the bench push later in the week. Push budget today: 0/2 used (all work was local + panel). GPU reserve untouched; $0 cloud spend. Queue: head = frozen fork, never empty. STOP.

### 2026-07-20 — filler 0.92; R16 circulation day; daemon audit-gate incident (window recovered manually)
- **Score:** overnight filler (00:07Z fire) = **0.92** (band-typical). Fork observed band 0.76-1.33.
- **Daily loop:** fired 08:23 on schedule; R16 circulated (round16/ + r16_circulation.md); as of 21:35 EDT the session is still live (13 procs) and R16 verdicts are NOT yet collected; no sentinel pushes observed in kaggle status; this entry written by the interactive session to restore the audit trail.
- **INCIDENT + RECOVERY: tonight's 20:07 EDT daemon fire was BLOCKED by the audit-trail gate** ("no ITERATION_LOG entry ### 2026-07-20" — the loop writes its entry at END of day and hadn't finished). Gate behaved as specified; spec has a single-point-of-failure hole. **Window recovered manually 21:32 EDT (user-directed): ref 54866117, frozen-fork filler v3 (the exact queued head), PENDING.** Queue bookkeeping updated (entry → history, fresh filler re-queued). Cadence intact: no day has been missed.
- **FIX (permanent):** daily_iterate_prompt.md STEP 1 now requires writing a minimal `### <date>` ITERATION_LOG stub IMMEDIATELY at session start (audit gate depends on its existence, not its completeness); full entry still appended at end.

### 2026-07-21 — (in progress; stub written at session start)
- (22:10 EDT, interactive) **Day-status: loop WEDGED** — session live since 08:23 (13 claude procs) but no file writes in ≥45 min, no round17, no sentinel pushes, no full log entry (stub only). Produced today: daily_brief_2026-07-21.md. **Tonight's window SAFE**: audit stub passed the gate (fix validated live), 20:07 fire succeeded (filler, PENDING), queue refilled to 1 after it drained to 0. Push budget 0/2 used today. Tomorrow's loop: (i) diagnose/kill the wedged session's leftovers FIRST (suspect: blocking wait on zombied KAOS reviewer agents — same class as the 07-17 R13 zombie incident; wait_all/communicate has no liveness escape), (ii) resume R16→R17 seal-repair docket + sentinel pushes (SENTINEL_BUDGET still unruled), (iii) consider hard wall-clock cap on loop sessions (e.g. schtasks /ET or a watchdog) so a wedge can't eat a full day again.

### 2026-07-22 — filler 1.14; R16 synthesized (A14 → R17); sentinel v2 pushed+ran; holdout collapse 10/11 (backfilled 07-23)
- **Score:** overnight filler (00:07Z) = **1.14**, band-typical upper half. Frozen control n=9 mean 0.992 σ̂ 0.155. zoli800 fork-diff: byte-identical artifact drew 1.39 publicly — confirms artifact tail ≥1.39, and fork-wave rank-erosion risk.
- **R16 synthesized** (wedge post-mortem: verdicts collected 07-21, session died pre-synthesis): 0 ACCEPT / 5 MAJOR-REVISION → **A14 does NOT seal; seals at R17** on a 9-item all-$0 checklist. Q2 SENTINEL_BUDGET=150 approved-with-conditions; Q6 A17′ REJECTED as drafted; Q5 banking restricted to full-replay-only v1.
- **Sentinel:** Q2 conditions 1–3 discharged (attempt-unit approximation FAILS on carriers → v2 re-key to cumulative game-envelope implemented; canary v3 PASS). **Pushed + ran on Kaggle** (arc3-duck-sentinel-eval, COMPLETE 14:59Z, 2h12m, 25 games, mean 0.85) — deep-dive belongs to 07-23.
- **Resolver holdout (R17 item 2): 10/11 in-sample ALIASED-RESOLVABLE games DROP to UNRESOLVED** (only sb26 keeps a held-out certificate, hist1). EWM carrier set collapses; banking stays FULL-REPLAY-ONLY everywhere non-CLEAN.
- **Schema traces mined** (runs/schema_traces_mining/report.md): claims verified locally on our exact engine versions; certification-as-resync (never step-0 abort) is the liftable contract for EWM v1.1; their 99% RHAE costs ~7 LLM-h/game — a 150-action cap structurally forbids their revise-loop (note for future sentinel-budget litigation).
- **Also:** a17_repair artifacts computed (false-NOGO bootstrap, per-seed table); r17_thresholds.json sealed; grinder_design_R17_sealing.md written. Session died before R17 panel launch, log entry, and queue refill (queue drained to 0 after the 20:07 fire — refilled 07-23 morning; no window missed).

### 2026-07-23 — (in progress; stub written at session start)

### 2026-07-23 (interactive; STUCK-REVIEW cycle) — verification + R17 + revised reset → R18
- **User-ordered full review (Jul 21): "we are stuck."** Independent raw-artifact verification (runs/verify_2026-07-21/report.md): ALL gate arithmetic exact; 4 discrepancies (fork band = 0.82-1.33; pricing STALE → E[max@102]≈1.35, P(1.44)≈0.18, down 40%; banking bound conservative-direction error; prose undercounted fatals). Throughput quantified: 9 straight windows zero new code; R10-R17 0/34 accepts, 169 majors; ≥7 mechanisms validated, 2 ever live (both killed); infra incidents 8/11 days.
- **R17 on stuck_review (5 reviewers): 3 FATAL — including the campaign-changing one: window pricing is denominated in MEAN currency but LB is MAX-scored** → opportunity cost of an experimental window ≈ 0.001-0.002 E[max]-equiv (2 orders below the +0.06-0.12 rule). Other fatals: quota-not-free (GPU-h table required), A17 envelope check required. Draws while cycle ran: 1.14 (2nd best), 0.82.
- **stuck_review_v2 (learnings/stuck_review_v2_2026-07-23.md) files A21-A25:** A21 12-window exploration budget (canary+screen entry, harm-pause, promotion still sealed); A22 two-track governance (build-rail = intent-only); A23 A17 start w/ envelope one-pager (43GB/96GB fits, SKU identical, >3.5x penalty = self-certifying envelope NO-GO); A24 heartbeat watchdog + bench exemptions; A25 seal-termination rule. **R18 running with R17 priors.** On 0-fatal R18: amendments file, sentinel + A17 push same day, first exploration draw this week.

### 2026-07-23 (execution) — amendment day-1: dataset verified live, C7 amended 140→150 (no push spent), C3 filed, (f) defaulted
- **Step 1 (arc-war-kit):** live dataset verified **byte-identical** to staging on all 6 files (sha256; budget_sentinel_patch.py v2 17296 B was already pushed 07-21 12:41Z; warpack_patch.py stays VERSION=v1). **No new dataset version pushed** — an identical push would be version churn; step's end state pre-existed.
- **Step 2 (sentinel eval) — CONFLICT + RESOLUTION:** slug arc3-duck-sentinel-eval NOT fresh — morning loop had independently pushed SEED=2/SENTINEL_BUDGET=150 (~09:00 local, RUNNING; seed 1/150 COMPLETE 07-22 14:59Z mean 0.85), violating C7-as-written (140) and consuming push 1/2. Escalated to coordinator; **ruling: C7 AMENDED to 150** (live two-seed 150-ledger authoritative; amendment file updated 09:57 EDT). The fully-staged, smoke-tested seed-1/140 build was **NOT pushed** (push discipline preserved; push 2/2 reserved for A17 bench tomorrow); archived as notebooks/ducksentinel-eval/arc3-duck-sentinel-eval.b140-archived-2026-07-23.ipynb + C7_AMENDMENT_NOTE.md sidecar. Staging rebuilt at 150 (seed-1 lines); sentinel_smoke gained **S2c/I1c budget-export checks** (C7 inert-sentinel risk) — **32/32 PASS** both warkit paths. preflight.py = N/A for this taaf-harness family (targets agents/-swarm arc3-baseline; smoke S1-S6 is the drift guard, per build-report design decision 6). Banner verification of BOTH seeds' logs (grep `SENTINEL v=2` — v1 in older docs is stale — + `SENTINEL_BUDGET=150` echo + `unit=game-envelope`) pending seed-2 COMPLETE (~11:15 local; monitor armed).
- **Step 3 (C3 filed, before bench push): learnings/a17_error_model.md** + runs/a17_error_model/{a17_error_model.py,.json} (MC 200k/cell, seed 20260723, exact-enumeration cross-check; inputs = verified runs/a17_repair/per_seed_table.json). Headline: **P(false GO | lift ≤ 0) = 0.000 everywhere** (structural: null throttled Σ ≤ 4 « capability bar 8); **P(false NO-GO | lift +1/+2/+3) = 1.000 at k=1** (structural: max achievable Σ = 7 < 8) — detection frontier P(GO)≥0.75 needs true lift ≥ +5 (k=1) / +4 (k=2). Filed interpretation: the screen is a ≥+4-to-+5-level capability-existence detector by construction; NO-GO at modest lift is the designed outcome, GO is near-unimpeachable. Also flagged (rule unchanged): marginal-rule dead zone at Σ=6 when ρ ∈ (2.6, 3.5].
- **Step 4 ((f) defaulted):** build_notebook.py + build_eval_notebook.py now append the game-over-continuation graft to cell 12 **by default in all modes** (--w0 excepted: it IS the graft; idempotence guard for v2 chain); kill switch CONTINUATION_DISABLE=1 stays, build opt-out `--no-continuation` reproduces pre-Jul-23 compositions (e.g. live sentinel seeds). Rebuilt duck_warpack{,_v2}.ipynb + duckwar-eval + duckwar-v2-eval + duckw0-eval; ducksentinel-eval left at live-matching sentinel-only composition (sidecar documents; seed-3 composition ruling belongs to loop/panel). Smokes: **war_v2_eval 40/40** (staged continuation module + new I2g check; default + --warkit), **w0_eval 20/20**, **sentinel 32/32**, warpack smoke_test 47/48 (1 FAIL pre-existing/unrelated: repo warpack_patch.py VERSION=v2 vs smoke's v1 expectation; the SHIPPED dataset copy is v1 as required).
- Push budget: 1/2 used today (loop's seed-2; this session pushed nothing). No submissions. $0 cloud.
- **Gate-chain (mandate extension, exploration draw #1 pulled forward to TONIGHT per A21/C4):**
  - **Seed-2 verified (COMPLETE 11:21:47 local):** SEED=2 banner, SENTINEL_BUDGET=150 exported+echoed, `sentinel v2 ... unit=game-envelope` ACTIVE, graft from arc-war-kit mount, **52 `SENTINEL v=2` events** + 20 sidecars (not inert), no PATCH FAILED. Seed-1 log re-verified same greps (56 events). Both-seed inert-check = PASS.
  - **Screen vs null10 (runs/sentinel_eval_v{1,2}/screen_report.md):** primary paired Δlc s1 **−0.128** (4W/12L, p=0.95) / s2 **+0.032** (7W/10L, p=0.40); pooled ≈ −0.05/game vs per-game sd ≈ 0.46, direction flips across seeds → **NON-HARM PASS** (screen criterion: mechanism fires AND Δlc not materially negative). **Caveat on record (C2: no stronger claim either way):** secondary Δlog1p(RHAE) negative on BOTH seeds (s1 −0.315 p=0.997; s2 −0.166 p=0.90) — flag for the sealed 3-seed look, not gating the screen.
  - **Live kernel built + pushed:** build_notebook.py gained `--sentinel` live mode (vanilla duck + (f) continuation + sentinel v2 @150 ONLY; NO warpack/ledger; cell-2 live budget stamp — scored regime uncapped, export mandatory). New smoke duck_eval/sentinel/sentinel_live_smoke.py **23/23 PASS** (structural + heavy-path grafts + cold-gate fast-submit dummy parquet + double kill-switch subprocess). **Pushed 13:52:21 EDT → canivel/arc3-duck-sentinel v1 (push 2/2 today; sanctioned — A17 bench moves to tomorrow slot 1, C3 already filed).** Build COMPLETE on fast path (dummy parquet 0.6s, both live banners in log, no tracebacks).
  - **Preflight trusted-fork (war precedent, self-upstream): ALLOW** (T1–T4 OK, 0 warns). **Queue head swapped 14:0x EDT** to arc3-duck-sentinel v1 with the mandated A21/C2 citation; filler stays entry 2; daemon (20:07) untouched. Composition recorded: **(f) rides as sealed hygiene-default** (49/49 screen + amendment order #4), **sentinel is THE experimental flag** — one experimental flag under the one-flag discipline.

### 2026-07-24 (early, interactive)
- **Exploration draw 1/12 (sentinel arm) = 0.71 — campaign-low draw. HARM-PAUSE TRIGGERED (A21/C2: draw < 0.80 pauses the arm; sealed pre-observation).** Sentinel arm PAUSED pending analysis. Three independent signals now align: eval-seed RHAE negative x2 (s1 −0.315, s2 −0.166) + scored draw 0.71. Per C2 no formal claim from n=1, but the pause is mandatory and the prior for this composition is now poor. Cost of the experiment: ~0.001-0.002 E[max]-equiv (max-currency pricing) — one window, versus warpack's five. LB best UNCHANGED at 1.33.
- Tonight's head: frozen-fork filler (already queued). Today's slot-1 push: A17 72B-VL bench (decision rule recalibration required BEFORE interpretation per a17_error_model findings — false-NO-GO=1.0 at k=1 for lift ≤+3).

### 2026-07-24 — (in progress; stub written at session start)

### 2026-07-25 — (in progress; stub written at session start)
- Midday state: filler 1.05 (in-band, frozen n=11); A17 canary v1 ERROR root-caused = Kaggle API silently drops model_sources pinned to 72b-instruct-awq VERSION 1 (probe-isolated, runs/model_attach_probe/); v3 pushed with /2 pin (48/48 weight shards size-identical; deviation recorded in daily brief 1a) - model attach verified by metadata round-trip, RUNNING, monitor armed. Queue re-armed (was EMPTY at session start - filler head restored). Sweeps done (Opus-5 30.2% external datapoint; boristown 1.47 unchanged). Panel R20 (3 reviewers, prior=R19) launched on the brief. Pipeline pushes 2/2 used (canary v2 diagnostic + v3 fix).

### 2026-07-26 — (in progress; stub written at session start)
- (backfill 07-27) arc-war-kit dataset version pushed (fenced_recovery_patch.py, pull-back verified); canary v4 (= v3 + fenced-recovery adapter, prereg `learnings/war_room/a17_v4_prereg_2026-07-26.md`) pushed as kernel push 1/2 → **ERROR: model mount dropped AGAIN at push** (the save-kernel API loss is nondeterministic even on the /2 pin). Model-mount route declared DEAD; principal's 07-27 addendum pins the DATASET-weights route.

### 2026-07-27 — A17 priority pin: 72B weights → DATASET route; v5 boot canary pushed
- **Weights dataset built and live: `canivel/qwen25-vl-72b-awq`** (private). No public Kaggle dataset carried Qwen2.5-VL-72B-Instruct-AWQ (searched 5 name variants; only 3B/7B/32B VL + one text-only 72B, which the multimodal contract bans). Downloaded the full HF snapshot `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` (24 files, 43,021,048,004 B, 11 shards) to F: (5.9 TB free — disk math trivial), **size-verified file-by-file against the HF API manifest (24/24)**, stripped hf transfer cache, `kaggle datasets create` at ~80 MB/s → status **ready** 12:15Z. Same serve pattern as the 27B (`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`): dataset mount + marker-based finder (v3 already proved the finder; it is mount-path agnostic, so the notebook needed NO path change).
- **Canary v5 pushed (kernel push 1/2 today; version 5) and RUNNING.** Composition = staged v4 notebook (fenced-recovery intact, serve config untouched) + 3 deltas via `duck_eval/a17/build_v5_boot_canary.py` (anchor-exact, idempotence-guarded): (1) metadata model_sources REMOVED / weights dataset attached; (2) banner `mode=boot-canary-v5-dataset-weights`; (3) SHORT window — `A17_WINDOW_S=1500`, offline soft_end = now()+1500 s at bm.run entry. Purpose per the addendum: the kernel IS the runtime test of the dataset route — vLLM boot + /v1/models identity + forced hermes tool-call round-trip + MM probe (FAIL-LOUD cell-8 boot asserts) + a ~25-min in-game slice for the fenced-recovery adapter. ~0.6–0.8 GPU-h. Smoke `duck_eval/a17/a17_v5_smoke.py` **50/50 PASS** (incl. model-finder replay against the real dataset layout + 27B-decoy refusal). **Pull-back verification (the 07-25 lesson): server metadata round-trip shows all 5 dataset_sources incl. the weights set, model_sources empty; pulled code = v5 cells.** Build memo + verdict grep list: `learnings/war_room/a17_v5_dataset_route_2026-07-27.md`. Verdict line to grep on pull: `A17-CANARY model_path=` MUST resolve under `/kaggle/input/qwen25-vl-72b-awq`.
- **Output discipline:** v5 is MEASUREMENT ONLY (boot/serve validation) — no ρ_action (needs full 7920 s window, scope v2 §3), no GO/NO-GO reading (a17_error_model.md: k=1 false-NO-GO = 1.0; interpretation only via the sealed walk + Sunday panel).
- **QUEUED (tomorrow slot-1, contingent on v5 PASS):** v6 = v5 with window restored to the sealed bench config (`A17_WINDOW_S=7920`, original budget-derived soft_end — original strings preserved verbatim in `build_v5_boot_canary.py`), banner `mode=throughput-canary-v6-dataset-weights`. That run discharges the 07-26 v4 prereg gates (G1 recovery ≥0.95, G2 ≥100 actions, G3 cadence, G4 no capability reading) AND delivers the ρ_action denominator (480/ΣN₇₂B); then freeze null_adj at measured ρ_action → seed-1 scored bench. ~2.5 GPU-h; C4 deadline Aug 3 has 6 days + retry slack. If v5 FAILs with `A17-CANARY FATAL`, today's slot-2 is the reserved retry after the fix.
- Hard-rule compliance: 1/2 kernel pushes used (dataset create is not a kernel push, per 07-26 quota classification); no submissions touched; zero cloud spend (HF download + Kaggle upload only).
