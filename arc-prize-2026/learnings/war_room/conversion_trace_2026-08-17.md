# CONVERSION TRACE — how the 2.5+ recipe actually moves
**Date:** 2026-08-17 · **Order:** trace the transferable recipe behind the 08-16/08-17 single-draw steps
**Rules honoured:** read-only (kernels pull, datasets list/download, forum via CLI). Zero pushes (both 08-17 slots already SPENT), zero submissions, zero spend.
**Provenance tags on every load-bearing claim:** **[V]** verified by direct read/download this session · **[V-doc]** verbatim claim inside a verified artifact (their words, not independently reproduced) · **[INF]** inference · **[UNK]** unknown.

---

## THE ONE SENTENCE

> **The 2.5+ recipe is not the engine and not a new harness — it is a public, CC0, default-OFF *score-mechanics graft stack* (`taaf_grafts`) that ships vendored inside the Kaggle dataset `thtennant/taaf-kaggle-source-share-fork` ("taaf source share fork **(banking)**", 1,122 downloads vs 253 views), plugs into the same Tufa duck/TAAF harness the whole 1.44–1.65 band already runs via one `install(bm, flags={...})` call in cell 12, and attacks the action denominator of the per-level score formula `min(115, (baseline/actions)² × 100)` — quadratic — with three mechanisms: win-then-replay banking, cross-clone level transfer, and no-op trimming. It moves SCORE without moving LEVELS, which is exactly why our sealed engine-swap eval (a levels metric) measured +1.67 levels ≈ nothing while the board jumped +1.1 in single draws.**

The delta the task asked for — "the delta between their 1.65 kernel and their 2.76 kernel" — **does not exist publicly for Lord Han Solo (zero public artifacts, verified)**, but the recipe itself is fully public and is quoted verbatim below from a downloaded copy.

---

## 1. PRIORITY 1 — Lord Han Solo: fully dark; the conversion is real but the artifact trail is not theirs

- `runs/lb_daily/lb_full_2026-08-17.csv`: team **Lord Han Solo = solo user `lordhansolo`**, teamId 16371045, 2.76, 34 subs, last sub 2026-08-16 22:04:57. **[V]**
- `kaggle kernels list --user lordhansolo` → **"Not found"** (zero public kernels, ever). `kaggle datasets list --user lordhansolo` → **"No datasets found."** **[V]**
- This matches the 08-13 census (`what_the_field_runs_2026-08-13.md`): Lord Han Solo was already recorded as **zero public artifacts** while sitting at 1.65 in the band. **[V]**
- ⇒ **There is no kernel to diff, no dataset upload, no discussion post.** Their +1.11-in-one-draw conversion (1.65 → 2.76, submitted 08-16 22:04) is **consistent with privately adopting the public graft stack below and flipping the aggressive flags** — the step size, the single draw, and the fact that they already ran the same harness family all fit — but **per-team attribution is [INF]. Nothing public proves what lordhansolo ran.**

The prompt's premise "the delta between their kernels IS the recipe if any of it is public" resolves as: **none of it is public — but the recipe is public anyway, one hop upstream.**

## 2. PRIORITY 2 — census of every 2.0+ team (13 teams today, not 10)

All member usernames from today's full-board CSV; every one checked for public kernels AND datasets. **[V]**

| # | team | score | members | public ARC-AGI-3 artifacts |
|---:|---|---:|---|---|
| 1 | Lord Han Solo | 2.76 | lordhansolo | **ZERO** (no kernels, no datasets) |
| 2 | cstl | 2.70 | gatamaz, tehnar | ZERO (one 2015 Theano notebook) |
| 3 | Daniel Franzen | 2.58 | dfranzen | ZERO for 2026 (only the 2024/2025 solution notebooks) |
| 4 | rellik13 | 2.53 | sirikilohit | **ZERO** — 5 lifetime subs |
| 5 | Fufront-RyanX | 2.33 | ryanxatlasai | one tiny dataset `ryanxatlasai/arc-agi3-traces-v3` (19.9 KB, 2026-05-01, 26 dl) — old, not the jump |
| 6 | Kevin E R MILLE | 2.22 | kevinermille | **one kernel: `arc3-sophia-no-action7-v1`** (08-10) — CPU, `dataset_sources: []`, agent code only, no engine attach; a bespoke agent, tells us the codename ("Sophia"), not the recipe |
| 7 | Ethan Lee | 2.13 | ethanlee43 | ZERO — 8 lifetime subs |
| 8 | Nikita Sorokin | 2.10 | nikitasorokin | ZERO — 6 lifetime subs |
| 9 | @Abstraction Lab | 2.05 | jcole75 + 4 | Jack Cole's known artifacts (runtime wheels, 2025 solutions); nothing new since the merge |
| 10 | Logical Arbitrage | 2.03 | markbarney, sonphamorg | ZERO ARC (AmnesiaBench notebooks only) |
| 11 | Extremis | 2.03 | gameryoulose | ZERO ARC |
| 12 | egangu | 2.01 | egangu | **ZERO** — 2 lifetime subs |
| 13 | Muhammad Haaris | 2.00 | muhammadhaaris27083 | ZERO ARC |

- **No new public dataset was uploaded by ANY 2.0+ member since 08-14.** **[V]**
- **No shared attached dataset is observable among the jumpers** — their scoring kernels are all private. **[V]** The common-component hypothesis therefore cannot be confirmed at the team level; it is carried by the artifact-side evidence in §3–4. **[INF]**
- The 08-17 morning finding stands: the steps arrive as **single draws on 2–8-lifetime-sub accounts** — egangu +1.96 on 1 (2 subs), rellik13 +1.25 on 1 (5 subs), Ethan Lee +0.81 on 1 (8 subs). That is the signature of *forking something that works*, not of iterating. **[V]**

## 3. THE RECIPE ITSELF — downloaded, read, quoted **[V]**

Dataset **`thtennant/taaf-kaggle-source-share-fork`** — title "**taaf source share fork (banking)**", CC0, datasetId 11022776, **1,122 downloads / 253 total views** (downloads ≫ views = attach-and-run forks, not browsing), current version 2026-08-17 00:26. Downloaded and diffed against Jakob Brüggen's stock bundle this session. **[V]**

It vendors the full Tufa duck/TAAF harness (`ARC3-Inference` + `tufa-arc-agi-framework`) **plus a 5,465-line graft layer `src/taaf-grafts/taaf_grafts/`** absent from stock: `banking_solver.py`, `transfer_solver.py`, `family_store.py`, `shortcircuit_solver.py`, `recovery.py`, `goalkeep.py`, `hudmask.py`, `retry_guard.py`, `schema_helpers/notes/void.py`, `agent_ext.py`, `composite.py`. One call installs it: `install(bm, flags={...})`, all flags default OFF, any failure degrades to stock. **[V]**

### 3.1 The scoring fact everything exploits **[V]**

Vendored `tufa-arc-agi-framework/src/taaf/game.py:403`:
```python
level_score = min(115.0, (baseline / actions) ** 2 * 100)
```
Per-level score is **quadratic in the action ratio and capped at 115**. Halve your actions on the same cleared levels → ~4× the per-level contribution (up to the cap). The board's +1.1 steps do not need one extra level of capability; they need the denominator. (Independently consistent with forum topic 728299 "finishing 4 of 6 levels scores 47.6, not 66.7". **[V]**)

The bundle also pins **`arc-agi-3-local` from `https://github.com/Tufalabs/re-arc-3.git`** — Tufa Labs' own reimplementation of the games — and `inference/utils/rearc_baselines.py` loads **`metadata_baseline_actions(game_id)`**: the per-level baseline action counts. **The score formula is fully computable offline.** **[V]**

### 3.2 The three score movers (docstrings quoted from the downloaded copy)

1. **`banking_solver.py` — "Win-then-replay banking"** **[V]**: a card's score is the **MAX over its plays**; RESET from the WIN state opens a **new play on the same card**; the framework refuses actions after a win, so the graft drives `arc_agi.EnvironmentWrapper` directly. Strategy: after a recorded win, **prune the winning trace per level** (drop every action that changed neither frame nor level) **and replay the pruned trace on a fresh play of the same card**. Divergence aborts free of charge — the recorded win still owns the card max. **Same levels, a fraction of the actions, squared.**
2. **`transfer_solver.py` — "Cross-clone replay + scout scheduler — the headline transfer knob"** **[V-doc]**: *"turns the 110 competition runs (25 public games cloned round-robin) into a cooperative pool"* — the first clone of a game family to clear a level publishes its pruned per-level action sequence to a process-global store (all 110 runs share one process, `Semaphore(28)`); **sibling clones replay it mechanically and "skip straight to the deepest already-solved level for free."** Clone identity via initial-frame fingerprint; every replayed action re-verified; any divergence falls back to the LLM loop. `transfer` implies `banking`.
3. **`shortcircuit_solver.py` — no-op overshoot trimmer** **[V]**: the stock batch loop executes all N of a repeated-action batch even after the mover hits a wall, and **every no-op still increments the scored action counter**; two-strike-confirmed no-ops are skipped. Directly cited against the quadratic penalty.

Support grafts (context/agent-side, not score-mechanical): `goalkeep` (stops the carried world model being wiped on game-over/level change — stock carried a non-empty model on only 33 of 481 turns **[V-doc]**), `hudmask` (segments the HUD band out of the board-change signal; 10 of 25 games are otherwise 100%-change **[V-doc]**), `recovery`, `retry_guard`, `schema_*`, plus a `context_window` module-global override.

### 3.3 Who wrote/carries it, and what they score

- Publisher: **`thtennant` (Teddy Tennant), team "Beyond Good and Eval" (alancai27+thtennant) — 1.28, rank ~216, BELOW US.** **[V]** Their own public kernels (`arc3-duck-v12` 40 votes, v18, v19) enable only the conservative flags (`efficiency, retry_guard, shortcircuit, goalkeep, hudmask`) — **never `banking`/`transfer`**. The distributor does not run the exploit publicly; the forkers flip the flags. **[V]**
- The graft code's own comments reference battle-plans, preregistered gates, byte-identity proofs, measured offline results — this is a disciplined engineering lineage that has been developing **in public since early July**: `kevin250304/arc3-duck-v9b-recovery-banking` (public kernel, last run **2026-07-12**) already ran `install(bm, flags={..., "recovery": True, "banking": True})`. **[V]** **Banking has been publicly available and publicly ENABLED for a month.**

## 4. PRIORITY 3 — the transfer channel, mapped end to end

```
Tufa Labs (milestone winner, harness authors)
  └─ jeroencottaar/taaf-kaggle-source-share   (2,135 downloads, the original source bundle)   [V]
  └─ github.com/Tufalabs/re-arc-3             (game reimpl + baseline_actions metadata)       [V]
       └─ community "arc3 duck vN" kernel lineage (kevin250304 v7/v9b → thtennant v12/v18/v19,
          caoyupeng fork 66 votes, boristown fast-eval 263 votes, juliancamilovilla 08-16 …)  [V]
            └─ thtennant/taaf-kaggle-source-share-fork  "(banking)" — vendored taaf_grafts,
               1,122 attach-downloads, versions through 08-17 00:26                            [V]
                 └─ ~7 public attaching kernels + an unobservable majority of PRIVATE forks   [V/INF]

PARALLEL, SEPARATE channel — the engine:
  Qwen3.8 mirrors: saltb0x 30→119 dl, mustangliu 25→99, johnlussier 6→13 (since 08-15)       [V]
  jakobbrggen/qwen3-8-27b-fp8-hf-snapshot (08-15 14:20) + jakobbrggen/taaf-kaggle-source
    (08-15 14:27, TAAF ported to Qwen3.8, branch feature/model-qwen38)                        [V]
      └─ forks: poby7722 (21 dl), chewkokwahibrainai "reasoning effort" (32 dl),
         helmirinaldi, nagabhanuja; FOYSAL Kaggle Model announced in topic 735479             [V]
      └─ first public duck+Q38 kernel: obirdy/arc3-duck-qwen-3-8-visible-memory-candidate
         (08-15 20:50, attaches saltb0x mirror + jeroencottaar source-share)                  [V]
```

- **The forum is NOT the channel.** Full topic sweep since 08-14: only 735590 ("run went backwards"), 735479 (FOYSAL's Q38 model announce), 735381, 735243, 735147. **No thread discloses banking/transfer/grafts.** The recipe moves through the kernel-fork/dataset-attach graph, silently. **[V]**
- GitHub: Tufalabs org has **no public push since 07-18**; `re-arc-3` is referenced by lockfile but not visible in the org's public repo list (private or hidden — the bundle vendors what matters anyway). **[UNK on repo status, V on lockfile pin]**
- Caveat, stated: the 1,122 download count includes attach-runs from some non-ARC kernels (two "Biohub" kernels attach it, likely dataset-spam) — the count is directional, not a clean adopter census. **[V]**

## 5. TIMELINE — and it retro-explains cstl

| when (UTC) | event | tag |
|---|---|---|
| ≤ 07-08 | share-fork dataset exists (attached by a kernel last run 07-08) | [V] |
| 07-12 | **`banking: True` runs in a PUBLIC kernel** (kevin250304 v9b) | [V] |
| 08-11 00:29 | thtennant v18 runs: grafts + goalkeep on Qwen3.6 | [V] |
| **08-11 18:25** | **cstl 1.59 → 2.52** (the previously "unexplainable" step) | [V] |
| 08-12 20:02 | cstl → 2.70 | [V] |
| 08-14 15:00 | Qwen3.8 ships; mirrors within 3h | [V] |
| 08-14→15 | first jump cluster (Franzen, Sorokin, Muroya, AbeLincoln…) | [V] |
| 08-15 14:20-27 | Jakob Brüggen publishes Q38 mirror + Q38 TAAF source bundle | [V] |
| 08-16 day | TAAF-Q38 fork bundles multiply; mirror downloads 61 → ~230 | [V] |
| 08-16 night | **single-draw steps on tiny histories**: egangu +1.96/1, LHS +1.11/1 → 2.76, rellik13 +1.25/1, Ethan Lee +0.81/1 | [V] |
| 08-17 00:26 | share-fork "(banking)" current version | [V] |

**cstl's 2.70 — flagged for five days as "chronologically immune to every explanation" — sits 18 hours downstream of a public graft-stack run and a month downstream of public banking. It is chronologically immune to the ENGINE story only. The graft story covers it with room to spare.** Attribution remains **[INF]** (cstl is dark), but the "single most important open question on the board" now has a mechanism that fits its dates, its step shape (1.59 → 2.52 in ONE submission), and its flatness since (an exploit banks; it does not compound).

## 6. PRIORITY 4 — reconciliation with our sealed refutation (the part that matters most)

Our sealed result (08-16): **Qwen3.8 at effort=medium in the duck harness → +1.67 LEVELS over 25 games, z=+0.41, REFUTE-2×; mean_score 2.795, below the best single baseline run.**

**There is no tension. The two results are about different variables, and together they PIN the mechanism:**

1. Our sealed primary was **levels**. The recipe moves **score at ~fixed levels** by shrinking the action denominator of a quadratic formula. A levels-metric instrument is structurally blind to it — correctly blind: the engine hypothesis it was testing was about capability, and capability is what it refuted.
2. **Public direct attribution now exists for "engine alone lands in the band":** Chew Kok Wah's kernel is titled *"**LB1.71** Qw3.8 27B FP8 Temperature1.0 kv16 xHigh"* **[V]** — Qwen3.8 in the TAAF harness at the xhigh default = **1.71**, i.e. the top of the old band, nowhere near 2.5. The field's own data agrees with our REFUTE.
3. The candidate "somethings" from the tasking, judged against evidence:
   - **Multi-pass/banking exploitation: CONFIRMED as the class.** Direct artifact evidence (§3), public since 07-12, "(banking)" dataset current, step signature matches. **Strongest.**
   - **A different harness (Tycho/Prime port): REFUTED** — every observed artifact is the same duck/TAAF family. **[V]**
   - **effort=xhigh/low with longer analyzer yield: real but second-order** — it tunes the same denominator the grafts attack mechanically; nothing public shows it producing a +1 step alone (1.71 at xhigh says it doesn't).
   - **TTT/LoRA on the new engine: NO EVIDENCE anywhere public.** Remains a possible private residual for Franzen specifically (his +1.34 predates the graft wave's visible spike and his pedigree is TTT), **[INF/UNK]**.
   - **Longer max_model_len: NO EVIDENCE** (stock bundle still 65536). **[V]**
4. What our refutation does NOT explain and this does: **single-draw +1 steps by 2-sub accounts.** No capability story survives that arithmetic; a fork-and-flip-flags story requires exactly that arithmetic.

**Standing corrections this trace forces:**
- The 08-13 census line "harness and agent policy are the entire public variance" survives again — but must be sharpened: **the decisive policy variance is not agent quality, it is score-mechanics exploitation of the efficiency formula.** Our own 08-12 "efficiency is the binding constraint" reframe was directionally right and we killed it for the wrong reason: we read efficiency as a *capability ceiling* when it is an *attack surface*.
- "The scores above 1.69 do not cluster, so it is not a shared artifact" (research restart §1.4) is now resolved: **shared artifact, per-team flag combinations and prune quality** — a shared *library* with knobs produces exactly a non-clustered top. **[INF, high confidence]**

## 7. VERDICT — ranked by evidence strength

1. **[V, decisive] The recipe exists, is public, is free, and is one cell:** duck/TAAF harness + `taaf_grafts` with `banking` (+`transfer` implying it) + `shortcircuit` (+`goalkeep`/`hudmask`) enabled, on either engine. Mechanism verified in source; score formula verified in source; distribution graph verified.
2. **[V] The engine is a riser, not the step:** Qwen3.8 alone = 1.71 public attribution + our own sealed REFUTE. Engine + grafts is the 2.5+ configuration; the wave timing (08-14→17) is the *coincidence* of the engine release with graft adoption reaching critical mass, which is why the engine got the credit in discussion 735243.
3. **[INF, ~85%] The single-draw jumpers (Lord Han Solo, rellik13, egangu, Ethan Lee, Kevin E R MILLE, UlinNuhaAbduh…) are running graft-class score mechanics.** No per-team attribution is possible (all private); the inference rests on step arithmetic + artifact availability + channel volume.
4. **[INF, ~65%] cstl = early graft-class adopter** (dates fit perfectly; zero direct evidence).
5. **[UNK] Franzen's residual** (+1.34 with 3 subs on 08-14): engine + his own machinery; TTT possible; nothing public.

## 8. WHAT IS LIFTABLE, WITHIN OUR ENVELOPE

Everything needed is free and public; our slot discipline is the only constraint (08-17: 2/2 spent — earliest action 08-18 slot 1).

- **Liftable today at zero cost (no slot):** the share-fork bundle is on disk in the scratchpad. Offline study of `banking_solver`/`transfer_solver` against our frozen fork's harness (same family — our `arc3-baseline` is the same duck lineage) and the vendored `re-arc-3` baselines. Also liftable free: `goalkeep` (it is literally our own "the agent FORGOT" root cause, fixed by someone else and measured — stock carries a world model on 33/481 turns) and `hudmask`.
- **The decision the coordinator must make first, and it is not technical:** banking/transfer is score-mechanics exploitation, not capability. It uses documented engine behavior (max-over-plays + full-reset-on-win), is distributed openly under CC0 by the harness authors' own community, and half the top-20 is plausibly running it — but it drives the env wrapper directly to bypass the framework's win-lock, and it colors what our leaderboard number *means* under `feedback_arc_generalization_first`. Separately, **transfer's payoff depends on the scored set actually being clones-of-25 (the bundle asserts it; we have never verified it on a scored run) — banking's does not** (scorecard mechanics are universal), so banking generalizes to the private final; transfer might not.
- **Fit to our stack:** cell-12 `install(bm, flags=...)` + one `dataset_sources` line on a frozen-fork copy; preflight D4 would see cells [12] + metadata only. Engine stays Qwen3.6 (one variable). Our q38-low arm (currently sealed for its own question) is orthogonal and unaffected.

## 9. THE SINGLE CHEAPEST DECISIVE TEST

**08-18 slot 1, one free build, one variable:** fork the frozen `arc3-baseline`, attach `thtennant/taaf-kaggle-source-share-fork`, add cell-12 `install(bm, flags={"efficiency": True, "retry_guard": True, "shortcircuit": True, "banking": True, "transfer": True})`, incumbent Qwen3.6 engine, incumbent everything else.
**Pre-registered read — SCORE-primary this time, not levels** (the mechanism predicts score, not levels): decisive CONFIRM if `mean_score` clears the baseline spread's max (3.420) by a step consistent with the formula (predict ≥2× best baseline); decisive REFUTE if inside the spread. Levels recorded as the guard metric (prediction: ~unchanged — that is the mechanism's signature, and the run doubles as a direct check that banking's replays cost us nothing). ft09-class analyzer behavior unaffected (grafts do not touch the analyzer loop except via flags we leave off).
**This one build simultaneously tests: the mechanism (§3), the reconciliation (§6), and — because it needs no new engine — cleanly separates recipe from engine, which no observation of the leaderboard can ever do.**

---

## Appendix — artifact index (all pulled/verified this session)

| artifact | why it matters |
|---|---|
| `thtennant/taaf-kaggle-source-share-fork` | **THE recipe carrier.** 1,122 dl, CC0, vendored `taaf_grafts` incl. banking/transfer. Local copy: scratchpad `conv_trace/taaf_banking/`. |
| `taaf_banking/src/taaf-grafts/taaf_grafts/banking_solver.py` | Win-then-replay banking, engine facts verified against arc_agi 0.9.8/arcengine 0.9.3 (their audit). |
| `.../transfer_solver.py` + `family_store.py` | Cross-clone replay; "110 competition runs = 25 public games cloned round-robin" claim. |
| `tufa-arc-agi-framework/src/taaf/game.py:403` | `min(115,(baseline/actions)²×100)` — the exploited formula. |
| `github.com/Tufalabs/re-arc-3` (pinned 57e46d6) | Per-level `baseline_actions` metadata → formula computable offline. |
| `kevin250304/arc3-duck-v9b-recovery-banking` | **`banking: True` in public since 07-12.** |
| `thtennant/arc3-duck-v12/v18/v19` | The public conservative-flags lineage (40 votes on v12); v19 = +hudmask, 08-17. |
| `jeroencottaar/taaf-kaggle-source-share` | Tufa's own source bundle, 2,135 dl — the root of the tree. |
| `chewkokwahibrainai/lb1-71-qw3-8-27b-fp8-temperature1-0-kv16-xhigh` | **Public attribution: Qwen3.8 + TAAF + xhigh = LB 1.71.** Engine alone ends in the band. |
| `jakobbrggen/taaf-kaggle-source` (+ poby7722/chew forks) | The Qwen3.8-ported TAAF bundle channel, 08-15 14:27. |
| `obirdy/arc3-duck-qwen-3-8-visible-memory-candidate` | First public duck+Q38 kernel (08-15). |
| `kevinermille/arc3-sophia-no-action7-v1` | Only public ARC3 kernel by any 2.0+ member; agent code, no engine. |
| Forum topics 735590/735479/735381/735243 | The field's own confusion; no disclosure of the graft channel anywhere. |

**Scratchpad working set:** `conv_trace/` under `C:\Users\dcani\AppData\Local\Temp\claude\f--kaggle\62c35e7c-0d05-4da2-99b0-f9b400a45a97\scratchpad` (taaf_banking, taaf_jakob, taaf_chew, taaf_poby, tennant_*, juliancamilovilla_*, kevin250304_*, obirdy, nikitagajbhiye30_*).

---

## DATED CORRECTION — 2026-08-18 (graft lane, coordinator-tasked verification; original left readable above)

The appendix row `chewkokwahibrainai/lb1-71-qw3-8-27b-fp8-temperature1-0-kv16-xhigh` ("Public attribution: Qwen3.8 + TAAF + xhigh = LB 1.71") **does not survive verification**:

1. **The kernel is not publicly accessible on 2026-08-18** (`kernels.get` denied; absent from the author's public kernel list). No pull of it exists in the 08-17 working set — only the author's *dataset* fork (`taaf_chew`). The 08-17 **[V]** attached to this row was a listing-title observation, not a content pull or board corroboration.
2. **The title's LB claim is contradicted by the author's own board.** LB score is max-over-submissions (monotonic). Team `Chew Kok Wah` (teamId 15657807, solo `chewkokwahibrainai`, 73 subs) reads 1.28 (08-15, 08-16) → **1.52** (08-17, 08-18). A team that had ever posted 1.71 cannot display 1.52. **"LB1.71" never appeared on the board for this team.**
3. **Consequence for §6.2:** the "public direct attribution now exists for 'engine alone lands in the band'" pillar is **DOWNGRADED from [V] to [UNSUPPORTED]** — its sole carrier was a now-inaccessible kernel's self-claimed title. The conclusion "engine is a riser, not the step" **still stands**, but on our own sealed Q38 REFUTE-2× alone, which is independent.
4. Alternates sweep (public kernel + author's board today): obirdy 1.48 (public duck+Q38 kernel 08-15 20:50, score posted 08-16 17:19 — the only candidate where a public kernel *precedes* the score, attribution [INF]); boristown 1.66 / caoyupeng 2.05 / saltb0x 1.65 / iamjasonfeng 1.59 — all either kernel-less, month-stale public kernels, online-API-only, or manipulation-flagged (caoyupeng dup-gate, 08-13 harness diff).
