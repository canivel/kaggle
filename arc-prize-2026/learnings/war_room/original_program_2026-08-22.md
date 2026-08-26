# THE ORIGINAL-WORK PROGRAM — 2026-08-22
**Order (principal):** "produce the ORIGINAL-WORK PROGRAM — what we build ourselves, now that there is nothing left to adopt." Standing goal: **TOP-10 OR FIRST**, explicitly not "accept the band."
**Owner:** strategy-0822 (panel). **Mode: READ-ONLY on campaign operations** — no pushes, no submissions, no queue edits, no writes to any other lane's staged artifacts. Lane lock registered.
**Tags:** **[V]** computed/verified from artifacts in this session · **[V-doc]** verbatim in a verified artifact · **[INF]** inference · **[UNK]** unknown.

---

## §0 — THE VERDICT IN FIVE LINES

1. **The quadratic is already harvested.** Perfect action-efficiency on every level we currently clear is worth **+0.60 local score (+9.6%)**, i.e. **≈ +0.06 LB**. Efficiency-first replay is arithmetically dead. **[V]**
2. **Score is capped by level DEPTH, quadratically.** Per-game score ≤ `100·k(k+1)/(N(N+1))`. We clear **28 of 183** available levels. **[V, from source]**
3. **LB 2.5 ⇔ roughly ONE MORE LEVEL ON EVERY GAME** (lc 28 → 45–53). LB 3.5 ⇔ **two more on every game** (lc ≈ 65–77). **[V arithmetic; INF on the local→LB map, 5-anchor fit R²=0.99]**
4. **Every game we have ever run, in all 27 recorded 25-game evals, ended `gave_up` at exactly the 7920 s wallclock cap — 675/675 game-runs, 100%.** Nothing has ever ended on its own merits. The agent gets **17 decision turns per game** against a harness designed for **132** (the 60 s yield). **[V]**
5. **Therefore: the binding constraint is the DECISION BUDGET, not the prompt.** That is the common cause of "everything harms" and it is the only lever that has never been touched. The first arm is a **budget-elasticity curve**, it costs two free builds and zero submission slots, and it kills or confirms the entire program in one week.

---

## §1 — THE SCORE ARITHMETIC, RE-DERIVED FROM SOURCE

### 1.1 The exact formula, and the theorem nobody stated

`taaf/game.py::GameRun._compute_final_score` (bundle 20260815, lines 381–412) **[V, read this session]**:

```python
for level_idx in range(number_of_levels):
    weight = level_idx + 1                      # 1-indexed, INCREASING with depth
    total_weights += weight
    completed = level_idx < levels_completed    # a PREFIX: you clear 0..k-1
    if completed and actions > 0:
        level_score = min(115.0, (baseline / actions) ** 2 * 100)
    else:
        level_score = 0.0
    if level_score > 0: max_weights += weight
    total_score += level_score * weight
score     = total_score / total_weights
max_score = max_weights / total_weights * 100
return min(score, max_score)                    # <-- THE CAP
```

Write `N` = levels in the game, `k` = levels completed (always a prefix), `W = N(N+1)/2`, `b_i` = baseline actions, `a_i` = our actions.

> **THE CAP THEOREM [V].** Because `max_weights` sums only the levels that scored, and the aggregate is clipped at `100` per scored weight:
>
> ```
> game_score  =  min( Σ_{i<k} w_i·min(115,(b_i/a_i)²·100) ,  100·Σ_{i<k} w_i )  /  W
>             ≤  100 · k(k+1) / (N(N+1))
> ```
>
> **Beating the baseline can never raise a game above its level-completion ceiling.** The 115 bonus is purely an *offset* — it can rescue one sloppy level using one tidy one, and nothing more. Efficiency is a **loss-avoidance** term, not a gain term.
>
> And the ceiling is **quadratic in depth**: `k(k+1)/(N(N+1)) ≈ (k/N)²`. Going from 1 of 8 levels to 2 of 8 triples the ceiling (2.78 → 8.33). Going 7 → 8 adds 22.2 points. **Depth pays like a square; breadth pays like a line.**

This inverts the campaign's standing reading. `conversion_trace_2026-08-17` (exp 11) framed the transfer vector as *"attacking the action denominator"* **[V-doc]**. The denominator is real but **bounded at 1.0× in aggregate**. The numerator — depth — is unbounded until you win.

### 1.2 Reader certification (before any number is read)

I re-implemented the formula independently and ran it over every artifact on disk. **The oracle reproduces `final_score` exactly (0 mismatches) on all 675 game-runs of all 27 contract-clean 25-game evals** **[V]**, and reproduces every sealed campaign value:

| sealed value | source | oracle | match |
|---|---|---|---|
| field-floor 28 / 6.173 / 1639 | exp 25 | 28 / 6.1725 / 1639 | ✓ |
| ArmA base 30 / 5.686 / 1463 | exp 32 | 30 / 5.6858 / 1463 | ✓ |
| edge-1 18 / 2.909 / 1113 | exp 33 | 18 / 2.9085 / 1113 | ✓ |
| edge-2 20 / 3.140 / 1555 | exp 36/37 | 20 / 3.1403 / 1555 | ✓ |
| Arm 3 18 / 3.217 / 1251 | exp 30 | 18 / 3.2167 / 1251 | ✓ |
| graft-confirm 14 / 1.202 | exp 26 | 14 / 1.2019 | ✓ |
| null10 lc `[16,11,16,15,16,15,14,18,18,13]` | `local_lb_transfer` | identical | ✓ |

`runs/null10/seed*/benchmark.json` carry `base_actions_per_level: null` (baselines hidden). Baselines are a property of the game, so I imputed them by game-id **prefix** from the 25 runs that do carry them; all 25 prefixes resolve, and **no prefix has two different level-counts anywhere in the corpus** (no version drift) **[V]**. This yields the campaign's **first score-space null distribution**: vanilla duck **mean_score 1.5711, sd 0.5848 (n=10 seeds)**.

### 1.3 Where our 6.17 actually comes from — the concentration problem

Certified field-floor run `q38_field_v1`, per game (`cap` = the depth ceiling of §1.1, `next+` = points the next level would add) **[V]**:

| game | N | k | score | cap | next+ | actions | tokens | turns |
|---|---|---|---|---|---|---|---|---|
| **sb26** | 8 | **7** | **77.778** | 77.778 | **22.222** | 120 | 61,934 | 23 |
| lp85 | 8 | 3 | 16.667 | 16.667 | 11.111 | 49 | 89,495 | 12 |
| ft09 | 6 | 2 | 14.286 | 14.286 | 14.286 | 61 | 51,964 | 10 |
| sc25 | 6 | 2 | 14.286 | 14.286 | 14.286 | 37 | 92,789 | 21 |
| re86 | 8 | 2 | 5.444 | 8.333 | 8.333 | 86 | 75,921 | 16 |
| 12 games at k=1 | 6–10 | 1 | 0.49–3.57 | 1.82–4.76 | 3.6–9.5 | 18–209 | 38–102k | 11–26 |
| **8 games at k=0** | 6–9 | **0** | **0.000** | 0.000 | 2.2–4.8 | 7–92 | **708,913 (33.7%)** | **119 (28.1%)** |
| **TOTAL / MEAN** | 183 | **28** | **6.1725** | **6.7680** | — | 1639 | 2,103,403 | 424 |

Three facts that reframe everything:

- **`sb26` alone is 77.778 of the 154.3 total points = 50.4% of our entire local score** **[V]**. One game. Our headline local metric is a **single-game statistic** with 24 games of noise attached. This is why `private_base_v1` scored **lower** (5.686) on **more** levels (30): its sb26 read 4, not 7.
- **Efficiency realized = 91.2%** of the depth ceiling (`6.1725 / 6.7680`). On the levels we *do* clear we spend **709 actions against a baseline of 790 — ratio 0.90, i.e. we already beat the designers' pace** **[V]**. In the old June-30 configs this realization was 45–70%; **the field floor already collected the entire action-economy prize.**
- **930 of our 1639 actions (57%) are spent on levels we never clear** **[V]**, and 33.7% of all tokens go to the 8 games that score zero.

### 1.4 The efficiency lever, priced exactly

| scenario | lc | mean_score | Δ vs now |
|---|---|---|---|
| current | 28 | **6.1725** | — |
| **perfect efficiency, same levels** | 28 | **6.7680** | **+0.5954 (+9.6%)** |

**[V]** Under the transfer map of §1.5 that is **+0.06 LB**. *The efficiency-first replay strategy, the shortcircuit graft, the whole "attack the denominator" family, is worth six hundredths of a point.* This is the single most important negative result in this document and it required no experiment.

### 1.5 The local → LB transfer map (the campaign's first calibration)

Five configurations have BOTH a local artifact and LB draws. Local scores by the certified oracle; LB means from `runs/ledger.json` + the 50-row submission history + `local_lb_transfer_2026-08-22.md` **[V]**:

| config | local mean_score | local trim1* | local lc | LB mean (n draws) |
|---|---|---|---|---|
| vanilla duck (frozen fork) | 1.5711 (10 seeds) | 1.082 | 15.20 | **0.9316** (37) |
| duck-sentinel v2 | 0.8545 | 0.664 | 12.0 | **0.7100** (1) |
| attempt-scheduler | 1.3138 | 1.123 | 17.0 | **0.9000** (1) |
| warpack-v1 (3 seeds) | 1.4538 | 1.089 | 16.67 | **0.9360** (5) |
| **field floor (Q38 xhigh 08-07)** | **6.1725** | **3.061** | **28.0** | **1.5850** (2: 1.59, 1.58) |

\* `trim1` = mean per-game score after **dropping the single best game** — the concentration-corrected score.

Three fits, all 5 points / 2 parameters **[V, computed]**:

| predictor | best form | R² | max residual (LB units) |
|---|---|---|---|
| mean_score | `LB = 0.7854 · s^0.3924` | 0.990 | **0.039** |
| **trim1** | `LB = 0.520 + 0.3508 · t` (LINEAR) | **0.990** | **0.043** |
| lc | `LB = 0.062 + 0.0538 · lc` (LINEAR) | 0.971 | 0.079 |

Two independent things are true and both matter.

- **The map is strongly CONCAVE in raw mean_score** (exponent 0.39): doubling local score buys only **+31% LB**. Corroborated *externally and independently* by forum 736578 (Nick Pellegrin): duck+Q38 local **2.1** → LB ~1.4; his own harness local **5.0–5.4** → LB **still ~1.4**. The power fit predicts **1.04** and **1.48** for those two points — i.e. it reproduces "a 2.5× local gain buys almost nothing," where a linear map would have predicted 1.4 → 3.5 **[V-doc for his numbers; V for the fit's prediction]**.
- **The concavity largely disappears when you remove the top game.** `trim1` fits LINEARLY with the same R² and the same residuals. **[INF, strong]** Mechanism: **the LB averages away exactly the concentrated deep-run component that dominates our local mean.** §1.8 supplies the structural reason.

**Targets (model ensemble, honest range):**

| target | via mean_score | via trim1 | via lc | READ |
|---|---|---|---|---|
| LB **1.90** | 8.2–12.7 | 4.0–5.0 | 34–40 | the redraw ceiling, nothing more |
| LB **2.23** (public ceiling, FOYSAL) | 10.4–26.7 | 4.9–6.3 | 40–55 | |
| **LB 2.50 (top-13 line today)** | **12.2–19.1** | **5.6–7.3** | **45–47** | **≈ +1 level on every game** |
| LB 3.00 | 15.6–30.4 | 7.1–10.2 | 54–57 | |
| **LB 3.57 (cstl)** | **19.4–47.4** | **8.7–14.0** | **65–68** | **≈ +2 levels on every game** |

The three predictors disagree by 1.5–2× in score space (score is sb26-dominated) but **agree tightly in lc space**. **Use lc for targets, trim1 for screening, and never mean_score alone.**

### 1.6 The scenario frontier — what actually reaches 2.5

Hypothetical level gains applied to the certified field-floor run (new levels priced at **baseline** efficiency, existing levels at our achieved efficiency) **[V]**:

| scenario | lc | mean | trim1 | LB: mean-pow / trim1-lin / lc-lin |
|---|---|---|---|---|
| S0 current | 28 | 6.17 | 3.06 | 1.60 / 1.59 / 1.57 |
| S1 **perfect efficiency, same levels** | 28 | 6.77 | 3.66 | **1.66 / 1.80 / 1.57** |
| S5 **win sb26 (7→8) — banking unlocked** | 29 | 7.06 | 3.06 | **1.69 / 1.59 / 1.62** |
| S6 the 8 zero-games → 1 level each | 36 | 7.47 | 4.36 | 1.73 / 2.05 / 2.00 |
| S2 +1 level on the 17 scoring games | 45 | 12.01 | 8.01 | 2.08 / 3.33 / **2.48** |
| **S3 +1 level on ALL 25 games** | **53** | **13.31** | **9.31** | **2.17 / 3.78 / 2.91** |
| S9 the 5 deepest games WON outright | 48 | 20.92 | 16.92 | 2.59 / 6.46 / 2.64 |
| S4 +2 levels on ALL 25 | 77 | 22.96 | 18.96 | 2.69 / 7.17 / 4.20 |
| S7 +3 levels on ALL 25 | 101 | 36.03 | 32.03 | 3.21 / 11.76 / 5.50 |

**READ [V arithmetic, INF on the map]:**
- **Efficiency (S1): +0.06 to +0.21 LB. Dead.**
- **Winning sb26 (S5): +0.02 to +0.09 LB.** As a *score* play it is negligible. Its only value is arming the banking trigger — a **reachability** asset, not a points asset. Do not spend a slot on it for points.
- **The 2.5 line is S2/S3: one more level, essentially everywhere.** Every model agrees this band brackets 2.5. Nothing cheaper reaches it.
- **The 3.5 tier (cstl) is S4-and-up: two more levels everywhere, or a handful of outright wins.**

### 1.7 THE BUDGET IDENTITY — and the number that ends the argument

Every level must be bought with actions; actions come from turns; turns come from tokens; tokens come from throughput × wallclock. Measured on the certified field floor **[V, from `benchmark.json` histories]**:

```
levels(28)  <-  actions(1639)  <-  turns(424 @ 3.87 act/turn)  <-  tokens(2,103,403 @ 4961 tok/turn)
            <-  throughput(265.6 tok/s aggregate = 10.6 tok/s per stream) x wallclock(7920 s)
```

The facts that were sitting in the artifacts unread:

- **`max_runtime_s_per_game = 7920.0`, `concurrency = 28`, `analyzer_timeout = 900`** — read verbatim from the solver banner in the kernel log **[V]**.
- **All 25 games in all 27 recorded 25-game evals terminate in state `gave_up` at 7920 s. 675 of 675 game-runs. 100%. No game in the history of this campaign has ever ended for any reason other than the clock.** **[V]**
- The designed cadence is the **60-second yield** (`_LOCAL_ANALYZER_YIELD_SECONDS`, field-wide, `tool_agent.py:2139` → `"turn_time_budget"`): think ≤60 s, then act. That implies **7920/60 = 132 turns per game.**
- **We achieve 17.0 turns/game (field) and 15.5 (base) — 12.8% of the designed cadence. 85.6% of turns overrun the 60 s yield**, because one request of 4961 tokens at 10.6 tok/s takes **468 seconds**. Observed single turns run to **2448 s / 14,817 tokens.** **[V]**

> **The harness is built for an agent that thinks for one minute. Ours thinks for eight. The 60-second yield — the field's central pacing knob — has effectively never fired.**

And the cost of the target:

- **+1 level on every game costs 1448 additional actions at baseline pace** (`Σ_g b_g[k_g]`) — **+88% on 1639**, i.e. **≈ 374 extra turns, ≈ 1.9× the decision budget.** **[V]**
- **Winning all 25 games costs 17,135 baseline actions = 4,428 turns = 10.4× our budget.** **[V]** Outright wins are arithmetically out of reach this year at current throughput; **+1 level everywhere is a 1.9× problem.**

Cross-run evidence on how the budget should be *spent* (27 evals × 25 games) **[V]**:

| correlate with lc_total | r |
|---|---|
| **tokens per turn** (thinking depth) | **+0.687** |
| aggregate tok/s | +0.467 |
| total tokens | +0.466 |
| number of turns | **−0.609** |
| number of actions | **−0.637** |

Between configs, **deep thinking wins and action volume is a symptom of flailing**. Inside a single run the sign flips (`corr(tok/turn, levels) = −0.394` within the field run: the games it is *stuck on* are the ones it thinks longest about) — a clean Simpson reversal, and the within-run sign is a usable **stuck detector**. **[V]**

The serving line, verbatim from the kernel log **[V]**:

```
vllm.entrypoints.openai.api_server --model <qwen3-8-27b-fp8-repacked-v1>
  --served-model-name Qwen/Qwen3.8-27B-FP8 --tensor-parallel-size 1
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm
  --enable-prefix-caching --default-chat-template-kwargs {"preserve_thinking": true}
  --reasoning-parser qwen3 --max-model-len 65536            [vLLM 0.19.0]
```

**Not set, all at defaults: `--speculative-config` (none), `--kv-cache-dtype` (fp16), `--gpu-memory-utilization`, `--max-num-seqs`, `--max-num-batched-tokens`, `--cuda-graph-sizes`.** A dense 27B at FP8 on a 96 GB Blackwell delivering **266–290 tok/s aggregate at 25–28 concurrent streams** — when the campaign's own measurement was **42.6 tok/s at c=1 and 312.8 at c=8** (exp 13/15) — means throughput **saturates at batch ≈8 and gains nothing from 8→28.** That anomaly is worth 20 GPU-minutes to characterise. **[V for all numbers; INF for "there is headroom"]**

**And the edge-1 correction [V, and it matters for the record]:** doubling the context ceiling took aggregate throughput to **146.2 tok/s — the lowest of all 27 runs, 49% of base** — and tokens/turn from **5910 → 3072 (−48%)**. Exp 33 recorded the mechanism as *"capability harm, not slowdown — the model overthinks."* **An overthinking model emits MORE tokens per turn. This one emitted half as many, at half the decode rate.** The lc 30→18 is fully consistent with a **halved decision budget**. The verdict (HARM, flag off) stands; **the mechanism on record is probably wrong**, and its replacement is testable in one serving probe with no games played.

### 1.8 THE SUBMISSION IS NOT THE SHAPE WE SCREEN — and one closed lane reopens

From `taaf/standard_benchmarks.py::make_benchmark_kaggle_official_110` and `taaf/competition_arcade.py` (`OFFICIAL_110_RUN_COUNT = 110`, `clone_game_ids`) **[V, read this session]**:

> *"The 25 official games are repeated round-robin to **110 independent GameAPI entries**, with n_passes=1 … the repeated entries use the same deterministic clone IDs exposed by `CompetitionArcadeServer.official_110()`, making the benchmark compatible with the competition-style Arcade."*

The harness authors' own model of the scored rail is **110 runs = the 25 games × ~4.4 clones each**, at concurrency 28 ⇒ **4 waves × 7920 s = 8.8 h against a 9 h cap** (`rules_verification_2026-07-28`: host-confirmed **v3 = 9 hours** for scored runs **[V-doc]**). Our screening rail runs **1 wave, 1 clone per game.**

Four consequences, in descending confidence:

1. **[INF, strong] This is the concavity of §1.5.** The LB scores the **mean over ~4–5 independent clones per game**; our local metric scores **one draw**. A 77.8-point sb26 that happens in one clone of five contributes 77.8 locally and ~15.6 to the LB. Concentrated local gains are averaged away; broad gains are not — which is exactly why `trim1` linearises the map and raw mean does not.
2. **[V] `competition_sim` exists and is free.** `run.py --simulate-competition-arcade --competition-clone-runs N` runs the submission-shaped arcade **locally, inline, on the free rail** (the same simulator `scripts/local_gate.py` group H already drives with a stub LLM). **We have been screening the wrong shape for the entire campaign and the right shape has been one flag away.**
3. **[V, and it is a correction to a sealed verdict] The graft lane's transfer arm was killed by a property of the SCREEN, not of the mechanism.** exp 12/14: *"transfer has 0 clone siblings on the eval rail"* — true, and *unavoidable*, on a 25-unique-game single-wave benchmark. On the scored rail every game has **3–4 clone siblings**. **Cross-clone transfer is REACHABLE where it is scored and unreachable where we test it.** Banking still needs a win and stays dead (0 wins in 675 game-runs **[V]**); **transfer does not need a win.**
4. **[UNK] The wave/time envelope on the scored rail.** 4 waves × 7920 s + vLLM start (409 s observed) ≈ 8.9 h against 9 h, with `soft_end_time = None` in submission mode (`taaf_kaggle_run.ipynb` cell 7 **[V]**). Whether the final wave is ever truncated — which would zero ~25% of runs — is **unknown and directly measurable** by the same competition-sim instrument.

---

## §2 — "EVERYTHING HARMS": FOUR FRAMINGS, RANKED BY EVIDENCE

The record to explain **[V, KAOS registry]**: graft stack on June-30 (exp 19/26/30 — flat/NULL/HARM) · 08-15 bundle vs 08-07 (exp 32 — NULL, 30 ≈ 28) · context ceiling 2× (exp 33 — HARM, 18) · visible-capture contract (exp 36/37 — HARM, 20, *mechanism independently proven to work as plumbing*) · and before them P1 and EFFNOTE (exp 3/4/9 — delivered on 94–96% of turns, behaviour indistinguishable from controls).

### Rank 1 — **THE DECISION BUDGET IS BINDING, AND EVERY ADDITION SPENDS IT.** [strongest]

**For:** 675/675 game-runs die on the clock **[V]**. 17 turns/game against a designed 132 **[V]**. Cross-run `corr(actions, lc) = −0.637`, `corr(tok/turn, lc) = +0.687` **[V]**. Edge-1: throughput −51%, tokens/turn −48%, lc −40% **[V]** — a budget signature, not a comprehension signature. Edge-2 at 1555 actions vs 1463 and −10 lc: a contract forcing extra *visible* emission on every turn is a **direct tax on tokens/turn**, and at 10.6 tok/s a tax on tokens is a tax on turns. P1/EFFNOTE injected text into every turn's prompt (96%/94% delivery) at zero behavioural gain — **pure spend** **[V]**.
**Explains what others cannot:** why a mechanism *proven to deliver* (exp 17/18; H3 measured the carry live) can still *harm*. Delivery costs tokens; tokens are turns; turns are levels.
**Against:** exp 32 (08-15 bundle) was NULL at comparable tokens — but a NULL is exactly what a budget theory predicts for a change that spends nothing.
**Program implied:** buy decision budget; never spend it. **Testable to destruction in one free build (Arm 1).**

### Rank 2 — **OUR MEASUREMENT IS UNDERPOWERED AND SHAPE-WRONG.** [strong; partly already ratified]

**For:** exp 35 (sealed today): single-seed ±5 lc bands are **1.06σ–2.33σ**, pooled **1.79σ**, one-tail FP **3.7%**, and we already shipped one false positive (war_eval_v1's 22 lc). Within-config seed sd: vanilla **2.15** (n=10), warpack **4.73** (n=3), pooled **2.80** **[V-doc]**.
**Detection floor, computed here [V]:** with pooled sd 2.80 and n=1 seed per arm, the minimum detectable effect at 80% power / α=0.05 two-sided is **Δlc ≈ 11.1** — **our screen can only see effects of about +40% on the field floor.** n=2 → **7.8**; n=3 → **6.4**.
**Plus, new today:** the screen has the **wrong shape** (§1.8) — 1 clone where the LB scores 4.4 — and its headline statistic is **50.4% one game** (§1.3).
**Explains:** the NULLs (exp 19/32) honestly. Does **not** explain the HARMs: −10 and −12 lc are −3.6σ and −4.3σ against pooled sd and survive comfortably **[V-doc, exp 35]**.
**Program implied:** two seeds by default (already on Sunday's agenda), `trim1` as primary screening statistic, and **move the rail to competition shape**.

### Rank 3 — **THE POLICY IS BRITTLE TO PROMPT PERTURBATION.** [live, unfalsified, weaker]

**For:** edge-2 is the cleanest case — plumbing worked (H3), play got worse; *"the hidden reasoning channel appears LOAD-BEARING — working memory the model needs, not a leak to be plugged"* **[V-doc, 08-22 log]**. 97.64% of content is hidden (exp 17/18) **[V]**.
**Against:** unfalsifiable as stated, and it cannot explain edge-1 (a *serving* knob, no prompt change) at all.
**Program implied:** stop perturbing the prompt — already ordered on 08-22 (*"stop proposing single-knob perturbations of the field config"*). Ranked 3 because it is **already actioned** and yields no new program.

### Rank 4 — **THE FLOOR IS AT THE MODEL'S CAPABILITY CEILING.** [weakest, refuted by the board]

**Against:** twelve teams above 2.5, several with 3–12 lifetime submissions **[V-doc, attribution 08-21]**; cstl was at 2.70 *before Qwen3.8 existed* **[V-doc]**; Scott reports 3.8 locally **[V-doc]**. The same model class reaches double our score in other hands. And **10.4× is the win gap (§1.7)** — an agent at its cognitive ceiling would not be failing by a factor that large on a *budget* metric.
**Program implied:** none. Do not build for this.

> **Ranking: BUDGET (1) ≫ INSTRUMENT (2) > BRITTLENESS (3, already actioned) ≫ CEILING (4, refuted).**
> Framings 1 and 2 are complementary, not competing: **the budget is the mechanism, the instrument is why we could not see it.**

---

## §3 — ORIGINAL-WORK CANDIDATES

None of these is "add a module to the agent's prompt/context." That class is empirically dead (exp 3/4/9/36/37).

### C1 — BUDGET-ELASTICITY CURVE (diagnostic; the keystone)

**Mechanism.** `max_runtime_s_per_game` is a single float in the deployed solver. Run the certified field floor at **0.5× (3960 s)** and **3× (23,760 s)** per-game wallclock. Our 25-game eval is one wave, so 3× fits in **6.6 h + 0.2 h setup, inside the 9 h kernel** **[V]**. **Time is a free simulator for throughput**: 2× the seconds and 2× the tok/s deliver the same tokens per game.
**Why it beats the floor when others did not.** It is a **subtraction/scaling of an existing constant**, not an addition. It changes nothing the model sees. It cannot be a "clever addition."
**EV against §1.** It does not score; it **prices every other arm**. If `lc(3×) ≥ 45` the throughput family is worth the whole 2.5 gap. If `lc(3×) ≤ 33` the family is dead and we pivot with certainty in week 1.
**Cost.** 2 free builds ≈ 1.1 + 6.8 = **7.9 GPU-h** of 30/wk. **Zero submission slots.**
**How it fails.** A 3× kernel dying at hour 7 for platform reasons (2 infra deaths on record, exp 16). Mitigation: 0.5× first (cheap); gate the 3× behind MOUNTCHECK and rely on `periodic_save_interval_s = 600`, which already writes partials **[V]** — even a truncated 3× run yields a readable curve.

### C2 — DECODE THROUGHPUT (the deployable twin of C1)

**Mechanism.** Raise aggregate tok/s at fixed policy. Ordered by neutrality: **(a) speculative decoding** (`--speculative-config`, ngram/EAGLE) — with rejection sampling it is **distribution-preserving: it provably cannot change the policy, only its speed**, and reasoning traces are exactly the self-similar text spec-decode wins on; **(b) `--kv-cache-dtype fp8`** — more KV, larger effective batch; **(c)** `--gpu-memory-utilization` / `--max-num-seqs` / `--max-num-batched-tokens` against the measured saturation at c≈8; **(d)** diagnose *why* 8→28 concurrency adds nothing — likely stream idleness during action execution and prompt assembly, i.e. an **agent-loop pipelining** problem rather than a vLLM one.
**Why it beats the floor.** It is the only lever in the campaign with a **theoretical guarantee of policy-neutrality**. Against a record where every behavioural change harmed, "provably does not change behaviour" is the most valuable property an arm can have.
**EV.** If C1 shows elasticity ≈1, a 2× throughput win is **+1 level/game ⇒ LB ≈ 2.2–2.9**. At elasticity 0.5, ≈ +0.4 LB. It composes with everything.
**Cost.** A serving-only probe (fixed synthetic prompts at real lengths, sweep concurrency 1/4/8/16/28) is **~20 GPU-min, no games**. Then one build per winning knob.
**How it fails.** vLLM 0.19.0 spec-decode may not support this model/quant pair; FP8 KV may degrade quality (then it is *not* neutral and must be read as an arm); or the saturation is in the agent loop, in which case C2 becomes C3.

### C3 — DEAD-GAME TRIAGE / THROUGHPUT REALLOCATION

**Mechanism.** All 25 streams share one saturated server. The 8 zero-level games consume **33.7% of tokens and 28.1% of turns for 0.000 points** **[V]**. Abandon a game once its posterior of clearing a level is low; freed capacity accelerates survivors — 25 → 17 streams gives each survivor **+47% throughput share** at unchanged aggregate. The abandon signal is measured, not guessed: within-run **`corr(tok/turn, levels) = −0.394`** **[V]**.
**Why it beats the floor.** It is a **scheduler** change, outside the agent entirely, and composes multiplicatively with C2.
**EV.** Free +47% budget on the games that can use it. Against S2 (+1 level on the 17 scoring games ⇒ **LB 2.1–3.3**), a substantial fraction of the 2.5 gap **at zero token cost**.
**Cost.** Real code, but **CPU-testable end-to-end** on `local_gate`'s stub-LLM harness smoke; one build to confirm.
**How it fails.** (i) On the scored rail games run in waves of 28, so freed capacity may idle rather than transfer — **must be validated in competition-sim shape, not 25-game shape**. (ii) A wrongly abandoned game costs its whole ceiling; the policy must be strongly asymmetric (abandon late, only on hard evidence). (iii) The 47% assumes throughput is flat between 17 and 25 concurrency — **exactly what C2's probe measures**, so C2 must precede C3.

### C4 — THE COMPETITION-SHAPE RAIL + CROSS-CLONE TRANSFER

**Mechanism.** Two parts. **(a) Instrument:** move screening to `--simulate-competition-arcade --competition-clone-runs`, the shape the LB actually scores (§1.8). **(b) Arm:** with 3–4 clone siblings per game, a cheap cross-clone memory (clone *i* writes what it learned; clone *i+1* reads it) converts the LB's **mean over clones** from flat to increasing. This best fits a 2.5+ tier with no public kernel: it is invisible in every 25-game screen the public runs.
**Why it beats the floor.** It is not an addition *to a turn* — it is state carried *between runs*, so it does not tax tokens/turn. And it exploits an averaging structure guaranteed present on the scored rail and guaranteed absent on ours.
**EV.** If clone-to-clone score variance is large (measurable in one build), converting mean-like to max-like behaviour is worth roughly the gap between our mean and best clone — plausibly the whole §1.5 concavity, i.e. **LB ×1.5–2** **[INF, speculative until measured]**.
**Cost.** Scoped first build: **5 games × 5 clones = 25 runs**, exactly one normal eval. Full 110-run shape ≈ 8.8 h = 29% of the weekly GPU budget — affordable **once a week**.
**How it fails.** TAAF's official-110 model may not match the live gateway (**[INF]**, their model, not host-confirmed). The gateway may serve genuinely different hidden environments — then (a) is still a strict instrument improvement and (b) dies. **Carrying state across clones must be checked against the competition rules before any build.**

### C5 — A LEARNED COMPONENT TRAINED ON OUR ~675 RECORDED GAME-RUNS

**Mechanism.** We own something nobody else does: 675 fully-instrumented game-runs with per-turn tokens, actions, frames and outcomes. Two credible products, both **outside the LLM's prompt**: (i) an **abandon/continue classifier** for C3 (label = "did this game ever clear another level"; features = turn cadence, tokens/turn trend, frame-delta entropy) — small, trainable on the 3080; (ii) an **action-proposal prior** that pre-filters `_get_valid_actions` so the LLM chooses among ~5 plausible actions rather than reasoning over the full click grid — **reduces tokens/turn without reducing thinking**, the only way found so far to buy budget without buying throughput.
**Why it beats the floor.** Offline-trained, deterministic, adds **zero** tokens to the prompt; (ii) *removes* them.
**EV.** (i) makes C3 real; (ii) a 30% cut in tokens/turn is a 1.4× budget win ≈ +0.3 LB via §1.6 **[INF]**.
**Cost.** Local only (3080 / 5090 from 08-28). Weeks 3–6. Zero Kaggle GPU-h.
**How it fails.** 675 runs is many turns but few positive level-transitions; the label is rare and the classifier may be underpowered. And a bad action prior is a policy change of exactly the kind that has always harmed.

### C6 — THE 5090 AS A PERMANENT LOCAL ITERATION RAIL (from ~08-28)

**Mechanism.** 32 GB cannot hold 27B-FP8 (28.8 GB) plus KV at useful concurrency **[V, arithmetic]**. It *can* host the full harness against a **smaller** served model (8–14B) for **structural** iteration (scheduler, clone transfer, abandon policy, action prior) at 20–50× our Kaggle cadence, plus C2's low-concurrency serving probes.
**Why it matters.** Our real scarcity is **calendar days and submission slots**, not GPU-hours (30/wk are underused). A rail that turns a 2-week Kaggle loop into a 1-day local loop is worth more than any single arm.
**How it fails.** Structural results on a 14B may not transfer to 27B. Mitigation: use it only for **plumbing and policy-shape** questions, never for level counts.

### C7 — SEARCH / BACKTRACKING OVER GAME STATE

**Mechanism.** `ONLY_RESET_LEVELS=true` **[V]** means reset returns to the level start, and the harness exposes `transitions` the agent never queries. A tree search over reset-and-retry would in principle beat linear play.
**Verdict: DEFER.** At **17 turns per game**, a search with branching factor above 1 spends the entire budget on one subtree. **Search requires C1/C2 to have already succeeded.** Every search variant proposed before the budget is fixed is a proposal to spend a budget we do not have. Revisit only if `lc(3×) ≥ 45`.

### C8 — MAX-OVER-SUBMISSIONS / 2-SELECTED EXPLOITS

**Mechanism.** Public = max over submissions; private = the twin of 2 selected submissions.
**Verdict: BOUNDED, KEEP AS FLOOR ONLY.** exp 29/31: redraw ceiling ≈ 2.0 **[V-doc]**.
**New caveat, not in the record:** the public max is an **order statistic** and is upward-biased; the private twin of that same submission is drawn once. Selecting our top-2 public draws yields expected private ≈ **μ + ρ·(selection bias)**, not the public max. With μ = 1.585 and public max climbing toward ~2.0 over 30 draws, **the private score we finish on will sit materially below the public number we celebrate** unless ρ ≈ 1 **[INF]**. Nightly redraws remain free and correct — but they buy **public rank, not final standing.** Do not let the board number set the plan.

### C9 — sb26 8/8 AS A TARGETED WIN

**Verdict: DO NOT SPEND A SLOT FOR POINTS** (S5 = +0.02–0.09 LB, §1.6). Keep the standing WATCH. Its only value is arming banking, whose payoff is itself unmeasured. `sb26_mechanics.md` documents a scripted 24-action clear of L1+L2 **[V-doc]** — a *diagnostic* asset (it proves a level is solvable well inside budget when the policy is right), not a submission play.

---

## §4 — THE PROGRAM: TEN WEEKS

Constraints honoured: 1 submission/UTC-day (Arm 0 nightly redraw banks the floor for free), 2 pushes/day, 30 GPU-h/week, $50 RunPod, 5090 from ~08-28, deadline 2026-11-02.

**Standing policy, unchanged.** Arm 0 (nightly field-floor redraw) continues every night it is not displaced by a certified SIGNAL. It is free and it is our floor. Per C8 it is a **rank** instrument, not a **standings** instrument.

**Methodology changes this program adopts:**
- **Primary screening statistic becomes `trim1`** (mean per-game score minus the best game), with `lc_total` co-primary. Raw `mean_score` is retired as a primary — it is 50.4% one game and it is the component the LB averages away.
- **Two seeds per arm by default** (already on Sunday's agenda; MDE 11.1 lc at n=1 → 7.8 at n=2).
- **Screening moves to competition shape** as soon as W2 validates it.

---

### ARM 1 — BUDGET ELASTICITY (Week 1, push-ready, prereg-sealable today)

**Question.** Is `lc` a function of the decision budget, and with what slope?
**Vehicle.** The certified field floor, byte-identical, **one variable**: `max_runtime_s_per_game`.
**Arms.** `T0.5 = 3960 s` and `T3 = 23,760 s`. The `T1 = 7920 s` point already has **two independent replicates** (field 28, ArmA-base 30 — mean 29.0).
**Order.** T0.5 first (1.1 GPU-h, cheap failure); T3 second (6.8 GPU-h) with MOUNTCHECK armed and `periodic_save_interval_s = 600` relied on for partial reads.
**Certification, before any number.** Served `Qwen/Qwen3.8-27B-FP8`; `reasoning_effort` ABSENT; 08-07 anim bundle; n=25; **`max_runtime_s_per_game` echoed in the solver banner equal to this arm's value and to no sibling's**; zero graft markers; contract markers absent. Add the forbidden-marker pair to every existing certifier per the exp-34 standing rule.
**Sealed read — elasticity `ε = Δln(lc) / Δln(budget)` across the three points:**
- **ε ≥ 0.60** (`lc(T3) ≥ 45` **and** `lc(T0.5) ≤ 22`) ⇒ **HIGH ELASTICITY.** Budget program confirmed; Arms 2 and 3 fire; C7 unlocks.
- **0.25 ≤ ε < 0.60** ⇒ **PARTIAL.** Throughput worth pursuing but cannot alone reach 2.5; C2 proceeds at reduced priority, C4 promoted.
- **ε < 0.25** (`lc(T3) ≤ 33`) ⇒ **KILL THE ENTIRE BUDGET FAMILY.** C1/C2/C3/C7 die together; pivot to C4 + C5 in week 2. *This is the outcome that saves us six weeks.*
- **ANOMALY branch (added by §5 self-review):** `lc(T3) < 24` **with** turns/game up ≥2× is **not** an elasticity read — it is a long-run pathology (context growth, transcript size, trimming) and must be diagnosed, not scored.
- Bands are wide because the screen is: pooled seed sd 2.80, so `lc(T3) = 45` is **+5.7σ** against the T1 mean of 29 — a bar no draw artifact reaches; `lc(T0.5) ≤ 22` is −2.5σ.
**Kill criteria.** Two INFRA DEATHS on T3 ⇒ arm parked; the T0.5-only read is published one-sided (a large drop at half budget still establishes ε > 0 and licenses Arm 2).
**What it teaches even if it fails.** A NULL at 3× budget is the most valuable negative result available to this campaign: it retires the wallclock/throughput hypothesis, retires search, and proves the agent is policy-limited — redirecting nine weeks with certainty instead of hope. **No other experiment on the table has a failure mode this informative.**
**Cost.** 7.9 GPU-h. **0 submission slots.** Both builds fit in one day's two pushes.

### ARM 2 — POLICY-NEUTRAL THROUGHPUT (Week 1–2, gated on ε ≥ 0.25)

**Step 2a (no games, ~20 GPU-min).** Serving probe on the real SKU with the real model: fixed synthetic prompts at the measured length distribution; sweep concurrency 1/4/8/16/28; report tok/s and TTFT. **Kill before any arm:** if aggregate tok/s at c=28 is within 20% of the best achievable across all knob settings, C2 is dead and C3 inherits (the saturation is in the agent loop).
**Step 2b.** Apply, in order, the knobs that survive 2a — **speculative decoding first** (distribution-preserving ⇒ a pure speed change, read against lc with the null "no change in policy, more turns"), then `kv-cache-dtype fp8`, then batching parameters.
**Sealed read.** Primary: **turns per game** — a pure throughput metric with ~424 observations rather than 25, so it is high-powered where lc is not. Secondary: lc / trim1 vs the T1 comparator. **Predicted relation, stated pre-data:** `Δlc` should equal `ε × Δln(turns)` from Arm 1. **A throughput gain that raises turns but not lc falsifies Arm 1's elasticity** and is itself decisive.
**Kill.** No knob delivers ≥1.5× aggregate tok/s in 2a ⇒ arm closed, one build spent.
**Teaches on failure.** Whether the 8→28 saturation is server-side or loop-side — which decides whether C3 is a scheduler problem or a pipelining problem.
**Cost.** ~0.4 + 2.2 GPU-h per knob. 0 submission slots unless a certified SIGNAL heads the queue.

### ARM 3 — THE COMPETITION-SHAPE RAIL (Week 2, runs regardless of Arm 1's verdict)

**Why unconditional.** §1.8 establishes that our screening rail has the wrong shape and that a sealed verdict (transfer-unreachable, exp 12/14) is an artifact of that shape. This is an **instrument correction**, and this campaign's own history says instrument corrections outrank arms (exp 6, 22, 34, 35).
**Build.** `--simulate-competition-arcade --competition-clone-runs 25` over **5 games × 5 clones**, per-game budget unchanged, one wave — the same GPU cost as a normal eval.
**Games chosen to span the outcome range:** sb26 (k=7), lp85 (k=3), sc25 (k=2), ls20 (k=1), tr87 (k=0).
**Sealed reads (descriptive-primary; instrument arm, not mechanism arm):**
1. **Clone-to-clone score sd per game.** Large sd relative to mean supports §1.8's clone-averaging explanation of the concavity and vindicates `trim1` as primary.
2. **Does mean-over-5-clones predict our LB better than the single-clone local score?** Directly checkable against the §1.5 anchors.
3. **Clone siblings confirmed present** ⇒ file the correction to exp 12/14 and re-open transfer as a buildable arm.
**Kill.** The simulator cannot start inline on the Kaggle rail in two attempts ⇒ arm parked; the local-gate stub-LLM path still validates the plumbing at zero GPU cost.
**Teaches on failure.** Whether our screening shape can be corrected at all before the deadline.

### WEEKS 3–10 (sequenced, contingent)

| week | if ε ≥ 0.60 (budget confirmed) | if ε < 0.25 (budget dead) |
|---|---|---|
| 3–4 | **C3 dead-game triage**, built on the 5090 + local_gate stub, validated in competition shape | **C4b cross-clone transfer** built and screened in competition shape |
| 5–6 | **C5(ii) action prior** — cut tokens/turn without cutting thinking; compose with C2 | **C5(i) abandon classifier** on the 675-run corpus; then **C5(ii)** |
| 7–8 | **C7 search/backtracking** — now affordable; the first arm that spends budget deliberately | full 110-run competition-shape confirmation build (weekly, 8.8 GPU-h) |
| 9 | **Compound build**: the surviving stack, 2 seeds, competition shape | same |
| 10 | Freeze. Redraw the best certified config nightly. **Select the 2 private submissions by config mean, not by public max** (C8). | same |

**Weekly rhythm, both branches:** Sunday panel + weekly; Mon/Tue builds; Wed read + prereg; Thu/Fri builds; nightly Arm 0 throughout; **two seeds per arm**; every arm's certifier gains the forbidden-marker pair for every richer sibling (exp 34 standing rule).

---

## §5 — ADVERSARIAL SELF-REVIEW: THE CASE AGAINST ARM 1

**A1. "You have re-discovered that the agent runs out of time, which everyone knows, and dressed it as a finding."**
Partly fair. What is new is not that the clock binds but **how far off the designed cadence we are (12.8%)** and **how the gap is priced (1.9× for +1 level everywhere; 10.4× for wins)**. Still, the campaign ran 27 evals into this wall without anyone counting turns per game. The novelty here is embarrassing rather than clever.

**A2. "Time is NOT a free simulator for throughput, and that is the load-bearing assumption."**
The strongest objection. 3× wallclock and 3× tok/s deliver the same *tokens* but not the same *run*: (i) KV and conversation grow with turns either way, so per-turn cost rises with turn count — a 3×-time run reaches longer contexts, and per edge-1 **long contexts are where things went wrong**; (ii) `analyzer_timeout = 900 s` is a per-request cap that a 3×-*time* budget does not relax but a 3×-*throughput* gain effectively does; (iii) a 3× run may hit different failure modes (context trimming, transcript size) entirely. **A high `lc(3×)` is therefore an upper bound on the throughput payoff, not an estimate of it.** I accept this as a stated limitation, not a reason to skip: an upper bound of "no gain" still kills the family, which is the decision the arm exists to make.

**A3. "Elasticity may be non-monotone, and 3× may land in the harmful regime."**
Real. Edge-1 shows the system can get *worse* when a resource is increased. If `lc(T3) < 29` I must not read "budget doesn't help"; I must read "something else broke," and the diagnosis lives in turns/game and tokens/turn, which the artifact carries. **The ANOMALY branch is now written into Arm 1 because of this objection.**

**A4. "The local→LB map is a 5-point fit with 2 parameters, and you extrapolate 3× beyond the data."**
Correct — which is why I gave three functional forms and a range rather than a number, and why the target coordinate is **lc**, where the forms agree within 5%. Every §1.5 target is an extrapolation and should be held loosely. The parts that do **not** depend on the fit are the ones that matter most: the cap theorem, the +9.6% efficiency headroom, 675/675 clock deaths, 17 vs 132 turns, and the 1.9×/10.4× cost ratios. **Those survive the map being wrong.**

**A5. "You are proposing a diagnostic while the board moves 30 ranks a night."**
Yes, deliberately. The alternative is six more weeks of single-knob arms of the kind that have gone 0-for-6. Arm 1 costs **zero submission slots**, so the nightly floor is untouched; it spends only GPU-hours we are not using. **Cost of delay ≈ 0; value of the kill ≈ six weeks.**

**A6. "C2's 'policy-neutral' claim is overstated."**
Fair for FP8 KV (a real numerical change) and for batching (which can alter numerics). It is defensible only for **speculative decoding with proper rejection sampling**. The knobs are ordered accordingly and the rest are flagged as arms, not free wins.

**What would kill Arm 1 outright, stated pre-data.** (i) A serving probe showing tok/s at c=28 is already ≥80% of achievable *and* `lc(T3) ≤ 33` — the budget family dies in both its time and throughput forms. (ii) Discovery that the scored rail's per-game budget differs materially from 7920 s, making the local curve inapplicable — checkable in Arm 3. (iii) A T3 run whose turns/game does **not** scale with the budget — which would mean the arm never manipulated the variable it claims to.

---

## §6 — FINAL RANKING

| # | arm | why here | first kill criterion |
|---|---|---|---|
| **1** | **C1 budget elasticity** | prices every other arm; 0 slots; the failure branch is worth six weeks | `lc(T3) ≤ 33` ⇒ the whole budget family dies |
| **2** | **C4a competition-shape rail** | our screen has the wrong shape and one sealed verdict is an artifact of it; unconditional | simulator will not start inline in 2 attempts |
| **3** | **C2 throughput** | the only lever with a policy-neutrality guarantee | no knob reaches 1.5× tok/s in the 20-minute probe |
| 4 | C3 dead-game triage | 33.7% of tokens buy 0.000 points | throughput flat 17→25 concurrency ⇒ no capacity to reallocate |
| 5 | C4b cross-clone transfer | reachable where it is scored; invisible in public screens | clone variance small, or rules forbid inter-run state |
| 6 | C5 learned components | our 675-run corpus is an asset nobody else has | abandon classifier AUC < 0.7 offline |
| 7 | C6 5090 rail | multiplies iteration rate, not score | structural results fail to reproduce at 27B |
| 8 | C7 search | unaffordable at 17 turns/game | gated behind ε ≥ 0.60 |
| — | C8 redraw / C9 sb26 | bounded at ~2.0 / worth ≤ +0.09 LB | keep as floor and watch; never as plan |

**THE FIRST PUSH-READY ARM: C1/Arm 1, `T0.5 = 3960 s`** — a one-constant change to the certified field-floor vehicle, sealable from §4 today, 1.1 GPU-h, zero submission slots, with `T3 = 23,760 s` as the same day's second push.

---

## APPENDIX — provenance

| claim | source |
|---|---|
| score formula, cap theorem | `duck_eval/private/bundle_20260815/src/tufa-arc-agi-framework/src/taaf/game.py:381-412` |
| oracle reproduces `final_score` 675/675 | recomputation this session over all 27 contract-clean 25-game `benchmark.json` |
| null10 score distribution 1.5711 ± 0.5848 | `runs/null10/seed10{1..10}/benchmark.json`, baselines imputed by game-id prefix |
| per-game table, sb26 = 50.4% | `runs/kernel_pulls/q38_field_v1/benchmark.json` |
| 7920 s cap, concurrency 28, analyzer_timeout 900 | solver banner, `runs/kernel_pulls/private_base_v1/kernel.log` line 768 |
| 675/675 `gave_up` at the cap | `final_wallclock_seconds` + `state` over all 27 runs |
| 60 s yield / `turn_time_budget` | `.../inference/agent/tool_agent.py:1092, 2139-2140` |
| turns, tokens/turn, decode rates | per-`history` reconstruction (non-zero `generated_tokens` = turn boundary) |
| vLLM launch line, no spec-decode | `runs/kernel_pulls/private_base_v1/kernel.log`, "Starting vLLM OpenAI server" |
| official-110 clone structure | `.../taaf/standard_benchmarks.py:34-90`, `.../taaf/competition_arcade.py:36,69-101,155-165` |
| submission-mode path, `soft_end_time=None` | `.../taaf/kaggle/taaf_kaggle_run.ipynb` cells 6, 7, 9 |
| 9 h scored runtime | `learnings/rules_verification_2026-07-28.md`, `learnings/sweep_discussions_2026-07-29.md` (host-confirmed) |
| LB anchors, ledger | `runs/ledger.json` (n=37, μ 0.9316, s 0.1771), `runs/submission_log.jsonl`, `local_lb_transfer_2026-08-22.md` |
| seed sd 2.15 / 4.73 / pooled 2.80 | exp 35 |
| verdict history | KAOS `experiments` table, exp_id 1–37 |
