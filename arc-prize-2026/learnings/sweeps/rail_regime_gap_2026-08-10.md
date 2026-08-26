# Free-rail vs scored-rail regime gap — quantified

**Date:** 2026-08-10 · **Owner:** R25 systems-FATAL discharge · **Cost:** $0, zero pushes, zero API calls.
**Reproduce:** `uv run python scripts/rail_regime_gap.py` → `runs/rail_regime_gap_2026-08-10.json`.

> R25 systems [FATAL]: *"free-rail and scored rail are different throughput regimes → 'build-rail
> instrument is our most important asset' is unestablished for the scored twin. Quantify the regime gap."*

Every number below is computed from a repo file and cited `path:line`. Where a quantity is not
derivable from anything on disk it is marked **NOT ESTIMABLE** rather than guessed.

---

## 0. Verdict up front

**DEGRADED, not invalidated — with one directional asymmetry and one previously-unnoticed cliff.**

1. The two rails share *the same pickled solver object*: concurrency 28, `max_runtime_s_per_game`
   7920.0, `max_actions_per_game=None`, `analyzer_timeout=900.0`
   (`duck_eval/taaf_bundle/preamble.txt:2`). The **7,920 s/game guillotine and the ≈54 s/scored-action
   figure are therefore structurally identical on both rails** — R24 §3.1's core finding survives the
   transfer. The action cap is `None` on both. *Confirmed, not assumed.*
2. What differs is **load and shape**, not the per-game contract: build rail runs **25 games in 1
   wave at 25 steady children in 2.21 h**; the scored rail runs **110 games in 4 waves at 28 steady
   children in a projected 8.80 h**. The concurrency step 25→28 is **+12%** into an engine that is
   already **saturated**.
3. **De-rating factor for any latency-priced mechanism: ×1.12 (floor) to ×1.25 (measured-fit).**
   Multiply any build-rail-measured latency/throughput cost by this before comparing it to a
   scored-rail budget. Equivalently: **per-child token throughput on the scored rail is 0.80–0.89×
   the build rail.** This is one-directional — the build rail systematically *under*-prices latency.
4. **New cliff (not previously on record):** the scored rail's projected total is **32,267 s against a
   `KaggleTarget.max_runtime_s` of 32,400 s** — a **133 s (0.4%) margin** — and the notebook sets
   **`soft_end = None` in the rerun branch**, so it has no graceful-exit guard. Add the notebook's own
   ≤600 s gateway wait and the projection goes **−467 s over the ceiling.**
5. **What the build rail genuinely cannot tell us:** gateway-HTTP per-action cost, the
   minimal-diagnostics saving, per-game action counts on the official 110, and whether wave 4 ever
   finishes. All **NOT ESTIMABLE** from repo data. §6 gives the cheapest measurement that would settle
   the one leg we can actually control.
6. **The "most important asset" claim survives on necessity, not on transfer.** Build rail yields
   **300–325 per-game observations/week**; the scored rail yields **7 scalars/week** — a **43×**
   information-rate gap — and at the scored ledger's σ=0.1525 a 0.10-LB effect needs **73 calendar
   days** of two-arm A/B, i.e. **87% of the ~84 days left to the 2026-11-02 deadline**. The scored rail
   is not an instrument. It is a readout.

---

## 1. Rail definitions (from the notebook, not from memory)

Both rails execute the **same notebook and the same deployed bundle**. The branch is one env var,
`KAGGLE_IS_COMPETITION_RERUN` → `TRUE_SUBMISSION`
(`notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb`, cell 2 / cell 14):

| | **BUILD rail** (interactive Save & Run) | **SCORED rail** (competition rerun) |
|---|---|---|
| `TRUE_SUBMISSION` | `False` | `True` |
| game source | `_offline_games(.../environment_files)` — **25 bundled public games** | `_competition_games()` — live gateway `arcade.available_environments` |
| game count | **25** (`duck_eval/taaf_bundle/preamble.txt:5`) | **110** (`learnings/gap_forensics_2026-07-09.md:29`; `runs/lb_process_model/lb_process_model.py:49`) |
| env transport | in-process offline arcade | **HTTP `http://gateway:8001/`** |
| gateway wait | none | up to **600 s** (`_wait_for_gateway(timeout_s=600.0)`) |
| `soft_end_time` | `start + max_runtime_s − min(600, budget/2)` | **`None` — no soft deadline at all** |
| diagnostics | full (`TAAF_MINIMAL_DIAGNOSTICS=0`, periodic save every 600 s) | **suppressed** (`=1`) |
| artifacts retrievable | benchmark.json, transcripts, diagnostics.html, vLLM log, per-game cards | **one public LB scalar** |
| solver object | identical pickle | identical pickle |

The solver line, verbatim (`duck_eval/taaf_bundle/preamble.txt:2`):

```
HarnessSolver(label='duck-harness', ..., analyzer_timeout=900.0, max_actions_per_game=None,
              max_runtime_s_per_game=7920.0, concurrency=28, ...)
```

`concurrency=28` is confirmed as R24 §5.4 stated. The **25** steady-state children is confirmed
independently: across 24 full 25-game runs the maximum spread between first and last child launch is
**3.18 s**, and `sum(per-game wall) / overall wall` sits at **23.69–24.98** — a single fully-parallel
wave, never a queue (`scripts/rail_regime_gap.py`, §2 below).

---

## 2. Build rail — measured (24 runs: 14 `runs/kernel_pulls/*`, 10 `runs/null10/seed*`)

| quantity | value | source |
|---|---|---|
| benchmark wall | **7,928–7,988 s** (mean 7,960 s = **2.21 h**) | `*/benchmark.json` `start_time`/`end_time` |
| kernel wall (incl. setup) | 8,520–8,597 s on the 25-game pulls | last `"time":` in `runs/kernel_pulls/*/*.log` |
| setup + teardown + nbconvert | **553–654 s** (mean **587 s**) | kernel wall − benchmark wall |
| effective parallelism | 23.69–24.98 | Σ game wall / overall wall |
| max child launch spread | **3.18 s** | `game_runs[].started_at` |
| games ending **at** the 7,920 s guillotine | **99.3%** (596/600 game-runs) | `final_wallclock_seconds ≥ 7920` |
| s per scored action | **38.4–63.4** (mean **50.7**) | Σ wall / Σ actions |
| actions per game | **125–206** (mean **159**) | ditto |
| LLM turns per game | **67.4** (1,686 / 25) | `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json` `M2/war_turns` |
| s per LLM turn | **117.4** | 7,920 / 67.4 |
| generated tokens per turn | **931** | `M2/war_tok_per_turn` |

R24 §3.1 is fully reproduced: **the wall clock is the binding resource, the action cap
(`max_actions_per_game=None`) is never approached, and 99.3% of game-runs die by guillotine.**

---

## 3. Why the scored rail is a *different load*, from the harness's own arithmetic

`inference/framework/run.py:583` (`_wave_count`) and `:603` (`_max_runtime_minutes_per_game`):

```
waves            = ceil(game_count * n_passes / concurrent_jobs)
per_game_runtime = max_experiment_runtime_minutes / waves     # only when derived
```

and `solver.py:885` / `:898` cap live children at `concurrency` with a `ThreadPoolExecutor` +
`asyncio.Semaphore`.

Crucially, **`bm.games` is swapped at runtime (cell 14) but `bm.solver.max_runtime_s_per_game` is
not recomputed.** The 7,920 s figure was fixed at deploy time. Feeding it 110 games therefore does
not shrink the per-game budget — it multiplies the wave count:

| | games | waves | steady children | projected benchmark wall |
|---|---|---|---|---|
| build | 25 | **1** | **25** | 2.20 h |
| scored | 110 | **4** | **28** (last wave **26**) | **8.80 h** |

Ratio: **4.0× total wall, 1.12× concurrency, 1.00× per-game budget.**

Note this is not an accident of ours: 4 waves × 132 min = 528 min = 8.80 h is **exactly** what Tufa
sized the 7,920 s for. Confirmed by `learnings/gap_forensics_2026-07-09.md:19` — "Kaggle bundle:
concurrency 28, 7920s (132 min)/game, analyzer_timeout 900s, 1 pass … these are **Tufa's own Kaggle
scalings**". The build rail is the *quarter-scale* corner of a rail designed for 110 games.

### 3.1 The ceiling cliff — new

`KaggleTarget.max_runtime_s = **32400.0**` (9.00 h), read without unpickling from
`duck_eval/taaf_bundle/deploy_target.pkl` (BINFLOAT at offset 568; `pickletools.dis`).

```
scored projection = 4 × 7,920 s (benchmark)  +  587 s (observed setup/teardown)  =  32,267 s
ceiling                                                                          =  32,400 s
margin                                                                           =     133 s  (0.4%)
margin if the gateway wait runs to its 600 s cap                                 =    −467 s
```

And on the scored rail **`soft_end_time` is `None`** — the graceful-exit path that protects the build
rail does not exist there. If the kernel is cut, **wave 4 = 26 games = 23.6% of the official set**
takes the hit, under a metric that gives *zero credit for unfinished levels*
(`learnings/war_room/r24_successor_lane_proposal_2026-08-08.md` §2.3).

**Sensitivity, stated as a bound and not as a measurement.** `runs/lb_process_model/lb_process_model.py`
calibrates build→scored with `c = 0.922/1.594 ≈ 0.58`, attributing the entire discount to
official-set difficulty. Our 27-draw ledger gives `c = 0.9407/1.594 = 0.590`. If wave 4 were fully
lost to the ceiling, the same ledger implies an untruncated `c = 0.773`. **Nothing on disk
distinguishes "the official set is harder" from "we lose the last wave".** That is a live confound in
a model the campaign has been quoting since 07-18, and it belongs in the R25 record.

---

## 4. The engine is saturated — so the +12% concurrency is not free

Pooled over 12 twenty-five-game pulls (`runs/kernel_pulls/*/vllm-openai-server.log`,
10 s `loggers.py` samples, n = 9,470 at Running ≥ 20):

| Running | n | engine total gen tok/s (median) | per-child tok/s |
|---:|---:|---:|---:|
| 20 | 70 | 264.4 | 13.22 |
| 21 | 163 | 242.8 | 11.56 |
| 22 | 242 | 248.6 | 11.30 |
| 23 | 737 | 235.2 | 10.23 |
| 24 | 2,399 | 220.6 | 9.19 |
| 25 | 5,859 | **207.8** | **8.31** |

**Total engine throughput falls as concurrency rises** (264 → 208 tok/s from 20 → 25 children). That
is past the knee: the RTX PRO 6000 serving Qwen3.6-27B-FP8 is decode-bound, and extra children buy
nothing in aggregate while diluting each child.

- log-log slope of per-child throughput vs Running (R ≥ 20): **−1.986**.
- KV-cache occupancy at R ≥ 20: **median 60.0–77.5%, max 83.5%** across runs.
- Fraction of samples with `Waiting > 0` at R ≥ 20: **1.3–5.4%** (max Waiting 4) — i.e. the scheduler
  is *already* occasionally queueing at 25.

**De-rating 25 → 28 children:**

| estimator | per-child throughput ratio | latency multiplier |
|---|---|---|
| conservative floor, pure `1/R` | **0.893** | **1.12×** |
| empirical log-log fit (`R^−1.986`), extrapolated one bucket | **0.798** | **1.25×** |

The fit is an **extrapolation of one bucket beyond observed data** and the Running variation inside a
25-game run is partly incidental (a child between calls), so treat 1.12× as the *floor* and 1.25× as
the *plausible upper end*. Either way the sign is not in doubt and the direction is one-way.

**Consequence, stated plainly: the build rail systematically UNDER-prices latency.** Any mechanism
measured as "free" or "mildly slower" at Running = 25 pays **12–25% more** at Running = 28, on top of
whatever the gateway costs. Combined with §3.1's 133 s margin, a mechanism that adds even 0.5% to
per-game wall on the build rail can push the scored rail through the ceiling.

---

## 5. The instrument asymmetry — why the build rail is still the asset

| | build rail | scored rail |
|---|---|---|
| observations per run | 25 per-game scores + lc + actions + tokens + full transcripts + vLLM telemetry | **1 scalar** |
| runs available | **12–13 builds/week** (30 GPU-h ÷ 2.2–2.4 h, R24 §5.4) | 1 draw/day |
| per-game observations/week | **300–325** | **7** |
| diagnostics | full | suppressed by `TAAF_MINIMAL_DIAGNOSTICS=1` and unretrievable anyway |
| noise | null-screen Δlc sd **0.1223** over 25 paired games (`runs/gate_recalibration_2026-08-09.json`) | draw sd **0.1525** on a mean of **0.9407** (CV **16.2%**), n=27 |

Resolving power of the scored rail alone (α=.05 two-sided, power .80, σ=0.1525):

| LB delta to detect | draws per arm | calendar days, 2 arms |
|---|---:|---:|
| 0.05 | 146.0 | 291.9 |
| **0.10** | **36.5** | **73.0** |
| 0.15 | 16.2 | 32.4 |
| 0.25 (our gap to gold) | 5.8 | 11.7 |
| 0.40 | 2.3 | 4.6 |

**84 days remain** as of 2026-08-10 (`learnings/state_of_campaign_2026-08-09.md:9` records "~85 days" as of 08-09; deadline 2026-11-02). The scored
rail cannot resolve anything below ≈0.15 LB within the campaign. It follows that **the build rail is
the only instrument we have** — the R25 objection is not that we should switch instruments, it is
that we have been reading the instrument as if it were calibrated to the scored twin. It is not.

---

## 6. Verdict, de-rating rules, and the cheapest measurement that would close the gap

### 6.1 Disposition of the claim under review

> "the build-rail instrument is our most important asset"

**AMENDED, not struck.** Correct restatement:

> *The build rail is our only instrument with usable bandwidth (43× the scored rail's), and it shares
> the scored rail's per-game contract exactly (7,920 s guillotine, no action cap, concurrency 28
> configured, same model, same solver pickle). It differs in load — 1 wave × 25 children versus
> 4 waves × 28 children — and in transport (offline arcade vs HTTP gateway). It is therefore
> **valid for mechanism and structure, and biased-optimistic for anything priced in latency or
> wall-clock margin.***

### 6.2 Mandatory de-rating rules (proposed for R25 seal)

- **RD-1 (latency).** Any build-rail-measured wall-clock or CPU cost must be multiplied by **≥1.12**
  (floor) — **1.25** if the mechanism is LLM-decode-bound — before it is compared against a
  scored-rail budget. Report both.
- **RD-2 (margin).** Because the 7,920 s guillotine is hard and 99.3% of games hit it, in-game work
  does **not** lengthen the run — it is paid in **displaced actions**, not wall. The ceiling risk in
  §3.1 is therefore driven by **fixed cost**: `4 × 7,920 + setup ≤ 32,400` ⇒ **`setup ≤ 720 s`**,
  against an observed **587 s**, i.e. **≤133 s of new setup budget** for the whole campaign (imports,
  module loads, artifact hydration, extra wheels), and that is before the gateway wait. Any lane that
  adds fixed startup cost must be budgeted here explicitly.
- **RD-2b (displacement currency).** In-game cost must be quoted in **actions displaced**, not
  seconds: at 117.4 s/LLM-turn and 54.4 s/action, **one extra LLM turn per game ≈ 2.16 scored
  actions** (of 145.5/game) on the build rail, and **2.4–2.7** after RD-1 on the scored rail.
- **RD-3 (no scored-rail A/B below 0.15).** Do not propose any decision that requires the scored rail
  to separate arms differing by less than 0.15 LB. It cannot, in the time remaining.
- **RD-4 (generalisation).** Build-rail evidence is on **25 public games**; the scored rail is **110**.
  Any per-game-tuned artifact (e.g. hand-migrated sims) carries **zero** measured evidence for the
  other ~85 games. Report public-25 results as public-25 results.

### 6.3 What is NOT ESTIMABLE from repo data

- **Gateway-HTTP per-action overhead.** The build rail runs `OperationMode.OFFLINE` with no gateway;
  there is no artifact anywhere in the repo that times a gateway round trip. Not estimable, and not
  measurable on the build rail *at all* — only a scored submission can produce it.
- **The minimal-diagnostics saving** (periodic 600 s saves + per-frame logging suppressed).
- **Per-game action counts, level completions, or wall clock on the official 110.** The rerun emits
  one scalar and its output is not retrievable.
- **Whether wave 4 completes.** §3.1 is a projection, not an observation. It cannot be observed.

### 6.4 Cheapest measurement that would settle the one controllable leg

**One ordinary build push — no extra cost over any normal screen — pins the concurrency leg exactly.**

> **M-1 (concurrency-matched screen).** In the cell-12 customisation hook, extend `bm.games` from the
> 25 offline games to **28 game-slots** (the 25 public games + 3 duplicates) so steady-state
> `Running` sits at **28**, the scored rail's true value, still inside **one wave** and therefore
> still a **2.2 h** build. Read `actions/game`, `s/action`, per-child tok/s and `Waiting>0` frequency
> against the 25-child baseline. This converts the 0.798–0.893 extrapolation in §4 into a
> measurement, and it is the *only* leg of the regime gap we can move without a submission.
> Cost: 1 of the 12–13 weekly builds. Prerequisite under K3′: it is a *telemetry* read, not a Δlc
> screen, so it does not need the m≥3 baseline stack.

**M-2 (free, zero pushes, owed anyway).** The latency instrumentation already required by R24 §5.3
should log, per action: wall-clock delta, the vLLM `Running` value at issue time, and a
`wave_index`. That makes RD-1 self-calibrating on every future build instead of extrapolated, and it
costs nothing.

**M-3 (free).** Re-open `runs/lb_process_model/` and re-run its calibration with the truncation
branch of §3.1 as an explicit alternative to the `c=0.58` difficulty story. It is a 20-line change to
an existing deterministic script and it may materially move the campaign's model of its own LB.

---

## 7. Source index

| claim | source |
|---|---|
| concurrency 28, guillotine 7920 s, no action cap, analyzer 900 s | `duck_eval/taaf_bundle/preamble.txt:2` |
| 25 games / 1 pass in the deployed bundle | `duck_eval/taaf_bundle/preamble.txt:4` |
| `KaggleTarget.max_runtime_s = 32400.0` | `duck_eval/taaf_bundle/deploy_target.pkl` (pickletools offset 568) |
| wave arithmetic | `duck_eval/taaf_bundle/src/ARC3-Inference/inference/framework/run.py:583,599,603` |
| concurrency enforcement | `.../inference/framework/solver.py:746,885,898` |
| rerun vs interactive branch, `soft_end=None`, gateway, 600 s wait | `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` cells 2 & 14 |
| official set = 110 games | `learnings/gap_forensics_2026-07-09.md:29`; `runs/lb_process_model/lb_process_model.py:49` |
| "Tufa's own Kaggle scalings" (28 / 7920 / 900 / 1 pass) | `learnings/gap_forensics_2026-07-09.md:19` |
| per-game wall, actions, launch spread, guillotine fraction | `runs/kernel_pulls/*/benchmark.json`, `runs/null10/seed*/benchmark.json` |
| kernel wall / setup overhead | `runs/kernel_pulls/*/[!vllm]*.log` last `"time":` |
| engine throughput / Running / Waiting / KV | `runs/kernel_pulls/*/vllm-openai-server.log` |
| 1,686 LLM turns, 3,638 actions, 1,569,582 gen tokens (war-eval seed-1) | `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json` `M2` |
| scored draw ledger n=27, mean 0.9407, s 0.1526 | `runs/lb_ground_truth.md` (08-10 refresh) |
| build budget 12–13 builds/week; concurrency 28 correction | `learnings/war_room/r24_minutes_2026-08-09.md` §5.4 |
| wall clock binds, ≈54 s/action | `learnings/war_room/r24_minutes_2026-08-09.md` §3.1 |
| null-screen sd 0.1223 | `runs/gate_recalibration_2026-08-09.json` |
| deadline 2026-11-02, ~85 days as of 08-09 | `learnings/state_of_campaign_2026-08-09.md:9` |
| `c = 0.58` build→scored calibration | `runs/lb_process_model/report.md` §0; `lb_process_model.py:49` |

---

## 8. Lead's check on the cliff — intermittent wave-4 loss is NOT supported (added 2026-08-10)

The §0.4 cliff (projected 32,267 s vs a 32,400 s ceiling, −467 s with the gateway wait) has a
testable implication the analysis did not check: **if wave 4 — 26 of 110 games, 23.6% — were
*intermittently* lost, the frozen-fork ledger would be bimodal**, with a low mode at roughly
0.764× the high mode. The ledger is the right instrument for this because it is 27 draws of a
byte-identical artifact, so every bit of its spread is environment, not code.

Tested against `runs/ledger.json` (n=27, sorted):

```
0.65 0.68 0.77 0.78 0.82 0.82 0.84 0.85 0.87 0.89 0.89 0.90 0.92 0.93 0.93 0.95
0.97 0.99 1.02 1.02 1.03 1.05 1.05 1.10 1.14 1.21 1.33
```

**Finding: no bimodality.** The draws are smoothly and near-uniformly spaced across 0.65–1.33.
The largest gap is **0.12, in the upper tail (1.21 → 1.33)** — a tail, not a mode boundary; the
next largest are 0.09 and 0.07. There is no low cluster separated from a high cluster.

**A discarded false positive, recorded so it is not re-derived.** Splitting the draws at 0.9 gives
group means 0.805 and 1.034, ratio **0.779** — temptingly close to the 0.764 truncation prediction.
**This is circular and is not evidence:** splitting any unimodal sample near its median and taking
the ratio of the two half-means produces a number in that neighbourhood regardless of mechanism.
It was computed, recognised as an artifact, and is reported only to stop a future reader treating
it as confirmation.

**What this does and does not establish.**
- It argues against **intermittent** wave-4 truncation, which is the failure mode that would have
  quietly inflated our variance estimate and contaminated the σ that every gate is priced against.
- It does **not** refute the cliff. A truncation that bites **every** run bites constantly, lands
  in the *level* rather than the *variance*, and is therefore invisible to this test — our whole
  ledger would simply sit on a 110-game score we have never seen un-truncated.
- It does not bear on the de-rating factor (§0.3), which is a throughput argument, not a
  completion argument.

Consequence: the §6 measurement stays owed, but its urgency is **lowered from "possible live
scoring defect" to "unquantified constant"**. Nothing about tonight's filler changes.
