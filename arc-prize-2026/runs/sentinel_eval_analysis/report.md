# Sentinel W1 (seed 1) eval deep-dive — 2026-07-23

Kernel: `canivel/arc3-duck-sentinel-eval` (ran 2026-07-22 12:47-14:59 UTC, free Kaggle build).
Pull: `runs/kernel_pulls/sentinel_eval_v1/`. Benchmark label `duck-harness-kaggle-sentinel-v2`.
Headline (summary.txt): mean score **0.85** across 25 games, 1 pass, 0 won, 4501 total actions, 1.61M tokens.

**Verdict up front: NULL vs pre-registration (nominal -0.60 vs baseline mean, but inside baseline seed-variance; NOT the priced positive lift). Mechanism is a clean PASS (fires correctly, budget=150, game-envelope keying, <=3/game, full provenance). The agent did NOT act on the warnings - the pre-registered "mechanism fires, doesn't pay" outcome (build-doc Open Risk #2). tu93's good result is NOT the sentinel; it never fired there.**

---

## 1. Pre-registered expectation vs. result

### The pre-registration (verbatim)

- Build doc `sentinel_build_2026-07-19.md` Open Risk #2: *"Per section 2(a) the honest expectation is **+0.01-0.03/draw** and the two canonical grinders (sb26, lp85) carry expected Delta-clears of ZERO at Qwen tier - the sentinel supplies awareness, not the missing concept. The canary proves the mechanism FIRES and warns before every budget death; it does NOT prove the model acts on the warning. If the sealed 3-seed gate shows the score prongs fail with the mechanism prong firing, the honest label is 'mechanism fires, doesn't pay' (A10 guarantees no regime excuse)."*
- The task framing prices W1 as "a credible >= +0.06-0.12 experiment" (optimistic end); the build doc's own honest number is +0.01-0.03/draw. Either way the expected sign is **positive lift**.

### Result (this single seed)

Local duck-harness weighted mean score (same metric family as certified baselines - NOT the LB metric; see note):

| run | benchmark | mean score |
|---|---|---:|
| **sentinel_eval_v1 (W1)** | sentinel-v2 | **0.855** |
| war_eval_v1 (baseline seed 1) | warpack-v1 | 1.579 |
| war_eval_v2 | warpack-v1 | 1.621 |
| war_eval_v3 | warpack-v1 | 1.162 |
| **war 3-seed pooled mean** | warpack-v1 | **1.454** |
| w0_eval_s1 (continuation) | continuation-v1 | 1.731 |
| war_v2_eval_s1 (ledger v2) | warpack-v1-ledger-v2 | 0.893 |

- **Mean delta vs war 3-seed mean: -0.599** (0.855 - 1.454).
- **Mean delta vs war_eval_v1 (nominally paired "prior-stack seed 1"): -0.724.**
- Per-game vs 3-seed mean: **6 wins / 12 losses / 7 ties**. 0 games "won", same as every baseline seed.

**Sign is NEGATIVE, opposite the pre-registered positive.**

### Is this real regression or noise? - NOISE-DOMINATED

1. **Baseline seeds span 1.16-1.73** (war 1.58/1.62/1.16; w0 1.73; ledger-v2 0.89). Vanilla seed-to-seed spread ~0.5-0.8. Sentinel at 0.855 sits at/just below that band - indistinguishable from a low baseline roll (~tied with ledger-v2 0.89).
2. **Score dominated by a few high-variance games.** ar25 swings 8.33->0.98 across baseline seeds; ft09 0.00->14.29; sp80 0.00->4.76. The sentinel seed drew low on ar25 (0.98 vs baseline mean 4.40, -3.4), ft09 (0.00 vs 4.93, -4.9), sp80 (0.00 vs 3.17, -3.2). These three unlucky **non-target** draws = -11.5 raw ~ the entire -15 raw-score gap. None sentinel-attributable.
3. **Frozen-fork LB band (n=9 mean 0.992, sigma 0.155) is a DIFFERENT metric scale (LB draws)**, not comparable to this local weighted mean. "0.85 vs 0.99" is a scale coincidence.

**Prong-1 conclusion: NO lift; nominal -0.60 entirely inside baseline seed-variance and driven by unlucky high-variance non-target games. NULL-with-negative-nominal at n=1, underpowered. Refutes the optimistic +0.06-0.12 framing but cannot establish real regression.**

> **Metric note:** scores are the duck-harness weighted `mean score` from each run's `summary.txt` (parser reproduces the 0.85 headline exactly). Same metric family across sentinel/war/w0/ledger pulls => cross-run deltas are apples-to-apples. LOCAL harness metric, not the Kaggle LB metric; the frozen-fork band lives on a different scale.

---

## 2. Sentinel v2 mechanism audit - **PASS (clean)**

Source of truth: 22 sidecars `artifacts/*_tool_sentinel_events.jsonl` + 56 stdout `SENTINEL v=2` lines in `arc3-duck-sentinel-eval.log`. Both channels agree exactly (56 = 56 events).

### (a) Each threshold fires at most once per game - **PASS**
Every sidecar has thresholds in {0.5, 0.75, 0.9}, **no duplicates**, **<=3 events** (hard cap honored). Distribution: 14 fired 3, 6 fired 2, 2 fired 1 (lp85, r11l). 0 games > 3.

### (b) Cumulative game-envelope keying (budget=150, not per-attempt) - **PASS**
- Fire action-nums **identical across every game**: 50% @ **75**, 75% @ **113**, 90% @ **135** (= 0.5/0.75/0.9 x 150). All events `budget=150`.
- Cross-attempt cumulative counting demonstrated: **ar25** 50% at `attempt=1` (act 75) and 90% at `attempt=2` (act 135) - counter did NOT reset on level-up. **bp35** `attempt 3->6->8`; **sp80** `2->3->4`; **sc25** `1->2->2`. `attempt` is metadata only. Exactly the v1->v2 re-key.

### (c) Event format `SENTINEL v=2` with `unit=game-envelope` - **PASS**
- Banner: `sentinel v2: budget sentinel ACTIVE (unit=game-envelope; thresholds=50%/75%/90%; FACT injected on crossing only)`; graft-applied line confirms `applied=True; NO warpack/ledger`.
- Cell-2: `sentinel-eval: SEED=1 (a) budget sentinel ON, NO warpack (pairs with the prior-stack seed 1)`; `SENTINEL_BUDGET=150/level-attempt (R16-ruled)`.
- Event line: `SENTINEL v=2 kind=budget_threshold game=<gid>_p0_tool action_num=... budget=150 remaining=... attempt=...`. **Zero `SENTINEL v=1` lines.** `bm.label` = `duck-harness-kaggle-sentinel-v2`.
- **Open Risk #1 (uncapped budget -> inert sentinel) CLEARED**: budget=150, 56 events fired, banner ACTIVE.

### Carrier-game cross-check (ka59 / re86 / tu93 - v1's blind spots)
- **ka59** (149 actions): fired **3/3**, attempts [0,1,1] - v1 would have missed (each attempt < 75). **PASS.**
- **re86** (251 actions): fired **3/3**. **PASS.**
- **tu93** (50 actions this seed): fired **0/3** - correctly; never reached 75 cumulative actions (sections 3/4). Not a miss.

v2 repair demonstrated live on the two carriers that crossed the envelope (ka59, re86 fired early where v1 was structurally blind), on scored data.

---

## 3. Missing sidecars (s5i5, tu93, vc33) - **EXPECTED, not a bug**

All three lack a `*_tool_sentinel_events.jsonl`. Root cause (code + traces):
- `_emit_event()` in `duck_eval/sentinel/budget_sentinel_patch.py` (L211-239) writes the sidecar via `path.open("a", ...)` **only on a threshold crossing**. No file created at game start (design decision #5: "best-effort per-game sidecar"). **Lazy write => file-exists iff >=1 fire.**
- These three never crossed 50% (75 cumulative actions): **s5i5 = 51**, **tu93 = 50**, **vc33 = 68** actions (verified from each `_events.jsonl`; all < 75). Each also has **0 stdout `SENTINEL v=2` lines**.
- Therefore no sidecar (not an empty file). **Artifact-writing conditional (lazy write) = expected, NOT a bug.**
- Non-blocking recommendation: document the invariant (file-exists iff >=1 fire) in the sealing text so a future auditor does not read "missing file" as "mechanism failure." Optionally write an empty sidecar at game-open if a strict one-file-per-game invariant is ever wanted.

---

## 4. Behavioral effect - **the sentinel did NOT change agent behavior** (decisive negative)

### 4a. Post-warning progress: 1/22 fired games advanced
Level/score at action 75 (first fire) vs game end, per fired game:
- **Only tn36 (1/22)** advanced after the first warning (level 1->2, 329 actions).
- **21/22 fired games made ZERO further progress after the 50% warning.** Egregious (all got all 3 warnings, kept grinding): **wa30 560 actions** (stuck L1), **sk48 402**, **cn04 380**, **re86 251**, **sp80 196**, **tr87 189** - all ended at the level held at action 75.

### 4b. Total grinding went UP, not down
Sentinel total actions **4501** vs war 3-seed mean **3883** => **+618**. Within seed variance, but zero evidence of a "stop grinding after warning" effect. Over-grinders vs baseline: wa30 560 vs 213, sk48 402 vs 169, cn04 380 vs 224, tn36 329 vs 128.

### 4c. tu93 - the hoped-for success case - is NOT the sentinel
tu93 scored **3.97, 50 actions, 2 levels cleared** (reached L3). Is this the sentinel? **No:**
- tu93 fired **zero** sentinel events (50 < 75).
- Trajectory: cleared L1 at act 24 (score 0->1), L2 ~act 44-45 (1->2), pass **ended at act 50 status "playing"** (no GAME_OVER, no budget crossing) - harness stopped the pass; lucky efficient run.
- v1-baseline tu93 (154 actions, 12 attempts, never-fired, worst cross-attempt-waste offender) vs this 50-action clean run is a **different stochastic draw**, not a sentinel intervention. Attributing the improvement to the sentinel would be a false-positive causal claim.

**The one headline hoped to validate the mechanism demonstrates the opposite: tu93 improved with the sentinel silent, and every game where it DID fire kept grinding.**

---

## 5. Verdict

**Lift/null/regression: NULL** (nominal -0.60 vs baseline mean, one seed, inside baseline seed-variance, driven by unlucky high-variance non-target games ar25/ft09/sp80). Not a credible regression, not the priced +0.06-0.12 lift. Score prong at n=1 underpowered; it refutes only the optimistic framing.

**Mechanism evidence quality: HIGH / clean PASS.** Full provenance (banner, SENTINEL_BUDGET=150, `unit=game-envelope`, `-sentinel-v2` label, 0x v1). All 22 firing games obey the v2 contract: <=3 events/game, no duplicate thresholds, budget=150, cumulative keying (75/113/135 regardless of attempt), attempt as metadata only. v1->v2 re-key verified live on carriers ka59 & re86. Missing sidecars expected lazy-write. Open Risk #1 cleared.

**Behavioral evidence: STRONG NEGATIVE.** 21/22 fired games made no progress after the warning; total actions +618; tu93's "win" had zero sentinel involvement. The sentinel supplies awareness the Qwen-tier model does not convert - exactly build-doc Open Risk #2. **"Mechanism fires, doesn't pay"** is the honest label, and A10 guarantees no regime excuse.

### Implications for R17 condition-4 sealing
- **Condition 4 (scored-envelope +/-15% of 63k tokens): PASS on this pull.** tokens/game mean **64.3k**, median 66.1k; band [53.55k, 72.45k]; **23/25 in-band**. Only s5i5 (48.7k) and sc25 (52.4k) below - low-action early-ending games (fewer turns -> fewer tokens), not envelope drift; tu93 (56.5k) in-band. **B=150 does not need re-derivation on token grounds.** (Orchestrator should stamp the formal tokens/game +/-15% grep; numbers already in-band.)
- Mechanism prong (R15 O5 "fires before every budget death") upheld: 0 canary violations + contract-clean live firing. Sealing the **mechanism** half of condition-4 is justified.
- The **score prong should seal as NULL/underpowered at n=1, with the honest "fires-doesn't-pay" label** - NOT a lift. Component (a) is NOT a lift contributor to the conversion stack on this evidence.

### Should W2 (seed 2) be queued? - **YES, as a $0 free-build, pre-registered as a confirmatory null**
- Mechanism proven correct; a 2nd seed adds the power the single-seed score prong lacks and distinguishes "null" from a small real regression (the gate is 3-seed).
- BUT the behavioral evidence (1/22 post-warning progress, +618 actions, tu93 non-attributable) already predicts W2 lands in the null/negative band. Run W2 to **seal the gate honestly**, not because lift is expected.
- Pre-register W2's expected outcome: mean inside 1.16-1.73 baseline band, mechanism clean, behavior unchanged. If confirmed, (a)'s sealing text reads "mechanism verified, no measurable score payoff at Qwen tier." Do NOT spend a W3 unless W2 is surprisingly positive.

---

### File references
- Sentinel pull: `runs/kernel_pulls/sentinel_eval_v1/` (summary.txt, arc3-duck-sentinel-eval.log, artifacts/*.jsonl)
- Baselines: `runs/kernel_pulls/war_eval_v{1,2,3}/summary.txt`, `runs/kernel_pulls/w0_eval_s1/summary.txt`, `runs/kernel_pulls/war_v2_eval_s1/summary.txt`
- Patch (lazy-sidecar proof): `duck_eval/sentinel/budget_sentinel_patch.py` L203-239
- Sealed thresholds: `runs/sealed/r17_thresholds.json`
- Canary: `runs/sentinel_canary_v3_b150.json`
- Build/discharge docs: `learnings/war_room/sentinel_build_2026-07-19.md`, `learnings/war_room/sentinel_q2_discharge_2026-07-22.md`
