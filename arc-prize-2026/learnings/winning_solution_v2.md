# WINNING-SOLUTION PROPOSAL v2 — ARC-AGI-3 Campaign

**Author:** Proposal Architect (claude-fable-5) · **Date:** 2026-07-06 · **Revision of:** v1 after 5× MAJOR-REVISION panel · **Targets:** Milestone #2 (Sep 30, top-3 paid), Final (Nov 2, top-5 paid, ~55 private games)

---

# Thesis

The money is decided on the private ~55-game set, where per-game heuristics compress (AERA, arXiv:2605.25931). We adopt the open-sourced Duck harness as our floor in week 1, then add three capabilities, ordered by risk, **with the riskiest premise tested before any build slots are committed**:

1. **A deterministic exploration substrate** — scripted-first `explore()` (Rudakov salience tiers, directed state graph) plus a state-dedup archive **gated per-game on a pre-registered determinism/aliasing audit**. Independent of LLM strength.
2. **A runtime Rodionov verify-refactor world-model loop.** Its load-bearing premise — that Qwen 3.6 27B (not Opus) can synthesize Class-A transition models under runtime caps — is a **Phase-0 pilot**, not an August discovery. Plainly: `bfs_solve()` is a generic planner over a *supplied* transition function; on unseen games it has value **only if** this loop produces that function. The "14 games" v1 claim was clone-coupled and is retracted as evidence of private-set value.
3. **TTT value net**, entry-gated on offline ranking correlation from logged trajectories.

Every merge uses a paired per-game permutation test against an **identical-build null measured at 8–10 seeds**, with fresh-seed confirmation after selection. We do not claim "we measure, they gamble" until Phase 0 proves the instrument.

# Evidence base

1. **Duck = 1.21% Kaggle RHAE**, Qwen 3.6 27B FP8, vLLM, open-sourced Jun 30. Coding agent: one `python` tool, `action()` batching, segmentation object graphs, persisted world model, 28 threads on one vLLM server (25 games × 2.2 h/game in ~12 h wall — **token-bound, not env-bound**). Weaknesses: author-admitted irreproducibility, level-transition amnesia, no systematic exploration.
2. **Diffusion is near-instant** (125 teams ≥1.0 in 6 days). The 1.7–2.0 Sep-30 top-10 figure is **a labeled guess**; no gate depends on it.
3. **Public LB noise:** two identical-code pairs (0.43/0.22, 0.33/0.24) — point estimates only. σ is **unmeasured**; we measure it.
4. **v35 family at structural ceiling** (mean 0.25; 9 games L=0 at 5× budget).
5. **Rodionov loop survives its own leakage audit** (58.12%, GPT-5.5 — evidence for the *loop*, none for 27B synthesis). Rudakov exploration: 3rd on Preview private LB, LLM-free, open code.
6. **Live assets:** 25 opus sims (22/24 ≥50% one-step replay-exact — a build-methodology result, **not** a runtime-capability result), BFS planner, GraphExplorer, RHAE-correct eval_harness, daemon + preflight.
7. **Hard process rules** unchanged: fork-never-build, preflight, byte-matched metadata, smoke test, generality over LB luck.

# Measurement protocol (pre-registered; replaces all v1 point-threshold gates)

- **Null first.** Phase 0 runs the frozen duck baseline **8–10 seeds** locally: per-game level std, aggregate RHAE distribution of the *unchanged* build.
- **Gate definition.** A component merges iff its paired per-game delta (25 games × 3 seeds ≈ **75 paired observations**) exceeds the **90th percentile of the identical-build null** under a paired permutation test. Per-game regression checks use Benjamini–Hochberg FDR at q=0.1 (kills v1's multiplicity trap).
- **Winner's curse correction.** Every selected winner is **re-run on 2 fresh seeds never used for selection**; the confirmation number is the reported number. Shrinkage expected and budgeted.
- **Variance-reduction claims** (voting) are no longer gated criteria (F(2,2) is undecidable); voting is gated on the same paired RHAE test, **in RHAE not levels**, with no-restart and clean-restart-vs-notes-carried arms.
- **Overfitting canary.** Public set split **18 dev / 7 holdout** now; holdout touched only at phase boundaries (one sweep each); gap reported per phase.
- **Token parity.** Phase 0 measures Qwen tokens/s on the A40 (Ampere runs the FP8 checkpoint via weight-dequant; AWQ-int8 fallback if unusable) vs. the Kaggle SKU, and extracts duck's realized per-game wall-clock/concurrency from the source bundle. Local budgets set to **token parity, not time parity**; all budget-denominated criteria re-denominated in (tokens, wall).
- **LB sigma.** 4–6 submissions of the **byte-identical frozen duck fork** (filler slots v1 wasted on v35 redraws) give empirical σ with a CI. Cross-environment check: rank correlation of per-game levels, local 3× vs. Kaggle draws, before local gating is trusted for submission selection.

# The plan (Jul 7 – Sep 30)

**Cadence invariant:** daemon queue never empty; weekly queue-depth check in preflight.

### Phase 0 — Instrument + kill-questions (Jul 7–20, 2 weeks)
**Build/measure (all pre-registered):**
(a) Duck fork verbatim, fresh slug, byte-matched metadata — **2 Kaggle attempts** (24 GPU-h, fits 30 h/wk); verify empirically whether scored reruns draw from the interactive quota.
(b) Local duck-eval on RunPod A40 (spot, per-game checkpoint/resume, instance type logged); tokens/s parity measurement; **8–10-seed frozen-baseline variance decomposition**.
(c) **Qwen-27B synthesis pilot (the riskiest premise):** feed recorded transition histories from 5–10 public games + template library to Qwen 27B under runtime-realistic caps (32k context, 60 s turns, capped refactor iterations); measure Class-A rate under the **new metric** (below). Includes leave-one-game-out scaffolding (templates from games ≠ i only) to audit template leakage, and re-scoring the 22 offline sims for **plan-transfer rate** (fraction of model-found plans that realize their predicted end state live).
(d) **Determinism/aliasing audit:** replay identical action prefixes twice per game, diff frames; scan logs for identical segmentation hashes preceding divergent transitions. Archive enabled per-game only where both pass.
**Class-A redefined (one definition everywhere):** ≥50% exact-match on **held-out 5-step open-loop rollouts** (transitions the refactor loop never saw), beating two scored degenerate baselines — identity-frame and pure-lookup — under a mechanical code-size/branch-count penalty (MDL in the acceptance rule, not a prompt).
**Slots:** 7+ (duck ×2; v35 fillers until a frozen fork exists for the sigma plan).
**Exit gates:** Kaggle ≥0.9 within 2 attempts (else bisect ≤4 days; dead Jul 24 → CPU-track pivot); null distribution published; pilot Class-A rate recorded; Phase 1–3 gates numerically instantiated from the null.

### Phase 1 — Exploration substrate, scripted-first (Jul 21–Aug 3)
**Build:** `explore()` fires **automatically after N no-progress turns (scripted trigger — the primary architecture)**; determinism-gated dedup archive persisting across level transitions; salience-tiered click prioritization; frontier routing. **LLM self-routing is the A/B variant.** `bfs_solve()` ships with numeric caps: 10 s wall / 2×10⁵ nodes per call, anytime best-partial return. **Entry artifact — context budget table (worst case):** duck prompt ~4k, tool schemas ~1.2k, world-model blocks ~1.5k, archive frontier summary ~0.8k (top-k only), transition window ~4k, history ~12k, thinking headroom ~5k ≈ 29.5k/32k; eviction order history → templates → transition window; world-model blocks never evicted.
**Gate:** paired delta > null 90th pct on dev-18, **reported with and without the 14 clone-history games**, fresh-seed confirmed, FDR regressions.
**Slots:** 14.

### Phase 2 — Runtime executable world models (Aug 4–24; **entry conditional on Phase-0 pilot**)
**Entry gate:** pilot Class-A (new metric) on ≥4 piloted games across **3 pre-registered scaffold variants** (fill-in-skeleton / free-form / diff-refactor — a kill then confirms the model ceiling, not one bad prompt). Below that: verify-only mode permanently.
**Build:** in-kernel loop: synthesize → verify (held-out split vs. degenerate baselines) → refactor under MDL penalty → plan via `bfs_solve(model)` with **replanning-on-contradiction as the default control loop** (a contradicting live transition aborts the plan, triggers re-exploration). Voting: restart-after-stagnation, both arms.
**Gate:** Class-A (held-out 5-step) on ≥8/25 **and** plan-transfer ≥40% on those games; paired delta > null 90th pct; voting in RHAE. **4–7/25 zone (pre-registered):** ship verify-only + contradiction signal, no BFS-through-model; retry with remaining scaffolds; still <4 → kill synthesis permanently.
**Slots:** 21.

### Phase 3 — TTT in-kernel (Aug 25–Sep 14)
**Entry gate (offline, before any slot):** CNN trained on logged Phase-1/2 trajectories reaches Spearman ρ ≥0.3 vs. realized progress, and logs show ≥30 informative progress events/game on the target sparse games — else Phase 3 is cancelled.
**Co-residency measurement (1 day):** vLLM tokens/s at gpu_memory_utilization 0.90 vs. 0.75; CNN train-step latency co-resident vs. solo; the tax enters the 15% kill rule ex ante.
**Build:** online value net ranks probes and frontier; JEPA-XXS one clean pre-registered trial (never actually executed; prior ERRORs structural).
**Gate:** paired delta > null 90th pct, or ≥15% action-efficiency gain at equal levels, in RHAE.
**Slots:** 21.

### Phase 4 — Hardening & milestone (Sep 8–30)
**Build:** ablation matrix at Phase-0-derived seed counts; **expected score from the measured joint effect, not summed marginals**. **Freeze Sep 12–15.** Draws: 2 scored submissions/wk × ~2.5 wks = **5–6 draws, inside quota** (more if Phase 0 shows scored reruns are quota-free). Pre-registered selection: the config with higher **fresh-seed holdout worst-case per-game RHAE**; the same rule picks the Nov 2 submission.
**Success criterion:** **mean of Phase-4 draws with CI ≥ frozen-fork mean + 2σ̂** (σ̂ from the identical-fork experiment). Best-draw ≥1.6 is a reported aspiration, not a gate. "Still worth submitting" floor: mean ≥ duck-fork mean.
**Slots:** 18.

**Oct 1–Nov 2:** absorb the Sep 30 open-source wave (5-day drill); post-release edge = assets that compose with anyone's code (archive, harness, TTT, measurement protocol); October spent on holdout-7 and procedural-variant generality testing.

# Differentiators

1. **We add search + measurement; they add prompts.** No fork has a dedup archive, determinism audit, or null-calibrated A/B instrument.
2. **Risk-ordered engineering:** the weakest premise is tested in week 1; three weeks of slots are contingent, not committed.
3. **World-model scaffolding paid for:** verification metric, degenerate baselines, templates — what a 27B needs *if* the pilot passes; verify-only contradiction signal pays even if it fails.
4. **Variance engineering** under an honest, RHAE-denominated metric.
5. **Generality enforced:** dev/holdout split, leave-one-game-out template audit, no game identifiers, worst-case-game selection rule.

# Resource budget (re-derived)

- **Kaggle GPU 30 h/wk, 12 h/run → hard cap 2 scored runs/wk + 6 h smoke.** Phase 0: 2 runs. Sigma experiment: 4–6 identical-fork runs over Phases 1–2 at ≤2/wk. Phase 4: 5–6 draws post-freeze. All scheduled inside this worst case; if reruns prove quota-free, scale up.
- **RunPod A40:** token-parity sweeps costed **after** the Phase-0 tokens/s measurement; planning envelope 20–40 A40-h/sweep → ~$8–16/sweep, ≤$150/wk, ≈**$1,900 through Sep 30** (up from v1's $780). Spot preemption: checkpoint/resume mandatory; preempted runs excluded from paired comparisons.
- **Local 3080:** CNN prototyping, unit tests, offline TTT gate. **Opus:** ~5M tokens/wk (KAOS, opus-4-8).

# Kill criteria (pre-registered)

- **P0:** duck <0.8 after 2 matched attempts + 4-day bisect (dead Jul 24) → CPU-track pivot. Pilot <4 games Class-A across all 3 scaffolds → Phase 2 pre-killed; plan continues on substrate + voting (honestly a well-run fork; we accept that floor rather than pretend otherwise).
- **P1:** substrate fails its permutation gate twice consecutively → scripted-fallback-only; still regresses → pure duck.
- **P2:** as specified, including the 4–7 zone rule.
- **P3:** offline entry gate fails → cancelled before any slots. JEPA-XXS: one clean ERROR on a preflight-clean notebook → dead.
- **Global:** any component costing >15% of the *measured* per-game budget without passing its gate is removed.

# Risks

1. **27B synthesis fails (high; now tested Jul 7–20, not discovered Aug 17).** Residual plan = duck + substrate + voting, band 1.2–1.5: milestone-competitive, not thesis-fulfilling. Accepted and stated.
2. **LB noise σ exceeds local signal (likely).** Local-first gating; LB used only as distribution samples with measured σ.
3. **Sep 30 open-source wave (certain).** Compose-with-anything assets + 5-day absorption drill.
4. **Kernel/env fragility (proven).** Fork-never-build, preflight, smoke tests — hard daemon gates.
5. **Context contention (32k, five payloads).** Budget table + eviction order are Phase-1 entry artifacts; token-per-level tracked in every ablation.

# Change log from v1

- **[meth-F1]** LB σ measured empirically: 4–6 byte-identical frozen-fork submissions; "≥0.21 noise" demoted to two point estimates.
- **[meth-F2, meth-M1, rl-F1, llm-M4, sys-M2]** Statistics rebuilt: 8–10-seed frozen-baseline null in Phase 0; gates = paired per-game permutation tests (~75 obs) vs. null 90th pct; variance-reduction de-gated; regression clause → BH FDR q=0.1.
- **[meth-M2]** Fresh-seed confirmation after every selection (winner's-curse correction).
- **[meth-M3]** 18/7 dev/holdout split from day one; gap reported per phase.
- **[meth-M4]** Phase-4 success = mean of draws with CI; final-submission rule pre-registered.
- **[meth-m1]** 1.7–2.0 extrapolation labeled a guess; no gate depends on it.
- **[llm-F1, llm-M1, ps-M1]** Qwen-27B synthesis pilot moved to Phase 0 (A40, recorded transitions, runtime caps, 3 scaffolds); Phase-2 entry conditional on it — riskiest dependency tested first.
- **[llm-F1, ps-M4, rl-M2]** `bfs_solve()` clarified as generic planner over a supplied transition function; "14 games" retracted as generality evidence; Phase-1 deltas reported excluding those games.
- **[llm-M2]** Scripted `explore()` trigger is primary; LLM self-routing is the A/B variant.
- **[llm-M3]** Per-turn context budget table (~29.5k/32k) + eviction order as Phase-1 entry artifact.
- **[llm-m1, rl-m1]** Voting ablations RHAE-denominated with no-restart and clean-vs-notes arms.
- **[llm-m2, sys-M1]** Token-parity budgeting + cross-environment rank-correlation check; AWQ fallback; budget re-costed to ~$1,900.
- **[ps-M2]** Anti-Goodhart verify: held-out split, identity/lookup baselines, mechanical MDL penalty.
- **[ps-M3, rl-M3]** Class-A = held-out 5-step rollout exact-match; plan-transfer ≥40% gates Phase 2; replanning-on-contradiction default; 22 sims re-scored in Phase 0.
- **[ps-M5]** Leave-one-game-out template-leakage audit; written audit artifact scheduled.
- **[ps-m1]** One Class-A definition; gates in RHAE.
- **[ps-m2, rl-m2]** P2 4–7 zone pre-registered; 3 scaffold variants before any "ceiling" verdict.
- **[rl-M1]** Determinism/aliasing audit gates the archive per-game.
- **[rl-M4]** TTT offline entry gate (ρ ≥0.3, ≥30 events/game) or Phase 3 cancelled.
- **[rl-M5]** Additive compounding removed; joint ablation effect + "still worth submitting" floor.
- **[sys-F1]** Quota closed: Phase 0 → 2 attempts; freeze Sep 12–15 → 5–6 draws at 2/wk; rerun-quota status verified in Phase 0.
- **[sys-M3]** vLLM co-residency measured before Phase 3; tax enters the 15% rule.
- **[sys-M4]** Duck concurrency stated (28 threads, token-bound); per-game wall/tokens extracted and criteria re-denominated.
- **[sys-m1]** Spot checkpoint/resume mandatory; preempted runs excluded.
- **[sys-m2]** Numeric BFS caps (10 s / 2×10⁵ nodes, anytime) in the Phase-1 spec.

---
*Sources: panel_research_{literature,winners,ourstack,lb}.md; round-1 reviews in learnings/panel/round1/.*
