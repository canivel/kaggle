# PATH FORWARD v1 — Revised Proposal After Phase-0/1 Execution

**Author:** claude-fable-5 · **Date:** 2026-07-13 · **Revises:** `winning_solution_FINAL.md` (v4, panel-approved 4/5) · **Evidence:** `failure_analysis_2026-07-13.md`, `headroom_analysis_2026-07-13.md`, `position_analysis_2026-07-13.md`
**Hard constraints:** ~$68 RunPod remaining (reserve-only, **zero default GPU spend**); free Kaggle quota 30 h/wk; ~80 daily submission windows to Sep 30; Milestone-2 Sep 30; Final Nov 2 (~55 private games).

---

# Thesis (what actually wins now)

Two weeks of instrumented execution changed the answer. Redraws are mathematically dead (P(top-100 by luck) ≈ 2.8% and shrinking), and the approved plan's additive strategy — inject exploration machinery into the duck context — is net-harmful as built: always-on context pollution costs a real −0.2 to −0.3 local even when explore never fires. The null-arm data shows exactly where the score is: **20.2% of all compute is burned on 5 games that never score, and the median good run burns 59% of its tokens grinding one level it never finishes.** The binding constraint is not budget, prompts, or exploration breadth — it is (1) compute allocation and (2) a level-2 capability wall on three games that clear level 1 in 10/10 seeds.

The revised strategy: **subtract and reallocate first, inject never-by-default, and spend the only remaining GPU dollars on one pre-registered shot at the level-2 wall.** Every gate runs on free instruments — the null10 corpus (offline, CPU) and scored Kaggle windows (free, best-score-kept) — except one $68-funded local A/B that unlocks only if forensics produces a falsifiable mechanism hypothesis.

**Re-baselined targets (honest arithmetic, difficulty ratio 0.55):** Sep-30 top-20 needs local null ≈ 2.9–3.3 (+70–95%); no identified lever stack reaches it. Config levers (+0.25 local → +0.14 official) roughly hold rank against the climbing cutoff. Levers + one grinder's level 2 (+0.42 local → +0.23 official, draws ≈ 1.15–1.25) contests top-100; all three grinders (+0.77 local → 1.35-class draws) makes Sep-30 top-100 solid, top-50 possible. **Formal objective: Sep-30 = instrumented checkpoint targeting top-100; the campaign is built for Nov-2 (~55 private games), where game-agnostic logic compresses the public-LB luck-chasers above us.** Nothing shipped may key on game identity.

# Evidence base (what 2 weeks proved and killed)

**Proved:**
1. **The headline "-0.5 in both arms" was wrong.** Rescored against null10: substrate **v1 = +0.13** (+0.42 version-stable); only v2 = −0.54, of which ~half is 3-seed lottery on bimodal ft09 (2-of-3 zeros, binomial p = 0.22). True systematic v2 cost ≈ −0.2 to −0.3.
2. **Context pollution is real and always-on.** Generated tokens/action: null 435 → v2 543 (+25%) → v1 633 (+46%). Replicated losers isolate it: ar25 loses in both arms (p = 0.009/0.008) with **zero** explores and **zero** animations — damage traced to the ~400×/game REPL-archive status line + tool-doc; su15 (p ≈ 0 both arms) is the animation-heaviest game. Mechanism: same action mix, more actions and wallclock per level → quadratic RHAE penalty (−0.33 levels, −0.19 efficiency).
3. **Death mode is STUCK, not out-of-time.** Only 9/126 good runs were still progressing in the final 15% of wall; extra budget buys little. Five games (dc22/g50t/m0r0/tr87/wa30) are dead in 10/10 seeds yet eat 20.2% of tokens. Four games (lp85/r11l/sb26/su15) clear level 1 in 10/10 seeds and level 2 in 0/10 — +1 level on the three w=2 grinders is worth **+0.52 local mean (+29%)**, the largest identified headroom.
4. **Position:** rank 187 at 1.02 (best of 5 draws; mean 0.922, σ̂ 0.074). Expected best-of-80 redraws ≈ 1.10 < today's top-100 cutoff (1.17). Minimum interesting local effect ≈ **+0.6–0.8**; substrate v1's +0.169 was 4–10× short.

**Killed / refuted:** token-budget displacement (actions/game 140→144); cache-break-fewer-turns (treated arms did *more* work); redraw-as-strategy; n=3 ungated A/Bs on bimodal games (ft09 alone swings ±2.5); the $1,900 GPU program (budget is $68); always-on REPL-archive injection and ungated animation summaries (now kill-switched-off defaults).

**Carried forward from the approved v4:** pre-registration discipline, game-as-exchangeable-unit statistics, shrinkage-aware confirmation, fork-never-build + preflight + smoke rules, no per-game parameters on private-set-facing logic. **Superseded:** the A40 sweep cadence, dev-18 sign-flip machinery, the 10-game 27B synthesis pilot at scale, and Phase-3 TTT as a Sep track (a 3080-offline gate may still feed Nov-2; it consumes no plan resources).

# The plan (phases with gates)

**Standing instruments (free):** (i) **null10** — 10 seeds × 25 games with per-action events, the permanent paired-control corpus; offline scorer reproduces the official gate report exactly. (ii) **Window gate** — each candidate build gets 2 scored windows; SE of (2-draw mean − 6-draw baseline mean) = σ̂·√(1/2+1/6) ≈ 0.060. **Promote** at Δ ≥ +0.12 (≈2 SE); **kill** at Δ < 0; **one extra window** if in between. σ̂'s χ² CI [0.044, 0.213] is printed beside every decision. Kaggle keeps best score, so a failed candidate costs only the window.

### R0 — Instrument close-out (Jul 14–16; free, CPU + 1 window)
Draw #6 completes the σ panel. Offline scorer + null10 baselines committed as the frozen control. All window-gate thresholds pre-registered in `ITERATION_LOG.md` before any candidate is submitted. **Exit:** thresholds logged; draw #6 scored.

### R1 — Reallocator build (Jul 14–27; free)
Vanilla duck + the two scheduler levers, **no context injection of any kind**:
- **L1:** deprioritize any game with `lc==0` after 120 actions (p90 time-to-first-level = 94; <10% FP), freeing ~20% of compute for progressing games. EV +0.10–0.30.
- **L2:** restart fresh episode if `lc==0` at action 90 (the pre-registered v2 detector; 39% budget remaining vs median-32-action first level). EV +0.13.
Both thresholds are null10 percentiles, game-agnostic, private-set-safe. Smoke on free Kaggle quota (mandatory `scripts/preflight.py`, runtime-test before push). **Gate:** 2-window rule. **Retry budget:** one, with L2 disabled (L1-only), if the combined build fails.

### R2 — Level-2 wall (Jul 21–Aug 10; free CPU; $68 reserve gated)
**Forensics (free):** mine null10 transcripts of sb26/su15/lp85 level-2 grinds (median 86–92% of tokens burned there): does the agent ever state the correct mechanic? Hypothesis churn, action entropy, repeated-plan signatures. Deliverable by Aug 3: a mechanism hypothesis + ONE intervention from the ranked shortlist — duck+BFS hybrid over the object graph, stall-scoped exec-WM verify loop, or capability router — with a falsifiable prediction.
**Reserve unlock (the only GPU spend):** iff the prediction is pre-registered, run 3 treated seeds × 25 games locally (~$15–25), paired against null10. **Primary gate:** ≥3/9 level-2 clears on {sb26, su15, lp85} (null10: 0/30 ever; rule-of-three p̂ < 0.1 → P(≥3/9 | p=0.1) ≈ 0.05). **Secondary (no-collateral):** mean paired RHAE delta ≥ 0 excluding ft09 (ft09 reported separately — it is 26% of all score and pure lottery at n=3). Pass → confirmation via 2-window rule (remaining reserve ≈ $45 covers one 3-seed retry OR one confirmation sweep, not both — retry only if primary ≥2/9).

### R3 — Stack, optional explore-min, freeze (Aug 11–Sep 30)
Merge every window-promoted component. **Explore-min** (optional, only on a promoted base): v2-gated explore with `PHASE1_ENABLE_REPL_ARCHIVE=0` and animation summaries capped at 5/game — the failure analysis's exact prescription; 2-window gate, one retry, then the injection family is closed until Nov. **Freeze Sep 12.** Sep 13–30: 4–5 selection draws of the frozen build; success criterion mean ≥ frozen-fork mean + 2σ̂·√(1/n_draws + 1/6) at both σ̂ CI endpoints; floor = never ship below the vanilla-duck fork.

# Submission policy (~80 windows)

- **~60 windows (75%): gated candidates** — R1 (2–3), R2 confirmation (2–3), explore-min (2–3), stacked builds and retries; every submission pre-registered (build hash + expected Δ + decision rule) before push.
- **~8 windows: frozen-fork drift sentinel**, 1/week — detects environment/version drift (15/24 games were version-unstable in July) and keeps the daemon queue never-empty.
- **~5 windows: final selection draws** of the frozen build, Sep 13–30.
- **Remainder: redundancy draws of the best promoted build** late-Sep (best-score-kept makes these strictly non-negative).
- Non-sentinel redraws of unchanged builds: **banned from draw #7** (near-zero information and rank EV).

# Kill criteria (pre-registered)

- **R1:** combined build fails 2-window gate → L1-only retry; that fails → revert to vanilla duck, levers dead for Sep.
- **R2:** no falsifiable mechanism hypothesis by Aug 3 → **$68 stays unspent**; R2 degrades to a prompt-free BFS-hybrid tested on windows only. Local gate <2/9 → intervention dead, no reserve retry.
- **Context injection:** always-on REPL archive **permanently dead** for Sep-30 (replicated p<0.01 harm, both arms). Explore-min gets exactly one gate + one retry; then the family is closed until Nov.
- **Global:** no game-ID-keyed logic ships, ever. Any component raising generated tokens/action >10% over null must have passed its own gate. Any A/B touching bimodal games uses ≥5 seeds or ft09-stratified/excluded primaries.
- **Re-baseline (Sep 1):** if stacked local Δ < +0.4, the milestone objective formally becomes top-100 defense; all remaining variant windows convert to selection/redundancy draws of the best build.

# Risks

1. **Level-2 wall is a capability gap, not a config knob (high).** If R2 fails, ceiling ≈ levers-only (+0.25 local) — we roughly hold rank, and the case for Nov rests on private-set compression of overfit rivals. Accepted; that is what the evidence supports.
2. **Seed/draw lottery.** ft09 = 26% of score, sd 9.9; σ̂'s CI is wide (df=5). Mitigated by paired-vs-null10 designs, ft09-stratified primaries, both-endpoint reporting — but a 2-window gate can still false-kill a +0.1 true effect (~30% miss); the window budget funds one retry per family.
3. **Cutoffs may outrun the S-curve projection** (+0.02/day now). Re-baseline clause absorbs this; targets are restated Sep 1 from the live LB.
4. **Difficulty ratio 0.55 is estimated from one build family**; local→official transfer may differ for scheduler-level changes. The window gate, not local delta, is the promotion authority.
5. **Env fragility (proven 5×).** Fork-never-build, byte-matched metadata, preflight, runtime smoke before every push — unchanged and mandatory.

---
*Supersedes the resource plan and Phases 1–3 of winning_solution_FINAL.md; statistical ethos and process rules carry over. Provenance: the three 2026-07-13 analysis docs and their scratchpad scripts.*
