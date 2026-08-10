# exec-WM Scale-Up Results (24 ARC-AGI-3 games)

Validation harness: `exec_wm/validate_sim.py` over all 200 tuples per game at `--split all`.
Source: live re-run on 2026-06-26 against currently-active `<game>_sim.py` (= chosen v1/v2 winner from per-game evolutions).
Zero sim crashes across all 24 games.

> **CORRECTION 2026-08-10 (R24 minutes §5.2 vi; `duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md`).**
> **These numbers are NOT held out.** `--split all` (`validate_sim.py:56,62-64`) evaluates every
> tuple, including the ones the authoring model read while writing the sim, and the reported figure
> is for the *selected* v1/v2 winner — so it is an in-sample fit-plus-selection number.
> **91.7% is a class share (22 of 24 games in Class A), not a `state_exact` rate**; per-game
> in-sample `state_exact` runs 23.0 (r11l) to 100.0 (9 games), mean 81.1.
> On-trajectory these numbers do not transfer: for a *fixed* sim, teacher-forced accuracy across the
> three `war_eval` pulls has a **median across-source range of 0.400** (sp80 0.026/0.879/0.067;
> su15 0.309/0.149/0.808) — `runs/ewm_dryrun/report.md`, `duck_eval/r24_prep/s1_seal_audit.json`.
> Do not quote 91.7% as evidence of sim fidelity anywhere.

## Aggregated table

| game | state_exact% | pixel_match% | reward_acc% | done_acc% | n_actions | notes |
|------|--------------|--------------|-------------|-----------|-----------|-------|
| ar25 |  80.00 |  99.5651 |  98.50 | 100.00 | 7 | v2; +12.5pp from logical-position translate model |
| cd82 |  60.50 |  99.9180 |  77.00 | 100.00 | 6 | v2; 16-entry (action, pose) rotation table |
| cn04 |  77.50 |  99.9442 |  84.00 | 100.00 | 6 | v2; a5=90deg CW blob rotate + wall-lock flip |
| dc22 |  50.50 |  99.9607 |  81.50 | 100.00 | 5 | v2; path-colour collision rule |
| ft09 | 100.00 | 100.0000 | 100.00 | 100.00 | 1 | v1 saturated (single-action game) |
| g50t |  73.00 |  99.6241 |  96.00 | 100.00 | 5 | v2; row-63 hidden countdown tick (module counter) |
| ka59 |  60.50 |  99.9796 |  83.50 | 100.00 | 5 | v2; counter decoupled + adjacent-goal merge |
| lf52 | 100.00 | 100.0000 | 100.00 | 100.00 | 6 | v1 saturated |
| lp85 | 100.00 | 100.0000 | 100.00 | 100.00 | 1 | v1 saturated (rotation game) |
| ls20 | 100.00 | 100.0000 | 100.00 | 100.00 | 4 | v2; counter-band-all-3 life-pair consume |
| m0r0 |  57.50 |  99.9792 |  78.00 | 100.00 | 6 | v1 kept (v2 strategies regressed) |
| r11l |  23.00 |  98.1376 | 100.00 | 100.00 | 1 | v2; +1.5pp from double-tick rule, borderline |
| re86 |  90.50 |  99.9958 | 100.00 | 100.00 | 5 | v2; period-50 binary timer pattern |
| s5i5 |  99.50 |  99.9978 | 100.00 | 100.00 | 1 | v2; controller-panel toggle model |
| sb26 | 100.00 | 100.0000 | 100.00 | 100.00 | 3 | v1 saturated |
| sc25 |  72.50 |  99.9663 |  93.00 | 100.00 | 5 | v2; a6 panel-stamp invariant + greedy life drop |
| sk48 |  38.00 |  98.6631 |  93.50 | 100.00 | 6 | v2; sidebar cursor decoding for a3/a4/a7 |
| sp80 | 100.00 | 100.0000 | 100.00 | 100.00 | 6 | v1 saturated |
| su15 |  99.50 |  99.9976 | 100.00 | 100.00 | 2 | v2; counter-row colour inheritance from row 62 |
| tn36 | 100.00 | 100.0000 | 100.00 | 100.00 | 1 | v2; 5-button H/V toggle decode |
| tr87 | 100.00 | 100.0000 | 100.00 | 100.00 | 4 | v2; step_index%2 parity tick (module state) |
| tu93 | 100.00 | 100.0000 | 100.00 | 100.00 | 4 | v1 saturated |
| vc33 |  99.50 |  99.9678 | 100.00 | 100.00 | 1 | v1 kept (1 one-off frame, curve-fit risk) |
| wa30 |  65.00 |  99.9785 |  89.00 | 100.00 | 5 | v2; +0.006pp pixel only, marginal |

## Classification

- **A. STRONG (>=50% exact)** — will likely add value at inference
- **B. MARGINAL (25-50% exact)** — useful for some actions only
- **C. WEAK (<25%)** — may be net-zero contribution
- **D. BROKEN (errors / crashes)** — exclude from inference hook

### Class A (STRONG) — 22 games

ft09 (100.0), lf52 (100.0), lp85 (100.0), ls20 (100.0), sb26 (100.0), sp80 (100.0), tn36 (100.0), tr87 (100.0), tu93 (100.0), s5i5 (99.5), su15 (99.5), vc33 (99.5), re86 (90.5), ar25 (80.0), cn04 (77.5), g50t (73.0), sc25 (72.5), wa30 (65.0), cd82 (60.5), ka59 (60.5), m0r0 (57.5), dc22 (50.5)

### Class B (MARGINAL) — 1 game

sk48 (38.0)

### Class C (WEAK) — 1 game

r11l (23.0)

### Class D (BROKEN) — 0 games

(none — zero sim crashes across the suite)

## Totals per class

| class | count | %     |
|-------|------:|------:|
| A     |    22 | 91.7% |
| B     |     1 |  4.2% |
| C     |     1 |  4.2% |
| D     |     0 |  0.0% |
| total |    24 |  100% |

## Top candidates for v64 inference hook

Sort priority = state_exact% (primary), then reward_acc% (secondary), preferring multi-action games over 1-action games (more inference leverage).

1. **ls20** — 100.0 / 100.0 / 100.0, 4 actions. Multi-action perfect sim, life-pair consume invariant.
2. **lf52** — 100.0 / 100.0 / 100.0, 6 actions. Perfect on 6-action surface, highest action coverage among saturated games.
3. **sp80** — 100.0 / 100.0 / 100.0, 6 actions. Perfect with 6 actions; rigid paddle + deterministic fuel drain.
4. **tu93** — 100.0 / 100.0 / 100.0, 4 actions. Perfect 4-action sim.
5. **tr87** — 100.0 / 100.0 / 100.0, 4 actions. Perfect 4-action sim (parity tick fully decoded).

Tier-2 (close behind, still A-grade with great multi-action coverage):
- **ar25** (80.0, 7 actions) — highest action count in suite, strong cross-action lift.
- **re86** (90.5, 5 actions) — period-50 timer fully decoded, mechanistic.
- **cn04** (77.5, 6 actions) — rotation + wall-lock.

## Budget estimate (opus-4-8 wall-clock + tokens)

Per-game evolution loop (v1 build, observe-diff, v2 design + validate + decide):

| component                | per game | x24 games |
|--------------------------|---------:|----------:|
| wall-clock (opus-4-8)    | ~25 min  | ~10 hours |
| input tokens             | ~250k    | ~6.0M     |
| output tokens            | ~40k     | ~960k     |

Estimate is bottom-up from typical per-game agent transcripts (tuples + sim source + notes back-and-forth dominate input; v2 source + report dominate output). Plus ~5 min and ~50k tokens for this aggregation pass.

## Recommended next step

**Build v64 = v35 + ExecWMHook using class A + B sims (23 games).**

- Include all 22 class-A sims + the 1 class-B sim (sk48). sk48 is 38% exact overall but its v2 wins on the action-3/4/7 buckets where it now hits 32-41% from 0-12% — those buckets are exactly where MCTS rollouts benefit most from a non-identity prior.
- **Exclude r11l** (class C, 23%) — gain is +1.5pp over identity and concentrated in a 4-positive-example rule; net-zero risk on the BFS frontier.
- **Hook design**: `ExecWMHook.step(state, action, x, y)` dispatches to `sims/<game>_sim.py::simulate`. Gate on `game_id in CLASS_A_OR_B`. For non-covered games, fall back to v35 BFS without prior.
- **Acceptance bar**: v64 must match v35 on the BFS-suite ceiling for non-covered games (no regression) and lift the score on each covered game versus v35's identity-prior baseline on the local harness.
- Files to create: `agent/exec_wm_hook.py` (dispatcher), `agent/v64.py` (= v35 + hook wiring), `eval_harness.py --agent v64` smoke test before Kaggle push.
- Per `feedback_test_before_submit.md`: runtime-test v64 locally on at least one tuple per covered game before any `kernels push`.
