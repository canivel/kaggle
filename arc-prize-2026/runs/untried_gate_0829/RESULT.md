# A30 — UNTRIED-SET FIREABILITY GATE — SEALED VERDICT (2026-08-29)

**VERDICT: FIRES — but the redirect target is materially WEAKER than the field
claims, and it carries ZERO observational support on our own archive.**

Instruments (both new today, both CPU-only, zero GPU / zero Kaggle slot / zero model call):
- `scripts/untried_probe.py` — 662 game-passes, 27 arms, 33,820 turns, `runs/kernel_pulls/`
- `scripts/action_profile_probe.py` — 24 official games driven on the real engine
  (`taaf.game_api`), the same instrument that replicated RESET at 192/192 on 08-28

Raw: `runs/untried_gate_0829/untried_gate.json`, `runs/untried_gate_0829/action_profile.json`

---

## 1. THE GATE — DOES A REDIRECT TARGET EXIST WHEN THE SUPERVISOR WOULD FIRE?

Stagnation window = ≥K consecutive turns with no level gain and no score gain.
The untried set is measured at the window's START — the moment a supervisor fires.

| K | windows | non-empty untried (discrete) | mean size |
|---|---|---|---|
| 10 | 758 | **65.8%** (499/758) | 2.42 |
| 25 | 510 | **72.7%** (371/510) | 2.74 |
| 50 | 263 | **75.7%** (199/263) | 3.03 |

Excluding ACTION7 entirely, K=25 still gives **60.8%** (310/510). The gate does not
depend on the one exotic control.

**★ THE SHARPEST SUB-FINDING — THE AGENT STAGNATES WITHOUT EVER TOUCHING A DIRECTION KEY.**
At K=25, in **53.1% of windows (271/510) EVERY declared arrow key was still unpressed.**
The top two untried-set compositions are the whole movement set:
`DOWN, LEFT, RIGHT, SPACE, UP` (115 windows) and `DOWN, LEFT, RIGHT, UP` (94).
Meanwhile MOUSE is pressed **28,270** times pooled. More than half of all long
stalls are the agent clicking MOUSE over and over having never once tried moving.

## 2. SUPPORTING RATES (all measured, all with n)

| measurement | ours | thtennant |
|---|---|---|
| turns carrying a `Valid actions right now:` line | **99.3%** (33,568/33,820) | — |
| declared set constant within a level | **100.0%** (948/948 blocks) | 25/25 games |
| passes that ever pressed the full declared set | **66.8%** (381/570) | 41% |
| passes that ever pressed the full DISCRETE set (MOUSE excluded) | **71.4%** (407/570) | — |

92 passes carried no declared line and are reported **UNMEASURED**, not dropped.

**★ ACTION7 IS DECLARED IN 137 PASSES AND PRESSED 0 TIMES IN 33,820 TURNS.**
This is the exact shape of tennant's named case, and stronger — his was zero
presses in 12 passes. Note the inversion alongside it: **RESET is pressed 229
times while never appearing in the declared list**, and ACTION7 is declared
constantly and never pressed. The agent reaches for the undocumented control and
ignores the advertised one.

## 3. THE LIVENESS HALF — AND WHERE WE FAIL TO REPLICATE THE FIELD

A never-pressed control only matters if pressing it does something. Our archive
cannot say (we never pressed it), so this was measured on the engine directly:
one press from a byte-identical level-1 opening, RESET between presses.

| action | declared | moved board | rate |
|---|---|---|---|
| ACTION6 (MOUSE) | 18 | 13 | 72.2% *(ANY of 20 coords — upper bound)* |
| ACTION3 | 16 | 9 | 56.2% |
| ACTION4 | 16 | 9 | 56.2% |
| ACTION1 | 15 | 8 | 53.3% |
| ACTION2 | 15 | 8 | 53.3% |
| ACTION5 | 8 | 4 | 50.0% |
| **ACTION7** | **6** | **2** | **33.3%** |

**WE DO NOT REPLICATE 91%.** Our discrete one-press rate is **~53%**. The honest
comparison is against tennant's *one-press-at-opening* figure of **69%** (411/600),
not his 91% (which allows an action to move the board anywhere, not on one press).
Ours is materially below even that.

**SCOPE, STATED SO NOBODY QUOTES THIS AS A 600-PAIR REPLICATION:** level-1 opening
only. `GameAPI` exposes `number_of_levels` but no setter, so deeper levels need
real play and are NOT covered. n=6 for ACTION7 is a small sample and is labelled
as such. One game (`cn04-65d47d14`) failed to start
(`base_actions_per_level has 6 entries; number_of_levels is 5`) and is reported,
not dropped — 24 games, not 25.

## 4. ★ THE COUNTER-EVIDENCE, KEPT IN FRONT

Observational, on our own 570 measured passes:

```
passes that pressed the full discrete set   n=407   mean max level 1.66
passes that left >=1 declared action untried n=163  mean max level 1.69
```

**There is no benefit visible. If anything the direction is backwards.** This is
confounded (games differ in control count and difficulty) and is weak evidence in
either direction — but it is the only outcome-linked evidence we have, and it does
**not** support the claim that pressing everything helps. `untried` has FIREABILITY
on our rail; it does not have demonstrated VALUE on our rail.

## 5. WHAT THIS LICENSES, AND WHAT IT DOES NOT

**LICENSED.** The stagnation supervisor remains the right build — the stall itself is
proven (88% of clock after last clear; 45.2% immediate repeats; 0 `hard_noop_guard`
fires in 5,255 actions; 675/675 games dying on the clock) and it now has a concrete,
cheap, non-empty target at the moment it would fire.

**THE REDIRECT RULE THE EVIDENCE ACTUALLY SUPPORTS** is narrower than `untried`:
> On a stagnation window, if no declared arrow key has ever been pressed this game,
> press the unpressed arrows — one action each.

That sub-case is 53.1% of windows, costs ≤4 actions against a ~600-action budget,
and targets controls with a measured ~53% one-press live rate. It is strictly
cheaper and better-evidenced than the general `untried` rule.

**NOT LICENSED.** Adopting `untried` wholesale as the supervisor's primary rule.
Three independent discounts apply: our one-press live rate is ~53% not 91%;
ACTION7's live rate is 2/6; and full-coverage passes show no outcome advantage.
**FALSIFIER 5 still stands — no individual graft has board validation, and this
result does not supply any.** This is a fireability gate, not an effect estimate.

**NEXT INSTRUMENT (free, CPU):** drive deeper levels by real play to lift the
liveness measurement off the level-1 opening, and measure whether the *arrow-key*
sub-case in particular is live at depth. Until then the 53% is a level-1 number.
