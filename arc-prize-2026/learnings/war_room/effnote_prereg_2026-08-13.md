# EFFNOTE — QUANTIFIED PER-TURN EFFICIENCY NOTE — SEALED PRE-EVAL INTENT

**Status: SEALED 2026-08-13, BEFORE any eval kernel is pushed and BEFORE any arm
run exists.** Every number in §1 and §5 is an offline reconstruction over
**already-recorded, block-free** runs (`animation_v1`, `a22_v2_seed1`,
`a22_compaction_v1`), produced on CPU, read-only, $0, by
`duck_eval/warpack/effnote_replay.py`, and sealed here **together with the
intent so the two cannot be reordered after the fact**. None of them is an
outcome of this arm.

Parent documents:
`learnings/war_room/harness_diff_2026-08-13.md` §4 item **#1** (the ranked item,
its canaries and its kill rule), `learnings/war_room/efficiency_diagnosis_2026-08-12.md`
(the binding constraint, its **correction notice**, and §2.3's capped-score
fact), `learnings/war_room/p1_prereg_2026-08-12.md` + its addendum **A1** (house
prereg shape, and the three rules below that its post-mortem paid for),
`duck_eval/SCREEN_PROTOCOL.md` (binding; K3″, §4.6 power-honesty clause).

Artefacts sealed with this file:
`duck_eval/warpack/_kaggle_dataset/effnote_patch.py` (VERSION `v1`),
`duck_eval/warpack/effnote_replay.py`,
`duck_eval/warpack/effnote_smoke.py` (**99 checks, 0 failures**),
`notebooks/duckeffnote-eval/arc3-duck-effnote-eval.ipynb`,
`runs/effnote_replay/control_spread.json` (+ `_target100_sensitivity.json`,
`control_spread.txt`).

---

## 0. The single question this arm answers

**The model is never shown the scoring rule and never sees its own action
count. Does showing it both, every turn, change how it spends actions?**

Our stock prompt's *entire* efficiency treatment is one unquantified sentence
(`inference/agent/prompts.py:17`):

> `"- Optimize for as few in-game actions as possible while still being reliable.\n"`

Meanwhile the per-level term is `min(115, (baseline/actions)^2 * 100)`, further
capped at the human-baseline contribution — **quadratic in waste, with zero
credit for beating the human count** (efficiency_diagnosis §2.3). Our 17 cleared
levels ran at **2.11× the human action count**; the same 17 re-scored at the
baseline give **2.549 local ≈ 1.48–1.58 LB**.

Every other item in the harness diff chases **more levels**. This one tries to
make the levels we **already clear** worth more — i.e. it targets the **mean**,
which is what pays at private selection (harness_diff §3.2: our per-draw
distribution is statistically indistinguishable from the field's; we are behind
on **k**, not on capability).

**Evidence class, stated honestly and up front.** Present in exactly **one**
≥1.40 kernel (`caoyupeng/arc3-duck-v12-1d7d88`, Tara Labs #37 @ 1.46, via
`install(bm, flags={"efficiency": True, ...})` over `thtennant/taaf-kaggle-source-share-fork`).
The graft's author, thtennant, is at **1.28**. **This carries no efficacy
evidence whatsoever.** It ranks #1 on mechanism-to-diagnosis fit, not on the
public record. Nothing in this document may be read as though it did.

---

## 1. THE CONTROL SPREAD — computed BEFORE the arm exists

**Why this section is first.** P1's mechanism C looked like a **4.4× behavioural
win** inside its own arm (dead-reissue 33.9% → 7.8%, z = 10.9) and was
**regression to the mean**: the identical statistic on three block-free control
runs spans **5.3–23.1%**, the arm's 7.8% sat *inside* that spread and *above*
the best control, and on one control the within-run direction **reversed**. The
standing rule that came out of it, and the rule this arm is built around:

> **Compute the control-side statistic for every behavioural metric BEFORE
> reading the arm. A within-run before/after contrast is not evidence.**

`effnote_replay.py` reconstructs, on each control, exactly what the EFFNOTE note
**would have said** at every turn — driving the **shipped** pure functions
imported from `effnote_patch.py`, never a re-implementation — and then measures
what the agent actually did next. Arm and control are scored by the **same
code** on the **same definitions**.

**Definitions (binding for both arm and control).**
*TURN* = one analysis step carrying ≥1 action; the note is built from the frame
history as of the **last action of the previous step**, which is exactly when
`_build_user_prompt` runs. *STALL TURN* = a turn on which ≥1 detector fires.

| metric | animation_v1 | a22_v2_seed1 | a22_compaction_v1 | **CONTROL SPREAD** |
|---|---:|---:|---:|---|
| **D1** note rate (turns with a note) | 0.9651 | 0.9609 | 0.9672 | **0.9609 – 0.9672** |
| **D2** stall rate (any detector) | 0.0996 | 0.0913 | 0.0712 | **0.0712 – 0.0996** |
| **D3** over-target rate | 0.2921 | 0.2197 | 0.1968 | **0.1968 – 0.2921** |
| **D4** note chars, mean | 297.5 | 281.3 | 273.0 | **273 – 298** |
| **D4** note chars, **max** | 603 | 602 | 603 | **602 – 603** |
| **B1** post-stall **revisit** rate | **0.3986** | **0.5487** | **0.4751** | **0.3986 – 0.5487** |
| **B1c** non-stall revisit rate (same run) | 0.1106 | 0.1992 | 0.1914 | 0.1106 – 0.1992 |
| **B2** post-stall **no-op** rate | 0.0779 | 0.3454 | 0.2777 | **0.0779 – 0.3454** |
| **B2c** non-stall no-op rate (same run) | 0.0837 | 0.1717 | 0.1675 | 0.0837 – 0.1717 |
| **B3** over-target burn, all levels (actions) | 2301 | 1267 | 2060 | **1267 – 2301** |
| **B3** over-target burn, **cleared** levels | 307 | 32 | 64 | **32 – 307** |
| **B4** mean actions per **stall** turn | 7.275 | 3.945 | 5.180 | **3.95 – 7.28** |
| **B4c** mean actions per non-stall turn | 3.943 | 3.458 | 3.718 | 3.46 – 3.94 |
| **M0** median actions per cleared level | 24 | 39.5 | 49 | **24 – 49** |
| **M0′** median of per-game medians | 24.25 | 44.5 | 50 | 24.25 – 50 |
| `levels_completed` | 17 | 14 | 17 | **14 – 17** |
| detector fired on N distinct games — net-zero | 20 | 14 | 16 | **14 – 20** |
| — revisit | 5 | 3 | 8 | **3 – 8** |
| — **stagnation** | **1** | **2** | **1** | **1 – 2** |
| turns / actions replayed | 1205 / 5151 | 997 / 3492 | 1250 / 4777 | — |

Source: `runs/effnote_replay/control_spread.json`. The replay reproduces each
run's recorded action total exactly (5151 / 3492 / 4777) — smoke check **R2**.

### 1.1 Three things the control spread already settles, pre-seal

1. **The harness_diff canary K-E1 as drafted would have FAILED on the controls,
   arm or no arm.** It asks each of the three detectors to fire on **≥3 distinct
   games**. **Stagnation fires on 1–2 games per control run.** That is a property
   of the rail (our agent almost never hammers a genuinely inert control for 8
   straight actions), not of the arm. **K-E1 is therefore re-pre-registered
   here** as: net-zero ≥3 games, revisit ≥3 games, **stagnation ≥1 game**. This
   is disclosed **before** the data and the incentive is on the record: the
   change makes K-E1 *more* likely to pass. It is also, per K-E1's own charter,
   a **detector-sanity** canary and **not** an efficacy endpoint — it cannot
   rescue a dead arm. (Same class of defect as P1's K-P6 denominator, ruled in
   addendum A1; same remedy — rule it before the number exists.)
2. **The detectors select real waste.** On all three controls the post-stall
   revisit rate is **2.5–3.6× the same run's non-stall rate** (0.399 vs 0.111,
   0.549 vs 0.199, 0.475 vs 0.191). Whatever the agent does with the note, the
   note is fired at moments that genuinely are wasteful. Smoke **R8**.
3. **The nag risk is not realised.** harness_diff's kill rule (iii) fires if any
   detector fires on **>40%** of turns. The control stall rate is **7.1–10.0%**.
   Smoke **R4**.

---

## 2. THE ARM — one flag, report-only, two seams

`EFFNOTE=1` on the duck baseline + the **(f) continuation default**. Nothing
else: **no** warpack, **no** ledger graft, **no** sentinel, **no** compaction,
**no** animation, **no** P1.

**What is appended to the USER turn**, only when there is something to say:

| part | content |
|---|---|
| (a) | the scoring rule, **verbatim and quantitative**: `EFFICIENCY BUDGET - this level scores (human_baseline_actions/your_actions)^2 x 100, capped at 100: baseline=100, 2x over=25, 3x=11, 5x=4. Waste costs you quadratically.` |
| (b) | the **live** action count on this level (read from `game_run.actions_per_level`, the same array the scorer reads) against the clamped `heuristic_action_target()` **proxy** |
| (c) | the over-target ratio (`… you are 2.2x over the typical target.`) |
| (d) | the three **pure** stall detectors — `detect_net_zero_cycle` (shortest ≥6-action round-trip back to an exact prior same-level grid, with a divergence requirement so a static board is not double-reported), `detect_stagnation` (≥8 consecutive same-level actions leaving the grid byte-identical), `count_recent_revisits` (≥4 exact recurrences of the current grid) |
| (e) | the commit-don't-scan reminder, gated on over-target **or** any stall so it is not boilerplate |

**REPORT-ONLY, proved at runtime.** The module writes exactly two seams —
`_HarnessGameSession.play` (bind + canary; the vanilla body runs unchanged) and
`ToolAgent._build_user_prompt`. `_execute_action` and `step_env` are the **same
objects** after the graft as before it (smoke **I2g**, **L5**). No action is
ever blocked, declined, injected or reordered. Zero LLM calls, zero GPU, zero
new tools.

### 2.1 Four DELIBERATE divergences from the reference implementation

The reference is `EfficiencyToolAgent` in `thtennant/taaf-kaggle-source-share-fork`
(`taaf_grafts/agent_ext.py`). Ours is not a port. Each divergence is a
constraint learned the expensive way:

1. **PROXY-ONLY BASELINES — no baseline is ever read.** The reference *prefers*
   a real per-level baseline (`game.base_actions_per_level`, else an rglob over
   the shipped `metadata.json`) and only falls back to the heuristic. **We
   delete that path entirely.** (i) A per-game baseline table is game-specific
   and, per the 08-12 P1 finding, **factually wrong on a rerun** — the
   latent-state game set is run-dependent (cn04/sc25 in, re86 out), which
   proved a hardcoded list would have been both illegal *and* wrong.
   (ii) Real baselines exist offline and are **stripped on the hidden set**, so
   preferring them means measuring one mechanism and shipping another — the
   exact class of error that killed the animation arm. The clamped
   game-agnostic proxy is **the only target that ships**, so the eval and the
   hidden set see the identical mechanism. Enforced by smoke **L1/L2**.
2. **COST BOUNDED IN CHARACTERS, NEVER AS A TOKEN FRACTION.** The note is an
   **input**-token cost; the rail reports **generated** tokens only. That
   denominator mismatch is what fired K-A3 and killed the animation arm, and is
   what P1 prereg addendum A1 had to rule on. `EFFNOTE_MAX_CHARS = 700` is a
   hard static clamp that drops **whole lines**, never a mid-sentence cut
   (smoke **U22/U24/U25**). **No token metric of any kind exists in the module
   or the canary** — asserted by smoke **L6** and **I6b**.
3. **Monkeypatch, not a `ToolAgent` subclass.** The reference installs an
   `analyzer_factory`. Our house pattern patches `_build_user_prompt` directly
   (as P1 mechanism C did), so the arm composes with the (f) continuation
   default and cannot fight another graft over the factory slot.
4. **caoyupeng's duplicate-game replay gate is NOT ported in any form.** His
   `taaf.game_api.GameAPI(..., external_game_id=f"{first.env_name}-dup")`
   scouting gate exists only behind `if not TRUE_SUBMISSION:`; replaying an
   environment under a second game id inside a real rerun would be **scorecard
   manipulation**. Out of scope, asserted absent by smoke **L4**.

### 2.2 The proxy target, stated plainly

`heuristic_action_target(valid_action_count, board_cells)`
` = clamp(50 + 5·V + min(30, √cells), 40, 100)`. Game-agnostic by construction:
it reads two observables and no identity. On this rail (64×64 boards) the clamp
binds at **V ≥ 4 ⇒ target = 100**; the observed control range is **85–100**
because early turns expose fewer valid actions.

**It is generous on purpose** and it is **not** a baseline: the human baselines
on our six penalty games are 7–39, so a level at the proxy target is already
2.5–14× the human count. The note therefore under-states the pressure rather
than over-stating it, and cannot nag a genuinely short level. **Sensitivity
sealed:** pinning the target to 100 everywhere moves D3 from 0.197–0.292 to
0.189–0.282 and B3-total from 1267–2301 to 1238–2245, and leaves **every other
metric bit-identical** (`control_spread_target100_sensitivity.json`). The proxy
choice is not load-bearing for any endpoint.

### 2.3 Shipped defaults

`EFFNOTE_MAX_CHARS=700` · `EFFNOTE_MIN_ROUNDTRIP=6` · `EFFNOTE_STAGNATION_MIN=8`
· `EFFNOTE_REVISIT_MIN=4` · `EFFNOTE_WINDOW=240` · `EFFNOTE_REVISIT_WINDOW=120`
· `EFFNOTE_TARGET_*` as §2.2 · `EFFNOTE_MAX_EVENTS=400` (log-volume bound only).
Kill switch `EFFNOTE_DISABLE=1`. Arm flag `EFFNOTE=1` — without it `apply()`
returns False and **nothing** is patched (smoke **K1–K4**, both flag-off and
kill-switch paths verified in a subprocess).

---

## 3. ENDPOINTS

**Power statement, sealed first (SCREEN_PROTOCOL §4.6).** The family
`duck-harness-kaggle-continuation-v1` is still **m = 2**. **Δ`levels_completed`
is NOT screenable** and may not be read as evidence in either direction. Our
per-draw sd is 0.1513 on n = 29; a single arm draw cannot resolve anything of
the size this mechanism could plausibly produce. **This arm buys a mechanism
reading, not a score reading.**

### M0 — PRIMARY, part 1: mechanism DELIVERY

The note must actually reach the model. From the build log:

* `note_rate` ≥ **0.80** on turns where a stall fires or the level is over
  target (harness_diff K-E0). Control reconstruction says the *overall* rate
  will be ≈0.96; the 0.80 floor is the binding form.
* `chars_max` ≤ **700** (K-E3) and `errors` = **0**.
* the canary line and ≥1 `EFFNOTE v=1 kind=note` line on ≥20 of 25 games.

**Delivery is necessary and NOT sufficient.** P1's mechanism C delivered on
96.3% of turns and was dead. Delivery failing kills the arm; delivery passing
licenses nothing on its own. This sentence is sealed so it cannot be forgotten
when the number arrives.

### M0 — PRIMARY, part 2: the BEHAVIOURAL test vs the CONTROL SPREAD

**B1 — post-stall revisit rate.** Of the actions the agent issues on a turn
whose note fired ≥1 stall detector, the fraction that land on a board state
already visited on that level.

* **CONTROL SPREAD: 0.3986 – 0.5487** (min = `animation_v1`).
* **PASS** requires the arm to fall **strictly below 0.3986** — below the
  *minimum* of the spread, not below its own first half, not below the mean.
* The arm's within-run first-half→second-half contrast **may not be cited**.
  That statistic is what fooled mechanism C.

**B3 — over-target burn on cleared levels.** Actions spent on a cleared level
*after* it first crossed the proxy target.

* **CONTROL SPREAD: 32 – 307 actions** (total across 25 games).
* Read as **supporting** only. The spread is a factor of 9.6 wide on three
  draws, so it can support a B1 pass but **cannot** carry a verdict alone, and
  a B3 "win" with B1 inside the spread is **not** a pass. Sealed as secondary
  for exactly that reason.

### M1 — SECONDARY, DESCRIPTIVE ONLY: `levels_completed`

Reported against the control range **14 – 17**. **NOT screenable (m = 2).** It
may be used **only** in the non-harm direction of kill rule (i) below, never as
evidence of gain.

### M2 — SECONDARY, DESCRIPTIVE ONLY: RHAE and M0-median

Local-25 RHAE via `duck_eval/warpack/*_score.py` / `scripts/phase1_gate.py`, and
the median actions per cleared level against the control range **24 – 49**. One
draw against an m = 2 family. **Attributing either to EFFNOTE would be exactly
the error this prereg exists to prevent.**

---

## 4. CANARIES (read from the build log; all pre-registered)

| id | check | grep |
|---|---|---|
| **K-E0** | graft installed, 2 seams, report-only banner | `effnote v1: ACTIVE` |
| **K-E0b** | note delivered on ≥80% of stall-or-over-target turns, on ≥20 games | `EFFNOTE CANARY` → `note_rate=` |
| **K-E1** | net-zero ≥3 games, revisit ≥3 games, **stagnation ≥1 game** (re-preregistered — §1.1 item 1) | `EFFNOTE CANARY` → `nz=…/Ng stag=…/Ng rev=…/Ng` |
| **K-E1′** | no detector fires on **>40%** of turns (nagging ⇒ ignored) | `stall_rate=` |
| **K-E2** | `levels_completed` ≥ **14** (the minimum of the three block-free controls) | `[finished] … level=` |
| **K-E3** | `chars_max` ≤ **700**, and **no token metric anywhere** | `chars_max=` / `bound=700` |
| **K-E4** | `errors=0`, no `PATCH FAILED`, no traceback | `errors=` / `PATCH FAILED` |
| **K-E5** | the graft is alone: no warpack / ledger / sentinel / compaction / animation / p1 banner in the log | `graft applied` |

---

## 5. KILL RULES (sealed)

1. **Non-harm.** `levels_completed` < **14** (the control minimum) ⇒ **KILL
   outright.** Any trade of levels for efficiency kills this arm; the whole
   thesis is that it makes *already-cleared* levels cheaper.
2. **The behavioural test.** **B1 ≥ 0.3986** (i.e. the arm does not beat the
   *minimum* of the control spread) ⇒ **NO-PROMOTE.** No re-reading against the
   control mean, against the arm's own first half, or against a subset chosen
   after seeing the data.
3. **Nagging.** Any detector firing on **>40%** of turns ⇒ **KILL** (the note is
   noise and the agent will learn to skip it).
4. **Delivery.** `note_rate` < 0.80 on stall-or-over-target turns, or
   `errors > 0`, or any `PATCH FAILED` ⇒ **INFRA DEATH**, re-run or abandon; the
   behavioural numbers may not be read at all.
5. **Cost.** `chars_max` > 700 ⇒ the static bound leaked; **KILL** and fix the
   clamp. (This is the character bound. **A token-fraction reading is
   forbidden**: the rail's denominator is generated tokens and the note is an
   input cost. K-A3 / addendum A1.)

---

## 6. WHAT A PASS WOULD AND WOULD NOT LICENSE

**Would:** a second seed of the same kernel to see whether B1 holds outside the
control spread twice; and only then a promotion discussion. Nothing more.

**Would NOT:** (i) any claim about LB score — m = 2, sd 0.1513, one draw;
(ii) any claim that "prompting the model about efficiency works" in general —
`feedback_prompt_is_noise` stands and this arm is deliberately **not** a prompt
A/B: it adds *state the model does not have* (its own action count) rather than
rewording instructions it already has; (iii) any move toward a real-baseline
read, a per-game table, or the replay gate — those stay permanently out of
scope; (iv) reading the efficiency lane's ceiling as anything other than
**~2.19 local ≈ 1.26–1.36 LB**, which the 08-12 diagnosis showed is **short of
the 1.48–1.58 gold line**. This arm is a mean-mover, not a gold path.

**Standing counter-evidence, sealed so it is not rediscovered as a surprise:**
efficiency_diagnosis §3 found **vc33** cleared at 3.00× the human count with
**zero** duplicates, **zero** no-ops, **zero** revisits and a provably minimal
path — 100% capability, untouchable by any note. §2.1 puts **40%** of the gap in
that class. And RedundancyBench (arXiv:2605.29893, sweep 08-12) measured the
best LLM-based redundancy detector at **24.88%**, some below random — the
standing rebuttal to "let the model notice it is repeating." **This arm does
not ask the model to notice; the runner notices and tells it.** That is the
whole distinction, and it is the reason the arm is worth one slot rather than
zero.

---

## 7. GATES RUN PRE-SEAL

* **Smoke:** `duck_eval/warpack/effnote_smoke.py` → **99 passed / 0 failed** —
  13 structural, 9 legality/report-only over the shipped source, 30 unit,
  10 control-spread replay, and integration against the **real offline
  arcengine** (real `GameAPI` on `ft09`, the real patched `play()` body, a real
  64×64 board through the real patched `_build_user_prompt`), plus the
  flag-off and kill-switch subprocess gates.
* **Structural diff gate:** `notebooks/duckeffnote-eval/arc3-duck-effnote-eval.ipynb`
  vs the war-eval baseline — **17 cells vs 17, exactly cells 2 / 12 / 14
  differ**, `kernel-metadata` delta = `{id, title, code_file}` only; the
  wheelhouse + taaf bundle + 27B snapshot triple, the docker sha, the machine
  shape and the GPU/internet flags are **byte-identical**
  (`feedback_kaggle_env_match`, 5× confirmed). Smoke **S9–S13**.
* **preflight.py was NOT used as the gate, and that is a known runbook debt,
  not a skipped check.** Filed 2026-08-12: preflight's K2/K4/K5/K6/K8 test the
  `arc3-baseline` agent-swarm notebook shape (`agents/__init__.py`, `.env`,
  `main.py --agent myagent`) and return **BLOCK on every member of the
  duck-harness eval family**, including `arc3-duck-animation-eval`, which built
  COMPLETE and produced our primary trace. `--mode trusted-fork` is likewise
  inapplicable (the war-eval baseline itself differs from the Cottaar upstream
  in 7 code cells, so every member of this family is a graft). The applicable
  gate is the structural diff above. **Preflight still needs a duck-harness
  family profile.**

---

### Provenance

* Control replay: `duck_eval/warpack/effnote_replay.py` over
  `runs/kernel_pulls/animation_v1/`, `runs/a22_v2_seed1/`,
  `runs/a22_compaction_v1/` (`benchmark.json` + `artifacts/*_p0_events.jsonl`).
  Drives the **shipped** `effnote_patch` pure functions. Action totals
  reproduced exactly: 5151 / 3492 / 4777.
* Reference read: `runs/harness_diff_0813/ds/thtennant_taaf-kaggle-source-share-fork/
  src/taaf-grafts/taaf_grafts/agent_ext.py`.
* Stock prompt read: `duck_eval/taaf_bundle/src/ARC3-Inference/inference/agent/prompts.py:17`.
* All analysis local CPU, read-only. **No submissions, no cloud spend.**
