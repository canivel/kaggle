# EXEC-WM observation layer — VERDICT, and a correction to the 08-27 addendum
**Mac iterate session, 2026-08-27. CPU-only. Slots spent: 0. GPU: $0. No kernel pushed.**

Instrument: `duck_eval/execwm/ewm_replay_obs.py` (new). Artifact: `runs/execwm_obs_replay.json`.
Source data: `runs/kernel_pulls/execwm_v1/artifacts/` — **re-pulled today**, complete (both large
logs present: 307KB kernel, 370KB vLLM; only 0-byte file is a legitimately empty `__init__.py`).

---

## 1. THE RE-SCOPE'S PREMISE IS FALSE: NOTHING IS STARVED

The 08-27 Windows handoff ordered: *"replay the retained `*_events.jsonl` and find why transition
extraction returns EMPTY in 9 of 18 games. Nothing else about that arm matters until n > 0."*
Done. **Transition extraction does not return empty, and the observation layer is not starved.**

**Test 1 — do the frames change?** In the zero-candidate games the board changes on **85–100%** of
move actions, *equal to or above* the games that worked. `lf52` and `tr87` change on **100%** of
move actions and mined **zero** rules; `g50t` changed on only **74.2%** and mined four.

**Test 2 — does change magnitude separate them?** No. Pooled median changed-cells: failing **46**,
working **54**, mixed **32**. `tr87` changes a sprite-sized **13** cells median and mines nothing;
`ka59` changes **19** and mines three rules.

**Test 3 — replay every transition through the SHIPPED classifier** (`detect_translation`, not a
reimplementation — a reimplementation would only measure a second opinion about the bug). 2394
move-action transitions across 25 games:

| class | n | share |
|---|---:|---:|
| `move` — a single translation explains EVERY interior diff cell | **1243** | **51.9%** |
| `residual` — rejected: some diff cell is neither departure nor arrival | **641** | **26.8%** |
| `noop` | 366 | 15.3% |
| `no-candidate` — no (dr,dc) within ±8 explained anything | 139 | 5.8% |
| `too-small` — best candidate moved < `MIN_SPRITE_CELLS` | 5 | 0.2% |

**Over half of all transitions are already clean, textbook single-sprite translations.** The feed is
rich. The loss is entirely inside the extractor. "Data starvation" is refuted; so is the probe-budget
theory (already closed arithmetically). This is the **third** successive wrong diagnosis of the same
number — and each was corrected only by looking at the artifact rather than at the previous summary.

## 2. WHERE THE DATA GOES — TWO GATES, BOTH TOO STRICT

**GATE B — `detect_translation`: `if diffset - departures - arrivals: continue`.** A move must
explain *every* interior diff cell. One co-occurring change — an animated tile, a second object, an
enemy, a counter inside the interior — and the entire transition is discarded as `unexplained`.
**This is the single largest loss channel: 26.8% of all transitions, 641 of them.** It is
concentrated exactly where the arm failed: `cn04` 80/95 residual, `re86` 119/167, `tr87` 103/129,
`ar25` 65/83, `cd82` 35/80.

**GATE A — `mine()`: `deltas = {...}; if len(deltas) == 1`.** A rule is admitted only if an action
produced **exactly one** distinct delta across *all* its instances. No mode, no majority. One
blocked move against a wall (delta `(0,0)` or a short move) discards the rule outright. The same
gate also fences **blocker mining** (line ~404), so the penalty is paid twice: no rule *and* no
knowledge of what stopped it.

**Measured cost of Gate A.** Replacing it with the **union** (unanimous **OR** ≥60% majority with
n≥3 — a union, not a replacement, because majority alone would *drop* a unanimous rule seen once or
twice, as `re86` does):

| game | rules under shipped Gate A | under union | note |
|---|---:|---:|---|
| `m0r0` | 0 | **4** | crosses `MIN_VERIFIED_MOVES=2` → **plannable, was not** |
| `tu93` | 0 | **3** | crosses `MIN_VERIFIED_MOVES=2` → **plannable, was not** |
| `sc25` | 2 | 4 | |
| `wa30` | 1 | 3 | |
| `ka59` | 3 | 4 | |
| `bp35` | 0 | 1 | |
| all others | — | unchanged | **no game loses a rule** |

**+12 rules across 6 games, and two games go from "no model at all" to plannable.**

**Why relaxing the mining gate is safe, and not just moving the failure downstream.** Mining emits
*candidates*; `verify()` is an independent prequential gate at `VERIFY_PRECISION = 0.90`,
`VERIFY_MIN_N = 3`. A majority-mined rule that is actually wrong is rejected there. The shipped
design already treats LLM hints this way — *"as CANDIDATES that PHASE V must pass before use"* —
so Gate A is holding mined rules to a **stricter** standard than the arm holds an LLM's guess.

## 3. THIS IS PRE-AUTHORIZED, AND IT IS NOT A TWEAK

`bench_meta_execwm_v1`'s sealed `prestated_v2_constraint`: *"a v2 may NOT be a probe-budget tweak;
it must widen the rule class / object model (click-addressable objects, non-constant deltas,
multi-object dynamics)."* Gate B is precisely *multi-object dynamics*; Gate A is precisely
*non-constant deltas*. **Both repairs are the pre-registered v2 class, named in advance, and the
measurement now says which to build first.** The original bench_meta lesson was right all along —
*"the binding constraint is UPSTREAM of the planner and upstream of the verifier — it is object
identification and rule-class coverage"* — and today's "data starvation" re-diagnosis had moved
away from it.

**Ranking, by measured yield:** Gate B (641 transitions, 26.8%) ≫ Gate A (+12 rules, 2 games made
plannable). Gate A is far cheaper — a bounded change to one `if` — and is the honest first build.

## 4. WHAT WAS NOT DONE
No v2 was built and no kernel was pushed: that is a slot decision, and the `q38-field`/`graft-floor`
lane locks in `runs/lane_locks.json` are stale (dated 08-20/08-21) and need clearing by their owner
before any push. The union gate is measured on **retained transitions only** — it predicts what
mining *would admit*, not what `verify()` would then confirm at 0.90, and not a score. Re-seeding
the arm remains gated on an actual build. The `residual` bucket is counted but not sub-classified
(animation vs. second object vs. enemy) — that is the next measurement, and it is free.
