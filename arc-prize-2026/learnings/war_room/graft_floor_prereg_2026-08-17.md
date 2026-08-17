# PREREG — GRAFT FLOOR ARM (`arc3-graft-floor-eval` v1), sealed 2026-08-17 BEFORE the push
**Slot:** 2026-08-18 slot 1 (08-17 is 2/2 spent; nothing is pushed today).
**Lane status:** NEW lane. Adopts the field-standard PUBLIC graft implementation as-is at its published conservative setting. **Does NOT reopen our own warpack/A9 banking line, which stays parked.**
**Binding evidence:** `duck_eval/graft/bundle_audit_2026-08-17.md` (this session, read it first) and `learnings/war_room/conversion_trace_2026-08-17.md`.
**Sealed means sealed:** reading the data and then adjusting a number in §4 voids the screen. `feedback_audit_the_instrument`.

---

## 1. WHY THIS ARM IS NOT THE ONE THAT WAS AUTHORIZED

The 08-17 coordinator ruling authorized `banking + transfer + shortcircuit` with a SCORE-primary read. **The audit falsified that arm's precondition before it was built:**

- `banking` gates on `run.state == "won"` (`banking_solver.py:180`). **Across 23 pulled eval artifacts / 470 game-runs — the entire recorded campaign — runs reaching `won` = 0.** Best levels-completed on any single game, ever = 4 of 6–10. **[V]**
- `transfer` needs clone siblings. Our eval rail is `n_passes=1`, 25 games, **25 unique `game_id`s**. Its own docstring: a non-clone set makes the stack *"a measured no-op"*. **[V]/[V-doc]**

⇒ The authorized arm reduces by construction to `shortcircuit` alone and would return a REFUTE containing **no information about the mechanism** — a precise repeat of the A9/warpack error (*"our gate measured LEVELS on an offline bench where banking's conditions never fired"*). **An arm whose treatment cannot fire is not screenable.**

**This arm therefore substitutes the reachable public floor** — thtennant's published **v19** flag set — and holds `banking`/`transfer` OFF *and asserted-absent*. The coordinator may overrule and demand the literal authorized arm; §7 states exactly what that would and would not measure.

**Strategic consequence, recorded here because it outranks the arm:** banking multiplies the score of cards you already win, and we win none. For us the field's recipe is **gated behind the capability we lack**, not a shortcut past it. Hence this arm's primary endpoint is **levels (capability)**, not score.

## 2. THE ONE VARIABLE

Frozen fork `canivel/arc3-duck-repro` (upstream `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`), with the source-bundle dataset **replaced** by `thtennant/taaf-kaggle-source-share-fork` and cell 12 calling `install()`. Verified **byte-identical harness**: the fork is stock + 16 additive files under `src/taaf-grafts/`, **0 stock files modified, 0 files missing** (recursive sha256, audit §1). Engine stays **Qwen3.6** (`vrfai/Qwen3.6-27B-FP8`, `max_model_len 65536`, analyzer ctx 32768) — the fork's `setup_commands.json` is in the identical-73. **No engine confound; orthogonal to the sealed Q38 lane.**

Flags (verbatim, = thtennant v19):
```python
install(bm, flags={"efficiency": True, "retry_guard": True, "shortcircuit": True,
                   "goalkeep": True, "hudmask": True}, expected_version=1)
```
Mechanisms, and which endpoint each is expected to move:
- `goalkeep` (+`hudmask`, which is nested under it) → **LEVELS.** Stops the carried world model being wiped on game-over/level-change; authors measured stock carrying a non-empty model on **33 of 481 turns** **[V-doc]**. This is *our own* documented root cause ("the agent FORGOT"), already fixed and measured by someone else.
- `shortcircuit` → **SCORE.** Trims no-op overshoot from repeated-action batches; every no-op otherwise increments the scored action counter, and score is `min(115,(baseline/actions)²·100)` — quadratic (`taaf/game.py:403`, verified).
- `efficiency`, `retry_guard` → the proven public floor riders (report-only analyzer note + pass-through chain layer).

Touched cells: **[6, 12] + `kernel-metadata.json`**. (Cell 6's `DATASET_SOURCES` literal must change too: index 0 is the bundle, and `setup_commands.json` resolves wheelhouse/model by exact ref string.)

## 3. BASELINE (P1/P2/P3 all satisfied)

Family **`duck-harness-kaggle`**, **m = 3**: `gate_eval_v1` lc 18 · `gate_eval_v2` lc 19 · `duckgate_v1post` lc 21.
- **P1 same-config legality:** baseline is stock `NO warpack/ledger-graft/sentinel`; the arm's single change is the graft install. The warpack band is *not* used (it is the illegal control per R17) — cited nowhere in this gate. ✔
- **P2:** m = 3 ≥ 3. ✔
- **P3:** σ̂ = **0.141740**, df = 6, standing pooled build-rail estimate (SCREEN_PROTOCOL §P3). ✔

Derived: baseline per-game lc = 58/(3·25) = **0.773333**. C(3) = **2.02**. SE(Δ) = σ̂·√(1+1/3) = **0.163667**.
**C(3)·σ̂ = 2.02 × 0.141740 = 0.2863148**, so the lines are **∓0.286315** (6 dp). *Note of record: the Q38 prereg sealed this quantity as "0.286320", which is not equal to the product it claims to be. The discrepancy is 5×10⁻⁶ lc/game and moves no boundary (the lines land at lc_total 12.175 / 26.491 either way), but `graft_score.py` now DERIVES the constant instead of transcribing it, so the class of defect cannot recur.*
Baseline `mean_score` spread = **1.427 / 1.939 / 3.420** (mean 2.262, sd 1.033, n = 3).

## 4. SEALED VERDICTS — evaluated in this order

**PRIMARY = mean Δlc over the 25 games** (SCREEN_PROTOCOL §0; levels is the decision statistic). The three-state-minimum rule applies: INFRA DEATH is mandatory and is never a REFUTE.

| verdict | rule | in lc_total terms (25 games) |
|---|---|---|
| **INFRA DEATH** (not decisive) | any §5 assertion fails · no `benchmark.json` · n_games ≠ 25 · window drift > 5% · served engine unconfirmable | — |
| **HARM** (decisive) | mean Δlc ≤ **−0.286315** (K3″ FAIL at m=3, σ̂=0.141740) | lc_total ≤ **12** |
| **SIGNAL** (decisive for the arm's own claim) | mean Δlc ≥ **+0.286315** | lc_total ≥ **27** |
| **NULL** (decisive: K3″ PASS, no signal) | −0.286315 < mean Δlc < +0.286315 | lc_total **13 … 26** |

- The HARM line is canonical K3″ verbatim (§2 of the protocol, measured type-I **4.4%** at m=3).
- The SIGNAL line is the **mirror** of the same null10-measured 5th-percentile envelope. Type-I ≈ 4.4% and power ≈ 56.7% against a true +0.20/game **by mirror symmetry of the measured table** — this symmetry is **[INF]**, not separately measured, and is declared as an assumption rather than asserted as fact.
- **Honesty about the bar:** lc_total ≥ 27 is **+28% over the highest lc ever recorded on this rail** (22, `war_eval_v1`). A real but modest capability gain will land NULL. NULL therefore means "not a big effect", **not** "no effect", and must be written that way.
- **STRUCK legs, not used:** `worst-game Δlc ≥ −1.0` (measured type-I 50%) and `#(Δlc ≤ −2) ≤ 2` (advisory only). `actions_per_level_completed` may not be a co-primary.

**SECONDARY = `mean_score`, explicitly NON-INFERENTIAL and carries no verdict** (sealed reasoning inherited from the Q38 scorer: with baseline sd 1.033 on n=3 a score test has ~60% power). Reported as: `mean_score` vs the spread 1.427/1.939/3.420, plus total actions and actions-per-level as **descriptive** evidence of whether `shortcircuit` actually shrank the denominator. **A score movement may not be converted into a verdict after the fact.**

**PRE-REGISTERED EXPECTATION (stated before the data, so it can be wrong):** most likely **NULL on levels with a modest positive Δ**, and a *descriptive* drop in total actions from `shortcircuit`. `goalkeep` is the only flag with a plausible route to SIGNAL. **P(SIGNAL) ≈ 20%** — this is a judgement, not a calculation.

## 5. MANDATORY GATE ASSERTIONS (the Q38-low lesson)

The Q38 low arm died because the gate asserted the *previous* arm's signature and read the new arm's correct behaviour as failure. **A gate's logic must be invariant across arms; its expected values are a function of the arm.** So: assert this arm's POSITIVE markers, and fail on the wrong arm's markers.

1. banner `TAAF_GRAFTS FEATURES=` present, with `API_VERSION=1`
2. `[goalkeep] armed` **and** `[hudmask] armed` present
3. `efficiency`, `retry_guard`, `shortcircuit` all present in the FEATURES dict
4. **`[banking] armed` ABSENT; `banking` and `transfer` ABSENT from FEATURES** ← the arm is defined by their exclusion
5. `[taaf_grafts] install failed -> stock` ABSENT and `cell-12 graft failed` ABSENT (a silent stock fallback must never be scored as a REFUTE)
6. attached bundle re-diffs to audited manifest sha **`df447f61caa181cca68049e28b139e02`** (grafts subtree `7705481551494b141d6a33ffec1d7a20`); mismatch ⇒ re-audit before reading
7. served model banner = `vrfai/Qwen3.6-27B-FP8`

Any of 1–7 failing ⇒ **INFRA DEATH**, never HARM/NULL/SIGNAL.

**Why 6 is not optional:** Kaggle attaches the LATEST dataset version and kernel metadata cannot pin one. The fork was republished **2026-08-17 00:26** and is actively maintained. `expected_version=1` makes an API bump fail closed; assertion 6 catches a same-API content change.

## 6. RISKS ACCEPTED
- Unknown flag names are **silently ignored** (no validation in `install()`): a typo = a silent stock run that looks like a clean arm. Mitigated solely by assertions 1–5.
- `goalkeep`/`hudmask` are module-global monkey-patches that change per-turn prompt content — a genuine behaviour change that can cut either way, and it adds prompt tokens against the documented 31,744-token ceiling. Window-drift gate watches this.
- `shortcircuit`'s "provably monotonic non-decreasing score" is **[V-doc]**, their claim, **not independently verified by us [UNK]**.
- `install()` never raises; every failure path degrades to stock. Safety and silence are the same mechanism here — hence the assertions.

## 7. WHAT THIS ARM DOES *NOT* SETTLE (so no one over-reads it)
- **Banking/transfer remain UNTESTED, in either direction.** Not refuted — **unreachable on this rail.** Reopening them requires first answering: *does the agent ever reach `state=="won"` on the COMPETITION 110-run rerun?* That is **[UNK]** (we retain no rerun logs) and is answerable from a rerun log, not from an eval build.
- Whether the 110 scored runs are really 25 games cloned round-robin: **[UNK]**, asserted by the bundle only.
- The field's 2.5+ attribution stays **[INF ~85%]**; nothing here is a per-team attribution.
- This arm cannot separate `goalkeep` from `shortcircuit` — it is a floor-adoption test, not a factorial. If it SIGNALs, the follow-up is the factorial, one flag per slot.

## 8. ARTIFACTS
Notebook `notebooks/graft-floor-eval/`, kernel `canivel/arc3-graft-floor-eval`, scorer `duck_eval/graft/graft_score.py` (sealed, selftested pre-push), push script `duck_eval/graft/graft_push.sh` (date-guarded to 2026-08-18, with the step-1c push-target integrity assert shipped 08-17). Results land at `runs/kernel_pulls/graft_floor_v1/`.
