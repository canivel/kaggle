# PREREG — ARM 3: Q38×GRAFT COMPOUND (`arc3-q38-graft-eval` v1), sealed 2026-08-21 BEFORE build
**Slot:** 08-21 slot 1 (pre-authorized 08-20; deep-test amendment executed this morning). Operator: graft-lane agent.

## 0. VEHICLE DECISION, RECORDED
A byte-faithful `thtennant/arc3-duck-v21` rebase was REJECTED after inspection (sha `71f0b1a8e1e5ab7b`, diff cells [6,8,12,14] vs frozen fork): (a) it hardcodes the OLD competition mount path — dies at t≈6s on the migrated build rail exactly as graft-floor v1–v3 did; (b) its offline branch is a **4-game commit run with a duplicated game deliberately firing the transfer family store** ("dup-game commit gate", cell 14) — not a 25-game eval, and it exercises machinery this lane excludes. **Vehicle instead: OUR layout-tolerant graft-floor v4 lineage + v21's deltas lifted VERBATIM from the pulled artifact:** cell-6 engine block (Q38 Kaggle-Model pin, mount probing across layout shapes, required-file + 18-shard validation, offline env pins), cell-8 setup patch (serve `Qwen/Qwen3.8-27B-FP8`), cell-12 flags verbatim `{efficiency, retry_guard, shortcircuit, goalkeep, hudmask, clickmap}` (+ our `expected_version=1`). v21's dup-gate/safety-pack is EXCLUDED. Expected diff-set vs frozen fork: **[2,4,6,8,12,14]**.

## 1. DEEP-TESTS DONE PRE-SEAL (per ruling)
- Fresh bundle re-audit (3rd republish in 4 days): +`searchmap.py` (18th module), composite/goalkeep diffs read line-by-line — ALL flag-gated on `searchmap` (nested under clickmap, "Absent → byte-identical" author invariant) — **inert for the v21 flag set**; checker shas updated checker-canonical (`dde323ab…`/`8a80c180…`, 91/18), BUNDLE CHECK OK.
- **Init-path smoke under REAL wheels: all 18 graft modules import clean against arc_agi 0.9.8 + arcengine 0.9.3** (competition wheels installed locally); `install(bm, flags, *, expected_version)` signature confirmed; clickmap wiring + `[clickmap] armed` banner confirmed in the fresh bundle.

## 2. CERTIFICATION (runtime, before any number; any failure ⇒ INFRA DEATH)
1. Banner `TAAF_GRAFTS FEATURES=` with API_VERSION=1; **all SIX flags incl. `clickmap`** in FEATURES.
2. `[goalkeep] armed`, `[hudmask] armed`, **`[clickmap] armed`** present ← REQUIRED-marker flip per pre-auth (the q38-low landmine class, handled: this arm's scorer is its own per-arm table; the closed v19 lane's scorer is untouched).
3. `banking`/`transfer`/**`searchmap`** ABSENT from FEATURES; no `[banking]`/`[transfer]`/`[searchmap] armed` lines.
4. No stock-fallback signatures; served model banner = `Qwen/Qwen3.8-27B-FP8`; `reasoning_effort` ABSENT from log (xhigh default); benchmark n=25.

## 3. SEALED BANDS — comparator = YESTERDAY'S FIELD-FLOOR SEED (lc 28, mean_score 6.173, actions 1639; n=1)
Same-config replicate diff-SD **5.011 lc** (validated pair, exp 13/15). n=1-vs-n=1 ⇒ these are **1σ screen bands, type-I ≈16%/side under normality — declared, not decisive**:
| verdict | rule |
|---|---|
| INFRA DEATH | any §2 failure · no benchmark.json · n≠25 |
| **HARM** | lc_total ≤ **23** (Δ ≤ −5) |
| **NULL** | 24…32 |
| **SIGNAL** | lc_total ≥ **33** (Δ ≥ +5) |
**SECONDARY (recorded, non-inferential):** mean_score vs 6.173 — mechanism prediction: grafts on xhigh should RAISE score at ≈level parity (shortcircuit trims the denominator; quadratic pays more at higher lc). total_actions vs 1639. games_won (predict 0).
**Pre-registered expectation:** NULL-to-SIGNAL on levels (P(SIGNAL)≈25%), score ≥ 6.2 (P≈55%). **Compound-on-compound declared: any delta attributes to the graft BUNDLE, not to any single flag.**

## 4. TONIGHT'S HEAD (coordinator ruling, verbatim logic)
Head = highest expected draw, deep-tested: **this arm heads IFF certified AND lc_total ≥ 28** (the field-floor-equivalent read). Otherwise **re-queue `arc3-q38-field-eval` v1** (public max-of-draws; 1.59 is one draw of a config whose reference shows 2.23) with trusted-fork tag + upstream; filler drops to entry 3.

## 5. ARTIFACTS
Builder `duck_eval/graft/build_q38graft_eval.py` (lifts v21 cells from the sha-pinned pull); kernel `canivel/arc3-q38-graft-eval` (fresh slug); scorer `duck_eval/graft/q38graft_score.py` (sealed pre-push, selftested, per-arm marker table); results → `runs/kernel_pulls/q38graft_v1/`. Standard gates: byte-audit, idempotence, push-target integrity, pull-back verify (model_sources silent-drop check), post-push preflight `--expect-diff-cells 2,4,6,8,12,14`.
