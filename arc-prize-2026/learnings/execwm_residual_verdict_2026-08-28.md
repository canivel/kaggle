# EXEC-WM GATE B — the `residual` bucket sub-classified, and GATE B's ranking re-priced
**Mac iterate session, 2026-08-28. CPU-only. Slots: 0 (this measurement is free). GPU $0.**

Instrument: `duck_eval/execwm/ewm_residual_class.py` (new).
Artifact: `runs/execwm_residual_class.json`. Source: the retained
`runs/kernel_pulls/execwm_v1/artifacts/` events, same pull as exp 61.

This is the measurement exp 61 named as next and called free: *"the `residual`
bucket is counted but not sub-classified (animation vs. second object vs. enemy)."*

## 1. CROSS-CHECK FIRST

The classifier reproduces **641** residual transitions — the exact number exp 61
measured independently yesterday. It also reuses the SHIPPED primitives
(`interior_diff`, `detect_translation`, `_why_unexplained`, `MAX_DELTA`,
`MIN_SPRITE_CELLS`) and converges `HudMask` over all pairs exactly as
`replay_game` and the live arm do. Two earlier drafts of this run returned 0 and
were wrong for instructive reasons, both fixed: `detect_translation` never
returns the string `residual` (only `noop`/`move`/`unexplained` — the label
belongs to the tracer), and a **fresh, unconverged HudMask is a different
instrument** that silently reclassifies every transition.

## 2. THE RESULT

| sub-class | n | share | median leftover | what it needs |
|---|---:|---:|---:|---|
| **overlap** | **354** | **55.2%** | **35 cells** | occlusion / partial redraw — **neither proposed repair** |
| co-mover | 126 | 19.7% | 4 cells | MULTI-DELTA (two objects, two deltas) |
| bystander | 117 | 18.3% | 2 cells | BOUNDED-LEFTOVER tolerance (disjoint, small) |
| diffuse | 44 | 6.9% | 12 cells | animation / global repaint — not recoverable |

**RECOVERABLE: 243 of 641 = 37.9%.**

**Sensitivity.** The only free parameter is the bystander cell bound. Swept 2→20
(10×): `co-mover` is **126 at every threshold** and `overlap` is **354 at every
threshold** — both are threshold-independent by construction. Recoverable moves
only between **197 and 246 (30.7%–38.4%)**. The headline does not depend on the knob.

## 3. WHAT THIS DOES TO THE GATE A vs GATE B RANKING

Exp 61 ranked **Gate B (641 transitions, 26.8%) ≫ Gate A (+12 rules, 2 games made
plannable)**, on the raw size of the loss channel. **That ranking does not survive
its own follow-up measurement.**

- Gate B's recoverable share is **at most 246 of 2394 transitions = 10.3%** of all
  move-action transitions, **not 26.8%**. The advertised number is 2.6× the
  achievable one.
- The **majority class (overlap, 55.2%, median leftover 35 cells) is untouched by
  either proposed repair** — and a bounded-leftover tolerance would *mis-mine*
  exactly there, because the leftover sits on the sprite's own footprint.
- The 243 recoverable transitions need **two independent repairs** (multi-delta
  and bounded tolerance), roughly 126 and 117 each. Neither alone is a
  Gate-B-sized win.

**Gate A is now the better first build on every axis** — it is one `if`, it is
already measured to recover +12 rules across 6 games with **none lost**, and it
takes `m0r0` 0→4 and `tu93` 0→3 across `MIN_VERIFIED_MOVES=2`, i.e. it converts
two games from "no model at all" to plannable. Gate B remains real but is a
**multi-delta object-model project**, not a threshold change.

## 4. LIMITS, STATED

This measures **transitions recovered, not rules admitted, and not score.** A
recovered transition only helps if it yields a delta consistent enough to survive
the independent `verify()` prequential gate at `VERIFY_PRECISION = 0.90`. That is
the same limitation exp 61 stated for Gate A's +12, and it is not discharged here
for either gate. `overlap` is diagnosed by geometry (leftover adjacent to or on
the sprite footprint), not by reading the games — the label is a shape claim, not
a semantic one.
