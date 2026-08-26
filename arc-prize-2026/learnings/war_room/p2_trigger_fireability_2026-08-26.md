# P2 STUCK-TRIGGER FIREABILITY GATE — 2026-08-26

**Ordered by:** the 08-25 handoff — *"test the H=4 counter against retained `benchmark.json` histories BEFORE the push, because D1 needs ≥15/25 games to enter `retry_mode` and `hard_noop_guard` has never fired in 5,255 actions; a trigger that cannot fire is this campaign's signature defect."*

**VERDICT: FIRES.** 19/25 on the certified field floor (bar ≥15/25), and ≥15/25 on all four independent real corpora. **P2 may take slot 1.**

## 1. The definition under test — as SEALED (the leg is not yet built)

`learnings/war_room/p2_reset_retry_prereg_2026-08-22.md` §3.2:
> After **H consecutive acting turns on the same level with no `level_completed`**, the `python` tool result carries `retry_mode: on, episodes_available: K`.

§3.3: **H = 4**, K = 5, episode cap 40 actions, retry disabled once k ≥ 4 on that game.

**Implementation status:** `duck_eval/p2/p2_patch.py` contains `H_STUCK_TURNS = 4` (line 49) and passes it into the tool-result message (lines 227, 231) — **but no counter, no increment site, no reset site exists.** The trigger leg is genuinely unbuilt, exactly as the 08-25 entry recorded. This gate therefore evaluates **the sealed definition**, not shipped code, and the build must implement precisely what was measured here.

## 2. Reconstruction from retained artifacts

Per game in `benchmark.json`: `history` is a per-**action** list; `actions_per_level` gives the action-index boundaries of each level; `levels_completed` gives how many levels actually cleared.

- **Turn boundary** = a history entry with `generated_tokens > 0` (subsequent actions in the same turn carry 0 — they are the batched actions of one generation).
- **Turn → level** = the level containing that turn's first action.
- **Cleared** = level index < `levels_completed`. A turn on a cleared level resets the counter; a turn on an uncleared level increments it.

### PROXY VALIDATION (this is a reconstruction, so it is checked against an independent instrument)

| quantity | this reconstruction | perturn program (independent, 08-22) |
|---|---|---|
| acting turns, field floor | **424** / 1,639 actions | **424** |
| turns per game | **17.0** | **17.0** |
| turns firing exactly 1 action | 50.7% | 53.5% |
| turns firing ≥8 actions | 14.6% | 12.2% |

The turn count reproduces **exactly** (424/424). The two action-multiplicity figures differ by ~3 pp because the two instruments break ties differently on zero-token continuation entries; the trigger depends only on turn boundaries, not on multiplicity, so this does not move the verdict. **Labelled PROXY on multiplicity, EXACT on turn count.**

## 3. RESULT — four independent real corpora

| corpus | games firing at H=4 | median turns/game | `max_stuck_run` distribution |
|---|---|---|---|
| **`q38_field_v1`** (certified field floor = P2's actual vehicle) | **19/25** | 18 | 0,0,1,2,2,2,**4**,5,5,5,7,7,8,8,9,9,11,13,15,15,16,17,19,23,26 |
| `budget_t3_v1` (3× budget) | **23/25** | 28 | 2,2,**4**,5,6,6,7,8,9,9,10,13,15,15,18,20,20,23,25,27,28,40,56,90,118 |
| `p1_notes_v1` | **19/25** | 16 | 0,0,2,2,3,3,**4**,4,6,6,7,7,8,10,10,11,12,12,13,14,17,21,23,24,29 |
| `execwm_v1` | **19/25** | 17 | 0,1,2,2,3,3,**4**,4,6,7,7,8,8,8,8,9,11,12,13,16,16,17,26,30,53 |

**Bar ≥15/25 is cleared on every corpus, with 4 games of margin on the vehicle.** The margin is not fragile: on the field floor, 15 of 25 games have `max_stuck_run ≥ 7`, so H would have to be raised past **7** before delivery drops to the bar and past **9** before it fails. H = 4 is safely inside the firing regime, not on its edge.

## 4. NEGATIVE CONTROL — the detector CAN refuse (mandatory)

**Real refusals.** 6 of 25 field-floor games never fire, and they are exactly the games that are *winning*:

| game | max run | turns | levels cleared | why it correctly refuses |
|---|---|---|---|---|
| `sb26` | 2 | 23 | **7** | clears constantly; counter never survives to 4 |
| `lp85` | 0 | 12 | 3 | every turn lands on a cleared level |
| `re86` | 0 | 16 | 2 | idem |
| `sc25` | 2 | 21 | 2 | idem |
| `cn04` | 2 | 18 | 1 | |
| `ka59` | 1 | 21 | 1 | |

**Synthetic refusals.** Clears-every-turn → `False`. Three-stuck-then-clear, repeated → `False`. Exactly 3 stuck turns → `False`. Exactly 4 → `True`. **The boundary is where it was sealed.**

## 5. THE CATCH WORTH STATING BEFORE THE BUILD

`sb26` is one of the six refusers — and `sb26` carries **50.4% of the certified field floor's entire `mean_score`**. The trigger is correctly silent on the single most valuable game on the rail. **P2 cannot lift our best game by construction**; it can only work on the 19 that are stuck. That is the right design, but it caps the arm's reachable upside and should be priced into the read, not discovered after it.

## 6. WHAT THIS GATE DOES **NOT** ESTABLISH

This proves the **stuck condition is reachable** — i.e. `retry_mode: on` will actually be emitted on 19/25 games. It says nothing about whether the model then **calls `attempt()`**.

That is D2 (≥25% of retry-mode turns calling `attempt()`), and it remains the arm's real risk: `feedback_advertise_where_model_reads.md` records a schema-only affordance delivering at **96.3%** and getting **1.3% use** against a 30% bar. **Measure USE, not delivery.** The build must instrument the call count per retry-mode turn, or the read will be unevaluable in exactly the way P1's was.

## 7. RECOMMENDATION

**P2 may take slot 1.** Build the trigger leg to the definition measured here — increment on an acting turn whose level is uncleared, reset on a level clear, fire at run ≥ 4 — and instrument `attempt()` **call** counts, not just emission counts.
