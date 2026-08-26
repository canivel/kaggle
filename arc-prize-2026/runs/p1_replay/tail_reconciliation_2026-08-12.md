# Blind-batch tail: 195 vs 101 — reconciliation by independent re-derivation (2026-08-12)

**Question.** `learnings/war_room/efficiency_diagnosis_2026-08-12.md` §1 publishes bucket (c)
"blind-batch tail" = **195** actions (17.6%) of the 1,110 actions on the 17 cleared levels of
`runs/kernel_pulls/animation_v1`. `duck_eval/warpack/p1_replay_validate.py` →
`runs/p1_replay/report.json` (`published_arithmetic[0].blind`) reports **101**. Both come from the
same three runs and the same `artifacts/*_p0_events.jsonl`.

**Method.** Everything below is re-derived from the raw event logs by a reader written from scratch
for this note (own md5 board fingerprint, own level counter, own batch segmentation) — neither
artifact's code was imported. Read-only, local CPU, no spend.

Reader validated before use, against quantities both artifacts agree on:

| check | result |
|---|---|
| per-level action counts vs `benchmark.json` `actions_per_level`, all 25 games | **exact** |
| the diagnosis's `act` column, all 17 cleared levels (sum 1,110) | **exact** |
| duplicate `(s,a)` actions on cleared levels | **117**, matches both artifacts |
| re-traversal actions on cleared levels (diagnosis column (e)) | **180**, matches the diagnosis |

---

## 1. The two definitions, stated precisely

Both artifacts agree on the *dead-batch rule*: inside a multi-action batch, once an executed action
either no-ops (`board_changed == false`) or lands on a board already visited this level, every
**later** action of that same batch is "blind" — the model never saw a frame for it. They differ in
**what happens to an action that is blind AND is also a duplicate `(board_hash, action)`**:

- **D-gross (the diagnosis, bucket (c)).** Count every action fired after its batch went dead,
  *regardless* of whether it is also a duplicate. Bucket (b) is counted the same way,
  independently. → **the two buckets overlap.**
- **D-marginal (the validator, `published_arithmetic`).** `bucket = "dup" if is_dup else ("blind" if
  dead else "nec")` — duplicates take precedence, so (c) counts only the blind actions that (b) has
  **not** already claimed. → **the two buckets partition the actions.**

Scope is identical in both: cleared levels only (`level < levels_completed`), all 25 games, the same
three runs. Scope is **not** the explanation.

## 2. Independently re-derived counts — animation_v1, 17 cleared levels, 1,110 actions

| level | act | (b) dup | (c) D-marginal | (c) D-gross | overlap | union removable | necessary |
|---|---|---|---|---|---|---|---|
| tn36 L1 | 29 | 0 | 0 | 0 | 0 | 0 | 29 |
| tn36 L2 | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| m0r0 L1 | 138 | 23 | 12 | 27 | 15 | 35 | 103 |
| bp35 L1 | 175 | 8 | 8 | 15 | 7 | 16 | 159 |
| tu93 L1 | 70 | 4 | 15 | 15 | 0 | 19 | 51 |
| tu93 L2 | 20 | 8 | 0 | 0 | 0 | 8 | 12 |
| lp85 L1 | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| ka59 L1 | 18 | 0 | 1 | 1 | 0 | 1 | 17 |
| vc33 L1 | 21 | 0 | 0 | 0 | 0 | 0 | 21 |
| lf52 L1 | 21 | 0 | 0 | 0 | 0 | 0 | 21 |
| sc25 L1 | 22 | 0 | 0 | 0 | 0 | 0 | 22 |
| sp80 L1 | 225 | 12 | 41 | 47 | 6 | 53 | 172 |
| ar25 L1 | 191 | 45 | 24 | 67 | 43 | 69 | 122 |
| ar25 L2 | 53 | 13 | 0 | 13 | 13 | 13 | 40 |
| sb26 L1 | 13 | 0 | 0 | 0 | 0 | 0 | 13 |
| cd82 L1 | 65 | 4 | 0 | 0 | 0 | 4 | 61 |
| su15 L1 | 17 | 0 | 0 | 0 | 0 | 0 | 17 |
| **TOTAL** | **1110** | **117** | **101** | **185** | **84** | **218** | **892** |

Arithmetic: 117 + 185 = 302 counted twice over 84 actions ⇒ union = 302 − 84 = **218**.
Equivalently 117 + 101 = 218. Necessary = 1110 − 218 = **892**.

**D-marginal = 101 reproduces the validator exactly.** **D-gross = 185 does not reproduce the
diagnosis's 195** — it is 10 short (bp35 −5, ar25 L1 −5, tu93 L1 +4, tu93 L2 −3, sp80 −1).

### 2.1 How hard 195 was hunted

≈1,000 rule variants were swept over: batch boundary (`batch_index == 1` vs the validator's
`(analysis_step, batch_size)` key vs `analysis_step` alone), dead trigger (no-op / level-revisit /
in-batch cycle / duplicate, and all combinations), whether the triggering action itself counts,
whether the `bs > 1` guard applies, dead-flag scope (per batch / per level), visited-set scope
(level / game) and update rule (every action / batch-final only), level attribution (derived counter
vs the `level` field), RESET handling, and all three precedence orders.

- **117 for bucket (b) is rock-solid** — invariant under every variant that does not give blind
  precedence (blind-first collapses (b) to 33, so the diagnosis is not using blind-first either).
- **The best reproduction of (c) is 185**, |Δ| = 10 in total and |Δ| = 18 summed per level.
  **No variant produces 195.** Neighbouring values reachable: 101, 140, 156, 185, 215, 218, 509,
  609, and 671 for the strictest "every action after the first in any batch" reading.
- Independent corroboration that 185 is the right D-gross value: the **validator's own**
  `baseline_stats()` — a different code path from its `published_arithmetic()` — reports
  `baseline.blind = 185` on cleared levels (`blind_rate_before = 16.67%`) in every arm of
  `report.json`. So `report.json` already contains **both** numbers, 185 and 101, and the two
  artifacts differ by only 10 actions on the gross figure, not by 94.
- Batch segmentation note (checked, benign): the validator keys batches on
  `(analysis_step, batch_size)`, which merges consecutive batches — 1,362 key changes against 3,150
  true batches by `batch_index == 1`. It changes **neither** count, because the merged neighbours are
  single-action batches that the `bs > 1` guard excludes anyway.

### 2.2 Replication on the other two runs (same re-derivation)

| run | cleared actions | (b) dup | (c) marginal | (c) gross | union |
|---|---|---|---|---|---|
| `animation_v1` | 1,110 | 117 | **101** | **185** | 218 |
| `a22_v2_seed1` | 607 | 30 | **27** | **42** | 57 |
| `a22_compaction_v1` | 869 | 43 | **23** | **41** | 66 |

The marginal column matches `report.json` (27, 23) exactly on both.

## 3. Score consequence — the +0.184 headline is NOT affected

Re-scored with `scripts/phase1_gate.py:rhae_score`, removing each removable action **once**:

| run | as-run | remove union (218 / 57 / 66) | remove (b)+(c) double-counted |
|---|---|---|---|
| `animation_v1` | 1.6352 | **1.8239 ×1.1154** | 1.8414 ×1.1261 |
| `a22_v2_seed1` | 1.4075 | **1.5637 ×1.1110** | 1.5657 ×1.1124 |
| `a22_compaction_v1` | 1.4509 | **1.5878 ×1.0943** | 1.6177 ×1.1149 |

The union column reproduces `report.json` bit-for-bit. The diagnosis published **1.8188** for
M1+M3 — *below* the disjoint-removal value, therefore the diagnosis's scoring path removed each
action once and did **not** inherit the double-count. **The double-count lives only in the bucket
table's prose, not in the score.**

## 4. Verdict

1. **101 is "blind-tail actions that bucket (b) has not already removed" — the marginal
   contribution of the batch-abort mechanism on top of the memo.** It is correct for the question
   "how many *extra* actions does M3 save that M1 would not have saved anyway?" It is **not** the
   size of the blind-tail phenomenon.
2. **185 is "all actions fired inside a dead batch" — the gross size of the phenomenon**, and the
   right number for the batch-abort canary. The diagnosis's **195 is not reproducible** under any
   definition tested and should be read as **185 (16.7%)**; the 10-action difference is unexplained
   and is a defect in the published figure, not a definitional one.
3. **Neither number is the correct figure for "preventable actions in the 17 cleared levels."**
   That is the **union: 218 actions = 19.6%**, and the necessary residual is **892 = 80.4%**.

### What needs correcting in `efficiency_diagnosis_2026-08-12.md` (not edited by this note)

| claim in the diagnosis | corrected |
|---|---|
| §1 (c) blind-batch tail **195** (17.6%) | **185** (16.7%) gross; **101** (9.1%) marginal-over-(b) |
| §1 read: "**312 (28%)** provably removable" | **218 (19.6%)** — 117 + 195 double-counts 84–94 actions |
| §1 read: "**798 (72%)** genuine probes"; column (a) computed as `act − b − c` | **892 (80.4%)**; (a) must be `act − \|b ∪ c\|` |
| §1.1 "ar25 L1: 117 of its 191"; "sp80 L1: 60 of 225" | **69 of 191**; **53 of 225** |
| §5 P1 canary "blind-batch tail 17.6% → 0%" | **16.7% → 0%** (gross) or **9.1% → 0%** (marginal) |
| §2 M1+M3 = 1.8188, ×1.10, +0.184 | **unaffected** (validator/this note: 1.8239, ×1.1154) |

The framing correction is real but modest: bucket (c) does **not** add 195 preventable actions on
top of bucket (b)'s 117 — it adds **101**. The diagnosis's own headline ("only ~20% of the gap is
bookkeeping") survives, and in fact its "28% of actions are removable" figure was the *optimistic*
side of its own argument; the corrected 19.6% strengthens, not weakens, its conclusion that the
waste is mostly genuine probing.

### Not resolved here (out of scope, flagged)
§4 of the diagnosis says `stopped_early` "fired 10 times out of **190** batches". Re-derived batch
counts are 3,150 batches run-wide, **529** multi-action batches run-wide, **126** multi-action
batches on cleared levels. None is 190; the denominator's provenance is unverified.

### Provenance
Source logs `runs/kernel_pulls/animation_v1/artifacts/*_p0_events.jsonl` (+ `a22_v2_seed1`,
`a22_compaction_v1`); `benchmark.json` per run; scorer `scripts/phase1_gate.py:rhae_score`.
Compared against `learnings/war_room/efficiency_diagnosis_2026-08-12.md` §1,
`duck_eval/warpack/p1_replay_validate.py` (`published_arithmetic`, `baseline_stats`) and
`runs/p1_replay/report.json`. No sealed P1 endpoint depends on 195: the prereg's M0 band is
`saved/requested ∈ [3%, 30%]`, and its §1 already carries the 195-vs-101 non-reproduction. Nothing
pushed, submitted or committed.
