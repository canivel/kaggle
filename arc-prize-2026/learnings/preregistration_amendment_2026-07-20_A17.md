# Pre-registration amendment — 2026-07-20 (A17 revision, post-R15)

**STATUS: DRAFT — NOT FILED.** Prepared 2026-07-20 for panel sign-off; to be appended
to the R16 circulation (or filed as `preregistration_amendment_2026-07-20.md` on
sign-off). Supersedes §A17 of `learnings/preregistration_amendment_2026-07-18b.md` in
full. Incorporates the four R15 A17 directives (repaired gate boolean, sealed
comparator statistic, hardware-SKU verification, quota ledger) per
`learnings/panel/round15/_directives.md`; the arithmetic is imported verbatim from
`learnings/war_room/a17_72b_screen_scope.md` (filed 2026-07-19). Everything below
seals BEFORE any bench observation: no 72B kernel has been pushed, no 72B tokens/s
number exists, and no term in this amendment conditions on one.

## A17′ — war-v4 72B capability screen (revised; pre-Aug-1, blocking)

### 1. Model artifact — sealed, with negative seal

1.1. The screen model is **Qwen2.5-VL-72B-Instruct-AWQ**, Kaggle Model
`qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1` (official QwenLM artifact,
verified present; 11 safetensors shards, 43,023,138,387 B ≈ 43.0 GB, AWQ W4A16),
attached as a Kaggle Model source — no upload, no download, no cloud spend.

1.2. **Negative seal:** the harness is MULTIMODAL — it renders the current grid as a
4× upscaled image (`MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`) and the
27B baseline is itself a VL model (`Qwen3_5ForConditionalGeneration` +
`Qwen2VLImageProcessor` in the server log). **Any text-only 72B artifact renders the
screen VOID** — it deletes the visual channel and confounds capability with a modality
regression. A run on a text-only model is discarded unscored; it neither GOes nor
NO-GOes the gate.

1.3. If a Qwen3-tier VL-72B AWQ artifact appears on Kaggle before the first push, it
may be substituted (same swap procedure) with a one-line filed note; no other
substitution is permitted. The original A17 phrase "Qwen3.6-72B-tier" is retired: no
such attachable Kaggle artifact exists (searched 2026-07-19, models + datasets, nil).

### 2. Games, comparator, and capability prong

2.1. Games: **ft09, sb26, lp85, vc33**, identical harness, full per-game fixed
~7920 s wallclock window, on the free Kaggle build rail.

2.2. **Comparator statistic (sealed): per-game MAX over the 3 certified 27B seeds
(war_eval v1/v2/v3 + W0), on BOTH sides.** The screen tests capability existence,
which is a max-property; max-on-both-sides is symmetric and matches the banking
line's order statistic. Frozen 27B side: ft09 MAX 2, sb26 1, lp85 1, vc33 2 →
**Σ 27B MAX = 6**.

2.3. **Capability prong:** Σ(72B per-game MAX lc) ≥ Σ(27B per-game MAX lc) + 2 =
**≥ 8**. The 72B-side max is taken over however many 72B seeds the quota affords (≥1).

2.4. **Marginal-result rule (pre-stated):** if the capability prong lands at exactly
+1 (Σ 72B MAX = 7), or either prong sits within one level of its threshold, run ONE
additional 72B seed on the two decisive games (largest 72B-vs-27B per-game gap) and
re-evaluate MAX. If still +1 after that seed → NO-GO. No further re-rolls.

### 3. Throughput-adjusted null — sealed formula and frozen arithmetic

3.1. The screen is a fixed-wallclock race, not fixed-action. Define
**ρ = tok/s(27B) / tok/s(72B)**, both measured from the `generated tokens/sec`
`summary.txt` line **on the same SKU** (§5). ρ is **self-measured: no external
throughput anchor exists for 72B-AWQ on this card** (daily brief 2026-07-19 §1c);
our bench is the reference. 27B reference: 192 tok/s (`w0_eval_s1/summary.txt`).

3.2. **N₇₂B(game) = ⌊(1/ρ)·N₂₇B(game)⌋**, where N₂₇B = the W0 27B baseline's total
actions in that game (Σ actions_per_level, `runs/kernel_pulls/w0_eval_s1/benchmark.json`).

3.3. **null_adj(game)** = the number of levels the W0 27B baseline had fully completed
by action N₇₂B (cumulative walk of its frozen `actions_per_level`; a level counts iff
its block closes within N₇₂B actions).

3.4. Frozen worked example, from W0 actions_per_level (ft09 [27,10,2]; sb26 [16,209];
lp85 [8,139]; vc33 [7,19,43]):

| game | N₂₇B | N₇₂B (ρ=2.5) | null_adj (ρ=2.5) | N₇₂B (ρ=3.0) | null_adj (ρ=3.0) |
|---|---:|---:|---:|---:|---:|
| ft09 | 39 | 15 | 0 | 13 | 0 |
| sb26 | 225 | 90 | 1 | 75 | 1 |
| lp85 | 147 | 58 | 1 | 49 | 1 |
| vc33 | 69 | 27 | 2 | 23 | 1 |
| **Σ** | | | **4** | | **3** |

Worked walk, ft09 at ρ=2.5: N₇₂B = ⌊39/2.5⌋ = 15; level-1 closes at cumulative action
27 > 15 → zero levels credited. A 72B merely matching 27B skill clears 0 ft09 levels
in the throttled budget — the throughput penalty made concrete. At ρ=3.0, vc33:
N₇₂B = ⌊69/3⌋ = 23; L1 closes at 7, L2 at 26 > 23 → null_adj = 1.

3.5. **Σ null_adj = 4 if measured ρ ≤ 2.5; = 3 if 2.5 < ρ ≤ 3.0.** If measured ρ
falls outside [2.4, 3.1], null_adj is recomputed at the measured ρ by the frozen walk
of §3.3 on the frozen W0 data — the PROCEDURE seals, leaving no post-hoc freedom.

### 4. Gate boolean — sealed decision rule (verbatim from the scope doc)

```
GO  iff
    ( CAPABILITY:  Σ(72B per-game MAX lc)  ≥  Σ(27B per-game MAX lc) + 2      # ≥ 8
      AND
      ACTION-PARITY:  Σ N₇₂B  ≥  0.90 · Σ N₂₇B )                             # throughput not binding
  OR
    ( CAPABILITY  (same ≥8 bar)
      AND
      THROUGHPUT-ADJUSTED:  Σ(72B per-game lc)  ≥  Σ null_adj  +  MARGIN )   # throughput binding, but wins anyway
NO-GO otherwise.
```

4.1. **MARGIN = +1 level**, registered: Σ(72B lc) ≥ 5 at ρ≤2.5 / ≥ 4 at ρ≤3.0. The
margin protects against a ρ-measurement error stepping null_adj by one integer.

4.2. The test is exact — integer level counts, no p-value; the margin IS the test.
n = 1–2 72B seeds cannot power a sign-flip test and the panel is asked not to demand
α on a capability-existence screen; this is stated now so it cannot be litigated later.

4.3. This repairs the R14-era defect the R15 panel named: the original A17 conjunction
(capability AND ≥90% action parity) auto-failed under any real 2.5–3× slowdown,
making NO-GO deterministic and null_adj dead code. The disjunction closes that branch.

### 5. Hardware SKU and self-measurement seal

5.1. Both rails are the **same verified physical SKU**: NVIDIA RTX PRO 6000 Blackwell
Server Edition ×1, ~96 GB (build-rail log `w0_eval_s1/…eval.log` CUDA check; scored
rail `machine_shape: NvidiaRtxPro6000` + harness hard-assert). Build-time throughput
therefore transfers to the scored budget with no cross-SKU correction.

5.2. The 72B tokens/s probe MUST run on this exact SKU before N₇₂B is computed. If
any kernel log prints a different GPU name, the null (§3) and the gate (§4) are
recomputed from scratch and the offending run is not scored.

### 6. Budget and deadline

6.1. **~7.5 GPU-h total** (canary push + scored bench + optional marginal seed, at
~2.5 GPU-h/push) on the free Kaggle rail's 30 GPU-h/wk; **$0 cloud spend** (zero-budget
rule). Contention with A14 cumulative look + A15 full-budget replicate is a stated
dependency: the weekly scheduler must keep (screen + A14 + A15 + open v3 windows)
≤ 30 GPU-h in each of the Jul 20–27 and Jul 27–Aug-3 weeks; the canary and scored
bench are the protected pair, the marginal seed yields first.

6.2. **The screen must READ OUT (gate evaluated, GO/NO-GO recorded) before Aug 1.**
It is blocking for war-v4 scoping: no v4 registration may file without this readout.

### 7. Pre-push runtime tests — sealed as BLOCKING

No scored push occurs until all three pass; a scored run made without them is discarded.

7.1. **Serve-config tool-call round-trip:** Qwen2.5-VL takes `--tool-call-parser
hermes`, has NO qwen3 reasoning parser and NO thinking mode — the 27B's
`qwen3_coder`/`--reasoning-parser qwen3`/`preserve_thinking` flags are removed, and
`LOCAL_ANALYZER_ENABLE_THINKING=false` is set. The boot smoke test is extended to
assert a TOOL CALL round-trips (not merely a chat completion), since a silent parser
mismatch is the highest-probability zero (`feedback_test_before_submit` class).

7.2. **Reset-path A/B — byte-identical to the frozen fork:** per the reset-fragility
caution (daily brief 2026-07-19 §1b: a reset-cap change turned a 9-min agent into a
1-hour 0-score run), the 72B swap changes ONLY the model + its serve-config constants.
The v4-eval builder asserts the reset constants (`ONLY_RESET_LEVELS=true`,
`max_runtime_minutes: 45`) and the ~7920 s game-window deadline are byte-identical to
the 27B baseline; the W0 27B seeds are the implicit control arm. A run whose window
is not ~7920 s voids the null comparison and is discarded, not scored.

7.3. **Preflight structural checks** (`scripts/preflight.py`) pass — no scratch-built
kernel drift (fingerprint family `provenance:scratch-built`, n=5, stays at n=5).

### 8. Fail consequences — sealed now

8.1. **NO-GO → the war-v4 line CLOSES for the campaign.** The frozen 27B stack is the
terminal model; the finding ("72B replicates the ~1-level grinder profile under the
binding budget") goes to the panel immediately, which then decides in July — the
campaign proceeds with no registered wall-closer. **No partial credit**: a near-miss
is a NO-GO after the §2.4 marginal seed, full stop.

8.2. **No re-screen without a materially different artifact.** "Materially different"
means (exhaustively): a different model family or generation (e.g. a Qwen3-tier
VL-72B appearing on Kaggle), a different parameter class, or a quantization change
that alters measured ρ by ≥ 0.5. Re-running the SAME artifact with tuned serve flags,
prompts, or seeds is not material and is prohibited.

8.3. **GO → the war-v4 build window opens**, gated by its OWN subsequent
pre-registration (A14-form gate: sealed prongs, sealed consequences) which must
circulate to the panel before v4 consumes any scored window. GO grants the right to
register, not the right to ship.

### 9. Seal hygiene

Per methodology N6 (R15): this amendment is the hash-committed threshold file for the
A17 screen; it is committed BEFORE any 72B measurement script runs. Bench results land
in separate append-only artifacts (`runs/…/summary.txt`, `benchmark.json`); the gate
evaluation (~Jul 30) cites this file's sha and applies §3–§4 arithmetic with no free
parameters.

— END A17′ (DRAFT — NOT FILED; seals on panel sign-off at R16 circulation) —
