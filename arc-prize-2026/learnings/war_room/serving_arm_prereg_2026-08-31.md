# PRE-REGISTRATION — A24 SERVING ARM (four vLLM flags)

**Sealed 2026-08-31 08:35 EDT, before the kernel was pushed and before any data landed.**
Kernel `canivel/arc3-serving-mtp3` v1. Build script `scripts/build_serving_mtp3.py`,
smoke `scripts/smoke_serving_mtp3.py` (18/18 PASS on CPU, run pre-push).

---

## 1. What the arm is

The certified field-floor notebook (`notebooks/q38-field-eval/arc3-q38-field-eval.ipynb`,
config mean 1.541 over n=8 public draws) with **exactly one changed cell (cell 5)**, which
adds four flags to the vLLM launch argv inside the bundled setup here-doc:

| flag | floor (verified in the mounted bundle) | armed |
|---|---|---|
| `--max-model-len` | `65536` | **`262144`** |
| `--kv-cache-dtype` | absent | **`fp8`** |
| `--speculative-config` | absent | **`{"method":"mtp","num_speculative_tokens":3}`** |
| `--async-scheduling` | absent | **present** |

Everything else is byte-identical to the floor. `ANALYZER_CONTEXT_WINDOW` stays `32768`;
gameplay, prompt and tool policy are untouched. **This is a serving change only.**

**Provenance.** `romantamrazov/arc-real-agi-solution` (The AGI Boys, board #22, **2.66** —
the current public ceiling) cell 8, whose own comment reads *"Fallback = exact MTP3+async
serving that produced 2.66 LB"* and *"V31 keeps the V22 gameplay/prompt/tool policy
unchanged."* Same wheelhouse (`driessmit1/arc3-vllm-h100-wheelhouse-v3`, vllm==0.19.0),
same model pin (`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`), same `machine_shape`
(`NvidiaRtxPro6000`), same `docker_image`. The attribution is **[V-doc] (the author's own
comment), not [V] (an isolated A/B)** — that limit is stated here so it cannot be quietly
upgraded later.

`feedback_vllm_params` is binding: **all four or none.** The patch refuses to half-arm —
if any anchor is missing it runs the floor command unchanged (proved by smoke S8/S8b).

---

## 2. Why this arm and not another mechanism

Every measured finding this campaign owns points at the same binding quantity:

- `feedback_decision_budget_binding`: **675/675 games died on the 7920 s clock at 12.8% of
  design turns.** The clock binds; the action cap never does.
- Our own bench pull today (`runs/bench_pull_0831/`): 4 games × 7920 s each, and the
  per-game generated-token counts are **242,119 ± 2,100 (0.9% CV)** on m0r0. The token
  budget per game is very nearly a physical constant of the rail.
- Tennant's independent replication: ~30.7 tok/s per stream, 97.6–98.3% of generated
  tokens spent in hidden reasoning, 12/12 games ending inside an analysis step that never
  emitted an action.

Speculative decoding multiplies exactly that quantity **at an unchanged output
distribution**. It is also architecture-neutral, so unlike every mechanism graft screened
this month it should transfer to the private set intact (`feedback_arc_generalization_first`).

---

## 3. GATE 0 — fireability (prerequisite, not the result)

`feedback_verify_treatment_can_fire` / `feedback_guard_never_fired`: the run writes
`arc3_vllm_launch.json` (the argv the child process was ACTUALLY launched with) and
`arc3_serving_arm.json`.

**GATE 0 PASSES iff** `state == "ARMED"` **and** `flags_present ==
["--async-scheduling", "--kv-cache-dtype", "--speculative-config"]` **and**
`max_model_len == "262144"`.

If `state` is `FALLBACK` or `FLOOR`, the armed launch never ran and **no throughput claim
of any kind may be made** from this artifact.

---

## 4. PRIMARY READ — throughput (certifiable at n=1)

**Metric:** `generated tokens/sec (job wallclock)` from `summary.txt` of the 25-game ×
1-pass × 7920 s offline eval, benchmark label `anim-20260807-anim-25g-p1`.

**Comparator — same bundle, same 25-game shape, same serving stack, n=3:**

| artifact | date | tok/s | total tokens | total actions |
|---|---|---|---|---|
| `runs/kernel_pulls/q38_field_v1` | 08-20 | 263.61 | 2,103,403 | 1639 |
| `runs/kernel_pulls/q38graft_v1` | 08-21 | 277.66 | 2,213,745 | 1251 |
| `runs/kernel_pulls/seed_0829` | 08-28 | 257.58 | 2,040,319 | 1449 |
| **mean / sd** | | **266.28 / 10.31** | 2,119,156 / 87,510 | 1446 / 194 |

(`q38graft_v1` differs from the floor in gameplay grafts but **not** in the serving stack,
so it is a valid throughput comparator and is included deliberately to widen the null.)

**Sealed decision bands (n=3 null, one-sided — throughput can only be bought, not lost, by
adding a decode accelerator; a *drop* is a distinct failure and is called out separately):**

- **FIRES** — tok/s **> 297.21** (= mean + 3 sd, i.e. **+11.6%**). Certified throughput gain.
- **INCONCLUSIVE** — 276.60 ≤ tok/s ≤ 297.21 (between +1 sd and +3 sd).
- **REFUTED** — tok/s **< 276.60** (inside the historical range). The four flags do **not**
  buy throughput at 25-game concurrency on our rail, whatever they did for The AGI Boys.
- **HARM** — tok/s **< 235.35** (mean − 3 sd) or the run ERRORs after GATE 0 passed.

**Named risk, recorded before the fact:** the comparator is measured at ~25 concurrent
streams. Speculative decoding pays most at *low* batch and can pay nothing (or cost) when
the server is already throughput-saturated. A REFUTED read is therefore a **live and
expected** outcome, not a build failure, and must not be re-scoped after the fact
(`feedback_screen_calibration_range`).

**Secondary, reported but NOT gating:** `total actions` (more tokens should mean more
decisions) and `total tokens`.

---

## 5. SCORE — explicitly NOT a promotion gate

Offline `mean score` on this shape has been 6.17 / 3.22 / 4.48 across the three comparator
runs. That spread makes an n=1 score read uninformative, and `feedback_stop_redrawing` plus
`feedback_screen_calibration_range` both say so. **Score is recorded, never gated on.**

---

## 6. SEALED SUBMIT RULE (decided now, before the data)

| GATE 0 | PRIMARY | tonight's queue head |
|---|---|---|
| ARMED | **FIRES** | **`canivel/arc3-serving-mtp3` v1** — a certified gain on the binding quantity, architecture-neutral, worth starting a new config mean at n=1 |
| ARMED | INCONCLUSIVE | **certified field floor** (`canivel/arc3-q38-field-eval`) |
| ARMED | REFUTED or HARM | **certified field floor**; log the refutation, retire the flags lane |
| FALLBACK / FLOOR | — | **certified field floor** |

The floor is the fallback in every branch, and the queue is never empty
(`ref_arc_daily_protocol`). No draw is spent on an unfired or unread arm.

---

## 7. What this arm CANNOT tell us

- It cannot certify that the flags are why The AGI Boys scored 2.66 — that remains their
  own comment plus a board step, never an A/B.
- It cannot settle whether extra tokens convert into score. A throughput gain is a
  *necessary* condition for the budget reframe, never a sufficient one
  (`feedback_verify_treatment_can_fire`: fireability ≠ effect).
- A single public draw settles nothing about the config mean either way.
