# REFERENCE-CONFIG PROVENANCE — what makes `runs/tufa_example_run` cost ~1.2k tokens per acting turn
**2026-08-22 · weekend-prep lane · ZERO GPU · forensics on artifacts already on disk**
**Answers:** the coordinator's 08-22 order (ITERATION_LOG 2026-08-22, "WEEKEND (zero GPU): … price K-token cap vs medium-effort pin"), and the caveat it raised: *if the reference is a weaker model, adopting its settings trades the wrong way.*
**Tags:** **[V]** measured this session from real artifacts · **[V-src]** read from vendored source · **[V-doc]** verbatim in a verified artifact · **[INF]** inference.

---

## 0. THE VERDICT IN FIVE LINES

1. **The reference's cheap turns are the MODEL, not a setting.** `runs/tufa_example_run` serves **`vrfai/Qwen3.6-27B-FP8`**; our floor serves **`Qwen/Qwen3.8-27B-FP8`**. Every other analyzer setting we can compare is **identical** (yield 60 s, tool_steps 0=unbounded, max_output 0=unbounded, ctx 32768, temp 0.6 / top_p 0.95 / top_k 20, thinking on). **[V]**
2. **Proof, not inference: 18 of our own pulls ran Qwen3.6 and reproduce the reference's cadence to ±25%.** Qwen3.6 pulls: **1,249–1,758** tokens/acting-turn (reference **1,406**). Qwen3.8 pulls: **2,462–6,047**. **Zero overlap across 29 pulls**, across three harness generations, two rails, and clocks from 2,700 s to 23,760 s. **[V]**
3. **The mechanism is generation LENGTH, not generation COUNT.** Generations per analyzer invocation is ~**1.22 on every config**; median per-generation length is **618 tok (Q3.6) vs 1,452 tok (Q3.8)**. Consequence: **52.5%** of Q3.8 analyzer invocations end `Yielded control to solver: turn_time_budget` with **no action**, against **22.1%** on Q3.6. The shipped 60 s yield was calibrated for the Q3.6 generation length and Q3.8 silently outgrew it. **[V]**
4. **We are squarely in the "weaker model" case the caveat feared, and the trade is not close.** On our own rail Q3.6 tops out at **lc 22 / mean_score 2.88**; Q3.8 reaches **lc 35 / 6.21** and owns the certified floor (**lc 28 / 6.173**). Median mean_score **1.63 (Q3.6) vs 2.95 (Q3.8)**. **Adopting the reference config = reverting the model = giving up the whole 08-20 field-floor adoption.** DO NOT DO IT. **[V]**
5. **But there IS an admissible "adopt the measured setting" arm, and it is not the reference's:** `reasoning_effort='medium'` on Qwen3.8 renders the Qwen3.6 prompt **byte-for-byte (1495 chars, 0 diff)** [V-doc] and measured **2,462–2,565** tokens/turn and **+72% acting turns** on our rail. It is the reference's *prompt shape* on the better weights. It has been read once (lc 21/17) — but only on the **June-30 harness**, never on the current floor. That is the cadence family's arm 1.

---

## 1. THE REFERENCE RUN, IDENTIFIED TO THE BYTE

`runs/tufa_example_run/` — 17 MB, 500 game-runs = 20 passes × 25 games, produced 2026-06-02.

| field | value | file |
|---|---|---|
| `model` | **`vrfai/Qwen3.6-27B-FP8`** | `run_config.json` |
| `agent` / `runner` | `inference` / `taaf` | `run_config.json` |
| `analyzer_timeout_seconds` | **120.0** | `run_config.json` |
| `max_runtime_minutes_per_game` | 45.0 (explicit) | `run_config.json` |
| `n_passes` / `game_count` | 20 / 25 (`include_tags:["official"]`) | `run_config.json` |
| deploy target | **slurm**, 2 GPUs, 32 concurrent jobs/GPU (64 effective), 13 h wall | `run_config.json`, `deploy_meta.json` |
| benchmark label | **`0-history-turns`** | `benchmark.json`, `deploy_meta.json` |
| kernel slug (unused; slurm ran it) | `taaf-0-history-turns` | `run_config.json` |
| source revision | `dd50b552a3b83af6339d1ee0c95f04f5af05ff07` | `git_info.txt` |
| source repos | `tufa-arc-agi-framework`, `re-arc-3`, main project `ARC3-Inference` | `deploy_meta.json`, `deployment-overrides.txt` |
| outcome | mean score **1.600**, median 0.07, 68,682 actions, 29.6 M tokens, 0 games won | `summary.txt`, `evaluation.json` |

**The `0-history-turns` label is an ablation name, and it is NOT the cause of anything.**
`_PERSISTENT_HISTORY_ASSISTANT_TURNS` is a module constant, **= 30 in every bundle generation on disk** (08-15 private bundle, `duck_eval/taaf_bundle`, `runs/fork_audit/tennant_src`, four `gpt56_probe` copies, four `harness_diff_0813` third-party datasets) [V-src], and the reference run records no history knob in `run_config.json`. The empirical settlement is stronger than the source read: **our own Qwen3.6 pulls carry the full 30-turn persistent history and land at 1,249–1,758 tokens/acting-turn against the reference's 1,406** (§3). Whatever `0-history-turns` was ablating, it is not worth a single token of the 7× gap. **[V]**

---

## 2. OUR CONFIG, SIDE BY SIDE — the comparison is one row long

Sources: `duck_eval/private/bundle_20260815/{preamble.txt,setup_commands.json}` (ours, 08-15) · `runs/kernel_pulls/q38_field_v1/arc3-q38-field-eval.log` (the certified floor's own runtime banner) · `runs/a22_compaction_v1/taaf_setup_env.json` + `runs/a22_v2_seed1/taaf_setup_env.json` (a **captured runtime env dump from our own Qwen3.6 era**, which is what makes this table a measurement rather than a source read).

| analyzer setting | reference (06-02, Q3.6) | our Q3.6 runs (08-01/08-06) | our floor (08-20, Q3.8) |
|---|---|---|---|
| served model | **`vrfai/Qwen3.6-27B-FP8`** | `vrfai/Qwen3.6-27B-FP8` | **`Qwen/Qwen3.8-27B-FP8`** |
| `LOCAL_ANALYZER_YIELD_SECONDS` | 60 | **60** | **60** |
| `LOCAL_ANALYZER_TOOL_STEPS` | 0 (= unbounded) | **0** | **0** |
| `LOCAL_ANALYZER_MAX_OUTPUT` | 0 (= no `max_tokens`) | **0** | **0** |
| `LOCAL_ANALYZER_CONTEXT_WINDOW` | 32768 | **32768** | **32768** |
| temp / top_p / top_k / thinking | 0.6 / 0.95 / 20 / on | **identical** | **identical** |
| `MULTIMODAL_CONTEXT` / upscale | current_grid / 4 | current_grid / 4 | current_grid / **4** |
| `reasoning_effort` | n/a (Q3.6 has no such knob) | n/a | **ABSENT ⇒ template default `xhigh`** |
| `analyzer_timeout` | **120 s** | 900 s | 900 s |
| `max_runtime_s_per_game` | 2,700 | 7,920 | 7,920 |
| concurrency | 64 (2 GPU) | 28 (1 GPU) | 28 (1 GPU) |

Two rows differ that are not the model. Both are dismissed on evidence:

- **`analyzer_timeout` 120 vs 900.** This is the **per-HTTP-request** timeout (`solver.py:922` → `ToolAgent(timeout=…)` → `requests.post(timeout=…)`) [V-src], not a turn or token cap. It cannot shorten a generation; it can only **discard** one. Our own Q3.6 runs used 900 s and still produced 1,457 tokens/turn, so 900 s is not what makes Q3.8 expensive. **Explicit warning for the next builder: do NOT "adopt" 120 s.** On Q3.8 the p95 turn is ~15,000 generated tokens ≈ **1,169 s** at our measured 12.88 tok/s per game-slot; a 120 s request timeout would abort the majority of turns, and the abort path sets `preserve_history=False` and **throws the turn's work away** (`tool_agent.py`, `requests.RequestException` handler) [V-src]. It would be a destruction arm wearing a cheapness costume.
- **clock / concurrency.** Covered by exp 39's elasticity curve; the T0.5/T1/T3 triple is all Q3.8 and spans 3,960→23,760 s with tokens/acting-turn **rising** 3,872 → 4,961 → 6,047 [V]. More clock does not make turns cheaper; it makes them dearer.

**Everything else that could be a "reference setting" is already our setting.** There is no cheap knob to adopt from `runs/tufa_example_run`.

---

## 3. THE DECOMPOSITION — 29 pulls, perfect separation by model

Instrument: `duck_eval/cadence/cadence_instrument.py` (written and validated this session; reproduces the 08-22 BP35 diagnostic table exactly, 31/31 checks). `tpt` = pooled generated tokens per acting turn; `turns`/`acts` = median per game; model column read from each pull's own runtime banner (`grep Qwen3.x-27B-FP8` in the kernel log).

| pull | model | tpt | turns/game | acts/game | lc | mean_score |
|---|---|---|---|---|---|---|
| war_eval_v1 | 3.6 | 1,251 | 49 | 117 | 22 | 1.58 |
| w0_eval_s1 | 3.6 | 1,249 | 44 | 147 | 16 | 1.73 |
| w0_cont_eval | 3.6 | 1,331 | 37 | 137 | 10 | 0.92 |
| sched_v1 | 3.6 | 1,364 | 42 | 172 | 17 | 1.31 |
| p1_v1 | 3.6 | 1,365 | 48 | 145 | 17 | 1.93 |
| sentinel_eval_v2 | 3.6 | 1,373 | 49 | 159 | 16 | 2.04 |
| phase1_v5 | 3.6 | 1,386 | 65 | 165 | 15 | 1.82 |
| war_eval_v3 | 3.6 | 1,388 | 41 | 165 | 13 | 1.16 |
| **REFERENCE (20 passes)** | **3.6** | **1,406** | **42** | **130** | **15.4/pass** | **1.60** |
| sentinel_eval_v1 | 3.6 | 1,429 | 42 | 140 | 12 | 0.85 |
| gate_eval_v1 | 3.6 | 1,457 | 48 | 182 | 18 | 1.43 |
| effnote_v1 | 3.6 | 1,487 | 43 | 126 | 16 | **2.88** |
| war_eval_v2 | 3.6 | 1,519 | 43 | 144 | 15 | 1.62 |
| gate_eval_v2 | 3.6 | 1,523 | 38 | 160 | 19 | 1.94 |
| a22_v2_1 / compaction_v2 | 3.6 | 1,532 | 40 | 76 | 13 | 2.17 |
| graft_confirm_v1 | 3.6 | 1,732 | 39 | 100 | 14 | 1.20 |
| graft_floor_v1 | 3.6 | 1,758 | 42 | 123 | 18 | 2.30 |
| — | | | | | | |
| engine_misfire_0817 / q38_engine_v3 | **3.8 (medium)** | 2,462 | 31 | 86 | 17 | 2.91 |
| q38_v2 | **3.8 (medium)** | 2,565 | 29 | 102 | 21 | 2.80 |
| private_edge1_v2 | 3.8 | 3,072 | 16 | 41 | 18 | 2.91 |
| budget_t05_v1 | 3.8 | 3,872 | 11 | 25 | 14 | 1.92 |
| **q38_field_v1 (certified floor)** | **3.8** | **4,961** | **18** | **60** | **28** | **6.173** |
| private_edge2_v3 | 3.8 | 5,304 | 15 | 41 | 20 | 3.14 |
| q38graft_v1 | 3.8 | 5,706 | 16 | 44 | 18 | 3.22 |
| private_base_v1 | 3.8 | 5,910 | 14 | 43 | 30 | 5.69 |
| budget_t3_v1 | 3.8 | 6,047 | 28 | 99 | **35** | **6.21** |

**Read it:** the Q3.6 block spans 1,249–1,758 and the Q3.8 block spans 2,462–6,047. Nothing crosses. The reference sits in the **middle of our own Q3.6 block**. Harness generation, prompt content, animation flags, graft flags, clock and rail all vary widely *within* each block and move `tpt` by ≤40%; the model moves it by 3.5×.

**A within-bundle control, because "different harness" is the obvious objection.** Four Q3.6 and four Q3.8 pulls carry the *same* solver label `duck-harness-kaggle`:
`gate_eval_v1 1457 · gate_eval_v2 1523 · graft_confirm_v1 1732 · graft_floor_v1 1758` (3.6) vs `q38_engine_v3 2462 · engine_misfire 2462 · q38_v2 2565 · q38graft_v1 5706` (3.8). Same harness bytes, same env, model swapped: **still no overlap.** **[V]**

### 3.1 Where the tokens actually go — length, not count

Per-generation forensics from the transcript corpora on disk (`content_chars + reasoning_chars` per `[MODEL RESPONSE META]`, chars→tokens calibrated per pull against `solver_note`):

| pull | model | chars/tok | gens per invocation | p50 tok/gen | p90 | p95 | max | invocations ending `Step executed` | ending `turn_time_budget` |
|---|---|---|---|---|---|---|---|---|---|
| war_eval_v1 | 3.6 | 2.35 | 1.28 | **477** | 1,716 | 2,262 | 8,414 | **74.0%** | 24.5% |
| p1_v1 | 3.6 | 2.27 | 1.23 | 530 | 2,044 | 2,886 | 8,624 | **78.7%** | 19.6% |
| gate_eval_v1 | 3.6 | 2.35 | 1.23 | 618 | 2,110 | 2,953 | 8,657 | **76.6%** | 22.1% |
| engine_misfire_0817 | 3.8 (medium) | 2.63 | 1.21 | 1,011 | 3,243 | 4,297 | 13,218 | 63.5% | 33.4% |
| private_edge2_v3 | 3.8 (xhigh) | 2.82 | 1.22 | **1,452** | 5,195 | 6,587 | 13,737 | **43.2%** | **52.5%** |

Three consequences, all load-bearing for the cadence arm:

- **`gens/invocation` is 1.21–1.28 everywhere.** The agent is not looping more; each single generation is 2–3× longer. A per-request `max_tokens` cap therefore *does* bound the turn — the mechanism is not defeated by a longer loop. **[V]**
- **The 60 s yield has silently stopped working.** At our measured **12.88 tok/s per game-slot** (median over the 25 field-floor games) the Q3.8 median generation is **≈113 s** — it cannot fit in a 60 s budget, and `control_yield_reason()` is only evaluated *between* generations (`tool_agent.py:2139`) [V-src], so the budget can only fire *after* the overrun. Q3.6's median generation is ≈55 s at 8.72 tok/s, i.e. **just inside** the shipped budget. **The 60 s constant was calibrated to the Qwen3.6 generation length and the model swap invalidated it.** [V for the rates; INF for "calibrated"]
- **`finish_reason: length` is 0.0% in every corpus** (0/1223, 0/1537, 0/1865, 0/2156) — nothing truncates today, so a `max_tokens` cap is a genuinely new constraint whose firing rate is a clean, previously-zero delivery instrument. **[V, and it re-confirms perturn_program §1.1 B6]**

---

## 4. THE CAVEAT, RESOLVED: WHICH CASE ARE WE IN?

The order asked this explicitly. **We are in the "the reference is the weaker model" case, and it is not marginal.**

| | Qwen3.6 (18 runs incl. reference) | Qwen3.8 (10 runs) |
|---|---|---|
| tokens per acting turn | 1,249–1,758 | 2,462–6,047 |
| acting turns / game | 37–65 | 11–31 |
| **lc_total** (25 games) | 10–**22** (median 16) | 14–**35** (median 19) |
| **mean_score** | 0.85–**2.88** (median 1.63) | 1.92–**6.21** (median 2.95) |
| best artifact | gate_eval_v2 lc 19 / 1.94 · effnote lc 16 / 2.88 | **budget_t3 lc 35 / 6.21** · **field floor lc 28 / 6.173** |

The certified field floor — the artifact the whole campaign is currently built on, and the comparator in `private_score.py` and `q38graft_score.py` — is a Qwen3.8 run. **Adopting the reference's configuration means reverting to Qwen3.6, which on our own rail has never produced an lc above 22 or a mean_score above 2.88 in eighteen tries.** The reference's own local mean is **1.600**, below eight of our Q3.6 pulls and below every Q3.8 pull except T0.5.

**This also settles the per-action point the BP35 diagnostic raised.** Our Q3.8 policy clears r11l L1 in 4–12 actions against the reference's median 14.5; it is action-*efficient* and turn-*starved*. The reference buys its cheap turns with a weaker policy. **Trading our per-action quality for its cadence is the wrong direction, and no arm in this family may propose it.**

### 4.1 The one admissible "adopt the measured setting" move

There is exactly one setting on **our** model that reproduces the reference's prompt, and it is documented and already tooled:

> `reasoning_effort='medium'` leaves Qwen3.8's `reasoning_instructions` **empty**, rendering **1495 chars — byte-identical to the Qwen3.6 template (0 diff)**; the default `xhigh` renders 1704 chars and `low` renders 1633. Re-measured every build by `duck_eval/q38/q38_smoke.py` §5. **[V-doc: `duck_eval/q38/build_q38_eval.py` header]**

So `medium` is not an invented cap — it is **the reference's prompt configuration, transplanted onto the better weights**. Measured effect on our rail (Q3.8-medium pulls): tokens/turn **2,462–2,565** (−48% vs the floor's 4,961), acting turns **29–31/game** (+72%), `Step executed` rate **63.5%** (vs 43.2%). The mechanism delivers.

**And its outcome read is the strongest pre-data evidence AGAINST the cadence thesis, which is why it must be stated here rather than discovered later:** those runs scored **lc 21 and lc 17** against the floor's 28. Halving tokens/turn and adding 72% more turns did **not** raise levels — it lowered them.

Three reasons that read is not decisive, and is worth one clean slot:
1. **Confounded by harness generation.** Both Q3.8-medium runs are the **June-30 `duck-harness-kaggle`** vehicle. The floor is the **08-07 anim** vehicle. Arm A (`private_base_v1`, 08-21) measured that generation difference alone at lc 30 vs 28 — so the vehicles are not interchangeable, and the medium read has never been taken on the current floor. The 08-22 log records this gap in terms: *"medium-on-CURRENT-floor is UNTESTED — the Q38 medium REFUTE was June-30-harness only."*
2. **Confounded by the engine-swap arm's purpose.** `q38_v2` was built to isolate *weights*, holding the prompt at Q3.6-equivalent; it was never read as a cadence arm and its turn counts were never the endpoint.
3. **n = 1 per point** against a pooled seed sd of 2.80 (MDE 11.1 at n=1). lc 21 vs 28 is **−2.5σ** — suggestive, not sealed.

---

## 5. CORRECTIONS THIS FORCES ON THE STANDING RECORD

1. **`perturn_program_2026-08-22.md` §1.5 calls the reference "a much weaker config" (A5 says the same).** Amend: it is not a *config*, it is the **predecessor model at our own settings**. Every clone-variance number taken from it (`max/mean = 2.97`, the 11-of-25 sd≥mean count, the `tr87`/`wa30` 0/20 walls) is a **Qwen3.6** property. A5 already flagged the caveat; this document supplies the reason and makes it specific.
2. **`BP35_DIAGNOSTIC_2026-08-22.md` §2's "~7× the reference's tokens per acting turn" is correct per-game on bp35/r11l/sp80 and is reproduced exactly by the instrument.** Pooled over all 25 games the same artifacts give **3.53×** (1,406 → 4,961). Both are true; quote the per-game figure only with its three games named. The instrument now asserts both.
3. **The cadence deficit is not a defect we introduced.** It arrived with the 08-15/08-20 model adoption, which simultaneously raised lc from ~19 to 28. **The campaign traded turns for per-turn quality and came out ahead.** Any arm that proposes to trade back must beat lc 28, not lc 19.
4. **`LOCAL_ANALYZER_YIELD_SECONDS=60` should be treated as a stale constant, not a design invariant.** It was fit to a generation length the current model no longer has. It is the strongest available anchor for choosing a token cap (§4 of the cadence prereg) precisely because it encodes the harness authors' intended turn.

---

## 6. WHAT THIS LICENSES, AND WHAT IT FORBIDS

**FORBIDDEN (state it once so nobody re-proposes it):**
- Reverting to `vrfai/Qwen3.6-27B-FP8` "for cheaper turns". −6 to −13 lc on our own record.
- Adopting `analyzer_timeout=120`. It discards turns; it does not shorten them.
- Adding prompt content to make turns "more efficient". Raises tokens/turn — the exact wrong direction, and 0-for-6 on this campaign.
- Reading `0-history-turns` as an efficiency setting. It moves nothing measurable.

**LICENSED — one family, two arms, ordered by attribution cleanliness:**
- **Arm 1 (cleanest): `reasoning_effort='medium'` pinned on the certified field floor.** One documented setting; renders the reference's exact prompt; no code patch; existing builder + smoke; de-confounds the June-30 read.
- **Arm 2: `LOCAL_ANALYZER_MAX_OUTPUT = K` on the floor.** An invented cap, but the only lever that bounds the *tail* rather than lowering effort uniformly, and the only one that can restore the 60 s yield's intended behaviour.

Both are sealed in `learnings/war_room/cadence_prereg_2026-08-22.md`.

---

## APPENDIX — PROVENANCE OF EVERY NUMBER

| claim | source |
|---|---|
| reference model, timeout, passes, slurm shape, git rev | `runs/tufa_example_run/{run_config.json, deploy_meta.json, deployment-overrides.txt, git_info.txt}` |
| reference outcome (mean 1.600, 68,682 actions, 29.6 M tokens) | `runs/tufa_example_run/{summary.txt, evaluation.json}` |
| our 08-15 config (model, analyzer_timeout 900, ctx, upscale) | `duck_eval/private/bundle_20260815/{preamble.txt, setup_commands.json}` |
| our Q3.6-era runtime env (the identical-settings proof) | `runs/a22_compaction_v1/taaf_setup_env.json`, `runs/a22_v2_seed1/taaf_setup_env.json` |
| the floor's own runtime banner (Q3.8, timeout 900, yield 60, tool_steps 0, max_output 0) | `runs/kernel_pulls/q38_field_v1/arc3-q38-field-eval.log` |
| `_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30` in all 10 bundle copies | `grep -rn` over `duck_eval/`, `runs/fork_audit/`, `runs/gpt56_probe/`, `runs/harness_diff_0813/` |
| yield checked only between generations; `max_tokens` only when MAX_OUTPUT>0; request-timeout abort discards the turn | `duck_eval/private/bundle_20260815/src/ARC3-Inference/inference/agent/tool_agent.py:1092-1095, 1503, 2139, 2331, 2341-2358`; `inference/framework/solver.py:922, 1466`; `inference/utils/openai_compat.py:41-58` |
| env var defaults and the Kaggle template values (yield "60", tool_steps "0", max_output "0") | `inference/agent/tool_agent.py:145-156`; `inference/framework/kaggle.py:95-135, 378-400` |
| the 29-pull cadence table | `duck_eval/cadence/cadence_instrument.py` over `runs/kernel_pulls/*/benchmark.json` + `runs/tufa_example_run/benchmark.json` |
| per-generation distributions, gens/invocation, yield-reason split | `[MODEL RESPONSE META]` + `[ANALYZER STATUS]` records in `runs/kernel_pulls/{war_eval_v1,p1_v1,gate_eval_v1,engine_misfire_0817,private_edge2_v3}/transcripts/*.txt` (8,938 generations) |
| 12.88 tok/s per game-slot (Q3.8), 8.72 (Q3.6) | generated tokens ÷ final cumulative `wallclock_seconds`, median over the 25 games of each run |
| `reasoning_effort='medium'` == Qwen3.6 template byte-for-byte | `duck_eval/q38/build_q38_eval.py` header; re-measured by `duck_eval/q38/q38_smoke.py` §5 |
| Q3.8-medium was the June-30 harness only | `learnings/war_room/q38_field_prereg_2026-08-20.md` §3; ITERATION_LOG 2026-08-22 |
