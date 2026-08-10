# S1 — SEALED SPEC (R24 minutes §5.2), 2026-08-10

**Status: SEALED. Nothing here has been executed. S1 has NOT been fired.**
$0, 0 Kaggle pushes, 0 cloud spend, CPU only. No campaign file was modified except the five
documents carrying the incorrect "91.7% held-out" claim (§6.2 below).

**Authority:** `learnings/war_room/r24_minutes_2026-08-09.md` §5.2 (six-item re-scope), §3.5 (the
FATAL that held S1), §4 decision 3 ("S1 is HELD — not run today, and not run as written").
**Scoped object:** `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md:299` (§4 row S1)
and `:304` (row S5, whose gate consumes S1's output).
**Evidence generator:** `duck_eval/r24_prep/s1_seal_audit.py` → `duck_eval/r24_prep/s1_seal_audit.json`.
Read-only; calls **no** `simulate()`, opens **no** engine, and therefore cannot pre-empt the S1
endpoint in either direction. Every number below is reproducible by re-running it.

**Headline: S1 as written is dead and is not resurrected. The carrier-set-expansion gate is
RETIRED and the coverage channel is REMOVED. A narrower, genuinely open endpoint replaces them.
Verdict: RE-SPECIFY, then FIRE (§8).**

### The six items at a glance

| §5.2 | Question | Resolution | § |
|---|---|---|---|
| **i** | which coverage channel does the gate read | **None exists.** `coverage_strict` is measured at exactly 1.0 (0 errors, 0 selfdiffs / 4,996 banked steps); the identity-abstention proxy is computed from the observed label, so it is circular *and* mechanically inflates `accepted_match`. **Coverage is removed from S1; S1 is de-listed as a falsifier.** Verified independently: 0/25 sims can abstain (AST: every return is a 3-tuple), 3/25 hold module state. | §1 |
| **ii** | numeric carrier definition ("~4" vs r16's "3") | Different objects. R16 = a **pass count** (≥3 of 5) *and* a 3-game 07-20 ship-now set; "~4" = the 07-23 set size. Fixed: criterion **`step_acc ≥ 0.92`**, baseline set **4** `{ft09, ls20, tn36, tu93}`, pass count **3**. All three retired with the gate. | §2 |
| **iii** | restrict to the 12 games with replay streams | **The "12" is wrong.** All **25** games have a complete `board`-carrying stream in each of `war_eval_v1/v2/v3`. "12" is `SAT12`, a *sim-quality* filter at `ewm_replay_dryrun.py:57-61`. S1 runs 25 games × 3 sources. **13 games have never been replayed at all.** | §3 |
| **iv** | the module-state bug's bug-fix effect | Bug confirmed present and still unfixed in `scripts/ewm_replay_dryrun.py`. **But its blast radius is zero:** g50t and re86 were never in SAT12 so have no prior number, and tr87's only boundaries sit at call index 128 — even, hence parity-neutral for a mod-2 counter. **0 of 127 / 253 / 182 steps change.** Nothing to attribute; a pre-registered `module_reset_delta_steps = 0` guards it. | §4 |
| **v** | engine-version drift | **Controlled on the trace side** — engine hashes are byte-identical across all 25 games in v1/v2/v3 (`gpt56_full` excluded: 2 of its 5 streams are drifted). **Declared on the authoring side** — the sims were fitted on *local* engines that differ from the Kaggle build on 15/25 games, and `exec_wm/observations/*.json` record no version (0/25). Mandatory stratification by `engine_matches_kaggle`. | §5 |
| **vi** | "91.7% held-out" | Confirmed wrong on both counts: it is the **Class-A share (22/24 games)**, not a `state_exact` rate, and it was scored at `--split all` on the same 200 tuples the authoring model read, for the *selected* v1/v2 winner. Mean per-game in-sample `state_exact` is **81.1** (the execution plan's 79.8 is also wrong). Corrected in-place in 5 files; frozen panel packets deliberately left intact. | §6 |

---

## 1. (i) Which coverage channel the gate reads — ANSWER: **none. Coverage is removed from S1.**

### 1.1 The abstention claim, verified independently

`s1_seal_audit.py` §A parses all 25 active sims (`exec_wm/sims/<gid>_sim.py`, excluding `_v1`/`_v2`
variants and backups — 25 files, matching `s1_s1b_execution_plan_2026-08-09.md:22`):

| property | measured | interpretation |
|---|---|---|
| sims exposing `simulate(state, action_id, x, y)` | **25/25**, all 4-arg | one uniform contract |
| distinct return arities across all return statements in all 25 sims | **{3}** — nothing else | no 4th "confidence/abstain" slot anywhere |
| return paths with a `None` or negative state element 0 | **0** | no `UNKNOWN = -1` token |
| occurrences of `UNKNOWN` / `ABSTAIN` / `NO_PREDICTION` / `unsupported` in any sim source | **0** | no documented abstain path |

**"0 of 25 sims implement abstention" is CONFIRMED**, and by a stronger test than a token grep:
the AST admits no return shape that could carry an abstention.

### 1.2 `coverage_strict` is degenerate — measured, not asserted

`coverage_strict = (steps − errors)/steps`. Over the entire banked on-trajectory corpus
(`runs/ewm_dryrun/raw.json`, 12 sims × 4 sources):

```
total steps  = 4,996
sim_error    = 0
selfdiff     = 0        (two independently-loaded module instances never disagreed)
```

⇒ `coverage_strict ≡ 1.0000` on every game and every source ever measured. The panel's
"degenerate at 1.0" is exact. Nothing in state threading can change this: the errors counted are
*shape/exception* errors, which are a property of the sim code, not of the input trajectory.

### 1.3 `coverage` (the identity-abstention proxy) is worse than degenerate — it is **circular**

In `duck_eval/r24_prep/s1_threaded_replay.py`:

```
:250   obs_changed = obs_next != obs_prev        # <- GROUND TRUTH
:273   identity    = pred_next == pred
:274   abstain     = identity and obs_changed    # <- abstention decided using the label
:280   match       = pred_next == obs_next
:284-5 if match and not abstain: n_match_threaded_committed += 1
:340-2 accepted_match = n_match_threaded_committed / n_committed ;  coverage = n_committed / steps
```

Two defects, both fatal, neither fixable at L0:

1. **The abstention indicator is a function of the observation.** A real coverage number says
   "the model declined." This one says "the model asserted no-change *and we, holding the answer
   key, know the board did change*." An executor at inference time cannot compute it. Reporting
   it as Tycho coverage misrepresents what was measured.
2. **The exclusion is guaranteed to remove only losses.** On a trajectory-aligned step
   (`pred == obs_prev`), `pred_next == pred` and `obs_next != obs_prev` jointly imply
   `pred_next != obs_next` — the step is *necessarily* a mismatch. So `accepted_match` is
   mechanically ≥ `match_all_steps`, with the gap equal to the abstention rate. The
   "Tycho-shaped success mode" the execution plan hoped to observe (`s1_s1b_execution_plan_2026-08-09.md:150-152`
   — accepted_match high while coverage falls) is **produced by the arithmetic**, not by the sims.
   24 of 25 sims can structurally return their input unchanged (AST heuristic; g50t is the lone
   heuristic false-negative — its own source documents identity-return as its action-5 fallback),
   so essentially the whole suite is eligible for this inflation.

### 1.4 Is there any non-degenerate, non-circular channel? Only one, and it is not abstention

The single label-free variant is: **abstain iff `pred_next == pred_t`**, regardless of the
observation. That *is* decidable at inference time and *is* non-degenerate. But it is not
abstention — the sim did not decline, it asserted "nothing changes" — and it cannot distinguish a
correct no-op prediction from a decline. It is therefore recorded here as **the channel L1 exists
to replace**, and it is **explicitly barred from gating S1**. It may be emitted as an unlabelled
descriptive counter (`n_identity_predictions`) and nothing more.

### 1.5 Rulings sealed under (i)

- **R1. `coverage`, `coverage_strict` and `accepted_match` are REMOVED from S1's output entirely.**
  Not demoted, not advisory — removed, so no later reader can quote them.
- **R2. The carrier-set-expansion gate is RETIRED** (see §2.3 for the additional, independent
  reason).
- **R3. S1 is de-listed as a falsifier.** The §6.4 row claiming "two independent pre-registered
  falsifiers … and L0's carrier-set-expansion test" has been corrected in-place at
  `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md:420`. **L0 cannot answer the
  round-18 unfalsifiability charge.** Real abstention requires L1 — which is exactly the
  circularity prog-synthesis filed (`r24_minutes_2026-08-09.md:181-182`) and it is conceded, not
  argued around.

---

## 2. (ii) A numeric carrier definition — the "~4" vs "3" discrepancy resolved

### 2.1 The two numbers are different objects, which is why the tilde survived

| source | number | what it counts |
|---|---|---|
| `learnings/daily_brief_2026-07-20.md:39` (R16-era consumer ruling) | **3** | *ship-now carrier set* `{ft09, tn36, tu93}`, membership criterion stated numerically: **`step_acc 0.92–0.97` AND clean/resolvable** |
| R16 §9.2 sealed cheap measurement (`learnings/panel/round16/_prompt_llm-agents.md:410-413,585`; `rl-planning.md:27`) | **5** measured, **≥3** to pass | measurement set `{tn36, tr87, tu93, ls20, ft09}`, rule *"≥0.70 @ depth ≤10 on ≥3 of 5 carriers"* |
| `learnings/stuck_review_v2_2026-07-23.md:13` (latest) | **4** | *clean carrier set* `{tn36, tu93, ls20, ft09-L1}` — the 5-set minus tr87, demoted to ALIASED-UNRESOLVED |
| `r24_successor_lane_proposal_2026-08-08.md:299` | **"~4"** | inherited the 07-23 count and lost the criterion |

So R16's "3" is a **pass count** (≥3 of 5) *and*, separately, a **set size** (the 07-20 ship-now
subset). The proposal's "4" is the 07-23 set size. Neither is wrong; they were never the same
quantity.

### 2.2 Numbers fixed (for the record; both now retired with the gate)

- **Membership criterion: `step_acc ≥ 0.92`** — the only numeric criterion anywhere in the record
  (`daily_brief_2026-07-20.md:39`).
- **Baseline carrier count: 4**, set `{ft09, ls20, tn36, tu93}` (`stuck_review_v2_2026-07-23.md:13`,
  the most recent statement, superseding 07-20's 3).
- **Pass count inherited from R16 §9.2: 3.**

The tilde is gone. These are recorded so nobody re-derives them; they no longer gate anything.

### 2.3 Independent reason the gate had to be retired: it was **unreachable**

Teacher forcing and state threading are **bit-identical up to and including the first mismatch**
(both feed the same grid until they first differ), and after that threading can only lose ground —
errors cascade and no sim in the suite is contracting. So the banked teacher-forced numbers are an
upper bound on what S1 could report.

From `runs/ewm_dryrun/raw.json` (re-derived in `s1_seal_audit.json` §F), games at TF accuracy ≥ 0.92:

| source | games ≥ 0.92 |
|---|---|
| `war_eval_v1` (**the S1 primary source**) | **ft09** — 1 of 12 |
| `war_eval_v2` | ls20, tn36 — 2 of 12 |
| `war_eval_v3` | ft09, tn36, tu93 — 3 of 12 |
| `gpt56_full` | **none** — 0 of 5 |

The prior carrier set has **4** members. On the primary source, at most **1** of the 12 measured
games can clear 0.92 under threading. The 13 unmeasured games all sit below 91% even *in-sample*
(`exec_wm/scale_summary.md:22-45`), while the nine games at in-sample 100.0 collapse to a median
on-trajectory TF accuracy near 0.30. **"Expand beyond 4" was therefore a foregone NOT_EXPANDED —
unless rescued by the circular `committed` denominator of §1.3.** A gate whose only route to PASS
is a circular statistic is not a gate.

### 2.4 And the threshold was finer than the measurement noise

Across-source range of TF accuracy for a **fixed** sim, over the three engine-identical `war_eval`
pulls (`s1_seal_audit.json` §F):

| game | v1 | v2 | v3 | range |
|---|---:|---:|---:|---:|
| sp80 | 0.026 | 0.879 | 0.067 | **0.853** |
| su15 | 0.309 | 0.149 | 0.808 | **0.658** |
| tn36 | 0.530 | 1.000 | 0.984 | **0.470** |
| lf52 | 0.301 | 0.496 | 0.752 | **0.451** |
| ft09 | 0.985 | 0.556 | 1.000 | **0.444** |
| vc33 | 0.239 | 0.667 | 0.368 | **0.428** |
| lp85 | 0.113 | 0.458 | 0.087 | **0.371** |
| ls20 | 0.637 | 0.923 | 0.803 | **0.286** |
| tu93 | 0.731 | 0.779 | 0.997 | **0.266** |
| s5i5 | 0.265 | 0.300 | 0.129 | **0.171** |
| sb26 | 0.162 | 0.106 | 0.162 | **0.056** |
| tr87 | 0.819 | 0.771 | 0.819 | **0.048** |

**Median across-source range = 0.400.** A 0.92 point threshold on a statistic whose own
source-to-source spread is 0.40 is a coin flip dressed as a criterion. **Any single-source carrier
verdict is a draw artefact.** This is the same defect as the 91.7% (§6): a point estimate with no
stated variance.

---

## 3. (iii) Restriction to games with replay streams — the answer is **25, not 12 and not 24**

`s1_seal_audit.py` §C walks every `*_events.jsonl` under each source and counts, per stream, how
many `action` events carry a `board` frame:

| source | streams | streams with a `board` on **every** action event |
|---|---:|---:|
| `runs/kernel_pulls/war_eval_v1/artifacts` | **25** | **25** |
| `runs/kernel_pulls/war_eval_v2/artifacts` | **25** | **25** |
| `runs/kernel_pulls/war_eval_v3/artifacts` | **25** | **25** |
| `runs/gpt56_probe/experiment_full/artifacts` | 5 | 5 |

**The minutes' "12" is not a stream-availability figure.** It is `SAT12`, the *sim-quality* filter
hard-coded at `scripts/ewm_replay_dryrun.py:57-61` and applied at `:257-258`
(`if gid not in SAT12: continue`) — the 12 sims that scored ≥99.5 in-sample. It restricts which
**sims** the 2026-07-18 dry-run scored, not which **games** have data:

```
SAT12 = ft09 lf52 lp85 ls20 s5i5 sb26 sp80 su15 tn36 tr87 tu93 vc33
```

The 13 games with a full replay stream and **no on-trajectory measurement of any kind**:

```
ar25  bp35  cd82  cn04  dc22  g50t  ka59  m0r0  r11l  re86  sc25  sk48  wa30
```

**SEALED: S1 runs on all 25 games × the 3 `war_eval` sources = 75 (game, source) replays.** The
`gpt56_full` source is **excluded** (only 5 games, and 2 of them are engine-drifted — §5).

**R8. The R24 minutes are themselves wrong here and have been corrected in place** with a dated
2026-08-10 note appended to §5.2 of `learnings/war_room/r24_minutes_2026-08-09.md`, plus the same
correction in `ITERATION_LOG.md` and `duck_eval/r24_prep/_log_entry_2026-08-09.md`. The frozen
round-24 review packets are untouched.

Recording for the record, per `s1_s1b_execution_plan_2026-08-09.md:437-438`: there are **25** active
sims, not 24. `bp35` has a sim and observations but was never scale-validated, so it carries no
in-sample reference number; it is included with `held_out_state_exact_pct: null`.

---

## 4. (iv) The `ewm_replay_dryrun.py` module-state bug — real in code, **zero blast radius in fact**

### 4.1 The bug is confirmed present, and is NOT fixed in `scripts/ewm_replay_dryrun.py`

`s1_seal_audit.py` §D, by source inspection:

- `load_sim(gid, "a")` / `load_sim(gid, "b")` at `scripts/ewm_replay_dryrun.py:144-145` — the module
  is loaded **once per game** and then reused for the whole trace.
- `calls_any_reset_hook = False` — the string `reset_state` / `reset_phase` / `reset_step_parity`
  appears **nowhere** in the file. No reset is ever performed at a RESET or level boundary.
- `teacher_forced = True` (`:146-155`, `boards[id(ev)] = prev_board`).

**Status: STILL PRESENT.** It is deliberately left unfixed — `runs/ewm_dryrun/report.md` was
produced by this exact file and patching it would break the artifact↔producer correspondence.
`duck_eval/r24_prep/s1_threaded_replay.py:234-235` calls `sim_reset()` at every segment start, so
the S1 runner does not inherit it.

### 4.2 The stateful set is exactly 3 — confirmed by AST, not by grep

`s1_seal_audit.py` §B looks for module-level names that some function rebinds via `global`:

| game | module state rebound | reset hook |
|---|---|---|
| g50t | yes | `reset_state` |
| re86 | yes | `reset_phase` |
| tr87 | `_step_parity`, `_seen_first_call` | `reset_step_parity` |
| the other 22 | **none** | — |

**"Only 3 sims hold state" is CONFIRMED**, and all three expose a named reset hook (none is
orphaned).

### 4.3 The claimed damage does not exist — attribution collapses to nothing

The minutes state (§5.2 iv) that *"g50t/re86/tr87 were measured with desynced hidden counters."*
Two independent checks show that is **not true of any banked number**:

**(a) Two of the three were never measured at all.** `g50t` and `re86` are **not in SAT12**
(`scripts/ewm_replay_dryrun.py:57-61`), so `runs/ewm_dryrun/` contains no number for them under
any source. There is no prior figure for a bug-fix to inflate. Only `tr87` was ever scored.

**(b) For `tr87`, the bug is provably a no-op on all three sources.** `s1_seal_audit.py` §G locates
every segment boundary by sim-call index:

| game | source | sim calls | segment boundaries (at call index) |
|---|---|---:|---|
| **tr87** | war_eval_v1 | 127 | **none** |
| **tr87** | war_eval_v2 | 253 | RESET @ **128** |
| **tr87** | war_eval_v3 | 182 | RESET @ **128** |
| g50t | war_eval_v1 | 62 | none |
| g50t | war_eval_v2 | 65 | none |
| g50t | war_eval_v3 | 89 | LEVEL_COMPLETED @ 86 |
| re86 | war_eval_v1 | 294 | LEVEL_COMPLETED @ 64, RESET @ 164, RESET @ 264 |
| re86 | war_eval_v2 | 163 | LEVEL_COMPLETED @ 32, LEVEL_COMPLETED @ 80 |
| re86 | war_eval_v3 | 235 | LEVEL_COMPLETED @ 35, RESET @ 135, RESET @ 235 |

`tr87_sim.py:276` flips `_step_parity ^= 1` exactly once per `simulate()` call, unconditionally, and
`reset_step_parity()` defaults to 0 (`:222,229`) — the same value the module initialises to
(`:218`). After **128** calls the parity is `128 mod 2 = 0`, which is **identical to what the reset
would have written**. On `war_eval_v1` there is no boundary at all. Therefore:

> **The module-state fix changes tr87's prediction on 0 of 127 steps (v1), 0 of 253 (v2), and
> 0 of 182 (v3). It is bit-identical to the buggy harness on every tr87 number the campaign holds.**

### 4.4 Rulings sealed under (iv)

- **R4. There is no bug-fix effect to attribute on the primary source.** The concern in §5.2 (iv) is
  discharged, not managed. `g50t` (v1, v2) and `tr87` (all three) have zero segment boundaries or
  parity-neutral ones; `g50t` (v3) and `re86` (all three) do reset, but neither game has a prior
  on-trajectory number, so any figure they produce is a **first measurement**, not a delta.
- **R5. Belt and braces.** The runner emits per game `sim_has_module_state`, `n_segments`, and a new
  `module_reset_delta_steps` (count of steps whose prediction differs between reset-at-boundary and
  no-reset, computed in the same pass). **Pre-registered: this must be exactly 0 for tr87 on all
  three sources and for g50t on v1/v2.** A non-zero value is a runner bug and **voids the run**; it
  is not evidence.
- **R6. `g50t` and `re86` are reported in a separate table** headed *"first on-trajectory
  measurement, no prior comparison exists"* — neither appears anywhere in `runs/ewm_dryrun/` —
  and their numbers may not be described as a gain over anything.

---

## 5. (v) Engine-version drift — **controlled where it binds, declared where it does not**

### 5.1 S1 invokes no engine

S1 replays recorded frames through pure-Python sims. `arcengine` / `kaggle-data/environment_files`
are never opened. The 15-of-25 local-vs-Kaggle mismatch recorded in
`runs/war_eval_v1/determinism_audit_25.json` (`version_mismatch_vs_kaggle`, mismatched: ar25, cn04,
dc22, ka59, m0r0, r11l, re86, s5i5, sc25, sk48, sp80, su15, tn36, tu93, vc33) binds **S1b**, not S1.

### 5.2 The trace side is CONTROLLED

Trace-id engine hashes are byte-identical across `war_eval_v1`, `v2` and `v3` on **all 25 games**
(`s1_seal_audit.json` §C, e.g. sp80 `589a99af`, sc25 `635fd71a`, tr87 `cd924810` in all three).
The three-source design is therefore an engine-held-constant robustness triple, and any across-source
spread (§2.4) is **draw variance, not engine drift**. This is the reason the sealed spec uses three
sources rather than one.

`gpt56_full` is **excluded**: of its 5 streams, `su15` (`4c352…` vs `1944f…`) and `vc33`
(`9851e…` vs `54305…`) carry **different engine hashes** from the war-eval build. Mixing it in
would silently reintroduce the drift the triple was chosen to exclude.

### 5.3 The sim-authoring side is DECLARED, not controllable

`exec_wm/collect_observations.py:49,60-63,124,147` harvests observation tuples from the **local**
`arcengine` and writes no version field. Verified: the union of top-level keys across all 25
`exec_wm/observations/*.json` is `{game_id, available_actions, tuples, summary}` — **0 of 25 record
an engine hash.**

⇒ **The sims were fitted on local engines; S1 scores them against Kaggle-recorded traces, and the
two builds differ on 15 of 25 games.** This is unrecoverable — the provenance was never written —
so it is declared, and S1 must **stratify every reported statistic by engine match**. The banked
data already hints the stratum matters (`s1_seal_audit.json` §F): mean `war_eval_v1` TF accuracy is
**0.503** on the 6 engine-matched SAT12 games (ft09, lf52, lp85, ls20, sb26, tr87) versus **0.350**
on the 6 mismatched ones (s5i5, sp80, su15, tn36, tu93, vc33). n = 6 per arm — **suggestive,
reported as a stratification, never as a finding.**

**R7. `engine_matches_kaggle` is a mandatory per-game field; every summary statistic is reported
both pooled and split by it. No S1 conclusion may be drawn from a pooled number alone.**

---

## 6. (vi) The "91.7% held-out" correction — verified and applied in-place

### 6.1 Verified

- `exec_wm/scale_summary.md:74,78` — **91.7% is the Class-A share, 22 of 24 games**, in a table
  whose other rows are B 4.2%, C 4.2%, D 0.0%. It is **not** a `state_exact` rate.
- `exec_wm/scale_summary.md:3` — the pre-correction text read *"over all 200 **held-out** tuples per
  game (split=all)"*, which is self-contradictory on its face. `exec_wm/validate_sim.py:56` defaults
  `--split all`, and `:62-64` shows a split is applied only when `--split != "all"`. **`split=all`
  scores every tuple, including the ones the authoring model read.** Never held out.
- `exec_wm/scale_summary.md:4` — the score is for *"the chosen v1/v2 winner from per-game
  evolutions"*, so it is **fit plus selection** on the same 200 tuples.
- Per-game figures (`scale_summary.md:22-45`): **23.0 (r11l) to 100.0 (9 games), mean 81.1** over
  n=24. The execution plan's "mean ≈ 79.8%" is also wrong and is corrected.

### 6.2 Applied in-place

| file:line | action |
|---|---|
| `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md:421` | §6.4 Goodhart row corrected: class share, `split=all`, in-sample-with-selection, mean 81.1 |
| `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md:420` | §6.4 unfalsifiability row corrected: **L0 withdrawn as a falsifier** (§1.5 R3) |
| `learnings/war_room/tycho_portability_2026-08-08.md:314` | *"maximise held-out `state_exact` (91.7% Class A…)"* → corrected, "held-out" struck |
| `exec_wm/scale_summary.md:3-4` | root fix: `--split all`, plus a standing CORRECTION block forbidding the 91.7% quote as fidelity evidence |
| `exec_wm/validate_sim.py:1` | docstring "held-out observations" → "recorded observations", plus an explicit in-sample warning on the default split |
| `duck_eval/r24_prep/s1_s1b_execution_plan_2026-08-09.md:136-143` | §1.6 superseded: still said "held-out"; mean 79.8 → 81.1 |

### 6.3 Deliberately NOT modified

- `learnings/panel/round24/_prompt_*.md:662` and `learnings/panel/round24/*.md` — **frozen panel
  inputs and reviews.** Editing what the reviewers were shown would falsify the record. The error is
  part of the packet and prog-synthesis's [MAJOR] at `round24/prog-synthesis.md:89-97` is the
  response to it.
- `learnings/state_of_campaign_2026-08-09.md:107,241`, `learnings/daily_brief_2026-08-09.md:132`,
  `ITERATION_LOG.md:608`, `duck_eval/r24_prep/_log_entry_2026-08-09.md:6`,
  `learnings/panel/round24/_ADDENDUM3.md:49`, `learnings/panel/round25/_prompt_*.md` — these
  **already state the correction**. No change needed.

---

## 7. What S1 can and cannot add — the honest inventory

**Cannot add (already on disk, or determined):**

- Segment-1 survival for the 12 SAT12 games. Threading is bit-identical to teacher forcing up to
  the first mismatch, so `first_div` in `runs/ewm_dryrun/raw.json` **is** the threaded survival of
  segment 1. On `war_eval_v1`: lf52 0, sb26 0, su15 0, vc33 2, s5i5 3, sp80 6, lp85 7, tn36 7,
  tr87 18, ls20 50, tu93 67, ft09 130 — median ≈ 6.5, and three sims die on step 0.
- Whether a diverged sim re-synchronises. It cannot; none of these sims is contracting.
- Any coverage number (§1) or any carrier verdict (§2).
- Any bug-fix delta on the primary source (§4.3).

**Can add (genuinely unmeasured):**

1. **On-trajectory behaviour for the 13 games nobody has ever replayed** (§3). This is the whole
   of the campaign's blind spot.
2. **Segment-restart survival** — the old harness never restarted at level frame 0, so survival in
   segments 2..n is unmeasured everywhere. `war_eval_v1` has ≥1 extra segment in 21 of 25 games
   (sp80 13, tu93 11, bp35 5, sc25 5, ar25 3, cn04 4, ft09 4, re86 3, su15 3, …).

**The decision that consumes it:** `r24_successor_lane_proposal_2026-08-08.md:304`, row **S5** —
the staged **L1** arm (migrating the sims to an abstention-carrying interface) is gated on S1. L1 is
workstation-LLM authoring time, priced by `exec_wm/scale_summary.md:97-102` at **~25 min and ~290k
tokens per game**. Whether that is spent turns on whether any sim can carry a plan on-trajectory.
That is a real, live, and currently unanswered question.

---

## 8. VERDICT: **RE-SPECIFY, then FIRE**

Not DROP: the run is $0, under 10 minutes, and items 1–2 of §7 are real gaps feeding a real
decision. Not FIRE-AS-WRITTEN: the gate and the endpoint of the original S1 are retired outright.

**S1 is reclassified from "free falsifier for lane (a)" to "L1 authorisation measurement."** It
answers one question: *is there any game whose sim can carry a plan on-trajectory?* It does **not**
answer the round-18 unfalsifiability charge, and may not be cited as doing so.

---

## 9. SEALED RUN SPEC — mechanical, fire as written

### 9.1 Pre-registered before execution (this section is the seal)

**Population.** All **25** games × **3** sources (`war_eval_v1`, `war_eval_v2`, `war_eval_v3`) = 75
replays. `gpt56_full` excluded (§5.2). No game may be dropped after seeing a number.

**Protocol.** State-threaded replay from level frame 0, segment = `initial` frame | post-RESET frame
| post-level-completion frame, module reset hook invoked at every segment start.

**Primary statistic — the threaded survival horizon.**
For each (game, source, segment): `survival` = number of consecutive threaded steps from segment
start that exactly match the recorded settled frame, before the first mismatch.
Per (game, source): `median_survival` over that source's segments.
Per game: **`H_g = min over the 3 sources of median_survival`** — a *robust* horizon, so a single
lucky draw cannot create a result. §2.4 is why the min, not the mean, is taken.

**Threshold.** `H_g ≥ 10` steps. Inherited, not invented: R16 §9.2's sealed cheap measurement is
*"≥0.70 @ depth ≤10"* (`learnings/panel/round16/_prompt_llm-agents.md:410-413`), i.e. the registered
executor plan depth is 10. A sim that cannot survive its own plan length is not usable by an
executor.

**PRIMARY ENDPOINT — E1, and it must be about the 13, not the 25.**
> **E1: how many of the 13 never-measured games (`ar25 bp35 cd82 cn04 dc22 g50t ka59 m0r0 r11l re86
> sc25 sk48 wa30`) reach `H_g ≥ 10`?**

E1 is restricted to the 13 **deliberately**, and this restriction is part of the seal. On the 12
SAT12 games the answer is already largely determined by banked data — exactly 3 of them (ft09 25,
ls20 14, tu93 67) clear the weaker *segment-1* form of the same bar on
min-over-3-sources `first_div`. **Stating that here, before the run, is what keeps a bare "3 games
passed" from being read as a result.** The 12 are still replayed (they anchor the segment-restart
comparison in E2), but they **cannot contribute to E1**.

**PRE-REGISTERED EXPECTATION: E1 = 0.** Rationale, stated before the fact: every one of the 13 has
in-sample `state_exact` ≤ 90.5 (`exec_wm/scale_summary.md:22-45`), whereas the nine games at
in-sample 100.0 collapse to a median on-trajectory accuracy near 0.30. The 13 start from a strictly
weaker position than a population that already failed.

**SECONDARY, DESCRIPTIVE, NON-GATING.**
- **E2 — segment-restart survival:** median survival in segments 2..n versus segment 1, per game.
  Answers whether failure is "one bad level" or systemic. No threshold. No verdict.
- **E3 — module-reset attribution:** `module_reset_delta_steps` per (game, source), with the §4.4 R5
  pre-registered zeros.
- **E4 — engine stratification:** every summary reported pooled **and** split on
  `engine_matches_kaggle` (§5.3 R7).

**BANNED OUTPUTS.** `coverage`, `coverage_strict`, `accepted_match`, `carrier`, `carrier_set`,
`gained`, `lost`, and any field whose value is `"EXPANDED"`/`"NOT_EXPANDED"`. If the runner emits
them the run is void. `match_all_steps` and `teacher_forced_match_all_steps` **may** be emitted as
comparability channels to `runs/ewm_dryrun/report.md`, clearly labelled as such, and may not be
thresholded.

### 9.2 What each outcome means — decided now, not after

| E1 | Reading | Action |
|---|---|---|
| **0 of 13** (the pre-registered expectation) | The executable-sim substrate tops out at ~3 usable games out of 25, after ~10 h of opus-4-8 authoring. Adding an abstention channel to sims that cannot survive 10 steps is polishing a dead instrument. | **L1 NO-GO.** S5 does not open. Bank the clean second negative that `proposal:299` already specified as the negative branch — but bank it on *this* endpoint, not on the degenerate one. exec-wm closes as an execution substrate; C1/C2/C3 retained as schema only. Lane (a) proceeds on P1/P3, which do not depend on the sims. |
| **1–2 of 13** | A small unmeasured carrier population exists. Real, but below R16's own inherited pass count of 3. | **L1 HELD.** Do not authorise the full migration. Author the abstention interface for those games only, at ≤2× 25 min, and re-read at R26. Record which games and why. |
| **≥3 of 13** | The SAT12 filter was selecting on the wrong axis (in-sample `state_exact`) and hid a usable set. This contradicts the pre-registered expectation and is the only outcome that earns L1. | **L1 GO** for the identified set, conditional on §5.3 ruling workstation authoring in-bounds (already RATIFIED, `r24_minutes_2026-08-09.md` §2 item 3). Carry the engine stratification into the prereg. |
| any E3 non-zero where R5 pre-registers 0 | Runner defect. | **VOID.** Fix, re-run, do not interpret. |

### 9.3 Runner changes required before firing

`duck_eval/r24_prep/s1_threaded_replay.py` must be amended to:
1. delete `coverage`, `coverage_strict`, `accepted_match`, `n_identity_abstain`, `n_committed`,
   `carrier*`, and the `verdict`/`gained`/`lost` summary block (§9.1 BANNED OUTPUTS);
2. emit per (game, source): `segment_survivals[]`, `median_survival`, and per game
   `H_g = min over sources`;
3. emit `module_reset_delta_steps` (second pass with reset hooks suppressed, same trace, same
   order) and `engine_matches_kaggle`;
4. **remove `gpt56_full`** from `SOURCES`, default `--source` to all three remaining `war_eval`
   pulls, and accept a comma-separated list;
5. partition the summary into `E1_never_measured_13` and `anchor_sat12_12`, with E1 computed only
   over the former.

### 9.4 Commands (do not run before §9.3 lands)

```bash
.venv/Scripts/python.exe duck_eval/r24_prep/s1_seal_audit.py           # evidence, re-runnable, read-only
.venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py --dry-run
.venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py \
    --source war_eval_v1,war_eval_v2,war_eval_v3 \
    --authorized-by "duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md §9" \
    --out runs/r24_prep/s1_threaded_replay.json
```

Cost: CPU only, ≈3–9 min (3 sources × 3,638-ish recorded actions), **$0, 0 Kaggle pushes, GPU
untouched.** Deterministic replay, no RNG draw, so a single run is the complete result and there is
no seed to vary.

---

## 10. Also sealed: what S1 may never be used for

1. It is **not** an answer to the round-18 unfalsifiability charge (§1.5 R3).
2. It is **not** evidence about the Kaggle engine — 15 of 25 sims were fitted on a different local
   build and the provenance was never recorded (§5.3).
3. It **cannot** promote or demote anything on the scored rail. It gates one thing: whether L1
   authoring hours are spent.
4. No number in it may be quoted without its across-source range (§2.4). That is the standing
   lesson of the 91.7% and it is now a rule, not an anecdote.
