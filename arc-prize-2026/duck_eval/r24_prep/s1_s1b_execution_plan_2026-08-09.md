# S1 / S1b execution plan — implementation scoping, 2026-08-09

**Status: SCOPING ONLY. Nothing here has been executed.** Both experiments are staged and
dry-run-verified; neither has been run. No Kaggle push, no cloud spend, no existing campaign file
modified. New files are confined to `duck_eval/r24_prep/`.

Scopes rows **S1** and **S1b** of `learnings/war_room/r24_successor_lane_proposal_2026-08-08.md`
§4 (read with §5.4 and §6), so that both can fire the moment R24 authorises them.

**Headline: both are RUNNABLE today.** Every asset exists on disk and imports. Three findings
below change what the panel is authorising, and one of them (§1.4, the coverage channel) needs a
ruling before S1's output is evidence-grade rather than advisory.

---

## 0. Asset inventory — the load-bearing output

Every path below was verified by direct read/import on 2026-08-09.

| # | Asset the proposal assumes | Actual path | Exists? | Note |
|---|---|---|---|---|
| 1 | "the 24 existing `exec_wm/` sims" | `exec_wm/sims/<gid>_sim.py` | **YES — 25, not 24** | 25 active sims (v1/v2 variants excluded). `exec_wm/scale_summary.md` validates **24**; **bp35 has a sim + observations but was never scale-validated** |
| 2 | sim validation harness | `exec_wm/validate_sim.py` | YES | held-out tuples only; not on-trajectory |
| 3 | held-out observation tuples | `exec_wm/observations/<gid>.json` (+ `.summary.json`) | YES | 25 games × 200 tuples |
| 4 | recorded traces (per-action settled frames) | `runs/kernel_pulls/war_eval_v1/artifacts/<gid>-<hash>_p0_events.jsonl` | **YES — 25/25** | plus `war_eval_v2`, `war_eval_v3`, `runs/gpt56_probe/experiment_full` |
| 5 | on-trajectory replay harness | `scripts/ewm_replay_dryrun.py` + `scripts/ewm_events.py` | YES | **wrong protocol for S1** — see §1.1 |
| 6 | prior on-trajectory result | `runs/ewm_dryrun/report.md`, `raw.json`, 4 `.log` streams | YES | source of the sp80/lp85/sb26 collapse numbers |
| 7 | event schema | `duck_eval/ewm_exec/EVENT_SCHEMA.md` | YES | `duck_eval/ewm_exec/` contains **only** this file — no sims live there |
| 8 | "25 recorded traces" for banking | `runs/kernel_pulls/war_eval_v1/benchmark.json` → `game_runs[].history` | **YES — 25/25, 3,638 actions** | matches the §1.1 control figure exactly |
| 9 | `prune_trace` code path | `duck_eval/warpack/warpack_patch.py` L164 (+ byte-identical copy in `_kaggle_dataset/`) | YES | **no env kill-switch** — see §2.2 |
| 10 | `_bank` replay path | `duck_eval/warpack/warpack_patch.py` L186+ | YES | divergence semantics reproduced verbatim in the new runner |
| 11 | prior bank fire validation | `duck_eval/warpack/bank_fire_validation.py` → `runs/war_eval_v1/bank_fire_validation.json` | YES | 4 games only (ar25, sc25, m0r0, s5i5) |
| 12 | prune-vs-replay diagnosis | `duck_eval/warpack/prune_replay_diag.py` → `runs/war_eval_v1/prune_replay_diag.json` | YES | 4 games; the root-cause artifact |
| 13 | 25-game determinism audit | `duck_eval/warpack/determinism_audit_25.py` → `runs/war_eval_v1/determinism_audit_25.{json,md}` | YES | **25/25 DETERMINISTIC, 0 divergent, 0 untestable, 13.5 s total** |
| 14 | 11-game prefix-splice-safe set | `learnings/war_room/grinder_design_R17_sealing.md` L319–324 (= `learnings/panel/r17_circulation.md`) | YES | see §2.3 — the sealed number is **10**, not 11 |
| 15 | `runs/kernel_pulls/war_eval_v1/` | as named | **YES** | `benchmark.json`, `artifacts/` (50 files), `intermediate_states.pkl`, logs, `summary.txt` |
| 16 | local offline engines | `kaggle-data/environment_files/` (25 dirs) + `arcengine`, `arc_agi` in `.venv` | YES | import-verified |
| 17 | framework bundle | `duck_eval/taaf_bundle/src/{ARC3-Inference,tufa-arc-agi-framework}` | YES | `taaf.game_api.GameAPI` imports |
| 18 | prior carrier set ("≈4 games") | `learnings/stuck_review_v2_2026-07-23.md` L13 | YES | `{tn36, tu93, ls20, ft09-L1}` |

**Nothing required by S1 or S1b is missing.**

---

## 1. S1 — L0 state-threaded, abstention-aware sim re-verification

### 1.1 What must be newly written, and why

The existing harness `scripts/ewm_replay_dryrun.py` cannot answer S1 as written. Four gaps:

1. **It is teacher-forced, not state-threaded.** Every prediction restarts from the *recorded*
   pre-action frame (`boards[id(ev)] = prev_board`, L146–155), so errors never cascade. That is
   precisely the protocol S1 exists to replace.
2. **It covers 12 sims, not 24/25.** `SAT12` (L57–61) hard-filters to the saturated set;
   `if gid not in SAT12: continue` (L256).
3. **It never restarts at level frame 0** and **never resets sim module state.** `g50t`
   (`reset_state`), `re86` (`reset_phase`) and `tr87` (`reset_step_parity`) carry module-level
   hidden counters. The old harness loads each module once per game and never resets it, so those
   three sims were measured with counters desynced from the first level/RESET boundary onward.
   **This is a real defect in the 2026-07-18 numbers**, not merely a protocol difference.
4. **It has no coverage/abstention channel at all.**

New file: **`duck_eval/r24_prep/s1_threaded_replay.py`** (thin wrapper — imports only the sims'
public `simulate()` contract and re-uses `ewm_replay_dryrun`'s trace-parsing conventions; touches
no existing file).

### 1.2 Commands

```bash
# asset + import verification, runs no experiment  (ALREADY EXECUTED, 25/25 OK)
.venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py --dry-run

# the S1 run, primary source (ledger-OFF seed-1 control)
.venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py \
    --source war_eval_v1 \
    --authorized-by "R24 minute <ref>" \
    --out runs/r24_prep/s1_threaded_replay.json

# robustness: all four recorded sources in one artifact
.venv/Scripts/python.exe duck_eval/r24_prep/s1_threaded_replay.py \
    --source all --authorized-by "R24 minute <ref>" \
    --out runs/r24_prep/s1_threaded_replay_allsrc.json
```

### 1.3 Wall-clock and cost

**≈1–3 min CPU** for `--source war_eval_v1` (3,638 recorded actions × 3 `simulate()` calls each:
two independently-loaded instances for the self-diff check plus one teacher-forced A/B call, on
64×64 `uint8`). `--source all` ≈ 4× that. **No GPU. RTX 3080 idle. $0. 0 pushes.**

### 1.4 BLOCKING UNKNOWN #1 — the sims have no abstention channel

Tycho's protocol requires abstention (`UNKNOWN = -1`) with coverage reported separately. **Our
sims cannot abstain.** The contract is
`simulate(state, action_id, x, y) -> (next_state, reward_class, done)` — a concrete 64×64 grid,
every time (verified: `exec_wm/validate_sim.py` L6–8; no `UNKNOWN` token anywhere in
`exec_wm/sims/`). Coverage as literally specified is **not measurable** without L1 (the sim
interface migration), which §4 S5 gates behind S1 itself and behind the §5.3 ruling.

The runner therefore emits **three** channels and pre-commits to none:

| field | definition |
|---|---|
| `coverage_strict` | `(steps − errors) / steps` — the sim produced a well-formed 64×64 grid |
| `coverage` | `(steps − errors − identity_abstentions) / steps` |
| identity abstention | the sim returned its input frame **unchanged** while the recorded transition **did** change the board |
| `accepted_match` | matches / **committed** steps (the Tycho numerator) |
| `match_all_steps` | matches / all steps — directly comparable to the 2026-07-18 figures |

The identity-abstention proxy is defensible because several sims *document* identity-return as
their explicit "cannot tell from one frame" fallback (e.g. `g50t_sim.py` action 5: *"From a single
frame we can't know which mode fires… v2 keeps the identity default"*). But it is a **proxy, and
an inference**. **R24 must state which coverage definition the gate reads**, otherwise the number
is chosen after seeing the data and the carrier-set verdict is post-hoc.

### 1.5 BLOCKING UNKNOWN #2 — "carrier" has no numeric definition in the proposal

The gate is "carrier set must **expand beyond ~4 games**", but no threshold defines membership.
The repo's own precedent (`learnings/daily_brief_2026-07-20.md` L39) is
*"EWM ship-now carriers = ft09/tn36/tu93 (**step_acc 0.92–0.97 AND clean/resolvable**)"*, later
shrunk to `{tn36, tu93, ls20, ft09-L1}` (`learnings/stuck_review_v2_2026-07-23.md` L13).

The runner defaults to that precedent and labels it **PROPOSED — NOT SEALED**:

```
accepted_match >= 0.92  AND  coverage >= 0.50  AND  engine verdict != ALIASED-UNRESOLVED
```
(overridable via `--carrier-match-min`, `--carrier-coverage-min`,
`--allow-unresolvable-carriers`). Engine verdicts are hard-coded from
`learnings/daily_brief_2026-07-20.md` L37: 11 CLEAN / 11 ALIASED-RESOLVABLE / 3 ALIASED-UNRESOLVED
(g50t, sk48, m0r0). The emitted `verdict` field carries
`"verdict_status": "ADVISORY until R24 seals the carrier thresholds"`.

**Seal the two numbers before the run, or S1 is a measurement, not a falsifier.**

### 1.6 Correction the panel needs — the "91.7%" figure is misdescribed

§6.4 of the proposal reads *"91.7% held-out `state_exact` across 24 games"*. It is not a
`state_exact` rate. In `exec_wm/scale_summary.md` **91.7% = 22/24 games classified Class A
(≥50% state_exact)**.

> **SUPERSEDED 2026-08-10** by `duck_eval/r24_prep/s1_sealed_spec_2026-08-10.md` §6, on two counts:
> (a) this paragraph still calls the figures "held-out" — they are **`split=all`, i.e. in-sample
> fit-plus-selection** (`exec_wm/validate_sim.py:56,62-64`; `scale_summary.md:3-4`), never held out;
> (b) the mean is **81.1**, not 79.8 (24 values from `scale_summary.md:22-45`, recomputed in
> `duck_eval/r24_prep/s1_seal_audit.py`). Range 23.0 (r11l) to 100.0 (9 games) is correct.

The rhetorical force of the §6.4 objection ("high fidelity number, near-uninformative
on-trajectory") survives — nine games really do sit at 100.0 in-sample and collapse
on-trajectory — but the sentence as written is corrected in the minutes and in the proposal.

### 1.7 Prediction worth pre-registering

State threading can only ever **lower** the all-steps match rate relative to teacher forcing
(errors now cascade). The interesting quantity is therefore **not** whether the numbers drop — they
will — but (a) `mean_survival_steps` (how many threaded steps a sim survives before first
mismatch), and (b) whether `accepted_match` on *committed* steps stays high while `coverage`
falls, which is the Tycho-shaped success mode. Both are emitted per game. The genuinely new
information is on **g50t / re86 / tr87**, where the module-state reset bug (§1.1 item 3) is being
corrected for the first time — those three are the only games where the number could legitimately
go **up**.

---

## 2. S1b — offline bank re-fire with `prune_trace` disabled

### 2.1 What must be newly written

`bank_fire_validation.py` covers **4 games** and replays **pruned** traces. `prune_replay_diag.py`
covers **4 games** and diagnoses the pruning artifact. Neither runs the unpruned arm across 25.

New file: **`duck_eval/r24_prep/s1b_bank_refire_noprune.py`** — imports `TraceStep` and
`prune_trace` from `warpack_patch` **unmodified**, and reproduces `_bank`'s divergence semantics
verbatim (per-step `grid_hash` + `levels_completed` comparison; new play via ≤2 RESETs at
`ONLY_RESET_LEVELS=false` until `full_reset`).

### 2.2 Why "disable pruning" needs a wrapper rather than a flag

`prune_trace` is a module-level function called unconditionally inside `_bank`
(`warpack_patch.py` L202). The nearest env knob, `WARPACK_BANK_STRICT`, gates frame *checking*,
not pruning. **There is no `--no-prune` switch anywhere.** Rather than edit campaign code, the
runner selects the arm over the trace:

| arm | trace fed to the replay |
|---|---|
| `pruned` | `prune_trace(trace)` — the 2026-07-15 baseline that aborted at step 0 |
| `trailing_only` | drop only what follows the last level completion; keep no-ops and RESETs |
| `unpruned_full` | every recorded step up to the recorded lc, verbatim |

Three fresh plays per game, so the comparison is **paired within a single engine session**.
`trailing_only` is included because R16 explicitly flagged *"unpruned (or trailing-only-pruned)"*
as the viable class, and it is the variant banking would actually ship (you do not replay past the
banked frontier).

### 2.3 DISCREPANCY the panel must resolve — 11 games or 10?

The proposal's 11-game list (ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, **tn36**, tu93)
matches the **R17 SEALED §7 carrier set** exactly. But the same sealed document states the banking
rule as *"prefix-splice restricted to the **10 CONFIRMED CLEAN** games (**tn36 excluded by its
flag**)"* (`grinder_design_R17_sealing.md` L67–68, L158–159; tn36's flag: det 1.000 on only 31
visits, Wilson LB 0.890). So the proposal's gate set is one game wider than the sealed banking
policy. The runner reports **both** verdicts (`gate_11_verdict`, `gate_10_verdict`) and states the
provenance of each. Since tn36 "carries zero priced value (already at base) and may not enter any
splice", the discrepancy is very unlikely to change the outcome — but it should be minuted rather
than silently resolved.

### 2.4 Commands

```bash
# asset + import verification, touches no engine  (ALREADY EXECUTED, all OK)
.venv/Scripts/python.exe duck_eval/r24_prep/s1b_bank_refire_noprune.py --dry-run

# the S1b run
.venv/Scripts/python.exe duck_eval/r24_prep/s1b_bank_refire_noprune.py \
    --authorized-by "R24 minute <ref>" \
    --out runs/r24_prep/s1b_bank_refire.json
```

### 2.5 Wall-clock and cost

**≈1–2 min.** Calibration: `determinism_audit_25.py` did 2 full plays × 25 games in **13.5 s
total** on these engines. S1b does 1 trace-build pass + 3 replay arms = 4 passes over ≤ the same
3,638 actions. **No GPU. $0. 0 pushes.**

### 2.6 Coverage of the gate set, verified

Of the 11 gate games, **11/11 have `levels_completed ≥ 1` in `war_eval_v1`**, so all 11 produce a
bankable trace and the gate is fully evaluable. Across all 25: **18 bankable, 7 `NO_BANK_TRACE`**
(sk48, cn04, dc22, wa30, cd82, tr87, g50t — all recorded `lc = 0`; `_bank` never fires on them by
construction, and the runner labels them rather than counting them as passes).

### 2.7 Non-blocking caveat to state in the minutes

Local engine versions differ from the Kaggle war-eval build on some games (`sc25` local
`f9b21a2f` vs Kaggle `635fd71a`) — already flagged in `determinism_audit_25.json` as
`version_mismatch_vs_kaggle` and reproduced per game in the new runner. S1b measures the mechanism
(does unpruned replay preserve phase alignment) on the **same engines the 2026-07-15 failure was
observed on**, so the comparison is internally valid; it is not a claim about the Kaggle build.

### 2.8 Honest note on how much S1b adds

`determinism_audit_25.py` probe A **already** fed the full unpruned recorded history through 25/25
games and found **0 divergent**. S1b's marginal contribution is therefore not "does unpruned replay
survive" (largely answered) but: does it survive **through the `_bank` fire path** — new play
opened `_bank`-style, `_bank`'s per-step `grid_hash`/`lc_after` checks, `bank_max_replay_actions`
enforced — with the **paired pruned arm reproducing the step-0 abort in the same session**. That
paired A/B is what converts the root-cause claim from inference into measurement. The panel should
price S1b as **confirmation at near-zero cost**, not as an open question.

---

## 3. Output artifact schemas (evidence-grade)

Both artifacts carry a provenance header, an explicit config block recording every free parameter,
per-game rows, and a summary carrying the gate verdict **plus its sealed/unsealed status**.

### 3.1 `runs/r24_prep/s1_threaded_replay.json`

```jsonc
{
  "provenance": {
    "script": "duck_eval/r24_prep/s1_threaded_replay.py",
    "script_sha256": "...", "generated_utc": "2026-08-09T..Z",
    "git": {"commit": "...", "dirty": true},
    "python": "3.12.x", "platform": "Windows-11-...", "numpy": "...",
    "proposal": ".../r24_successor_lane_proposal_2026-08-08.md §4 S1",
    "authorized_by": "R24 minute <ref>  |  UNSEALED-R24-PENDING",
    "cost": {"usd": 0, "kaggle_pushes": 0, "gpu": "none (CPU only)"},
    "rng": "none — replay is fully deterministic; no seeds are drawn",
    "sources": {"war_eval_v1": "runs/kernel_pulls/war_eval_v1/artifacts"},
    "wallclock_s": 0.0
  },
  "config": {
    "mode": "state_threaded",
    "segment_rule": "initial | post-RESET frame | post-level-completion frame",
    "reset_module_state_at_segment_start": true,
    "teacher_forced_ab_in_same_pass": true,
    "coverage_definition": {
      "commit": "valid 64x64 grid AND not identity-while-observation-changed",
      "abstention_channel_in_sims": false,
      "note": "OPERATIONAL PROXY — must be sealed by R24"
    },
    "carrier_match_min": 0.92, "carrier_coverage_min": 0.50,
    "carrier_require_resolvable": true,
    "carrier_thresholds_status": "PROPOSED — NOT SEALED BY R24",
    "prior_carrier_set": ["ft09","ls20","tn36","tu93"],
    "gate": "carrier set must EXPAND beyond the prior 4"
  },
  "games": [{
    "game": "sp80", "source": "war_eval_v1", "trace_id": "sp80-589a99af",
    "trace_file": "runs/kernel_pulls/war_eval_v1/artifacts/sp80-589a99af_p0_events.jsonl",
    "sim_file": "exec_wm/sims/sp80_sim.py", "sim_sha256": "...",
    "sim_reset_hook": null, "sim_has_module_state": false,
    "engine_determinism_verdict": "CLEAN", "held_out_state_exact_pct": 100.0,
    "n_segments": 0, "n_steps": 0, "segment_lengths": [],
    "n_error": 0, "n_selfdiff": 0, "n_obs_changed": 0,
    "n_identity_abstain": 0, "n_committed": 0,

    "accepted_match": 0.0,            // ** S1 metric (i) ** matches / committed
    "coverage": 0.0,                  // ** S1 metric (ii) ** committed / steps
    "coverage_strict": 0.0,           //    (steps - errors) / steps

    "match_all_steps": 0.0,                  // comparable to 2026-07-18
    "teacher_forced_match_all_steps": 0.0,   // legacy protocol, same pass (A/B)
    "mean_survival_steps": 0.0, "max_survival_steps": 0,
    "done_flag_agree": "0/0",
    "per_action": {"6": {"n":0,"match_threaded":0,"match_tf":0,"abstain":0,"error":0}},
    "carrier": false, "carrier_fail_reasons": ["accepted_match ... < 0.92"]
  }],
  "summary": {
    "n_rows": 25, "primary_source": "war_eval_v1", "n_games_primary": 25,
    "carrier_set": [], "n_carriers": 0,
    "prior_carrier_set": ["ft09","ls20","tn36","tu93"], "n_prior_carriers": 4,
    "gained": [], "lost": [],
    "gate": "carrier set must EXPAND beyond ~4 games",
    "verdict": "EXPANDED | NOT_EXPANDED",
    "verdict_status": "ADVISORY until R24 seals the carrier thresholds"
  }
}
```

### 3.2 `runs/r24_prep/s1b_bank_refire.json`

```jsonc
{
  "provenance": {
    "script": "duck_eval/r24_prep/s1b_bank_refire_noprune.py", "script_sha256": "...",
    "generated_utc": "...", "git": {...}, "python": "...", "platform": "...",
    "proposal": ".../r24_successor_lane_proposal_2026-08-08.md §4 S1b",
    "authorized_by": "R24 minute <ref>  |  UNSEALED-R24-PENDING",
    "cost": {"usd": 0, "kaggle_pushes": 0, "gpu": "none (CPU only)"},
    "rng": "none — replay is fully deterministic; no seeds are drawn",
    "trace_source": "runs/kernel_pulls/war_eval_v1/benchmark.json",
    "trace_source_sha256": "...", "warpack_patch_sha256": "...",
    "engines": "kaggle-data/environment_files (local offline arcengine)",
    "wallclock_s": 0.0
  },
  "config": {
    "arms": ["pruned","trailing_only","unpruned_full"],
    "divergence_semantics": "warpack _bank verbatim: per-step grid_hash + levels_completed; new play via <=2 RESETs at ONLY_RESET_LEVELS=false",
    "bank_max_replay_actions": 1500,
    "prune_disable_method": "arm selector over the trace; warpack_patch imported unmodified",
    "splice_safe_11": ["ar25","bp35","ft09","lf52","lp85","ls20","r11l","sp80","su15","tn36","tu93"],
    "confirmed_clean_10": ["... same minus tn36 ..."],
    "gate": "step-0 frame_divergence must clear on the 11-game prefix-splice-safe set"
  },
  "games": [{
    "game": "sc25", "benchmark_game_id": "sc25-635fd71a", "local_game_id": "sc25-f9b21a2f",
    "version_mismatch_vs_kaggle": true,
    "recorded_lc": 2, "recorded_actions": 184,
    "in_splice_safe_11": false, "in_confirmed_clean_10": false,
    "status": "TESTED | NO_BANK_TRACE | UNTESTABLE | ERROR",
    "driver": "gameapi | raw_env (...)",
    "actions_fed": 0, "lc_reached_on_local_engine": 0,
    "local_engine_reproduced_recorded_lc": true,
    "arm_lengths": {"pruned":0,"trailing_only":0,"unpruned_full":0},
    "n_dropped_by_prune": 0,
    "prune_dropped_before_pruned0": {
      "recorded_index_of_pruned0": 2, "noops": 2, "board_changing": 0, "resets": 0
    },
    "arms": {
      "unpruned_full": {
        "n_replay_actions": 0,
        "outcome": "survived | aborted | bank_skip_trace | no_new_play",
        "abort_step": null,
        "abort_kind": null,                     // frame_divergence | lc_divergence | empty_frame | step_error:*
        "step0_frame_divergence": false,        // ** the S1b gate flag **
        "final_lc": 2, "reached_recorded_lc": true
      },
      "pruned": {"...": "same shape"}, "trailing_only": {"...": "same shape"}
    },
    "scorecard": {"total_plays": 4, "levels_per_play": [2,2,2,2], "actions_per_play": [184,12,180,184]},
    "wallclock_s": 0.0
  }],
  "summary": {
    "n_games": 25, "n_tested": 18, "n_no_bank_trace": 7,
    "step0_frame_divergence": {"pruned": ["sc25","m0r0"], "trailing_only": [], "unpruned_full": []},
    "survived": {"pruned": [], "trailing_only": [], "unpruned_full": []},
    "gate_11_tested": [], "gate_11_step0_divergent_unpruned": [],
    "gate_11_verdict": "CLEAR | FAIL",
    "gate_10_strict_step0_divergent_unpruned": [], "gate_10_verdict": "CLEAR | FAIL",
    "note": "gate_11 follows the R24 proposal text; gate_10 follows the R17 SEALED banking rule, which excludes tn36 from splice"
  }
}
```
(Values above are shape illustrations, not results. Nothing has been run.)

### 3.3 Seeds

Both experiments are **deterministic replays of recorded data with no RNG draw**. There is no seed
to vary and no variance to average over; `"rng": "none"` is recorded explicitly in both headers so
a reader cannot mistake a single run for an n=1 sample. (The only seeded object in the vicinity is
`determinism_audit_25.py`'s probe B, which S1b does not use.) The "seed 1 / seed 2" language of
§6.6 applies to build-rail screens (S2), **not** to S1/S1b.

---

## 4. Dry-run verification already performed

Both runners were syntax-checked (`py_compile`), `--help`-checked, and `--dry-run`-executed.
**No experiment was run.**

* `s1_threaded_replay.py --dry-run` → **25/25 sims import and expose `simulate()`; 25/25 traces
  resolved; 0 failures.** Module-state reset hooks auto-detected: `g50t → reset_state`,
  `re86 → reset_phase`, `tr87 → reset_step_parity`; the other 22 are stateless.
* `s1b_bank_refire_noprune.py --dry-run` → benchmark 25 game_runs / 3,638 actions; 25 local
  engines; `arcengine`, `arc_agi`, `taaf.game_api`, `warpack_patch.{TraceStep,prune_trace}` all
  import; 18 bankable / 7 `NO_BANK_TRACE`; **11/11 gate games bankable**.

### 4.1 Single-game runtime smokes (per `feedback_test_before_submit`)

A dry run proves assets resolve; it does not prove the runner works. Each script was therefore
executed **once, on one game, writing to a scratch path outside `runs/`**, purely to prove the code
path completes. Neither smoke is an S1/S1b result and neither is banked.

* `s1_threaded_replay.py --games ft09 --out <scratch>` → completed, exit 0, full artifact written
  with every schema field populated. **ft09 was chosen because it is already inside the prior
  4-game carrier set**, so the smoke cannot pre-empt the gate in either direction.
* `s1b_bank_refire_noprune.py --games sc25 --out <scratch>` → completed, exit 0. **sc25 was chosen
  because it is NOT in the 11-game gate set**, so the smoke cannot touch the gate. It nonetheless
  reproduced the known 2026-07-15 signature exactly — `pruned` arm **aborted at step 0**, while
  `trailing_only` and `unpruned_full` both **survived** — which is independent evidence that the
  runner's `_bank` divergence semantics are faithful. The `gate_11`/`gate_10` fields printed
  `CLEAR` **vacuously** (no gate game was in the filter); they are meaningless on a filtered run
  and must be read only from the full 25-game invocation.

Both scratch artifacts were discarded. Nothing was written under `runs/`.

---

## 5. What R24 must decide before either run counts as evidence

1. **Coverage definition for S1** (§1.4) — `coverage` (identity-abstention proxy) or
   `coverage_strict` (errors only)? The sims have no abstention channel; the proxy is an
   inference and must be sealed *before* the run.
2. **Carrier thresholds for S1** (§1.5) — proposed `accepted_match ≥ 0.92`, `coverage ≥ 0.50`,
   exclude `ALIASED-UNRESOLVED`. Without a sealed number, "expand beyond ~4" is unfalsifiable.
3. **11 vs 10 for the S1b gate** (§2.3) — proposal text vs R17 sealed banking rule. Both verdicts
   are emitted; the minutes should say which binds.
4. **Correct the 91.7% description** (§1.6) — it is a class share (22/24 games), not a
   `state_exact` rate.
5. Note for the record that there are **25 sims, not 24** (bp35 is unvalidated by
   `scale_summary.md` but present and importable), and that if bp35 is included the S1 denominator
   changes.

None of the five blocks *execution*. Items 1–3 block **interpretation**.

---

## 6. Recommended firing order once authorised

1. `s1b_bank_refire_noprune.py` first — ~1–2 min, cheapest, and its answer is close to known
   (§2.8). Bank the confirmation.
2. `s1_threaded_replay.py --source war_eval_v1` — ~1–3 min, the decisive read.
3. If and only if S1 is near the carrier boundary, `--source all` for the robustness pass; a
   carrier that appears on `war_eval_v1` but on no other recorded source is a draw artefact and
   should be reported as such rather than counted.

Total workstation cost for the entire S1 + S1b programme: **well under 10 minutes, CPU only, $0,
0 Kaggle pushes.** The GPU is not touched.
