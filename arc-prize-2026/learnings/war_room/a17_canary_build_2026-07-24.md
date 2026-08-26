# A17 72B-VL canary — build report (2026-07-24)

The A17 screen's canary push (scope v2 §5 row 1: 4-game FULL 7920 s window,
measures ρ_action denominator; scope v1 §5 "Jul 24 push #1"). LOCAL BUILD
ONLY — nothing pushed; the accelerator-selector check and the push are the
ORCHESTRATOR's (see §7 checklist). Binding specs: `a17_72b_screen_scope.md`
(v1), `a17_72b_screen_scope_v2.md`, `preregistration_amendment_2026-07-20_A17.md`,
`stuck_review_v2_2026-07-23.md` §3 (A23 envelope).

## What was built

| deliverable | path | status |
|---|---|---|
| `--a17-canary` builder mode | `duck_eval/warpack/build_eval_notebook.py` | done (other modes rebuild byte-identical — verified) |
| assembled canary notebook | `notebooks/a17-canary/arc3-a17-72b-canary.ipynb` | built, smoke-verified |
| kernel metadata | `notebooks/a17-canary/kernel-metadata.json` | id `canivel/arc3-a17-72b-canary`; ONLY delta vs duckwar family beyond id/title/code_file = `model_sources: ["qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1"]` |
| smoke test | `duck_eval/a17/a17_canary_smoke.py` | **56/56 PASS** |
| A23 envelope one-pager | `learnings/war_room/a17_envelope_onepager.md` | filed |

### Composition (minimal diff from the proven duckwar baseline)

Source = `notebooks/duckwar/arc3-duck-war.ipynb`; ONLY cells 2/8/12/14 differ,
each byte-equal to the builder's reconstruction (smoke S2–S6):

- **Cell 2**: eval-force line (`WARPACK_FORCE_OFFLINE_BENCH=1`) + A17 seed
  stamp + banner. Reset pin `ONLY_RESET_LEVELS=true` untouched (risk A).
- **Cell 8**: runtime rewrite of the bundle's `setup_commands.json` (the vLLM
  serve script ships inside the attached read-only taaf dataset, so it can
  only be patched at runtime): 10 exact-string anchors, each must match
  exactly once. Delivers: model path via marker finder (config.json with
  `Qwen2_5_VLForConditionalGeneration` + `quantization_config` + safetensors
  present — mount-path agnostic, refuses ambiguity), served name
  `Qwen2.5-VL-72B-Instruct-AWQ`, `max-model-len 32768`,
  `--quantization awq_marlin`, `--tool-call-parser hermes`, qwen3 reasoning
  parser + `preserve_thinking`/`enable_thinking` REMOVED,
  `LOCAL_ANALYZER_ENABLE_THINKING=false`, early `A17-CANARY gpu=` banner +
  RTX-PRO-6000 hard gate, serve-cmd persisted to `a17_vllm_cmd.json` (liveness
  restart source), and boot serve asserts: served-model identity, forced
  tool-call round-trip (hermes), real PNG through the vision tower.
- **Cell 12**: = the W0 continuation-only graft (duck + (f), NO warpack, NO
  ledger).
- **Cell 14**: 4-game screen filter (versioned ids), heartbeat + liveness
  gate, post-run ρ_action report. Everything else (reset/deadline/soft-end/
  n_passes logic) byte-identical (amendment §7.2).

## Design decisions (recorded)

1. **FAIL-LOUD policy inversion (the doc's own requirement).** Every other
   eval graft falls back to VANILLA duck on failure ("never 0"). Here vanilla
   = the 27B weights, which stay attached (kaggle_env_match keeps
   dataset_sources identical), so a silent fallback would produce a plausible-
   looking run that POISONS the ρ_action denominator. Therefore: any rewrite
   anchor mismatch, missing 72B model, wrong served-model id, failed tool-call
   round-trip, empty MM probe, or wrong GPU RAISES → kernel ERROR. A dead
   canary costs a retry slot; a silently-27B canary corrupts the sealed
   measurement. Three independent guards: rewrite vetoes (27B strings must be
   gone), model finder (refuses to return anything but the VL-AWQ dir), boot
   `/v1/models` identity assert.
2. **(f) continuation: ON — W0 composition.** Checked as instructed: war_eval
   v1–v3 did NOT carry (f); **w0_eval_s1 IS the (f) continuation run — and it
   is the run the 27B numerator (480 actions/7920 s) is frozen from** (v2 §3).
   The canary measures throughput against that numerator, so like-for-like
   demands the same harness composition: duck + (f), NO warpack. This also
   matches the amendment 2026-07-23 item-4 default. Cell 12 is therefore the
   W0 graft itself (not warpack+appended-(f)).
3. **Game drift rule** implements scope v2 §7.2 literally: 1 missing screen
   game → `A17-CANARY DRIFT ... MISSING` banner, run continues (dropped from
   both sides); ≥2 missing → FATAL (screen VOID). Exact versioned ids
   ft09-0d8bbf25 / sb26-7fbdac44 / lp85-305b61c3 / vc33-5430563c.
4. **MM evidence, two-tier.** Boot probe (real 64×64 PNG through the vision
   tower) is the HARD gate — empty reply = FATAL before any game starts. The
   post-run `MM cache hit rate` scan is corroboration: `mm_cache=NONZERO`
   banner, or WARN on all-zero (discard-grade per risk E) / not-found (format
   drift) — not fatal, because a working-but-uncached MM path must not destroy
   a completed denominator measurement; humans adjudicate on the banner.
5. **ρ_action report source** = `benchmark.json` `history` length per game
   (equals Σ`actions_per_level`, cross-printed), the same fields the frozen
   numerator was computed from. Window-drift check: any game off ~7920 s by
   >5% prints a `window_drift` VOID-flag banner (amendment §7.2).

## Hang-risk mitigation (ops #684625 + panel R19 liveness directive)

Community report: vLLM silently hangs on RTX Pro 6000 with this notebook
family after 15–20 min at ≥8–25 concurrent sessions. The canary's effective
concurrency is **4 games** (below the reported threshold, and the certified
27B 25-game runs completed clean) — but R19 is right that 72B-AWQ KV pressure
makes "below threshold" non-demonstrable, so both layers were added:

- **Heartbeat (log-only, zero harness contact), every 120 s:**
  `A17-CANARY HEARTBEAT t=<s> actions_total=<n|NA> vllm_log_bytes=<b>
  gen_tps=<x|NA> running_reqs=<r|NA> stall_s=<s> restarts=<k>`.
  actions_total is best-effort from the harness's periodic `benchmark.json`
  saves; gen_tps/running from the vLLM engine stats tail. A post-run log read
  distinguishes "hung at minute 18" from "slow but alive".
- **Liveness GATE:** progress = new engine-log bytes with gen_tps > 0, OR
  actions_total increase. ≥600 s with neither → ONE vLLM server restart from
  the persisted `a17_vllm_cmd.json` (identical cmd+env); a second stall (or a
  failed restart) → `A17-CANARY LIVENESS-FAIL t=<s> restarts=<n>` +
  `os._exit(70)` — the window dies loudly instead of burning 2.2 h silently.
- **Kill disarm window (deliberate):** the kill (not the heartbeat) is
  disarmed in the final 10 min of the 7920 s window and after it — the drain
  (`cancel_drain_timeout_s=120`) and diagnostics phases have legitimately zero
  generation activity, and a false kill there would destroy a COMPLETED run's
  artifacts (the exact asset the gate protects). A post-window stall prints
  `LIVENESS-STALL-POSTWINDOW` instead.
- **Restart caveat (open risk):** in-flight analyzer requests fail during the
  restart; the harness's retry behaviour around a mid-run server bounce is
  unproven. The restart is best-effort salvage per the directive; the FAIL
  path is the guarantee.

## Systems Q4 / R19 report items

1. **Window vs load:** the 7920 s per-game window starts at `bm.run()`
   (`max_runtime_s_per_game=7920.0`, clocked from each game's `started_at`),
   which runs AFTER cell 8 completes — i.e. after wheel install, 72B weight
   load, engine init, and all boot asserts. **Load/warmup is OUTSIDE the
   denominator**; the 27B numerator was measured under the identical
   structure, so the ρ_action ratio is load-free on both sides. (Evidence:
   w0_eval_s1 server ready 12:44:28 → all games `started_at` ≈ 12:44:57,
   every `final_wallclock_seconds` ≈ 7920.)
2. **Session concurrency:** harness solver param stays `concurrency=28`
   (byte-untouched, risk A); effective concurrent sessions = **4** (the
   filtered game list). The post-run report prints it:
   `A17-CANARY concurrency: ... effective concurrent games this run = 4`.
3. **First-party GPU device print:** `nvidia-smi --query-gpu=name` printed as
   `A17-CANARY gpu=<name>` during setup (before the server starts), plus the
   bundle's own `CUDA GPU check passed for rtx-pro-6000 x1` assert; non-RTX-
   PRO-6000 → FATAL (run VOID per scope §0).

## Smoke results — 56/56 PASS

`uv run python duck_eval/a17/a17_canary_smoke.py` → `RESULT: 56 passed, 0 failed`.
Coverage: structural byte-parity (18), cell-2 execution (3), serve-config
rewrite against the REAL bundle incl. patched-script compile + tamper/shape
FATALs (8), boot serve asserts against a scripted server incl. silent-27B /
no-tool-call / empty-MM FATALs (5), model finder incl. 27B-only-refusal and
ambiguity FATALs (3), game filter incl. drift/VOID (3), post-run report (8),
heartbeat + post-window disarm (3), liveness escalation in a subprocess with
exit-code-70 assert (3), builder idempotence (2). Regression: `--w0`,
`--sentinel --sentinel-budget 150 --no-continuation`, and default war-eval
modes rebuild **byte-identical** to the live certified dirs (none touched).

## §7 precondition checklist (scope v2)

| # | precondition | status |
|---|---|---|
| 7.1 | accelerator selector still offers RTX 6000 Pro (UI check before EVERY GPU build) | **PENDING-ORCHESTRATOR** |
| 7.2 | versioned-game-id identity (4 exact ids; 1 drop = flag, ≥2 = void) | enforced at runtime by the cell-14 filter (banners above) |
| 7.3 | serve-config smoke: tool-call round-trip + MM path | built in as boot asserts; DISCHARGED BY THE CANARY RUN itself (that is this push's purpose) |
| 7.4 | ρ_action measured on canary, null_adj frozen at it BEFORE seed-1 push | this run produces the denominator; freezing = post-run analysis step, **PENDING-ORCHESTRATOR** after the pull |
| — | the push itself (`kaggle kernels push`), quota slot (~2.5 GPU-h, protected pair) | **PENDING-ORCHESTRATOR** |

## For the orchestrator — push + post-run verification

Push (after the 7.1 UI check):

    kaggle kernels push -p notebooks/a17-canary

Post-run greps on the pulled build log (in order of diagnostic value):

    grep "A17-CANARY gpu="                        # MUST be RTX PRO 6000 (else run VOID)
    grep "A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ"   # 72B actually served
    grep "A17-CANARY setup-commands rewrite OK"   # all 10 anchors hit
    grep "A17-CANARY tool-call-roundtrip=OK"      # risk D discharged
    grep "A17-CANARY mm-image-roundtrip=OK"       # risk E boot probe
    grep "A17-CANARY games="                      # expect n=4 of 4 (any DRIFT line = flag)
    grep "A17-CANARY HEARTBEAT"                   # liveness trace (gaps >4 min = trouble)
    grep "A17-CANARY LIVENESS"                    # MUST be empty for a clean run
    grep "A17-CANARY N("                          # per-game action counts
    grep "A17-CANARY rho_action_denominator="     # THE number: rho_action = 480 / this
    grep "A17-CANARY mm_cache="                   # NONZERO expected; ZERO = discard-grade
    grep "A17-CANARY WARN"                        # window_drift / mm anomalies
    grep "generated tokens/sec" summary.txt       # tok/s diagnostic (demoted, still recorded)

Envelope verdict (A23): ρ_action = 480/denominator > 3.5 → ENVELOPE-INFEASIBLE,
self-certifying NO-GO (no panel needed). Kernel status ERROR with a FATAL
banner → fix + use the reserved retry slot (Jul 25). Kernel ERROR with exit 70
→ the hang reproduced at 4-game/72B; that finding goes to the war-room loop.

## Open risks

1. **`tool_choice` named-function forcing** (guided decoding) on vLLM 0.19 +
   hermes + VL arch is the least-proven part of the boot assert; if
   unsupported it fails LOUDLY at boot (never silently), and the fix is to
   fall back to `tool_choice:"auto"` + assert in the retry slot.
2. **awq_marlin × Qwen2_5_VL on vLLM 0.19.0** unverified on this exact combo
   (scope risk C); loud failure mode; hard blocker → panel if it errors.
3. **Mid-run restart salvage** unproven (see hang-risk section); the loud-fail
   path is the guarantee, the restart is best-effort.
4. **actions_total heartbeat field** depends on the harness's periodic
   benchmark.json saves; if absent mid-run it reads NA and liveness relies on
   the engine-log signal alone (still sufficient: hang = no new gen_tps>0
   lines).
5. **MM cache log-format drift** in a newer-model serve would degrade the
   post-run MM banner to WARN not-found; the boot probe remains the hard gate.
6. **27B weights stay attached** (env-match discipline) — the silent-fallback
   hazard this creates is exactly what the three loud-fail guards close.
