# Daily brief — 2026-07-25

## §1a Result deep-dive

### Scored window (00:07Z): frozen-fork filler = 1.05 — in-band, ledger n=11

**Draw:** `canivel/arc3-duck-repro` v3 (frozen-fork filler, eternal fallback)
scored **1.05**. Pre-registered expectation: a frozen-control draw from the
band 0.82–1.33 — met (0.6σ above the n=10 mean 0.975). No mechanism claims;
this is a control draw and ENTERS the frozen stratum per the prospective
pooling rule (amendment DRAFT §(b), pending R20): frozen n=10 → **n=11:
mean 0.982, s ≈ 0.150** (chronological append 1.05; exact recompute belongs
to the ratified machinery, not this brief). Pooled n=15 → n=16. No trend or
changepoint claim; the 07-24 MK/CUSUM verdict (no trend) stands.

### A17 72B canary v1: ERROR — root-caused to a MODEL-ATTACH DEFECT, not envelope

The headline result of the day is forensic, and it is GOOD news for the lane:

- **v1 (pushed 07-24 12:50Z) died at t≈8.5 s** with the designed loud-fail:
  `A17-CANARY FATAL: Qwen2.5-VL-72B-Instruct-AWQ not found under
  /kaggle/input`. The runtime input paths show only the three datasets + war
  kit — **the Kaggle Model was never mounted**. The fail-loud policy worked
  exactly as sealed (no silent 27B fallback; ρ_action never poisoned).
- **This is NOT an envelope NO-GO.** No weights were loaded; the A23
  serve-fit question (43 GB AWQ + vision tower + KV at max_model_len 32768 on
  96 GB) remains open and is settled by the retry, not by v1.
- **Root cause (isolated this morning with a scratch CPU probe kernel,
  `runs/model_attach_probe/`):** the Kaggle save-kernel API **silently drops
  `model_sources` pinned to version 1 of qwen-lm/qwen2.5-vl/transformers/
  72b-instruct-awq** (reproduced 3×: CLI 2.0.0 and 2.2.2, notebook and
  script, lowercase and display-case framework). Pinning **version 2**
  round-trips (server confirms
  `qwen-lm/qwen2.5-vl/Transformers/72b-instruct-awq/2` on metadata pull).
  Version diff audited via the models API: **51 files each; all 48 weight
  shards byte-size-identical; only tokenizer_config.json differs (5776 →
  5702 B)** — same artifact, config touch-up. The sealed scope's "/1" pin is
  amended by necessity to "/2" with this paragraph as the deviation record.
- **v3 pushed 07-25 with the /2 pin (metadata-only change; notebook
  byte-identical to the smoke-tested build). Model attachment verified by
  metadata round-trip; kernel RUNNING; terminal-status monitor armed.**
  Interpretation discipline unchanged: C3 error model
  (`learnings/a17_error_model.md`) — capability-existence detector, NO-GO at
  modest lift is the designed outcome, false-GO ≈ 0.
- Push accounting: canary re-push (v2, still /1 — failed same way) + v3 =
  **2/2 pipeline pushes used today**. The v2 push is charged to
  incomplete diagnosis (pushed before the probe isolated the version-pin);
  cost = one dead-canary retry, no GPU quota beyond ~10 s boots. Scratch CPU
  probe pushes (2, no GPU) logged for transparency, not counted against the
  pipeline cap.

### w0-continuation-eval (25-game, pushed 07-24 by the morning loop): COMPLETE

Mean **0.92**, 10 levels total, 5162 actions, 2h12m — consistent with the
certified W0 eval-rail band (build-time seeds ranged 0.85–0.95 mean; levels
16 on the 07-19 seed vs 10 here is within known per-seed level variance; no
new mechanism claim). Pull archived at `runs/kernel_pulls/w0_cont_eval/`.
Value: one more W0 (duck + (f), no warpack) seed for the eval-rail ledger and
fresh per-game action counts for the A17 screen-game rows (ft09 104, sb26 96,
lp85 42, vc33 41 — note these are seed-level counts, NOT a revision of the
sealed 480-action numerator, which stays frozen per scope v2 §3).

### LB context (API-verified this morning; full top-25 pulled)

KOJIMA 1.86 · Tecnod8.AI 1.61 · dense 1.45–1.61 band · **gold cutoff ≈ 1.49**
(top-13). We hold 1.33 and have slid out of the loaded top-50 (~#50–53);
erosion ~2–4 ranks/day at constant score. boristown's public 1.47 is now
seeded across the 1.44–1.47 band (7+ teams at 1.46–1.47). Strategic read
unchanged: efficiency channels price ≈ 0; only a depth event moves us.

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-25.md)

- **NEW #728934 "Claude Opus 5 achieves 30% on ARC-AGI-3"** (arcprize.org,
  Jul 24): new external SOTA on the benchmark via API at High reasoning
  effort. **IGNORE for config** (no API models on-node; Kaggle regime is
  quantized/time-limited local) but it is a second same-week datapoint that
  **capability, not harness micro-tuning, is the lever** — directional
  support for A17 and the depth lane.
- #684625 vLLM silent-hang: unchanged, no root cause. ADAPT holds (canary
  concurrency 4 < reported ≥8 threshold; heartbeat observable aboard).
- boristown 1.47: unchanged, upvotes 144→151, no new version, still top
  public artifact. Defensive posture stands (fork-diff filed 07-24;
  amendment §(i) monitored-continuation protocol awaiting R20).
- "Deep Reasoning Agent (179/183)": still unscored. WATCH.
- Host: no announcements; "3 months to go" banner consistent with ~Nov 2.

## §1c Research sweep (learnings/war_room/research_2026-07-25.md)

- **0 ADOPT, 1 ADAPT, 2 PARK.** ADAPT (low, design-only): aTTT
  (arXiv:2607.03441) — within-episode drift diagnosis (revisited states /
  re-failed actions); its in-context analogue (harness-injected prior-failed-
  action digest on deterministic cues) goes into the tr87 depth-lane doc as
  one sentence. No build.
- PARK: TTA-via-environment-interaction 2511.04847 (probe-before-act spends
  the budget the depth lane concentrates; hold for banking re-open).
- 2607.15439 turns out to be the Rodionov team's own ablation follow-up
  (already a standing exclusion): model + reasoning effort dominate harness
  ablations. Same directional read as Opus-5.
- Standing watches: schema replication still ZERO; no new 72B-VL-on-96GB
  serving datapoint — **the canary running right now is the datapoint**.

## §2 Today's plan

1. **A17 canary v3 in flight** (model attached, verified). On COMPLETE: pull,
   attach first-party device line, walk the sealed §9.1 gate boolean with the
   C3 recalibration discipline. On ERROR: post-mortem before any further
   push (0 pipeline pushes remain today).
2. **Panel R20 (reduced bench: rl-planning, methodology, systems; prior-dir
   round19):** ratify/amend `learnings/preregistration_amendment_2026-07-24_DRAFT.md`
   §(a)–(i) + the two same-day deviations (model-version pin /1→/2; v2
   diagnostic push). R19's non-question objections all have dispositions on
   file (`learnings/panel/r19_response_memo.md`).
3. **Queue:** frozen-fork filler re-armed (head verified, pending n=1).
   Daemon 18:37/20:07 untouched.
4. **tr87 slot:** no pushes left today; depth-lane confirmation build staged
   for tomorrow slot 1 unless R20 reprioritizes (A17 GO/NO-GO may arrive
   first and §(f) reallocation would then govern).

## Open questions (for R20)

1. **Ratify amendment §(a)–(i)?** All numbers derive from the single
   t-predictive model (`runs/r19_hygiene/`); Gaussian formally retired.
2. **Model-pin deviation:** accept the /1→/2 pin on the evidence above
   (48/48 weight shards size-identical), or demand a stronger equivalence
   artifact (e.g. sha256 of shards — NOT obtainable without a 43 GB download;
   we argue size+name+card identity suffices for a canary whose GO output is
   re-certified at promotion anyway)?
3. **boristown adoption timing:** §(i) monitored-continuation is written; the
   filler-replacement push (with hygiene graft) is $0 and low-risk — schedule
   tomorrow alongside tr87, or hold until A17 outcome to avoid confounding
   the changepoint monitor's first 5 post-gate draws?
4. **Exploration draw 2/12:** entry bar is now §(c) (aggregated prior +
   positive right-tail evidence). No candidate arm currently clears it
   (sentinel shelved; war-v4 waits on A17 GO). Confirm: no exploration draw
   until a §(c)-clearing entry case exists — filler rides.
