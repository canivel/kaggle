# Daily Brief — 2026-07-28 (user-ordered full evaluation cycle)

Panel note: yesterday's round ran as **round 21** (learnings/panel/round21/,
5/5 MAJOR-REVISION, directives synthesized 07-27). Today's cycle therefore runs
as **round 22** with round 21 as prior — re-running "--round 21" would have
overwritten the existing round-21 record.

## 1. A17 v5 boot canary — VERDICT: PASS (dataset-weights route ALIVE)

Pull: `runs/kernel_pulls/a17_v5/` (kernel `canivel/arc3-a17-72b-canary`,
status COMPLETE). All greps from the 07-27 build memo hit, evidence lines
(kernel log timestamps in seconds):

- `A17-CANARY model_path=/kaggle/input/qwen25-vl-72b-awq` (t=8.09/8.30) — THE
  dataset-route verdict line: weights found under the DATASET mount.
- `A17-CANARY gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition` (t=8.12).
- `A17-CANARY setup-commands rewrite OK (10 anchors replaced; loud-fail mode,
  no 27B fallback)` (t=7.71).
- `A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ` (t=344.9) — 72B actually
  served; boot-to-serve ≈ 337 s on the dataset mount.
- `A17-CANARY tool-call-roundtrip=OK parser=hermes name=submit_action
  args={"action": "submit_action", "x": 3, "y": 7}` (t=345.8) — risk D
  discharged on the dataset route.
- `A17-CANARY mm-image-roundtrip=OK reply='Yellow'` (t=345.9) — risk E boot
  probe green; vision path intact.
- `fenced-recovery v1: graft applied` (t=347.0); `fenced-recovery v1 hits=1`
  (t=354.8) — adapter live, ≥1 markup-fenced tool call recovered.
- `A17-CANARY games=ft09-0d8bbf25,lp85-305b61c3,sb26-7fbdac44,vc33-5430563c
  (n=4 of 4)`; 12 HEARTBEAT lines; no runtime `A17-CANARY FATAL` (the FATAL
  strings in the log are source-code echoes at t≈7.7 s, not raises).
- 1500 s slice raw counts (MEASUREMENT ONLY, no interpretation at k=1 per
  `learnings/a17_error_model.md`): N(ft09)=2, N(sb26)=1, N(lp85)=0, N(vc33)=2,
  lc=0 across all four.

## 2. A17 v6 full-window bench — PUSHED (today's push slot 1 of 2)

- Built by `duck_eval/a17/build_v6_full_window.py` (notebook was already staged
  v6 locally; idempotence guard confirmed). Smoke
  `duck_eval/a17/a17_v6_smoke.py`: **56/56 PASS** (window 7920 restored,
  budget-derived soft_end restored verbatim, cell-8 serve graft untouched,
  fenced-recovery composition intact, metadata = duckwar family + weights
  dataset, model-finder replay vs the real 43.0 GB / 11-shard local snapshot:
  exactly 1 hit, 27B decoy refused).
- Pushed: **kernel version 6** of `canivel/arc3-a17-72b-canary`. Pull-back
  verification: served `kernel-metadata.json` has `model_sources: []` and
  `dataset_sources` includes `canivel/qwen25-vl-72b-awq`; pulled notebook
  carries `mode=throughput-canary-v6-dataset-weights` + `A17_WINDOW_S = 7920`.
- Regime: free Kaggle kernel BUILD (zero-budget rule) — consumes NO scored
  submission slot. Runs under the 07-26 v4 prereg (G1 recovery ≥ 0.95, G2
  ≥ 100 executed actions, G3 cadence measurement, G4 no capability
  interpretation) and delivers the ρ_action denominator (480 / Σ N₇₂B).
  Output is MEASUREMENT ONLY; numbers go to the sealed walk + Sunday panel.
- R21 named-condition status, stated plainly: v6 fired mid-week on a boot-PASS
  under an explicit user order (the panel is advisory under the 07-27
  restructure). NC-3 (hard resource abort): the kernel session cap + stall-kill
  + zero-action-abort machinery bound the burn to one session (~2.5 GPU-h
  planned). NC-4 (≥200-replay offline parse study) and NC-5 (numeric ρ_action
  kill threshold) are NOT yet discharged — flagged as open question 3 below;
  no scored draw is being consumed by v6, which is the asset NC-4 protects.

## 3. Week's scored results vs verified posterior (ledger refreshed TODAY)

`runs/lb_ground_truth.md` refreshed 2026-07-28 from live API (submissions +
full leaderboard CSV 11:24Z). All numbers below come from that file.

- New overnight draw: **0.90** frozen filler (07-28 00:07Z, COMPLETE).
- Frozen ledger now **n=14, mean 0.9686, s ≈ 0.1384** (was 13/0.974/0.143).
  Week's draws vs posterior (z vs prior n=13 stats): 1.05 (07-25, z ≈ +0.53) ·
  0.84 (07-26, z ≈ −0.94) · 1.02 (07-27, z ≈ +0.32) · 0.90 (07-28, z ≈ −0.53)
  — all interior to band 0.82–1.33, alternating around the mean; no monotone
  run, no low-streak, MK/CUSUM no-trend verdict (07-24) stands. s has
  tightened 3 refreshes in a row (0.148 → 0.143 → 0.138).
- Sentinel arm: n=1 (0.71, 07-24, harm-paused). Under the honest t-predictive
  that draw is one-sided p ≈ 0.07 — suggestive, not significant; C2 forbids
  any single-draw claim and none is made. Status: SHELVED by
  `learnings/war_room/sentinel_disposition_2026-07-24.md` ("certified
  observable, no lift channel"; eval-rail evidence load-bearing, scored draw
  consistent-with only). R21 directive #4 (3/5) calls the n=1 shelving
  statistically indefensible and wants a pre-registered un-shelve rule
  (date for draw #2, n ≥ 4–5 per disposition, sequential stopping boundary).
  Disposition is OPEN — question 2 below.
- LB context (same refresh): our best 1.33, rank #51 (47 strictly above, 6
  tied); leader KOJIMA 1.86; gold cutoff 1.49 (#13–14 = 1.49, #15 = 1.48);
  boristown fork anchor 1.47 unchanged. P(single frozen draw ≥ 1.49) remains
  ≈ 2×10⁻⁴ — filler holds rank, it does not climb (R21 directive #5 stands).

## 4. Open questions for round 22

1. **A17 v6 result handling:** v6 is in flight (full 7920 s window, seed 1).
   Before its numbers are read: does the panel ratify a numeric ρ_action kill
   threshold NOW (R21 NC-5 wants Y stated pre-observation: ρ_action < Y ⇒ 72B
   route dead, slots revert to frozen/gated-A/B)? Propose the panel name Y and
   the expected-LB mapping (R21 directive #2 arithmetic) before the sealed
   walk reads v6.
2. **Sentinel arm disposition:** shelved-at-n=1 vs R21 directive #4. Options:
   (a) pre-register un-shelve rule + date for draw #2 (e.g., after v6 lands),
   n ≥ 4–5, sequential stop (re-shelve only on 2 consecutive < 0.80 or mean of
   first 3 < 0.80); (b) uphold the disposition memo (eval-rail evidence
   load-bearing, doctrinal zero-upside) and spend those slots on the
   boristown readiness-gate A/B instead. Panel to pick one, with error rates.
3. **Boristown vLLM readiness-gate A/B (R21 directive #1, 5/5):** still
   unscheduled. If ratified, it competes with sentinel draw #2 for the same
   filler slots — panel to rank the two and set the pre-registration (n=3–5
   gated vs frozen, one-sided at the 1.47-anchor-implied effect size).
4. **EWM Stage-1 (due Aug 4, r16 §9 window Jul 28–Aug 3):** still BLOCKED by
   the latent-state audit (r16 §10, 0.99 bar sealed) + §9.2 cheap measurement.
   Six days remain. Does the panel re-affirm the gate-then-window sequence, or
   re-price given the A17 rail now occupies the build slots through ~Aug 3?

## 5. Discussions sweep (1b) — 2026-07-28

Method: chrome-devtools MCP, discussion list sorted by recent-comments. Window =
posts/comments NEW since 2026-07-27. (The MCP browser had orphaned itself on a
stale profile lock from 07-27; cleared it — killed PID 35336 tree + removed
`chrome-profile/lockfile` — then relaunched cleanly.)

**5 threads had activity since 07-27.** One-line verdicts:

1. **Three clarifications on final scoring mechanics** — Ahmed Mobasher, posted
   ~21h ago, host (Greg Kamradt) reply ~17h ago (07-27). **ADOPT (as fact, not
   code).** Three official answers that pin our end-game model: (a) **private
   scores are computed at each submission's original run time and are NOT
   re-run** at final-2 selection — so a good frozen/A17 draw, once scored, is
   banked and cannot regress at the deadline; (b) **each scored run plays BOTH
   datasets (public + private hidden set); only the 50% public tasks feed the
   public LB** — one notebook run sees the full hidden set, so private ≠ a
   separate rerun; (c) **wall-clock is exactly 9h for v3** (the "under 12 hours"
   in one official page is an error Greg said "we should switch"). Confirms our
   A17 8–9h kernel-window planning and removes final-selection rerun risk.
   URL: /discussion/729985.
2. **Tufa Labs' Winning Solution for ARC-AGI-3 Milestone 1** — InfiniteCreativity
   (post a month old), new comment 3h ago by Mustang Liu. **IGNORE.** This is the
   Duck-harness writeup our frozen fork already descends from; the fresh comment
   adds nothing. (Re-confirmed base = Qwen 3.6 27B FP8, 64k cap / 32k eviction,
   4x-upscaled image + segmentation tool, world-model note, UNDO withheld —
   all already in our fork.)
3. **Is 100% Accuracy Realistic With the Available Compute?** — OverfitOracle
   (post 6d old), new comment ~1d ago (Doruk Doğrular). **IGNORE.** Speculation
   on VRAM/model ceilings; one unverified claim that "Kimi K3 hit 70% on public
   data" from a rank-184 competitor, no artifact. No adoptable technique.
4. **A clarification for the input that enters the agent…** — Maren Sajdaras
   (post 6d old, −4 votes), new comment ~1d ago (Son Pham). **IGNORE.** Answer
   just restates that the 64×64 grid is the frame's numeric representation.
5. **Clarification on deadline for milestone prizes** (pinned) — María Cruz,
   new comment 3h ago (KostasMouratidis). **IGNORE.** Competitor procedural Q
   about the Sept-30 open-source deadline; no host answer, no plan impact. (For
   the record: Milestone-2 = Sept 30, ranked on **public** position.)

Threads checked and found NOT new (last activity ≥ 3d, pre-07-27, no change):
"Claude Opus 5 achieves 30%" (3d), "schema-harness 99%" (6d), "Constraint
Before Control" (6d), "500 Submissions Analyzed" (10d) — nothing to re-open.

**Net:** 1 ADOPT (scoring-mechanics facts: no final rerun, both datasets per
run, 9h window). No new public notebook/fork scoring >1.33; no harness/API
change; no model-serving trick for the 8h window. Plan unchanged.

## 6. Research sweep (1c) — 2026-07-28

Window: papers NEW or newly-surfaced since ~2026-07-25. Dedup checked against
07-24/25/26/27 briefs (2607.08716, 2607.13591, 2607.09493, 2607.20972,
2607.07196, 2607.08964, 2607.03441 aTTT, 2607.15439 Rodionov-ablation,
2607.03441 already covered — NOT re-reported). arXiv listings are running
thin in-window; the freshest genuinely-new on-topic item is 2607.18754
(Jul 21). Constraint frame for verdicts: zero-cloud-budget, Kaggle-kernel-only,
frozen-fork filler + A17 72B bench = the single build priority until Aug 3.

Findings (6):

1. **arXiv 2607.18754 — AgentDebugX (Jul 21, newest in-window).**
   https://arxiv.org/abs/2607.18754 — Closed-loop Detect→Attribute→Recover→Rerun
   toolkit for LLM-agent failures; core insight "the step where an error
   surfaces is often not the one that caused it."
   **ADAPT (low).** Matches our exact pain: A17 tool-call/fenced-recovery
   failures surface downstream of the true cause. We already log
   `fenced-recovery v1 hits=N` + per-game N counts; the Attribute stage
   (root-cause vs surface step) is a cheap post-hoc lens for the ≥200-replay
   offline parse study (R21 NC-4). No runtime adoption — offline log-analysis
   only, so it costs zero build slots.

2. **arXiv 2607.05775 — "Beyond the Leaderboard": synthesis of tool-use,
   planning & reasoning failures (Jul 7, newly surfaced this sweep).**
   https://arxiv.org/abs/2607.05775 — Six failure clusters; cluster (1) =
   "tool invocation and parameter-level errors"; failures compound
   nonlinearly with task length.
   **ADAPT (low).** Gives a named taxonomy to bin our fenced/malformed
   tool-call recoveries against, and the "nonlinear-with-length" claim is a
   prior for why long ARC episodes (7920 s A17 window) degrade. Use as the
   coding scheme for the NC-4 parse study; nothing to run.

3. **arXiv 2607.12227 — "Rethinking the Evaluation of Harness Evolution for
   Agents" (Jul 14, newly surfaced).**
   https://arxiv.org/abs/2607.12227 — Automatic harness evolution does NOT
   consistently beat simple test-time scaling and generalizes poorly (harness
   tuned on public tests overfits).
   **ADOPT (as caution).** Directly de-risks any temptation to auto-evolve the
   A17 harness against Kaggle-visible games; it argues our frozen-fork +
   fixed-harness discipline is the correct posture and that harness "cleverness"
   overfits — same lesson as our simplicity-wins memo. Reinforces holding the
   plan, not changing it.

4. **arXiv 2607.08124 — TTHE: Test-Time Harness Evolution (Jul 9, surfaced).**
   https://arxiv.org/abs/2607.08124 — Treats the executable harness as the
   test-time-adaptation state; refines candidate harnesses from execution
   traces, no gold labels or weight updates.
   **IGNORE (park).** Paired counterpoint to (3): appealing in principle
   (label-free, weight-free) but (3)'s negative result + our zero-budget /
   single-build-priority frame means we do not open a harness-search lane
   before Aug 3. Revisit only if A17 ρ_action verdict reopens the harness axis.

5. **arXiv 2607.08233 — ZendoWorld: active visual concept induction (Jul 9,
   surfaced; VLM-on-grid, our modality).**
   https://arxiv.org/abs/2607.08233 — 22-game grounded visual rule-discovery
   testbed; headline: "VLM-based agents propose near-uninformative experiments,
   failing to actively reduce hypothesis uncertainty"; perception vs induction
   are distinct bottlenecks.
   **ADAPT (low-med).** Closest external evidence to our A17 exploration
   problem: a multimodal agent (like our Qwen2.5-VL-72B-AWQ) that observes
   grids but explores un-informatively. Predicts our N(game) action counts may
   be high while information-gain stays low — a reason ρ_action can look alive
   yet not convert to score. Feeds directly into naming the R21 NC-5 ρ_action
   kill threshold Y (open question 1): action-count alone is a weak proxy;
   consider an info-gain caveat in the verdict language. No code.

6. **ARC Prize blog / leaderboard (non-arXiv, live context).**
   https://arcprize.org/blog — Opus 5 (High) reported as top ARC-AGI-3 model at
   **30.2%** (Jul 24), clearing 5 previously-unbeaten Public Demo envs; public
   snapshot still shows GPT-5.6 Sol-family single-digit on the hidden split.
   **IGNORE (context only).** Frontier closed-model result on a different
   (hosted) track; no transfer to our Kaggle-kernel 72B-AWQ build. Logged so
   the panel has the current ceiling when pricing the A17 rail.

Net: 6 findings, all low-to-medium. **Nothing new forces a plan change** — the
build priority stays A17 72B bench + frozen-fork filler through Aug 3. Two items
earn quiet ADOPT/ADAPT that cost zero build slots: 2607.12227 (ADOPT-as-caution:
do NOT auto-evolve the harness against visible games — validates frozen posture)
and 2607.08233 + 2607.18754 + 2607.05775 (ADAPT-low: an info-gain caveat for the
NC-5 ρ_action threshold, and a failure taxonomy/attribution lens for the
offline NC-4 replay parse study). No runtime, no cloud spend.

## 7. Panel R22 outcome + threshold seal (appended 08:40)

- R22: **5/5 MAJOR-REVISION** (scores 5,4,5,5,5; 17 MAJOR, 0 FATAL).
  Synthesis: `learnings/panel/round22/_directives.md`. Top directives:
  D1 publish v5-slice projection (done), D2 schedule boristown readiness-gate
  A/B this week (5/5), D3 authors name Y (done), D4 no kill off k=1 (adopted
  as branch B2), D5 hash-commit before v6 COMPLETE (done).
- **Threshold seal: `learnings/a17_threshold_commit_2026-07-28.md`, commit
  `4ecf49a` @ 2026-07-28T08:29:22-04:00, v6 status RUNNING at seal time.**
  Governing kill line = the 07-27 seal ρ_action > 3.5 ⇔ ΣN₇₂B < 138 (panel's
  looser G2-collapse value rejected as post-hoc). v5-slice projection
  ΣN₇₂B ≈ 26–33 published; honest prior = v6 FAILs the line; branches
  B1 / B2→B2a/B2b sealed (kill requires k=2 concordant seeds per D4).
- N-definition verified at source (`build_eval_notebook.py:651`):
  N = `len(history)` = executed actions, same units as the frozen 480 —
  the reviewers' doom projection is unit-correct.
- Sentinel disposition (open question 2): adopt pre-registered un-shelve rule
  (draw #2 after v6 lands, n≥4–5, re-shelve on 2 consecutive <0.80 or
  mean-of-3 <0.80, ≈6–7% false-kill), queued BEHIND the boristown A/B;
  ratification at Sunday panel.
