# A30 — UNTRIED-SET FIREABILITY GATE (build the instrument, run it, report)

## WHY (read this first; it constrains the whole task)
We have never built the STAGNATION SUPERVISOR, and our own data says it is the
largest single gap: 88% of each game's clock elapses AFTER its last level clear;
45.2% of actions are immediate repeats; `hard_noop_guard` fired 0 times in 5,255
actions; 675/675 games died on the 7920s clock.

What the supervisor has never had is a REDIRECT TARGET — something to do instead
of stalling. Competitor `thtennant/arc3-duck-v28` shipped a graft (`untried`) whose
offline sweep claims: 91% of declared actions move the board, 69% on ONE press at
level opening, and yet only 41% of his archived game-passes ever pressed the full
declared set.

CAMPAIGN RULE `feedback_verify_treatment_can_fire`: **prove an arm's TREATMENT CAN
FIRE on OUR rail before building it.** Banking died because it needed a win and we
had 0 wins in 470 recorded game-runs. Do not repeat that.

**Your job is the GATE, not the supervisor.** Answer, on OUR OWN archived traces:
does a non-empty UNTRIED set exist at the moments a supervisor would fire?

## THE DATA (all local, all free, zero GPU, zero Kaggle)
Root: `F:\kaggle\arc-prize-2026`
- 662 files: `runs/kernel_pulls/*/artifacts/*_events.jsonl` (one JSON object per line).
- Event keys seen: `board`, `board_ascii`, `score`, `state`, `level`, `run_status`,
  `type`, `title`, `action_num`, `analysis_step`, `transcript`, and on some rows
  `action_display`, `action_name`, `board_changed`, `reward`, `level_completed`,
  `game_over`, `run_complete`.
- **DECLARED ACTIONS ARE IN THE TRANSCRIPT.** The user prompt of each turn contains a
  line of the exact form `Valid actions right now: MOUSE.` (comma-separated list;
  values seen include `MOUSE`, `ACTION1`..`ACTION7`, `RESET`). Parse it per turn.
- **PRESSED ACTIONS** are in `action_display` / `action_name` rows, and also appear in
  the transcript header `--- analysis_step=N | action=M | ... ---`.
- `runs/kernel_pulls/` subdirs are ARMS (a17_canary_v3, q38_v1, p1_v1, ...). Keep the
  arm and the game id (from the filename, e.g. `ft09-0d8bbf25_p0_events.jsonl`)
  as grouping keys.

`scripts/affordance_audit.py` already reads this exact layout — READ IT FIRST and
reuse its file-walking and its taxonomy discipline. Do not reinvent the walker.

## DELIVERABLE 1 — the instrument
Write `scripts/untried_probe.py` (CPU-only, artifact-reading only, idempotent,
`--json out.json`, `--pull <dir>` to scope to one arm, no writes outside `runs/`).
Match the house style of `scripts/affordance_audit.py`: a docstring that states WHY
it exists and what previously-paid failure it prevents, and a verdict taxonomy.

## DELIVERABLE 2 — the measurements (this is the actual gate)
Compute and report, per arm and pooled:
1. **Declared-set recoverability.** In what % of turns is a `Valid actions right now:`
   line present and parseable? If it is absent for an arm, say so — an unmeasurable
   arm must be reported as UNMEASURED, never silently dropped. (`feedback_audit_the_instrument`)
2. **Declared-set stability.** Is the declared list constant within a level? Within a
   game? Tennant claims 25/25 identical on every level. Report OUR rate.
3. **Coverage.** Per game-pass: fraction of passes that ever pressed EVERY action in
   the declared set. Tennant's archive: 41%. Report ours with n.
4. **★ THE GATE — untried-at-stagnation.** Define a stagnation window as a run of
   >= K consecutive actions with no `level_completed` and no score increase
   (report for K = 10, 25, 50). At the START of each such window, compute the set
   {declared actions} minus {actions pressed so far in this game}. Report:
   - fraction of stagnation windows where that set is NON-EMPTY  <- **the fireability rate**
   - its size distribution
   - how many actions elapsed before the agent eventually pressed one (if ever)
5. **The named case.** Tennant reports a control that moved the board 40/40 and was
   pressed ZERO times in 12 passes, while a dead one was pressed in 10. Look for the
   same shape in ours: per (game, action), presses vs board_changed rate. Name the
   worst offenders explicitly.

## HONESTY REQUIREMENTS (non-negotiable — these have cost us real slots)
- We only have `board_changed` on SOME rows. If you cannot measure liveness for an
  arm, say UNMEASURED. Do NOT infer liveness from a proxy and present it as measured.
- Distinguish MOUSE (a parameterised action with coordinates) from the discrete
  ACTIONn set. "Untried MOUSE" is almost certainly meaningless; state how you handled it.
- Report n for every rate. A rate over <10 passes is not a rate; label it.
- If the gate FAILS (untried-set is empty at stagnation, i.e. the agent already presses
  everything), SAY SO PLAINLY AND RECOMMEND KILLING THE ARM. A clean kill today is
  worth more than a build we cannot read. This is the single most valuable outcome
  you can produce if it is what the data says.
- Do not tune anything. Do not touch the submission queue. Do not push a kernel.

## DELIVERABLE 3 — the write-up
`runs/untried_gate_0829/RESULT.md`: a sealed verdict of the form
FIRES (rate, n) / DOES-NOT-FIRE (rate, n) / UNMEASURABLE (why), plus the tables, plus
one paragraph on what the supervisor's redirect rule should be IF it fires.
Also emit `runs/untried_gate_0829/untried_gate.json` with the raw numbers.

Work in F:\kaggle\arc-prize-2026. Run the probe yourself and report REAL numbers.
