# Pre-registration amendment — 2026-07-15

In response to panel round 11 (3× MAJOR-REVISION, 0 fatal; `learnings/panel/round11/`).
Amends `learnings/preregistration_2026-07-14.md`. Filed BEFORE war-eval seed 2 launches
(llm-agents N1 requirement) and before any Jul-17 gate look.

## A1. Build-rail gate rule — COMPOUND (supersedes §2's single primary)

The 2026-07-14 §2 gate statistic (paired Δlc) was chosen on power grounds; the war-eval
seed-1 screen (Δlc +0.272 p=0.0074 / Δlog1p(RHAE) −0.036 p=0.61 / LB draw flat) shows Δlc
can dissociate from the scoring currency. Registered rule for the war-v1 3-seed gate look
(earliest Jul 17, when seeds 2–3 land):

- **PASS requires BOTH:** (i) paired Δlc exact sign-flip p < 0.0125 (α unchanged), AND
  (ii) secondary Δlog1p(RHAE) mean ≥ 0 across the 3 seeds (non-regression, point estimate).
- Diagnostic (reported, non-gating): Δlc per 100 actions, and leave-one-game-out range.
- **A pass licenses:** war-v2 (ledger flags ON, same warpack base) entering scored R2
  windows per §4, and continued warpack-line development.
- **A fail on (ii) alone** (levels up, RHAE down/flat): warpack line enters
  "conversion-first" mode — no new warpack-derived scored arms beyond completing the
  pre-registered R2 A/B; development pivots to converting clears into clean clears.
- **A fail on both:** warpack build-rail line closes (LB control ledger continues to n=5
  for the §3 record; R2 A/B launch decision escalates to a full 5-reviewer panel).

## A2. Handoff green-light — banking clause replaced (was vacuous; R11 unanimous)

Old (2026-07-14 handoff): "if eval shows banking replay divergence … convene panel."
That condition could not fire (1 pass/game, 0 wins → replay structurally unreachable) and
is struck as a gate condition. New standing conditions:

- Further **war-v1 control draws** are licensed by prereg §3 accumulation alone (n≥3/n≥5),
  plus preflight ALLOW. Banking integrity is explicitly **UNVERIFIED** and carried as such
  in the ledger record (rl-planning: "on faith" — acknowledged, on the record).
- **war-v2 scored windows are BLOCKED until** an engineered local validation shows
  `replay_attempted > 0 AND replay_succeeded > 0` end-to-end with per-game score invariance
  vs the replayed trace (multi-pass config on games where warpack wins: sc25, m0r0, ar25,
  s5i5). The §7 canary must count **attempts, not only successes**, so vacuity ≠ silence.
  If replay cannot be made to fire locally under multi-pass, R2 slips and the panel convenes
  on warpack composition.

## A3. R2 ledger A/B — MDE published (methodology R11 major 1; answers its Q1)

- Assumed per-draw σ: 0.074 (frozen-fork LB draws, n=5; 95% χ² CI on σ [0.044, 0.213]).
  war-v1's own σ̂ is unknown at n=1 and will be recomputed at n=3 and n=5 (§3).
- Alternate-nightly, windows/arm k: SE(Δ) = σ√(2/k). At σ=0.074:
  k=3 → SE 0.060, **MDE(80%, α=.05, two-sided) ≈ 0.17**; k=6 → SE 0.043, MDE ≈ 0.12.
- **On the record: the LB A/B is unpowered for any plausible ledger effect** (predicted
  Δlc ≥ +0.08 ≈ +0.05–0.10 LB at full conversion; MDE 0.12–0.17). Therefore the A/B's
  DECISION endpoint is P1–P5 mechanism observables on build-rail transcripts (§5, verbatim
  there; ≥4/5 = concept validated); the LB ledgers are monitoring + long-horizon
  accumulation, interpreted only under §3 n-minimums and the §4 stopping rule (CI
  half-width < 0.10 or 6 windows/arm). If war-v1 σ̂ at n=3 exceeds 0.15, LB windows are
  downgraded to monitoring-only for BOTH arms and the stopping rule is void (no
  "no-significant-difference" reading will be quoted).
- No mid-course threshold changes; this amendment is the last edit before window 1.

## A4. Screen sensitivity (methodology R11 minor; computed today, same artifacts)

Reported in `runs/war_eval_v1/screen_report.md` addendum: Δlc with sc25 removed, with
m0r0 removed, with both removed; plain sign-test p alongside the magnitude-weighted p.
The mechanism narrative ("recovery buys stuck-game L1s at full action cost") is a
**one-seed hypothesis** pending the 3-seed look.

## A5. Daemon decision record (llm-agents N3 vs rl-planning minor — conflict resolved)

Chosen: **zero-code schedule fix** — second Task-Scheduler trigger at 20:07 EDT (00:07Z),
added 2026-07-15, alongside the existing 18:37 EDT safety-net trigger. The idempotent
`already-submitted-today` UTC-date check is left untouched: rl-planning's argument
(untested code in the path guarding a hard-capped resource) outweighs the logic-purity
fix; the 22:37Z fire remains a deliberate skip whenever the 00:07Z fire already used the
window. Residual known behavior: a manual midday submit makes both same-UTC-day fires
skip — correct, since the quota is spent. Unit test deferred; regression surface is the
schedule, not the code.

## A6. Dispositions of remaining R11 items

- Order-stats curve + per-mechanism reach table (rl-planning, dodged twice): will appear
  in `daily_brief_2026-07-16.md` §Instruments, from prereg §6 numbers (k=30→1.07,
  k=110→1.11 at σ̂; 1.46 only at CI-hi σ). Per-mechanism reach: order-stats +0.15 floor;
  warpack per-draw mean = unknown pending compound gate; R2–R5 grinder cracking = the only
  budgeted wall-closer. Owner: tomorrow's brief, non-negotiable.
- P1–P5 verbatim + thresholds: restated in the Jul-16 brief from §5 (deadline "before
  first R2 window" holds).
- Wheel-formula reconstruction (rl-planning part i): **disputed as infeasible** per §7
  (hidden set ≠ public 25; rails differ in population). The §7 cross-rail tripwire is the
  registered substitute. Recorded as a standing dispute, not silence.
- R1b leave-one-game-out jackknife on (90, cap 2): scheduled **2026-07-17** (post gate
  look; free, existing artifacts). Note: sched-v1 is killed, but the recovery module
  inherits the concentration bet — jackknife runs against the war-eval seeds instead.
- Provenance (rl-planning minor): submission IDs are in `runs/submission_log.jsonl` and
  `kaggle competitions submissions` output; LB screenshot artifacts to be attached to
  briefs going forward (starting Jul 16).
- Panel-objections disposition section: adopted as a standing brief section from Jul 16.
