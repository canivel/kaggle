# Daily brief — 2026-07-16

Author: daily-loop agent. Inputs: overnight LB draws #2/#3 (war-v1 ledger → n=3), war-eval
seed-2 build-rail pull + screen, discussions sweep, research sweep. Panel round 12 runs on
this document.

## 1a. Result deep-dive

### LB: war draws #2 = 1.08, #3 = 0.88 → war-v1 ledger n=3
- Ledger {0.91, 1.08, 0.88}: mean **0.957**, σ̂ **0.108**, χ² 95% CI on σ (df 2) [0.056, 0.678].
- **Amendment A3 check: σ̂ = 0.108 < 0.15 → LB windows remain live** for the R2 A/B
  (stopping rule stands). Recompute again at n=5.
- Pre-registered expectation (prereg §3): accumulation only, no gate — **met**. Draws #2/#3
  were licensed by the accumulation rule alone; banking integrity carried as UNVERIFIED on
  the record, as amended.
- 1.08 is the **highest single draw of the campaign** (previous max: frozen #4 = 1.02), and
  0.88 is comfortably inside the frozen band — but war-vs-control Welch t = 0.49, p = 0.66
  (Δmean +0.035): **descriptively indistinguishable, exactly as the MDE analysis predicted**
  (MDE ≈ 0.14 at n=5/5). No standardized claim is made at n=3; that is the design working,
  not a disappointment. Notable only as: no evidence of harm, kill line never approached.
- Mechanics note: draw #3 fired at 00:07Z via the new 20:07 EDT trigger (A5 daemon fix
  verified working end-to-end on its first night).

### Build rail: war-eval seed 2 = NULL screen; pooled 2-seed evidence weakens
Seed-2 pull `runs/kernel_pulls/war_eval_v2/`, report `runs/war_eval_v2/screen_report.md`
(validated scorer, max err 0e+00):

| screen | Δlc mean | W/L | p (sign-flip) | Δlog1p(RHAE) | p |
|---|---|---|---|---|---|
| seed 1 (07-15) | **+0.272** | 12W/5L | **0.0074** | −0.036 | 0.61 |
| seed 2 (today) | **−0.008** | 6W/11L | 0.539 | **−0.159** | 0.86 |
| pooled 2-seed | **+0.132** | 11W/6L | 0.089 | **−0.098** | 0.85 |

- **Interpretation (validated, not just the number):** seed 1's positive primary was
  plausibly a 1-seed draw from a noisy panel — 11 of 25 games flip ≥1.0 lc between seeds
  (ar25, bp35, ft09, ls20, m0r0, r11l, re86, s5i5, sp80, tn36, vc33). This is the
  variance-reconciliation lesson repeating at the per-game level.
- **Stable signal that survives both seeds:** sc25 (+1.8 both seeds — recovery cracking a
  game the null never opens), tu93 (+0.7), re86 (+0.7 s2), ka59 (+0.6 s2). The warpack
  mechanism story ("recovery buys stuck-game L1s") lives, but is narrower than seed 1
  suggested.
- **Compound gate (A1) forecast for tomorrow's look (seeds 1–3):** on pooled-2 evidence the
  gate currently fails BOTH prongs (p 0.089 > 0.0125; secondary −0.098 < 0). Seed 3
  (kernel v3, pushed 12:xx local today, RUNNING) decides. Most likely landing per A1:
  **fail on (ii) alone or on both** → conversion-first mode or line-close + 5-reviewer
  escalation. Both branches are pre-registered; no discretion needed.
- Banking canary seed 2: **zero replay events again** — vacuous (0 wins → replay
  structurally unreachable), consistent with A2. Not new evidence for or against.
- Budgets: seed 2 byte-comparable to null10 (161 vs 140 actions/run mean; same regime as
  seed 1's 146).

### Leaderboard
- #1 YUTO KOJIMA **1.86**, resubmitted overnight, **40 entries** (screenshot artifact:
  `learnings/artifacts/lb_screenshot_2026-07-16.png`). Order-stats kill-shot: even at our
  χ²-CI-**upper** σ = 0.213, E[max of 40 draws] ≈ 0.922 + 0.213×2.16 ≈ **1.38 ≪ 1.86** —
  the #1 edge is per-draw mean, not resubmission luck. Strategy implication unchanged and
  sharpened: only per-draw mean gains matter.
- Wall region thickening: 11 teams ≥ 1.44 (was ~8); 1.56 is the new "cluster above the
  wall" (Mathurin Ache, anngle, NoOneAhead). Cutoff for prize contention is drifting UP.
- Our best remains 1.02.

## 1b. Discussions sweep (new since 07-15) — verdicts
1. **"Run-to-run variance in the public score for a fixed agent"** (Alvaro Camacho, #726552)
   — two byte-identical submissions 0.20 vs 0.03; his cause: *unseeded* random ACTION6
   fallback; open question whether the eval env is seeded per run. **ADAPT**: (a) TODAY —
   seed-audit duck-harness+warpack for unseeded `random`/`numpy` fallback paths so our
   ledger draws are seeded-agent draws; (b) monitor thread for an official
   environment-seeding answer (would sharpen every ledger inference we run).
2. **"GPT-5.6 Sol sets a new SOTA on ARC-AGI-3: 7.8%"** (#726340) — **IGNORE**: frontier
   pure-model news; confirms harness engineering (our lane) dominates raw models.
3. **AGI-timeline poll** (#726367) — **IGNORE**: chit-chat.
4. **Milestone #1 thread, new Reki comment**: replica of his 0.86 notebook scored 0.00 —
   "almost certainly infra/timeout, not variance." **ADAPT**: treat 0.00 LB draws as
   censored (infra) rather than legitimate samples in ledger stats — file as a prereg
   footnote BEFORE we ever observe one in a war ledger (we have not yet).
5. No YUTO KOJIMA footprint; no new public notebooks; host re-links the three open-sourced
   winners → 1.44 wall keeps feeding.

## 1c. Research sweep — verdicts
(Additional to `learnings/war_room/research_2026-07-16.md` from the early pass.)
1. **AutoMem** (arXiv:2607.01224) — post-episode LLM trajectory review decides what enters
   memory; 2–4× on Crafter/MiniHack/NetHack with a 32B open model. **ADAPT** (the one real
   find): inference-side pattern = LLM reviews its own trajectory and curates what gets
   *banked* — a direct upgrade path from warpack's raw-trace banking, targets L2+
   progression. War-v3 material, behind the compound gate.
2. **ECHO** (arXiv:2606.31650) — per-turn compressed indexed records + selective context
   reconstruction; fewer turns for higher score. **ADAPT (low priority)**: prompt-side
   context compression targets action-efficiency (RHAE conversion). Behind the gate.
3. AGI Maze (2607.00627), TTT-memory audit (2607.00368), MineExplorer (2605.30931) —
   **IGNORE** (no adoptable mechanism / confirms existing skepticism).
4. ARC-AGI-3 citations: nothing new in 48 h.

## Instruments (standing section; closes the twice-dodged rl-planning item)

### Order-statistics curve — E[max of k LB draws], control mean 0.922
| k | at σ̂ = 0.074 | at CI-hi σ = 0.213 |
|---|---|---|
| 5 | 1.01 | 1.17 |
| 10 | 1.04 | 1.25 |
| 30 | 1.07 | 1.36 |
| 40 | 1.08 | 1.38 |
| 60 | 1.09 | 1.42 |
| 110 | 1.11 | 1.46 |

### Per-mechanism reach table (what can close the 0.42/0.84 gap to 1.44/1.86)
| mechanism | reach (LB points) | status |
|---|---|---|
| order stats over draws (k→110) | **+0.15 floor-raise max** at σ̂; never wall-breaking at point estimate | arithmetic, published |
| warpack v1 (banking/recovery) per-draw mean | unknown; build-rail pooled Δlc +0.13 (p 0.089) with **negative RHAE conversion**; LB Δmean +0.035 (p 0.66, unpowered) | compound gate look Jul 17 |
| R2 ledger+escalation (war-v2) | predicted Δlc ≥ +0.08 ≈ +0.05–0.10 LB **if** conversion holds | blocked on A2 engineered-replay condition + gate outcome |
| R3–R5 grinder cracking (L2+ on sb26/su15-class) | the only budgeted wall-closer; sized in path_forward v3 | queued behind R2 |
| YUTO-class per-draw mean edge (1.86 @ 40 entries) | existence proof that ≥ +0.9 per-draw mean over our control is achievable | opaque; no public footprint |

### P1–P5 — verbatim (prereg §5, restated as required before first R2 window)
> P1: sb26 leaves the fill-in-order family before action 80 and states ≥3 distinct goal
> families. P2: su15 states a third goal family within 30 actions of refuting the second.
> P3: verbatim-paragraph recurrence drops >70%. P4: SPACE/no-op re-probes ≤2 per run.
> P5: sb26 post-restart does NOT re-execute a refuted plan. ≥4/5 = concept validated
> even if L2 doesn't fall.

Observables: pulled build-rail transcripts, GOAL:/RESULT: regex extraction; P3 =
SequenceMatcher ≥ 0.9 paragraph-pair rate vs 13-run baseline; P4 from FACT-ledger no-op
table. Effect target in gate currency: Δlc ≥ +0.08.

## Panel-objections disposition (standing section, per A6)
| item (round) | disposition today |
|---|---|
| order-stats curve + reach table (R9/R10/R11, rl-planning) | **DONE above** — no longer outstanding |
| P1–P5 verbatim in brief (R10/R11) | **DONE above** |
| LB screenshot provenance (R11 minor) | **DONE** — `learnings/artifacts/lb_screenshot_2026-07-16.png` |
| compound gate rule (R11 major 1) | filed 07-15 (A1); gate look tomorrow uses it unmodified |
| banking engineered-replay condition (R11, A2) | **partially satisfied**: `bank_fire_validation.json` shows fired+score-invariant on ar25 & s5i5; sc25/m0r0 abort `frame_divergence` (per-play randomization → strict guard, by design). OPEN QUESTION Q1 below: does this satisfy A2's multi-pass condition, or must the canary-counting war-v2 build demonstrate it on-kernel? |
| daemon window-loss (R11, A5) | verified fixed live — draw #3 fired 00:07Z |
| LOO jackknife on recovery concentration (R11 minor) | scheduled tomorrow (post gate look), unchanged |
| wheel-formula reconstruction (rl-planning pt i) | standing dispute per §7 — unchanged |

## Open questions for the panel (round 12)
- **Q1 — tonight's LB window:** (a) war draw #4 (ledger → n=4; σ̂ recheck at n=5 per A3), or
  (b) frozen-fork control draw #6 (control ledger df 4→5 sharpens the σ̂ every inference
  uses), or (c) first R2 A/B window with war-v2 flags ON — only if the panel rules the
  A2 replay condition satisfied by `bank_fire_validation.json` AND the war-v2 build with
  attempt-counting canary passes smoke today. Author lean: (a) if gate look pending seems
  decisive tomorrow, else (b); (c) feels one day early.
- **Q2 — dev priority today** (after seed-3 push, done): (i) build+smoke war-v2 (ledger
  flags ON + attempts-not-successes canary) so a licensed window can happen without delay;
  (ii) seed-audit unseeded fallback paths (discussion #726552); (iii) conversion forensics —
  why lc gains don't convert (action-cost accounting on recovered clears; feeds
  conversion-first mode which is the modal A1 branch). Author lean: (i)+(ii) today ((ii) is
  ~an hour), (iii) tomorrow with 3-seed data.
- **Q3 — censoring rule:** adopt the "0.00 LB draws are censored infra failures, excluded
  from ledger stats, logged separately" footnote now, before one ever occurs? (Cheap,
  pre-registered before contact with data — the right time is now.)
- **Q4 — pooled screen usage:** confirm the gate look uses the 3 seeds as pre-registered
  (no peeking adjustment needed — today's pooled numbers were computed for monitoring, the
  A1 rule is unchanged and was filed before seed 2 existed).

## State (mechanics)
- Pushes: 1/2 used (war-eval v3 = seed 3, RUNNING, ETA ~2.4 GPU-h). GPU reserve untouched.
- Queue: head EMPTY (frozen filler only) — MUST be set after panel; hard deadline 18:00.
- Phase-1 line CLOSED; retry look unspent. Sched-v1 dead. JEPA dead.
- Compound gate look: tomorrow (Jul 17), seeds 1–3, rule A1, no discretion.
