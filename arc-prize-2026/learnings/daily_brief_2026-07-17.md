# Daily brief — 2026-07-17

Author: daily-loop agent. Inputs: overnight LB draw #4 (war-v1 ledger → n=4), the SEALED
3-seed compound gate look (executed this morning per amendment A1, no discretion), the
war-v2-eval ledger-ON screen (landed late 07-16), discussions + research sweeps. Panel
round 13 runs on this document at FULL 5-reviewer strength (mandated by the gate outcome,
see 1a).

## 1a. Result deep-dive

### THE GATE LOOK (A1 compound rule, seeds 1–3): FAIL ON BOTH PRONGS → warpack line CLOSES
Artifact: `runs/war_gate_look_2026-07-17.json`. Pooling = per-game Δ averaged across the
3 certified seed replicates, exact sign-flip on the 25 pooled deltas (same convention as
the 07-16 2-seed monitor).

| prong | rule | observed | verdict |
|---|---|---|---|
| (i) primary | pooled Δlc sign-flip p < 0.0125 | Δlc +0.059, **10W/10L**, p = **0.225** | FAIL |
| (ii) secondary | mean Δlog1p(RHAE) ≥ 0 across seeds | **−0.132** (all 3 seeds negative: −0.036/−0.159/−0.202) | FAIL |

- **Interpretation:** seed 1's +0.272 (p 0.0074) was a 1-seed draw from a noisy panel —
  by seed 3 the primary is a coin-flip (10W/10L) and the RHAE secondary is *uniformly*
  negative. Warpack-v1's composition (banking+recovery+retry_guard as built) clears no
  more levels than null on pooled evidence and pays action-cost for what it does clear.
  LOO diagnostic: p range [0.14, 0.42] — no single game rescues the gate.
- **Only stable signal:** ka59 positive in all 3 seeds (+0.6 mean); cd82/cn04/sk48 stable
  negatives. The "recovery cracks stuck games" story survives only as a per-game anecdote.
- **Pre-registered consequence (A1 fail-on-both, verbatim):** "warpack build-rail line
  closes (LB control ledger continues to n=5 for the §3 record; R2 A/B launch decision
  escalates to a full 5-reviewer panel)." → R13 is that 5-reviewer panel. No discretion
  was exercised: the look was sealed 07-15, executed once, on the pre-registered rule.
- This is the campaign's second full line-close by its own pre-registered gate (sched-v1
  07-14, warpack build-rail today). The discipline is working; the cost is that **no
  built mechanism has yet moved the per-draw mean** — which 1a-LB below says is the only
  thing that matters.

### war-v2-eval (ledger-ON) screen: NEGATIVE — ledger-as-built is a constant context tax
Landed late 07-16 (`runs/war_v2_eval_s1/screen_report.md`): tripwires PASS (per-game
keying verified live, canary `stores=25 attempts=1552 digests=1552 skips=0 aborts=0`)
but **Δlc −0.128 (p 0.86), Δlog1p(RHAE) −0.314** — below all three ledger-OFF seeds.
1552 digests, **0 escalations**: the ledger's firing trigger never fires, so it is pure
context tax — phase-1's always-on-injection failure repeated at the ledger layer.
**Consequence:** ledger-as-built does NOT enter scored windows (would have been blocked
by A1's fail anyway). Its distill-derived upgrades (budget-sentinel FACTs,
submission-fingerprint refutation, PREDICT→RESULT wiring) are precisely the components
that give it firing triggers — they are war-v3 material, ranked in §Q1 below.

### LB: war draw #4 = 1.05 → ledger n=4 {0.91, 1.08, 0.88, 1.05}
- Mean **0.980**, σ̂ **0.0997**, χ² 95% CI on σ (df 3) [0.056, 0.372]. **A3 check:
  σ̂ < 0.15 → LB windows remain live.** A5 variance gate (χ²-CI-hi < 0.25 at df ≥ 4)
  evaluable at n=5 — tonight's draw completes it.
- vs frozen control (0.922, n=5): Δ +0.058, Welch t 0.97, p ≈ 0.36 — unpowered by design
  (MDE ≈ 0.14); no claim made. Descriptive note for the record: 3 of 4 war draws ≥ 0.91
  and the war mean now sits above every frozen draw except 1.02; IF a small true lift
  exists (banking/recovery occasionally converting on LB's budget regime), n=5 t-stats
  won't see it — the §3 record accumulates regardless.
- Prereg expectation met: accumulation only, no gate consumed, banking still UNVERIFIED
  on the record.
- **Tension worth the panel's eyes:** build rail says warpack ≈ null (gate FAIL); LB
  ledger says +0.06 unpowered. These are consistent (build rail forces offline-bench
  regime; LB runs the full 8h budget where soft-time banking/recovery conditions differ)
  — but per A1 the build-rail evidence governs build decisions. The LB ledger completes
  n=5 for the record either way.

### Leaderboard
- #1 YUTO KOJIMA 1.86 (no footprint, no change). Wall region: 11+ teams ≥ 1.44; 1.56
  cluster persists. Our best 1.08 (war draw #2). Nothing new strategically; per-draw
  mean gains remain the only path (order-stats ceiling 1.11@k=110 at σ̂).

## 1b. Discussions sweep (new since 07-16) — verdicts
1. **#726903 "x(-1)/week is scary"** (Doruk Doğrular) — speculation the public
   score-multiplier decays weekly. **IGNORE**: no mechanism/data; our gate currency is
   Δlc per draw, not the multiplier.
2. **#716295 "Submission parallelism questions" — NEW comments** (Gabriel Mirea, Nick
   Pellegrin): swarm.py runs one daemon thread per game; per-agent LLM routing and
   per-game budget allocation are overridable (`swarm.py#L67`); open tradeoff "110
   parallel × 9h vs concentrated budget". **ADAPT**: confirms budget re-allocation
   across games is a legal harness lever — concentrating action budget on
   winnable/almost-cracked games (and starving stable-negative ones like cn04) is a
   candidate war-v3 mechanism that directly targets per-draw mean. Added to Q1 ranking.
3. #726367 poll, #725002 Milestone — trivial chatter, **IGNORE**. No YUTO footprint; no
   official seeding answer in #726552; no new public notebook > 0.79.

## 1c. Research sweep — verdicts
1. **Proactive Memory Agent** (arXiv:2607.08716) — separate memory agent watches the
   trajectory and decides *per-step* whether to inject a reminder or stay silent; ablation
   shows **selective intervention beats always-on injection** (+8.3pp Terminal-Bench).
   **ADOPT**: external confirmation of exactly what our war-v2-eval measured (1552 digests,
   0 escalations, negative screen = always-on tax). Design template for any war-v3 ledger
   revival: sidecar with a silence-default firing policy. Feeds Q1(d).
2. **MemCon** (arXiv:2607.13591) — when/what/how-much-to-retrieve as a lightweight UCB
   bandit, no extra LLM calls, converges in tens of tasks; +task success with 5–20% fewer
   tokens. **ADAPT** (idea, not code): a per-game fire-vs-suppress gate over hand-designed
   state (episode stage, stuck-flag, goal recurrence). Behind Q1 ranking.
3. **Shared Selective Persistent Memory** (arXiv:2607.09493) — keep only 4 reusable
   categories, drop session reasoning traces. **ADAPT (weak)**: taxonomy for what belongs
   in a cross-game ledger; no firing policy.
4. Observation-masking (2508.21433, resurfaced): "simple masking ≈ LLM summarization" —
   mildly **confirms** the cheap *scripted* probe-diff instinct in Q1(b) over any
   LLM-written summarizer. MRMS 2607.04617, agentic-RL survey 2607.01120 — **IGNORE**
   (overlap/survey; the survey's "deterministic replay" category matches our N5 result).
5. Nothing new on ARC-AGI-3 itself, action-cost accounting, or budget sentinels in the
   window (BAGEN 2606.00198 remains the closest budget-sentinel template).

## Instruments (standing)
- Order-stats curve unchanged (07-16 brief) — E[max] 1.11@110 at σ̂; floor-raiser only.
- Per-mechanism reach table DELTA vs 07-16: warpack v1 per-draw mean now **measured ≈ 0
  on build rail** (gate FAIL); R2 ledger+escalation **as-built refuted** (0 escalations,
  negative screen); R3–R5 grinder cracking now the ONLY live budgeted wall-closer
  (A6: unconditional, build work NLT Jul 20); GPT-5.6 probe decomposition says grinders
  = model gap, su15 = info-theoretic wall (accept), ft09 = frontier-convertible.
- P1–P5 verbatim: unchanged from 07-16 brief §Instruments (prereg §5). Note P1–P5 were
  designed for the R2 A/B whose launch decision is now escalated to this panel.

## Panel-objections disposition (per A6)
| item | disposition |
|---|---|
| R12 option (c) refusal (no early A/B) | vindicated — gate FAILED; A/B as designed is moot |
| R12 M2 accumulation to n=5 | draw #4 done; draw #5 = tonight's head (pending panel confirm) |
| A5 variance gate at df≥4 | evaluable tomorrow at n=5 |
| A6 grinder-cracking scoping | THIS panel ranks the backlog; build NLT Jul 20 |
| A7 seed audit | PASS (07-16); ledgers valid |
| LOO jackknife on recovery concentration (R11 minor) | subsumed by gate-look LOO (p range [0.14,0.42], published) |

## Open questions for panel round 13 (FULL 5-reviewer per A1 escalation)
- **Q1 — rank the war-v3 backlog** (one flag per window, simplicity-wins prior, all
  game-agnostic). Candidates with current evidence grade:
  (a) **budget sentinel** (shrinking-bar detector → FACT; every Qwen grinder death was an
      unseen budget death) — distill-grounded, cheap, targets RHAE conversion;
  (b) **scripted probe-diff summarizer** (~120-token fixed block per action result) —
      distill-grounded (GPT's separating behavior), medium complexity;
  (c) **submission-fingerprint refutation** (hash layout at submit; blocks verbatim
      re-submits) — distill-grounded, cheap;
  (d) **PREDICT→RESULT ledger wiring + no-effect FACTs** (ft09 distill) — gives the
      (refuted-as-built) ledger a firing trigger;
  (e) **banking unpruned-replay + scored-budget soft-time** (N5: all 25 games
      deterministic; prune bug + soft-time threshold are engineering fixes) — but
      banking's parent line just gate-FAILED on build rail;
  (f) **game-over-continuation prompt fix** (su15 deadlock: correct restart computed
      every turn, never executed) — free hygiene;
  (g) **per-game budget re-allocation** (1b#2: concentrate budget on convertible games,
      starve stable negatives) — NEW today, legality confirmed by hosts' own repo.
  Author lean: (a)+(f) as war-v3.0 single window (both are subtractive/cheap, both
  target the conversion failure the gate exposed), (g) scoped next, (b)/(d) A/B-only,
  (e) parked with its line.
- **Q2 — R2 A/B disposition** (this panel owns it per A1): formally retire the A/B as
  designed (ledger-as-built refuted on build rail; LB A/B unpowered per A4-0716), OR
  keep window structure with war-v3 as the treatment arm once built?
- **Q3 — tonight's window:** war draw #5 (completes n=5 → A5 variance gate + final σ̂
  recompute; author lean, licensed by R12 M2) vs frozen control draw #6.
- **Q4 — GPT-5.6 rig follow-ups:** su15 re-probe at frontier tier ($10 topped up) is
  optional; author lean: skip (info-theoretic wall verdict stands; spend nothing on a
  closed question). Any objection?

## State (mechanics)
- Pushes: 0/2 used today. GPU reserve untouched. Probe budget: $25.33 remains + $10 top-up.
- Queue: frozen filler only; head MUST be set post-panel (deadline 18:00; daemon 18:37/20:07).
- Lines: phase-1 CLOSED, sched-v1 DEAD, JEPA DEAD, warpack build-rail CLOSED (today),
  ledger-as-built REFUTED. Retry look unspent. War LB ledger OPEN (n=4→5 tonight).
