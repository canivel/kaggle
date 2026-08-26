# Daily brief — 2026-07-21

## §1a Result deep-dive

**Draw:** frozen-fork filler = **0.93** (ref 54866117, submitted manually 01:32Z
after the 00:07Z daemon fire was blocked by the audit-trail gate — see incident,
below). Band-typical: frozen observed band 0.76–1.33.

**Pre-registered expectation:** a plain draw from the frozen-fork distribution.
**Met.** No mechanism claim attaches to a filler draw; no kernel pull done
(filler runs are not evidence artifacts).

**Ledger update:**
- Frozen control n=8 {0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93}:
  mean **0.974**, σ̂ **0.155**.
- Pooled (frozen + closed war arm) n=13: mean **0.959**, σ̂ **0.141**.

**Interpretation:** two consecutive band-typical draws (0.92, 0.93) are diluting
the 1.33 outlier; σ̂ is drifting DOWN from the 0.154 pooled estimate. Descriptive
naive-normal update: E[max over ~104 remaining windows] ≈ **1.31**,
P(any filler draw ≥ 1.44 wall) ≈ **0.05** (the 07-18 process model said ≈1.39 /
0.29 at σ̂ 0.154 with common-night correlation; both estimates agree on the
direction). **Filler stays a lottery, and its odds are getting worse, not
better** — every credible ≥ +0.06–0.12 experiment window is better-priced than
filler. This strengthens the case for spending windows on the sentinel W1 and
EWM lines as soon as their gates clear.

**Incident (process, resolved):** the 2026-07-20 20:07 EDT daemon fire was
blocked by the audit gate (ITERATION_LOG heading written only at end-of-day;
the long-running session hadn't finished). User recovered the window manually
21:32 EDT with the exact queued head. Permanent fix live: the loop now writes a
`### <date>` stub as its FIRST action (done today 08:23). Cadence intact — no
window missed.

## §1b Discussions sweep (learnings/war_room/discussions_2026-07-21.md)

Quiet window — zero new threads, zero new public notebooks, leader (1.86) still
opaque. Two IGNOREs (#684625 lament, #697720 accelerator housekeeping — already
our war-v4 target env). **Watch-item (ADAPT, no code change):** unresolved
1.15x-vs-1.0x per-level efficiency-cap discrepancy (methodology page vs Kaggle
/data + arXiv:2603.24621), host-unanswered. We treat LB math as 1.0x =
completion-weighted → breadth/generalization dominates (matches our
generalization-first prior). If Kaggle flips to 1.15x, fast-solve efficiency
becomes worth up to +15%/level — re-price then.

## §1c Research sweep (learnings/war_room/research_2026-07-21.md)

Thin window; no ADOPT-and-build.
- **ADAPT (medium)** AgentAbstain 2607.10059: post-hoc-abstention failure mode =
  twin of our EWM step-0 aborts → verify/abstain pre-flight on phase-aliased
  games feeds EWM contract v1.1.
- **ADAPT (low)** Infinite Agentic Loops 2607.01641: externally validates the
  budget-sentinel premise; cite only.
- **ADAPT (serving)** 70B quant delta: FP8 ≈ −0.4 pt vs FP16, AWQ-INT4 ≈ −1.6 pt
  (text; VL extra) → **pre-register FP8/W8A8 as the A17 fallback SKU** if
  Qwen2.5-VL-72B-AWQ misses the capability bar (still fits 96GB). A17 remains
  self-benched (no external throughput anchor exists).
- **PARK** Harness Effect 2607.06906: harness leverage is model-agnostic —
  argues port-harness-verbatim for A17 (already mandated by A17′ byte-identity
  tests).

## §2 Panel state

R16 (the A14 SEALING round, full 5-reviewer, circulation
`learnings/panel/r16_circulation.md`, 5 parts sha-stamped): yesterday's 12:38Z
spawn **zombied** (all 5 'running' 24h, zero result rows — same failure family
as R13). **Relaunched today 08:5x on the identical circulation**; collection in
progress. Open questions to R16 unchanged: Q1 gate seal on republished
arithmetic; Q2 SENTINEL_BUDGET=150; Q3 (c)+Reki kill vs audit-informed
resurrection; Q4 EWM Stage-1 re-price + gate; Q5 banking scope from latent-state
audit; Q6 A17′ sign-off; Q7 W0 control-band n=4 pooling; Q8 dream digest.

## §3 Development state (as of brief writing)

Sentinel W1 push is FULLY STAGED, blocked only on R16 Q2:
- arc-war-kit dataset version pushed + **live-verified byte-identical**
  (budget_sentinel_patch.py 16790 B).
- build_eval_notebook.py gained `--sentinel-budget N`; notebook rebuilt with
  SENTINEL_BUDGET=150 exported in cell 2 + greppable banner; smoke **29/29**.
- B=150 canary re-run (mandatory pre-seal): **PASS 3/3 certified seeds**, 33/75
  firing units, 6/6 budget deaths warned, 0 predicate violations
  (runs/sentinel_canary_b150.json).
- Push plan on R16 approval: kernel seed 1 (push 1/2), seed 2 (push 2/2) if
  same-day; banner + `SENTINEL v=1` event grep post-run (inert-sentinel = #1
  risk, covered by the cell-2 export).

## Open questions
1. R16 Q1–Q8 (above) — verdicts pending collection today.
2. Tonight's LB window: frozen-fork filler (sentinel is build-rail, not a
   submission head). Queue verified non-empty.
3. Panel-zombie robustness: 2nd relaunch in 4 rounds — the spawn-side hang (not
   collection) is now the live failure mode; needs a heartbeat/timeout fix in
   panel_round.py or KAOS runner (backlog).
4. If R16 rules a different SENTINEL_BUDGET value: one-command rebuild
   (`--sentinel-budget <B>`) + canary re-run at that B before push.
