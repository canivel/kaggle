# Daily Brief — 2026-08-08 (Saturday)

**Day type:** R24-prep day (no panel — full R24 panel tomorrow, Sunday 08-09). No build-rail pushes
(sealed post-death disposition forbids compaction pushes; successor lane needs R24 authorization).
Measurement + synthesis only.

## 1. Result deep-dive: overnight draw 0.87 (interior recovery)

Frozen-fork filler scored **0.87** (submitted 00:07Z, API COMPLETE) → ledger **n=25, mean 0.9384,
s 0.1569**. Interpretation: z ≈ −0.45 vs the n=24 record — a clean interior draw, not a tail event.
Coming after the two consecutive sub-0.80 draws (0.77, 0.78) that fired the watch-rule on 08-07, this
recovery is consistent with the same-day STATIONARY resolution (`runs/stationarity_recheck_2026-08-07.json`:
change-point p=0.757 min-seg≥3, MK p=0.62, pair-prob 0.19–0.51 under sealed nulls). The watch-rule stays
resolved and re-arms only on a future consecutive sub-0.80 pair. No per-game pulls available for filler
draws (API-only); pre-registered expectation (interior draw from the stationary record) was met. Ledger
remains safe for gate arithmetic; σ=0.24 regime stays rejected (χ² p=0.0097).

**LB (from 06:00 archive, `runs/lb_daily/lb_2026-08-08.csv`):** KOJIMA 1.86 frozen #1; head static
(Andy liu 1.69, Lord Han Solo 1.65, GeniusYY 1.64); **gold/top-13 cutoff 1.56 — fourth flat day**;
top-5 prize cutoff 1.61; 1.58 pack 4 names + cstl 1.59; our 1.33 below #49.

## 2. Sweeps

**Discussions:** deferred per the every-other-day cadence rule (2 quiet days met 08-07); resumes
tomorrow 08-09. Quick WebSearch check inside the research sweep found no leaderboard-relevant public
news requiring an early break of cadence.

**Research** (`learnings/sweeps/research_2026-08-08.md`) — three load-bearing items:
1. **Provenance correction (de-rates yesterday's framing):** Prime Agent's 95.5% is vendor-reported
   and independently unreplicated; ARC Prize keeps harness results off the official leaderboard
   (official Opus 5 = 30.2%); the 95–100% figures live on a self-reported community leaderboard.
   "Public set saturated" is a statement about harness-design maturity, not a verified bar.
2. **Backfill: the Schema harness** (Impossible Research, ~07-16) predates Tycho and Prime Agent —
   executable-program world models, ~99% RHAE public with **Opus 4.8** (not frontier-latest), and it
   published **50 agent traces + a dependency-free scorer on HuggingFace** — the only external artifact
   so far compatible with our offline/zero-budget kernel rails. State-externalization is a
   **three-team convergence** (Schema → Tycho → Prime Agent), which carries the argument even after
   the headline numbers are de-rated. Tycho's code is public (`github.com/NIMI-research/Tycho`;
   correct metric: mean RHAE 88.49 on Opus 4.8 orchestrator; 100.00 = metric ceiling on frontier).
3. **arXiv delta thin** (Fri/Sat indexing lag; re-check 08-07 submissions tomorrow). Best pickup:
   **SkillHEX 2608.05628** — executable failure-tests give dense signal without spending scored
   actions; enabling mechanism for a banking/replay revival. ADAPT-level: 2608.05891 (transition
   deltas), 2608.04530 FocusMem, 2608.06257 MASS. Honest empties: no new ARC-AGI-3 paper since
   2608.04066; zero follow-ups to compaction theory 2608.01326; A22 closure unchallenged.

## 3. R24 prep (the day's work product)

Three assessments written, then synthesized into the panel decision document
**`learnings/war_room/r24_successor_lane_proposal_2026-08-08.md`** (453 lines):

- **Prime Agent portability** (`learnings/war_room/prime_agent_portability_2026-08-08.md`): repo read
  directly (agent-loop, compaction, refinement, skills, rlm-runtime, state-snapshot). Nothing literally
  portable (TS monorepo + authenticated provider login; `enable_internet: false` bars it; no
  ARC-specific code exists — vendor's only ARC change was the task prompt). **But the duck is two
  additive mechanisms short of the RLM shape:** (i) persistent kernel namespace (our sandbox spawns a
  fresh subprocess per call, destroying model-built state) and (ii) durable cross-level memory
  (`_summarized_knowledge` is wiped at level transitions). Their own compaction is generation-side
  summarization at safe cut points — never eviction — corroborating the A22 post-mortem. Lift M
  (1–2 dataset push cycles), $0. Cheapest kill-path: `namespace_reuse_rate` canary (≥0.15 floor)
  answers the 27B-adoption NULL without needing a score.
- **Tycho portability** (`learnings/war_room/tycho_portability_2026-08-08.md`): paper + public code
  read verbatim. **Headline: Tycho is a point-by-point diagnosis of why our shelved exec-wm lane
  died** (stateless sims → latent-state aliasing; IID split → on-trajectory collapse 0.026–0.16; no
  abstention → confident-wrong step-0 aborts; untyped frames; whole-plan beam). Four of five fixes are
  deterministic code, no in-kernel LLM. **Reconciliation:** Tycho's frontier config runs `tail_evict`
  aggressively and is safe *because state already lives in a verified external program* — eviction is
  harmful when not preceded by externalization; this single sentence unifies our A22 harm curve with
  the winners' designs and sets the ordering constraint. Hard blocker on the LLM half: Tycho spends up
  to 3,500 LM calls/game vs our ~67/game (~52× gap), zero weak-model ablation. Verification is offline
  replay — zero scored actions. Unknown: no wall-clock reported (9h-envelope compatibility unverified).
- **Metric resolution (closes the Prime Agent file's homework item):** the Kaggle metric IS
  action-efficiency-weighted — quadratic action penalty, hard 5× action cap, linear level weighting,
  zero credit for unfinished levels (`docs/community_research_apr1.md:261-303`). Makes
  `actions_per_level_completed` (baseline 165.4) a legitimate co-primary; completions stay primary
  (Goodhart guard).
- **Candidate (c) unblocked on paper:** the 07-15 frame-divergence root cause is the N5 `prune_trace`
  phase desync (no-ops advance the phase counter; pruning leading no-ops desyncs replay → step-0
  aborts on sc25/m0r0; `learnings/panel/r16_circulation.md:~1250`); fix = full unpruned replay from
  RESET (fired + score-invariant already shown on ar25/s5i5, `bank_fire_validation.json`). borro1980
  variance map: 2 games = 65% of ledger variance → banking binary clears is variance-efficient.

**Proposal recommendation (for R24 to decide):** select lane **(a) state-externalization**, Tycho as
the artifact schema *inside* the lane; (b) additive memory demoted to a component arm (the
`_summarized_knowledge` un-wipe); (c) banking kept as variance-efficient complement on its own clock.
Sequenced authorization: S1 = L0 free zero-push offline re-verification of the 24 existing exec_wm
sims under Tycho's protocol (kill: carrier set stays ~4 games); S1b = free offline bank re-fire with
pruning disabled on all 25 traces; S2 = one dataset push for the persistent-namespace seed-1 screen;
S3 = decision point; S4/S5 later. Gates inherited verbatim from A22 preregs (no retuning charge) +
new `namespace_reuse_rate ≥ 0.15` adoption canary. Refuted-list micro-arm: **drop** (M3 instrument
confounded — moved −6.49pp p=0.0001 with the injection channel provably closed). Governance rulings
requested: (i) does workstation LLM regeneration count against the zero-budget rail (load-bearing for
L1)? (ii) adopt provenance de-rating of self-reported community numbers as a standing rule.

## 4. Open questions → R24 (Sunday 2026-08-09)

1. Ratify (or amend) the successor-lane selection + S1/S1b/S2 authorization per the proposal doc.
2. The two governance rulings above.
3. Refuted-list micro-arm: confirm drop.
4. Compaction lane: confirm formal DEAD record (revival bar = generation-side + new mechanism theory).
5. Weekly: KAOS ingest + dream digest, fingerprint report (run before panel per protocol).
6. Discussions sweep resumes (every-other-day cadence); re-check 08-07 arXiv submissions once indexed.
7. No-regret action available immediately if panel concurs: attach Schema traces as a Kaggle dataset.

## 5. Today's mechanics

Queue: frozen-fork filler armed (trusted-fork, n=25 message current) — tonight's fire is the eternal
fallback, as every night. Kernel pushes 0/2 used. $0 cloud. No panel (Saturday). Nothing passes
promotion gates; measurement + synthesis day. Daemon healthy (08-08 00:07Z fire ok=true).
