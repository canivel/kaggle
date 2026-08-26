# Daily brief — 2026-08-07

## 1. THE result: A22 v2.1 seed-1 screen — K3 FAIL → **A22 COMPACTION LANE DEAD**

Third independent K3 strike (v1 08-03, v2 08-06, v2.1 today) under the sealed prereg
(`learnings/war_room/a22_compaction_v2_1_prereg_2026-08-06.md`), which accepted this consequence at seal
time. The run is fully VALID — every canary passed, and the arm's defining check (injection channel closed:
`digest_tokens=0` AND `reserve_applied=0` on 2,780/2,780 events, RETAIN-OFF clean) held — so this is a
**decisive negative answer**, not a void.

**Pre-registered expectation vs outcome:** the §0 question was whether pure region-aware eviction (the
LightMem cell, entered by construction) passes non-harm. It does not: mean Δlc **−0.360** (cap −0.128),
worst **ar25 −2** with sc25 also −2 (cap −1.0), 2W/10L, 13 levels vs war-eval's 22 — the worst arm of the
three.

**Mechanism evidence (what the lane bought):**
- **Harm is eviction itself.** Monotonic worsening as injection was removed: v1 −0.200 → v2 −0.320 → v2.1
  −0.360; pearson(evicted_chars, Δlc) = **−0.403** (v2: −0.13). The duck's policy uses its full episodic
  context; deleting "stale" episodes deletes load-bearing information. Theory-consistent: arXiv:2608.01326
  proves selection (eviction) is the strictly weaker compaction class vs generation.
- **M3 confound, exactly as prereg §3 anticipated:** reprop −6.49pp (p=0.0001) with ZERO injection channel
  ⇒ v2's −4.57pp "first lane win" was a compaction side-effect on thinking-text similarity, NOT the
  refuted-list digest working. The standalone refuted-list micro-arm (R24 pile) is weakened — re-argue from
  scratch or drop.
- stuck-suppress never fired (0 events); eviction classes 61% episode / 39% user; sc25 never recovered in
  any arm.

**Post-death disposition (sealed 08-06, executed today):** lane closed in project memory; carried forward =
M3 confound + borro1980 variance map + "additive-only memory" datum; **no compaction push of any kind
without an R24 (Sunday 08-09) revival decision.** Evidence: `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json`,
memo `learnings/sweeps/a22_v2_1_seed1_screen_2026-08-07.md`.

## 2. Draw + ledger: watch-rule fired and RESOLVED same day (STATIONARY)

Overnight filler 0.78 (z −1.06) after 0.77 = second consecutive sub-0.80 → pre-registered watch-rule fired.
Re-check (fresh script, sealed original untouched; `runs/stationarity_recheck_2026-08-07.json`, memo
`learnings/sweeps/stationarity_recheck_2026-08-07.md`): **STATIONARY** — change-point p=0.757 (min-seg≥3;
the unconstrained |t|=5.32 split is the same n₂=2 near-equal-pair artifact NC-15 discharged), Mann-Kendall
p=0.62, tabular CUSUM no breach, and the headline: **P(some adjacent pair both <0.80 in a 24-draw record) =
0.19–0.51** across the sealed nulls — a ~1-in-5 event at strictest, a coin-flip under the record's own fit.
Ledger n=24, mean 0.9413, s 0.1596 — safe for gate arithmetic. σ=0.24 regime stays rejected (χ² p=0.0097).
**Re-fire condition:** third consecutive sub-0.80 tomorrow (per-draw prob ≈0.10–0.19).

## 3. Sweeps

**Research** (`learnings/sweeps/research_2026-08-07.md`) — plan-relevant:
- **Public ARC-AGI-3 is effectively saturated by state-externalizing harnesses:** Prime Intellect **Prime
  Agent** (08-05, open-source MIT) 95.5% with Opus 5, 183/183 levels, above human baseline; **Tycho**
  (arXiv:2607.28287) 183/183 with 61% fewer actions via programmatic world models. Both ADOPT-as-blueprint
  for the successor lane: state/program lives OUTSIDE the context window — consistent with our eviction-harm
  finding (don't shrink context; externalize state instead).
- Successor-lane inputs ranked in the file: Activity Frames 2608.05784 (deterministic typed-frame digests),
  replay-verified skill store 2608.06153 + Skill-Use compliance 2608.04828, TTCD 2608.01672. Cautionary
  null: 2608.04066 (deterministic-executive, 0 completions).
- 2608.01326 (compaction theory) — see §1; any future compaction revival must be generation-side.

**Discussions** (`learnings/sweeps/discussions_2026-08-07.md`): zero new topics since 08-06 (newest thread
still 08-05). Comment-level only: Jason Feng third notebook (IGNORE), his "Kaggle errors" resolved
self-inflicted (IGNORE), borro1980 merge solicitation naming teams at ranks 11–25 (MONITOR). **Cadence rule
met (2 quiet days) → discussions sweep goes every-other-day (next 08-09), cutoff check folds into the daily
brief.**

**LB:** KOJIMA 1.86 #1 frozen; gold/top-13 cutoff 1.56 third flat day (#13–15 all 1.56); top-5 prize cutoff
1.61; 1.58 pack 4 names + cstl 1.59; page-1 floor 1.40; our 1.33 below #49. Archived
`runs/lb_daily/lb_2026-08-07.csv`.

## 4. Open questions → R24 agenda (Sunday 2026-08-09)

1. **Successor lane selection** — the central decision. Candidates, ordered by evidence fit:
   (a) **state-externalization / programmatic world model** (Prime Agent open-source + Tycho blueprint;
   additive, consistent with eviction-harm datum; largest engineering lift — assess what's portable into the
   duck within fork-never-build + zero-budget rails);
   (b) **additive typed memory** (Reki typed-causal-memory +0.098 local anchor; Activity Frames; LeanMem
   write-once — adds retrieval without subtracting context);
   (c) **replay/banking revival** (borro1980: 2 games = 65% of variance → banking the binary clears is the
   variance-efficient target; needs the 07-15 frame-divergence fix).
2. Refuted-list micro-arm: drop or re-argue from scratch given the M3 confound.
3. Compaction lane: formally record DEAD; revival bar = generation-side design + new mechanism theory.
4. Weekly KAOS dream + fingerprint report (Sunday per protocol).

## 5. Today's mechanics

Queue: frozen filler armed (trusted-fork, message updated with lane-DEAD + stationarity state). Kernel
pushes 0/2 used (no push today — sealed disposition forbids compaction pushes; successor lane needs R24).
$0 cloud. Nothing passes promotion gates; measurement-only day.
