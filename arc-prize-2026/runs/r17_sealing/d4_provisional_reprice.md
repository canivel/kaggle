# D4 — Provisional re-pricing through the verified level-number-weighted oracle

**Status: PRE-SCORECARD / PROVISIONAL. NO SEALED THRESHOLD MOVES.**
Directive R17 D4 / OBJ-B / ruling R-DEPTH-REPOINT. Filed 2026-07-23.

methodology (the statistics remit) rules the binding look's sign test is on **Δlc
(depth events)** and is **unaffected**; this document moves no seal. It is a $0
recompute + a labeling amendment: every efficiency-denominated price from
`learnings/panel/r17_circulation.md` §7 (EWM tu93/ls20/ft09 channels) and §2
((a)/(b)/EWM-in pair rows) is re-published through the verified scoring oracle
and marked **PRE-SCORECARD/PROVISIONAL exactly as the B+ rows are marked
pre-A16**. Consumers must cite these numbers, not the old ones, before any
window-priority decision.

## Method (verified authority + legal baselines)

- **Scorer:** `duck_eval/scoring_oracle.py`, the shipped
  `arc_agi.scorecard.EnvironmentScoreCalculator` wrapper, validated to
  **0.00e+00** vs the harness on all 25 sentinel-run games
  (`runs/atlas_oracle/validation.md`). `using_real_scorer()` == True at run time.
- **True aggregate (verified):** `game_score = Σ score_i·i / Σ i` over attempted
  levels (level-number-weighted), `level_score_i = min((b_i/a_i)²·100, 100)` if
  level i completed else 0. LB = mean game score / 25.
- **Baselines:** the LEGAL control **w0_s1** (methodology N1), per-run
  `base_actions_per_level` via `load_baselines_from_benchmark(
  runs/kernel_pulls/w0_eval_s1/benchmark.json)` — NOT the atlas (drifts 20/25).
- Reproducer: `runs/r17_sealing/d4_provisional_reprice.py` → `.json`.

## Headline: level-number weighting sharpens depth ≫ efficiency beyond §12.2

The panel's key expectation is quantified and **confirmed, more strongly than
§12.2 framed it**: an early-level (L1) efficiency channel is worth a *tiny*
LB-rail slice because (i) L1 carries the **smallest** level-number weight in its
game (weight 1 of Σi = 21–45, i.e. 2.2–4.8% of the game), and (ii) the game score
is then averaged over 25 for the LB. A **frontier depth event** (the thing the
Δlc sign test actually rewards) is worth **4–15×** a comparable L1 efficiency
event on the same games:

| event (best case, at human parity) | game-pts | **LB rail** | ratio vs its own L1-speed |
|---|---:|---:|---:|
| tu93 L1 speed (shave 42→19) | +1.77 | **+0.071** | 1× (reference) |
| tu93 frontier depth (complete L3) | +6.67 | **+0.267** | **3.8×** |
| ka59 frontier depth (complete L2) | +7.14 | **+0.286** | — |
| re86 frontier depth (complete L2) | +5.56 | **+0.222** | — |
| sc25 frontier depth (complete L1→new-clear) | +4.76 | **+0.190** | — |

And the *realistic* (not oracle-parity) L1 efficiency gains are far below even the
best-case row: a half-gap shave on tu93 is **+0.017 LB**, and two of the three
§7 carriers price to **exactly 0** (see below). The one depth mechanism the
portfolio still owns is worth an order of magnitude more per event — which is the
whole R-DEPTH-REPOINT point, now on the oracle to the digit.

## §7 EWM per-subset channels — re-priced (PROVISIONAL)

| channel | old price (pts) | re-priced value (oracle, w0_s1 base) | LB rail | method | verdict |
|---|---|---|---:|---|---|
| **tu93 L1 speed** | +0.1–0.9 | best-case shave 42→19: +1.77 pts; **realistic half-gap: +0.44 pts** | **+0.017** (realistic), +0.071 (parity ceiling) | L1 completed at 42 (base 19); shave L1 actions, re-score | **survives but tiny**; the +0.9 pts upper end requires exact human parity, unattainable by BFS-in-sim |
| **ft09 L1 reliability** | +0.0–1.0 | **+0.0 pts** (L1 already at the 100 cap: 27 < base 43, zero headroom) | **+0.000** central | L1 already below baseline → level_score already capped at 100 | **prices to ZERO on the mean**; pays only in the variance tail where a run *fails* to complete L1 at all (reliability-of-completion, not speed) |
| **ls20 L1 speed** | +0.0–0.6 | **N/A as efficiency** (lc = 0: no L1 to speed up) | 0 as efficiency | control completed 0 levels | **misnomer**: cannot pay as efficiency. Re-cast as a **completion** channel it is +3.57 pts / **+0.143 LB** — but that is a DEPTH new-clear, not an efficiency line, and belongs in the Δlc accounting, not the efficiency rail |

**Per-subset re-price (the sealed §7 form), through the oracle:**
- `{tu93}` → **+0.017 LB realistic** (ceiling +0.071 at unattainable parity)
- `{ls20}` → **+0.000** as an efficiency channel (0 headroom; any value it has is a
  DEPTH L1-completion of +0.143, re-filed to Δlc)
- `{ft09}` → **+0.000** central (L1 already capped)
- any passing subset with none of {tu93 with real headroom} → **0**, Stage-1 parked.

**Consequence (provisional):** the §7 sum "+0.1–2.5 pts ≈ +0.04 central rail" is
**over-stated for the LB** once level-number-weighted and denominated in rail: the
efficiency-only central value is **≈ +0.017 rail** (tu93 alone; ft09/ls20 add ~0),
i.e. **less than half** the §7 headline, and the +0.9-pt tu93 ceiling is
scorer-unreachable. This *strengthens* the sealed §7 conclusion ("EWM Stage-1 is
no longer the largest registered non-model line; re-prioritizes below (a),(b),
A17") — it is now even smaller.

## §2 (a)/(b)/EWM-in pair rows — re-priced (PROVISIONAL)

The §2 P(pass) rows and the §2R (a) +0.06 / (b) +0.06 ceiling rows derive from
"expected positive pairs," where a positive pair is a **Δlc (depth) event** at the
binding look. **These are already depth-denominated, so the binding sign test is
unaffected** (methodology's ruling) — but the EWM-in *increment* to those rows was
priced off the efficiency channels above and is re-pointed here:

| §2 row | old label | provisional re-price | note |
|---|---|---|---|
| B− / EWM-out | P(pass)=0.04 | **unchanged** | pure Δlc sign test; no efficiency input; **SEALS AS-IS** |
| B− / EWM-in | P(pass)=0.08 ("+ ≤1 expected pair: ls20/ft09/tu93") | **PROVISIONAL: the EWM-in increment shrinks** — ls20/ft09 efficiency channels price to ~0 on the mean; the only live EWM pair is a tu93/ls20 **L1 completion** (a depth event, +0.14–0.19 LB if it fires), not an efficiency pair | the "+1 expected pair" central case should be read as a *depth* pair contingent on a new L1 clear, not an efficiency shave |
| B+ / EWM-out | P(pass)=0.19 | **unchanged by D4** (still owns its own pre-A16 flag) | banking = depth (replay-to-frontier); D4 does not touch it |
| B+ / EWM-in | P(pass)=0.27 | **PROVISIONAL: "EWM adds ls20 only" re-reads as a depth-completion contingency**, not efficiency | same haircut logic |
| §2R (a) sentinel +0.06 ceiling | efficiency-flavored | **W1 already refuted the efficiency lift** ("fires-doesn't-pay", §12.1); under the oracle a stop-grinding signal buys **0** unless freed actions convert to **depth**. (a)'s rail value is depth-gated, central ≈ 0 | consistent with the W1 −0.60/−0.72 behavioral-negative |
| §2R (b) diff +0.06 ceiling | efficiency-flavored | same: any (b) value must show up as **Δlc**, not action-trimming; provisional central ≈ 0 pending its own window | |

**No B− row moves** (they are depth sign-test rows). The EWM-in *increments* and
the (a)/(b) efficiency-flavored ceilings are marked **PRE-SCORECARD/PROVISIONAL**;
their only legitimate value is a depth (Δlc) contribution, which the binding sign
test already prices directly.

## Per-channel ledger (old price → re-priced → method → delta)

| channel | old price | re-priced (LB rail) | method | delta vs old |
|---|---|---|---:|---|
| tu93 L1 speed | +0.1–0.9 pts (~+0.004–0.036 rail if naïvely /25) | **+0.017 realistic / +0.071 ceiling** | oracle shave L1 42→19 on w0_s1 base | ceiling higher in *pts* but gated by unattainable parity; realistic value confirmed small |
| ft09 L1 reliability | +0.0–1.0 pts | **+0.000 central** | L1 at 27 < base 43 → already 100-capped | **↓ to 0** on the mean |
| ls20 L1 speed | +0.0–0.6 pts | **+0.000 as efficiency** (+0.143 rail only as a *depth* L1-clear) | lc=0, no efficiency headroom | **re-classified** efficiency→depth |
| (a) sentinel +0.06 | efficiency ceiling | **depth-gated, ~0 central** | W1 "fires-doesn't-pay" + oracle: stop-grinding pays only via depth | consistent with §12.1 |
| (b) diff +0.06 | efficiency ceiling | **depth-gated, ~0 central** | same | provisional |
| **[contrast] frontier depth event** | — | **+0.19 to +0.29 rail** | complete +1 level at frontier at parity | the Δlc sign test's true unit; 4–15× any L1 efficiency event |

## Bottom line

Under the verified level-number-weighted oracle, the efficiency-denominated
channels are worth **even less than §12.2 implied**: tu93's realistic L1-speed
value is **+0.017 rail**, ft09 and ls20 price to **0** as efficiency lines, and
the (a)/(b) efficiency ceilings collapse to depth-gated ~0 (matching W1's
fires-doesn't-pay). A single frontier depth event is **4–15× larger**. All of the
above is **PRE-SCORECARD/PROVISIONAL**; the binding Δlc sign test is untouched and
**no sealed threshold moves**.
