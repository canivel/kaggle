# Independent verification — 2026-07-21/23 (agent-recomputed from raw artifacts)

## Discrepancies (4, none fatal to the gate record)
1. Fork band claim "0.76-1.33" WRONG → fork never drew below 0.82; correct fork band 0.82-1.33 (n=10); pooled all-arm band 0.76-1.33 (0.76 = war #5).
2. LB pricing STALE: model reproduces bit-exact (E[max@107]=1.3897, P(1.44)=0.2917) but 4 new sub-best draws (0.92, 0.93, 1.14, 0.82) → recomputed on 15-draw pooled ledger (mean 0.962, σ̂ 0.1444): **E[max@~102]≈1.35, P(touch 1.44)≈0.18** (down ~40%).
3. Grinder doc: banking feasible number understated (own-method computes +0.208/draw for ft09+sc25+re86, doc says ≤+0.15, conservative-direction); +0.31 stack ceiling is asserted overlap discount of +0.37 sum, not computed; 0.56x rail→LB factor = registered assumption (0.4-0.8 band).
4. Prose undercounts fatals: R15 raw = 1 fatal (prose said 0 new); R14 raw = 3 (prose implied 1).

## Verified exact (every printed digit)
- Pooled n=11 mean 0.965455 σ̂ 0.153972; war ledger + both sealed looks reproduce exactly (gate p=0.22483, prong-ii mean −0.13229; A5 CI-hi 0.3761 FAIL).
- EWM: step-0 aborts = 1331/1362 (97.7%); carrier top-5 confirmed; **tr87 CONFIRMED ALIASED-UNRESOLVED on binding holdout (Wilson LB 0.927 < 0.95, EWM no-go)** → clean carrier set {tn36, tu93, ls20, ft09-L1}.
- Banking Δ(max2) table exact (+0.2866/draw across 8 games).

## Throughput audit 07-13→07-23 (the stuck-claim, quantified)
Scored windows: 11 (0 errors) — 2 carried new code (both lines since killed), 4 war-control resubmits, 5 filler. **9 consecutive windows with zero new mechanism live (ongoing).** Panels R10-R17: 8 rounds / 34 verdicts / **0 ACCEPTs** / 169 majors / 9 fatals; A14 unsealed after 4 rounds. Mechanisms built+validated ≥7; ever live in scored window: 2 (both killed). Infra incidents on 8 of 11 loop-days; ~3 full days lost; 1 window unused + 1 manually recovered.
Raw ledger: submissions_raw.csv. Full agent output in session transcript 2026-07-21.
