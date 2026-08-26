You are Professor of ML Systems (GPU inference, vLLM serving, wall-clock budgets, quota economics; kills plans that don't fit the compute envelope).

You are reviewer #5 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026).

GROUND-TRUTH LEADERBOARD STATE (refreshed 2026-08-09 from the live Kaggle API; the
draw-by-draw submission ledger is at runs/lb_ground_truth.md; treat THESE numbers as canonical —
any different numbers you may remember from earlier rounds were a stale briefing):
# LB ground truth — refreshed 2026-08-09 (live Kaggle API: submissions + leaderboard head)

**STEP-1 verification (state-of-campaign review, 2026-08-09, independent recompute):** live LB
re-pulled (`kaggle competitions leaderboard`, top-20 archived `runs/lb_daily/lb_2026-08-09.csv`) —
head order matches this doc exactly (KOJIMA 1.86 #1, Andy liu 1.69, Lord Han Solo 1.65, GeniusYY
1.64; gold/top-13 line 1.58; top-5 prize line 1.61; our 1.33 below #49). Full submission ledger
re-pulled from the Kaggle API (36 scored rows since 2026-06-26). Independent recompute of the daily
draw distribution: over the 26 most-recent draws the mean is 0.931–0.939 and σ̂ 0.160–0.166
depending on ledger-membership convention (whether the 07-24 A21 exploration draw and the five
war-v1/scheduler draws are counted); the daily process's canonical **n=26, mean 0.9365, s 0.1540**
falls inside that envelope and is confirmed to central-tendency. **The strategic point is
unchanged and is the headline: our per-draw MEAN is ~0.93, and per host thread 729985 the prize is
the PRIVATE twin of the two selected submissions, not public E[max] — see
`learnings/state_of_campaign_2026-08-09.md` §A.**

Refresh 2026-08-09: incorporated the 08-09 00:07Z frozen draw **0.89** (API COMPLETE,
frozen-fork filler) → record ledger **n=26, mean 0.9365, s 0.1540**. Interior draw
(z ≈ −0.31 vs n=25 stats), ABOVE the 0.80 line — the resolved-STATIONARY watch-rule
does **NOT** re-arm (needs a fresh consecutive sub-0.80 pair); second consecutive
interior recovery (0.87 → 0.89), trailing-4 mean 0.9025. LB cross-check 08-09T~06:00
local (archived `runs/lb_daily/lb_2026-08-09.csv`, top-20): KOJIMA 1.86 frozen #1
(resubmitted 08-09 00:03, unchanged); head order static (Andy liu 1.69, Lord Han Solo
1.65 resubmitted unchanged, GeniusYY 1.64). **GOLD CUTOFF MOVED: top-13 line rises
1.56 → 1.58, ending a four-day flat** — Helmut AGI enters at 1.61 (#7) and the whole
1.58 pack (Biubiu, ippeiogawa, Nkosi Ndwandwe, Yuchen20) is now #10–13, pushing
Mathurin Ache / anngle / NoOneAhead (1.56) out of the gold band to #14–16. Top-5 prize
cutoff HOLDS at 1.61 (Tecnod8.AI, FOYSAL — Helmut AGI is the third 1.61 and lands #7
on tiebreak). Our 1.33 below #49; gap to gold widens to 0.25. No builds in flight
(compaction lane DEAD 08-07; `arc3-duck-compaction-eval` COMPLETE-terminal); day
session = R24 FULL PANEL on the successor-lane proposal + weekly KAOS/fingerprints.

Refresh 2026-08-08: incorporated the 08-08 00:07Z frozen draw **0.87** (API COMPLETE,
frozen-fork filler) → record ledger **n=25, mean 0.9384, s 0.1569**. Interior draw
(z ≈ −0.45 vs n=24 stats), ABOVE the 0.80 line — the fired-and-resolved-STATIONARY
watch-rule (0.77, 0.78 on 08-06/08-07) does **NOT re-fire**; it re-arms only on a
future sub-0.80 pair. Third-lowest of the last 5 draws but unremarkable. LB cross-check
08-08T~06:00 local (archived `runs/lb_daily/lb_2026-08-08.csv`, top-20): KOJIMA 1.86
frozen #1 (resubmitted 08-08 00:00, score unchanged); head order unchanged (Andy liu
1.69, Lord Han Solo 1.65 — resubmitted 08-07 unchanged, GeniusYY 1.64); **gold cutoff
(top-13) HOLDS at 1.56 — fourth flat day** (#13–15 Mathurin Ache / anngle / NoOneAhead);
top-5 prize cutoff holds 1.61 (Tecnod8.AI, FOYSAL); 1.58 pack unchanged (Biubiu,
ippeiogawa, Nkosi Ndwandwe, Yuchen20) + cstl 1.59. Our 1.33 below #49. No builds in
flight (compaction lane DEAD 08-07; kernel COMPLETE-terminal); day session = R24 prep
(Prime Agent / Tycho portability assessment → successor-lane proposal doc).

Refresh 2026-08-07: incorporated the 08-07 00:09Z frozen draw **0.78** (API COMPLETE,
frozen-fork filler) → record ledger **n=24, mean 0.9413, s 0.1596**. Interior-low draw
(z ≈ −1.06 vs n=23 stats), but it is the SECOND consecutive sub-0.80 (0.77 → 0.78):
**the pre-registered two-consecutive watch-rule FIRED** — a stationarity re-check is
owed (precedent: 08-02 fire → NC-15 repro, verdict stationary; note both draws sit
just under the 0.80 line vs the 0.65/0.68 dip, so prior is tail-noise, but the check
must run). Trailing-4 mean 0.9325. LB cross-check 08-07T~06:00 local (archived
`runs/lb_daily/lb_2026-08-07.csv`, top-20): KOJIMA 1.86 frozen #1 (resubmitted 08-07
00:30, score unchanged); head order unchanged (Andy liu 1.69, Lord Han Solo 1.65,
GeniusYY 1.64); **gold cutoff (top-13) HOLDS at 1.56 — third flat day** (#13–15 all
1.56, Mathurin Ache new name at the line); 1.58 pack now 4 names + cstl 1.59 above it.
Our 1.33 below #49. A22 v2.1 (pure-eviction, digest-OFF) kernel v3 COMPLETE overnight —
banner canary + seed-1 K3 screen is the day-session's first action (K3 FAIL ⇒ lane DEAD).

Refresh 2026-08-06: incorporated the 08-06 00:07Z frozen draw **0.77** (API COMPLETE,
frozen-fork filler) → record ledger **n=23, mean 0.9483, s 0.1594**. Low draw
(z ≈ −1.18 vs n=22 stats): below the quoted 0.82 band-low but above the 0.65 record
low; FIRST sub-0.80 since the 07-31/08-02 dip (0.65/0.68) — **watch-rule ARMED, not
fired** (fires on two consecutive sub-0.80; prior draw was 1.21). Snapping back from
the 1.21 high to 0.77 in one day re-confirms the frozen artifact's own variance spans
~0.6 of LB. LB cross-check 08-06T~06:00 local (archived `runs/lb_daily/lb_2026-08-06.csv`
— first day of the daily CSV archive, process-gap fix from the 08-04 intel sweep):
KOJIMA 1.86 frozen #1; head order unchanged (Andy liu 1.69, Lord Han Solo 1.65,
GeniusYY 1.64); gold cutoff (top-13) HOLDS at 1.56 (second flat day); top-10 cutoff
1.58, the dense pack at 1.58 persists and gained a NEW name (Nkosi Ndwandwe) — shared
public-artifact signature strengthening. Our 1.33 below #49; A22 v2 build COMPLETE
overnight, banner read + M1 screen is today's action.

Refresh 2026-08-05: incorporated the 08-05 00:07Z frozen draw **1.21** (API COMPLETE,
frozen-fork filler) → record ledger **n=22, mean 0.9564, s 0.1582**. Interior but strong
(z ≈ +1.76 vs n=21 stats): highest draw since the 07-18 record 1.33; no band change
(0.82–1.33 holds), no watch-rule (rules watch the low side); fourth consecutive
interior draw — stationarity verdict keeps holding, and the high tail is a reminder
the frozen artifact's own variance spans ~0.7 of LB. LB cross-check 08-05T~12:30Z:
KOJIMA 1.86 frozen; **NEW #3 Lord Han Solo 1.65**; gold cutoff (top-13) HOLDS at 1.56
(first non-rising day since 07-28), top-10 cutoff 1.58 with a dense 4-way pack at
1.58 — the pack suggests a shared public artifact at 1.58, i.e. the effective
"published ceiling" may have moved above the 1.47 boristown anchor. Our 1.33 slide
continues; A22 v2 (region-aware eviction) is the lane — eval push today,
measurement-only.

Refresh 2026-08-04: incorporated the 08-04 00:07Z frozen draw **0.97** (API COMPLETE,
frozen-fork filler) → record ledger **n=21, mean 0.9443, s 0.1514**. Interior draw
(z ≈ +0.17 vs n=20 stats): no watch-rule fire, no band change; third consecutive
interior draw post the 0.65/0.68 dip (stationarity verdict holding). LB cross-check
08-04T~13:00Z: head frozen (KOJIMA 1.86, Andy liu 1.69, GeniusYY 1.64); **gold cutoff
(top-13) risen AGAIN to 1.56** (08-03: 1.54; 07-28: 1.49), top-10 cutoff 1.58; new
name FOYSAL 1.61 at #5. Drift rate ~0.02/day at the gold line — our 1.33 keeps
sliding on pure drift; only a mechanism win moves us (A22 v2 is the lane).

Refresh 2026-08-03: incorporated the 08-03 00:07Z frozen draw **0.99** (API COMPLETE,
"frozen-fork filler (eternal fallback)") → record ledger **n=20, mean 0.9430, s 0.1552**.
0.99 is interior (z ≈ +0.31 vs prior n=19 stats): no band change, no watch-rule fire; the
0.65/0.68 dip did NOT continue (consistent with the NC-15 stationarity verdict — n₂=2 tail
artifact, not a regime change). Paired harm-pause rule (trailing-4 −1.5s) applies to gated
arms only; not evaluated on filler draws. LB cross-check 08-03T~12:30Z: head frozen
(KOJIMA 1.86), #2 Andy liu 1.69 (NEW since 08-02), #3 GeniusYY 1.64; **gold band risen
again — 1.54 is now ~#13–14, top-10 cutoff ≈ 1.56–1.58** (was 1.54 on 08-02, 1.49 on
07-28). Our 1.33 continues its pure-drift rank slide.

Refresh 2026-08-02: incorporated the 08-02 00:07Z frozen draw **0.68** (API COMPLETE,
"frozen-fork filler (eternal fallback)") → record ledger **n=19, mean 0.9405, s 0.1590**.
SECOND consecutive sub-0.80 control filler (0.65 → 0.68): pre-registered watch-rule FIRED;
stationarity re-check executed (`learnings/sweeps/stationarity_2026-08-02.md`): MK no-trend
(p=0.65); change-point Welch |t|=8.64 after draw 17 — **CORRECTED per NC-15 repro
(`learnings/sweeps/nc13_nc15_discharge_2026-08-02.md`): permutation p=0.0117 (memo's 0.0032
overstated ~3.7×), NOT significant at 0.01; with min-segment≥3 the break collapses (|t|=1.40,
p=0.72, an n₂=2 tail artifact); pipeline false-alarm calibrated (1.04% vs nominal 1%). Record
is CONSISTENT WITH STATIONARITY.** σ=0.24 regime REJECTED by our record (χ² p=0.0073 at sealed
n=15) — struck from decision rules. Original memo verdict INCONCLUSIVE-PROCEED-WITH-GUARD
superseded by the repro; the A/B hold rests on NC-14 (mechanism-null), not on drift. LB cross-check 08-02T12:25Z (2011 teams):
our 1.33 INTACT at **#65** (churn only), head frozen (KOJIMA 1.86), gold cutoff #13 = **1.54**.

Account: canivel (Danilo Canivel, d.canivel@gmail.com). Competition:
arc-prize-2026-arc-agi-3. Verification command:
`uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`.

- OUR BEST (public LB): **1.33** (frozen-fork filler draw, 2026-07-18). Current rank
  **#63** (leaderboard CSV pull 08-01: team "Canivel" at #63, 58 teams strictly above, 7
  tied at 1.33 spanning ranks 59–65 — the 07-28 #51 → 08-01 #63 slip is pure competitive
  drift from other teams climbing the dense band, NOT any change to our banked draw, which
  is byte-for-byte intact at 1.33).
- LEADER: YUTO KOJIMA **1.86**. #2 Tecnod8.AI 1.61, #3 DhanaLakshmiMalla 1.60,
  #4 ippeiogawa / Yuchen20 1.58. Gold cutoff ≈ **1.49** (top 13; #13–14 both 1.49,
  #15 = 1.48). Dense band 1.46–1.61 unchanged (boristown's public 1.47 seeding).
- External context: Claude Opus 5 posted 30.2% on the ARC-AGI-3 benchmark (arcprize.org,
  Jul 24) via API at High reasoning effort — different regime (unconstrained API vs
  Kaggle quantized/time-limited local), no artifact to lift; directional support for
  capability-over-harness.
- The "best 0.43 / leader 1.56" figures in pre-R19 briefings were a STALE HARDCODED
  TEMPLATE (May-era), root-caused and fixed 2026-07-24 (panel_round.py now reads this
  file). Reconciliation: 0.43 was the team's best in early May (forge-era agents);
  the frozen duck fork lifted the floor to the 0.82–1.33 band from 2026-07-05 on.

## Draw-by-draw scored ledger (all API-verified)

Frozen-fork record ledger (n=20): 0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14,
0.82, 1.05, 0.84, 1.02, 0.90, 1.03, 0.85, 1.10, 0.65, 0.68, 0.99 → mean 0.9430, s ≈ 0.1552
(recomputed `uv run python`; was n=17 / 0.9729 / 0.1332 before the 08-01 draw — mean
dropped 0.018, s widened 0.013 from the single 0.65 low draw). **A/B control
parameters remain FROZEN at n=15 (mean 0.9727, s 0.1343) per the sealed prereg draft
`learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md` §3 — later fillers accrue to
the record, not to the sealed control.** War arm (n=5, CLOSED per A9):
0.91, 1.08, 0.88, 1.05, 0.76. Sentinel exploration arm (n=1, HARM-PAUSED 07-24, SHELVED
by disposition memo; R22 disposition: pre-registered un-shelve rule adopted, queued
BEHIND the boristown readiness-gate A/B): 0.71.

Recent tail (newest first): 0.65 filler (08-01, campaign-low draw) · 1.10 filler (07-31) · 0.85 filler (07-30) · 1.03 filler (07-29) · 0.90 filler (07-28) · 1.02 filler
(07-27) · 0.84 filler (07-26) · 1.05 filler (07-25) · 0.71 sentinel (07-24) · 0.82
filler (07-23) · 1.14 filler (07-22).

Refresh 2026-08-01 (live API `competitions submissions` + full leaderboard CSV
2026-08-01T12:28Z): incorporated the 08-01 00:07Z frozen draw **0.65** (API status
COMPLETE, description "frozen-fork filler … record ledger n=17 after 07-31 draw 1.10; A/B
control frozen n=15…"). This is a **campaign-low draw** (prior min 0.71 sentinel / 0.76
war / 0.82 filler; below the historical 0.82–1.33 filler band). Record stats recomputed
numerically (`uv run python`): n=18, mean 0.9550, s 0.1500 (was n=17 / 0.9729 / 0.1332).
z(0.65) = **−2.40** vs the frozen n=15 control, **−2.42** vs the n=17 record. TAIL
ARITHMETIC: P(single draw ≤ 0.65 | N(0.9727,0.1343)) = 0.81% (Gaussian) / 1.78%
(t-predictive ν=14); **P(≥1 of 18 draws ≤ 0.65) = 13.7% (Gaussian) / 27.6% (t)** —
tail-consistent with a stationary frozen distribution, NOT distribution-shift evidence
(a −2.4σ single draw is expected roughly one time in seven-to-four over 18 draws). LB
cross-check (live top-20 CSV, 2001 teams): head UNCHANGED from 07-31 — KOJIMA 1.86 #1,
Andy liu 1.69 #2, GeniusYY 1.64 #3, Tecnod8.AI 1.61 #4, DhanaLakshmiMalla 1.60 #6; gold
cutoff drifted UP to **#13 = 1.54** (#15–16 at 1.50, #17–18 at 1.49) from normal
new-submission churn in the dense 1.47–1.61 band (FOYSAL 1.61, Nkosi 1.58, paul/Seok 1.54
new). **Our banked 1.33 is intact at #63 (team "Canivel"): 58 above, 7 tied, ranks
59–65.** No platform-wide rescoring / game-set change / eval-infra shift: the top scores
are frozen and our own historical draw was NOT rewritten. Verdict: **isolated left-tail
low draw**, no band change re-classification (the band floor is now formally 0.65 as a
record low, but this is a single tail observation, not a regime), no drift signal, no
trigger; A/B control stays frozen at n=15 (0.9727/0.1343) per prereg §3 (no drift/harm/
control-invalidation clause is touched by a control-arm draw — see
`learnings/sweeps/draw_deepdive_2026-08-01.md`).

Refresh 2026-07-31 (live API `competitions submissions` + leaderboard head 2026-07-31):
incorporated the 07-31 00:07Z frozen draw **1.10** (API status COMPLETE, description
"frozen-fork filler (eternal fallback)"). Record stats recomputed numerically (`uv run
python`): n=17, mean 0.9729, s 0.1332 (was n=16 / 0.9650 / 0.1334). 1.10 is interior
(z ≈ +0.95 vs the frozen n=15 control): no band change, no drift signal, no trigger; A/B
control stays frozen at n=15 (0.9727/0.1343) per prereg §3. Leaderboard head cross-check
(live CLI top-20): KOJIMA 1.86, #2 Andy liu 1.69 (NEW at #2 — first movement in the head
since 07-24; submitted 07-30 06:42Z), #3 GeniusYY 1.64, #4 Tecnod8.AI 1.61,
#5 DhanaLakshmiMalla 1.60, then 1.58×3. Gold cutoff moved UP: #14–15 at 1.50, #16–17 at
1.49 → cutoff now ≈ **1.50** (was 1.49). Dense band 1.47–1.61 intact; boristown 1.47
now #19–20 band.

Refresh 2026-07-30 (live API `competitions submissions` + leaderboard head 2026-07-30):
incorporated the 07-30 00:07Z frozen draw **0.85** (API status COMPLETE, description
"frozen-fork filler (eternal fallback)"). Record stats recomputed numerically (`uv run
python`): n=16, mean 0.9650, s 0.1334 (was n=15 / 0.9727 / 0.1343). 0.85 is interior
(z ≈ −0.91 vs the frozen n=15 stats): no band change, no drift signal, no trigger; A/B
control stays frozen at n=15 per prereg §3. Leaderboard head cross-check: KOJIMA 1.86,
#2 Tecnod8.AI 1.61, #3 DhanaLakshmiMalla 1.60, #4 ippeiogawa/Yuchen20 1.58; gold cutoff
still ≈ **1.49** (#13–14 at 1.49, #15 = 1.48); 1.46–1.61 dense band unchanged (boristown
1.47 and twin at #16–17).

Refresh 2026-07-29 (live API `competitions submissions` + leaderboard head): incorporated
the 07-29 00:07Z frozen draw **1.03** (API status COMPLETE, description "...n=14 after
07-28 draw 0.90..."). Stats recomputed numerically: n=15, mean 0.9727, s 0.1343 (was
n=14 / 0.9686 / 0.1384). 1.03 is interior (z ≈ +0.44 vs prior stats): no band change, no
drift signal, no trigger. Leaderboard head cross-check: leader KOJIMA 1.86, #2 Tecnod8.AI
1.61, #3 DhanaLakshmiMalla 1.60, gold cutoff 1.49 (#13–14 at 1.49, #15 = 1.48) — all
unchanged from 07-28.

Refresh 2026-07-28 (live API `competitions submissions` + full leaderboard CSV
2026-07-28T11:24Z): incorporated the 07-28 00:07Z frozen draw **0.90** (API status
COMPLETE, description "frozen-fork filler ... n=13 after 07-27 draw 1.02"). Stats
recomputed numerically: n=14, mean 0.9686, s 0.1384 (was n=13 / 0.974 / 0.143 —
mean −0.005, s tightened). 0.90 is interior (z ≈ −0.53 vs prior stats): no band
change, no drift signal, no trigger. Leaderboard cross-check: our best 1.33 rank #51
(47 strictly above, 6 tied at 1.33), leader KOJIMA 1.86, gold cutoff 1.49 (#13–14 at
1.49, #15 = 1.48) — all unchanged from 07-27.

Refresh 2026-07-27 (live API `competitions submissions` + full leaderboard CSV):
incorporated the 07-26 (0.84) and 07-27 (1.02) frozen draws that previously existed
only in briefs (stale-at-n=11 flagged by panel R21 directive #3). Both cross-checked
against runs/submission_log.jsonl (ok=true, arc3-duck-repro v3, trusted-fork
preflight). Recomputed stats agree exactly with
learnings/artifacts/result_deepdive_2026-07-27.md (n=13, mean ≈ 0.974, s ≈ 0.143) —
no discrepancy.

External anchors: byte-identical public forks of the same duck artifact family have
drawn 1.39 (zoli800) and 1.47 (boristown agi-duck-harness-fast-eval, whose only real
functional diff is a vLLM readiness gate — see
learnings/war_room/fork_diff_boristown_2026-07-24.md). Artifact tail ≥ 1.47 confirmed.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

### [FATAL] The P1 screen is a fixed-wall-clock tournament, so §6.2's primary gate confounds mechanism with per-turn latency — and §2.3's "in-sandbox reasoning is free" is false on this rail

`runs/kernel_pulls/war_eval_v1/arc3-duck-war-eval.log` L657 and `runs/kernel_pulls/a22_v2_1/arc3-duck-compaction-eval.log` L660 both print `max_actions_per_game=None, max_runtime_s_per_game=7920.0`. The two `summary.txt` files report `total wallclock: 198068.6s` and `198115.2s` over 25 games — i.e. **7922.7s and 7924.6s mean per game against a 7920s cap: all 50 game-runs hit the guillotine**. Actions are therefore an *output* of throughput, not a fixed budget: baseline took 3,638 actions, A22 v2.1 took 3,994 in the identical 2h12m. §2.3 consequence #1 explicitly discharges the Prime-Agent homework item "in the favourable direction" because in-sandbox reasoning costs zero *scored* actions — but under a 7,920s cap it costs the one resource that is actually exhausted, and P1's whole design intent is to induce the model to compute more inside the sandbox (30s of tool wall clock per call is ~0.55 of a mean inter-action interval, since 3,638 actions / 198,069s = one action per 54.4s). A P1 arm that is genuinely better per-turn but 15% slower per turn will read as `mean Δlc` harm and trip the inherited −0.128 gate, and per §6.3 K3 that PAUSES the arm; a seed-2 repeat of the same latency artifact kills it. This is exactly how a lane dies for a systems reason and gets recorded as a mechanism negative — which is the fourth such event this campaign would be risking. **Fix, and it is cheap and offline:** pre-register per-game reporting of turns, actions, generated tokens, and cumulative tool wall-seconds for both arms; add a pre-registered *secondary* Δlc computed at matched action prefix (truncate each game-pair to `min(actions_baseline, actions_P1)` from the already-pulled transcripts); and pre-commit that a §6.2 FAIL accompanied by a >10% action-count deficit is reported as INCONCLUSIVE-ON-LATENCY, not as K3.

### [MAJOR] The concurrency constant in the §6.2 safety canary is wrong — the rail runs 28, and all 25 games run simultaneously

§6.2 pins "live-child count ≤ concurrency (16)"; §4 likewise justifies excluding `rlm()` sub-agents because they "contend for the same GPU that already serves 16 concurrent games." 16 is the *dataclass default* (`solver.py` L746) and the `--concurrent-jobs` argparse default (`run.py` L1276). The production eval rail overrides it: `concurrency=28` in both pulls, and the per-game wallclock arithmetic above confirms all 25 games were live for the entire run. The persistent-child steady state is therefore **25 simultaneous children, +56% on the assumed budget**, and the canary as written would either fire on every healthy run (25 > 16) and be misread as a fault, or get silently "corrected" post hoc — the exact class of post-seal parameter movement §6.2's pinned-free-parameter clause exists to prevent. Restate the canary as `live_children ≤ observed_concurrent_games` with the constant read from the run's own solver banner, and re-derive every per-child resource budget at 25, not 16.

### [MAJOR] `_set_limits` has no memory rlimit, and that is safe only because children currently die within 30s

`python_tool_sandbox.py::_set_limits` (L252–267) sets `RLIMIT_CPU`, `RLIMIT_FSIZE` (1 MB) and `RLIMIT_NOFILE` (32) — and **nothing for address space or data segment**. Today a runaway allocation is bounded by the fact that the child is torn down after at most one 30s call. P1 inverts that by design: the child now lives for a whole game (~67 LLM turns/game at 1,686 turns / 25 games) and the added system-prompt sentence instructs the model to *keep its world model there*. 25 children with unbounded, monotonically-growing heaps on a shared Kaggle kernel is an OOM path, and a kernel OOM is a whole-notebook ERROR — the single largest recurring failure family in §A4's table (`class:ERROR:none`, n=7). §6.2's canary list covers child faults but names no memory instrument at all. Add `RLIMIT_AS` (or `RLIMIT_DATA`) to `_set_limits` at a pre-registered per-child ceiling, report peak child RSS per game, and pre-register the aggregate ceiling (25 × ceiling) that must hold below the kernel's RAM.

### [MAJOR] The lifecycle risk that actually breaks K4 is the per-call *timeout* path, not `RLIMIT_CPU` — and the proposal names only the latter

§3(a) correctly flags that `RLIMIT_CPU` becomes per-game and "will kill a long-lived child silently" (it is `timeout+1` = 31 cumulative CPU-seconds, which ~67 turns will exhaust). But the more probable namespace-destroying event is the host-side deadline in `run_sandboxed_python` L499–509: on any single call exceeding `timeout_seconds`, the host calls `_kill_process_group(process)`. In a persistent design that one timeout **destroys the world model for the remainder of the game**, and the ephemeral/restart fallback in §6.2 then silently converts the rest of that game to a control condition. Two consequences neither §6.2 nor §6.3 covers: (i) `namespace_reuse_rate` becomes structurally near-zero for any game that timed out early, so K4's 0.15 floor can fire on an infrastructure artifact and be recorded as "27B does not use the substrate" — the precise misreading K4 was written to prevent; (ii) a child that dies and is reused rather than restarted returns an error on every subsequent call, producing a run that is VALID by K1 (banner present, ≥1 namespace event) but is 100% tool-failure and reads as mechanism harm. Add: per-game child-restart count and first-restart turn index; per-game tool-error rate vs the baseline pull; `namespace_reuse_rate` reported per game *and* restricted to turns after the last restart; and pre-register that K4 is evaluated only on games with zero restarts, with the restart count itself gating validity.

### [MAJOR] Three concrete state-leak traps in the child bootstrap that would violate §6.1 without §6.1 detecting them

The bootstrap creates `stdout = io.StringIO()` and `action_results = []` **once inside `main()`** (L320–321) and the host spawns a fresh `_stdout_reader` thread per call on the same pipe (L481–486). Naively persisting the interpreter therefore yields: cumulative stdout returned on every turn; `action_results` reporting every action of the whole game each turn (which changes `payload["result"]` from a single dict to `{"action_calls": N, ...}` at L1650–1656 of `tool_agent.py`); and, if the reader thread is respawned per call, two threads racing on one pipe. Each of these changes the *rendered tool payload* — which then interacts with the 1024-token `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` truncation and alters the context bytes. §6.1 audits `evicted_chars` and the output of `_trim_messages_for_context()`, i.e. the trim layer; the corruption enters one layer upstream and could pass the stated invariant while the arm is no longer additive. Extend the §6.1 byte-identity assertion to the rendered tool payload for a matched no-op code path, and add explicit per-call reset of stdout/`action_results`/`sandbox_cwd` to the patch spec.

### [MINOR] The binding budget is GPU-hours/week, not pushes/day — §6.6 quotes only the looser constraint

§6.6 states the budget as "max 2 kernel pushes per day." The real ceiling is the ~30 GPU-h/week free quota (`duck_eval/README.md`, `duck_eval/a17/gate_eval_readiness_2026-07-30.md` L97) against ~2.2–2.4 GPU-h per build, i.e. **≈12–13 builds/week, a sustained rate below 2/day**. To be fair to the authors: S2 as scoped *is* affordable — zero builds have run since 08-07, so this week's quota is untouched, and A22 v1/v2/v2.1 each got a canary-clean run on their first push, so "one push" is a defensible point estimate. The gap is that the *sequence* S2 → seed-2 → S4 is a minimum of three builds and three calendar days (build ≈2.2h, read next session), plus one extra build per K1/K2 VOID, and the phase-1 precedent (ITERATION_LOG L304–305: kernel v4 ran v1 defaults because the dataset half of the cycle was not synced) shows the dataset-then-kernel sequence has burned a push before. State the S2 budget as "1 push, +1 reserved for a K1/K2 VOID re-push," and denominate the lane's plan in GPU-h/week.

### [MINOR] Addendum §A3's two "free harness checks" do not apply to our substrate, and §A4 mis-assigns P1's execution risk

Per the brief's instruction to flag contradictions: §A3 tells us to check, before any P1 push, that "Swarm hardcodes `record=True` (~1 GB JSONL per game)" and that "the base `Agent` never prunes `self.frames`." Neither symbol exists anywhere in the duck/taaf substrate P1 patches — grep for `record=True` / `class Swarm` / `self.frames` returns **zero** matches under `duck_eval/`, and hits only under `kaggle-data/ARC-AGI-3-Agents/` and the `runs/runs-202604*/` snapshots, i.e. the dead forge-era stack. The empirical scale is also off by ~2.5 orders: the complete 25-game `war_eval_v1` pull is **224 MB total** (artifacts 105 MB, transcripts 34 MB), not 25 GB. Those checks are no-ops and must not be used to justify any recording change riding on the P1 push. Separately, §A4 asserts P1's kernel-side code "would sit on the same `provenance:scratch-built → ERROR` path that produced five incidents" — that is wrong for P1 as designed (§6: monkeypatch of `run_sandboxed_python` via `canivel/arc-war-kit`, cell-12 hook untouched, zero notebook change, which is precisely the fork-never-build discipline that ended that failure family on 2026-07-08). The fingerprint warning is correct and should be retained for S5/L1, not for S2.

---


=====================================================================

THE PROPOSAL (sha256 of the full document: 86cb963fe620d9fd; full length 35094 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# State of the Campaign — ARC-AGI-3 (Kaggle) — master strategic review, 2026-08-09

**Author:** campaign orchestrator (this is an orchestrator-authored synthesis; sub-analyses are
cited inline, but every load-bearing number here was re-checked against the repo or the live Kaggle
API this session).
**Purpose:** a full, honest strategic review of the whole campaign, written to be torn apart by the
R25 adversarial panel (run on Opus 5). It is deliberately unsparing: the point is to decide whether
we have a path to gold or whether the campaign is structurally capped, and to say so either way.
**Deadline:** 2026-11-02. **~85 days remain.**

Provenance convention: **[V]** verified by direct read this cycle / against the live API; **[V-ours]**
read from our repo; **[SR]** self-reported on a non-official venue (community LB), always de-rated;
**[INF]** inference, asserted by no source.

---

## 0. Executive summary (read this if you read nothing else)

1. **We are ~#70 at 1.33 public.** Gold (top-13) is **1.58 today** and drifting **+~0.01/day**; the
   Nov-2 gold line projects to **~1.6**. Our score has not moved since the 07-18 record draw of 1.33;
   the rank slide (#49→#70) is pure field drift on a frozen artifact. [V, live LB]
2. **We are optimizing the wrong random variable, and we know it.** Host thread 729985 (Greg
   Kamradt) confirms private-LB scores are **banked at each submission's original run time and never
   re-run at selection**, and **every scored run plays both public and private sets**. The prize is
   the **private twin** of the two submissions we select on Nov 2 — not public `E[max@k]`. Public is
   a noisy proxy (ρ<1). Per-draw **MEAN** (≈0.93) and **capability** are the currency; public
   luck-harvesting pays *nothing* at selection unless its private twin is good. [V, intel_sweep_2026-08-04 §1]
3. **The mechanism graveyard is real but its headstones are now suspect.** Warpack, hypothesis
   ledger, sentinel, EWM, the 72B wall-closer, and A22 compaction were all killed or shelved by
   sealed gates. **On 2026-08-09 the A22 death certificate was VACATED**: the −0.128 non-harm
   threshold behind three lane deaths is an *arithmetic identity, not a statistic*, and A22 v2.1's
   headline −0.360 harm is bit-for-bit the gap between two identical no-compaction runs (seed noise
   on a lucky-outlier baseline). **The instrument, not the strategy, is what has been broken.** [V,
   r24_minutes §3.3a; gate_recalibration_2026-08-09]
4. **The field moved to state-externalizing harnesses and it is all open-source.** Three independent
   teams (Schema, Tycho, Prime Agent) reached public-set saturation by moving load-bearing state
   **out of the context window into an executable, verifiable artifact**. Tycho is Apache-2.0; Prime
   Agent is MIT. But all three run at **frontier scale** (Opus 4.8/5); **no weak-model ablation
   exists anywhere**, and our in-kernel actor is **Qwen3.6-27B-FP8** on local vLLM. [V, intel_sweep §2; SR]
5. **The real strategic question is portability, and it is genuinely open.** Can a state-externalizing
   architecture be ported to 27B / 96GB in-kernel and lift the *private* twin above ~1.6? R24
   ratified this lane (a) unanimously — and then five reviewers took the *instrument* apart. As of
   today we cannot yet screen it: the binding resource is wall clock (not actions), the non-harm gate
   is miscalibrated, our exec-wm falsifier is degenerate, and the artifact schema is literally not
   constructible in our sandbox. **These are all fixable and mostly free — but they are unfixed.**
6. **Honest position:** we are **structurally capped below gold on the current artifact** (its own
   family tail tops out at 1.47 on byte-identical public forks; gold is 1.58→~1.6). The only
   plausible route to gold is a **capability lift on the private twin**, which means porting a
   state-externalizing arch to 27B. That port is **unproven at our scale and unscreened**. It is the
   campaign's single best remaining shot, and it may fail on the substrate question alone. **This
   document recommends we run it as a time-boxed, kill-gated program and pre-commit to the fallback.**

---

## A. The selection-currency correction — the most important thing we learned late

For most of the campaign the whole apparatus — the daily filler lottery, `sched_pooled_ev.py`,
`E[max@k]`, the "band 0.82–1.33" framing — was denominated in **public max**. That was wrong.

**Host-confirmed facts (thread 729985, ~Jul 27; ADOPTed 08-04):** [V-ours, intel_sweep_2026-08-04 §1]
1. Private-LB scores are computed at each submission's **original run time and are NEVER re-run** at
   selection.
2. **Every scored run plays BOTH datasets** (public + private) in the same ≤9h run; the public LB
   shows only the ~50% public slice.
3. Wall clock is exactly 9h.

**Consequences, which re-price everything:**
- Each daily draw is a **joint (public, private) pair**. Public is a noisy proxy for the private twin,
  correlated only through same-run luck. Our banked 1.33 has a **fixed, unknowable private twin**.
- **`E[max@k]` on public is largely irrelevant to the prize.** What we select on Nov 2 is two
  already-banked private scores. Chasing a high public draw buys rank-cosmetics and Milestone-2
  eligibility (public-judged) but **not the prize** unless the private twin rode up with it.
- **Per-draw MEAN and capability are the currency.** A capability lift shifts the *whole joint
  distribution* and therefore pays on both sets — it "pays double." Public luck pays nothing at
  selection if its private twin is mediocre.
- **The frozen-fork family tail bounds the pure-draw ceiling.** Byte-identical public forks of our
  own artifact have drawn **1.39 (zoli800)** and **1.47 (boristown)** [V-ours, lb_ground_truth]. So our
  artifact has ≥1.47 *somewhere in its public draw distribution* — but even that tail top is **below
  the 1.58 gold line**. A pure-draw strategy cannot reach gold even on a lucky public draw, and the
  private twin of such a draw is not guaranteed to be the good one.

**What this retires and what it keeps.** It retires: `E[max]`-driven window pricing, the idea that
"one more filler draw" is meaningfully +EV on the prize, and any framing that treats the public 1.33
as "our position" for prize purposes. It keeps: **daily cadence** (every filler still banks a private
draw at ~zero cost and maintains the 33-day streak / eternal-fallback discipline), and the control
ledger as the **denominator** for gate arithmetic (its value is that it is a clean stationary process,
confirmed repeatedly — most recently the 08-07 watch-rule firing resolved STATIONARY).

**The uncomfortable corollary:** because the private twin is unknowable, we cannot *measure* progress
on the actual prize target from the LB at all. We can only measure it indirectly, on the free build
rail, via paired non-harm/lift screens against a same-config baseline. **That makes the build-rail
instrument the single most important asset in the campaign — and §C shows it was broken.**

---

## B. The graveyard — every killed lane, and what each one actually taught

This is the honest ledger of mechanism work. Throughput first, because it frames everything:
**panels R10–R23 produced 0/34 ACCEPTs and ~169 MAJORs; ≥7 mechanisms were built-and-validated;
2 ever went live on the scored rail; both were killed.** [V-ours, stuck_review_v2_2026-07-23 §0]

| Lane | Status | What killed it | What it taught |
|---|---|---|---|
| **Warpack / banking** (max-over-plays replay of winning traces) | Shelved | War-v1 sigma draws n=5; A5/A8 look FAILED at n=5 (pooled σ̂ 0.154); banking never fired on the scored rail (`bank_min_time_s=120`, every game time-starved) | Banking is a **public-set optimization by construction** (replays what we already solved), colliding with the generalisation rail. Root cause of the fire failure is now known & deterministic (N5 `prune_trace` phase desync + time-starvation) but the lane is a variance-efficiency complement at best, not a gold route. |
| **Hypothesis ledger** (typed causal memory, banking=max) | Never cleanly live | Banking integrity carried UNVERIFIED on the record; folded into warpack, died with it | Additive typed memory is plausible (Reki +0.098 local anchor) but **"generic memory has no notion of being wrong"** — the round-18 unfalsifiability charge. Survives as **component arm P3** inside lane (a), not a lane. |
| **Sentinel** (budget/stuck telemetry, v2 @150) | SHELVED 07-24 | "Certified observable, no lift channel": fires but doesn't pay (21/22 fired games kept grinding, +618 actions, no depth gain under the completion-weighted scorer); eval-rail Δlog1p(RHAE) −0.315/−0.166; scored draw 0.71 (harm-paused) | A mechanism can be **real, live, and verified and still have no scoring channel**. Efficiency observables don't help under a scorer that gives **zero credit for unfinished levels**. Passive telemetry retained ≈free. [V, sentinel_disposition_2026-07-24] |
| **EWM / exec-wm** (executable per-game simulators, Rodionov-style) | Effectively closed; schema-only survivor | Round-18 [MAJOR]: **no world-model fidelity metric → unfalsifiable by construction**. Then the "91.7% held-out `state_exact`" turned out to be `split=all` (never held out), and on-trajectory accuracy for a fixed sim swings 0.026–0.879 — a range larger than any carrier threshold. | The central lesson was **misdiagnosed**: we thought we had 22/24 high-fidelity sims; we had a selection artifact evaluated on the training tuples. Only ~3 sims hold hidden state; 0/25 implement abstention. **This is why lane (a) needs Tycho's abstention+coverage channel** — precisely to answer the unfalsifiability charge. [V, r24_minutes §3.5] |
| **72B wall-closer (A17)** (Qwen2.5-VL-72B-AWQ, the "harness is multimodal" bet) | DEAD 07-30 | B2a: ΣN=5 executed actions vs 138 needed; two seeds concordant. **Format non-compliance livelock**: 0/1008 native hermes tool-calls, freeze at t=720s, byte-identical re-emission; actions only in the first ~96s via fenced fallback. ρ_action=96 ≫ 3.5 kill line. | A bigger/multimodal model is **worthless if it can't emit the tool-call format the harness parses**. The Qwen2.5-VL-72B × hermes-parser contract is broken and not permitted to change between seeds. **Successor queue exists** (GLM-4.6V-AWQ native-FC, gpt-oss-120b) but the lane is expensive and unscreened. [V, a17_v7_concordance_2026-07-30] |
| **JEPA** (learned world-model MCTS) | DEAD (3 strikes v45/v62/v63) | Build always succeeds, rerun always ERRORs on Kaggle | Structural: don't build kernels from scratch; the ERRORs traced to wrong `agents/__init__.py` + `.env`. Now guarded by `scripts/preflight.py`. |
| **A22 compaction** (context eviction + digest, LightMem-style) | **VACATED 08-09 — UNRESOLVED, not dead** | Three K3 strikes (−0.200/−0.320/−0.360) — **but the gate was miscalibrated and the baseline was a lucky outlier (see §C)** | Two survivors independent of the death: (1) **eviction-after-externalisation ordering** — eviction harms when it deletes the *only* copy, is safe when a verified external artifact exists first (reconciles A22 with Tycho's own `tail_evict`); (2) **the M3 confound** — refuted-list re-proposal moved −6.49pp with the injection channel provably closed, so a metric can move hard when the mechanism is absent. |

**The single most expensive systemic lesson:** we killed lanes on a build-rail non-harm gate that we
never calibrated. §C is that story.

---

## C. The broken instrument — R24's real product, and why it reframes the whole graveyard

R24 (2026-08-09) was convened to pick a successor lane. It did (lane (a), unanimously). But its
*substantive* product was discovering that **our measuring instrument, not our strategy, is what has
been failing** — and then vacating a death certificate it had ratified hours earlier.

**C1. The non-harm gate fails a true null ~half the time.** [V, r24_minutes §3.2; gate_recalibration_2026-08-09]
Measured on the 90 null-vs-null pairs in `runs/null10` (10 identical vanilla runs × 25 games):
mean-Δlc leg FPR **12.2%**, worst-game leg **50.0%**, conjunction **51.1%**. **A −2 on some game is
the modal null outcome.** The −0.128 threshold was estimated against a 10-run-averaged baseline but
applied against a single run — "inherited verbatim" transported the digit, not the calibration.

**C2. The −0.128 threshold is an arithmetic identity.** It is exactly (12 − 15.2)/25 — i.e. "an arm
that scored 12 levels." Its source arm (`sentinel_eval_v1`, 12 levels) is **not** the arm it was
applied against (`war_eval_v1`, 22 levels — which itself screens at **+0.272, p=0.0074**).

**C3. A22's death is bit-for-bit seed noise on a lucky baseline.** `war_eval_v1/v2/v3` are **three
runs of the identical warpack config** (byte-identical `git_status.txt`, 07-14/15/16) scoring
**22/15/13** levels. `v3 − v1 = −0.360 worst −2` — **exactly A22 v2.1's headline "harm," with no
compaction in either run.** Warpack run variance is **4.83× vanilla's** (p=0.038), so `null10`
understates the null for warpack arms; and A22 was screened against `war_eval_v1`, the **22-level
high outlier** of its own family (flagged as lucky back on 07-16). Re-baselined on the 3-run mean:
**v1 +0.013, v2 −0.107, v2.1 −0.147 — all PASS a calibrated line.**

**C4. The fix — K3′, effective immediately.** Pair against the **per-game mean of m ≥ 3 same-config
baseline runs**; PASS iff `mean Δlc ≥ −t(0.95, df=m−1)·s_base·√(1+1/m)`. Fallbacks: **−0.200 at m=1**
(2.2% type-I), **−0.190 at m≥3**, **−0.160 at α=0.10**. **Worst-game leg dropped** (structurally
uninformative at 25 games). Pairing inflation corrected to **1.28×** (not the reviewer's 1.78×) — the
primary defect was the *number*, not only the procedure.

**C5. What this costs us going forward.** Every future arm now needs **m ≥ 3 same-config baseline
runs before it can be screened at all**, and a **warpack-specific null** before any warpack-family
screen. On a rail that yields ≈**12–13 builds/week** (30 GPU-h ÷ 2.2–2.4h; and the auto-spawned
save&run twin can be cancelled to recover ~9 GPU-h/day, per intel_sweep §5), that is a real budget
increase. **This is the tax we pay for having been sloppy earlier.**

**C6. The honest reframe of §B.** We do **not** get to claim A22 is alive — lane (a) was ratified on
independent grounds and keeps the budget, and the compaction push budget is still better spent
elsewhere. But we **must** stop citing the graveyard as evidence that "mechanisms don't work here."
At least one lane we buried was killed by its own baseline's lucky draw measured with a broken ruler.
**No lane may be declared dead on the old gate.** The "eviction is intrinsically harmful" story — which
*motivated the entire successor-lane rationale* — is no longer evidenced by our own data (the
eviction-after-externalisation reconciliation now rests on 2608.01326 and Tycho, not on our −0.360).

---

## D. The field shift — state-externalizing harnesses, all open

Three independent teams reached near-saturation on the 25 public games by externalizing state into an
executable, verifiable artifact: [V, intel_sweep §2; r24 proposal §2.1; all [SR]]

| Team | Model | Reported [SR] | Artifact |
|---|---|---|---|
| **Schema** (Impossible Research) | **Opus 4.8** + Fable 5 | ~99% RHAE; Claude-Code baseline 42.83% | 50 traces + dependency-free scorer on HF |
| **Tycho** (arXiv:2607.28287, NIMI) | Opus 4.8 (matched), Opus 5/GPT-5.6 (frontier) | RHAE **88.49** matched best-of-4; **100.00** frontier = metric ceiling, 183/183 levels, −61% actions | **Apache-2.0**, github.com/NIMI-research/Tycho |
| **Prime Agent** (Prime Intellect) | Opus 5 | **95.5%** (Best@3 99.97%), 183/183 | **MIT** TypeScript monorepo, no ARC-specific code |

**The de-rating that must ride with every one of those numbers (now a STANDING RULE):** all three live
on the ARC Prize **community leaderboard** — self-reported, harness-driven. The **official** LB
(vendor-model-only, harnesses excluded) reads **Opus 5 = 30.2%**, GPT-5.6 = 7.8%, Opus 4.8 = 1.5%.
Never place 88.49/95.5/99 and our 1.33 in the same sentence — different venues, different verification.
"Public ARC-AGI-3 is saturated" is a claim about **harness-design maturity on a self-reported venue**,
**not** about our private-set position. Tycho's "100.00" is the *ceiling* of the RHAE formula (full
completion pins the second term), i.e. "maxed the benchmark," not "beat it by 61%." [V, rules_verification_2026-07-28; r24 proposal §2.1]

**What survives the discount is the convergence, not any number.** Two out-of-domain 08-06 papers
confirm the architecture from outside ARC: **AppDeltaWorld** (next observation as executable code
deltas) and **MASS** (logic-engine/rendering-engine split = Tycho's state-vars + `render()`). The
decisive detail: **Schema reached the regime with Opus 4.8, a previous-generation model** — which
weakens "only frontier-latest works" while saying **nothing** about 27B.

**The reconciliation with our own A22 finding** (this is the intellectual core of lane (a)):
> Eviction is not harmful per se. Eviction is harmful when it is **not preceded by externalisation.**
> A22 deleted the only copy (harm ∝ `evicted_chars`, ρ=−0.403). Tycho deletes the *second* copy — the
> first lives in `world_model.py` + `notes/` and is re-derivable by replay. Same operation, opposite
> sign; the discriminator is whether a **verified** external artifact exists first. **[INF]**

Prime Agent corroborates from the third team: it compacts **generation-side** (`generateSummary()`,
never splits a tool result), not selection-side — exactly the class distinction 2608.01326 proves and
our −0.200/−0.320/−0.360 measured (before we learned that number was noise).

---

## E. The port lane (a) and the R24 six FATALs — verbatim, with proposed fixes

R24 ratified lane (a) — **state-externalisation, Tycho as the artifact schema (not a competing lane),
(b) additive typed memory as a component arm, (c) banking on its own clock** — and then filed **six
FATALs** (methodology filed two) plus 27 MAJORs, all **against instrument specification, not the
strategic choice.** Four of five reviewers said the lane and the free offline work should be authorised
today with only the S2 seal held. The six FATALs verbatim [V, learnings/panel/round24/*.md]:

1. **rl-planning [FATAL]:** *"§2.3's budget model is wrong on the team's own data, and it is
   load-bearing for the one authorised push."* — `war_eval_v1/benchmark.json`: `final_wallclock_seconds`
   is **7920.2–7939.9 s for every one of the 25 games**; ~54 s per scored action; caps never
   approached. **The binding resource is LLM latency under concurrency, not the scored-action budget.**
   In-sandbox reasoning is **not** free — it converts directly into fewer actions. *Fix (free):*
   re-derive §2.3 against the wall-clock regime; add latency + matched-action-prefix instrumentation;
   drop or jointly-report `actions_per_level_completed`.
2. **systems [FATAL]:** *"The P1 screen is a fixed-wall-clock tournament, so §6.2's primary gate
   confounds mechanism with per-turn latency — and §2.3's 'in-sandbox reasoning is free' is false on
   this rail."* — the Δlc gate **cannot separate "harmful" from "slower"**; P1 could die on latency
   and be recorded as the fourth mechanism negative. *Fix (free):* pre-register mean/p95
   seconds-per-scored-action, actions/game, tokens/turn, peak child count/RSS with a voiding band.
3. **methodology [FATAL #1]:** *"The inherited non-harm gate has a measured 51% false-FAIL rate under
   a true null; 'inherited verbatim' is rhetorical, not legitimate."* — see §C. *Fix:* **done** — K3′.
4. **methodology [FATAL #2]:** *"S1b's kill condition selects on the outcome: it excludes both
   documented failures and tests a proposition already recorded as proven."* — the 11-game
   "prefix-splice-safe" set is *defined* as the clean det=1.000 games; both games that actually threw
   step-0 `frame_divergence` (**sc25, m0r0**) are excluded; full unpruned replay already survived all
   25 on record. *Fix:* re-point the S1b gate at sc25/m0r0; run under a time-starvation regime.
5. **llm-agents [FATAL]:** *"P1's mechanism is a prompt change that the proposal forgot to read the
   existing prompt for; K4 can only fail invalidly."* — the harness tells the model its Python is
   ephemeral in ~6 places (`prompts.py:80,82,107`, tool schema `:1347`, etc.), all of which §6 keeps;
   at 27B tool schema beats system prompt, so a `<0.15` reading measures instruction conflict, not the
   substrate. Also: `SAFE_MODULES` lacks `dataclasses`/`typing`/`enum` — **the Tycho `State` dataclass
   is not constructible in our sandbox as configured.** *Fix (free):* promote those strings to declared
   patch surface; add the missing safe modules; restate the §6.1 byte-identity invariant as a
   drop-*policy* invariant (it is self-voiding — `_trim_messages_for_context` passes `tools` into its
   token estimate, so any prompt/tool edit voids the arm on event 1).
6. **prog-synthesis [FATAL]:** *"L0's protocol cannot be run on the artifacts it names, and its gate
   cannot pass by construction."* — all 25 sims are `simulate(state, action_id, x, y)` and **0
   implement abstention**, so Tycho "coverage" is the constant 1.0 (the channel meant to answer the
   round-18 unfalsifiability charge is **degenerate**); only 3 sims hold hidden state; "expand beyond
   ~4 games" is unreachable and circular (real abstention needs L1, which L0 is supposed to gate).
   Bonus: the "91.7% held-out `state_exact`" was **never held out** (`split=all`). *Fix:* re-scope L0
   before any run; abstention requires building L1 first (now affordable — workstation authoring ruled
   in-bounds).

**Ratified and uncontested (now policy):** lane (a) with Tycho as artifact schema; workstation-LLM
authoring **in-bounds** of the zero-budget rail (no metered API spend; artifacts committed &
byte-audited before riding in a dataset; provenance disclosed); provenance de-rating a standing rule;
sandbox host-mode acceptable only while executed code is ours/offline/byte-audited (re-taken if
model-authored in-kernel code lands); refuted-list micro-arm dropped as a standalone.

**Engineering-lift reality (from the proposal, [V-ours]):** the duck is **two mechanisms short of the
RLM shape, not ten** — `ToolAgent._tools()` already exposes exactly one `python` tool, `step_env` is
already passed into the sandbox (the model already calls `action([...])` inside its own Python), and
the board is already externalised. Missing: a **persistent namespace** (P1: 1–2 push cycles, the
subprocess+JSON protocol exists; work is child lifecycle + `RLIMIT_CPU` per-game re-accounting) and
**durable cross-level memory** (P3: the un-wipe — `_summarized_knowledge` is wiped on level transition
and game-over, exactly when later-level-dominant scoring makes it most valuable; ~1 cycle). The
**large** item is L1 (sim interface migration; the June generation pass cost ~10h / ~7M tokens for 24
games) — affordable only now that workstation authoring is ruled in-bounds.

**What is explicitly NOT in scope** (and why gold-via-full-Tycho is not simply "port it"): Tycho
budgets **3,500 LM calls per game** against our **1,686 turns total (≈67/game): a 52× gap**; its
builder subagent, `/refine` self-editing, and `rlm()` sub-agents all either blow the budget, contend
for the single GPU serving ~25 concurrent games, or make runs non-reproducible.

---

## F. Assets we own (the campaign is not starting from zero)

- **A working, saturated-understanding engine stack:** the Cottaar duck harness fork (Milestone-1
  winner) runs clean; **12 exec_wm sims are fully saturated** (100% on-tuple), the board is already
  externalised, one `python` tool, `step_env` in-sandbox. Two mechanisms from the target shape.
- **Deep game mechanics knowledge:** per-game forensics (borro1980 variance map — **2 games ≈ 65% of
  ledger variance**), determinism/aliasing audits, the N5 phase-desync root cause, the sc25/m0r0
  blocker inventory.
- **Verified infra & discipline:** `scripts/preflight.py` (blocks the 5-ERROR structural-drift class),
  fork-never-build rule, dataset-code-sync banner checks, canary discipline, a **33-day submission
  streak** with an eternal-fallback frozen fork, daemon automation (`ARCDailySubmit` @18:37,
  `ARCDailyIterate` @08:23, morning-check). Failure fingerprints: **no new failure family in ~4.5
  weeks**.
- **A clean, calibrated measurement denominator:** the frozen-fork control ledger (n=26, mean ~0.9365,
  σ̂ ~0.154), repeatedly confirmed stationary (change-point p=0.757, Mann-Kendall p=0.62). **Now paired
  with a calibrated numerator (K3′).**
- **An adversarial governance process:** 24 panel rounds, preregistration amendments A1–A25 +
  conditions C1–C7, sealed gates, harm-pause machinery validated end-to-end. It is slow (0/34
  ACCEPTs) but it **caught the broken instrument** — that is the process working, not failing.
- **Two free falsifiers ready to fire** once re-scoped (L0 sim re-verification; offline bank re-fire),
  a wall-closer model queue (GLM-4.6V-AWQ, gpt-oss-120b), and the Schema HF trace set as a no-regret
  dataset attach.

---

## G. Honest position and the real strategic question

**Position.** Public **1.33 (~#70)**; gold **1.58** and drifting to **~1.6** by Nov 2; **~85 days
left**. Our artifact's own public family tail tops out at **1.47** — **below gold**. On the current
artifact we are **structurally capped below gold**, on both the public proxy and (by inference) the
private twin, because a pure-draw strategy cannot manufacture capability it does not have.

**The real question — stated plainly:**
> Is porting a state-externalizing architecture to **27B / 96GB in-kernel** viable enough to lift the
> **private twin** above ~1.6 within ~85 days — or is the campaign structurally capped below gold, and
> should we say so and stop spending scored windows pretending otherwise?

**The case that it is viable:** the duck is 2 mechanisms short of the shape; three teams converged on
it; one (Schema) did it with a *previous-gen* model; the metric rewards it (completion-dominant, later
levels linear-weighted, our sims already exist); the eviction-after-externalisation theory is
coherent; the fixes to our instrument are free; and workstation authoring (the L1 cost) is ruled
in-bounds.

**The case that it is not:** **every saturating result is at frontier scale and no weak-model ablation
exists anywhere.** The most likely null mode is 2608.04828's **affordance-trigger failure** — a 27B
model may simply never *use* a namespace it is told about (K4). Tycho's own H2 shows fidelity is
**necessary, not sufficient** (its `trigger` policy hit 88.1% transition match and still *lost* on
completions). The 52× LM-call-budget gap means we can only afford the *substrate*, not the full loop.
And even a working port has to beat **1.6 on the private set**, which we can never directly observe
before Nov 2 — we would be betting the campaign on a free-rail non-harm/lift screen as a proxy for an
unobservable target.

**My orchestrator read (stated for the panel to attack):** the port is **worth exactly one
time-boxed, kill-gated program**, because it is the only lever that can move the private twin and
because it is now *cheap to falsify* — K4 (`namespace_reuse_rate`, properly instrumented) and the
re-scoped L0 answer the substrate question **without a scored draw**. If K4 fails, we have our answer
and we stop. If it passes, we have earned the right to spend pushes on P3/L1. **We should NOT bet the
campaign on it silently; we should bet a bounded budget on a decisive falsifier and pre-commit to the
"accept the band" fallback if it fails.**

---

## H. Decision options (EV / cost / risk) — the actual choice

These are mutually-informative, not mutually exclusive; the recommendation is a specific combination.

### Option 1 — Port the state-externalizing arch to 27B (the capability bet)
- **What:** Fix the instrument (K3′ done; add latency + matched-action-prefix endpoints, patch-surface
  strings, `SAFE_MODULES`, warpack null). Re-scope and run the **free** falsifiers first: L0
  (carrier-set expansion under threading+abstention, needs L1 built) and the affordance-compliance
  transcript pass. Then **P1 persistent-namespace screen** (1 push), gated on `namespace_reuse_rate ≥
  0.15` properly defined. If P1 passes non-harm **and** shows exploration change, proceed to **P3
  un-wipe** (separate arm) and staged **L1–L4**.
- **EV:** The only option with a path to gold (private-twin capability lift). If the port works even
  fractionally, Tycho's fixed-model ablation is worth +9.4 RHAE — a large lever. **But EV is
  bimodal:** high if K4 passes, ~zero if 27B won't use the substrate.
- **Cost:** ~1 push for P1 + m≥3 baseline runs each screened arm; L1 authoring ~10h workstation
  (in-bounds, $0). Weeks of the ~85 remaining. Free-rail budget ~12–13 builds/week is the real
  constraint, not money.
- **Risk:** the substrate-null (2608.04828) is the modal outcome; the 52× budget gap caps how much of
  Tycho we can run; latency confound could still masquerade as harm if instrumentation is incomplete;
  and success on the free rail is only a *proxy* for the unobservable private twin.
- **Kill gate:** `namespace_reuse_rate < 0.15` (properly instrumented) ⇒ lane NULL on the substrate,
  no further pushes. This is the cheap, decisive off-ramp.

### Option 2 — Screen the wall-closer model tier (the other capability bet)
- **What:** Free-rail screen **GLM-4.6V-AWQ** (106B MoE, native tool-calling VLM, fits 96GB), then
  gpt-oss-120b, on the existing duck harness — a bigger/native-FC model is the only known mechanism for
  a +0.2-class jump at fixed harness, and it directly addresses the A17 death cause (format-livelock).
- **EV:** Potentially large and *orthogonal* to Option 1 (better model helps any harness). But the
  entire 72B line died on exactly this route (format non-compliance), so the prior is guarded.
- **Cost:** GPU-heavy screens (each ~2.2–2.4h build); throughput penalty vs 27B may zero out the turn
  budget under the 9h wall clock (the A17 lesson: fewer-but-smarter turns lost).
- **Risk:** high — format-parser contract, decode-throughput collapse under the wall clock, no in-harness
  public evidence any of these models work.
- **Relationship to Opt 1:** a **parallel free-rail track**, not a substitute — different failure modes,
  can share the weekly build budget.

### Option 3 — Accept the band; optimize for Milestone-2 and streak, stop spending on mechanism
- **What:** Concede gold is out of reach on the private twin; keep the daily frozen-fork cadence (banks
  a private draw at ~zero cost), harvest the best public draw for **Milestone-2** (Sep 30, public-judged,
  top-3 — though at 1.33 vs ~1.64 we are not in reach there either), and stop spending scored windows
  and free-rail budget on unproven mechanism.
- **EV on the prize:** ~zero (we do not reach gold), but **honest** and **zero further cost/risk**. This
  is the correct choice *if and only if* Option 1's K4 falsifier fails.
- **Cost/Risk:** none beyond opportunity. The risk is choosing it **prematurely** — before running the
  now-cheap, now-calibrated falsifier that could still surprise us. §C is a direct warning against
  declaring things dead too early.

### Option 4 (RECOMMENDED) — Time-boxed port with a pre-committed off-ramp
- **What:** Run **Option 1's free, decisive falsifiers first** (instrument fixes + re-scoped L0 +
  affordance pass + one P1 screen), on a **hard 3-week box** (through ~Aug 30), **in parallel** with a
  single free-rail GLM-4.6V-AWQ screen (Option 2's cheapest leg). Keep the daily cadence throughout
  (Option 3's zero-cost banking). **Pre-commit now:** if K4 fails and the GLM screen format-livelocks,
  we adopt Option 3 and say so publicly-internally — the campaign is capped, and we stop pretending.
- **EV:** captures nearly all of Option 1's upside (the falsifiers are the high-VOI part) while bounding
  the downside to 3 weeks and ~2–3 pushes; the pre-committed off-ramp prevents the R10–R23 failure mode
  of grinding a dead lane forever.
- **Cost:** ~3 weeks, free-rail-dominated ($0 cloud), 1–2 scored pushes at most (none if the falsifiers
  kill it first).
- **Risk:** the main risk is *analysis paralysis* — the R24 fixes are free but must actually be built,
  not just enumerated. Second risk: 3 weeks is ~⅓ of remaining time; if we over-run the box we lose the
  chance to bank an honest fallback cleanly. **The box and the off-ramp are the whole discipline here.**

**Recommendation: Option 4.** It is the only option that both preserves the one real shot at the prize
(a private-twin capability lift) and pre-commits to the honest fallback the moment the cheap falsifier
says no — which, after §C, is the discipline this campaign most needs.

---

## I. Open questions carried to R25 (for the panel)

1. What replaces the non-harm gate beyond K3′ — is a worst-game leg salvageable at 25 games, or must
   everything become a mean/quantile statistic?
2. Is lane (a) screenable **at all** on a wall-clock-bound rail? Is latency + matched-action-prefix
   instrumentation sufficient, or does the 9h guillotine make any in-sandbox-reasoning arm
   un-measurable in principle?
3. **P1 before P3, or P3 before P1?** `transcript_forensics.md` ranks a persistent hypothesis ledger
   (=P3) as fix #1; the only 27B-on-our-substrate precedent (Feng, [SR], 177th) is memory; P3 is
   cheaper; and P1's "cheap decisive falsifier" advantage evaporated under FATAL #5. Note §3(b) is
   factually wrong — `cross_level_notes` is **deliberately not wiped**, so P3 reverses a design choice
   and must argue against it.
4. Is L0 rescuable, or does the abstention gap mean exec-wm cannot be re-verified without building L1
   first — and is that the right first spend now that workstation authoring is in-bounds?
5. Does A22 get re-screened under K3′, or left formally open and unworked? (Recommendation: unworked —
   lane (a) keeps the budget — but the death record is vacated either way.)
6. **The meta-question for the panel:** given the private-twin selection currency (§A) and that we can
   never observe the target before Nov 2, is a free-rail non-harm/lift screen a *legitimate* proxy to
   bet the campaign's remaining effort on — or is the honest move to size the port purely as
   information-gathering and pre-commit to Option 3?

---

*Bottom line for the minutes: the artifact is capped below gold, the prize is the private twin we
cannot see, the field's winning shape is open-source but only proven at frontier scale, and our own
measuring instrument was broken badly enough to have buried at least one lane by mistake. The
recommendation is a 3-week, kill-gated port of the state-externalizing shape to 27B with the fallback
pre-committed — spend the cheap decisive falsifiers before the one push that matters, and take the
honest off-ramp the moment they say no.*

## END OF PROPOSAL ##

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
