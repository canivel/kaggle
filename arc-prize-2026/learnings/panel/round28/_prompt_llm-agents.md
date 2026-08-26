You are Professor of LLM Agents and Scaffolding (tool-use, agentic harnesses, prompt-based control of foundation models; reviews for NeurIPS/ICLR; allergic to 'we will prompt it better' hand-waving).

You are reviewer #2 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026).

GROUND-TRUTH LEADERBOARD STATE (refreshed 2026-08-23 from the live Kaggle API; the
draw-by-draw submission ledger is at runs/lb_ground_truth.md; treat THESE numbers as canonical —
any different numbers you may remember from earlier rounds were a stale briefing):
# LB ground truth — refreshed 2026-08-23 (live Kaggle API: submissions + FULL leaderboard)

Refresh 2026-08-23: the 08-23 00:07Z draw scored **1.63, COMPLETE** (`canivel/arc3-q38-field-eval` v1,
A21 FIELD-FLOOR REDRAW draw 3). **★ NEW CAMPAIGN BEST — this retires the 1.33 banked 07-18.**
The certified field-floor config is now **THREE-FOR-THREE: 1.59 / 1.58 / 1.63, mean 1.600, spread
0.05**. Three draws inside 0.05 is a **config-level level, not a run of luck**, sitting **+3.8σ above
the frozen-filler ledger** (`runs/ledger.json`: n=37, mean 0.9316, s 0.1771, sealed promotion bar
**1.1087** — the bar drifts with the record; re-read it from `runs/ledger.json` at prereg time,
never cache it). The ledger must **NOT** absorb these three — they are a different config, and a
separate field-floor record is the right instrument. Public max moved 1.59 → **1.63 (+0.04)**, which
UNDERSTATES the event: public score is a MAX over submissions, so what actually improved is the
*config mean* going from a one-draw estimate to a three-draw one — and the config mean is the
quantity the deadline rule selects on (`project_arc_final_selection_rule.md`).

LB cross-check 08-23T10:01Z (full **2489-row** board via `leaderboard -d` → `lb_full_2026-08-23.csv`,
heartbeat asserted `HEARTBEAT OK … rows=2489 … sha=92ca36ee0fbd`, exit 0; top-15 archived to
`lb_2026-08-23.csv` in the real 5-column shape): **OUR RANK IS #250 OF 2489** (VERIFIED, full
download, `Rank` column; 1.63 on 119 lifetime subs). Yesterday **#239 of 2465 at 1.59** ⇒ **we GAINED
+0.04 AND STILL LOST 11 RANKS** — real gain, partially offsetting the ≈30/day field drift measured in
exp_id 29. **Top-5 prize line 2.72 → 2.73**; **gold/top-13 line 2.47 → 2.50**; top-50 **2.14**,
top-100 **1.98**, top-200 **1.73**. Board grew 2465 → 2489 (24 entries, 0 exits). The field-floor
stays the free nightly FLOOR — redrawing it asymptotes near ~2.0, **below a gold line that is already
2.50 and climbing.** A better floor does not close this gap; only a different ceiling does.

CONTROL ARM 08-23 (verbatim, `scripts/lb_diff.py`): `Jack Cole (MindsAI) 2.05, prev 2.05, +0.00,
subs 127, +1 sub, dRank −6, 0.0000/draw`; `Tufa Labs 3.04, prev 3.04, +0.00, subs 115, +1 sub,
dRank −1, 0.0000/draw`. **★ FIRST WINDOW IN WHICH BOTH CONTROL TEAMS ARE FLAT.** Jack Cole is flat a
**fifth** straight day; Tufa — which stepped +0.45 then +0.07 on the two prior windows — stepped
**+0.00** here. Both bought a draw and gained nothing. **The commodity-engine / shared-regime story is
at its WEAKEST yet**: the two teams with the means, the motive and the cadence to swap an engine did
not gain. The 1.55–1.65 band is **FLAT FOR A SEVENTH DAY on score** while growing 14% in population
(37 → 42 teams, median 1.60 → 1.61, median subs 13.0 → 12.5) — that is **arrivals stacking at the
duck-harness level, not a bloc being lifted.** **METHOD UNKNOWN [UNK]** for every team named here, and
none of it can be dated — `LastSubmissionDate` is a team's MOST RECENT submission while `Score` is
their BEST, and they need not be the same submission
(`learnings/top6_evidence_audit_2026-08-15.md`). Today's tally: 0 DISCLOSED / 1 INFERRED / 7 UNKNOWN.

Large moves this window are quoted **per Δsubmission**, never bare: `wking edewd +2.47 on +1 sub
(2.4700/draw, 3 lifetime subs, now #7)`, `automatylicza +1.37 on +1`, `galatea +1.14 on +1 (2 subs)`,
`Lindicaphxag +1.02 on +1 (3 subs)`, `Daniel Franzen +0.30 on +1`. Against them sits a long
DREW-NO-GAIN column — `zhen931`, `shineef`, `stephenn`, `NoOneAhead`, `Vaibhav`, `Munshi-PremChand`,
`GeniusYY`, `ocean240812` each bought **+2 draws for +0.00**. Extra draws buy score for free under a
max-over-submissions rule; a team climbing that order statistic is not a team whose agent improved.

★ ARCHIVE DEFECT (repaired 08-22, re-checked clean 08-23): `lb_2026-08-20.csv` had held a header and
**15 EMPTY rows**, and `lb_2026-08-21.csv` was **never written**; both were regenerated from their
surviving `lb_full_*.csv`. A broken daily and a healthy one are indistinguishable by filename — the
full-board archive lane is the only reason those days are recoverable. Today's daily was verified
non-empty at write time.

---

# LB ground truth — refreshed 2026-08-20 (live Kaggle API: submissions + FULL leaderboard)

Refresh 2026-08-20: incorporated the 08-20 00:07Z frozen draw **0.41** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro` v3, AUTO-REFILL again — **eighth consecutive day** where the only
thing on the board was the eternal fallback). **★ 0.41 IS A NEW RECORD MINIMUM: it sits 0.24 BELOW
the prior floor of 0.65**, z = **−3.44** against the standing ledger (`runs/ledger.json` still at
n=36, mean 0.9461, s 0.1558, trailing-4 1.0325, max 1.33, sealed promotion bar **1.0848** — the bar
drifts with the record; re-read it from `runs/ledger.json` at prereg time, never cache it). The last
three frozen draws read **1.01 → 1.15 → 0.41**, a −0.74 swing in one night on byte-identical code.
**The record can no longer be called resolved-STATIONARY without re-derivation** — the low tail has
a counterexample. **NOTE: the ledger has NOT yet been re-derived to include 0.41**, so every n=36 /
bar-1.0848 figure is stale by exactly one draw. **Public max UNCHANGED at 1.33 since 2026-07-18**
(the public score is a MAX over submissions, so a record-low draw costs rank nothing — what it costs
is the variance estimate every promotion gate is priced against).

LB cross-check 08-20T10:00Z (full **2428-row** board via `leaderboard -d` → `lb_full_2026-08-20.csv`,
heartbeat asserted `HEARTBEAT OK … rows=2428 … sha=a8b81e66bd8a`, exit 0; top-15 archived to
`lb_2026-08-20.csv`): **THE TAIL INFLATED FOR A FIFTH STRAIGHT NIGHT.** **Top-5 prize line
2.53 → 2.58** and **gold/top-13 line 2.24 → 2.37**. **OUR RANK IS #299 OF 2428** (VERIFIED, full
download, `Rank` column; 1.33 on 116 lifetime subs). Yesterday #261 ⇒ **−38 ranks in ONE day, −82 in
two, −124 in three, −169 in four, on a byte-unchanged 1.33.** **Gap to gold widens 0.91 → 1.04; gap
to the prize line 1.20 → 1.25.** Threshold counts: **53 teams ≥2.00** (was 33), 73 ≥1.90 (was 49),
113 ≥1.75 (was 76), 154 ≥1.65 (was 112), 162 ≥1.62 (was 124), 177 ≥1.58 (was 141), 206 ≥1.50
(was 169), 237 ≥1.44 (was 192). **p99 2.10 → 2.22; median UNCHANGED at 0.26 for a fifth day — the
tail moves, the body does not.** Board n went 2408 → 2428 on **21 entries and 1 exit**. Top of board:
**cstl 3.57 FLAT, still #1 by +0.81**, Lord Han Solo 2.76 flat, **The AGI Boys 2.66 (+0.95 on one
draw)**, Jonathan Wang2022 2.59, Daniel Franzen 2.58 flat, **cocoaAI 2.56 (+1.32 on TWO draws →
0.6600/draw)**, rellik13 2.53, **Pathetic384 2.47 (+1.56 on one draw, 4 lifetime subs)**.

**★ THE CONTROL ARM SPLIT — after two clean double no-move days, TUFA MOVED AND COLE DID NOT.**
`lb_diff.py` §1, verbatim: **`Jack Cole (MindsAI)` 2.05 → 2.05, +0.00 on +1 sub (124 lifetime),
Δ/draw 0.0000, DREW-NO-GAIN, dRank −18**; **`Tufa Labs` 1.62 → 2.07, +0.45 on +1 sub (112 lifetime),
Δ/draw 0.4500, STEP, dRank +79** — its verdict line: *"the shared-regime story survives this window.
It is NOT confirmed: a score move is a score move. Method remains UNKNOWN unless someone discloses
it."* **Tufa sat flat for four consecutive draws and then stepped +0.45 in one; Jack Cole is now
flat for a third and lost 18 ranks standing still.** A single control team stepping once is the
**WEAKEST non-null form** this evidence can take. And it is **undercut in the same run** by the
**1.55–1.65 band**: 40 → **34 teams (−6**; 5 entered, 11 left), median score **1.59 → 1.59 (FLAT)**,
median subs 41.5 → 24.5 → **14.5** (**composition artifact** of the churn — survivors and entrants
carry short histories). **Fifth consecutive day the INCUMBENT median does not lift; the band is now
DRAINING upward by exits, not lifting as a bloc** — the opposite of what a drop-in engine swap
would look like. *The merge-blind Δ/draw divisor defect named on 08-17 remains UNFIXED; it will
fabricate a DRIFT verdict on the next merge.* **METHOD UNKNOWN for every team named here.**

**★ WHAT THE DATA STILL SUPPORTS: THE STEPS ARE SINGLE DRAWS ON TINY HISTORIES.** Δscore per
Δsubmission (never bare Δscore — the public score is a max over submissions): **WENJIE_Wang-dev
+2.28 on 1 draw (3 lifetime subs — largest Δ/draw on the board), 史永刚 +1.67 on 1 (3 lifetime),
MarkDjadchenko +1.58 on 1 (TWO lifetime), Pathetic384 +1.56 on 1 (4 lifetime), Dmitry Belan +1.48
on 1 (TWO lifetime), Sid Devinen +1.41 on 1 (4 lifetime), JerrySun +1.30 on 1 (TWO lifetime).**
Board-wide base rate this window: **321 teams submitted, only 114 (35.5%) gained anything, 345 new
submissions, median Δ/draw among GAINERS just 0.2650, max 2.2800** — so +2.28 on one draw is ~8.6×
the median gainer. Counter-evidence in the same table: **Extremis, Araadhay Kanojia, Halla Yang,
Akagha Chimgozirim, Data Dreamers and Jason Feng each bought 2 draws for +0.00.**

**SCOPE.** This instrument measures Score, SubmissionCount, Rank, TeamName and LastSubmissionDate
only. It does NOT observe method, model, engine or prompt, and it **CANNOT date a scoring run** —
`LastSubmissionDate` is a team's MOST RECENT submission while `Score` is their BEST, and they need
not be the same submission (proven on our own row: 1.33 banked 07-18, dated to today's 0.41 filler).
Evidence classes per `learnings/top6_evidence_audit_2026-08-15.md`: **0 DISCLOSED / 1 INFERRED /
7 UNKNOWN.**

---

## Prior refreshes (archive)

# LB ground truth — refreshed 2026-08-19 (live Kaggle API: submissions + FULL leaderboard)

Refresh 2026-08-19: incorporated the 08-19 00:07Z frozen draw **1.15** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro` v3, AUTO-REFILL again — **seventh consecutive day** where the only
thing on the board was the eternal fallback). `runs/ledger.json` re-derived from the API today:
**n=36, mean 0.9461, s 0.1558**, z(1.15) = **+1.36** vs the prior n=35 (0.9403/0.1541) — the
**highest frozen draw since the 1.33 of 07-18**, but **interior to the record and BELOW the bar**,
trailing-4 **0.9675 → 1.0325**, max 1.33, min 0.65, sealed mean-of-4 **promotion bar 1.0848**
(was 1.0778 at n=35 — the bar drifts with the record; re-read it from `runs/ledger.json` at prereg
time, never cache it). Record stays **resolved-STATIONARY**. **Public max UNCHANGED at 1.33 since
2026-07-18.**

LB cross-check 08-19T10:00Z (full **2408-row** board via `leaderboard -d` → `lb_full_2026-08-19.csv`,
heartbeat asserted `HEARTBEAT OK … rows=2408 … sha=6a9dcb3d4efa`, exit 0; top-15 archived to
`lb_2026-08-19.csv`): **THE TAIL INFLATED FOR A FOURTH STRAIGHT NIGHT.** **Top-5 prize line
2.35 → 2.53** and **gold/top-13 line 2.05 → 2.24**. **OUR RANK IS #261 OF 2408** (VERIFIED, full
download, `Rank` column; 1.33 on 115 lifetime subs). Yesterday #217 ⇒ **−44 ranks in ONE day, −86 in
two, −131 in three, on a byte-unchanged 1.33.** **Gap to gold widens 0.72 → 0.91; gap to the prize
line 1.02 → 1.20.** Threshold counts: **33 teams ≥2.00** (was 18), 49 ≥1.90 (was 29), 76 ≥1.75
(was 45), 112 ≥1.65 (was 70), 124 ≥1.62 (was 79), 141 ≥1.58 (was 97), 169 ≥1.50 (was 127),
192 ≥1.44 (was 146). **p99 1.93 → 2.10; median UNCHANGED at ~0.25–0.26 for a fourth day — the tail
moves, the body does not.** Board n went 2383 → 2408 on **26 entries and 1 exit**. Top of board:
**cstl 3.57 is #1 by +0.81** (+0.76 on ONE draw), Lord Han Solo 2.76 FLAT, **Jonathan Wang2022 2.59
(+1.22 on one draw, 8 lifetime subs)**, Daniel Franzen 2.58 flat, rellik13 2.53, **Cyrus 2.43
(+1.24 on one draw, TWO lifetime subs)**.

**★ THE CONTROL ARM IS A CLEAN DOUBLE NO-MOVE FOR THE SECOND CONSECUTIVE DAY.** `lb_diff.py` §1,
verbatim: **`Jack Cole (MindsAI)` 2.05 → 2.05, +0.00 on +1 sub (123 lifetime), Δ/draw 0.0000,
DREW-NO-GAIN**; **`Tufa Labs` 1.62 → 1.62, +0.00 on +1 sub (111 lifetime), Δ/draw 0.0000,
DREW-NO-GAIN** — its verdict line: *"NEITHER CONTROL TEAM MOVED … the commodity-engine /
shared-regime story is WEAK on this evidence."* **Tufa Labs has now spent four consecutive daily
draws for zero gain; Jack Cole two.** The **1.55–1.65 band** agrees: 38 → **40 teams (+2**; 10
entered, 8 left), median score **1.59 → 1.59 (FLAT)**, median subs 41.5 → 24.5 (**composition
artifact of the churn** — the entrants carry short histories). **Fourth consecutive day the
INCUMBENT median does not lift; the band grows by influx, not by incumbents improving.**
*The merge-blind Δ/draw divisor defect named on 08-17 remains UNFIXED; it will fabricate a DRIFT
verdict on the next merge.* **METHOD UNKNOWN for every team named here.**

**★ WHAT THE DATA STILL SUPPORTS: THE STEPS ARE SINGLE DRAWS ON TINY HISTORIES.** Δscore per
Δsubmission (never bare Δscore — the public score is a max over submissions): Aditya Sasidhar
**+2.09 on 1** (→2.21, **2 lifetime subs**), Udit Jain #2 **+1.37 on 1** (→1.47, 2 subs), Daniel #3
**+1.33 on 1** (→1.72, 5 subs), Cyrus **+1.24 on 1** (→2.43, **2 subs**), Jonathan Wang2022
**+1.22 on 1** (→2.59, 8 subs), Jeki Wan Taufik **+0.97 on 1** (→2.13), 骐骥驰骋CreateAMind
**+0.94 on 1** (→1.99), cstl **+0.76 on 1** (→3.57). Measured best-of-N background the same window:
**307 teams submitted, only 116 (37.8%) gained anything**, 357 new submissions, **median Δ/draw
among gainers 0.2575**; Jayaprakash Sundararaj bought **+15 draws for +1.14 (Δ/draw 0.0760)** —
that is what climbing the order statistic looks like. **A +1σ.. +4σ jump on one draw from a 2-sub
history is not.** Evidence classes unchanged: **0 DISCLOSED / 1 INFERRED / 7 UNKNOWN**
(`learnings/top6_evidence_audit_2026-08-15.md`).

**SCOPE.** This instrument measures Score, SubmissionCount, Rank, TeamName and LastSubmissionDate.
It does NOT observe method, model, engine or prompt, and **it CANNOT date a scoring run** —
`LastSubmissionDate` is a team's MOST RECENT submission while `Score` is their BEST, and they need
not be the same submission. **Do not infer method from movement.**

---

# LB ground truth — refreshed 2026-08-18 (live Kaggle API: submissions + FULL leaderboard)

Refresh 2026-08-18: incorporated the 08-18 00:07Z frozen draw **1.01** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro` v3, AUTO-REFILL again — **sixth consecutive day** where the only
thing on the board was the eternal fallback). `runs/ledger.json` re-derived from the API today:
**n=35, mean 0.9403, s 0.1541**, z(1.01) = **+0.46** (interior), trailing-4 **0.89 → 0.9675**,
max 1.33, min 0.65, sealed mean-of-4 **promotion bar 1.0778** (was 1.0777 at n=34 — the bar drifts
with the record; re-read it from `runs/ledger.json` at prereg time, never cache it). Record stays
**resolved-STATIONARY**. **Public max UNCHANGED at 1.33 since 2026-07-18.**

LB cross-check 08-18T10:01Z (full **2383-row** board via `leaderboard -d` → `lb_full_2026-08-18.csv`,
heartbeat asserted `HEARTBEAT OK … sha=dcdea367ccf5`, exit 0; top-20 archived to `lb_2026-08-18.csv`):
**THE TAIL INFLATED FOR A THIRD STRAIGHT NIGHT.** **Top-5 prize line 2.33 → 2.35** and
**gold/top-13 line 2.00 → 2.05**. **OUR RANK IS #217 OF 2383** (VERIFIED, full download, `Rank`
column; 1.33 on 114 lifetime subs). Yesterday #175 ⇒ **−42 ranks in ONE day, −87 in two, −98 in
three, on a byte-unchanged 1.33.** **Gap to gold widens 0.67 → 0.72; gap to the prize line
1.00 → 1.02.** Threshold counts: **18 teams ≥2.00** (was 13), 29 ≥1.90 (was 19), 45 ≥1.75 (was 27),
70 ≥1.65 (was 44), 79 ≥1.62 (was 54), 97 ≥1.58 (was 71), 127 ≥1.50 (was 99), 146 ≥1.44 (was 116).
**p99 1.79 → 1.93; median UNCHANGED at 0.25 for a third day — the tail moves, the body does not.**
Board n went 2365 → 2383 on **18 entries and ZERO exits**. Top of board: **cstl 2.81 is #1 again**
(+0.11 on ONE draw after four flat days), Lord Han Solo 2.76 FLAT (loses #1 after a day),
Daniel Franzen 2.58 flat, rellik13 2.53, **Tanaka Ai24 2.35 (+0.74 on one draw)**,
Fufront-RyanX-AGI-Team 2.33.

**★ THE CONTROL ARM IS A CLEAN DOUBLE NO-MOVE TODAY** (and unlike 08-17 the reading is not
contaminated: the Jack Cole → `@Abstraction Lab` merge now sits on BOTH sides of the diff, so the
Δ/draw divisor is honest). `lb_diff.py`, verbatim: **`Jack Cole (MindsAI)` 2.05 → 2.05, +0.00 on
+1 sub, Δ/draw 0.0000, DREW-NO-GAIN**; **`Tufa Labs` 1.62 → 1.62, +0.00 on +1 sub, Δ/draw 0.0000,
DREW-NO-GAIN** — its verdict line: *"NEITHER CONTROL TEAM MOVED … the commodity-engine /
shared-regime story is WEAK on this evidence."* **Both teams — the TTT originators and the authors
of the harness we fork — spent a real draw and gained nothing; Tufa Labs for a third consecutive
day.** The **1.55–1.65 band** agrees: 37 → 38 teams (+1; 8 entered, 7 left), median score
**1.60 → 1.59 (−0.01)**, median subs 49.0 → 41.5 (**composition artifact of the churn**).
**Third consecutive day the INCUMBENT median does not lift — and today the band barely grew, so it
is not even influx.** *The merge-blind Δ/draw divisor defect named on 08-17 remains UNFIXED; it will
fabricate a DRIFT verdict on the next merge.* **METHOD UNKNOWN.**

**★ WHAT THE DATA STILL SUPPORTS: THE STEPS ARE SINGLE DRAWS ON TINY HISTORIES.** Δscore per
Δsubmission (never bare Δscore — the public score is a max over submissions): jiashuo Ma
**+1.49 on 1** (→1.72, 5 lifetime subs), Norman Neira **+1.26 on 1** (→1.30, **2 lifetime subs**),
Kaiser **+1.12 on 1** (3 subs), mark22 **+1.04 on 1**, greyhound **+0.91 on 1**, muhualiushui1217
**+0.89 on 1**, keithtyser **+0.87 on 1** (→2.13), Evgenii Rudakov **+0.85 on 1** (→1.58), Dong
**+0.82 on 1**, Tanaka Ai24 **+0.74 on 1**, sukeke **+0.71 on 1** (→1.93, 5 subs), Tara
Labs-speedsci **+0.59 on 2**. Board-wide: **308 teams submitted, only 103 (33.4%) gained anything,
330 new submissions, median Δ/draw among gainers 0.2100, max 1.4900.** **These are not the
max-over-N order statistic, and 2–8 lifetime submissions rules out grinding.** Consistent with the
graft-stack conversion trace sealed 08-17, but **not confirmation of it: every one is UNKNOWN, none
disclosed** (`learnings/top6_evidence_audit_2026-08-15.md`). Today's tally: **0 DISCLOSED /
1 INFERRED / 7 UNKNOWN.**

Refresh 2026-08-17: incorporated the 08-17 00:07Z frozen draw **0.80** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro` v3, AUTO-REFILL again — **fifth consecutive day** where the only
thing on the board was the eternal fallback). `runs/ledger.json` re-derived from the API today:
**n=34, mean 0.9382, s 0.1559**, z(0.80) = **−0.91**, trailing-4 **0.885 → 0.89**, max 1.33, min 0.65,
sealed mean-of-4 **promotion bar 1.0777** (was 1.0826 at n=33 — the bar drifts with the record;
re-read it from `runs/ledger.json` at prereg time, never cache it). Record stays
**resolved-STATIONARY**. **Public max UNCHANGED at 1.33 since 2026-07-18.**

LB cross-check 08-17T10:00Z (full **2365-row** board via `leaderboard -d` → `lb_full_2026-08-17.csv`,
heartbeat asserted `HEARTBEAT OK … sha=21624abc24b8`; top-20 archived to `lb_2026-08-17.csv`):
**THE LARGEST SINGLE-NIGHT TAIL MOVE WE HAVE ARCHIVED.** **Top-5 prize line 1.98 → 2.33** and
**gold/top-13 line 1.65 → 2.00**. **OUR RANK IS #175 OF 2365** (VERIFIED, full download; `Rank`
column): **170 teams strictly above**, 1.33 tie block spans **#171–#177** (7 teams). Yesterday #130
with 125 above ⇒ **−45 ranks in ONE day, −56 in two, −75 in three, on a byte-unchanged 1.33.**
**Gap to gold widens 0.32 → 0.67; gap to the prize line widens 0.65 → 1.00.** Threshold counts:
**13 teams ≥2.00** (was 4), 19 ≥1.90 (was 7), 27 ≥1.75 (was 10), 44 ≥1.65 (was 13), 54 ≥1.62,
71 ≥1.58, 99 ≥1.50, 116 ≥1.44, 177 ≥1.33. **p99 1.61 → 1.79; median UNCHANGED at 0.25 — the tail
moved, the body did not.** Top of board: **Lord Han Solo 2.76 is the NEW #1** (+1.11 on a SINGLE
draw), cstl **2.70 FLAT for a fourth day** (+1 draw, no gain, now #2), Daniel Franzen 2.58,
**rellik13 2.53 (#4 on 5 LIFETIME submissions, +1.25 on one draw)**, Fufront-RyanX-AGI-Team 2.33.

**★★ THE CONTROL ARM SPLITS TODAY, AND `lb_diff.py`'s OWN CONTROL LINE IS WRONG — READ THIS, NOT IT.**
The differ printed *"Jack Cole (MindsAI) MOVED +0.14 on 97 new submissions (0.0014/draw) — DRIFT"*.
**That is a TEAM MERGE, not 97 draws.** On 08-16 the board carried **two** rows: `Jack Cole`
(teamId **15587108**, 1.59, 96 subs, solo `jcole75`) and `aRc (binary relation)` (teamId **15490570**,
1.91, 24 subs, 4 members). On 08-17 **15587108 is GONE** and 15490570 is renamed **`@Abstraction Lab`**
with `jcole75` on the roster at **2.05, 121 subs**. **121 = 24 + 96 + 1** ⇒ the merged entity bought
**ONE** real new draw, and its score went **max(1.91, 1.59) = 1.91 → 2.05**, i.e. **≈ +0.14 per draw —
a STEP, misreported as DRIFT by ~100×.** *Evidence class: the ID/roster/count arithmetic is DISCLOSED
by the board; "the 2.05 came from the post-merge submission" is INFERRED — `LastSubmissionDate` is the
team's LATEST and `Score` is their BEST, so it cannot date a scoring run.* **METHOD UNKNOWN.**
**Instrument defect, named not fixed: the differ divides Δscore by ΔSubmissionCount across merges,
where the count jumps by the absorbed team's whole history. Every merge will fabricate a DRIFT
verdict on a real STEP.** By contrast **`Tufa Labs` 1.62 FLAT, +1 sub, Δ/draw 0.0000 — a genuine
no-move, second consecutive day.** The authors of the harness we fork spent a draw and gained nothing.
The **1.55–1.65 band** agrees with the weak-shared-regime read: 25 → 37 teams (**+17 entered, 5 left**),
median score **1.61 → 1.60 (−0.01)**, median subs 46 → 49 (**composition artifact**). **The INCUMBENT
median did not lift; the band grew by influx.** A drop-in commodity engine adopted broadly would raise
the incumbent median. **This does not say nobody swapped an engine — it says the score data does not
show one, and score data names no method.**

**★ WHAT THE DATA DOES SUPPORT: THE STEPS ARE SINGLE DRAWS ON TINY HISTORIES.** Δscore per Δsubmission
(never bare Δscore — the public score is a max over submissions): egangu **+1.96 on 1** (0.05 → 2.01,
**2 lifetime subs**), Syed Salman **+1.43 on 1**, rellik13 **+1.25 on 1**, Lord Han Solo **+1.11 on 1**,
UlinNuhaAbduh **+0.95 on 1**, Kevin E R MILLE **+0.90 on 1**, AI Winter **+0.88 on 1**, Ethan Lee
**+0.81 on 1**, Pascal Ledesma **+1.00 on 2**, SireeshLimbu **+0.84 on 2**. Board-wide: **289 teams
submitted, only 110 (38.1%) gained anything, 415 new submissions, median Δ/draw among gainers 0.2100.**
**You cannot buy +1.96 with one draw off a 0.05 base — these are not the max-over-N order statistic,
and 2–8 lifetime submissions rules out grinding.** Something cheap and transferable is being adopted.
**Every one of these is UNKNOWN — none disclosed** (`learnings/top6_evidence_audit_2026-08-15.md`).

Refresh 2026-08-16: incorporated the 08-16 00:07Z frozen draw **1.17** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro` v3, AUTO-REFILL again — **fourth consecutive day** where the only
thing on the board was the eternal fallback). `runs/ledger.json` re-derived from the API today:
**n=33, mean 0.9424, s 0.1563**, z(1.17) = **+1.53** (highest draw since 08-05's 1.21), trailing-4
**0.86 → 0.885**, max 1.33, min 0.65, sealed mean-of-4 **promotion bar 1.0826** (was 1.0731 at n=32 —
the bar drifts with the record; re-read it from `runs/ledger.json` at prereg time, never cache it).
The binding **paired harm-pause** needs a trailing-4 move of −1.5s ⇒ −0.234; realized move is
**+0.025**, wrong sign. Record stays **resolved-STATIONARY**. **Public max UNCHANGED at 1.33 since
2026-07-18** — a high draw on a byte-frozen fork is the process, not a gain.

LB cross-check 08-16T10:00Z (full **2345-row** board via `leaderboard -d` → `lb_full_2026-08-16.csv`,
heartbeat asserted `HEARTBEAT OK … sha=e7fc52bf83d4`; top-20 archived to `lb_2026-08-16.csv`):
**THE BOARD MOVED AGAIN, ONE DAY AFTER THE BREAK-OPEN.** **Top-5 prize line 1.90 → 1.98** and
**gold/top-13 line 1.62 → 1.65**. **OUR RANK IS #130 OF 2345** (VERIFIED, full download; `Rank`
column): **125 teams strictly above**, 1.33 tie block spans **#126–#132** (7 teams). Yesterday #119
with 114 above ⇒ **−11 ranks in ONE day, −30 in two days, on a byte-unchanged 1.33.**
**Gap to gold widens 0.29 → 0.32; gap to the prize line widens 0.57 → 0.65.** Threshold counts:
**4 teams ≥2.0** (was 3), 7 ≥1.90, 10 ≥1.75, 13 ≥1.65, 20 ≥1.62, 31 ≥1.58, 49 ≥1.50, 132 ≥1.33.
Top of board: cstl **2.70 FLAT for a third day** (+1 draw, no gain), Daniel Franzen 2.58,
**Fufront-RyanX-AGI-Team 2.25 (#3, +0.85 on a SINGLE draw)**, Nikita Sorokin 2.10, Yusaku Muroya 1.98.
**★ THE CONTROL ARM DID NOT MOVE — and this is the line that matters for the engine-swap story.**
`Jack Cole (MindsAI)` **1.59 flat**, +1 sub, **Δ/draw 0.0000**; `Tufa Labs` **1.62 flat**, +1 sub,
**Δ/draw 0.0000**. The two teams who wrote the TTT literature and the harness we fork each spent a
draw and gained nothing. The **1.55–1.65 band** agrees: 22 → 25 teams (**+3 entered, 0 left**), median
score **1.60 → 1.61 (+0.01)**, median subs 51 → 46 (a **composition artifact** — entrants carry fewer
draws). A drop-in commodity engine would lift the *incumbent median*; the movement here is teams
*entering* the band. **This weakens the shared-regime / Qwen3.8-swap reading of 08-15 — but it is a
measurement about SCORES, not about method, and it does not say nobody swapped an engine.**
**Best-of-N confound, measured on this window:** 218 teams submitted, **only 48 (22.0%) gained
anything**; 230 new submissions; **median gain among gainers 0.1650/draw, max 1.0100/draw**. The big
movers are per-draw large, not draw-count artifacts (4–6× the median gainer): mina wailin
**+1.0100/draw**, kakuteki **+0.9700**, Fufront **+0.8500**, aRc (binary relation) **+0.7400**,
george yazaji1234556 **+0.6600**; Ryan #3 is **+0.2850/draw** (2 draws) where the bare +0.57 flatters
it. **Always quote Δscore per Δsubmission — the public score is a MAX over submissions.**
**EVIDENCE DISCIPLINE (unchanged and binding):** this instrument measures Score, SubmissionCount,
Rank, TeamName, LastSubmissionDate. **`LastSubmissionDate` is a team's MOST RECENT submission and
`Score` is their BEST — they need not be the same submission, so it CANNOT date a scoring run** (this
is proven on our own row: 1.33 was banked 07-18 but the row is dated to today's 1.17 filler).
**Zero methods are DISCLOSED for any mover in this window.** See
`learnings/top6_evidence_audit_2026-08-15.md`.
**Operator note:** bare `kaggle` was NOT on PATH in either shell on 08-16; all CLI calls ran as
`uvx --from kaggle==2.0.0 kaggle …` (what `lb_archive.py` uses internally via `DEFAULT_KAGGLE`).
A check that shells `kaggle` directly fails with "not recognized" — indistinguishable from a quiet
board. **Silence from an automation is NOT success.**


Refresh 2026-08-15: incorporated the 08-15 00:07Z frozen draw **0.89** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro`, AUTO-REFILL again — **third consecutive day** where the only thing
on the board was the eternal fallback). **`runs/ledger.json` IS fresh today** (unlike 08-14): **n=32,
mean 0.9353, s 0.1533**, z(0.89) = **−0.30**, trailing-4 **0.91 → 0.86**, max 1.33, min 0.65, sealed
mean-of-4 **promotion bar 1.0731** (was 1.0821 at n=30 — the bar drifts with the record; re-read it
from `runs/ledger.json` at prereg time, never cache it). Interior draw, comfortably above the retired
0.80 leg; the binding **paired harm-pause** (trailing-4 −1.5s ⇒ −0.230) sees a realized move of
**−0.05**. Record stays **resolved-STATIONARY**. Public max **UNCHANGED at 1.33**.

LB cross-check 08-15T14:17Z (archived `runs/lb_daily/lb_2026-08-15.csv` top-20; **full 2331-row
leaderboard also pulled via `leaderboard -d`**): **THE BOARD BROKE OPEN.** After six flat days the
**GOLD/TOP-13 LINE MOVED 1.58 → 1.62** and the **TOP-5 PRIZE LINE MOVED 1.64 → 1.90 (+0.26 in one
day)**. Both streaks are over. **Five new names entered the top-20, four of them above the old prize
line**: Daniel Franzen **2.58** (#2), Nikita Sorokin **2.10** (#3), Yusaku Muroya **1.98** (#4),
AbeLincoln1865 **1.90** (#5), MLRush **1.75** (#7). Three scores above 2.0 where there had been one
for four days. **cstl FLAT at 2.70 for a THIRD day** (last sub 08-13 20:08) — the leader did not move;
everything below it did.
**MECHANISM CANDIDATE, and it is not a technique — it is an engine version.** Alibaba released
**Qwen3.8-27B open weights at 2026-08-14 15:00 UTC** (Apache 2.0, 27.78B, 262K ctx). **Every one of
the five new top-20 names has a last-submission timestamp AFTER that release**, and 29 of the top-40
do. Forum 735243: **Ya Xu (148th) reports "a consistent 2x score on the local 25 dataset"** with
Qwen-3.8-27B-8bit vs Qwen-3.6-27B-8bit. **OUR FROZEN FORK'S ENGINE IS QWEN3.6-27B** — VERIFIED,
`notebooks/duckfork/kernel-metadata.json` `dataset_sources` carries
`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`. So the candidate is a **source swap on the artifact we
already run, with zero solver-code change**. Qwen3.8-27B is arch `Qwen3_5ForConditionalGeneration`
(**reuses the existing `qwen3_5` implementation** — the reason this is plausible on a frozen offline
wheelhouse) but **carries a `vision_config`, i.e. it is a native VLM — the #1 failure risk**. No FP8
Kaggle artifact exists; two anonymous community uploads made 08-14 do
(`trailblazeranemo/qwen3-8-27b` bf16, `overseer66/qwen3-8-27b-nvfp4`), zero Kaggle datasets, no
official `qwen-lm`. **Gate before spend, do not push blind.**

> **★ CORRECTION APPENDED 2026-08-16 — the two sentences above were already stale when written and they
> propagated into panel round 26 as factual errors F1/F2/F3. Original text left readable, as change control
> requires.**
> **(a) "No FP8 Kaggle artifact exists" is FALSE.** `saltb0x/qwen3-8-27b-fp8` was catalogued by our own
> scout the same day (**25,346,275,232 B, 08-14 22:55Z**), and on 08-16 FOYSAL published a second,
> `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (**30.89 GB**, Apache-2.0, vLLM-compatible). **≥5 community
> uploads now exist, not "two".**
> **(b) The `vision_config` "#1 failure risk" is EMPIRICALLY DISCHARGED, not merely gated.** Q38 v1's kernel
> log shows **vLLM READY at 394.8 s** serving `Qwen/Qwen3.8-27B-FP8` (25.3 GB load + boot = **295 s**), the
> **stock smoke passed at 417.3 s**, and **tool-calls succeeded in BOTH forced and auto mode**
> (`parser=qwen3_coder`). v1 died at **425.5 s on our own MM boot probe** — i.e. *after* the model was
> serving correctly. **The native-VLM risk did not materialise, and the load-time budget is measured, not
> estimated.** (Panel `systems` stated v1 "died before load" — exactly backwards.)
> **(c) The mirror is vetted.** Prereg §1.1 is a three-mirror **file-level hash comparison**; all three
> mirrors' config/tokenizer/template files hashed byte-identical, with `text_config` identical in all 33
> fields. "Unvetted mirror" is not a live objection.
> **Standing lesson:** a ground-truth file that is not corrected on the day it goes stale becomes an input
> that manufactures confident errors downstream — this one cost four reviewer-MAJORs in a single round.
**cstl is NOT explained by this and the flag does NOT trip**: cstl banked 2.70 on **08-12 20:02**,
~43 h *before* the release, and the forum reached the same conclusion independently (Ravindra, 82nd).
cstl stays traced-to-WHO / untraced-to-WHAT and stays not-a-target.
**#2 Daniel Franzen (`dfranzen`) is the ARC Prize 2024 grand-prize winner** — VERIFIED profile: Deep
Learning Researcher, University of Mainz, mutual-follow with **Jan Disselhoff** (together, *the
ARChitects*). The most credentialed ARC entrant on this board is now second at 2.58.
**OUR RANK IS #119 OF 2331** (VERIFIED, full download): **114 teams strictly above**, 1.33 tie block
spans **#115–#120** (6 teams, we are 5th; 1.34 at #112–114). Yesterday #100 with 94 above ⇒
**−19 ranks in ONE day on a byte-unchanged 1.33; twenty teams passed us.** **Gap to gold widens
0.25 → 0.29; gap to the prize line widens 0.31 → 0.57.** Even the comfortable number got worse.
Threshold counts: 3 teams ≥2.0, 5 ≥1.90, 7 ≥1.75, 11 ≥1.64, 15 ≥1.62, 26 ≥1.58, 41 ≥1.50, 120 ≥1.33.
Gold-boundary robustness: #13 and #14 are both 1.62 so the cutoff is 1.62 either way, but team count
(2331) is rising and the medal boundary index must be re-derived, not cached.
**INSTRUMENT DEFECT FOUND AND LOGGED:** the paginated `leaderboard --show --page-token` route returns
**non-contiguous** windows — 425 rows with a hole straight through our own tie block. **Use
`leaderboard -d`** (full 2331 rows with a `Rank` column) for any rank claim. Prior days' rank history
is unreconstructible because only top-20 was archived; start archiving the full board.
Discussions (2 new topics + 1 new Kaggle-staff comment): **1 ADOPT (conditional on a serving-compat
gate) / 1 ADAPT (schedule only) / 1 IGNORE**. Host María Cruz confirmed the **RTX 6000 pool capacity
constraint** is real and unresolved (3–8 h queues) — budget latency, do not read `Queued` as a defect.
**None of the three plan-forcing flags (cstl-method disclosure / rules-deadline-scoring change /
private-LB mechanics) tripped.** Full sweep in `learnings/community_sweep_2026-08-15.md`.

Refresh 2026-08-14: incorporated the 08-14 00:07Z frozen draw **0.70** (API COMPLETE, frozen-fork
filler `canivel/arc3-duck-repro`, description **"AUTO-REFILL … auto-armed by scripts/daily_submit.py
because the queue was empty"**). **`runs/ledger.json` is STALE as of this write (still n=30 / latest
0.78 / 08-13) — the ledger recompute by `scripts/ledger.py` is OWED.** Pending that, INFERRED here
only: n=31, mean ≈ **0.9368**, z(0.70) vs the n=30 stats ≈ **−1.61**, trailing-4 **0.9975 → 0.9100**.
0.70 is the **second-lowest draw of the campaign** (record min 0.65, 08-01) and the **second
consecutive sub-0.80** (0.78 → 0.70). The retired fixed-0.80 leg would have fired; the leg that
actually binds since **R23 (08-02)** is the **paired harm-pause (trailing-4 −1.5s ⇒ a −0.228 move)**,
and the realized move is **−0.0875**, comfortably inside it — the record stays **resolved-STATIONARY**.
Public max **UNCHANGED at 1.33** (per R25-N3 ρ̂_draw ≈ 0 the max is not the selection currency).
Second day running that the only thing on the board was the eternal fallback.
LB cross-check 08-14T~06:00 local (archived `runs/lb_daily/lb_2026-08-14.csv`, top-20):
**the board did not move at all.** Every top-20 name and every top-20 score is **identical to 08-13**;
only submission timestamps changed. **cstl FLAT at 2.70** (resubmitted 08-13 20:08, no gain) — the
first non-rising day for the leader since it stepped up. **TOP-5 PRIZE CUTOFF FLAT at 1.64 for a
SECOND day** (#5 BambooCopter Analytics 1.64 on submission-time tiebreak ahead of GeniusYY 1.64;
Tufa Labs 1.62 still out at #7). **GOLD CUTOFF (top-13) HOLDS at 1.58 for a SIXTH flat day** (#13
Biubiu). No new entrant, and the largest score delta anywhere in the top-20 is **0.00**.
**CORRECTION TO THE 08-12 ENTRY BELOW — cstl did NOT "enter" at 2.52.** Our own archived CSVs show
cstl at **1.59 from 08-04 through 08-09**, i.e. parked inside the dense 1.58–1.61 shared-public-
artifact (duck) band, then **1.59 → 2.52 in one submission (08-11 18:25, +0.93) → 2.70 (08-12 20:02)
→ flat**. It is a *band team that found a step*, not an outsider — so the +0.93 is a delta on top of
an artifact **we also run**, and the artifact family's ceiling is therefore **≥ 2.70**, which refutes
the ≈1.26–1.36 efficiency-reframe ceiling as a property of the *family* rather than of *our
configuration of it*. Do not read cstl's flat scores as evidence of a deterministic agent: the public
LB shows **best** score with **latest** submission date, so flatness only means later runs did not beat
the best. **cstl is now traced to WHO but not to WHAT**: teamId 16364346 is a two-person team,
`tehnar` ("Tehnar", Software Engineer, Amsterdam NL, 11y; pedigree is *competitive agents in
simulation* — NeurIPS 2024 Lux AI S3 116/701, ACM ICPC Finals 2016, Russian AI Cup tooling) and
`gatamaz` ("TG", San Francisco, 9y, this is the only competition he has ever entered). Both handles
carry **zero ARC artifacts** — tehnar's only public notebook is an 11-year-old Theano CNN and his only
forum post is an 8-year-old PyCharm comment; `github.com/tehnar` (Vsevolod Stepanov, INFERRED from
handle + SPbAU/ICPC/AI-Cup repos) has **0 public events** and no ARC or agent repo; no arXiv, no X, no
ARC-Prize community-leaderboard entry. **No trace found for the mechanism — none is proposed here.**
**OUR RANK IS #100, NOT "below #49"** — that figure had been carried unrecomputed since 08-09 and the
last real count was #63 on 08-01. Directly counted from the rank-ordered top-200 today: **94 teams
strictly above, 9 tied at 1.33 spanning ranks #95–#103** (we are 6th in the tie block; Peter 1.34 at
#94). That is **−37 ranks in 13 days on a byte-unchanged banked score.** Gap to gold **0.25** and gap
to the prize line **0.31**, both flat vs 08-13 — the flat *gaps* are the comfortable number and the
*rank* is the honest one. Full sweep, including the discussion dispositions (3 new posts, **0 ADOPT**)
and an independent 50,140-response replication of the 08-13 memory-channel finding, is in
`learnings/discsweep_2026-08-14.md`.

Refresh 2026-08-13: incorporated the 08-13 00:07Z frozen draw **0.78** (API COMPLETE,
frozen-fork filler `canivel/arc3-duck-repro` v3) → record ledger **n=30, mean 0.9447,
s 0.1519** (re-derived from the API by `scripts/ledger.py`, written to `runs/ledger.json`
— no retyped prose). Interior draw, z ≈ **−1.13** vs the n=29 stats; the record minimum is
unchanged at **0.65**, so this is well inside the observed range. It **ends the five-draw
interior recovery** (0.87 → 0.89 → 1.05 → 1.09 → 1.07 → **0.78**). **Trailing-4 mean falls
1.025 → 0.9975**, back under 1.00 after a single day above it; mean down (0.9503 → 0.9447)
and dispersion up (0.1513 → 0.1519), reversing both of yesterday's moves. **The watch-rule
does NOT arm.** 0.78 sits below the old fixed-0.80 line, but that leg was **retired at R23
(08-02)** and replaced by the **paired harm-pause (trailing-4 −1.5s ⇒ a −0.228 move)**; the
realized trailing-4 move is **−0.0275**, an order of magnitude inside the trigger, so the
record stays **resolved-STATIONARY**. Public max **UNCHANGED at 1.33** — per the R25-N3
ρ̂_draw ≈ 0 finding the max is not the selection currency. Sealed mean-of-4 promotion bar at
n=30 = **1.0821** (was 1.0876 at n=29 — a low draw *lowers* the bar; it drifts with the
record and must be re-read from `runs/ledger.json` at prereg time, not cached).
LB cross-check 08-13T~06:00 local (archived `runs/lb_daily/lb_2026-08-13.csv`, top-20):
**the leader extends, the chased band is flat.** **cstl 2.52 → 2.70** (+0.18, submitted
08-12 20:02) — still **zero public kernels, zero datasets on both handles**, and now
roughly **z ≈ +5** beyond the field's single-draw process; it remains untraced and is not a
target. **YUTO KOJIMA 1.86** re-submitted (08-13 00:09) with no score change. **TOP-5 PRIZE
CUTOFF FLAT at 1.64** for the first time after the 1.62 → 1.64 tightening: #5 is now
**BambooCopter Analytics 1.64** on submission-time tiebreak ahead of GeniusYY 1.64, and
**Tufa Labs 1.62 stays out of the prize band at #7**. **GOLD CUTOFF (top-13) HOLDS at 1.58
for a FIFTH flat day** — Biubiu #13 / ippeiogawa #14, DhanaLakshmiMalla 1.60 holding #12.
Head otherwise static: Andy liu 1.69, Lord Han Solo 1.65, Tecnod8.AI / FOYSAL / hvp /
Helmut AGI all 1.61; anngle / NoOneAhead 1.56 at the #17–18 mark. **Our 1.33 unchanged —
gap to gold 0.25 and gap to the prize line 0.31, both flat vs 08-12.** Consistent with the
08-11 sweep's monotone-deceleration read: the head runs away while the band we chase stalls.
No builds in flight: **`arc3-duck-effnote-eval` v1 (EFFNOTE arm, pushed 08-13) is terminal
COMPLETE**, and `arc3-duck-animation-eval` v1 is terminal COMPLETE with its **M0 result
still NOT pulled** (owed since 08-12). **Two result pulls are outstanding**; per the 08-13
prereg the effnote arm's only legal primary is **B1 vs the control-spread minimum 0.3986**,
with the arm's own first-half/second-half contrast barred. `arc3-duck-repro` and
`arc3-duck-compaction-eval` also remain terminal COMPLETE.

Refresh 2026-08-12: incorporated the 08-12 00:07Z frozen draw **1.07** (API COMPLETE,
frozen-fork filler `canivel/arc3-duck-repro` v3) → record ledger **n=29, mean 0.9503,
s 0.1513** (re-derived from the API by `scripts/ledger.py`, written to `runs/ledger.json`
— no retyped prose). Interior draw, z ≈ **+0.81** vs the n=28 stats, far above the 0.80
line ⇒ the resolved-STATIONARY watch-rule does **NOT** re-arm; **fifth consecutive
interior result** (0.87 → 0.89 → 1.05 → 1.09 → 1.07), a marginal step down from 1.09 but
the second-highest draw since 08-05's 1.21. **Trailing-4 mean 1.025 (was 0.975) — first
trailing-4 above 1.00 in the record**; mean up, dispersion down again (0.1523 → 0.1513).
Public max UNCHANGED at **1.33** — per the R25-N3 ρ̂_draw ≈ 0 finding the max is not the
selection currency. Sealed mean-of-4 promotion bar at n=29 = **1.0876** (was 1.0848 at
n=28 — the bar drifts with the record, so it must be re-read from `runs/ledger.json` at
prereg time, not cached).
LB cross-check 08-12T~06:00 local (archived `runs/lb_daily/lb_2026-08-12.csv`, top-20):
**a new #1 by a wide margin.** **cstl enters at 2.52** (submitted 08-11 18:25) — **+0.66
over the previous leader** YUTO KOJIMA (1.86), the largest single-entrant jump of the
campaign and the first score above 2.0 on this board. **TOP-5 PRIZE CUTOFF TIGHTENS
1.62 → 1.64**: cstl's insertion shifts everyone below down one rank, so #5 is now GeniusYY
1.64 and **Tufa Labs 1.62 falls out of the prize band one day after entering it**.
**GOLD CUTOFF (top-13) HOLDS at 1.58 for a fourth flat day**, composition shifting again:
DhanaLakshmiMalla enters at **1.60 (#11)**, so the 1.58 pack at #12–13 is Biubiu /
ippeiogawa and **Nkosi Ndwandwe slips #13 → #14, out of gold** (yesterday Tufa's entry
pushed out Yuchen20, who now sits #15). Head otherwise static: Andy liu 1.69, Lord Han
Solo 1.65, Tecnod8.AI / FOYSAL / hvp / Helmut AGI all 1.61; Mathurin Ache / anngle /
NoOneAhead 1.56 at #16–18. Our 1.33 remains below #49; **gap to gold 0.25 unchanged, gap
to the prize line widens 0.29 → 0.31** — i.e. the head is running away while the band we
actually chase is flat, consistent with the 08-11 sweep's monotone-deceleration read
(+0.003/day realized vs +0.01/day planned) at the gold line.
No builds in flight: `arc3-duck-animation-eval` (animation-awareness arm, pushed 08-11)
is now terminal **COMPLETE** with its **M0 result NOT yet pulled** (first item for the day
session; M0 is the only readable endpoint per the prereg power-honesty clause), and
`arc3-duck-repro` / `arc3-duck-compaction-eval` both remain terminal COMPLETE.

Refresh 2026-08-11: incorporated the 08-11 00:07Z frozen draw **1.09** (API COMPLETE,
frozen-fork filler `canivel/arc3-duck-repro` v3) → record ledger **n=28, mean 0.9461,
s 0.1523** (re-derived from the API by `scripts/ledger.py`, written to `runs/ledger.json`
— no retyped prose). Interior-high draw, z ≈ **+0.98** vs the n=27 stats, far above the
0.80 line ⇒ the resolved-STATIONARY watch-rule does **NOT** re-arm; **fourth consecutive
interior recovery** (0.87 → 0.89 → 1.05 → 1.09) and the highest draw since 08-05's 1.21.
Trailing-4 mean **0.975** (was 0.8975). Public max UNCHANGED at **1.33** — 1.09 does not
touch it, and per the R25-N3 ρ̂_draw ≈ 0 finding the max is not the selection currency.
Sealed mean-of-4 promotion bar at n=28 = **1.0848** (was 1.0801 at n=27 — the bar drifts
with the record, so it must be re-read from `runs/ledger.json` at prereg time, not cached).
LB cross-check 08-11T~06:00 local (archived `runs/lb_daily/lb_2026-08-11.csv`, top-20):
**one real entrant, and it is upstream.** **Tufa Labs enters the top-20 at 1.62 (#5)** —
the authors of the harness our eternal-fallback fork descends from
(`tufa-labs-duck-harness-june-30-milestone-winner`) — shifting every team below them down
exactly one rank. **GOLD CUTOFF (top-13) HOLDS at 1.58 for a third flat day**, but the
band composition changed: Yuchen20 is pushed **#13 → #14** and out of gold by Tufa's
insertion, so the 1.58 pack is now Biubiu / ippeiogawa / Nkosi Ndwandwe at #11–13.
**Top-5 prize cutoff TIGHTENS 1.61 → 1.62** (Tufa takes the #5 slot; Tecnod8.AI, FOYSAL,
Helmut AGI at 1.61 fall to #6–8). Scott Le Grand (1.50) drops out of the archived top-20.
Head above is otherwise static: KOJIMA 1.86 #1 (resubmitted 08-11 00:00, unchanged), Andy
liu 1.69, Lord Han Solo 1.65, GeniusYY 1.64. Our 1.33 remains below #49; gap to gold 0.25
unchanged, gap to the prize line widens 0.28 → 0.29. No builds in flight
(`arc3-duck-compaction-eval` and `arc3-duck-repro` both terminal COMPLETE).

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

Refresh 2026-08-10: incorporated the 08-10 00:07Z frozen draw **1.05** (API COMPLETE,
frozen-fork filler `canivel/arc3-duck-repro` v3) → record ledger **n=27, mean 0.9407,
s 0.1526**. Interior-high draw (z ≈ **+0.74** vs n=26 stats), far above the 0.80 line —
the resolved-STATIONARY watch-rule does **NOT** re-arm; **third consecutive interior
recovery** (0.87 → 0.89 → 1.05) and the highest draw since 08-05's 1.21, trailing-4 mean
0.8975 (was 0.9025). Our public max is UNCHANGED at **1.33** — 1.05 does not touch it,
and per the R25-N3 ρ̂_draw ≈ 0 finding the max is not the selection currency anyway.
LB cross-check 08-10T~06:00 local (archived `runs/lb_daily/lb_2026-08-10.csv`, top-20):
**the top-20 is score-static vs 08-09 — every rank/score identical.** KOJIMA 1.86 frozen
#1 (resubmitted 08-10 00:00, unchanged); Andy liu 1.69, Lord Han Solo 1.65, GeniusYY
1.64; Tecnod8.AI / FOYSAL / Helmut AGI 1.61 (#5–7), DhanaLakshmiMalla 1.60, cstl 1.59.
**GOLD CUTOFF (top-13) HOLDS at 1.58 — second flat day** after the 08-09 step 1.56 →
1.58 (Biubiu, ippeiogawa, Nkosi Ndwandwe, Yuchen20 = #10–13). Top-5 prize cutoff HOLDS
at **1.61**. The only apparent top-20 delta is **cosmetic and must not be read as churn**:
teamId 15520570 renamed *Dinesh kumar Thiyagarajan* → *"Whatever it takes..."* (same team,
same 1.50, same #19). Our 1.33 remains below #49; gap to gold 0.25, unchanged. No builds
in flight (`arc3-duck-compaction-eval` and `arc3-duck-repro` both terminal COMPLETE).

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

**[MAJOR] The Q38 v2 engine artifact `saltb0x/qwen3-8-27b-fp8` has no stated provenance chain — the arm may be measuring unknown weights.** The 08-15 ground truth records exactly three community uploads (trailblazeranemo bf16, overseer66 nvfp4, and — per 1b-A — FOYSAL's `foysalemonshanto/...-repacked-v1`). `saltb0x/qwen3-8-27b-fp8` appears nowhere in the record before this brief, yet it is the 25 GB payload attached to a sealed measurement. The `Q38_VETO` gate protects against the *incumbent* poisoning the arm; nothing described verifies the *challenger* — no tensor-level hash comparison against the official HF release, no config/tokenizer diff, no statement of who saltb0x is. If those weights are not byte-faithful Qwen3.8-27B (or are a quantization with different behavior), both CONFIRM and REFUTE license nothing about the engine claim, and the arm's slot is spent on an unlabeled variable. Required before reading the result: publish the provenance verification (shard hashes vs. official release, or vs. FOYSAL's independently verified repack) and log it next to the sealed fingerprint.

**[MAJOR] The re-anchored prior (+0.17 LB) and the sealed read (levels: ≥32 / ≤25) are in incommensurable units, so open question 3 is ill-posed as written.** The +0.17-class anchor is a public-LB score delta on other teams' unknown configurations; the read is a level count on our local gauntlet. Without a pre-registered mapping from Δscore to Δlevels (or at least the historical score-per-level ratio of this harness), the panel cannot decide whether REFUTE-at-+0.17-true-effect indicts the claim or the power — the question the proposal itself says must be decided "now, before the data." Worse, the arm is *already running*; if the disposition is not fixed before the result is pulled, the sealed prereg's whole purpose is defeated by an interpretive degree of freedom held in reserve. Fix: state the score↔levels conversion and the conditional disposition in writing, timestamped, before any `kernels output` pull on `arc3-q38-engine-eval` v2.

**[MAJOR] The verifier was modified twice, mid-push, during a sealed measurement — with no change-control regime for the gate suite itself.** Both fixes are plausibly correct (the em-dash mojibake and the pre-`Q38_VETO` assert), and fixing the verifier rather than the artifact was the right call for byte-identity. But "we edited the checks until the push passed" is observationally identical to the failure mode the checks exist to prevent, and the brief offers no mitigation: no logged diff of the verifier, no second-party sign-off, no rule for which verifier edits are legal during a sealed arm. Given that `feedback_audit_the_instrument` is per the proposal's own accounting the highest-frequency failure family, verifier edits need the same discipline as artifact edits: pre/post sha256 of the verifier, a written rationale, and a rule that any edit *loosening* a check voids the push. The claim that fix 2 is "strictly stronger" is asserted, not demonstrated (show the predicate before/after).

**[MAJOR] The 1c-1/1c-2 reframing overclaims transfer and, as proposed, produces another non-submittable diagnostic with no read and no consequent intervention.** Two papers from different substrates (an explicit CEM cost swap; constraint-repair routing in a different task family) are elevated to "makes our 96.3% null the *predicted* result" — that is an analogy promoted to a prediction, and the proposal's own 1c-1 caveat ("the transferable content is the diagnostic, not the fix") concedes we cannot apply either paper's remedy. The proposed CPU-only monotonicity probe has no pre-registered statistic, no threshold, and — critically — no branch: what artifact gets built if the criterion is found non-monotone, given that prompt-side intervention is exactly what 1c-2 says fails and the harness exposes no explicit objective to swap? Before adopting as a rail item, require: (a) the monotonicity metric and its decision rule, and (b) the named intervention lane each outcome triggers (e.g., decode-side action reranking or scorer-in-the-loop selection, which *are* available in this harness), so the diagnostic cannot terminate in "interesting."

**[MAJOR] Question 5 is the plan's center of gravity and the brief leaves it empty.** Five filler days, −30 ranks in two, and the only rail item that could touch the board is Q38 — whose own honest prior (+0.17 against a 0.94 mean) clears the 1.0826 bar only marginally and only if the LB anchor transfers to our configuration at all. A daily brief convening a full panel should arrive with at least one ranked candidate for "next artifact that could clear 1.0826" (Q38-on-CONFIRM promotion path, boristown-style readiness gate on the frozen fork, LoRA-serve retry post-heredoc-fix) with slot costs; asking the panel cold guarantees another day of AUTO-REFILL. As written this is a well-instrumented description of standing still.

**[MINOR] The +0.17-class re-anchor rests on n=2 dateable points with censoring in both directions.** Ya Xu's +0.17 is one draw from a max-statistic (his true per-draw effect could be higher or lower), and FOYSAL's +0.00 is uninformative if that submission simply errored — a COMPLETE-with-failure and a genuine null are indistinguishable in this instrument. The re-anchor is directionally right (down from "2×") but should be stated as "≤ small, n=2, censored," not as a point-class estimate.

**[MINOR] Q38 v1's death at t=425 s is cited but never root-caused in this brief, and the 08-15 record's #1 risk (native-VLM `vision_config`) is not shown to be gated in v2.** If v1 died on the vision-tower load path, v2's "still RUNNING at ~11 min wall-clock" is weak evidence given queue time, as the brief itself admits. State v1's proximate cause and the specific v2 change (or gate) that addresses it.


=====================================================================

THE PROPOSAL (sha256 of the full document: b0ada9beaa0e67dd; full length 14834 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
# DAILY BRIEF — 2026-08-23 (Sunday; weekly-consolidation day)

**Session type:** Sunday. Full panel is IN SCOPE (weekday panels are suspended per the 07-27 restructure).
**GPU spent today:** 0. **Kernel pushes today:** 0. **Submission slots used today:** 0 (tonight's fire is the 00:07 queue head).

---

## 1a. RESULT DEEP-DIVE — no new score landed since the last read

The overnight draw of **1.63 COMPLETE** (submitted 2026-08-23 00:07:10) was already pulled, interpreted and logged under the
08-22 ITERATION_LOG entry, which was written after midnight. **There is no unread score today.** What changed today is only the
*status* of that number once it is read as a config rather than a draw:

**The certified Q38 field-floor config is now 3/3: 1.59 / 1.58 / 1.63.**
- config mean **1.6000**, sample sd **0.0265**, **sem 0.0153**.
- This is the campaign's first config with a *replicated level* rather than a lucky maximum, and it is the exact statistic the
  **final-selection invariant** reads (`project_arc_final_selection_rule`: pick the two private twins by CONFIG MEAN, never by public max).
- Pre-registered expectation MET and then some: the sealed prereg called this a "typical draw of ~N(1.6, 0.2)". The realised spread is
  **0.0265, i.e. ~7.5x tighter than the assumed sd**. That is worth flagging as its own finding — *we have been pricing draw variance
  on this config far too pessimistically.* Caveat before anyone spends it: n=3 gives the sd itself only 2 df; the 95% CI on a 3-sample sd
  runs roughly 0.014–0.17, so the honest claim is "materially tighter than 0.2", not "0.027".

**The consequence that matters, stated plainly:** with sem 0.0153, a **4th redraw of this config buys essentially nothing** — it would
move the mean estimate by ~0.01 and cannot change any decision we face. The floor is measured. Further redraws are not evidence-gathering,
they are lottery tickets on a public max that the final-selection rule explicitly ignores.

**And the floor is not a path to gold.** +0.04 on the last draw bought **-11 ranks** (#239 to #250 of 2489); the gold line is **2.50**.
Redrawing this config asymptotes near ~2.0 as a *max over draws*, which is a display number with no private-board meaning.

## 1b. DISCUSSIONS SWEEP — **no new posts since the last sweep**

Feed pulled via `kaggle competitions topics list -c arc-prize-2026-arc-agi-3 --sort-by recent` (CLI 2.2.2). Newest two topics are
**736578** (Public vs. Private Discrepancy, 08-21 15:56Z) and **736540** (non-official games for training, 08-21 12:05Z). **Both were
already evaluated and dispositioned in `daily_brief_2026-08-22.md`.** Nothing has been posted in the ~44 h since.

Restated for the record because it remains the most load-bearing external datapoint we have:
- **736578 — ADAPT (unchanged).** Pellegrin reports duck+Q3.8 local 2.1 -> LB ~1.4, own harness local 5.0–5.4 -> LB **still ~1.4**.
  Our own answer is on file (`war_room/local_lb_transfer_2026-08-22.md`): the failure mode exists in *our* record too but resolves as a
  **single-seed artifact, not a transfer failure** (war-v1 read +3.16 sigma on one seed, 22/15/13 across three, and its family mean 16.67
  predicted the null LB correctly). Our one large replicated local effect **did** transfer near-proportionally: local x1.84 -> LB x1.70,
  agreeing to 8%. The untested cell is his: a **from-scratch** harness has far more freedom to overfit 25 public games than our
  duck-lineage fork does. No change to our gates.
- **736540 — IGNORE (unchanged).** Third-party non-official games; no bearing on a 25-game public / hidden-private scored rail.

## 1c. RESEARCH SWEEP — **no new results**

`arXiv:2607.03441` (Agentic Test-Time Training) and `arXiv:2511.04847` (Test-Time Adaptation via Environment Interaction; WebArena
multi-site 2% -> 23% via deployment-time dynamics grounding) both re-surfaced. **Both are already on file** and dispositioned
(`artifacts/research_sweep_2026-07-27.md`, briefs of 07-19/07-25/07-28, and 08-22). Disposition unchanged: **ADAPT-not-ADOPT** — the
transferable idea is *search over observed dynamics*, which is precisely the gap the per-turn program measured independently
(agent queries `transitions` in 16.3% of generations but shows search idioms in 5.8% and explicit candidate scoring in **0.2%**).
It is already the surviving program; nothing in these papers is a drop-in for a 27B served model under a 7920 s/game clock.

## 1d. WEEKLY MECHANICS (Sunday)

**FAILURE FINGERPRINTS — writer run first, then reader (protocol order).**
`fingerprint_backfill.py` reported **2 NEW incidents** the store on disk did not have — so the reader would again have described a stale
store had the order been reversed. Post-write the reader asserts **`store FRESH: 51 retained logs all scanned`**.

| family | n | first | last |
|---|---|---|---|
| class:ERROR:none | 7 | 2026-05-26 | 2026-06-28 |
| provenance:scratch-built | 5 | 2026-05-26 | 2026-06-28 |
| slug:canivel/arc3-final | 4 | 2026-05-26 | 2026-06-10 |
| class:COMPLETE:0.00 | 3 | 2026-03-29 | 2026-06-10 |
| slug:canivel/arc3-forge35 | 3 | 2026-04-24 | 2026-06-22 |
| slug:canivel/arc3-pilot-eval | 3 | 2026-07-07 | 2026-07-08 |
| t1:07d0f5248c48401d | 3 | 2026-07-07 | 2026-07-08 |
| class:COMPLETE:null-band | 2 | 2026-06-01 | 2026-06-08 |
| slug:canivel/arc3-a17-72b-canary | 2 | 2026-07-25 | 2026-07-25 |
| t1:fb1e96c3815797ad | 2 | 2026-07-25 | 2026-07-25 |

23 incidents / 11 recurring families / 5 deaths flagged-in-advance (strict). New: `inc-t1-010` (08-17, q38low) and `inc-t1-011`
(08-18, graft-floor v2) — both already-known events, now indexed. **`provenance:scratch-built` (n=5) remains the single most
expensive family in the campaign** and is exactly what `feedback_arc_kernel_structural_drift` and `preflight.py` now block.

**KAOS consolidation.** `kaos_ingest.py` -> inserted 7 / unchanged 218 / total 291. `kaos dream run` -> digest
`Dreams/2026-08-23-122518.md` (17 episodes, 4 ok / 12 failed, dry_run, 0 skills promoted — the expected shape; skills never auto-promote).
Hot memory is dominated by the six war-room documents written on 08-22, i.e. the consolidation is tracking the live program.

**Bench.** `kaos bench rejections` -> `{"rejections": []}`. Still no cross-workspace pull upstream; the gap remains filed.

## 1e. STATE OF THE RAIL

- **Kernels:** all five recent slugs (`q38-field`, `q38-graft`, `duck-repro-pathsafe`, `q38-private`, `graft-floor`) report COMPLETE.
  **No build is open.** No pull is pending.
- **Queue:** non-empty. Head = `canivel/arc3-duck-repro-pathsafe` v1 (pathsafe insurance fork).
- **Ledger** (`runs/ledger.json`): **n=37, mean 0.9316, sd 0.1771, latest_date 2026-08-20** — this is the NULL/filler family and it is
  correctly dated; the 1.59/1.58/1.63 draws are treatment and do not enter it. Promotion bar **1.089**.
- **GPU week boundary:** the weekend-prep lane's standing finding is that the accounting window appears to open **Monday**
  (31.4 GPU-h reconstructed for 08-17..08-22, reproducing two independent coordinator checkpoints only under a Monday open).
  **It has never been observed** — no Kaggle-side quota banner or refusal exists anywhere in our logs. Treat as inferred.

---

## 2. TONIGHT'S HEAD — the recommendation put to the panel

**Recommend: leave the pathsafe filler as the head. Do NOT redraw the field floor a 4th time.**

1. The floor is already measured to **sem 0.0153**; a 4th draw changes no decision (section 1a).
2. Public max is **not** what the final-selection rule reads, so a lucky redraw is worth exactly zero private information.
3. The pathsafe fork is **insurance that has never been scored end-to-end on a competition rerun**, and `arc3-duck-repro` — the entry it
   replaces — is retiring. Submitting it converts an untested fallback into a certified one.
4. **It cannot cost anything.** The public score is a max over submissions; a null-band pathsafe draw cannot displace the banked 1.63.

## 3. MONDAY — three sealed arms, two slots. The panel's job is to rank them.

All three are sealed, pre-data, with instruments validated **before** their data lands (`feedback_audit_the_instrument` discharged):

| arm | prereg | claim under test | self-registered prior |
|---|---|---|---|
| **C1 `cadence-effort`** | `cadence_prereg_2026-08-22.md` | bounding deliberation (`reasoning_effort=medium`) converts into levels | **P(SIGNAL) ~ 20%** (author's own) |
| **P1 seed 2** | `p1_notes_prereg_2026-08-22.md` | owed replicate; a single seed is never read | — |
| **P2 reset-retry** | `p2_reset_retry_prereg_2026-08-22.md` | retry allowance where 20/25 games sit at k<=1 | — |

The cadence prereg orders **C1 = first slot of the new GPU week, unconditional**, and states its own strongest counter-evidence:
`reasoning_effort=medium` has run twice on Q3.8 and delivered the mechanism (-48% tokens/turn, +72% acting turns) while scoring
**lc 21 and 17 against the floor's 28** — on the *June-30 vehicle*, never on the current floor, n=1 per point. **KILL-F** is designed so
that a delivered-but-null C1 retires the uniform-effort branch and a delivered-but-null C2 retires the whole family.

**Open questions for the panel (ranked):**

1. **Is the cadence family worth the first slot of the week at a 20% self-declared prior?** Its *positive* value is mostly the KILL — it
   closes the last "more of the same currency" lever. Is buying a kill the best use of the week's first, cheapest-to-verify slot,
   given the Monday quota boundary has never actually been observed and slot 1 is also the natural place to test that boundary?
2. **The SCREEN-SHAPE defect may dominate all three arms.** taaf's own submission benchmark is **110 runs = 25 games x ~4.4 clones**
   (`make_benchmark_kaggle_official_110`); our rail screens **1 clone**. Our sealed lc bands, our pooled sd 2.80, and every kill
   criterion above are computed on a screen whose *shape* differs from the thing being scored. Does this invalidate reading C1/P1/P2 on
   the current screen, or is it orthogonal? Note the cost is **not** free in GPU (~4.4x runtime, ~10 GPU-h vs 2.3), only free in code.
3. **The consistency lever vs the capability lever.** Per-turn arithmetic says recovering *all* wasted turns caps out at **lc 32 / LB ~1.78** —
   it cannot reach 2.50. Clone-consistency on near-certain level-1 games (bp35 / r11l / sp80) is worth **~+3 lc of pure consistency**,
   but that estimate is a **Qwen3.6 property** (weekend-prep's correction) and may not survive on Q3.8. Which lever gets the week?
4. **Do we still believe the 2.50 gold line is reachable on this program at all**, and if not, what is the honest objective for the
   remaining window (entry deadline 2026-10-26, Milestone 2 on 2026-09-30)? A defensible answer of "maximise a certified, replicated
   config and bank the milestone" is a legitimate output of this panel.

---

# ADDENDUM (same session, written BEFORE the panel verdicts were read) — I overstated the variance finding in 1a

**What is solid.** The three draws are three competition reruns of the **identical artifact**: `canivel/arc3-q38-field-eval` **v1**,
submitted 08-21 / 08-22 / 08-23 (daemon log, `runs/daily_submit_stdout.log`). So the 0.05 spread is *pure rerun noise on one fixed
build* — not a comparison across builds. That part strengthens the config-mean reading.

**What I overstated.** I wrote that the realised sd is "~7.5x tighter than the assumed sd" and let that stand as a finding.
Two problems:

1. **n=3 gives the sd 2 df, and that is a very wide sampling distribution.** Testing the observed sd against the null family's
   0.1771: chi-square stat 0.0446 on 2 df, **P(sd <= 0.0265 | true sd = 0.1771) = 0.022**. Suggestive on a single comparison,
   nowhere near sealed, and we have not been counting comparisons.
2. **The 0.1771 is not a clean same-artifact control.** The ledger's n=37 mixes forks, slugs and months, and the frozen duck fork's
   own draws run **0.41 to 1.33** — a ~0.9-wide range on what is largely one artifact, including the 0.41 tail draw this campaign
   already de-escalated as unexplained. **Same-artifact reruns demonstrably CAN swing hard on this platform.**

**Corrected claim:** the field floor's rerun spread *looks* tighter than the null family's (p ~ 0.02, single comparison, 2 df), and
the config mean is **1.600**. Whether the true rerun sd is 0.03 or 0.18 is **not settled by n=3**.

**Does this change tonight's recommendation? No — but for a different reason than I gave.** My argument was "sem is 0.0153, so a 4th
draw is worthless". Under the pessimistic sd the sem is **0.1022**, and a 4th draw would tighten it to 0.089 — a real reduction.
The recommendation survives because **no decision we currently face turns on +/-0.1 of this config mean**: the field floor is our only
certified config, the gold line is 2.50 and the prize line ~1.90, and the final-selection rule only needs to *rank* configs against
each other. Replication precision starts to matter the moment we have **two** certified configs within ~0.1 of each other — at which
point redraws become genuinely valuable and should be budgeted. Flagging that now so it is not rediscovered late.

**Consequence for the panel's question 1:** the pessimistic sd also means our LB instrument is coarser than section 1a implied, which
*raises* the relative value of the local rail (sealed lc bands) over LB draws for reading arms — and therefore raises the stakes on
question 2 (the screen-shape defect), since the local rail is then carrying more inferential weight, not less.

## Rail verification run this session (zero GPU)

- `scripts/local_gate.py --self-test` -> **PASS 13/13, 0 fail** (40.9 s). Includes **S13 `cadence_instrument_can_refuse`** — Monday's
  C1 instrument is proven able to report failure against a poisoned expectation, and **S10 cross-arm refusal** still fires.
- Daemon: healthy, only `already-submitted-today` skips; queue head correctly armed for tonight's 00:07 fire.
- **Monday prerequisite discovered:** `local_gate.py --arm` currently registers 10 arms and **none of them is a cadence arm**
  (`budget-t05, budget-t3, graft-confirm, graft-floor, private-base, private-edge1, private-edge12, private-edge2, q38-field, q38-graft`).
  C1's per-arm certification suite therefore does not exist yet. The cadence instrument (P9) and its negative control (S13) DO exist.
  **The Monday build session must register the C1 arm before pushing** — that is a cadence-lane edit, not this session's to make
  (one-lane-one-operator, 08-18 ruling).
- **GPU-week boundary: NOT observed this session.** The Chrome profile is locked by a running browser, so the web-console quota page
  could not be read without killing the user's session. The boundary remains **inferred**. Weekend-prep's advice stands: Monday,
  cheapest build first.

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
