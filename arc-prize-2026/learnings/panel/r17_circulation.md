# Grinder-cracking design — R17 SEALING (A14 seals on this circulation)

Filed 2026-07-22. This is the sealing revision R16 demanded (0 ACCEPT / 5×
MAJOR-REVISION, 0 FATAL, `learnings/panel/round16/_directives.md`): the R16
republication's arithmetic was ratified to the digit, the seal was withheld on a
9-item checklist, and every computational item on that checklist was discharged
today from data on disk ($0, 0 GPU-h). **The recalibrated A14 gate SEALS ON THIS
CIRCULATION**, as amended below (control band → n=3 fallback per §5; B+ rows
flagged pre-audit per §3; composition sealed per §2). Nothing below conditions on
any unobserved measurement; every new threshold is sealed before its measurement
runs and mirrored in `runs/sealed/r17_thresholds.json` (§9).

Base document: `learnings/war_room/grinder_design_R16_republication.md`
(unaltered on disk; superseded section-by-section here; everything not amended
here carries forward verbatim). Evidence base unchanged, plus today's artifacts:
`runs/latent_state_audit/holdout_report.md` (held-out resolver validation),
`learnings/war_room/sentinel_q2_discharge_2026-07-22.md` (Q2 conditions 1–3
discharged; sentinel re-keyed v1→v2), `learnings/war_room/
a17_72b_screen_scope_v2.md` (A17″ re-filed per Q6), `runs/r17_sealing/
sketch_sensitivity.{py,json}` (this document's computed annexes),
`runs/schema_traces_mining/report.md` (external evidence annex, §11),
`learnings/daily_brief_2026-07-22.md` (ledger n=14 mean 0.972, σ̂ 0.144;
zoli800 1.39 fork-diff = byte-identical artifact).

---

## §1. Held-out resolver validation — the audit verdicts, revalidated (checklist item 2)

**Method (per C6, exactly):** for each in-sample ALIASED-RESOLVABLE game, streams
of the benchmark engine version only were split 4/4 (alternating sorted stream
ids; versions never pooled — the cn04/ka59 drift rule); the resolver was fit and
selected on TRAIN streams alone; it certifies iff held-out pooled augmented
determinism ≥ 0.99 AND Wilson 95% LB ≥ 0.95. Any failure → ALIASED-UNRESOLVED,
with no fallback to the in-sample resolver (that fallback is the selection leak
C6 bans). Selftest PASS on both synthetic arms (hidden-mod3 → KEEP; coin-flip →
UNRESOLVED; low-support mod3 → DROP despite held-out det 1.0). Per-stream
resolver table published in full in the report. A non-binding 3/5-split
sensitivity table is also published; the 4/4 split is the binding certificate.

**Result: 10 of 11 in-sample RESOLVABLE games DROP to ALIASED-UNRESOLVED**
(cd82, cn04, dc22, ka59, re86, s5i5, sc25, tr87, vc33, wa30 — five by
fit-failure on TRAIN alone, five by holdout certificate failure, e.g. tr87
parity: held-out det 1.000 but Wilson LB 0.927 < 0.95 on 49 visits). **The sole
survivor is sb26**, and not via parity: the TRAIN fit selects **hist1** (a
history-class resolver) with held-out det 1.000 on 190 visits, Wilson LB 0.980 —
verdict ALIASED-RESOLVABLE(hist1), HISTORY-AUGMENT class. prog-synthesis's R16
sentence is now the record: "'parity IS the mechanism' was an in-sample
hypothesis" — and it did not survive the held-out test anywhere except sb26,
where the surviving mechanism is a different class.

**CLEAN certificates (Q5 splice-legality inherits the Wilson standard):** 10 of
11 CLEAN games CONFIRMED at pooled base-key determinism with Wilson LB ≥ 0.95
(ar25 0.980, bp35 0.991, ft09 0.984, lf52 0.962, lp85 0.974, ls20 0.970, r11l
0.963, sp80 0.992, su15 0.989, tu93 0.994). **tn36 is FLAGGED: det 1.000 but
only 31 repeat visits, Wilson LB 0.890 < 0.95** — tn36 does not hold a
splice-grade certificate. g50t/m0r0/sk48 unchanged (ALIASED-UNRESOLVED).

**Sealed downstream consequences (the held-out numbers are now the ONLY binding
ones; every consumer re-pointed per the checklist):**

1. **PHASE-AUGMENT / resync-viable set = EMPTY.** No game holds a held-out
   phase-resolver certificate. sb26's hist1 is history-class and licenses **no
   resync** (resync-before-abort keys on phase recovery from the observed frame;
   hist1 keys on interaction history — a different contract).
2. **EWM Stage-1 re-prices on ship-now CLEAN carriers only** (§7). tr87 — the
   only surviving new-clear candidate in §9.1R16 — is UNRESOLVED and struck.
3. **Banking = FULL-REPLAY-ONLY from RESET everywhere non-CLEAN; prefix-splice
   restricted to the 10 CONFIRMED CLEAN games** (tn36 excluded by its flag).
   Full replay requires NO resolver — a replay is RESET + the recorded action
   sequence, which is exactly why it survives UNRESOLVED verdicts; splice
   requires the CLEAN certificate it now has, held-out grade, in 10 games.
4. **(c)+Reki resurrection prong (i)** re-pointed: it is now discharged ONLY for
   sb26 (hist1, held-out certified) and for no other game. Prong (ii) (Reki-keyed
   predict_metric ≥ 0.90) remains unrun; the Q3 kill stands; no window (a pass
   still leaves the component below MDE/2).
5. **su15's CLEAN certificate (LB 0.989) does not re-admit it.** A12 exclusion
   holds; the §13R16 post-A13 re-admission path is unchanged.

---

## §2. The sealed composition sentence + per-branch P(pass) (checklist item 1)

**Sealed composition sentence:** *At the A14 binding cumulative look, the
evaluated artifact is exactly: base harness + (f) continuation + (a) budget
sentinel v2 (game-envelope keyed, SENTINEL_BUDGET=150) + (b) diff summarizer —
minus any of {(a),(b)} killed OFF by its own §8R16 per-window guard before the
look, with the kill recorded and the branch re-labeled — plus banked-trajectory
replay (full-replay-only; prefix-splice only in the 10 CONFIRMED CLEAN games)
if and only if B+ (full-panel sign-off + A16 recompute recirculated + §1
held-out audit consequences honored) opened W3 before the look, plus the EWM
Stage-1 executor (phase-blind base key, resync OFF, no augmentation) if and only
if the §7 gate passed and its window ran before the look; no other component may
be present, and EWM's score prong is evaluated ONLY at this cumulative look —
it has no per-window score look.*

Four admissible compositions. P(pass) is computed by the sealed sketch model —
positives uniform over the branch's §3 pair range; spurious S+ ~ U{0,1,2},
S− ~ U{0,1,2} independent (the §5R16 assumption (iii) made exact); pass = §3R16
sign-test critical at α = 0.05 met by P + S+ among n = P + S+ + S− pairs; exact
enumeration, no MC (`runs/r17_sealing/sketch_sensitivity.py`):

| branch | stack at look | expected positive pairs | **P(pass)** | status |
|---|---|---|---:|---|
| B− / EWM-out | (f)+(a)+(b) | 1–3 | **0.04** | unconditional floor; near-certain honest FAIL (§5R16 language stands) |
| B− / EWM-in | + EWM (CLEAN carriers) | 1–4 | **0.08** | EWM's small channels add ≤1 expected pair (central case: ls20 or ft09; tu93 overlaps (a); all channel lower bounds are 0.0) |
| B+ / EWM-out | + banking (replay-priced) | 2–5 | **0.19** | PROVISIONAL — pre-audit rows flagged, superseded by A16 recompute (§3, §4) |
| B+ / EWM-in | + banking + EWM | 2–6 | **0.27** | PROVISIONAL — same flag; EWM adds ls20 only |

Consistency: the B− numbers sit inside R16's published 0.02–0.10; the B+
numbers sit inside 0.10–0.30. The holdout+replay haircut is visible in the
model: the pre-audit B+ range 2–6 evaluates to 0.27; the post-haircut 2–5
evaluates to 0.19 — the haircut costs ≈ 0.08 of P(pass). **P(pass|B+) is
republished by the A16 recompute before W3 opens (methodology's condition);
the two B+ rows above are labels for scheduling, not sealed pass
probabilities.** The B− rows are sealed as-is (C7: "the B− branch may seal
as-is").

---

## §3. §2R/§4R tables republished with the audit propagated (checklist items 2, 4)

**§2R sums:** unchanged in ceiling arithmetic — (a) +0.06/(b) +0.06 rows stand;
B− ceiling +0.09 / expectation +0.02–0.06 **seals as-is**. The banking-fixed row
(+0.15 ceiling, +0.03–0.08 expectation) is hereby **flagged PRE-AUDIT,
SUPERSEDED BY THE A16 RECOMPUTE** — it was computed without the replay-action
cost bound that FULL-REPLAY-ONLY imposes ("the exact O1 defect, one document
later" — prog-synthesis; conceded). B+ ceiling +0.21 survives only as the
pre-haircut upper envelope until A16 republishes.

**§4R per-game table, republished with held-out verdicts and replay pricing:**

| game | held-out class | B− Δclears | B+ adds | B+ Δclears | replay-to-frontier (actions, 4 certified runs) |
|---|---|---|---|---|---|
| ft09 (6) | CLEAN CONFIRMED (LB 0.984) | **0** | banking, **splice-eligible post-prong** | **0–1** *(pre-audit flag)* | splice: no replay bound |
| ka59 (7) | **UNRESOLVED** → full-replay-only | **0** ((a)) | banking, full-replay | **0–1** *(pre-audit flag)* | 16–77 (77/16/21/34) = 11–51% of the 150-action envelope |
| re86 (8) | **UNRESOLVED** → full-replay-only | **0–1** ((a)) | banking, full-replay | **0–1** *(pre-audit flag)* | 35–80 (64/80/35/41) = 23–53% |
| sc25 (6) | **UNRESOLVED** → full-replay-only | **0** | banking, full-replay | **0–2 → interim 0–1** | 107–114 (2 banked seeds; 2 seeds lc=0, nothing to bank) = **71–76%**; residual live envelope 36–43 actions |
| tu93 (9) | CLEAN CONFIRMED (LB 0.994) | **0–1** ((a)) | banking, **splice-eligible post-prong** | **0–1** *(pre-audit flag)* | splice: no replay bound |
| sb26 (8) | RESOLVABLE(**hist1**) | **0** | — (hist1 licenses no banking key change; full-replay-only) | **0** | — |
| lp85 (8) | CLEAN CONFIRMED (LB 0.974) | **0** | — | **0** | — |
| su15 (9) | CLEAN (excluded, A12) | — | — | — | — |

**Sums:** B− = 0–2 extra clears (unchanged, seals). B+ = **1–4 pre-audit → 1–3
interim post-replay-haircut**; the interim number is a scheduling input only —
the A16 recompute owns the final B+ Δclears and must circulate before W3 opens
(§4). Expected nonzero positive pairs: B− ≈ 1–3 (unchanged); B+ ≈ 2–5 interim
(was 2–6). The two canonical grinders still carry zero at Qwen tier.

---

## §4. A16 mandate — EXTENDED (checklist item 4)

The A16 recompute mandate now reads, sealed:

1. **Scope: FULL-REPLAY-ONLY v1 everywhere** (Q5 ruling adopted verbatim:
   prefix-splice legality in CLEAN games is an on-support claim used off-support;
   splice is deferred to a per-game Kaggle-side replay-success prong on the
   actual banked trajectory, and may then be used ONLY in the 10 CONFIRMED CLEAN
   games of §1).
2. **Replay-action cost priced from data, denomination sealed:** the pricing
   above (§3 last column) is computed from `actions_per_level` in the four
   certified `benchmark.json` files — replay-to-frontier = Σ actions over
   completed levels at the banked frontier. It is denominated in ACTIONS against
   the 150-action envelope (conservative). The recorded ambiguity: the binding
   envelope is the ~63k TOKEN envelope, and a replayed recorded action generates
   ≈0 tokens (32–86% of actions in the banked games already log
   generated_tokens = 0), so the true replay tax may be mostly wall-clock
   against the 7920 s per-game deadline, not tokens. **Sealed rule: the A16
   recompute prices in the action denomination unless it MEASURES the realized
   token+wall-clock cost of replay on the Kaggle-side replay-success prong; it
   may re-denominate only with that measurement in hand.**
3. **Sentinel interaction stated:** sentinel v2 counts CUMULATIVE game actions —
   replayed actions consume sentinel budget and can fire warnings. Default,
   sealed: replayed actions ARE counted (conservative; a replay that eats 76% of
   the envelope should look like it). A16 may propose a replay-flagged exclusion
   only as a registered patch revision with its own canary.
4. **Deliverables before W3 opens:** recomputed B+ Δclears per game, recomputed
   B+ ceiling/expectation, republished P(pass|B+), all under FULL-REPLAY-ONLY
   with the replay pricing — **recirculated to the panel before W3 opens.** The
   B− branch is sealed now and does not wait on A16.

Interim consequence already visible in the data: sc25 — three of five B+
Δclears carriers' worth of variance-harvest hope — retains only 36–43 live
actions after a frontier replay; its row is cut to 0–1 pending A16. ka59/re86
retain 49–89% / 47–77% of the envelope and keep their 0–1 rows, flagged.

---

## §5. Control band: engine-drift precondition + the three Q7 conditions (checklist item 3)

### §5.1 Engine-drift precondition (methodology R2 — in the sealing text, as demanded)

**Sealed:** (i) every control-band game must carry the SAME versioned game id as
its binding-look counterpart (the 25 reference ids are published in
`runs/latent_state_audit/holdout_report.md` and §5.3's table; verified identical
across all four certified runs today, 25/25). (ii) **A drifted game's pair is
DROPPED — counted for neither side, in neither prong.** (iii) **If ≥ 3 of the 24
look games drop on drift, the look is postponed and the band is refreshed once:
2 additional control seeds on the then-current engines (4.4 GPU-h, §12R16
headroom); if after the one refresh ≥ 3 pairs still drift against the ON seeds,
the binding look proceeds on the surviving pairs with every drop published and
the panel notified before the look is scored.** No other drift handling is
permitted.

### §5.2 Q7 condition (i): the structural-bound sentence

**CONFIRMED STRUCTURAL, in one sentence:** (f)'s 0.00 counting bound is
structural, not empirical — (f) formalizes the base harness's existing
post-GAME_OVER continuation path (the same recovery behavior is present and was
exercised in all certified control seeds; W0's 49/49 recoveries were all
recoverable by that base path), introduces no new action-selection or scoring
channel, and therefore cannot complete a level the base path would not.

### §5.3 Q7 condition (ii): the n=4 band, frozen and published

Per-game control value = 4-run mean levels_completed over {war_eval_v1,
war_eval_v2, war_eval_v3, w0_eval_s1}, frozen at these values (su15 excluded,
A12; versioned ids as shown; full 25-game table):

| game | id | v1/v2/v3/W0 | mean | game | id | v1/v2/v3/W0 | mean |
|---|---|---|---:|---|---|---|---:|
| ar25 | 0c556536 | 2/1/1/1 | 1.25 | r11l | 495a7899 | 1/0/1/1 | 0.75 |
| bp35 | 0a0ad940 | 1/0/1/1 | 0.75 | re86 | 8af5384d | 1/2/1/1 | 1.25 |
| cd82 | fb555c5d | 0/0/0/0 | 0.00 | s5i5 | 18d95033 | 1/0/1/0 | 0.50 |
| cn04 | 2fe56bfb | 0/0/0/0 | 0.00 | sb26 | 7fbdac44 | 1/1/1/1 | 1.00 |
| dc22 | fdcac232 | 0/0/0/0 | 0.00 | sc25 | 635fd71a | 2/2/0/0 | 1.00 |
| ft09 | 0d8bbf25 | 1/2/0/2 | 1.25 | sk48 | d8078629 | 0/0/0/0 | 0.00 |
| g50t | 5849a774 | 0/0/1/0 | 0.25 | sp80 | 589a99af | 1/0/1/1 | 0.75 |
| ka59 | 38d34dbb | 1/1/1/1 | 1.00 | su15 | 1944f8ab | (1/1/0/1) | excl. |
| lf52 | 271a04aa | 1/1/0/1 | 0.75 | tn36 | ef4dde99 | 1/0/0/0 | 0.25 |
| lp85 | 305b61c3 | 1/1/1/1 | 1.00 | tr87 | cd924810 | 0/0/0/0 | 0.00 |
| ls20 | 9607627b | 1/0/0/0 | 0.25 | tu93 | 0768757b | 2/2/1/2 | 1.75 |
| m0r0 | 492f87ba | 1/0/0/0 | 0.25 | vc33 | 5430563c | 2/1/2/2 | 1.75 |
| | | | | wa30 | ee6fef47 | 0/0/0/0 | 0.00 |

Frozen NOW as directed — and then ruled on by condition (iii) below. The freeze
plus the mechanical rule is what removes the post-hoc band-vs-fallback choice:
the decision is made in this document, on config evidence already on disk,
before any ON seed of the final stack exists.

### §5.4 Q7 condition (iii): config-diff enumeration — **THE BAND IS ILLEGAL; THE SEALED FALLBACK TRIGGERS**

Enumeration from the pulled artifacts, complete:

- **Identical across all four runs:** `taaf_setup_env.json` (byte-identical,
  empty diff), `git_status.txt` (byte-identical), the solver config line
  (`max_actions_per_game=None, max_runtime_s_per_game=7920.0, concurrency=28`,
  same model/serve config), all 25 versioned game ids, duration (2h12–13m).
- **Different:** war_eval_v{1,2,3} ran the WAR KIT — their logs carry
  `warpack: banking=True recovery=True shortcircuit=True retry_guard=True
  bank_min_time=120.0 bank_strict=True recovery_repeats=30` (benchmark label
  `duck-harness-kaggle-warpack-v1`); w0_eval_s1 ran **no warpack** with (f)
  continuation ON (`w0-continuation-eval: SEED=1 (f) game-over-continuation ON,
  NO warpack`, label `duck-harness-kaggle-continuation-v1`).

The diff set is {(f)} ∪ {warpack: banking, recovery, shortcircuit, retry_guard}
≠ {(f)}. **The sealed Q7 rule fires mechanically: the n=4 pooled band is
ILLEGAL, and the §11R16 pre-registered fallback triggers — 2 additional W0
seeds run before the binding look (2 pushes, 4.4 GPU-h, already scheduled as
the §12R16 conditional line), and the control band becomes {w0_s1, w0_s2,
w0_s3}, n=3.** No other configuration is permitted, per the §11R16 decision
tree sealed on 07-20. We note for the record that the warpack modules logged
no events beyond the config banner in the certified runs, and that §11R16's
recovery-equivalence claim (49/49 recoverable by the base path) still holds —
but the rule is the rule precisely so that this kind of judgment call is never
made post-hoc by the authors. The band question is closed by the sealed rule,
not by our opinion of the warpack's inertness.

**Propagation of the n=3 band (every dependent number recomputed):**

- **§6.2R16 dismantle calibration:** SE(Δ) = 0.189·√(2/3) = **0.154** (was
  0.144); **P(trip | Δ=0) = 0.26** (was 0.24). Sealed threshold unchanged at
  −0.10; we accept the 26% and say so before observation. Power: a true −0.10
  trips with 50%, a true −0.25 with ~83%.
- **§8R16 per-window guard: unchanged** — it was already 3 ON vs 3 control
  (SE 0.154, boundary −0.28, familywise 0.097 ≤ 0.10).
- **A17″ comparator: unaffected.** The 72B screen's R27 = 4 certified runs is a
  capability-MAX comparator ratified by the panel in its own right, not the
  control band; Σ 27B MAX = 6 with or without W0 (per-seed table, A17″ §2).
  Flagged for panel awareness, not re-opened: the 27B comparator runs carried
  the warpack — any inflation of the 27B baseline biases the capability prong
  toward NO-GO, i.e. adversely to the 72B; the A17″ §4 false-NO-GO computation
  already prices the observed comparator as-is.
- **The fallback seeds inherit §5.1's drift precondition** (they run on
  then-current engines; identity is checked per game against the ON seeds).
- The frozen §5.3 n=4 values stand in the record as what the band WOULD have
  been; they are void as controls except the w0_s1 column.

---

## §6. A17″ — re-filed, referenced (checklist item 5)

The corrected amendment is **`learnings/war_room/a17_72b_screen_scope_v2.md`**
(filed today), which discharges every Q6 defect in one pass: exactly ONE gate
boolean (the ≥8 conjunct dropped from branch 2 — systems' option (i), Part 1
§13's own form, dead-code check shown); **ρ := ρ_action** = pooled actions/s
ratio at identical 7920 s windows (27B numerator frozen from disk: 480 actions;
tokens/s demoted to diagnostic); comparator population stated as the **4
certified runs** with Σ MAX = 6 recomputed from disk (unchanged) and per-game
MAX over all 72B seeds as the single statistic in both prongs; the **per-seed Σ
table** published (Σ = {5, 5, 4, 6}, shortfall distribution {0,1,1,2}); **false-
NO-GO computed by exact enumeration with the chosen remedy** — P(false NO-GO |
+1/game) = 0.000 in the throttled regime and ≤ 0.0625 (k=1) / 0.0039 (k=2)
under the most conservative model, remedy = 2 seeds budgeted from the start +
marginal trigger extended to Σ∈{6,7} and branch-2-within-1; **one reconciled
budget** (AWQ arm cap 10.0 GPU-h, modal 7.5; 7920 s log-verified as a per-game
concurrent soft deadline; the 8.8 GPU-h reading retracted; 45-min figure
struck; Jul 21–27 closes ≈ 23.2/30 unchanged); **FP8/W8A8 fallback SKU
pre-registered** (RedHatAI w8a8 candidate named in research_2026-07-22; two
TBDs carried honestly) and the **accelerator-selector precondition** added
before every GPU build. The v4-registration ledger requirement is answered YES.
Not duplicated here; that document is part of this circulation and seals with
it.

---

## §7. EWM Stage-1 measurement config — SEALED (checklist item 6)

**Carrier set (sealed):** the ship-now CLEAN games with held-out-confirmed
certificates — **ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tu93**
(10 CONFIRMED), plus **tn36 admitted with its FLAG stated (det 1.000 on only 31
visits, Wilson LB 0.890)** — tn36 carries zero priced value (already at base)
and may not enter any splice, so its flag prices nothing. su15 remains excluded
from any sealed look (A12). No other game is a carrier.

**Sim configuration (pinned):** phase-blind BASE key on every carrier; NO
augmentation of any kind; resync-before-abort remains contract v1.1, OFF, and —
with the resync-viable set EMPTY (§1) — **cannot be enabled for any game in
this campaign without a new held-out certificate and a new panel look.**

**PHASE-AUGMENTATION IS NOT REGISTERED.** No game qualifies (the held-out
phase-certified set is empty), so there is no contract revision to register, no
canary to run, and no place for an augmented key in any measurement. A future
held-out phase certificate on any game does not self-execute: it requires
registration as a numbered sim-contract revision (with the synthetic mod-3
selftest as its canary) AND a new panel look before it may touch a measurement.
The same applies to any history-augment (hist1/sb26) use beyond banking's
no-resolver full replay.

**Re-price on the sealed carriers (tr87 struck):** surviving priced channels —
tu93 L1 speed +0.1–0.9 pts, ls20 L1 speed +0.0–0.6 pts, ft09-L1 reliability
+0.0–1.0 pts (overlap-adjusted per §9.1R16); tn36 0. **Sum +0.1–2.5 pts ≈
+0.00–0.10 rail per draw, central ≈ +0.04** (was +0.02–0.18 central +0.08).
There is no surviving new-clear channel at any tier. Consequence, stated
plainly and pre-empting the panel: **EWM Stage-1 is no longer the largest
registered non-model line; it re-prioritizes below (a), (b), and A17** (as
llm-agents predicted), and its window competes with filler at roughly 2× filler
EV, gate costs included.

**Per-subset pricing (prog-synthesis N6, pre-registered):** price = the sum of
the surviving channel values over exactly the carriers that pass the cheap
measurement: {tu93} → +0.1–0.9 pts; {ls20} → +0.0–0.6; {ft09} → +0.0–1.0;
unions additive (channels are disjoint per §9.1R16 overlap accounting); any
passing subset containing none of {tu93, ls20, ft09} prices to ZERO and Stage-1
is parked.

**Stage-1 gate, explicit (all four conditions BEFORE any window):**
1. Latent-state audit + held-out validation complete — **DISCHARGED (§1)**.
2. **Cheap measurement, re-sealed for the new carrier set:** BFS-plan
   step-accuracy on the local engines matching the Kaggle build, on sim-derived
   (not teacher-forced) states, phase-blind base key. **Threshold: ≥ 0.70 at
   plan depth ≤ 10 on ≥ 2 of the 3 priced carriers {tu93, ls20, ft09-L1}**;
   FAIL → Stage-1 parked at zero window cost. (Sealed here, before the
   measurement runs; mirrored in `runs/sealed/r17_thresholds.json`.)
3. A10 canary: plan/abort/fallback triggers ≥ 1/run on ≥ 5 games (dry-run
   replay evidence stands).
4. Full-panel sign-off (new asset class in the kernel).
Window prongs unchanged from §9.2R16; score prong = cumulative look ONLY (§2).

**llm-agents Q6 (Rodionov fixed-interface), the required paragraph:** Yes — our
sim is fixed-interface in Rodionov's sense. It consumes the harness's fixed
action vocabulary (ACTION1–ACTION5 + ACTION6(x,y)) and emits settled frames in
the same 64×64 observation space for every game; per-game dynamics are learned
content BEHIND that interface, and nothing about the interface varies across
games or levels. That is precisely why phase- or history-augmentation is a
contract change and not a tuning knob: augmenting the key alters the state
interface itself, which is the one thing the executor contract holds fixed —
consistent with treating any augmentation as a numbered v1.x revision with its
own canary, as sealed above.

---

## §8. Sentinel — Q2 discharged, condition 4 sealed, trigger table published (checklist item 7)

**Conditions 1–3: DISCHARGED** (`sentinel_q2_discharge_2026-07-22.md`).
Condition 1's deciding statistic came back decisive AGAINST the single-attempt
approximation: only 1/6 budget-attributable GAME_OVERs sat in multi-attempt
games (median actions at fatal-attempt start: 0) — the benign-looking number
the reviewers predicted — while the envelope view showed **15 of 33
envelope-crossing (game,seed) units received NO v1 warning by 0.9×B** and 13
cross-attempt-waste episodes were structurally invisible to v1, with the (a)
carrier games ka59/re86/tu93 multi-attempt in EVERY certified seed. **The
mandated re-key landed: sentinel v2, keyed on the cumulative per-game
envelope** (thresholds fire once per GAME, attempt ordinal demoted to
metadata), smoke 30/30 including the new cross-boundary assertions, dataset
copy byte-identical. Condition 2's defect-sensitive canary v3: A10 PASS (20/25
games fire per seed), O5 predicate PASS (54 budget deaths, 0 violations), and
**13/13 cross-attempt-waste episodes now fire AND warn by 0.9×B = 135** (every
one first fires at cumulative action 75). Condition 3's context-tax sentence is
in the build-doc addendum: **tokens-per-fire ≤ ~95 (≤ 124 conservative); v2
worst case 3 fires/game ≈ 285 tokens ≈ 0.45% (≤ 0.59%) of the 63k envelope**;
the v1 multi-attempt blow-up scenario (up to 42 fires ≈ 6–8% of envelope) is
structurally impossible in v2. W1 status: seed-1 eval kernel pushed today
(`canivel/arc3-duck-sentinel-eval` v1, RUNNING at filing time).

**Trigger-frequency table (canary v3, fires per (game,seed) unit over 75
units) — published before the W2 gate seals, as systems demanded:**

| fires/game | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| units (of 75) | 15 | 14 | 9 | 37 |

Per-seed: 20/25 games fire in each of v1/v2/v3. Carrier games: ka59 {2,2,3},
re86 {3,3,3}, tu93 {3,2,3} fires across seeds — the target population warns
mid-envelope in every seed. The corresponding (b)-side table (recurrence-counter
trigger frequency) cannot exist before (b) runs; **pre-registered: it is
computed from the W1/W2 ON-seed transcripts and published before the W2 gate
seals**, same format.

**Condition 4 — SEALED (pre-ship, pre-binding-look):** *Before any scored ship
and before the A14 binding look, the tokens/game grep on a scored-run pull must
land within ±15% of the 63k per-game envelope; outside the band,
SENTINEL_BUDGET is re-derived by the frozen formula (measured envelope ÷
measured tokens/action) with no other discretion; and the token-implied action
capacity is recomputed from the W1/W2 ON-seed transcripts before the binding
look, because (b) changes tokens/action by design.* VOID-not-FAIL stands: zero
sentinel events on a run containing any ≥75-action game ⇒ budget unset ⇒
window VOID.

---

## §9. Timestamped hash commitment (checklist item 8)

**The claim being evidenced:** the §10R16 0.99 recurrence-determinism bar (and
the holdout protocol's 0.99/Wilson-0.95 certificate bar) preceded the audit
measurements that were judged against them; and every threshold newly sealed in
this document precedes its measurement.

**Timestamp chain (verifiable):** `latent_state_audit_protocol.md` carrying the
0.99 bar is dated 2026-07-20 08:30:20 local (file mtime) and the bar is quoted
verbatim in the R16 republication (2026-07-20 08:36:28), which was circulated
to and reviewed by the R16 panel — reviewer verdicts collected 2026-07-21 08:32
(`learnings/panel/round16/` artifacts) — strictly before the holdout artifacts
existed on disk (2026-07-22 08:37:32). The in-sample `report.json` on disk is a
same-day regeneration (08:38:06); precedence for the in-sample run rests on the
panel record (the R16 circulation discusses its results), not on its mtime —
stated honestly. These files are currently untracked in git; **this commitment
block, once committed with the circulation, becomes the external timestamp for
everything below** (the daily-loop commit is the standing procedure; hashes
recomputed by any reviewer must match this table).

SHA-256 (computed 2026-07-22, `sha256sum`):

| artifact | sha256 |
|---|---|
| learnings/war_room/latent_state_audit_protocol.md | `f8720e8c001b574bb3fcb21f4350ac4c7b0f9b92ce00b08a6532a0e131676038` |
| scripts/latent_state_audit.py | `45a62efa47e90b1c5e8bd133a7db7f1b144cb871ecb29ebc71868007544c21e0` |
| runs/latent_state_audit/report.json | `fe5e1500ad5be68ce34e3adbc4ee432a48d52e4c76138874dfbb847726430479` |
| runs/latent_state_audit/holdout_report.json | `59e321b9fee5663319c2e929d53a1e02a02c195c5c0045c8863834994dd12e22` |
| runs/latent_state_audit/holdout_report.md | `ede25b5db2d72e8fc98dfa963831c36450804c18ea184094cea68dd872996375` |
| duck_eval/sentinel/budget_sentinel_patch.py (v2) | `6a28592c4a0ff637c524a23da4457239c90368ddbadb2640b4ed0db1bf659ed6` |
| runs/sentinel_attempt_unit_b150.json | `7e38d5a07b211bdcf3d7937e2eade3f42492ede2fb72b208723c232989bcb279` |
| runs/sentinel_canary_v3_b150.json | `5c2196a34771e38dbdbb56ff22d33c2c4ad60df206a6f719062e65ddf2e09e10` |
| runs/a17_repair/a17_repair_compute.py | `f2dd404d4cc0568b2bac5f6b985ca767a30027d1fbd8b13f57eedba66de51e34` |
| runs/a17_repair/per_seed_table.json | `932dc45a91c14e861cde3e02740107ff0dcf9871b0941c53707e47b2d3e66c8b` |
| runs/a17_repair/false_nogo_bootstrap.json | `8c62e63f39f122ab9cdeaa16fdd9e0372d2824920b783caa90be3983efc6beea` |
| runs/kernel_pulls/war_eval_v1/benchmark.json | `42fcc7459471f05ee4a5a893477124be83ed4a786d6064d267e410293c3b4b30` |
| runs/kernel_pulls/war_eval_v2/benchmark.json | `7e70e924ea6e0b91a4080febe2525be37a1367206c34164592953b6160157652` |
| runs/kernel_pulls/war_eval_v3/benchmark.json | `4c2aaca0bee7130e8dff50c10decb72acc50f5915b8881ba9e42350151aeab46` |
| runs/kernel_pulls/w0_eval_s1/benchmark.json | `75a88d03373e48f8b3df0dce04811706d3b3f32b94530641a8eb55b76c705cf1` |
| runs/r17_sealing/sketch_sensitivity.py | `452661d9a4394521a56b00d95d48f0b54e693083737b1b2b98321e8d454c576e` |
| runs/r17_sealing/sketch_sensitivity.json | `ccfd0e966e9257b178917ca04d489a1e1f44e3e5835e4a8cdf5a365e6d73d064` |
| runs/sealed/r17_thresholds.json | `d4a27e51f7faa5a99af8e6947ced2d63af7c2020cd13b80aca965bfed52b38b7` |
| learnings/war_room/a17_72b_screen_scope_v2.md | `3150fadb72e800d6611ebb898799fa996bda09904cf7f97ac22b93e97f6a87cf` |
| learnings/war_room/sentinel_q2_discharge_2026-07-22.md | `314045800be30dc376c89c3a210307b34dcf86bada80353d9428b80194a5ac48` |
| learnings/war_room/grinder_design_R16_republication.md | `f670127de642bd2887134221066346a85c43f9efde2566048a9f736ac2fa3338` |
| runs/schema_traces_mining/report.md | `bd6f7280bddac9a18d92d2916a7114eb6a05362f322d1a469f0fb0b1ac127e6e` |

`runs/sealed/r17_thresholds.json` (created before any newly sealed measurement
runs) carries every threshold sealed in this document in machine-readable form,
per the §13R16 standing procedure — the same standard applied to all R17 seals.

---

## §10. Statistical sensitivity annexes (checklist item 9)

### §10.1 df=2 σ̂ sensitivity band (methodology R3)

σ̂_run = 0.189 on df=2 is the frozen null; its 90% χ² CI is **σ ∈ [0.109,
0.835]** — brutally wide, published as such. The panel ratifies the RANGE below,
not a scalar; all boundaries stay frozen (guard −0.28; dismantle −0.10); the
n=3 fallback band (§5.4) is used throughout (SE(Δ) = σ·√(2/3)):

| quantity (frozen boundary, true σ varying) | σ=0.109 | σ=0.189 (point) | σ=0.835 |
|---|---:|---:|---:|
| dismantle P(trip \| Δ=0) | 0.13 | 0.26 | 0.44 |
| dismantle power at true −0.25 | 0.95 | 0.83 | 0.59 |
| guard false-kill / window at −0.28 | 0.001 | 0.035 | 0.34 |
| guard familywise (3 windows) | 0.003 | 0.10 | 0.71 |

Honest reading, stated before any look: at the df=2 upper tail the per-window
guard is toothless and the dismantle branch approaches a coin-flip in both
directions — the real net-negative protection at high σ is the deterministic
mechanism prongs (false-kill ≈ 0 by construction) plus the cumulative look's
paired sign test, whose size does not depend on σ̂ at all. Non-binding
corroboration that σ is not at the pathological upper end: the pooled LB ledger
(different unit — LB draws, not lc/game) sits at σ̂ = 0.144 on n=14.

### §10.2 Binomial sketch at 4 and 6 spurious pairs (methodology R4)

§5R16 assumed 0–2 spurious pairs. Republished under the sealed sketch model at
total spurious T ∈ {4, 6} (sign fair-coin), exact enumeration:

| branch | baseline (S± ≤ 2) | T=4 | T=6 |
|---|---:|---:|---:|
| B− / EWM-out | 0.04 | 0.06 | 0.08 |
| B− / EWM-in | 0.08 | 0.13 | 0.09 |
| B+ / EWM-out | 0.19 | 0.19 | 0.17 |
| B+ / EWM-in | 0.27 | 0.21 | 0.20 |
| **size under pure null (0 true positives)** | **0.000** | **0.000** | **0.016** |

Two facts the panel should have in hand: the gate's SIZE stays ≤ α even at 6
spurious pairs (the sign-test critical scales with n — the design is
noise-robust in exactly the way that matters); and the B− pass probability
RISES with spurious count (0.04 → 0.06 → 0.08) — i.e. at B−'s expectation a
PASS is substantially a luck event, which is R5's point and feeds the rule
below.

### §10.3 Sealed B− PASS reporting rule (methodology R5)

**Sealed:** a B− PASS at the binding look is reported with (i) the realized
exact sign-test p-value, and (ii) the likelihood ratio P(PASS | B− stack as
priced) / P(PASS | zero-effect world) computed under the sealed sketch model at
the OBSERVED nonzero-pair count, both published in the look report. **A lucky
B− PASS does not upgrade the stack's evidentiary label**: the label remains
"mechanism-verified, hygiene-grade; score evidence weak," no wall claim may cite
it, and it licenses no re-pricing of any component. (Illustration at the sealed
expectation: P(PASS) ≈ 0.04 under the priced B− effect vs 0.000–0.016 under
no-effect-with-underestimated-noise — a PASS moves belief mostly toward "the
spurious-pair count was underestimated," which is why the label freeze is the
rule.) A B− FAIL is likewise ≈ uninformative against the mechanism claims (they
are carried by the deterministic prongs); it is informative only via the
dismantle branch.

---

## §11. EVIDENCE ANNEX — Schema-harness released traces (external; nothing here re-opens a seal)

`runs/schema_traces_mining/report.md` (mined today; integrity: their bundled
scorer reproduces 25/25 wins, 183/183 levels, mean RHAE 98.98% on our copy;
**engine versions hash-identical to our audited set**, so this is evidence on
our exact game class). Unverified externally; public set only. Three
confirmations FOR this sealing text, three agenda items for the panel:

**Confirmations (cited as external validation; no verdict changes):**
1. **Full-replay is the working contract at 99% RHAE.** Schema's certified
   world models are stateful-from-RESET (`init_state(entry_grid)`, state
   threaded through every predict) — independent running-code confirmation of
   the FULL-REPLAY-ONLY banking rule sealed in §1/§4.
2. **The holdout DROPs are support starvation, not wrong physics.** Their
   certified models implement latent action-counters with affine-modular laws —
   exactly our parity/mod-k hypothesis class, on the same versioned games
   (wa30 mod-rate fits, ka59 parity-inverted bar, tr87 floor(n/2) bar). The §1
   verdicts STAY UNRESOLVED (our certificates failed on OUR support; that is
   what the certificate means), but the panel should read them as "not
   certifiable from 7–8 streams," not "no phase mechanism exists."
3. **mp@0 ≈ 0 (5 events / 10,303 actions):** re-rooting every plan in the
   observed frame + recovering latent state by history replay eliminates
   step-0 divergence — the exact failure mode our EWM dry-run died on.

**Agenda items (questions for reviewers, NOT sealed claims):**
- **(i) Sentinel-budget tension, flagged for the NEXT budget litigation (not
  now):** Schema's winning regime spends median 283 / mean 412 actions per game
  (their cap: 3000); games needing model revision under aliasing took 3–8× our
  B=150 game-envelope (wa30 956, dc22 1205, s5i5 643, re86 615), and their 18%
  mispredict tax alone would consume ~27 of 150 actions. Any sub-200-action
  budget structurally forbids the revise loop that produced these numbers.
  B=150 is sealed for THIS regime (it models our token-implied capacity, which
  is real); the tension is tabled for W2/post-look review, not acted on.
- **(ii) EWM contract v1.1 direction (certification-as-resync):** proposed
  contract for the NEXT registration cycle — sim state = frame + explicit phase
  register ticked per action-class (clicks may not tick — the N5 prune_trace
  trap, now seen in their sk48 model as "clicks and a7 are free"); on
  mispredict: truncate plan, re-root in observed frame, re-derive phase by
  replaying history through the step function, backtest, revise; abort only
  after post-revision certification fails; UNRESOLVED trio g50t/m0r0/sk48
  demoted from abort-by-default (Schema closed all three first-line). This is a
  registration proposal for panel ruling; it changes nothing sealed today.
- **(iii) Escalation economics for A17 context:** their operating rule was
  cheap-first, re-run <80-scorers with the stronger config, keep per-game max —
  11/25 games escalated (Claude side), and the both-stacks hard core (bp35,
  dc22, lf52, sc25, sk48, sp80, su15, tn36) is NOT our aliased class;
  long-horizon level structure drives escalation. Same shape as our 27B→72B
  plan; offered to the panel as context for the A17″ gate's economics, not as
  gate input.

**Q8 re-tabled:** the 2026-07-19 dream digest ruling (R16: NOT-ADDRESSED) is
re-tabled for an affirmative ruling this round; the §15R16 review stands
("nothing actionable; nothing panel-worthy") as the authors' unobjected default.

---

## §12. Changelog — R17 checklist item → where discharged

| # | R16 checklist item | discharged at |
|---|---|---|
| 1 | Composition sentence, all four branches, per-branch P(pass) | §2 (sealed sentence + table; P = 0.04 / 0.08 / 0.19 / 0.27; B+ rows provisional pre-A16; sketch model sealed + artifact) |
| 2 | Held-out resolver validation; failures → UNRESOLVED; Wilson LB ≥ 0.95; per-stream table; consumers re-pointed | §1 (10/11 DROP; sb26 hist1 sole survivor; CLEAN 10/11 confirmed, tn36 flagged; EWM/banking/resurrection re-pointed); propagated into §2 (P(pass)), §3 (§2R/§4R tables), §7 (carriers) |
| 3 | Engine-drift precondition; drop rule; fallback trigger; + Q7 conditions (structural sentence, frozen band, config-diff enumeration) | §5 (identity + DROP + ≥3-drops rule; structural sentence §5.2; n=4 band frozen §5.3; **enumeration found warpack diff → band ILLEGAL → sealed fallback to n=3 W0 band triggered**, dependents recomputed) |
| 4 | B+ rows flagged pre-audit; A16 mandate extended to replay-action cost; recirculate before W3; B− seals as-is | §3 (flags in §2R row + §4R rows) + §4 (replay cost priced from disk: ka59 16–77 / re86 35–80 / sc25 107–114 actions; denomination rule sealed; deliverables before W3); B− sealed |
| 5 | A17′ corrected and re-filed per C1–C3 + v4-ledger requirement | §6 pointer → `a17_72b_screen_scope_v2.md` (single boolean, ρ_action, 4-run comparator, per-seed Σ table, false-NO-GO 0.000/≤0.0625 with chosen remedy, cost reconciliation, v4 ledger = YES) |
| 6 | EWM Stage-1 measurement config sealed (carriers, sim config, phase-augment status, per-subset pricing) | §7 (11 CLEAN carriers listed, tn36 flag stated; phase-blind key pinned; **phase-augment NOT registered — no qualifying game; future certificate ⇒ new panel look**; per-subset prices; gate explicit; Rodionov paragraph) |
| 7 | Sentinel condition 4 sealed; trigger-frequency table before W2 | §8 (±15% envelope check sealed; capacity recompute from ON-seed transcripts; fires/game table {15,14,9,37}/75 published; (b)-side table pre-registered) |
| 8 | Timestamp-verifiable hash commitment (0.99 bar precedes audit; same standard for all seals) | §9 (sha256 table; timestamp chain incl. honest regeneration caveat; `runs/sealed/r17_thresholds.json` created pre-measurement) |
| 9 | Sensitivity annexes: df=2 band; 4/6-spurious sketch; B− PASS reporting rule | §10 (σ ∈ [0.109, 0.835] band with trip-rate table; T=4/6 sketch with size ≤ α shown; B− PASS rule sealed with label freeze) |

Binding single-reviewer items also carried: llm-agents Q6 Rodionov paragraph
(§7); systems #2 trigger table (§8); systems #3 context-tax line (§8);
methodology R5 reporting rule (§10.3); rl-planning Q5 v4-ledger (§6); Q8
re-tabled (§11).

---

The recalibrated A14 gate — as amended by §2 (sealed composition), §3/§4 (B+
pre-audit flags + A16 extension; B− sealed), §5 (n=3 fallback control band +
drift precondition), §7 (EWM config), §8 (sentinel v2 + condition 4), and
governed by the §9 hash commitment — **SEALS ON THIS CIRCULATION.** The two
fallback control seeds (4.4 GPU-h, scheduled) and the A16 recompute
recirculation are pre-look obligations, not seal conditions. The wall-closer
remains war-v4 (A17″, pre-Aug-1); nothing here is smuggled toward the wall.

END OF R17 SEALING


---

## §12. ADDENDUM (2026-07-23, pre-circulation) — post-filing evidence

Filed the morning after the base document. Three pieces of evidence landed
after §1–§11 were frozen; none alters a sealed threshold (all thresholds in
`runs/sealed/r17_thresholds.json` predate these measurements — the seal-before-
measure discipline held), but all three bear on how the panel should rule.

### 12.1 Sentinel W1 live run (seed 1) — the condition-4 measurement

The v2 sentinel ran on Kaggle 2026-07-22 12:47–14:59Z (free build, 25 games,
`canivel/arc3-duck-sentinel-eval`; pull + analysis:
`runs/sentinel_eval_analysis/report.md`). Verdict: **mechanism PASS / score
NULL / behavior STRONG-NEGATIVE** — "fires, doesn't pay" (build-doc Open Risk
#2 realized):

- **Mechanism:** 22 sidecars + 56 stdout `SENTINEL v=2` lines agree exactly;
  every threshold fires ≤ once per game (≤3 events/game cap held everywhere);
  cumulative game-envelope keying proven live (fires at identical cumulative
  actions 75/113/135; no re-arm across attempts — e.g. ar25 crosses 50% in
  attempt 1 and 90% in attempt 2 with no repeat). The v1→v2 re-key is verified
  on the carrier games: ka59/re86 fired 3/3 early where v1 was structurally
  blind. Open Risk #1 (inert-if-uncapped) is cleared: the budget export was
  live and the banner present. Missing sidecars for s5i5/tu93/vc33 are
  EXPECTED (lazy file creation on first crossing; all three ended <75 actions;
  file-exists ⇔ ≥1 fire).
- **Envelope (condition 4):** tokens/game mean 64.3k; 23/25 games inside the
  sealed 63k ±15% band; B=150 requires no re-derivation. **Condition 4 passes
  on this pull.**
- **Score:** sentinel-arm mean 0.855 vs certified war 3-seed baseline mean
  1.454 (−0.60; −0.72 vs the paired seed). The gap is carried almost entirely
  by three high-variance NON-target games (ar25/ft09/sp80); baseline seed
  spread is 1.16–1.73. The honest call is **NULL/underpowered at n=1**, not
  established regression — but the pre-registered positive-lift framing
  (+0.01–0.03/draw) is refuted as optimistic.
- **Behavior:** the warnings did not change play. 1/22 fired games advanced a
  level after the first warning; 21/22 kept grinding (wa30: 560 actions stuck
  on L1 after all 3 warnings); total actions rose +618 vs baseline. tu93's
  efficient 3.97 draw fired ZERO events and must not be claimed as sentinel
  evidence.

**Proposed ruling for the panel:** seal the sentinel as a certified *observable*
(mechanism half sealed; condition 4 discharged), record the score prong as NULL
with the fires-doesn't-pay label, and register **W2 as a $0 confirmatory-null
free build** (pre-registration: mean inside 1.16–1.73, mechanism clean,
behavior unchanged; no W3 unless W2 is positive). The sentinel's certified
function was always warn-only; lift was window pricing, not a gate premise.

### 12.2 Scoring-function dissection (community, verified) — depth ≫ efficiency

Discussion #728299 reverse-engineers the shipped `arc_agi/scorecard.py`
(reproduced to 1e-9): the 115% figure is a **per-level efficiency cap**, and
the game/LB aggregate is **completion-weighted with a completion cap**. Two
consequences the base document's objectives should be read under:

1. **An unreached level costs its weight twice** (it contributes zero AND
   shrinks the completion factor): 4/6 levels ≈ 47.6, not 66.7. Deeper levels
   dominate marginal score.
2. **Overshoot decays quadratically** (2× baseline actions on a completed
   level ≈ 25%, not 50%) — inefficiency on completed levels is cheap relative
   to failing to reach the next level.

This independently explains 12.1's fires-doesn't-pay: a stop-grinding signal
cannot buy score unless the freed actions convert into level *depth*. It
re-points EWM/A17 value at reaching deeper levels, not action-trimming, and it
resolves the long-standing 1.15x-vs-1.0x watch-item from code (1.0x
completion-weighted is confirmed as the right LB mental model). The same post
ships a no-API-key offline scoring atlas of the 25 bundled games — adopted as
a free deterministic local scoring oracle (zero cloud spend).

### 12.3 External literature (2 ADAPTs, both strengthening the gate)

- **arXiv 2607.12227 (Jul 14), "Rethinking the Evaluation of Harness Evolution
  for Agents":** held-out evaluation of the tune-on-public/report-on-public
  pattern shows only +0.6 avg transfer. This is the external charter for the
  base document's gate discipline: **held-out beat-null10, never
  beat-baseline-on-the-tuning-games**, and semi-private weighted over
  public-25. (Also the right lens on Schema's public-only 99%.)
- **arXiv 2606.24842 (Jun 23), "World Models in Pieces: Structural
  Certification":** certification is **transition-local**, not model-global.
  This reframes §1's holdout collapse as the expected outcome (sb26 is the one
  transition-local certificate that generalized) and tightens the EWM v1.1
  wording: BFS-in-sim is sound only over transitions carrying a live local
  certificate. Proposed as wording (not threshold) amendment to the sealed EWM
  measurement config.
- **A17 boundary note (Kamradt critique, Jul 21):** the 27B→72B
  escalate-on-low-score template stays a **serving-cost-only** policy —
  per-game score feedback must never re-enter the agent's context, or the
  public-set leak the critique penalizes is imported. To be recorded in the
  A17″ amendment text.

### 12.4 Ledger as of circulation

Frozen control n=10: mean 0.975, σ̂ 0.156. Pooled n=15: mean 0.962, σ̂ 0.144.
Overnight draw 0.82 (band-typical). LB: field compressed — the 1.44 wall is
now the bottom of a dense 1.44–1.60 band; our 1.33 slid #44→#45 (erosion, not
regression). No new public clones above zoli800's 1.39 byte-identical draw.
