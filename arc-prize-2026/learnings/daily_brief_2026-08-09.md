# Daily brief — Sunday 2026-08-09 (R24 DAY)

Protocol: STEP 1 collect + deep review, STEP 2 panel (full, Sunday cadence), STEP 3 develop,
STEP 4 validate/submit, STEP 5 loop. Weekly items (KAOS dream, fingerprint report) run.

---

## 1a. Result deep-dive — the 0.89 draw

**Not just the number.** The 2026-08-09 overnight submission (frozen-fork filler,
`canivel/arc3-duck-repro` v3) came back **COMPLETE at 0.89**, verified directly against the Kaggle
API rather than taken from the daemon log.

- **Pre-registered expectation: met.** No expectation was pinned to this draw beyond "interior".
  z ≈ −0.31 against the prior ledger — squarely mid-band.
- **Ledger: n=26, mean 0.9365, s 0.1540** (from n=25, 0.9384, 0.1569).
- **Watch-rule: stays resolved-STATIONARY.** The 08-07 firing (0.77, 0.78) was resolved the same
  day by change-point p=0.757 / Mann-Kendall p=0.62 / no CUSUM breach, with pair-probability
  0.19–0.51 under sealed nulls. This is now the **second consecutive interior recovery**
  (0.87 → 0.89), trailing-4 mean 0.9025. Re-arms only on a fresh consecutive sub-0.80 pair.
- **Mechanism evidence: none available and none expected.** This is a byte-identical frozen
  artifact; the draw carries information about the *artifact-noise distribution*, not about any
  agent mechanism. It is filler by design.
- **What the delta implies:** nothing actionable. The honest reading is that the control ledger
  is behaving exactly as a stationary process should, which is what makes it usable as the
  denominator in gate arithmetic — and §5 below is precisely about discovering that the
  *numerator* side of our gate arithmetic was miscalibrated.

**Promotion arithmetic, recomputed at n=26** (illustrative only — no scored draw requested):
`0.9365 + t(0.95, df=28) × 0.1540 × sqrt(1/4 + 1/26)` = **1.0773** mean-of-4 (was 1.0823 at n=25).

**Leaderboard — the gold cutoff moved for the first time in five days.** Top-13 line rose
**1.56 → 1.58**. Helmut AGI entered at 1.61 (#7); the four-name 1.58 pack slid to #10–13, evicting
three 1.56 entries to #14–16. Top-5 prize cutoff holds 1.61. Head static (KOJIMA 1.86 resubmitted
unchanged, Andy liu 1.69, Lord Han Solo 1.65, GeniusYY 1.64). **Our 1.33 remains below #49; gap to
gold is now 0.25** and widening. Archived `runs/lb_daily/lb_2026-08-09.csv`.

## 1b. Discussions sweep — `learnings/sweeps/discussions_2026-08-09.md`

Cadence resumed today (every-other-day since two quiet sweeps). Kaggle's SPA discards the `sort`
param, so ordering was cross-checked by monotonic topic ID (highest on board = 733865) plus page-2
enumeration. **Two new topics since 08-07.**

| Post | Disposition | Reason |
|---|---|---|
| **733865 — RPS ARC-AGI 3 Technical Report** (Jason Feng, 08-08) | **ADAPT (design) / MONITOR** | Three solutions on **the Tufa Duck harness with Qwen3.6-27B — our exact substrate**, via notebook-level runtime hooks. Only ARC-AGI-3 harness work found in three sweeps at 27B rather than frontier ⇒ retires the "implementable at 27B?" question for the P3 memory arm **on design grounds only**. Contributes two-timescale memory (consolidate at first level-clear, then refine-not-rewrite) and a working prior for the L4 consult gate. **Severely de-rated: he ranks 177th — below our own 1.33 — with no quantitative comparison, no ablations, four mechanisms confounded.** Design evidence, zero efficacy evidence. |
| **733697 — fresh-kernel fix** (Antoine Matemane Mahirwe, 08-07) | **ADOPT (operational)** | Independent third-party confirmation of our `feedback_fresh_kernel_slug` rule: 7 generic `system error` submissions on an iterated slug incl. a faithful rebuild of known-good code; brand-new slug worked first try. Unconfirmed n=1 claim that ERROR submissions may not count against the daily limit — **flag, do not act**. |

Monitors: borro1980's merge solicitation still has **zero uptake** from all five 1.47–1.58 targets
since 08-05. Reki 732854 still unanswered. Next sweep **08-11**.

*Correction carried from the panel:* the two "free harness checks" this sweep proposed (Swarm
`record=True`, unpruned `self.frames`, ~1 GB JSONL/game) target the **dead forge-era stack** — zero
matches under `duck_eval/`, and the full 25-game pull is **224 MB, not 25 GB**. Withdrawn.

## 1c. Research sweep — `learnings/sweeps/research_2026-08-09.md`

**The 08-07→08-09 arXiv window is genuinely empty** — API frontier is 08-06, no Sat/Sun
announcements; next real batch Monday 08-10. A quiet window is a valid result.

The real finding is **a correction to yesterday's own sweep**: the claim that the banking/replay
field "produced exactly one hit… nothing new" is **false**. Re-enumerating 08-05/06 surfaced a
**~20-paper skill-library / self-evolution cluster logged nowhere in `learnings/`**. R24 §3(c)
leaned on that premise and must withdraw it — though the correction does **not** promote lane (c):
the cluster is cross-task skill libraries, not within-game trace replay, and **2608.05810 (VaG)**
finds skill accumulation **non-monotonic and irreversibly harmful** past a pool-size threshold.

| Item | Disposition | Substance |
|---|---|---|
| **2608.06370 — "The Bitter Lesson of Tool Calling"** | **ADOPT (design)** | Programmatic Python-stub tool calling beats JSON in **11/14 models** and **improves +5.5% under context flood** where JSON degrades −2.3% — **opposite sign to everything A22 measured**. The duck's single-`python`-tool shape is already the favoured form. **Caveat stated at panel: zero open-weight models**, so the weak-model hole is NOT closed. |
| ↳ same paper, **[UNVERIFIED — 2nd hand]** | hold | Filesystem-backed store reported degrading 32%; if it survives a direct read it favours **in-process namespace over Tycho workspace files**. Not sealable on second-hand evidence. |
| **2608.05906 — MERIT** | **ADAPT** | Training-free dual-polarity memory (verified corrections + observed *unsuccessful* directions) on **Qwen2.5-7B** — weak-model range. Lane (b)'s blocker moves from "infeasible without training" to "feasible, thin evidence" (+3.45pp, text-to-SQL, needs an oracle). Rank unchanged, reason changed. |
| **2608.06196** | **ADOPT (two ways)** | Self-authored query benchmarks inflate **up to 44pp** — a number under the provenance de-rating rule. And a **typed knowledge graph over skills *hurt* retrieval −11.2pp (p=0.0007)** ⇒ **do not build a relational graph for P3.** |
| **2607.20709 — OO Agents** (07-22, backfill gap) | **MONITOR only** | Possibly a 4th convergent team but publishes **no ARC-AGI-3 numbers** — may **not** be counted toward the three-team convergence argument. |
| ARC-AGI-3 / test-time learning / compaction theory | nothing new | Third consecutive quiet compaction sweep ⇒ §5.2 death record sealable as written. |

Declared coverage gap: arcprize.org leaderboard fetch **failed** (client-rendered), so the official
Opus 5 = 30.2% counterweight is **≥2 days stale**.

## 2. Weekly items — `learnings/weekly/weekly_2026-08-09.md`

- **KAOS dream** (run_id=8, dry_run): 3 episodes, 221 memories scored, 0 skills scored. Hot memory
  is pure recency (all 0 hits). Consolidation proposals, **verbatim**: *"No structural changes
  proposed this cycle. Library is stable."* — **nothing for the R24 agenda**, as expected.
  Digest `Dreams/2026-08-09-122435.md`.
- **Failure fingerprints:** 16 incidents, 8 recurring families, **newest incident 2026-07-08** — no
  new failures in ~4.5 weeks under the preflight regime.

```
family                         n  first       last
--------------------------------------------------------------
class:ERROR:none               7  2026-05-26  2026-06-28
provenance:scratch-built       5  2026-05-26  2026-06-28
slug:canivel/arc3-final        4  2026-05-26  2026-06-10
class:COMPLETE:0.00            3  2026-03-29  2026-06-10
slug:canivel/arc3-forge35      3  2026-04-24  2026-06-22
slug:canivel/arc3-pilot-eval   3  2026-07-07  2026-07-08
t1:07d0f5248c48401d            3  2026-07-07  2026-07-08
class:COMPLETE:null-band       2  2026-06-01  2026-06-08
```

All three top families are **build/provenance-infra** modes, not agent-algorithm modes: they bound
**execution risk at S2**, and say nothing about which lane is right. *(Panel correction: P1 does
**not** sit on the `scratch-built → ERROR` path — it is a monkeypatch on a frozen fork.)*

## 3. R24 panel — full record in `learnings/war_room/r24_minutes_2026-08-09.md`

**5/5 MAJOR-REVISION** (5,6,6,6,6). 6 FATAL, 27 MAJOR. Pass criterion not met — **but every
reviewer independently ratified the lane decision**, and four of five said in their own words that
the lane and the free offline work should be authorised today with only the S2 seal held. The
verdicts attach to **instrument specification, not strategy**.

**Ratified (uncontested):** lane (a) state-externalisation with Tycho as artifact schema, (b) as
component arm, (c) on its own clock · workstation-LLM authoring in-bounds of the zero-budget rail ·
provenance de-rating as a standing rule · sandbox risk-class trigger · refuted-list micro-arm
dropped. **The A22 death record was ratified at panel close and VACATED hours later** — see §6.

**The four findings that changed the plan:**
1. **Wall clock, not actions, is the binding resource** (rl-planning + systems, independently):
   all 50 game-runs in two pulls ended at ~7,920 s against `max_runtime_s_per_game=7920.0` with
   `max_actions_per_game=None`. §2.3's rationale for the lane's *attractiveness* is false on our
   rail, and the Δlc gate **cannot separate "harmful" from "slower"** — P1 could die on latency and
   be recorded as the fourth mechanism negative.
2. **The inherited non-harm gate fails a true null ~half the time** (methodology, measured on 90
   null-vs-null pairs in `runs/null10`): mean-leg FPR 12.2%, worst-game leg **50.0%**, conjunction
   **51.1%**. The −0.128 digit was transported without its calibration (estimated vs a 10-run
   average, applied vs a single run). **This is the gate the campaign has been killing lanes with.**
3. **S1/L0 would measure a degenerate constant** (prog-synthesis): **0 of 25 sims implement
   abstention**, so Tycho "coverage" is constant 1.0; only 3 sims hold hidden state; the carrier
   gate is unreachable and circular (real abstention needs L1, which L0 gates). Independently
   corroborated by the runnability check from the implementation side. Bonus finding: **the
   "91.7% held-out" number was never held out** (`split=all`).
4. **P1's mechanism is contradicted by ~6 strings in our own harness** (llm-agents), so **K4 can
   pass validly but cannot fail validly**; §6.1's byte-identity invariant is self-voiding; and
   `SAFE_MODULES` lacks `dataclasses`/`typing`/`enum`, so **the Tycho `State` dataclass is not
   constructible in our sandbox**.

**Decisions:** lane (a) ratified · **S2 not sealed, push not spent** · **S1 held** (runnable but
uninterpretable as written) · **S1b demoted** to confirmation and deferred (it selects on its own
outcome — the 11-game set excludes both games that actually failed) · **gate recalibration
promoted above all lane work** · no submission change · no second round; next full panel 08-16.

## 4. Development — what actually got built today

Per the process restructure, all substantial work ran as background agents.

- **`duck_eval/r24_prep/s1_s1b_execution_plan_2026-08-09.md`** + two dry-run-verified runners
  (`s1_threaded_replay.py`, `s1b_bank_refire_noprune.py`). Verdict **RUNNABLE** — all assets
  verified present (**25** sims, not 24; 25 traces; the 11-game set; `prune_trace` located),
  CPU-only, <10 min, $0. Held per §3 rather than fired, because the panel showed the output would
  not be evidence. Also surfaced a **real bug**: `scripts/ewm_replay_dryrun.py` never reset sim
  module state, so g50t/re86/tr87 were measured with desynced hidden counters.
- **Gate recalibration** commissioned as the top free work item — independent reproduction of the
  null-calibration FATAL, calibrated replacement thresholds at α=0.05/0.10, a ruling on whether a
  worst-game leg is salvageable at 25 games, and a verified re-examination of the A22 death record.

## 6. POST-PANEL — the gate recalibration, and the A22 death record vacated

Full record: minutes §3.3a and §5.1; `learnings/sweeps/gate_recalibration_2026-08-09.md`,
`runs/gate_recalibration_2026-08-09.json`.

The recalibration **reproduced every null-calibration digit** independently (mean-leg type-I
12.22%, worst-leg 50.00%, conjunction 51.11%, null sd 0.1223) and audited `runs/null10` as a
genuine same-config null. Then it found what the panel did not have — and I verified the decisive
arithmetic myself rather than take it from an agent:

**`war_eval_v1/v2/v3` are three runs of the identical warpack config** (same `label`, same
`solver_label`, same `n_passes`, **byte-identical `git_status.txt`**, 07-14/15/16) scoring
**22 / 15 / 13** levels:

```
v3 - v1:  mean dlc -0.3600   worst -2      <-- A22 v2.1's headline "harm", exactly
v2 - v1:  mean dlc -0.2800   worst -1
v3 - v2:  mean dlc -0.0800   worst -2
```

**A22 v2.1's −0.360 / worst −2 is bit-for-bit the gap between two runs with no compaction in
either.** Warpack run variance is **4.83× vanilla's** (p=0.038), so `null10` understates the
correct null for warpack arms; A22 was screened against `war_eval_v1`, the **22-level high
outlier** of its own family. Re-baselined on the 3-run mean: **v1 +0.013, v2 −0.107, v2.1 −0.147 —
all PASS.** And **−0.128 is an arithmetic identity, not a statistic**: (12 − 15.2)/25, sourced from
a 12-level arm and applied against a 22-level one that itself screens at **+0.272, p=0.0074**.

**Ruling: A22 is UNRESOLVED, not DEAD** — three strikes by a broken instrument against a lucky
baseline. This does **not** resurrect the lane by default; lane (a) was ratified on independent
grounds and keeps the budget. But **no lane may be declared dead on the old gate**, and the
"eviction is intrinsically harmful" story is no longer evidenced by our own data.

**K3′ replaces K3 effective immediately:** pair against the per-game mean of **m ≥ 3 same-config
baseline runs**; PASS iff `mean Δlc ≥ −t(0.95,df=m−1)·s_base·√(1+1/m)`; fallbacks −0.200 at m=1
(2.2% type-I), −0.190 at m≥3, −0.160 at α=0.10. **Worst-game leg dropped** as structurally
uninformative at 25 games. One reviewer claim corrected: pairing inflation is **1.28×**, not 1.78×
— so the defect was primarily the *number*, not the procedure.

**Cost this imposes:** every future arm needs **m ≥ 3 same-config baseline runs before it can be
screened at all**, and a **warpack-specific null** is owed before any warpack-family screen.

## 5. Open questions → R25 (2026-08-16)

1. **What replaces the non-harm gate?** Everything downstream waits on this. Is a worst-game leg
   salvageable at n=25 games, or must it become a mean/quantile statistic?
2. **Does the A22 death record survive recalibration in full?** Provisional ruling: the death
   stands on the mean leg; **"harm is monotonic in eviction pressure" is downgraded to "consistent
   with, not demonstrated by"** (steps 0.120/0.040 vs null sd 0.1223; ρ shift p ≈ 0.33).
3. **Can lane (a) be screened at all on a wall-clock-bound rail?** A matched-action-prefix endpoint
   plus latency instrumentation is the proposed answer — is it sufficient?
4. **P1 before P3, or P3 before P1?** Reopened. Our own `transcript_forensics.md` ranks a
   persistent hypothesis ledger (= P3) as fix #1, the only 27B-on-our-substrate precedent is
   memory, P3 is cheaper — and P1's "cheap decisive falsifier" advantage evaporated under FATAL #4.
   Note §3(b) is factually wrong: `cross_level_notes` is deliberately **not** wiped, so P3 reverses
   a deliberate design choice and must argue against it.
5. **Is L0 rescuable at all**, or does the abstention gap mean the exec-wm line simply cannot be
   re-verified without building L1 first — and is L1 affordable now that workstation authoring is
   ruled in-bounds?
6. Direct read of 2608.06370 to confirm or drop the 32% filesystem-store claim.
