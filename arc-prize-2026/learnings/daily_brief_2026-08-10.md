# Daily brief — Monday 2026-08-10

Protocol: STEP 1 collect + deep review, STEP 2 panel, STEP 3 develop, STEP 4 validate/submit,
STEP 5 loop. **No panel today** — full panels are Sunday cadence; R24 ran 08-09 and the next full
panel is **2026-08-16**. **Discussions sweep is not due** — cadence resumes **08-11**. Weekly items
ran yesterday and are not repeated.

**The day's spine is the K3′ fallout and the A22 disposition, not lane-(a) building.**

---

## 1a. Result deep-dive — the 1.05 draw

**Not just the number.** The 2026-08-10 overnight submission (frozen-fork filler,
`canivel/arc3-duck-repro` v3, fired 00:07:11Z) returned **COMPLETE at 1.05**, verified against the
Kaggle API rather than taken from the daemon log.

- **Pre-registered expectation: met.** No expectation was pinned to this draw beyond "interior".
  **z ≈ +0.74** against the prior n=26 ledger (0.9365 / 0.1540) — comfortably interior, upper half.
- **Ledger: n=27, mean 0.9407, s 0.1526** (from n=26, 0.9365, 0.1540). Mean up, dispersion down.
- **Third consecutive interior recovery.** The 08-06/08-07 sub-0.80 pair (0.77, 0.78) has now been
  followed by **0.87 → 0.89 → 1.05**, the highest draw since 08-05's 1.21. Trailing-4 mean
  **0.8975**, up from **0.8275** yesterday.
  *(Correction, 2026-08-10: the "0.9025 yesterday / 0.8025 two days before" figures carried in
  ITERATION_LOG.md are wrong — recomputed from the Kaggle API, yesterday's trailing-4 was 0.8275
  (0.77+0.78+0.87+0.89). The direction of travel was right; the comparison numbers were not.
  Root cause: the ledger existed only as hand-carried prose. Now reproducible via
  `scripts/ledger.py` → `runs/ledger.json`, which re-derives the record from the API and
  confirms the headline n=27, 0.9407, s=0.1525, z=+0.74 exactly.)*
- **Watch-rule: stays resolved-STATIONARY.** 1.05 is far above the 0.80 line, so the rule does not
  re-arm; it re-arms only on a *fresh consecutive sub-0.80 pair*. The 08-07 firing was resolved the
  same day (change-point p=0.757, Mann-Kendall p=0.62, no CUSUM breach).
- **Mechanism evidence: none available and none expected.** This is a byte-identical frozen
  artifact. The draw carries information about the **artifact-noise distribution**, not about any
  agent mechanism. It is filler by design.

**The interpretation the log does not yet make explicit — and it is the one that matters.**

Per the **R25-N3 finding**, `ρ̂_draw ≈ 0` (`runs/rho_estimate/`, `scripts/rho_analysis.py`; three
independent routes agree — variance-components σ²_a clamps to 0, direct subset-split ≤ 0 at every
split, LB frozen-fork cross-check ≈ 0; flip threshold is ρ ≳ 0.3). **A high public draw therefore
pays approximately nothing at private selection.** So 1.05 is a **mean/dispersion datum, not a win**.
It is worth exactly what it does to `n=27, 0.9407, 0.1526` — which is a small, real, favourable
move in the two quantities that *are* the currency — and worth nothing as a public number. Public
max is **unchanged at 1.33**; 1.05 does not touch it, and under ρ̂ ≈ 0 that is not a loss.

**Two caveats travel with ρ̂ and must not be dropped when this reasoning is reused:**
1. **It is underpowered on the high side.** The nominal "13 seeds" are really **3 usable same-config
   war draws** (`null10` is all-zero, zero variance, and contributes nothing). Resampling puts the
   **upper CI near ~0.8**. ρ̂ ≈ 0 is the point estimate and the direction is consistent across three
   routes, but the interval does not exclude a materially positive ρ. Re-measure on any private
   readout or on replicated ~55-game draws.
2. **"Capability pays double" is NOT confirmed.** It is the natural companion claim to ρ̂ ≈ 0, and it
   remains an **assumption, not a fact** — we have no replicated configs to test it with. Do not
   quote the pivot as if both halves were established.

Root cause of ρ ≈ 0, for the record: per-game RHAE is so heavy-tailed that at 25 games neither
draw-luck nor capability produces a positive disjoint-subset correlation. The aggregate is a
which-games-hit lottery.

**Promotion arithmetic, recomputed at n=27** (illustrative only — no scored draw requested):
`0.9407 + t(0.95, df=26) × 0.1525 × sqrt(1/4 + 1/27)` = **1.0801** mean-of-4 (was 1.0777 at n=26).
*(Corrected 2026-08-10: the brief first wrote `df=29`, which is not n−1 for n=27; and 1.0796 came
from the rounded s. Canonical value is now emitted by `scripts/ledger.py`, so this stops being
retyped by hand each day. The `sqrt(1/4 + 1/n)` term is load-bearing — a naive `s/sqrt(4)` bar
gives 1.0661 and would promote arms the sealed arithmetic rejects.)*
The bar went *up* slightly, which is the mechanical consequence of a good draw and is worth noting
before anyone treats a rising ledger as free progress.

**Leaderboard — score-static, second flat day at the gold line.** Zero score movement anywhere in
the top 20. KOJIMA 1.86 #1 (resubmitted 08-10 00:00, unchanged), Andy liu 1.69, Lord Han Solo 1.65,
GeniusYY 1.64. **Gold / top-13 cutoff HOLDS at 1.58** — a second flat day after 08-09's 1.56 → 1.58
step. **Top-5 prize line HOLDS at 1.61.** The only apparent top-20 change is cosmetic: teamId
15520570 renamed *Dinesh kumar Thiyagarajan* → *"Whatever it takes..."* — same team, same 1.50, same
rank 19, **not a new entrant**. **Our 1.33 stays below #49; gap to gold 0.25, unchanged.** Archived
`runs/lb_daily/lb_2026-08-10.csv`; `runs/lb_ground_truth.md` refreshed.

**Net read of the morning:** all five automated rails green, nothing to fix. Queue was already armed
with the eternal-fallback filler (no append needed — fourth straight day of queue discipline), no
kernel builds in flight (`arc3-duck-compaction-eval` and `arc3-duck-repro` both terminal-COMPLETE),
daemon log shows only `ok:true` submits plus benign `already-submitted-today` skips. **The field gave
us no new information today.** The agenda is unchanged from the 08-09 handoff.

## 1b. Discussions sweep — **NOT DUE**

Every-other-day cadence; last ran 08-09, next runs **08-11**. Standing monitors carried without
re-checking: borro1980's merge solicitation still shows zero uptake as of 08-09; Reki 732854 still
unanswered.

## 1c. Research sweep — `learnings/sweeps/research_2026-08-10.md`

**The window is NOT empty.** Yesterday's forecast was that the weekend gap would break with the
Monday batch, and it did: **`cat:cs.AI` returns 157 submissions for 08-07**, announced this morning
and never swept. Five genuinely new items are decision-relevant; nine more are filed. All 17
candidate IDs were grep-checked against `learnings/` and none had been logged.

Per the discipline rule added yesterday, `totalResults` is recorded for every query in §1 of the
sweep. Query 1 returning **157** is precisely what turned "probably still quiet" into a five-item
day — the rule earned its keep on its first use.

| Item | Disposition | Substance |
|---|---|---|
| **2608.07077 — "Transformers Struggle to Use Their Emergent World Models"** | **ADOPT (mechanism)** | The item of the week. Probes show **Qwen3.6-27B — our exact backbone** — encodes a faithful, causally-involved world model **near-perfectly at the end of the prompt**, then **fails the majority of tasks beyond 3 rings** because the representation **decays during planning**. Causality established by **re-injecting the prompt-time representation**, which partially recovers performance. *"Models build a world model, and then lose it."* This is lane (a)'s thesis stated as a measured mechanism, on our substrate: **the deficit externalisation targets is maintenance, not construction.** First same-backbone datum lane (a) has ever had. **Three discounts, all load-bearing:** Tower of Hanoi is fully observed and is not ARC-AGI-3; the intervention is in **activation space**, which our frozen-fork/vLLM rail cannot reach, so **text-space externalisation is the assumed analogue, not the tested one**; and "partially recoverable" carries no number in the abstract. |
| **2608.07169 — Agent Memory Distillation (AMD)** | **ADAPT (strong)** | **Training-free**, students at **4B–8B**, GPT-5-mini teacher: **+27.2pp AppWorld / +11.2pp BFCL V3 / +3.4pp ToolSandbox**. Three memory tiers — Workflow (proactive), **Subtask (proactive, and the ablation says it carries the largest gains)**, Function (**retrieved reactively on tool-calling errors**). Two concrete consequences: R24 §5.6's P3-as-**two**-timescale may be **the wrong granularity**, and the reactive tier is a **measured** L4 consult gate where we previously had only RPS's design-grounds prior. Feasibility: AMD needs a large teacher, which we cannot call inside a kernel — but **workstation-LLM authoring is ratified in-bounds**, so an offline-authored teacher memory shipped as a static dataset artifact is exactly this shape at $0. Discounts: tool-use not games; effect range +27.2 → +3.4pp is highly domain-dependent; largest gains are at 4B, below our 27B. |
| **2608.07429 — TEPA** | **ADAPT (schema)** | Revocable **keyed precedents**: revoke on contradicting evidence, preserve revoked history for audit, allow later re-promotion. Measured: under full reversal, **append-only (0.210) and last-write-wins (0.210) both fall BELOW no-memory-at-all (0.309)**, TEPA 0.950 — reproduced under real file-backed drift. Our P3 is an accumulating store in a setting where hypotheses are falsified mid-game: **that is exactly this failure mode**, and it would surface as P3 failing K3′ for reasons unrelated to the mechanism. Also gives the refuted-hypothesis content (dropped as a micro-arm at R24) a principled home — **revoke-with-audit inside the store**, not a digest injected into the window, which is the A22 design that failed. Independently concordant with MERIT's dual-polarity, with a much larger effect. Discounts: fact-consolidation benchmark, **no backbone or parameter count stated**, and on the *clean* benchmark TEPA only **matches** last-write-wins ⇒ **the entire gain is a drift/reversal effect**. |
| **2608.06984 — HarnessSafe** | **ADAPT (risk control)** | 328 executable cases across **seven persistent-carrier families**; **containment is carrier-specific and depends jointly on harness AND model backend**, and end-to-end attack-success rate **cannot reveal at which stage a chain was stopped**. Our P1 persistent namespace is a persistent carrier in this taxonomy — this is the first published taxonomy the ratified sandbox risk-class trigger could be specified against rather than invented. Methodological echo worth stating: *an aggregate that conceals which leg fired* is structurally the defect we found in K3 this week. |
| **2608.07440 — "Blast Radius"** | **ADAPT (design pointer only) — heavily de-rated** | **First compaction-theory item in four sweeps**, landing the day after A22's death record was vacated. Reversible eviction with a **byte-exact verbatim archive** plus recurring-transcript burial; **17–26% token reduction** across seven OpenAI models. **The number the authors bury: of 450 evictions, ZERO were ever recalled** — the reversibility channel, the paper's whole novelty, was never exercised, so this is a **compression** result, not a **reversibility** result. Provenance: closed models only, no named benchmark, self-coined framework, no stated affiliation ⇒ **`[SR-adjacent]` under the standing de-rating rule**, with 2608.06196's measured **up-to-44pp** self-authored-benchmark inflation as the governing prior. **Official-LB counterweight, stated separately as the rule requires: the only externally-adjudicated ARC-AGI-3 figure remains arcprize.org's Claude Opus 5 at 30.2%, dated 2026-07-24.** Filed for two narrow reasons: it **withdraws the external half** of R24 §5.2's "compaction field is quiet ⇒ closure unchallenged" premise (the internal half was vacated yesterday), and it **names the affordance A22's design lacked** — archive **plus a recall channel**, rather than a lossy digest. |
| 2608.06880 SkillAligner · 2608.06909 trajectory attribution · 2608.06968 state encodings · 2608.07346 A²E | **MONITOR** | Respectively: training-free execution-time skill adaptation that reduces skill-induced regressions (the VaG mitigation) but publishes **no numbers**; leave-one-out **component attribution** — our K3′ problem in a safety domain, no transferable estimator; *"the schema is the intervention"* but multi-agent, behind the P8 exclusion; harness auditing, possibly relevant to §5.3 instrumentation, no numbers. |
| 22 further items (07110 Modular TTT, 07068 MemOPD, 07107 MemWM, 07408, 07420/07409 UniJEPA, 06706, 07449/07056/06891, 06862, 07437, 07023, …) | **IGNORE** | Training/RL-gated, off-domain, inference infrastructure, or already-sealed negatives. Two flags: **test-time learning is NOT empty this window** — it is architectural (fast-weight TTT, 100B-token pretraining) rather than agentic, which is a different claim from "quiet"; and the skill-library cluster added **3 more papers in one day**, so that field must not be declared empty a fourth time. |
| **ARC-AGI-3 field** | **NOTHING NEW** | Newest ARC-AGI-3 paper remains **2608.04066** (08-04). **Tycho 2607.28287: no v2, no replication, no citing work**; repo unchanged; only the 07-31 press. Official board unchanged at Opus 5 = 30.2%. |
| **Sandbox-persistence efficacy** | **NOTHING NEW** | Two persistence papers this batch and **both are safety papers**. Nobody published an efficacy result for persistent sandboxes. Lane (a)'s efficacy hole is exactly where it was. |

**Net effect on the plan: lane ranking unchanged, (a) > (c) > (b).** But two evidential shifts are
real and should be stated at R25 rather than discovered there. **(a)** now has its first
same-backbone mechanism datum — the R24 §3(a) weak-model objection is not *closed*, but it is no
longer *unaddressed*. **(b)/P3** gets a specification rather than a promotion: AMD and TEPA between
them say the P3 in R24 §5.6 is the wrong shape in two identifiable ways — wrong granularity, and
missing a validity lifecycle — and both corrections are free and text-only. **Nothing found today
bears on ρ̂, on K3′'s calibration, or on the m ≥ 3 baseline cost.** Declared coverage gaps: queries 4
and 5 were paged not exhausted; the arcprize.org counterweight is search-confirmed but **~2.5 weeks
stale**; no OpenReview sweep.

## 2. Open questions for today

Framed against the actual spine — the K3′ fallout, not lane building. Sources:
`learnings/war_room/r24_minutes_2026-08-09.md` §4–§5, `learnings/panel/round25/_directives.md`.

1. **A22 disposition — re-screen under K3′, or leave formally open-and-unworked?** This is the day's
   first decision and it must be **sealed in writing either way**. The death record was **VACATED**
   yesterday, not upheld: `war_eval_v1/v2/v3` are three runs of an **identical** config scoring
   22 / 15 / 13 lc, and **v3 − v1 = mean Δlc −0.360, worst −2 — bit-for-bit A22 v2.1's headline
   "harm", with no compaction in either run**. Re-baselined on the 3-run mean, all three A22 arms
   **PASS**. So A22 is **UNRESOLVED, not dead** — and the successor lane's premise that *"eviction is
   intrinsically harmful"* is **no longer evidenced by our own data**. Two live sub-questions: does
   the campaign owe A22 a real screen, or is "open and unworked" an honest disposition given lane (a)
   holds the budget on independent grounds? And if it is ever re-screened, the sweep says the variant
   worth screening is **eviction with a byte-exact archive and a recall channel** (2608.07440), not a
   re-run of v2.1's lossy digest.
2. **What does K3′ cost, and can we afford it?** K3′ requires pairing against the per-game mean of
   **m ≥ 3 same-config baseline runs** before an arm can be screened *at all*. Offline analysis is
   free; **generating baselines may consume build-rail runs**, and the real budget is
   **≈12–13 builds/week** (30 GPU-h ÷ 2.2–2.4 h), *below* the nominal 2 pushes/day. **Price this
   before committing.** Related and unresolved: a **warpack-specific null is owed** before any
   warpack-family screen — `null10` is a vanilla null and understates warpack variance by **4.83×**
   (p=0.038).
3. **K3′ is itself not yet clean.** R25 methodology N1 [FATAL]: on `null10`, **K3′ is LOOSER at m=3
   than its own m=1 fallback** — a type-II miscalibration in the replacement gate we sealed
   yesterday. Recalibration is free and must precede any use. Also open from R24 §5: is a
   worst-game leg salvageable at 25 games at all, or is mean/quantile the only honest statistic?
4. **The free instrumentation owed before ANY P1 screen** (R24 §5.3) — none of it is built. Latency
   + matched-action-prefix endpoint (because **wall clock, not actions, is the binding resource**:
   all 50 game-runs in two pulls ended at ~7,920 s, so Δlc **cannot separate "harmful" from
   "slower"**); `namespace_reuse_rate` **defined and validated on baseline transcripts before its
   floor is sealed** — which R25 N1/N2 escalated, since that undefined statistic is the gate that
   **concedes the campaign**; prompt/tool-schema strings promoted to declared patch surface; §6.1
   restated as a drop-*policy* invariant; the `SAFE_MODULES` gap (`dataclasses`/`typing`/`enum`)
   resolved, without which the Tycho `State` dataclass is **not constructible in our sandbox**.
   Systems adds a live hazard: the namespace-destroying event is the per-call **timeout**, so
   **K4 can fire on infrastructure** rather than on mechanism.
5. **S1 re-scope** (R24 §5.2) — seal before firing, do not fire before sealing. Six items: which
   **coverage channel** the gate reads (strict coverage is degenerate at **1.0** — *0 of 25 sims
   implement abstention*); a **numeric** carrier definition (the proposal says "~4", `r16` says
   **3**); restriction to the **12 games that actually have replay streams**, not 24/25; explicit
   separate attribution of the `ewm_replay_dryrun.py` module-state bug-fix effect on g50t/re86/tr87
   (a **bug-fix effect, not a protocol effect**); engine-version drift controlled or declared; and
   correction of the sentence claiming **"91.7% held-out"**, which was `split=all` and **never held
   out**.
6. **Sequencing: P1 before P3, or P3 before P1 — still open, and today's sweep pushes on it.** P1's
   "cheap decisive falsifier" advantage evaporated under R24 §3.4 (K4 can pass validly but cannot
   fail validly). Meanwhile P3 gained two free design corrections this morning and a second
   weak-model datum. Counterweight: 2608.07077 is the strongest same-backbone evidence lane (a) has
   ever had, and it is a *maintenance* argument, which is P1's territory. Note the standing
   correction: **§3(b) is factually wrong** — `cross_level_notes` is deliberately *not* wiped, so P3
   **reverses a deliberate design choice** and must argue against it.
7. **Is L0 rescuable at all**, or does the abstention gap mean the exec-wm line cannot be re-verified
   without building L1 first — and is L1 affordable now that workstation authoring is in-bounds?
8. **Standing rail items.** The panel rail was fixed and upgraded to Opus 5 yesterday (agent_sdk
   nesting resolved; `--panel-model` added) — it should be exercised once before 08-16 rather than
   trusted. R25's other unfixed fatals stay open: the **in-scope port subset excludes the components
   carrying Tycho's measured lift** (defanged test), and the **free-vs-scored rail regime gap is
   unquantified**.

**No pushes owed today. $0 spend. Keep the queue armed with the frozen fork.** Nothing in the sweep
or the draw changes the submission.
