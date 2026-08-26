# DAILY BRIEF — 2026-08-16 (Sunday: full panel + weekly)

Sources merged: morning-check entry in `ITERATION_LOG.md` (§`### 2026-08-16`), `learnings/sweeps/sweep_2026-08-16.md`,
`learnings/war_room/q38_engine_swap_prereg_2026-08-15.md` §12, `learnings/war_room/lora_serve_canary_postmortem_2026-08-16.md`,
`runs/ledger.json`, `runs/failure_fingerprints.json`, `Dreams/2026-08-16-122943.md`.

**Promotion bar today: 1.0826** — re-read from `runs/ledger.json` (n=33, mean 0.9424, s 0.1563). It drifts; never cache it.

---

## 1a. RESULT DEEP-DIVE — the number, and what it is not

**Overnight draw 1.17** (`canivel/arc3-duck-repro` v3, 2026-08-16T00:07:11Z, COMPLETE). **z = +1.53** against the prior n=32
record — the highest draw since 08-05's 1.21.

**Pre-registered expectation: met, and it means nothing.** This was the **fifth consecutive** AUTO-REFILL filler day: a
**byte-frozen fork**, unchanged between 08-15 and 08-16. A high draw off an unchanged artifact is a **measurement of
variance, not of progress** — and the record already says so, because our **public max is still 1.33, set 07-18 and
untouched for 29 days**. The correct reading of 1.17 is: *the ledger's dispersion is real and large* (s rose 0.1533 →
0.1563), which is exactly why the mean-of-4 promotion bar exists and why single draws are never a verdict.

**The delta that actually matters is the one we did not move.** Rank **#119 → #130 of 2345 on a byte-unchanged 1.33** —
eleven ranks in one day, thirty in two. Prize line **1.90 → 1.98**; gold **1.62 → 1.65**. **Both gaps widened for a second
consecutive day** (to gold 0.32, to prize 0.65). We are not being overtaken by a better draw of our own distribution; we
are stationary while the board moves.

**Per-mechanism evidence — the control arm is the load-bearing observation.** `Jack Cole (MindsAI)` **1.59 flat** (+1 sub,
Δ/draw 0.0000) and `Tufa Labs` **1.62 flat** (+1 sub, Δ/draw 0.0000). The two teams who wrote the TTT literature and the
harness we fork each **spent a draw and gained exactly nothing**. The 1.55–1.65 band agrees: median **+0.01** while 3
teams entered and 0 left — movement *into* the band, not a lift *of* it. **This is the opposite of what a shared
commodity-engine story predicts.** Caveat held: this measures **scores**, not methods; it does not say they didn't swap an
engine, only that nothing they ran beat their own best. Evidence class for what either ran: **UNKNOWN**.

Meanwhile real motion is **concentrated and per-draw large** — `Fufront-RyanX-AGI-Team` **+0.85 on ONE draw → 2.25** (#3),
plus three more at 4–6× the median gainer's 0.165/draw — against **78% of the 218 teams who submitted gaining nothing at
all**. Signature of specific teams doing specific work. **0 DISCLOSED methods.** And `LastSubmissionDate` is LATEST while
`Score` is BEST, so this instrument **cannot date a scoring run** (one narrow exception below).

### Today's own results (developed this session)

- **★ Q38 v2 PUSHED, VERIFIED, RUNNING** — `canivel/arc3-q38-engine-eval` v2, 08-16 **slot 1 of 2**. All three §11.6 steps
  performed deliberately. Artifact byte-matches the sealed v2 fingerprint: `code_sha256=8babf6de9934c3e5`, 17 cells, diff
  cells `[2,6,8]`, **smoke 109/0**, **scorer 22/0**. Preflight **ALLOW**, 0 fail / 0 warn. **3/3 dataset_sources survived
  including the 25 GB engine** `saltb0x/qwen3-8-27b-fp8`. Read remains sealed: **CONFIRM-2× ≥ 32 levels · REFUTE-2× ≤ 25 ·
  HARM ≤ 12 · INFRA DEATH.** *Status note: still RUNNING ~11 min after push. v1 died at t=425 s of KERNEL time; wall-clock
  since push includes queue time, so this is encouraging but is **not yet** proof the probe gauntlet was cleared.*
- **★ The instrument was the defect again — twice in one push.** Step 3 aborted on `CODE MISMATCH`. Root cause: **the
  frozen fork's OWN em-dash** (`U+2014`, baseline cell 16 offset 471) mojibaked by Kaggle's push path. Cell 16 is not one
  of our arm cells, so ASCII-hardening it would have manufactured a 4th differing cell and broken the very byte-identity
  D2/D3/D4 protect. `preflight.py` D4 had already been hardened for this class; step 3 had not. Second defect: an assert
  demanding the incumbent name be absent, written before v2 added **`Q38_VETO`** — the poisoning gate whose whole job is to
  name it. **The check was demanding the deletion of the gate that protects the measurement.** Both fixed in the verifier,
  not the artifact; fix 2 is strictly *stronger* than what it replaced. **Generalisable lesson: a gate suite that aborts on
  its cheapest check never runs its most load-bearing one** — a cosmetic em-dash suppressed the 25 GB-engine attachment
  check, the single likeliest way to void the arm.
- **★ LoRA canary ERROR — OWED item DISCHARGED, diagnosed from the real log.** `kernels logs` on **CLI 2.2.3**, 236,029 B /
  1,506 entries. Died **t = 99.049 s**, `NameError: name '_source_path_entries' is not defined` inside `_lora_install_guard()`
  — the `"$PYTHON" - <<'PYSETUP'` heredoc is a **separate interpreter that cannot see notebook cell-8 names**; the body
  compiled cleanly, which is why a `compile()`-only build check waved it through. Class: **CONFIG/AUTHORING DEFECT, NOT
  DECISIVE**. It licenses **nothing** about the LoRA-serve lane: vLLM never launched, the 35.9 GB brain never loaded,
  `--enable-lora` never exercised, the noop/probe differential never ran. Banked positives: adapters shipped byte-exact
  (r=16, 41,962,184 B each), 4/4 datasets attached, Blackwell allocated, 6-anchor rewrite correct, wheelhouse 82.2 s.
  **Correction to the standing record: cost was 102.5 seconds, not the "one GPU-hour" claimed in `lora_lane_2026-08-13.md`
  §12.4 and `build_lora_serve_canary.py:530` — the slot was lost, the GPU-hour was not.**

---

## 1b. DISCUSSIONS SWEEP — 1 new topic, 7 new comments on 2 threads

| # | Item | Verdict | Reason |
|---|---|---|---|
| 1b-A | topic 735479 "Qwen 3.8 27B", FOYSAL (#22, 1.61) — links `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, verified 30.89 GB Apache-2.0 vLLM-compatible repack | **ADAPT** | Discharges gate 2 of the 08-15 ADOPT — "no FP8 artifact exists on Kaggle" is now false. Supplies an **artifact, not a result**. |
| 1b-B | Scott Le Grand (#47) on 735243 — suspects the Qwen3.8 lift may hit only public/validation-split games; calls for per-game ablation | **ADAPT (risk)** | Only stated *risk* mechanism on the record. Evidence UNKNOWN (hypothesis, no measurement). Converts the swap from free upside to upside with an untested **private-set transfer** assumption — lands squarely on `feedback_arc_generalization_first`, and hands us a free pre-registerable falsifier. |
| — | remaining comments | IGNORE | No method, no number, no plan impact. |

**★ Attribution finding (INFERRED, derived from our own LB archives).** `OverfitOracle` — author of the "Qwen 3.8 release"
thread, who wrote *"we are currently using qwen 3.6 27b an older model"* — is a member of **`aRc (binary relation)`, today's
#6**. That team sat at **1.17 for 18 days**, made **exactly one** post-release submission, and landed **1.91**. This is
datable *only* because ΔSubmissionCount == 1 and Score improved, so the new draw **must** be the new best — a narrow,
provable exception to the standing `LastSubmissionDate` rule, and it does not generalise.

**★★ But it cuts the other way, harder — RE-ANCHOR THE Q38 PRIOR.** Ya Xu, the **sole** source of the "2× on local 25"
claim, moved **1.30 → 1.47 (+0.17)** on his one dateable draw. FOYSAL — who cared enough to repack 30.89 GB — drew
**1.61 → 1.61**. The expected effect should be re-anchored from *"2×"* to **+0.17-class**, which against the bar of
**1.0826** is ordinary and **well inside our own ledger noise** (s = 0.1563; we drew +1.53σ this morning on a frozen fork).
*This does not touch the sealed Q38 read — no constant was moved and none may be — but it is the honest prior going in, and
it materially raises the probability the arm lands REFUTE rather than CONFIRM.*

---

## 1c. RESEARCH SWEEP — 361 arXiv entries screened, 12 abstracts pulled, 9 adjudicated. **0 ADOPT · 4 ADAPT.**

| # | Paper | Verdict | Reason |
|---|---|---|---|
| 1c-1 | **arXiv:2608.12959** — *The Objective Is the Bottleneck: Latent World Models Encode What Their Planners Cannot Use* (Aug 13) | **ADAPT** | The paper we have been waiting for on our open question. Information provably present (ridge probe **R²=0.9922**); predictor is not the limit; failure is entirely in the **consumer's objective** (tracks true distance at r=0.426 then *decreases* — moving away from goal lowers cost). Replacing **only the objective** — nothing retrained, no GPU — lifts long-horizon success **26.0% → 98.0%**, reproduced in the authors' released weights. Substrate differs (they swap an explicit CEM cost; we cannot), so **the transferable content is the diagnostic, not the fix**. |
| 1c-2 | **arXiv:2608.12321** — *LLMs Know the Constraint But Do Not Use It* (submitted 29 May 2026; **zero prior hits in our record**) | **ADAPT (elevated)** | Probes decode the constraint **>88%**, behaviour still doesn't follow; **"no prompted intervention reaches the repair corner — all inflate conservative bias"**; *"routing problem, not a knowledge problem."* **Every intervention we have run is prompt-side.** It makes our 96.3% delivery null the **expected** outcome — and it **predicts raising the 31,744 ceiling will null too**. |
| 1c-3 | **arXiv:2608.13087** — *Sampling Luck Masquerades as Allocation Gain* (Aug 13) | **ADAPT (measurement discipline)** | In-sample oracle allocation reports 2.2–2.6% gain with intervals excluding zero; **out-of-sample the same gain is 0.457 / 0.015 / −0.512% — zero**. Bias does **not** shrink with more samples or instances. Our best-of-N confound, peer-reviewed, and the correct lens on today's board (218 submitted, 48 gained). |
| — | 6 others | IGNORE | No bearing on the plan; no result we could score against 1.0826. |

**★ Instrument defect found in the sweep itself, process change owed.** The **arXiv API search index is one day stale and
fails silently** — it returns nothing after 2026-08-13 in cs.AI/cs.LG under either sort, while `/list/cs.AI/recent` shows a
full 204-entry **Fri 14 Aug** cohort. **A sweep trusting the API would have reported "zero new papers" and missed
2608.12959** — the single most relevant paper of the week. Same failure class as the 07-06 registry freeze.
**Fix adopted: screen `/list/<cat>/recent` HTML; use the API only to resolve IDs and abstracts.**

---

## WEEKLY (Sunday duties)

- **KAOS ingest:** `inserted=36 updated=0 unchanged=155 total_rows=257`.
- **Dream run:** digest `Dreams/2026-08-16-122943.md`. As expected, **recency digest only** — 3 episodes, 2 complete, skills
  library empty, **0 consolidation proposals**, 0 tokens, $0. Nothing for the panel agenda.
- **★ Fingerprint report — the instrument was stale, and it retro-flagged a death it should have prevented.** The weekly
  duty is `fingerprint_report.py --brief`, a **READ**; `fingerprint_backfill.py` is the **WRITE**, and the protocol never
  invokes it. **The store is only ever filled by hand.** Running the backfill today took it **16 → 19 incidents / 8 → 9
  families with no new kernels** — purely by scanning logs already on disk. The newly surfaced family is
  **`t1:fb1e96c3815797ad`, n=2, both dated 2026-07-25**, material `t1|PYSETUP|CalledProcessError: Command '"$PYTHON" - <<'PYSETUP'`,
  from the A17 72B canary v1/v2 logs — **the same `"$PYTHON" - <<'PYSETUP'` heredoc surface that killed the LoRA canary 20
  days later.** *Stated precisely:* the A17 incidents are `CalledProcessError` (subprocess exited nonzero) while LoRA's
  proximate cause is a `NameError` inside it — **same surface, arguably different proximate bug**, so this does not license
  the claim that the family would have predicted the specific defect. What it **does** establish is that a recurring family
  sat unqueryable in retained logs for 20 days because nothing feeds the store. Fix in flight (backfill-before-report,
  staleness banner, real `--help`/`--dry-run`, regression test).

```
family                         n  first       last
class:ERROR:none               7  2026-05-26  2026-06-28
provenance:scratch-built       5  2026-05-26  2026-06-28
slug:canivel/arc3-final        4  2026-05-26  2026-06-10
class:COMPLETE:0.00            3  2026-03-29  2026-06-10
slug:canivel/arc3-forge35      3  2026-04-24  2026-06-22
slug:canivel/arc3-pilot-eval   3  2026-07-07  2026-07-08
t1:07d0f5248c48401d            3  2026-07-07  2026-07-08
class:COMPLETE:null-band       2  2026-06-01  2026-06-08
t1:fb1e96c3815797ad            2  2026-07-25  2026-07-25   <-- NEW, and see above
```

---

## OPEN QUESTIONS (for today's panel)

1. **★ The open question has a paper now. Does it change the arm we run next?** 2608.12959 and 2608.12321 converge on the
   same claim from different substrates: **the information arrives and the consumer's selection criterion does not use it.**
   Our own mech-C measured 96.3% delivery with no behaviour change — which those papers make the *predicted* result rather
   than a null. **Proposed reframing: retire "did transitions arrive?" (settled, 96.3%) and ask "is the agent's
   action-selection criterion monotone in what scores?"** — answerable **CPU-only on trajectories already on disk**, zero
   slots, zero spend. Panel: adopt as the next build-rail item or not?
2. **★ 2608.12321 predicts the context-ceiling fix will null.** If "routing, not knowledge" holds, raising the 31,744
   ceiling buys nothing. **Pre-register that prediction before any spend on context budget** — it is free to state now and
   expensive to learn later.
3. **Q38 prior re-anchored to +0.17-class (from "2×") on the only two dateable data points.** The read stays sealed and no
   constant may move. Question for the panel is the *disposition*: if v2 lands REFUTE at a +0.17-class true effect, is that
   a refutation of the engine claim or of our power to detect it? **Decide the answer now, before the data.**
4. **Le Grand's public/private split risk (1b-B) is a free pre-registerable falsifier.** Should the Q38 read carry a
   per-game secondary read so a public-split-only lift is visible rather than inferred?
5. **Five consecutive filler days.** The queue has never been empty and cadence is 44 days — but nothing we have developed
   since 07-18 has been submittable. What is the *next artifact that could actually clear 1.0826*, and is anything on the
   rail aimed at it?
6. **Three instrument defects found today** (step-3 verifier ×2, arXiv API silent staleness, fingerprint store never fed).
   `feedback_audit_the_instrument` is now the highest-frequency failure family in this campaign. **Is a standing
   "instrument audit" a rail item rather than an incidental discovery each day?**

---

## APPENDIX (added after the panel was dispatched — feasibility check on OPEN QUESTION 1)

**The CPU-only diagnostic proposed in OQ-1 is not a hope; the data is on disk and I verified it end to end today.**

- `runs/a22_v2_seed1/intermediate_states.pkl` (38.8 MB) unpickles to **25 games × ~630 steps** of `taaf.game.GameState`.
  Each step exposes **`previous_action`**, **`levels_completed`** (the score signal — the competition scores levels),
  `just_won_level`, `won`, `frame`, `available_actions`, `game_over`.
- `runs/a22_v2_seed1/benchmark.json` independently carries `game_runs[].history` — **629 steps for game 0**, each with
  `action`, `generated_tokens`, `uncached_input_tokens`, `wallclock_seconds`.

So for every step we hold **the action chosen, the score at that step, and the observation** — precisely the three
quantities 2608.12959's diagnostic needs to ask *"is the action-selection criterion monotone in what scores?"*

**Two frictions, both trivial, both now documented so nobody rediscovers them:** the pickle needs (i) the local taaf
package on the path — `PYTHONPATH=duck_eval\taaf_bundle\src\tufa-arc-agi-framework\src` (note: **not** the `src/` dir
above it, which is the wrong level and fails with `No module named 'taaf'`), and (ii) **`imageio`**, absent from the
project env; `uv run --with imageio` supplies it ephemerally without mutating `pyproject.toml`.

**Cost to run: 0 GPU-hours, 0 kernel slots, 0 dollars, 0 submissions.** Whatever the panel rules on the *strategy*, the
*feasibility* question is closed: this is a keyboard-and-CPU task against data we already paid for.
