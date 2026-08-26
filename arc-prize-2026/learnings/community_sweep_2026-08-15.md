# Community sweep — 2026-08-15 (STEP 1b)

Read-only on Kaggle. One write: the leaderboard fetch. No pushes, no submissions, no spend.
Every number is tagged **VERIFIED** (pulled live this session) or **INFERRED** (my derivation).

Artifacts: `runs/lb_daily/lb_2026-08-15.csv` (top-20, archived, format matches prior days),
`runs/lb_ground_truth.md` (refreshed).

---

## §0 — Headline

1. **The board broke open.** After six flat days the **gold/top-13 line moved 1.58 → 1.62** and the
   **top-5 prize line moved 1.64 → 1.90 (+0.26 in one day)**. Both streaks are over.
2. **Five teams entered or jumped above the old prize line inside 12 hours**, and **every one of them
   has a last-submission timestamp AFTER 2026-08-14 15:00 UTC**.
3. **That timestamp is the Qwen3.8-27B open-weights release** (VERIFIED: Alibaba released
   Qwen3.8-27B at 15:00 UTC on 2026-08-14, Apache 2.0).
4. **Our frozen fork's engine is Qwen3.6-27B.** VERIFIED from
   `notebooks/duckfork/kernel-metadata.json`: `dataset_sources` contains
   `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`. The community's claim is a **drop-in engine swap
   of exactly that artifact**, not a code change.
5. **cstl is NOT explained by this.** cstl banked 2.70 on **08-12 20:02**, ~43 h *before* the release,
   and has been flat for three days. Flag (a) does **not** trip. cstl remains untraced.
6. **#2 is Daniel Franzen (`dfranzen`) at 2.58** — VERIFIED Kaggle profile: Deep Learning Researcher,
   University of Mainz; mutual-follow with **Jan Disselhoff**. That pair is *the ARChitects*, the
   **ARC Prize 2024 Kaggle grand-prize winners**. This is the most credentialed ARC entrant on the
   board and he is now second, with 41 submissions.
7. **Our rank is #119 of 2331** (VERIFIED, full LB download). Yesterday #100. **−19 ranks in one day
   on a byte-unchanged 1.33.** 114 teams strictly above; the 1.33 tie block shrank 9 → 6.
8. Discussions: **2 new topics + 1 new Kaggle-staff comment. 1 ADOPT (conditional), 1 ADAPT, 1 IGNORE.**
   The ADOPT is the first non-zero verdict of the campaign's recent sweeps and it is *not* a technique
   — it is an engine version number.

---

## §1 — Leaderboard (VERIFIED, `kaggle==2.0.0`, pulled 2026-08-15 ~14:17Z)

### 1.1 The three lines

| Line | 08-14 | **08-15** | Move | Holder |
|---|---|---|---|---|
| **Top-1** | 2.70 | **2.70** | **FLAT — 3rd day** | cstl (last sub 08-13 20:08) |
| **Top-5 (prize)** | 1.64 | **1.90** | **+0.26 — streak broken** | #5 AbeLincoln1865 1.90 |
| **Top-13 (gold)** | 1.58 | **1.62** | **+0.04 — 6-day flat streak broken** | #13 "I forgot the name.." 1.62 |
| **Us** | 1.33 (#100) | **1.33 (#119)** | **−19 ranks** | Canivel, 1.33 |

Gaps: **to gold 0.25 → 0.29**; **to the prize line 0.31 → 0.57**. Both widened. The *gaps* have been
the comfortable number all campaign; today even the comfortable number got worse.

Robustness note on the gold line: total teams **2331** (VERIFIED). #13 and #14 are both **1.62**, so
the gold cutoff is 1.62 regardless of whether the medal boundary sits at 13 or 14. As team count
grows the boundary index drifts — like the promotion bar, it should be re-derived, not cached.

### 1.2 Top-20 today, with the release marker

Release line = 2026-08-14 15:00 UTC. `*` = last submission after it.

| # | Team | Score | Last sub (UTC) | |
|---|---|---|---|---|
| 1 | cstl | 2.70 | 08-13 20:08 | — pre-release |
| 2 | **Daniel Franzen** | **2.58** | 08-14 21:37 | `*` |
| 3 | **Nikita Sorokin** | **2.10** | 08-14 19:30 | `*` |
| 4 | **Yusaku Muroya** | **1.98** | 08-15 02:36 | `*` |
| 5 | **AbeLincoln1865** | **1.90** | 08-15 00:22 | `*` |
| 6 | YUTO KOJIMA | 1.86 | 08-15 00:00 | `*` |
| 7 | **MLRush** | **1.75** | 08-15 00:01 | `*` |
| 8 | Andy liu | 1.69 | 08-03 12:09 | — pre-release |
| 9 | Lord Han Solo | 1.65 | 08-15 03:52 | `*` |
| 10 | BambooCopter Analytics | 1.64 | 08-15 02:15 | `*` |
| 11 | GeniusYY | 1.64 | 08-15 00:03 | `*` |
| 12 | Tara Labs | 1.63 | 08-15 03:23 | `*` |
| 13 | I forgot the name.. | 1.62 | 08-14 05:44 | — |
| 14 | Marseilles Bogano | 1.62 | 08-15 02:27 | `*` |
| 15 | Tufa Labs | 1.62 | 08-14 20:19 | `*` |
| 16 | Tecnod8.AI | 1.61 | 08-15 03:23 | `*` |
| 17 | FOYSAL | 1.61 | 08-14 15:28 | `*` |
| 18 | Van-Phuc Huynh (was "hvp") | 1.61 | 08-15 04:56 | `*` |
| 19 | Helmut AGI | 1.61 | 08-15 05:03 | `*` |
| 20 | DhanaLakshmiMalla | 1.60 | 08-14 18:53 | `*` |

**Bold = new to the top-20 vs 08-14.** Five new names, four of them above the old prize line.
**29 of the top-40 have post-release submissions** (VERIFIED). Threshold counts today: **3 teams ≥2.0,
5 ≥1.90, 7 ≥1.75, 11 ≥1.64, 15 ≥1.62, 26 ≥1.58, 41 ≥1.50, 120 ≥1.33.**

**Honest confound:** the top of this board resubmits daily anyway, so "post-release timestamp" is not
by itself evidence. What is *not* routine is the size and the simultaneity: three scores above 2.0
where there had been one for four days, and the first movement in the gold line in six days, both
landing in the same 12-hour window.

### 1.3 Our position (VERIFIED, full 2331-row leaderboard download)

- 114 teams strictly above 1.33; tie block at 1.33 spans **#115–#120** (6 teams); we are 5th in it → **#119**.
- Immediately above: Peter / "put me in harness" / "blatant warrior" at 1.34 (#112–114).
- 08-14 → 08-15: teams above us **94 → 114**. Twenty teams passed a score we did not change.

---

## §2 — Discussions

Route: **chrome-devtools MCP worked** (public read, signed out, no OAuth attempted). Sorted by
`?sort=published`. Cross-checked against `learnings/discsweep_2026-08-14.md`, which closed at topic
**735147**. Everything with a higher id is new.

### 2.1 New topics (2)

| # | Post | Verdict | Reason |
|---|---|---|---|
| A | **735243** · "Qwen 3.8 release" · OverfitOracle (307th) · posted 08-14 ~17:00Z · 4 comments, 3 votes | **ADOPT — conditional on a serving-compat gate (§2.3)** | Mechanism, not vibe: our banked artifact's engine is `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`, i.e. **Qwen3.6-27B**. A strictly stronger same-size dense model shipped 24 h ago. **Ya Xu (148th): "already achieving a consistent 2x score on the local 25 dataset … Qwen-3.8 27B 8Bit is significantly better"** than both Qwen-3.6-27B-8bit and DeepSeek-V4-Flash. Independently corroborated by the board: 5 teams jumped past the old prize line, all post-release. This is a **`dataset_sources`/`model_sources` swap with zero solver-code change** — the one intervention shape that `feedback_simplicity_wins` *endorses* rather than punishes. |
| B | **735381** · "Too many High Score" · FOYSAL (17th) · posted 08-15 ~12:00Z · 1 comment, 0 votes | **IGNORE** | Content-free ("What is happening today?"). Its only value is as an independent witness that a team *inside the gold band* also noticed the jump and had no explanation. The single reply just points back at 735243. No technique, no artifact, no number. |

### 2.2 New comments on existing threads (1 that matters)

| # | Comment | Verdict | Reason |
|---|---|---|---|
| C | **735147** · **María Cruz (KAGGLE STAFF)**, 18 h ago: *"we are actively investigating potential capacity constraints for the RTX 6000 pool and are working to free up more resources soon."* | **ADAPT (schedule only) — upgrade of yesterday's ADAPT** | Yesterday we ADAPTed on three *user* reports of 28 min / 3 h / 8 h queues. This is **host confirmation** that the constraint is real and unresolved. Any build we push today must budget multi-hour queue latency, and a long `Queued` state must not be read as a build defect. Nothing here changes the agent. |

### 2.3 The ADOPT, stated as a gate rather than a push

I am ruling ADOPT because the mechanism argument is unusually clean, **not** because the evidence is
strong. It is one forum claim from a rank-148 competitor with no artifact, no seeds and no numbers,
plus a temporal coincidence on the LB. That justifies **spending a screen**, not a submission.

**Facts established this session (all VERIFIED):**
- Qwen3.8-27B: released **2026-08-14 15:00 UTC**, Apache 2.0, 27.78B params, 262K native context.
- Architecture is **`Qwen3_5ForConditionalGeneration`** — it **reuses the existing `qwen3_5`
  implementation**, no new architecture registered. Hidden layout
  `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`, i.e. the same hybrid
  linear-attention family our Qwen3.6-27B already is. **This is the single biggest reason the swap is
  plausible on a frozen offline wheelhouse.**
- `config.json` carries a **`vision_config`** — 3.8-27B is a **native VLM**; our 3.6 snapshot is
  served as a text model. **This is the #1 failure risk.**
- Kaggle availability as of this sweep: **no FP8 artifact exists.** Two community uploads, both
  created 08-14: **`trailblazeranemo/qwen3-8-27b`** (id 741119, transformers/bf16, Apache 2.0,
  fine-tunable: No) and **`overseer66/qwen3-8-27b-nvfp4`** (id 741328). **Zero Kaggle *datasets*.**
  No official `qwen-lm` upload.
- Size: bf16 27B ≈ 54 GB weights (fits 96 GB RTX PRO 6000, but eats the KV budget);
  NVFP4 ≈ 24.6 GiB per the vLLM recipe.

**Pre-registered blockers, in the order they will kill it:**
1. **VLM path.** Does the pinned vLLM in `driessmit1/arc3-vllm-h100-wheelhouse-v3` serve
   `Qwen3_5ForConditionalGeneration`, or only the causal-LM path? Frozen offline wheelhouse, no
   internet, no upgrade available. Cheapest possible check, do it first.
2. **Quantization.** No FP8 artifact ⇒ either bf16 (KV-cache shrinks, throughput drops — and
   **action-count efficiency is our binding constraint**, so a slower engine can *lose* score even if
   it reasons better) or NVFP4. NVFP4 on this stack **killed the b122 lane two days ago**
   (FlashInfer SM120 JIT). Caveat in our favour: that failure was **MoE**-specific and 27B is dense.
3. **Attach fragility.** `feedback_kaggle_model_attach`: Kaggle silently drops unattachable
   `model_sources` on push. Pull-back-verify is mandatory, and note our fork currently gets its engine
   from `dataset_sources`, not `model_sources` — that path changes.
4. **Queue.** Host-confirmed RTX 6000 capacity constraint (§2.2). 3–8 h queues observed.
5. **Provenance.** Both Kaggle artifacts are anonymous community re-uploads made within hours of
   release. Neither is `qwen-lm`.

**Cost check against `feedback_arc_zero_budget`:** a Kaggle kernel BUILD is free. It spends a push
slot and GPU-hours from the 30 h/wk quota, not money. Compliant.

**What I am NOT claiming:** that Qwen3.8 explains cstl (it cannot — see §3), that 2x local
translates to 2x LB, or that this closes a 0.29 gap. The honest read is that the *band we chase*
just got a new engine and our banked artifact is now running a superseded one.

### 2.4 Base-rate note

Verdict tally **1 ADOPT (conditional) / 1 ADAPT / 1 IGNORE**. The campaign's honest base rate is
0-ADOPT and I want the deviation on the record: I adopted an **engine version bump to an artifact we
already run**, which is the narrowest possible class, and I gated it behind a compat screen rather
than a submission. If gate 1 fails, this reverts to IGNORE with no further spend.

---

## §3 — The three flag categories

**(a) Anyone publicly disclosing a method consistent with the cstl jump — NO.**
The Qwen3.8 hypothesis is *timing-refuted* for cstl and the forum reached the same conclusion
unprompted: **Ravindra (82nd): "Cstl had already scored 2.70 before Qwen3.8-27B was released."**
VERIFIED from our own archives — cstl 1.59 → 2.52 on 08-11 18:25 → 2.70 on 08-12 20:02, all before
08-14 15:00Z. cstl stays **traced to WHO, untraced to WHAT**, and stays not-a-target.
The release *does* plausibly explain the **rest** of the 08-15 wave, which is a different question.

**(b) Host/organizer post changing rules, deadlines or scoring — NO.**
The only new staff activity is María Cruz's RTX-6000 capacity comment (infrastructure, §2.2). The
pinned "Clarification on deadline for milestone prizes" is 5 days old and unchanged. Nothing touches
rules, deadlines or the scoring function.

**(c) Discussion of private-LB / final-selection mechanics — NO.** Nothing, in any thread, new or old.

**Therefore: zero of the three plan-forcing categories tripped.** The thing that *should* force a
plan change today is not on that list — it is a **base-model release our own frozen fork can consume
as a source swap**. I flag it explicitly so it is not filed under "interesting."

---

## §4 — Ledger cross-check (VERIFIED, `runs/ledger.json`, already recomputed today)

Draw 2026-08-15 00:07Z = **0.89** (API COMPLETE, frozen-fork filler `canivel/arc3-duck-repro`,
AUTO-REFILL — third consecutive day where the only thing on the board was the eternal fallback).
`runs/ledger.json` is **fresh** today (unlike 08-14): **n=32, mean 0.9353, s 0.1533, z(0.89) = −0.30**,
trailing-4 **0.91 → 0.86**, max 1.33, min 0.65, **promotion bar (mean-of-4) = 1.0731** (down from
1.0821 at n=30 — the bar drifts; re-read it from `runs/ledger.json` at prereg time, never cache it).
Interior draw, well above the retired 0.80 leg; the binding **paired harm-pause** (trailing-4 −1.5s
⇒ −0.230) sees a realized move of **−0.05**. Record stays **resolved-STATIONARY**.
Public max **UNCHANGED at 1.33**.

---

## §5 — What I hand to the day session

1. **The single cheapest high-value action available today is gate 1 of §2.3**: determine whether the
   frozen wheelhouse can serve `Qwen3_5ForConditionalGeneration`. That is a config/registry question,
   answerable before spending a GPU-hour.
2. **Re-derive the gold boundary index** — team count is now 2331 and rising; #13 vs #14 happens to be
   score-degenerate today (both 1.62) but will not stay that way.
3. **Start archiving the FULL leaderboard, not the top-20.** `kaggle competitions leaderboard -d`
   returns all 2331 rows with a `Rank` column; the paginated `--show` route silently returns
   **non-contiguous** windows (I confirmed this — 425 rows with a hole straight through our own tie
   block). Our rank history is unreconstructible before today because of this.
4. **Do not read today's gap widening as our regression.** 1.33 is byte-identical to 08-09. The board
   moved; we did not. That is the whole event.
