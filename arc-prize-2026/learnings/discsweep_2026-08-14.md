# Discussions + Leaderboard sweep — 2026-08-14

Read-only sweep. One write to Kaggle: the leaderboard fetch. No pushes, no submissions, no spend.
Every number below is tagged **VERIFIED** (pulled live this session) or **INFERRED** (my derivation).

Artifacts: `runs/lb_daily/lb_2026-08-14.csv` (top-20, archived), `runs/lb_ground_truth.md` (refreshed).

---

## §0 — Headline

1. **cstl is FLAT at 2.70.** No drift today. It resubmitted 08-13 20:08 UTC and did not improve.
2. **The whole top-20 is score-static vs 08-13** — every name, every score identical; only submission
   timestamps moved. No new entrant. Largest score delta anywhere in the top-20 = **0.00**.
3. **Correction to the campaign record: cstl did NOT "enter" at 2.52.** Our own archived CSVs show
   cstl sitting at **1.59 inside the dense duck band from 08-04 through 08-09**. It is a *known
   band team that found a step*, not an outsider arriving with unknown pedigree. This changes the read.
4. **We have a trace for WHO cstl is** (a 2-person team, both accounts named below, one with a serious
   competitive-simulation-agent pedigree) and **no trace whatsoever for WHAT they did**.
5. **Our rank is #100, not "below #49".** The ground-truth doc has carried "#49" unrecomputed since
   08-09. Directly counted today: **94 teams strictly above us.**
6. **Today's draw is 0.70** (VERIFIED, API) — second-lowest in the record, second consecutive sub-0.80,
   and it was an **AUTO-REFILL**: the queue was empty and the daemon armed the frozen fallback itself.
7. Discussions: **3 new posts. 0 ADOPT.** The one item with real content (Jason Feng's Dynamic Value
   Model) is a self-narrating prompt with an n=1, author-acknowledged-confounded eval.

---

## §1 — Leaderboard (VERIFIED, `kaggle==2.0.0 competitions leaderboard`, pulled 2026-08-14)

### 1.1 The three lines

| Line | 08-13 | **08-14** | Move | Holder |
|---|---|---|---|---|
| **Top-1** | 2.70 | **2.70** | **FLAT** | cstl (resubmitted 08-13 20:08, no gain) |
| **Top-5 (prize)** | 1.64 | **1.64** | **FLAT — 2nd day** | #5 BambooCopter Analytics 1.64 (tiebreak over GeniusYY 1.64) |
| **Top-13 (gold)** | 1.58 | **1.58** | **FLAT — 6th day** | #13 Biubiu 1.58 |

Full top-13 unchanged in composition and order: cstl 2.70 · KOJIMA 1.86 · Andy liu 1.69 · Lord Han
Solo 1.65 · BambooCopter 1.64 · GeniusYY 1.64 · Tufa Labs 1.62 · Tecnod8.AI / FOYSAL / hvp / Helmut
AGI 1.61 · DhanaLakshmiMalla 1.60 · Biubiu 1.58. Then ippeiogawa / Nkosi Ndwandwe / Yuchen20 1.58,
kglctf 1.57, Mathurin Ache / anngle / NoOneAhead 1.56.

**Flags: none.** No new entrant in the top-20. No jump > 0.2 anywhere — no jump at all.

### 1.2 Our standing (VERIFIED, direct count over the rank-ordered top-200)

- Banked public best **1.33**, intact.
- **Rank #100.** 94 teams strictly above; **9 teams tied at 1.33 spanning ranks #95–#103**; we are 6th
  within the tie block. Neighbours: `today` 1.33 above us, `aarrcc` 1.33 below; Peter 1.34 at #94.
- **Gap to gold (1.58) = 0.25** — flat vs 08-12 and 08-13.
- **Gap to prize line (1.64) = 0.31** — flat vs 08-13.

**Process defect found: the rank in `runs/lb_ground_truth.md` is stale by ~50 places.** The doc has
said "below #49" every day since 08-09 without a recompute; the last real count was **#63 on 08-01**.
Today's direct count is **#100**. That is **−37 ranks in 13 days on an unchanged banked score** — pure
competitive drift, and it is the cleanest available measure of how fast the field is filling in
underneath us. The gold/prize *gaps* being flat is the misleadingly comfortable number; the *rank*
is the honest one. Recommend the daily loop count rank from the CSV rather than reusing prose.

### 1.3 Today's draw (VERIFIED, `competitions submissions`)

`submission.parquet` @ 2026-08-14 00:07:11 → **COMPLETE, public 0.70**. Description reads
*"AUTO-REFILL 2026-08-13 — frozen-fork filler (eternal fallback; auto-armed by scripts/daily_submit.py
because the queue was empty)"*.

- **Second-lowest draw of the campaign** (record min 0.65, 08-01).
- **Second consecutive sub-0.80** (0.78 → 0.70).
- `runs/ledger.json` is **stale** — still n=30, `latest 0.78`, `latest_date 2026-08-13`.

INFERRED (authoritative recompute belongs to `scripts/ledger.py`, not to this file):
n=31, mean ≈ **0.9368**; z(0.70) vs the n=30 stats = **−1.61**; trailing-4 **0.9975 → 0.9100**, a move
of **−0.0875**. The retired fixed-0.80 leg would have fired; the rule that actually binds is the R23
paired harm-pause at trailing-4 −1.5s = **−0.228**, and −0.0875 is well inside it. **No trigger.**
Flagging rather than acting — the ledger write is the day session's job.

Second flag: the **AUTO-REFILL** path means yesterday's queue was empty at submit time. That is the
daemon working as designed, but it is also the second consecutive day the only thing on the board was
the eternal fallback.

---

## §2 — Discussions

**Route.** `uvx --from kaggle==2.2.2 kaggle competitions topics list <slug> -s new` and `-s recent`,
plus `topics show <id>`, with `PYTHONIOENCODING=utf-8`. Note the CLI signature: it is
`competitions topics list <slug>`, **not** `competitions topics <slug>` (the bare form errors).
chrome-devtools was not needed for the forum; it *was* needed for §3.

**Window.** Posts with `postDate ≥ 2026-08-13`, plus new comments on older threads via `-s recent`.
Prior disposition read from `learnings/sweeps/sweep_2026-08-13.md` so nothing already ruled on is
re-adopted.

### 2.1 New posts (3)

| # | Topic · title · author · date | Verdict | Reason |
|---|---|---|---|
| A | **735147** · "Submission stuck in *Queued* (RTX Pro 6000) for 28+ min — anyone else?" · Dinesh kumar Thiyagarajan · 2026-08-14 07:52Z, 4 comments through 10:49Z | **ADAPT (schedule only, not method)** | Three independent reports today of RTX PRO 6000 queueing: 28 min, **3 h**, and **8 h then self-cancelled** (Benedicte HELFER, OverfitOracle). The 122B NVFP4 envelope screen is scheduled onto exactly this GPU today — budget for multi-hour queue latency, and do **not** read a long Queued state as a build defect. Nothing here changes the agent. |
| B | **734994** · "Dynamic Value Model: 14% on ARC-AGI 3 public eval with DeepSeek V4 Flash (vs 6% baseline)" · Jason Feng · 2026-08-13 13:13Z | **IGNORE** | Read the repo, not just the title. It is *"a lightweight test-time method"* that maintains a visible JSON of value judgements and has the model *"explain how its current value judgements inform the next action"* — **explicitly** *"requires no weight updates, auxiliary model calls, search branches, or environment copies."* That is the advisory/self-narration class three independent lines have now retired (RedundancyBench 24.88%, the animation arm, the executive>advisory finding). Eval is **single-run on 25 games**, and the author himself writes the comparison shares *"a temporary API-balance outage."* The repo **does not address action efficiency at all**, which is our binding constraint. |
| C | **734989** · "All my submissions score 0.00 — even an exact copy of the official Random Agent sample" · mina wailin · 2026-08-13 12:47Z, 1 comment | **IGNORE (method) / note (instrument)** | A platform-side scoring failure on someone else's account: `KAGGLE_IS_COMPETITION_RERUN` unset, `Could not resolve host: gateway`, `Game list: []`, and the run still lands **Succeeded + 0.00**. No technique. Worth one line only because the failure is *silent* — it scores 0.00 rather than erroring. Our fork already gates on `TRUE_SUBMISSION = KAGGLE_IS_COMPETITION_RERUN` (cell 2 of `notebooks/duckfork/...ipynb`) and our banner canary reads it, so we would see it; no change needed. |

### 2.2 New comments on older threads (1)

| # | Thread · new activity | Verdict | Reason |
|---|---|---|---|
| D | **734585** · Colab Pro / GPU-quota linkage, Ya Xu + Benedicte HELFER, 08-13 12:30–16:40Z | **IGNORE** | Entirely about buying and re-linking Colab Pro quota to work around the 30 h cap. That is cloud spend, which is out under `feedback_arc_zero_budget`, and our daemon already resubmits a frozen version rather than needing a fresh build to submit. |

**Base rate holds: 0 ADOPT of 4 items.** Nobody at ≥ 1.40 posted or commented in this window.

### 2.3 Not new — 734843 was already adopted, and I independently replicated it at 11× the sample

`learnings/sweeps/sweep_2026-08-13.md` §1.1 already ruled **ADOPT (narrow form)** on Jason Feng's
"persistent memory issue with the Tufa Duck harness". I re-derived it from scratch before reading
their file, so this is an independent check rather than a re-adoption.

**Code, VERIFIED in our own tree** (`runs/harness_diff_0813/ds/jeroencottaar__taaf-kaggle-source-share/src/ARC3-Inference/inference/agent/tool_agent.py`):
`_update_summarized_knowledge_from_assistant(content)` is called at `:1896` and `:1930` — **`content`
only**. `reasoning` is captured four lines earlier at `:1889–1892` into
`assistant_message["reasoning"]` and is **never parsed for the labels**. The prompt at `:1250` and
`:1918` calls the `World model:` / `Goal model:` / `Plan:` prefixes *"helpful optional"*. Confirmed.

**My measurement — LM-free, zero-cost, over 596 archived `*_events.jsonl` under `runs/`,
37,320 transcript events, 50,140 model responses** (pooled across run families, so broader but
noisier than the 08-13 estimate):

| quantity | count | rate |
|---|---|---|
| model responses | 50,140 | — |
| responses with a THINKING section | 45,170 | 90.1% |
| responses with an ASSISTANT (visible) section | 17,208 | 34.3% |
| **no visible content** | ~32,900 | **65.7%** |
| ASSISTANT sections carrying ≥1 world-model label | 12,262 | 24.5% of responses |
| **THINKING sections carrying ≥1 world-model label (= lost)** | **158** | **0.32% of responses** |

Two conclusions, and they point in opposite directions:

- **The symptom replicates.** My 65.7% "no visible content" matches Jason's 66.8% and the 08-13 pull's
  64.6–75.3%. Three independent counts now agree.
- **The loss is smaller than the 08-13 pull measured, not larger.** Captured:lost = **12,262 : 158**,
  i.e. the harness already captures **98.7%** of every world-model label the model ever emits. My
  pooled rate is **0.32% of turns**; the 08-13 pull got 0.7–1.6% on three same-rail 25-game pulls.

**Actionable warning on the adopted rider.** The 08-13 ADOPT pre-registers a delivery endpoint of
*"recovered label-updates per run, band [0.5%, 3%] of turns."* **My pooled estimate, 0.32%, sits below
that band's floor.** The band was calibrated on n=4,441 turns from three same-era pulls; mine is
n=50,140 but heterogeneous, so theirs is the better-matched estimate for the current rail and I am
not overturning it. The point is that the floor is **tight enough that the rider could fail its own
delivery gate while working exactly as designed** — which is precisely the `feedback_audit_the_instrument`
failure mode, one day after that feedback was written. Recommend either lowering the floor to ~0.2%
or redefining the endpoint over *label-bearing* turns (where my ratio is 1.27%, comfortably interior).

**And the number that is actually strategic:** 65.7% of turns emit **no world-model update at all** —
not lost to the wrong channel, simply never written. The channel bug is ~1%. The *silence* is ~66%.
That is the same finding as "the agent FORGOT", measured a second way, and it is two orders of
magnitude larger than the thing we adopted a patch for yesterday.

---

## §3 — cstl

### 3.1 The record was wrong about the shape of the event (VERIFIED, our own archives)

`runs/lb_ground_truth.md` (08-12 refresh) says *"cstl enters at 2.52 … the largest single-entrant jump
of the campaign."* Our own archived CSVs contradict the "enters":

| date | cstl submissionDate | score |
|---|---|---|
| lb_2026-08-06 | 08-04 23:49 | **1.59** |
| lb_2026-08-07 | 08-06 20:55 | **1.59** |
| lb_2026-08-08 | 08-07 20:59 | **1.59** |
| lb_2026-08-09 | 08-07 20:59 | **1.59** |
| lb_2026-08-10 | 08-09 10:13 | **1.59** |
| lb_2026-08-11 | 08-09 10:13 | **1.59** |
| lb_2026-08-12 | **08-11 18:25** | **2.52** |
| lb_2026-08-13 | 08-12 20:02 | **2.70** |
| **lb_2026-08-14** | **08-13 20:08** | **2.70 (flat)** |

So the true shape is **1.59 → 2.52 (+0.93, one submission, 08-11 18:25) → 2.70 (+0.18, 08-12 20:02)
→ flat**. cstl spent at least a week parked at 1.59, i.e. **inside the same 1.58–1.61 shared-public-
artifact band** the 08-05/08-06 sweeps identified as the duck-fork pack.

**Why this matters more than the "mystery outsider" framing.** If cstl was a duck-band team at 1.59,
the +0.93 is a **delta applied on top of an artifact we also run**, not an unknown system from
nowhere. That makes it the single most relevant unexplained result on the board for us — and it also
means the ceiling of the artifact family is at least 2.70, well above the 1.47 boristown anchor and
the ≈1.26–1.36 efficiency-reframe ceiling we computed. INFERRED, and it is inference, not evidence:
the +0.93 in one step is not a tuning move, it is a mechanism.

**One thing I will not over-read.** The public LB shows a team's **best** score with its **latest**
submission date. cstl's score being flat across four resubmits at 1.59 and two at 2.70 therefore says
**nothing** about whether their agent is deterministic — later submissions simply did not beat the
best. Any "cstl runs a low-variance artifact" claim is unsupported.

Resubmit cadence (VERIFIED): 08-04 23:49, 08-06 20:55, 08-07 20:59, 08-09 10:13, 08-11 18:25,
08-12 20:02, 08-13 20:08 — roughly daily, clustered ~20:00–21:00 UTC. 25 entries total. Consistent
with a daily-submit routine like ours.

### 3.2 Who they are (VERIFIED via chrome-devtools on the rendered leaderboard)

`kaggle.com/cstl` is a **404** — "cstl" is a *team* name (teamId 16364346), not a username. The
rendered leaderboard row links two member profiles:

**`tehnar` — "Tehnar"** · Software Engineer · **Amsterdam, North Holland, Netherlands** · joined 11
years ago · last seen in the past day · 6 followers.
- Competitions (5 completed): **NeurIPS 2024 — Lux AI Season 3, 116/701** (Featured, *Simulation*);
  TalkingData AdTracking 2958/3943; Denoising Dirty Documents 70/161; **Хакатон AI.Hack СПб
  Revisit prediction 2/16** and **Churn Detection 4/21** (invite-only, St. Petersburg).
- Public notebooks: **1** — *"Theano conv network"*, updated **11 years ago**. Nothing else.
- Forum contributions: **1** — a comment on *"PyCharm Installation taking forever"*, **8 years ago**.
- **Zero** ARC-related public artifact of any kind on Kaggle.

**`gatamaz` — "TG"** · **San Francisco, California** · joined 9 years ago · last seen in the past day
· **1 follower** · **1 competition ever, and it is this one.** No notebooks, no datasets, no posts.
Effectively a dormant account activated for ARC-AGI-3.

### 3.3 Off-Kaggle (VERIFIED where stated)

- **`github.com/tehnar` = Vsevolod Stepanov.** 26 repos, 7 followers. Identity link is **INFERRED but
  strong**: same handle, and the repo set matches the Kaggle profile's Russian-hackathon history —
  `SPbAU-Generation-Z-Team-Reference` (ACM ICPC Finals 2016), `SPbAU-Speech-Recognition`,
  `au_dl_course`, and **`rewind-viewer` — "Fast match viewer with rewinding support for Russian AI Cup
  championship."**
- **Recent GitHub activity: none.** Public events feed returns **0 events**. Newest pushes are a
  `spark` fork (2025-12-04) and `rewind-viewer` (2024-08-08). **No ARC repo, no agent repo, nothing
  from the 2026 campaign.**
- **arXiv:** no paper by Vsevolod Stepanov on ARC-AGI / ARC Prize / agents.
- **X/Twitter, blogs, ARC Prize community leaderboard:** no entry, no post, no mention of cstl,
  tehnar, gatamaz, or Vsevolod Stepanov in connection with ARC-AGI-3.
- **Kaggle datasets:** none public on either handle.

### 3.4 Verdict

**A trace exists for WHO. There is no trace at all for WHAT.**

We now know cstl is two people; that one of them has a genuine *competitive-agent-in-simulation*
pedigree (Russian AI Cup, ACM ICPC finals, NeurIPS Lux AI S3) rather than an LLM-prompting one; and
that he builds **instrumentation for rewinding and inspecting agent match state**. That is a
biographical fact and a suggestive one, and it is the honest limit of what can be said.

**There is zero public description of the method.** No notebook, no dataset, no forum post, no
comment, no repo, no paper, no thread, no talk. **No trace found.** I am not proposing a mechanism,
because there is no evidence for one and a plausible story here would be indistinguishable from
fiction. The only defensible operational conclusion is the one in §3.1: **the jump happened from
inside the duck band, so the artifact family's ceiling is ≥ 2.70** — which refutes, on its own, the
efficiency-reframe ceiling of ≈1.26–1.36 as a property of the *family* rather than of *our
configuration of it*.

---

## §4 — What I am handing to the day session

1. **Ledger write owed:** today's draw **0.70** is not in `runs/ledger.json` (still n=30 / 0.78).
   No rule fires; the paired harm-pause move is −0.0875 against a −0.228 trigger.
2. **Rank correction owed:** `runs/lb_ground_truth.md` has carried "below #49" since 08-09.
   Truth today is **#100** (94 strictly above, 9 tied at 1.33 spanning #95–#103). Refreshed.
3. **Instrument warning on the 08-13 ADOPT:** the pre-registered delivery band `[0.5%, 3%]` of turns
   for the reasoning-channel memory rider has a floor **above** my pooled-corpus estimate of 0.32%.
   Widen the floor or redefine the endpoint over label-bearing turns before the rider ships.
4. **Ops warning for the b122 envelope screen:** RTX PRO 6000 queues are 3–8 h today per three
   independent reports (#735147). A long Queued state today is not a build defect.
5. **The 66% number:** two-thirds of agent turns emit no world-model update at all. That is ~66×
   larger than the channel bug we patched, it is the same phenomenon as "the agent FORGOT", and no
   open lane currently targets it.
6. **cstl reframed:** not an outsider — a 1.59 duck-band team that found a +0.93 step on 08-11.
   Artifact-family ceiling is therefore ≥ 2.70, not ≈1.36.
