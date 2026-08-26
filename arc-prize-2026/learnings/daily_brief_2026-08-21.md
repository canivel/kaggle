# Daily brief — 2026-08-21 (iterate session)

Session context: **0 submission slots available** — both were spent by the graft lane before this
session opened (06:45 `arc3-q38-graft-eval` v1 = Arm 3; 06:50 `arc3-duck-repro-pathsafe` v1).
Per the 08-18 lane-ownership ruling this session pushes nothing and duplicates no locked action.

---

## 1a. RESULT DEEP-DIVE — the 1.59, and what it actually bought

**The number.** Submission 55656892 (00:31:23Z, the A21 exploration draw: field-floor adoption arm,
byte-faithful FOYSAL-2.23 rebase, Q38 xhigh, 08-07 harness) scored **public 1.59, COMPLETE**.
That is the **campaign's best public draw ever** — prior all-time max 1.33, banked 07-18 and stale
for 33 days. The board row updated: `Canivel, Score 1.59, Rank 211 / 2453, SubmissionCount 117`.

**Was the pre-registered expectation met? Yes, and cleanly.** The arm was certified 6/6 pre-read and
sealed-read SIGNAL (lc_total 28 vs bar 27) *before* the draw; the draw decision was carried by
FOYSAL's board-verified 2.23, not by our own lc/score bands. The offline SIGNAL and the online draw
agree in direction — the first time this campaign that an offline screen has predicted a public
improvement. **Rank moved 1.33≈#328 → 1.59 #211, i.e. +117 places.**

**The statistical read, and an asymmetry worth recording.** Against the frozen-fork null
(n=37, mean 0.9316, s 0.1771) the draw is z = **+3.72**. Yesterday's 0.41 deep-dive correctly
*deflated* its own z because 0.41 was the **minimum of 37 draws** — a selected extreme, so
min-of-n arithmetic applied. **That deflation does not apply symmetrically here**: 1.59 is a
**pre-designated single draw of a newly pre-specified config**, not an extremum picked out of a
series. No multiplicity correction is owed, so the +3.72 stands as written. (What is still owed:
this config has **n=1**. Its own draw-to-draw spread is an assumption, not a measurement.)

### ★★★ THE FINDING: we set a campaign record and *lost* rank. The field is outrunning us ~3×.

Re-derived from our own daily full-board snapshots (`runs/lb_daily/`, plus today's pull). Holding a
**fixed** score of 1.59 and asking where it would have ranked on each day:

| date  | teams | #1   | top-5 (prize) | top-13 (gold) | top-50 | top-100 | rank @ a fixed 1.59 |
|-------|-------|------|---------------|---------------|--------|---------|---------------------|
| 08-15 | 2331  | 2.70 | 1.90          | 1.62          | 1.47   | 1.35    | **#21**             |
| 08-16 | 2345  | 2.70 | 1.98          | 1.65          | 1.49   | 1.38    | #26                 |
| 08-17 | 2365  | 2.76 | 2.33          | 2.00          | 1.62   | 1.49    | #64                 |
| 08-18 | 2383  | 2.81 | 2.35          | 2.05          | 1.72   | 1.56    | #88                 |
| 08-19 | 2408  | 3.57 | 2.53          | 2.24          | 1.88   | 1.67    | #130                |
| 08-20 | 2428  | 3.57 | 2.58          | 2.37          | 2.03   | 1.79    | #169                |
| 08-21 | 2453  | 3.57 | 2.72          | **2.47**      | 2.08   | 1.86    | **#209**            |

* The **same score loses ≈30 places per day.** Six days ago 1.59 was a **top-25** score; today it is #209.
* **Gold line +0.14/day** (1.62 → 2.47 in 6 days). **Top-50 +0.10/day. Top-100 +0.085/day.**
* **Our best improved +0.043/day** (1.33 → 1.59 over the same 6 days).
* Net: on 08-15 we were **#119 at 1.33**; today, at our **best-ever 1.59**, we are **#211**.
  **We gained 0.26 of score and still lost ~92 places.**

**Consequence for the standing Arm-0 order (this is the part that changes the plan).** The
"field-floor config = default nightly head, redraw nightly ≈ +0.3 LB at zero GPU" order treats the
draw distribution as the asset. Priced properly as max-of-n on the assumed N(1.6, 0.2): 10 nightly
redraws → E[max] ≈ **1.91**; 30 redraws → ≈ **2.01**; the curve is flat after that. So the *entire*
remaining value of redrawing this config is **≈ +0.4, asymptoting near 2.0** — which is below
**today's** gold line of 2.47, and the gold line is moving +0.14/day. **Redrawing is a treadmill: a
one-time ~0.3–0.4 gain that the field erases in ~3 days.** This is the same *class* of finding as
the 08-15 "efficiency reframe is dead arithmetically" — not an argument against keeping the
field-floor as the nightly floor (it is free and it is our best config), but a decisive argument
that **draw-exploitation cannot reach a prize position and must not be mistaken for progress.**
Deadline is **2026-11-02 (73 days)**, so there is runway — but only for a capability step-change.

**Honest limits.** (i) The linear extrapolation of the gold line will saturate; the argument does
not depend on it — even a *frozen* 2.47 is above the ~2.0 redraw ceiling. (ii) n=1 for this config,
so 1.59 may itself be a high or low draw. (iii) Rank comparisons use public LB only.

---

## 1b. DISCUSSIONS SWEEP

Enumerated by CLI (`kaggle==2.2.2 competitions topics list`, the reliable route — browser/WebFetch
remain dead ends). Max topic id yesterday **735662**; today exactly **one** new topic, confirmed by
an id-sorted scan across all pages.

* **736540 — "non-official community game environments for the ARC-AGI-3 benchmark"** (robenten,
  08-21 12:05Z, 0 votes / 0 comments). Content is a *question*, not a disclosure: "are there any
  non-official ARC-AGI-3 games created by the community?", linking one repo.
  **Verdict on the post: IGNORE** (no plan-relevant claim, unanswered).
  **Verdict on the lead inside it: ADOPT-CANDIDATE — see §3, the most valuable thing the sweep has
  surfaced in weeks.**

The forum still discloses **nothing** about banking/transfer/grafts and **nothing** about cstl
(#1 at 3.57, still untraced).

---

## 1c. RESEARCH SWEEP

* **Sensi — "Learn One Thing at a Time: Curriculum-Based Test-Time Learning for LLM Game Agents"
  (arXiv 2603.17683)** — evaluates **directly on ARC-AGI-3**. Abstract read.
  **Verdict: IGNORE as a performance claim — and log it as a third instance of the campaign's
  non-comparable-headline pattern.** Its banner number is "50–94× greater sample efficiency
  (32 attempts vs 1600–3000)", but **v1 solved 2 levels and v2 solved 0 levels.** The efficiency is
  efficiency-of-completing-its-own-curriculum, *not* of solving. For scale: our certified field-floor
  arm scores **lc 28 across 25 games**. We are far ahead of this published system.
  (Prior instances: MAP's "22/25" = beat-ReAct rate; ARChitect's ARC-AGI-1 grid-transduction win.)
  **ADAPT (one idea, no slot):** its diagnosed failure mode — a *self-consistent hallucination
  cascade originating in the perception layer*, concluding the bottleneck has moved from learning
  efficiency to **perceptual grounding**. That is an independent third-party diagnosis pointing at the
  same place as `feedback_arc_jepa_dead`'s pivot note (Rudakov visual-priors), and it is a
  **competing explanation for our own open question** (`forgetting REFUTED or DELIVERY-WITHOUT-USE?`
  — mech-C delivered 96.3% and behaviour did not change). If perception is the binding constraint,
  delivery-without-use is exactly what we would observe. **For Sunday.**
* Surfaced, **titles/abstracts only — not read, no disposition claimed**: Graph-Based Exploration for
  ARC-AGI-3 (2512.24156); Workspace Optimization: How to Train Your Agent (2605.09650);
  ARC-AGI-3 technical report (2603.24621, context only — frontier LLMs <1% human efficiency).
* Carried from 08-20, still parked for Sunday: **BeliefMem (2605.05583) → ADAPT.**

---

## 2. INSTRUMENT AUDIT (pre-data, completed before Arm 3 reached COMPLETE)

Per `feedback_audit_the_instrument`, today's sealed scorer `duck_eval/graft/q38graft_score.py`
(Arm 3) was audited **while the kernel was still RUNNING**. **Verdict: PASS — and proven beyond its
own selftest.**

* Its own selftest: **17/17**.
* **The named landmine is genuinely defused.** The 08-20 ruling required `[clickmap] armed` to flip
  from FORBIDDEN-marker to **REQUIRED**-marker for this arm. It does, and both directions were proven.
* **Stronger test added by this session — certification against REAL log bytes, not handcrafted
  fixtures.** The real 08-19 graft-confirm pull was structurally mutated into an Arm-3-shaped log
  (clickmap injected into the FEATURES banner + `[clickmap] armed` record + served-model swap,
  edited *through* the CLI-2.2.3 JSON envelope so the escaping is real). Result: **certifies, all six
  flags detected through the escaping, lc computed correctly (14, the real run's true value).**
  **5/5 negative controls refuse for the right reason** (clickmap absent → wrong arm; wrong served
  model; effort pin present; searchmap armed; stock fallback). This is the exact gap behind the
  standing lesson that internal consistency is not correctness.
* **Two residual risks checked and cleared:** (i) the FORBIDDEN-flag test could false-fire if the
  runtime banner printed disabled flags as `"banking":false` — checked against the real banner, which
  **emits only enabled flags**, so it cannot; (ii) `"reasoning_effort" in log` is a whole-log
  substring test and would false-refuse a healthy run if the notebook echoed that token — the Arm-3
  notebook source contains **0 occurrences**.
* **Incidental confirmation:** `graft_score._read_log` is **slug-agnostic** (globs all log artifacts),
  so the new kernel name cannot orphan the reader; and it **fail-closes** on an unparseable log
  (proven accidentally when a malformed fixture of mine broke JSON validity — the scorer refused
  rather than mis-read).

---

## 3. OPEN QUESTIONS / PROPOSALS FOR SUNDAY'S PANEL

1. **★ THE 249-GAME OFFLINE SET (new, from §1b).** `github.com/theredbluepill/arc-interactive`,
   **MIT licensed**, verified by clone-and-inspect (no execution): **252 game packages**, same 4-char
   ID convention, standard package shape (`<id>/<ver>/{<id>.py, metadata.json}`) with
   `baseline_actions`, levels and frame-based interaction; the README documents ACTION1–6 (+undo)
   mapping and a Kaggle-style competition mode. It contains **copies of official games** (`ft09` is
   present as `ft09-9ab2447a`, the official id-hash convention, downloaded 2026-03-17) **alongside
   ~249 community-authored ones**.
   **Why this matters more than it looks:** every screen we run is **n=25 games**, which is the root
   cause of our chronically underpowered gates (SIGMA 0.1417 lc/game, C(3)=2.02, promotion bars set
   at +28% over the all-time high just to clear noise). A larger novel set attacks that directly, and
   it tests **generalization on games the agent has never seen** — the standing PRIORITY
   (`feedback_arc_generalization_first`: the private LB has more games).
   **Costed honestly:** these still need an LLM to play them, and our 3080 cannot serve Qwen3.8-27B,
   so this is **not** free local compute — it is Kaggle build-time GPU (30 GPU-h/wk). At the observed
   ~2h20m per 25-game run, a **50–75 game screen fits one kernel** and would cut the standard error
   of a screen by ≈1.4–1.7×. **Proposal: a 50-game screen set (25 official + 25 community-novel).**
   **Risks to state:** community games are a **proxy of unknown fidelity** to the private set;
   quality/difficulty is uncontrolled; third-party code would eventually need to be executed
   (inspected only, so far).
2. **Does the Arm-0 nightly-redraw order survive §1a?** It should be **kept as the floor** (free, best
   config) but **explicitly re-labelled as a floor, not a strategy**, with the ~2.0 ceiling on record.
3. **Perceptual grounding vs delivery-without-use** — Sensi's diagnosis vs our mech-C observation.
4. **cstl (3.57) remains completely untraced** at 2.2× our best draw.

---

## 4. STATE AT WRITING

* Ledger (`runs/ledger.json`, re-read not cached): **n=37, mean 0.9316, s 0.1771, trailing-4 0.8425,
  promotion bar (mean-of-4) 1.089.** The 1.59 is **not** in it and must not be — that ledger is the
  frozen-fork **null** distribution, and 1.59 is a different config.
* Board: **Canivel 1.59, #211 / 2453.** Gold 2.47, prize 2.72, top-50 2.08, top-100 1.86. #1 cstl 3.57.
* Kernels: `arc3-q38-graft-eval` (Arm 3) **RUNNING** at 08:30 EDT, ETA ≈09:05;
  `arc3-duck-repro-pathsafe` **RUNNING**. Arm-3 read gate: certified **AND lc ≥ 28** to head tonight.
* `kaos bench rejections`: **empty** (nothing rejected to consume).
