# cstl → 2.52: the first 2.0+ score on the ARC-AGI-3 board (investigation, 2026-08-12)

**Scope.** Team **cstl** (`teamId 16364346`, members `gatamaz`, `tehnar`) went from **1.59 → 2.52**
in a single submission on **2026-08-11 18:25:39 UTC**, taking #1 by **+0.66** over YUTO KOJIMA
(1.86) and sitting at **1.89× our 1.33**. This file is the exhaustive read: timeline, evidence,
the capability 2.52 implies, and what (if anything) is liftable.

**Method.** Read-only Kaggle CLI (live LB `--show`, full LB CSV snapshots `2026-08-11T11:24:31Z`
and `2026-08-12T10:53:53Z`, kernel list by `dateRun`, dataset list by `updated`, user/kernel/dataset
search on all three handles), our own archived `runs/lb_daily/lb_2026-08-{06..12}.csv`,
chrome-devtools browser on the Kaggle discussion feed (lock cleared — see §6), `gh` API (user +
repo + commit reads), HF model API, WebSearch. Local scoring arithmetic with
`scripts/phase1_gate.py` (exact RHAE mirror) against `runs/kernel_pulls/animation_v1/benchmark.json`
and the `runs/lb_process_model` calibration. **Zero pushes, zero submissions, zero spend.**

**Provenance markers:** **[V]** verified by direct read; **[INF]** my inference; **[UNK]** unknown.

---

## 1. TIMELINE — a single leap, not a staircase

| when (UTC) | source | cstl state |
|---|---|---|
| ≤ 2026-08-04 | `intel_sweep_2026-08-04.md` §3 | **first appearance**, "cstl 1.59 (new name)" |
| 2026-08-04 23:49:20 | `runs/lb_daily/lb_2026-08-06.csv` | 1.59, rank **8** |
| 2026-08-06 20:55:01 | `lb_2026-08-07.csv` | 1.59, rank 8 |
| 2026-08-07 20:59:30 | `lb_2026-08-08.csv` | 1.59, rank 8 |
| 2026-08-07 20:59:30 | `lb_2026-08-09.csv` | 1.59, rank 9 (pushed down) |
| 2026-08-09 10:13:38 | `lb_2026-08-10.csv`, `lb_2026-08-11.csv` | 1.59, rank 9 → 10 |
| **2026-08-11 11:24:31** | official LB CSV snapshot | **1.59, rank 10, 22 submissions** |
| **2026-08-11 18:25:39** | LastSubmissionDate on the 08-12 snapshot | **the jump submission** |
| **2026-08-12 10:53:53** | official LB CSV snapshot | **2.52, rank 1, 23 submissions** |

**[V] The jump is one submission.** The team had **22** submissions at the 08-11 11:24Z snapshot
(score still 1.59) and **23** at the 08-12 10:53Z snapshot (score 2.52), with `LastSubmissionDate`
= 08-11 18:25:39. Only one submission exists in that window, so **submission #23 is the 2.52**.
There is no staircase: 1.59 held flat across at least four submissions and seven days, then
+0.93 in one draw.

**[V] Cadence.** 23 submissions over ~8 days of visible activity — roughly 3/day at the start,
then a 2-day gap (08-09 10:13 → 08-11 18:25) immediately before the jump. That gap is the only
behavioural tell on the board: they stopped grinding, then came back with a different number.
Contrast KOJIMA (65 subs, resubmits daily, frozen at 1.86 for weeks).

**[V] The rest of the board did not move.** Gold/top-13 cutoff held at 1.58 for a fourth flat day;
KOJIMA 1.86, Andy liu 1.69, Lord Han Solo 1.65 all unchanged. This is not a rescoring event or a
metric change — it is one team, one submission.

---

## 2. IS IT A LUCKY DRAW? — decisively no

The house model of the LB is a lottery: our own record is **n=29, mean 0.9503, s 0.1513**
(CV 0.159), and yw8837's published 11-submission ledger has CV ≈ 0.26 — the two independent
estimates of draw dispersion on this rail. Ask what a 2.52 costs under either.

Constraint: cstl's **best of 22** was 1.59. Using Blom's `E[max of n]` (a₂₂ = 1.910) to back out
the implied process mean from their own history:

| assumed CV | implied μ | implied σ | z of 2.52 | P per draw | P(any of 23 draws) |
|---|---|---|---|---|---|
| 0.16 (our ledger) | 1.218 | 0.195 | **+6.68** | 1.2e−11 | 2.7e−10 |
| 0.26 (yw8837) | 1.062 | 0.276 | **+5.28** | 6.6e−08 | 1.5e−06 |

**[V] Draw variance is falsified as an explanation** by 5–7 orders of magnitude. (For scale, our
own best-ever draw 1.33 is only z = +2.51 against our record.) Either cstl changed something
between 08-09 and 08-11, or their process is not the one their first 22 submissions sampled.
**[INF]** A capability or configuration change is by far the most economical reading.

---

## 3. PUBLIC ARTIFACTS BY cstl — none. This is a silent team.

**[V] Verified empty on every route:**

- `kaggle kernels list -s gatamaz` / `--user gatamaz` → **Not found**. `kaggle datasets list -s
  gatamaz` → **No datasets found**.
- `kaggle kernels list --user tehnar` → one 2015 notebook (`tehnar/theano-conv-network`,
  Theano conv net). `kaggle datasets list -s tehnar` → none.
- `kaggle kernels list -s cstl` / `datasets list -s cstl` → unrelated hits only (a stock dataset
  for the ticker CSTL, three unrelated notebooks).
- Kaggle profile `gatamaz` = **Tamaz Gadaev**, San Francisco CA, joined 9 y ago, "Quietly working
  away", **Competitions (1)** — ARC is his only competition, no code, no datasets, no discussion.
- Kaggle profile `tehnar` = **Tehnar**, Software Engineer, Amsterdam NL, joined 11 y ago,
  6 competitions, **Code (1)** (the 2015 notebook), **Discussion (1)** — a 2018 comment on a
  PyCharm install thread. Nothing about ARC, ever.
- Full sweep of ARC-competition kernels by `dateRun` (top 40, covering 08-10 → 08-12) and of
  `taaf` / `arc-agi-3` datasets by `updated`: **no artifact by any of the three handles.**
- HuggingFace model API search for `gatamaz` and `tehnar`: **empty**.
- No forum post, comment, or reply by either handle anywhere in the competition.

**Identity, as far as public evidence goes [V-2nd / INF]:**
- `gatamaz` → GitHub **`tamazgadaev`** (name match; secondary sources put a Tamaz Gadaev in SF as
  Lead ML Engineer at Jhourney, MIPT applied physics/maths, ex-consulting for Huawei/Samsung —
  matching his `jhana_eeg` repo). **[INF, name-based — not confirmed by either party.**
  A second GitHub account `ttgadaev` (MIPT/ResearchGate) exists with 0 repos.]
  His 10 repos: `llm-optimizer-benchmark` (his own, Python, pushed 2026-01-19), and **forks** of
  `vllm` (2026-01-12), `prime`/ZeroBand, `gonka`, `inspect_evals`. **Nothing ARC-related, nothing
  pushed since January.** GitHub public events: empty.
  - **Worth flagging honestly:** the one self-authored recent repo is an **LLM inference-optimisation
    benchmark**, and he keeps a **vLLM fork**. On a rail where the binding constraint is LM calls
    per 9h wall (our own read, and Jakob Brüggen's "tokens are the real currency"), an
    inference-optimisation specialist producing a 1.9× is *coherent*. **That is a coherence
    observation, not evidence.** [INF]
- `tehnar` → GitHub **`tehnar`** = Vsevolod Stepanov (26 repos, newest activity 2025-12 on a Spark
  fork; HSE/SPbAU coursework, an AI-Cup match viewer). **Nothing ARC-related.** Events: empty.

**Conclusion for §3:** **nothing is liftable from cstl.** They have published no code, no dataset,
no model, no notebook, no post. Any claim about *how* they did it is speculation.

---

## 4. REALITY CHECK — is 2.52 even reachable, and what capability does it represent?

### 4.1 The scale (so the number is not misread)

`scripts/phase1_gate.py` mirrors the official scorer exactly (max abs error 1.78e−15 over 1,000
cross-checks vs Tufa's 500 stored runs):

```
level_score_i = min(115, (baseline_i / actions_i)^2 * 100)     [0 if level not completed]
game_score    = min( Σ w_i·level_score_i / Σ w_i ,  (Σ w_i over scored levels)/Σ w_i · 100 )
                with w_i = i+1
LB score      = mean game_score over the FIXED official ~110-game set, one pass each
```

So the metric runs **0 → ~115**, not 0 → 1. **2.52 is 2.52% of a ~100-point ceiling.** Public SOTA
(Retrodict, `gpt-5.6-sol` at max effort, $654, internet-connected, *not* Kaggle-legal) is **99.86**.
**[V] 2.52 is nowhere near the metric's ceiling and there is nothing arithmetically suspicious
about it.**

### 4.2 What 2.52 buys, in levels

Using the level-count distribution of the 25 public games as a proxy for the official set
(T = N(N+1)/2 per game; computed from `runs/kernel_pulls/animation_v1/benchmark.json`):

| profile, every game | mean score |
|---|---|
| clear **level 1** at exactly the human action baseline | **3.524** |
| clear **levels 1–2** at the human baseline | **10.571** |
| clear **levels 1–3** at the human baseline | **21.141** |

Therefore, expressed as a fraction of games solved at human-equal efficiency:

| LB score | ≈ % of games at perfect L1 | ≈ % of games at perfect L1+L2 |
|---|---|---|
| our 1.33 (best draw) | 37.7% | 12.6% |
| cstl's old 1.59 | 45.1% | 15.0% |
| KOJIMA 1.86 | 52.8% | 17.6% |
| **cstl 2.52** | **71.5%** | **23.8%** |

**2.52 ≈ "clears level 1 on about seven of every ten games, at roughly human action efficiency."**
That is a real step up — but it is a *first-level* agent, not a deep one. **[V]**

### 4.3 The finding that actually matters — and it is about *us*, not them

Our own animation-arm run (25 public games, 2026-08-11) clears **17 levels** and scores **1.635**.
Rescoring **the same 17 levels at exactly the human baseline action count**:

```
arm actual mean                                    1.635
same levels, human-perfect efficiency              2.549   (×1.56)
+1 level on every game we already clear, perfect    6.618   (×4.05)
```

**We already complete enough levels to score ~2.55 on the public 25. We score 1.635 because we
spend 3–8× the human action count getting there.** The per-level efficiency table:

| game | levels | actions / baseline per level | scored | if perfect |
|---|---|---|---|---|
| bp35 | 1 | **8.33×** | 0.03 | 2.22 |
| ar25 | 2 | **5.97×**, 1.06× | 5.02 | 8.33 |
| sp80 | 1 | **5.77×** | 0.14 | 4.76 |
| m0r0 | 1 | **4.60×** | 0.23 | 4.76 |
| tu93 | 2 | **3.68×**, 1.25× | 3.01 | 6.67 |
| vc33 | 1 | **3.00×** | 0.40 | 3.57 |
| cd82 | 1 | 1.18× | 3.41 | 4.76 |
| tn36 | 2 | 0.91×, 0.33× | 10.71 | 10.71 |
| (ka59, lf52, lp85, sb26, sc25, su15 all ≤ 0.77× — already at or better than human) | | | | |

Six games throw away **0.91 points of mean score** (56% of everything we score) purely to
inefficiency, because `level_score = (baseline/actions)²` — 2× the human actions gives 25%,
3× gives 11%, 8× gives 1.4%.

### 4.4 Placing 2.52 on our own scale

`runs/lb_process_model/report.md` calibrates local-25 → LB with **c ≈ 0.58–0.62** (official mix is
harder than the public 25). Applying it in both directions:

| capability | local-25 | → LB equivalent |
|---|---|---|
| our arm as it ran | 1.635 | 0.95 – 1.01 (**matches our ledger mean 0.9503** ✔) |
| our 17 levels, at human efficiency | 2.549 | **1.48 – 1.58** (the gold line) |
| **cstl 2.52 LB** | **4.07 – 4.34** | 2.52 |
| +1 level everywhere, at human efficiency | 6.618 | 3.84 – 4.10 |

**[INF] 2.52 sits between "our current level depth played at human efficiency" and "half a level
deeper everywhere, at human efficiency".** In plain terms: cstl is roughly *one additional level
per already-solved game, played efficiently* ahead of where our current agent's raw depth would put
us if we stopped wasting actions.

### 4.5 Is it consistent with a legal in-kernel offline run?

**Yes, and nothing found argues otherwise.** [V]
- The metric ceiling is ~100+; 2.52 uses 2.5% of it. No cap is being exceeded and no per-level
  score can exceed 115 by construction.
- Discussion **734414** (mina wailin, 08-11) documents that a submission which does *not* execute a
  real Phase-B rerun finishes in ~30 s with a dummy parquet and **scores 0.00** — so a 2.52 required
  a genuine gateway + long `main.py` competition rerun.
- The publicly known *architectures* that reach far higher (Retrodict 99.86, Schema ~99) are all
  frontier-API, internet-connected and cost hundreds of dollars per run — they are **illegal
  in-kernel** and 2.52 is nowhere near their numbers anyway. A local-model agent at 2.52 is an
  ordinary point on the observed public distribution: the same forum thread (**732854**) has
  **Son Pham (LB #191 @ 1.22) reporting "Got 2.8 locally so far"** on the public 25.
- **[UNK]** Whether they used the duck/TAAF line, a fork of it, or something of their own.
  No evidence either way.

**I see no basis to question the number.** The only honest residual is that *we cannot verify it*
— private-set scores are not reproducible from outside, and cstl has published nothing.

---

## 5. RETRODICT / TYCHO / PRIME AGENT — checked, no connection found

- **Retrodict** (`ryanbbrown/Retrodict`, 28 ★, **no license**): pushed 2026-08-11T17:33Z — **52
  minutes before cstl's 18:25Z submission**. I checked the commits: `Add Retrodict article links`
  (17:32) and `Update public harness comparison` (16:58) are **documentation-only**; there is no
  local-model port, no Kaggle harness, no code change. `gpt-5.6-sol` at max effort with internet is
  structurally impossible in-kernel. **[V] Temporal coincidence only. No connection.**
- **Prime Agent** (`PrimeIntellect-ai/prime-agent`, assessed in
  `prime_agent_portability_2026-08-08.md`): the `tamazgadaev` GitHub account holds a **fork of
  `prime`/ZeroBand** — but that is Prime Intellect's *distributed-training* framework, not
  prime-agent, the fork is from **2024-11**, and forking a repo is not evidence of anything.
  **[V] No connection found.** [INF: I flag it only because it is the single tenuous thread between
  a cstl member and a lane we have assessed, and it should be written down rather than rediscovered.]
- **Tycho / Schema**: nothing new; no cstl handle appears in either project.

---

## 6. FORUM SWEEP 08-10 → 08-12 (browser route restored)

**Process note, worth keeping.** The chrome-devtools MCP lock that blocked the 08-10 and 08-11
sweeps was **an orphaned Chrome from 2026-08-09 18:13** holding
`~/.cache/chrome-devtools-mcp/chrome-profile`, launched with `--remote-debugging-pipe` (hence no
`DevToolsActivePort`, hence no out-of-band CDP attach — exactly as
`discussion_sweep_2026-08-11.md` §0 diagnosed). It was **not** a live session. Killing only the
processes whose command line contains `chrome-devtools-mcp*chrome-profile` and removing the
`Singleton*` files restored the route immediately. **Do this first next time the lock appears;
check process creation dates before assuming another agent owns it.**

Frontier is now **topic 734585** (was 734369 in `discussion_sweep_2026-08-11.md`).

**[V] No post anywhere mentions cstl, the 2.52, or a new #1.** No congratulation thread, no host
statement, no technique thread dated 08-08..08-12 that would explain it. cstl is silent and the
forum has not noticed.

New / changed since the 08-11 07:55Z frontier:

1. **734369 "Write Up: Taaf Anim Agent"** (Jakob Brüggen, Helmut AGI #10) — already swept, but it
   now has **two comments** that were not there at 07:55Z:
   - **Xuan** (15 h ago): *"the frames are important for Sol but seems to blow up the context
     window for Qwen"*. Points at **`vista-research.github.io`** — a **vision-only** approach whose
     entire idea is to **upscale the 64×64 grid to 512×512** so the model's vision stack can resolve
     it ("because the model is trained more with online images that are like 512×512"); reports
     **100% for Claude / Sol**. His A/B: with VISTA, Qwen *"is able to reason that the bottom right
     is the target area immediately after seeing the image"* on **ft09**, but *"cannot accurately
     infer coordinates based on image and is also worse in reasoning"*; the animation toolkit is
     *"vital for sol for problem like tn36"* but *"for qwen the vista approach just does not work as
     well"*. He asks about **SFT/GRPO on Qwen**.
     → **This is a new, unswept public artifact (VISTA) and an independent same-model negative on
     vision-in-the-loop at 27B.** It reinforces `intel_sweep_2026-08-11.md` finding 7
     (demote the VLM swap) rather than contradicting it. Worth a proper read in the next sweep.
   - **Greg Kamradt (COMPETITION HOST)** (7 h ago): *"Thank you for sharing"* — acknowledgement
     only, no rules or scoring content.
2. **734414** (mina wailin, 08-11) — Phase-B submissions finishing in ~30 s with the commit dummy
   parquet and scoring 0.00. Operational; used above as legality evidence. 0 comments.
3. **734585** (Jason Feng, 08-12) — *"i can't submit to the competition if i have used up my gpu
   quota, but the competition scoring does not use my gpu quota"*. Operational.
4. **732854** (Reki, "What are your agents scoring on the 25 public games?") — the "2.8 locally"
   comment is now attributed: **Son Pham**, LB #191 @ 1.22 (`sonphamorg`, team *Logical Arbitrage*).
5. Artifact churn confirming the animation write-up is propagating: `iseesmth/prolong-eval` +
   `taaf-kaggle-source-prolong-eval` (08-11 16:24), `iamjasonfeng/chimpanzee-1-1-anim` (08-12
   04:39), `cascadematrix/arc-agi-3-causal-animation-v1`, `finalsunflower/arc3-anim-lb161-exact-
   validation`. **No cstl artifact among them.**

---

## 7. WHAT IS LIFTABLE, AND WHAT IS LEARNABLE

**Liftable from cstl: nothing.** Zero published artifacts. Anyone claiming to know their method is
guessing.

**Learnable, and it is the most actionable number in this file:**

1. **Our binding constraint is action efficiency, not level depth.** Our own 17 levels re-scored at
   the human baseline give **2.549 local ≈ 1.48–1.58 LB — the gold line — with no new capability at
   all.** Six games (bp35 8.3×, ar25 5.97×, sp80 5.77×, m0r0 4.6×, tu93 3.68×, vc33 3.0×) burn 56%
   of our achievable score. Because the term is squared, halving actions on those six is worth
   ~4× their current contribution. **This is a self-derived, verified, zero-cost finding and it
   should re-order the queue.**
2. It converges with three independent external reads: Retrodict's lift is **token/action
   efficiency in the runner** (5.5× fewer tokens than the previous public best), Jakob Brüggen's
   write-up says **"tokens are the real currency, not actions… every run hits the wall-clock cap"**,
   and our own M2 shows the same coupling on our rail (§ animation score, 2026-08-12).
3. **Cadence is not the lever.** cstl got +0.93 on submission #23 after a 2-day pause; Andy liu
   holds #3 on 7 submissions; KOJIMA is frozen at 1.86 on 65. Our 108 submissions have produced one
   1.33. Nothing on this board rewards grinding.
4. **Board effect to carry:** cstl's insertion tightens the **top-5 prize line 1.62 → 1.64** and
   pushed Tufa Labs out of the prize band one day after they entered it. The **gold/top-13 line is
   unmoved at 1.58 for a fourth day** — the band we actually chase is still flat, consistent with
   the 08-11 sweep's deceleration read. Our gap to gold stays **0.25**; gap to the prize line
   widens **0.29 → 0.31**.

## 8. UNKNOWNS, STATED PLAINLY

- **[UNK]** What cstl changed between 08-09 and 08-11. No artifact, no post, no commit.
- **[UNK]** What model they run, whether they use the duck/TAAF line at all, and whether 2.52 is
  their capability mean or the top of a wider distribution around a new, higher mean. One
  submission cannot distinguish those.
- **[UNK]** Whether the `gatamaz` → `tamazgadaev` identification is correct. It rests on a name
  match plus a location match through a secondary aggregator. **Do not repeat it as fact.**
- **[UNK]** Whether the official private set behaves like the public 25 in level-count structure;
  §4.2's percentages use the public distribution as a proxy and the true T-distribution is not
  published.
- **[UNK]** Whether 2.52 will hold on the private LB. Public-LB position has never been our
  selection currency (R25-N3: ρ̂_draw ≈ 0).
