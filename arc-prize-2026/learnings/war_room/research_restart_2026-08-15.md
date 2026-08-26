# RESEARCH RESTART — what the top of ARC-AGI-3 is actually doing
**Date:** 2026-08-15 · **Order:** clean-slate investigation, deliverable is UNDERSTANDING
**Rules honoured:** read-only. Zero pushes, zero submissions, zero spend. One write (this file) + a leaderboard fetch.

Provenance tags are used on every load-bearing claim:
**[V]** verified by direct read this session · **[V-ext]** verified by an external primary source this session ·
**[INF]** my inference · **[UNK]** unknown / could not establish.

---

## THE ONE SENTENCE

> **The 2.5+ regime is not a new technique — it is a new engine: Alibaba released Qwen3.8-27B under Apache 2.0 at 15:00 UTC on 2026-08-14, a complete FP8 Kaggle mirror was public 2h48m later, a competitor who ran it reports "a consistent 2× score on the local 25", and every single team that jumped has a best-submission timestamp after that mirror went live — with the sole and important exception of cstl, whose 2.70 was banked 43 hours *before* the release and remains unexplained by any public evidence.**

**Top 3 actionable findings** (detail in §7):
1. **Qwen3.8-27B-FP8 is a structurally byte-identical drop-in for the engine we already serve** — same `Qwen3_5ForConditionalGeneration`, same 64 layers / 5120 hidden / 4 KV heads / head_dim 256 / `full_attention_interval` 4, same 248320 vocab, same vision tower (27/1152/16), same `image_token_id` 248056, and an identical tool-call block syntax so `--tool-call-parser qwen3_coder` carries over unchanged. Two complete public mirrors exist. **[V]** This is the cheapest high-value screen available to us and it is a config edit, not a research project.
2. **The new template adds a `reasoning_effort` knob defaulting to `xhigh`.** On a rail our own evidence says is **wallclock-bound, not action-bound**, an unexamined `xhigh` default is a live regression risk *and* `reasoning_effort: "low"` is a new, free tuning arm we never had. **[V]** Ship the swap and the knob as separate arms.
3. **Daniel Franzen is confirmed as the ARChitects** (ARC Prize 2024 grand prize, 2025 2nd place) **[V]**, and his published recipe is *per-task test-time training* — 128 gradient steps/task in 2025. He is the one jumper whose +1.34 is far larger than everyone else's, which is the residual the engine swap does **not** explain. **[INF]** Treat "engine + TTT" as the leading hypothesis for #2, and note that ARC-AGI-3 has no demo pairs, so the only label-free gradient signal available is the **transition log our own agent has never queried**.

---

## 1. WHAT ACTUALLY HAPPENED — the leaderboard forensics

Three full-leaderboard CSVs were diffed team-by-team (`lbdiff.py`, `lbdiff2.py` in scratchpad).
Snapshots: **08-11T11:24Z (n=2229)**, **08-13 (n=2263)**, **08-15T14:21Z (n=2331)**. All **[V]**.

### 1.1 The distribution moved at the top only

| snapshot | max | p99 | p95 | median | #≥1.7 | #≥1.6 | #≥1.5 | #≥1.33 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 08-11 | 1.86 | 1.50 | 1.30 | 0.25 | 1 | 9 | 23 | 86 |
| 08-13 | 2.52 | 1.50 | 1.30 | 0.25 | 2 | 12 | 26 | 93 |
| **08-15** | **2.70** | **1.58** | **1.33** | **0.25** | **7** | **20** | **41** | **120** |

Median is unchanged at 0.25. The bulk did not move; the top 5% did. **[V]**

### 1.2 It is NOT a host rescore, and this is proven

The decisive control: **1,919 teams made no new submission between 08-13 and 08-15. Exactly ZERO of them changed score.** The same test on the 08-11→08-13 window: 1,921 teams, zero changes. **[V]**

A scoring-formula change, an added eval game, or a leaderboard recompute would have moved teams that did not submit. None moved. **The host did not change anything.** Every gain is a real new submission.

Second control, and it is ours: **we submitted at 08-15 00:07 — inside the window — and stayed at exactly 1.33.** **[V]** We got no free lift.

### 1.3 The gains are concentrated, and the jumpers are individually huge

| # | team | 08-13 | **08-15** | Δ | new subs | last submission (UTC) |
|---:|---|---:|---:|---:|---:|---|
| 1 | cstl | 2.52 | **2.70** | +0.18 | 2 | **08-13 20:08** |
| 2 | **Daniel Franzen** | 1.24 | **2.58** | **+1.34** | 3 | 08-14 21:37 |
| 3 | **Nikita Sorokin** | 1.33 | **2.10** | **+0.77** | **1** | 08-14 19:30 |
| 4 | **Yusaku Muroya** | 1.25 | **1.98** | **+0.73** | **1** | 08-15 02:36 |
| 5 | **AbeLincoln1865** | 1.07 | **1.90** | **+0.83** | 3 | 08-15 00:22 |
| 6 | YUTO KOJIMA | 1.86 | 1.86 | **0.00** | 3 | 08-15 00:00 |
| 7 | **MLRush** | 1.46 | **1.75** | +0.29 | 4 | 08-15 00:01 |
| 8 | Andy liu | 1.69 | 1.69 | 0.00 | 0 | 08-03 |
| 15 | Tufa Labs | 1.62 | 1.62 | **0.00** | 3 | 08-14 20:19 |
| 19 | Helmut AGI | 1.61 | 1.61 | **0.00** | 3 | 08-15 05:03 |
| 22 | Jack Cole | 1.33 | **1.59** | +0.26 | 1 | 08-15 03:17 |
| **119** | **Canivel (us)** | 1.33 | **1.33** | **0.00** | 1 | 08-15 00:07 |

**[V]** Aggregate: teams that submitted gained mean **+0.1024** (08-13→08-15) vs **+0.0572** in the prior 2-day window; teams with a ≥0.5 gain went **11 → 23**. The churn roughly doubled. **[V]**

### 1.4 The scores above 1.69 do NOT cluster — so it is not a shared public fork

Below 1.65 the board is full of duplicate values (1.50 ×6, 1.46 ×5, 1.61 ×4, 1.58 ×4, 1.49 ×4) — the signature of many teams running one shared artefact. **Above 1.69 every score is distinct: 2.70, 2.58, 2.10, 1.98, 1.90, 1.86, 1.75.** **[V]**
**[INF]** These are independently-tuned systems, not N forks of one notebook. A shared *component* (the engine) is compatible with this; a shared *notebook* is not.

### 1.5 Correction carried forward from `runs/lb_ground_truth.md`

**cstl did not "enter" at 2.52.** cstl sat at **1.59 from 08-04 to 08-09** — inside the dense duck band — then **1.59 → 2.52 in one submission (08-11 18:25) → 2.70 (08-12 20:02) → flat for three days.** **[V]** cstl is a band team that found a step, on an artefact family we also run. That means the duck family's ceiling is **≥ 2.70**, which independently refutes the 1.26–1.36 "efficiency ceiling" as a property of the *family*.

---

## 2. THE CAUSE — Qwen3.8-27B, with a timeline that closes

### 2.1 The release **[V-ext]**

Alibaba released **Qwen3.8-27B open weights, Apache 2.0, on 2026-08-14 at 15:00 UTC (08:00 PT)**, on HuggingFace and ModelScope. 27.78B dense, natively multimodal (text/image/video), 262,144-token native context. Announced 08-03, shipped 08-14.
Sources: [latent.space AINews](https://www.latent.space/p/ainews-qwen-38-max24t-and-27b-new) · [kingy.ai specs](https://kingy.ai/blog/qwen3-8-27b-specs-benchmarks-local-hardware/) · [cryptobriefing](https://cryptobriefing.com/alibaba-qwen3-27b-open-weights-release/) · [aireleasetracker](https://aireleasetracker.com/model/qwen/qwen3.8-27b)

**It is NOT on Kaggle Models.** The `qwen-lm` official org has no Qwen3.8 entry — verified by direct `kaggle models list --owner qwen-lm`. **[V]** So it had to arrive via community dataset mirrors, which is exactly what happened and is what makes the timeline auditable.

### 2.2 The mirrors — this is the load-bearing evidence **[V]**

`kaggle datasets list -s qwen3.8 / qwen3-8-27b --sort-by updated`:

| dataset | size | created (UTC) | Δ from release | dl | owner LB rank |
|---|---:|---|---:|---:|---|
| `johnlussier/qwen3-8-27b-fp8-hf-snapshot` | 25.35 GB | **08-14 17:48:25** | **+2h48m** | 6 | JohnLussier **#145 @ 1.30** |
| `mustangliu/qwen38-27b-fp8-hf-snapshot` | 25.35 GB | 08-14 20:23:48 | +5h24m | 25 | Mustang Liu **#76 @ 1.40** |
| `saltb0x/qwen3-8-27b-fp8` | 25.35 GB | 08-14 22:55:20 | +7h55m | 30 | Akhil Tolani **#83 @ 1.39** |

Plus two Kaggle **Models**: `overseer66/qwen3-8-27b-nvfp4` (an NVFP4 quant) and `trailblazeranemo/qwen3-8-27b`. **[V]**

**All three mirror owners are ARC-AGI-3 leaderboard competitors.** **[V]** `mustangliu`'s README states plainly: *"unmodified mirror of `Qwen/Qwen3.8-27B-FP8` for internet-off Kaggle notebooks (ARC Prize 2026)."* **[V]** These were built for this competition, on release day.

Both `johnlussier` and `saltb0x` mirrors are **complete, servable HF snapshots**: 64 `layers-N.safetensors` + `model.safetensors.index.json` + `tokenizer.json` + `tokenizer_config.json` + `preprocessor_config.json` + `video_preprocessor_config.json` + `chat_template.jinja`. **[V]** Nothing is missing.

### 2.3 The timeline closes

```
08-14 15:00 UTC  Qwen3.8-27B open weights ship (HF/ModelScope)          [V-ext]
08-14 16:53 UTC  Kaggle discussion 735243 "Qwen 3.8 release" opens      [V, 6 comments]
08-14 17:48 UTC  FIRST complete Kaggle FP8 mirror public (johnlussier)  [V]
08-14 19:30 UTC  Nikita Sorokin  1.33 -> 2.10   (+0.77, ONE submission) [V]
08-14 20:23 UTC  2nd mirror (mustangliu)                                [V]
08-14 21:37 UTC  Daniel Franzen  1.24 -> 2.58   (+1.34)                 [V]
08-14 22:55 UTC  3rd mirror (saltb0x / Akhil Tolani)                    [V]
08-15 00:01 UTC  MLRush          1.46 -> 1.75   (+0.29)                 [V]
08-15 00:22 UTC  AbeLincoln1865  1.07 -> 1.90   (+0.83)                 [V]
08-15 02:36 UTC  Yusaku Muroya   1.25 -> 1.98   (+0.73)                 [V]
--------------------------------------------------------------------------
08-12 20:02 UTC  cstl 2.52 -> 2.70   *** 43 HOURS BEFORE THE RELEASE *** [V]
08-11 18:25 UTC  cstl 1.59 -> 2.52   *** 3 DAYS BEFORE THE RELEASE ***   [V]
```

**Every jumper is downstream of the mirror. cstl is upstream of the release and is therefore a separate phenomenon.** **[V]**

### 2.3b The participants say so themselves — direct testimony **[V]**

Kaggle discussion **735243 "Qwen 3.8 release"** (opened 08-14 16:53 UTC, 1h53m after the drop, 6 comments):
- **Ya Xu, 08-15 07:10:** *"It's already achieving a consistent **2× score on the local 25 dataset**. I'm actually surprised because DeepSeek-V4-Flash seems only slightly better than Qwen-3.6 27B 8Bit, but **Qwen-3.8 27B 8Bit is significantly better**."* **[V]**
- **Ya Xu, 08-15 08:59:** *"I don't think 3 × 2.0+ score in 12 hours is a coincidence. The 1.86 has been there for quite a while."* **[V]**
- **Ravindra, 08-15 13:05:** *"**Cstl had already scored 2.70 before Qwen3.8-27B was released.**"* **[V]** — an independent participant reaching the identical carve-out.

Discussion **735381 "Too many High Score"** (FOYSAL, 08-15 12:34): the field noticed the explosion; the sole reply attributes it to Qwen3.8. **[V]**

**"A consistent 2× on the local 25" is the only quantitative claim about this engine anywhere in the public record.** It is one participant's unaudited report on a local harness, not a leaderboard result — but it is a *measured* number from someone who ran it, and it is the right order of magnitude to explain +0.73 to +1.34.

### 2.3c The host did NOT change anything — refuted a second, independent way **[V]**

Full forum enumeration across 5 sort orders: the **most recent host announcement is 2026-06-24** ("Clarification on deadline for milestone prizes") — 52 days ago. No rule change, no scoring change, no new eval games, no reset. Greg Kamradt's only activity in the window is two one-line thank-yous. Deadline unchanged at 2026-11-02. **[V]**
This corroborates the §1.2 rescore test by a completely different route.

### 2.4 Honest weaknesses in this account — stated, not buried

- **Sorokin's 1h42m.** He submitted 1h42m after the first public mirror. The duck harness runs ~2h21m (yw8837's public ledger). The run must precede the submit, so he cannot have used `johnlussier`'s mirror. **[INF]** He mirrored the weights himself in parallel — entirely feasible for someone watching the release — or his harness is materially faster than the duck's. Either way the *release* still precedes him by 4h30m. This does not break the account but it is the tightest joint in it.
- **`LastSubmissionDate` is the last submission, not necessarily the best-scoring one.** For Sorokin (5 subs @1.33 → 6 subs @2.10) and Muroya (70 @1.25 → 71 @1.98) the last submission *must* be the scoring one, so those two are airtight. For Franzen (+3 subs) and AbeLincoln (+3 subs) the best could be up to 3 submissions earlier. **[V/INF]**
- **Many sophisticated teams submitted after the release and did NOT move**: Tufa Labs (08-14 20:19, flat 1.62), Helmut AGI (08-15 05:03, flat 1.61), Lord Han Solo (08-15 03:52, flat 1.65), GeniusYY, BambooCopter, Mathurin Ache, YUTO KOJIMA (3 subs, flat 1.86). **[V]** **This is the strongest counter-evidence and must not be waved away.**
  **It has a good answer, and the answer is a number: the three mirrors had ~61 downloads *combined* in the first 18 hours** (6 + 25 + 30). **[V]** Almost nobody has the weights yet. A 25.3 GB dataset attach is a slow, deliberate act; the pack re-submits daily out of habit with the engine it already has attached. **This converts the counter-evidence into a dated, falsifiable prediction — see §8.**
- **We have not observed a single scored submission that we know serves Qwen3.8.** **Zero public kernels attach these datasets** — verified across 37 pulled `kernel-metadata.json` files covering every public kernel run since 08-11; ~30 of 37 still attach `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`. **[V]** The engine→score link is **circumstantial timing plus participant testimony, not attribution.**

**Confidence that Qwen3.8 is the principal driver of the 08-14/08-15 cluster: HIGH (~85%).**
**Confidence that it explains Franzen's 2.58 on its own: LOW (~25%).**
**Confidence that it explains cstl: ~0% — it is chronologically impossible.**

---

## 3. DANIEL FRANZEN — the lead, run down

### 3.1 Identity: CONFIRMED, by direct read, not inference **[V]**

`kaggle kernels list --user dfranzen` returns exactly two notebooks:
- **`dfranzen/arc-prize-2024-solution-by-the-architects`** — 141 votes. Cell 1: *"This notebook contains our winning submission to the ARC Prize 2024 Kaggle competition, scoring 53.5 points on the private evaluation set. the ARChitects (Daniel Franzen and Jan Disselhoff)."* **[V]**
- **`dfranzen/arc-prize-2025-by-the-architects`** — 15 votes. **[V]**

Corroborated externally: [da-fr.github.io](https://da-fr.github.io/) (PhD candidate, JGU Mainz), [github.com/da-fr](https://github.com/da-fr) (`arc-prize-2024`, 189★, profile links Kaggle `dfranzen`), [huggingface.co/da-fr](https://huggingface.co/da-fr). **[V-ext]**

**This is the most credentialed ARC entrant on the board and he is now #2.**

### 3.2 What he actually does — read from his own code, not from the paper

**2024 (1st place, ARC-AGI-1, 53.5%)** — `kernel-metadata.json`: T4×2, `enable_internet: false`, model `dfranzen/wb55l_nemomini_fulleval` (Mistral-NeMo-Minitron-8B), dataset `dfranzen/unsloth-2024-9-post4` (offline unsloth wheelhouse). **[V]**
From the driver cell **[V]**:
- `train_epochs = 4`, LoRA fine-tune **on the eval set's own demonstration pairs**, in-kernel, across both GPUs (`ds.remove_replies()` → `ds.augment(tp=True, rot=True, perm='rnd_all', shfl_ex=True)`). lr 1e-4, embedding lr 1e-5, adamw_8bit, cosine.
- Inference with `min_prob=0.17`, D8 + colour-permutation + example-shuffle augmentation.
- `use_aug_score = True` → `calc_augmented_scores` → `run_selection_algo(score_full_probmul_3)` — **product-of-experts candidate selection over augmented views.**
- Notably: **`prime_on_single_task = False`** — the per-task `Retrainer(n=32, ...)` exists in the code but was **switched OFF** in the winning run. The TTT was joint over the eval set, not per-task. **[V]**

**2025 (2nd place, ARC-AGI-2, 16.53%)** — `model_sources` are **`dfranzen/lladamix1400k-...-s175k-4bit`** and a companion `...-size230k-4b`. **"LLaDA" = a masked *diffusion* language model.** **[V]** From the driver **[V]**:
- `dispatch_to_workers(gpus=4, ..., ttt_steps=128, ...)` with the comment **`# single task ttt`** → **128 gradient steps of test-time training per task**, 4 GPUs, one task at a time.
- A separate *size model* runs first (`ttt_steps=0`) to predict the output grid shape with a confidence, and tasks are then **ranked by confidence and scheduled** into the remaining budget (`max_infer_tasks`).
- Custom `llada_calculate_loss` with per-example random masking.

External corroboration of the 2025 report **[V-ext]**: [ARChitects 2025 technical report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) — LLaDA-8B with 2D RoPE, LoRA r512 pretrain (175k steps), **test-time finetuning 128 steps/task at LoRA r32**, 102 recursive refinement steps. [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis).

**A detail worth flagging:** their 2025 "what didn't work" section reports they tried **synthetic data generation using Atari-like game screens** — "promising conceptually but lacked scalability." **[V-ext]** That is ARC-AGI-3-shaped work, attempted a year ago.

### 3.3 What he has published about 2026: **NOTHING** **[V]**

- Zero ARC-AGI-3 kernels. Zero 2026 datasets (latest: 2025-10-27). Zero 2026 Kaggle models. **[V]**
- Zero arXiv entries in 2026 for Franzen/Disselhoff/Hartmann. Zero GitHub activity beyond his site (Apr 2026). Zero HF uploads since Jun 2025. No X account exists. **[V-ext]**

**He is running dark, exactly like cstl.** **[INF]** Note that ARC-AGI-3 weights may be attached as a **private** dataset or model, so the absence of a public artefact is *expected* and is not evidence of absence of a fine-tune.

### 3.4 The residual — what the engine does not explain

Franzen gained **+1.34**, roughly double the next-largest jump (+0.83). He was at **1.24 — below our 1.33** — for the whole campaign until 08-14. **[V]**
**[INF, moderate confidence]** The most economical reading: he had a strong bespoke agent parked at 1.24 for reasons unrelated to the brain (or was simply not trying), swapped in Qwen3.8 like everyone else, and the combination of *his* selection/augmentation/TTT machinery with the new engine produced a superadditive result. **There is no public evidence for this. It is a hypothesis with a named falsifier** (see §8).

---

## 4. THE OTHER JUMPERS

| handle | who | public ARC artefacts | read |
|---|---|---|---|
| **cstl** (`gatamaz`,`tehnar`) | 2-person. `tehnar` = Vsevolod Stepanov (SPbAU/ICPC, Lux AI S3 116/701 — **competitive agents in simulation**); `gatamaz` has entered no other competition. | **ZERO.** One 11-year-old Theano notebook. | **[V]** Untraced. Chronologically immune to the Qwen3.8 explanation. **The single most important open question on the board.** |
| **Nikita Sorokin** | **[V-ext]** MWS AI / MTS AI, Moscow. arXiv 2604.02340 — *"Not All Denoising Steps Are Equal: Model Scheduling for Faster **Masked Diffusion** Language Models."* **NOT** the Sber/Skoltech person; **NOT** Ivan Sorokin of NVARC. | **ZERO** Kaggle artefacts. | **[V]** 2.10 on his **6th ever submission**. A specialist in making diffusion LMs sample faster. |
| **Yusaku Muroya** | **[V-ext]** Kaggle **Grandmaster**, RF engineer at Murata. Track record in tabular/optimisation/**RL** (CAFA-6, PhysioNet, Santa 2025). | 17 public notebooks, **none ARC**. | **[V]** A long-running grinder: 71 submissions, sat at 1.25, jumped +0.73 on **one** submission. |
| **AbeLincoln1865** | **Nothing found on any platform.** High team-id (late registration). | **ZERO.** | **[V]** 1.07 → 1.90 in 3 submissions. Pure anonymous. |
| **MLRush** | **[V-ext]** "Independent Researcher and Investor at VOID MAIN LAB", Hong Kong. | **ZERO.** | **[V]** Smallest jump of the cluster (+0.29). |

**[INF, speculative — flagged hard]** #2 and #3 are both people whose most recent published work is on **masked diffusion language models** — Franzen *builds ARC solvers out of them*, Sorokin *optimises their sampling schedules*. That is either coincidence or signal. **I have no evidence either used one here**, and a diffusion LM is not servable on the duck's vLLM path without substantial work. Recorded so it can be checked later, not acted on.

---

## 5. RECONCILING WITH OUR OWN CENSUS (08-13)

The census concluded: *"the 1.4–1.62 band serves exactly one brain, Qwen3.6-27B-FP8; 52 teams span 0.00→1.62 on identical weights; the model explains **none** of the variance; harness and agent policy are the entire public variance."*

**That conclusion was correct on 08-13 and is now partially obsolete. Both halves matter:**

- **Still true:** *within a fixed engine generation*, the model explains nothing. 52 teams, same weights, 0.00→1.62. The dense 1.44–1.65 band is still there today, still clustered, still flat — Tufa Labs, Helmut AGI, FOYSAL, Tecnod8, GeniusYY, Lord Han Solo all submitted after the release and all sat still. **[V]**
- **Now false:** the census's implied corollary that *a brain swap is a speculative bet against the public record*. That rested on every public attempt to leave the 27B scoring worse (GPT-OSS-120B 0.16, Gemma-4-26B-NVFP4 0.17, Qwen3.6-35B-A3B 1.04, Qwen3-VL-30B-A3B 1.26). **Every one of those was a swap to a *different or older* family.** Qwen3.8-27B is the **same family, same architecture, same size, one generation newer**. It is the one swap the census never had the chance to evaluate, and it is categorically unlike the four that failed. **[V]**

**The census's method was sound; its data was one day too old.** The lesson is not "the census was wrong" — it is that *"the brain explains nothing"* was a statement about a snapshot of available brains, and a new brain shipped.

**The 122B-A10B-NVFP4 recommendation the census made should be demoted.** It was justified on mechanism + one circumstantial upload. Qwen3.8-27B-FP8 is smaller (25.3 GB vs 83.5 GB), architecturally identical to what we serve, needs no ModelOpt/NVFP4/SM120 gamble, and has a *timing* correlation with five teams jumping. **It strictly dominates the 122B bet on every axis: cost, risk, and evidence.** **[INF, high confidence]**

---

## 6. IS TTT LEGAL AND FEASIBLE HERE?

**Legality: YES, unambiguously.** `learnings/rules_verification_2026-07-28.md` **[V]** — internet is disabled at scoring time (mechanically: the Submit button is inactive otherwise), but *"Freely & publicly available external data is allowed, including pre-trained models."* Training **inside** the kernel is not restricted at all; the 2024 ARChitects winner did exactly that and won. The only binding constraint is the **9-hour wall** (extended from 6h on 2026-05-07 when Kaggle moved H100→RTX PRO 6000).

**Feasibility: the constraint is not legality or VRAM — it is the wallclock, and this is where the ARC-AGI-1 intuition breaks.**

Two structural differences from ARC-AGI-1/2 that must be stated plainly:

1. **There are no demonstration pairs.** ARC-AGI-3 hands you a game and an action interface. The ARChitects' entire 2024/2025 recipe — fine-tune on the eval task's own `(input, output)` pairs — **has no direct analogue here.** Anyone claiming "they're doing TTT" must say *on what data*.
2. **The rail is wallclock-bound, not action-bound.** Independently established twice: Jakob Brüggen's Taaf Anim write-up (Kaggle discussion 734369) — *"Every run in both arms hits the 132-minute wallclock cap. Nothing ends because of an action limit"* **[V-ext]** — and our own animation-arm kill. **Any compute spent on gradients is compute not spent on actions.** The published agentic-TTT method (arXiv 2607.03441, "No Time Like the Present: Agentic Test-Time Training for LLM Agents", 2026-07-03) reports **1.9× total overhead** for **+5.0 on ALFWorld**. On this rail 1.9× overhead ≈ halving the action budget. **[INF]** That trade is very likely negative for us and this is the arm most likely to look brilliant on paper and regress on the LB.

**The one label-free gradient signal that is actually free:** every action produces an exact, self-supervised training example — you took action `a` in state `s`, the environment returned `s'`. **Our own harness exposes `transitions` and our agent has never queried it** (`project_arc_prize_2026` memory). A small dedicated next-frame predictor — grids are small and discrete-colour, this is a tiny model, not the 27B — is minutes of training and does not compete with the policy for the wallclock in the way an in-episode LoRA on a 27B does. **[INF]** No published ARC-AGI-3 work does this: Rodionov (2605.05138, 2607.15439), Tycho (2607.28287) and OPINE-World (2607.01531) all build world models as **Python code via in-context synthesis with zero weight updates.**

**Bottom line on TTT: legal, feasible, and almost certainly NOT what caused the 08-14 jump** (the timing says engine). It remains an interesting second-order lane; it is not the explanation and it is not the priority.

---

## 7. WHAT IS LIFTABLE BY US, ON A FREE RAIL, IN 79 DAYS

Ranked by impact on the gap. Our position: **1.33, rank #119 of 2331. Gold (top-13) = 1.62. Prize (top-5) = 1.90. Leader = 2.70.** **[V]**

### RANK 1 — Screen Qwen3.8-27B-FP8 as a drop-in engine swap. **[V] on compatibility, [INF] on gain**

This is the highest expected value action available and it costs one free Kaggle build.

**Compatibility, verified field-by-field against our incumbent's `config.json`:** **[V]**

| field | Qwen3.6-27B (ours) | Qwen3.8-27B (new) |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | **identical** |
| `model_type` | `qwen3_5` | **identical** |
| hidden / layers / heads / KV heads / head_dim | 5120 / 64 / 24 / 4 / 256 | **identical** |
| `full_attention_interval` | 4 | **identical** |
| `intermediate_size` / `vocab_size` | 17408 / 248320 | **identical** |
| `max_position_embeddings` | 262144 | **identical** |
| `attn_output_gate` / `mtp_num_hidden_layers` / `partial_rotary_factor` | True / 1 / 0.25 | **identical** |
| vision tower depth/hidden/patch | 27 / 1152 / 16 | **identical** |
| `image_token_id` | 248056 | **identical** |
| **quantisation** | `compressed-tensors` W8A8 per-tensor | **`fp8` blockwise e4m3, `weight_block_size [128,128]`, dynamic activations** ← the one real difference |
| size on disk | 35.92 GB | **25.35 GB** (−29%) |

**Chat-template diff (verbatim read):** **[V]**
- **Tool-call block syntax is byte-identical** (`<tool_call><function=NAME><parameter=...>`) → **`--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` carry over unchanged.** This is the single most important compatibility fact and it is verified, not assumed.
- `preserve_thinking` **is present** and its default **flipped to true** (`preserve_thinking is undefined or preserve_thinking is true or ...`). Our harness passes it explicitly; it is still honoured. *(Contrast the 122B candidate, where it would have silently no-op'd.)*
- **NEW:** a `reasoning_effort` parameter — `xhigh` (**default**) / `medium` / `low`, injected as a system-prompt instruction, and it **raises an exception** on an unrecognised value (fails loud, not silent).

**Gates (free Kaggle BUILD rail only, per `feedback_arc_zero_budget`):**
- **Step 0.** Fork the frozen `arc3-baseline.ipynb`, run `scripts/preflight.py`, use a **fresh kernel slug**. Never hand-build (`feedback_arc_kernel_structural_drift`: 5 ERRORs).
- **Step 1.** Swap `dataset_sources`: `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` → **`saltb0x/qwen3-8-27b-fp8`** (prefer this one: complete snapshot, most downloads, and the owner is an LB competitor who presumably got it working). Keep `enable_gpu`, `enable_internet: false`, `machine_shape: NvidiaRtxPro6000`, and the **same pinned `docker_image` sha** (`feedback_kaggle_env_match`, 5× confirmed).
- **Step 2. GATE 1 — does vLLM come up?** The named risk: the duck wheelhouse pins **vLLM 0.19.0 / transformers 4.57.6**, and this config was written by **transformers 5.8.0.dev0**. Blockwise FP8 has been supported in vLLM since ~0.6 so the quant path should be fine, but the config version gap is the real unknown. **Fallback already exists and is public: `jcole75/arc3-qwen36-runtime-wheels` (vLLM 0.24.0, transformers 5.13.0, CUDA 13.3, built for this exact card).** Note Jack Cole gained +0.26 on 08-15 03:17 — coherent with him having the modern runtime. **[V]**
- **Step 3. GATE 2 — throughput.** Measure actions per window. 25.3 GB vs 35.9 GB of weight traffic per token predicts this should be **faster** than the incumbent. **[INF]** Bar: must not regress vs the frozen fork.
- **Step 4. GATE 3 —** local harness vs frozen fork, same seeds, `runs/null10`, `scripts/phase1_gate.py`, against the promotion bar re-read live from `runs/ledger.json` (it drifts — never cache it).

**Cost: zero.** Public dataset, free build allowance.

### RANK 2 — Treat `reasoning_effort` as a separate, preregistered arm. **[V] the knob exists; [INF] the direction**

The default is `xhigh`, described as *"think carefully… validate key assumptions, consider plausible alternatives."* On a **wallclock-bound** rail that is a tax paid in actions. `low` is *"keep your thinking brief… moving directly to the conclusion."*
**Do not bundle this with the swap.** Two changes at once is exactly how the campaign has burned draws before. Swap first at default; then A/B the knob. **[INF]** Given every efficiency finding we have, `low` or `medium` is the more likely winner, but that is a prediction, not a finding.

### RANK 3 — Query the transition log. **[V] the log exists and is unused**

Independent of everything above and independent of any TTT ambition. The harness exposes `transitions`; the agent never reads it. Our own root-cause finding for the capability gap was that **the agent FORGOT**. This is a prompting/context change with no compute cost, and it is a prerequisite for any world-model or gradient work later. **The cheapest item on this list.**

### RANK 4 — DEMOTE the 122B-A10B-NVFP4 screen

Superseded by Rank 1 on every axis (§5). Do not spend the build allowance on it while Rank 1 is unscreened.

### RANK 5 — cstl remains the real prize, and remains untraced

2.70, banked before the release, flat for three days, zero public artefacts across both handles. `tehnar`'s pedigree is **competitive agents in simulation** (Lux AI S3). **[V]** Nothing is liftable. **[INF]** If Qwen3.8 lifts the field to ~2.0–2.6, cstl's 2.70 stops being a 2× outlier and becomes a normal top score — which would suggest cstl found *early* whatever the new engine now gives everyone. That is a testable prediction (§8).

---

## 8. IS OUR CURRENT STRATEGY OBSOLETE? — blunt answer

**Partially, and in a specific way that is good news.**

**Obsolete:** the working assumption that *the engine is settled and the only lever is harness/agent policy*. That was correct for two months and stopped being correct at 15:00 UTC on 2026-08-14. Every lane premised on squeezing more out of Qwen3.6 — compaction, suppressors, animation, efficiency reframes — is now being run against an engine that five teams have already left behind. **The 122B lane should be parked immediately.**

**Not obsolete:** the *method*. The census method found this in one session. The prereg/gate discipline is exactly what stops us from panic-shipping an unscreened swap. The "audit the instrument" reflex is what caught that the "everyone jumped" framing needed a rescore test — and the rescore test is what turned a vague narrative into a dated, falsifiable timeline. **The instruments are fine. The world moved.**

**The honest strategic read:** we are #119 with a byte-unchanged 1.33 while the top of the board re-based on a free artefact that we can attach in one line. **We are not behind on research. We are behind on a supply-chain event**, and that is a far cheaper problem than the one we thought we had on 08-13.

### What would REFUTE this report — name the falsifiers now, before the data lands

1. **We screen Qwen3.8 and it does not beat the frozen fork.** Then the engine is not the driver, the five jumps need another explanation, and §2 collapses. **This is the primary falsifier and it resolves in one free build.**
2. **THE DATED PREDICTION — the 1.44–1.65 wall migrates upward within 2–5 days.** The three mirrors had ~61 downloads combined in 18h **[V]**; as they propagate, the ~200-team dense pack should start moving. **If by 2026-08-20 the 1.5–1.65 band is still flat and the top is still exactly these seven teams, the engine explanation is wrong and the jumpers did something else.** This is free to check: one leaderboard download per day, which the daily loop already does.
3. **A team that demonstrably has NOT swapped engines posts ≥1.9 in the next few days.** Would show the jump is technique, not supply chain.
4. **cstl's 2.70 is matched or passed by several Qwen3.8 teams within a week.** Would *support* the reading that cstl found the same capability early by other means.
5. **A public kernel appears attaching a Qwen3.8 mirror with a scored result.** Converts §2's circumstantial timing into direct attribution — in either direction.
6. **Franzen posts a write-up.** He has published a full solution notebook after each of the last two ARC Prizes. **[INF]** He is likely to do so again, and it will be the single most informative artefact of the competition.

### Two free findings from the sweep, unrelated to the engine but worth banking

- **Jason Feng, discussion 734843 (08-12):** he measured that **66.8% of Qwen tool-call responses in the Tufa Duck harness return hidden reasoning with ZERO visible content**, across all 25 games. **[V-ext]** This lands directly on our own root-cause finding that *the agent FORGOT* — it suggests a large fraction of the model's work never reaches the transcript the next turn reads. Cheap to verify against our own logs, and it is a *harness* bug, not a capability limit.
- **Xuan (LB 1.52), comment on 734369:** flags [vista-research.github.io](https://vista-research.github.io/) — upscale the 64×64 grid to 512×512 to exploit VLM vision priors; reports 100% for Claude/Sol but that **Qwen "cannot accurately infer coordinates based on image."** **[V-ext]** Note we already run `MULTIMODAL_UPSCALE=4`; this argues the upscale factor itself is a tunable worth re-examining **on the new engine**, whose vision tower is byte-identical but whose language model is not.

### The discipline note this campaign has earned

This report asserts a causal story from **timing correlation plus mechanism plausibility**. It has **no** direct attribution: not one scored submission is verified to serve Qwen3.8. The counter-evidence in §2.4 — seven strong teams submitting post-release and not moving — is real and is not explained away. **This has been called at ~85% confidence, not 100%, and the falsifier is one free build.** Two hypotheses have hardened into belief on this campaign already. This one is written to be killed cheaply.

---

## Appendix — artefact index

| artefact | why it matters |
|---|---|
| `saltb0x/qwen3-8-27b-fp8` | 25.35 GB, complete FP8 HF snapshot, 08-14 22:55Z, 30 dl. **Recommended swap target.** Owner Akhil Tolani #83. |
| `johnlussier/qwen3-8-27b-fp8-hf-snapshot` | First complete mirror, 08-14 17:48Z (+2h48m from release). Owner JohnLussier #145. |
| `mustangliu/qwen38-27b-fp8-hf-snapshot` | 08-14 20:23Z. README explicitly states "for internet-off Kaggle notebooks (ARC Prize 2026)". Owner Mustang Liu #76. |
| `overseer66/qwen3-8-27b-nvfp4` | Kaggle **Model**, NVFP4 quant of the same weights. Blackwell-native path if FP8 disappoints. |
| `jcole75/arc3-qwen36-runtime-wheels` | vLLM 0.24.0 + transformers 5.13.0 + CUDA 13.3, "built for RTX PRO 6000 Blackwell". **The escape hatch if gate 1 fails on the pinned 0.19.0 wheelhouse.** Owner Jack Cole, who gained +0.26 on 08-15. |
| `dfranzen/arc-prize-2024-solution-by-the-architects` | The 2024 grand-prize notebook. In-kernel TTT + D8/colour augmentation + product-of-experts selection. |
| `dfranzen/arc-prize-2025-by-the-architects` | The 2025 2nd-place notebook. **LLaDA masked diffusion + `ttt_steps=128` per task + shape-model scheduling.** |
| arXiv **2607.03441** | "No Time Like the Present: Agentic TTT for LLM Agents" — in-episode LoRA on own trajectory, vLLM runtime LoRA API, **1.9× overhead for +5.0 ALFWorld**. The honest price tag on agentic TTT. |
| Kaggle discussion **735243** | "Qwen 3.8 release", opened 08-14 16:53 UTC, 6 comments. The community timestamp on the event. |
| Kaggle discussion **735381** | FOYSAL, "Too many High Score", 08-15. The field noticed and has no explanation. **We now do.** |
| `cstl` / `gatamaz` / `tehnar` | Zero artefacts. 2.70 banked 43h before the release. **Still the one thing on this board nobody can explain.** |

**Scratchpad working files:** `lbdiff.py`, `lbdiff2.py`, `nbdump.py`, `f24.txt`, `f25.txt`, `newnames/`, `q38/` under
`C:\Users\dcani\AppData\Local\Temp\claude\f--kaggle\62c35e7c-0d05-4da2-99b0-f9b400a45a97\scratchpad`.
