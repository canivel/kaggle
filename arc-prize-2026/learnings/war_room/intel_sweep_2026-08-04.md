# Intel Sweep — 2026-08-04 (comprehensive, per USER ORDER "we are stuck")

Method: kaggle CLI (LB CSV 11:28Z, 2048 teams; kernels score-sort), chrome-devtools MCP
(discussion feed + 8 threads, anonymous session), WebSearch/WebFetch (arXiv, arcprize.org,
model news), gh CLI (repo forensics). Dedup baseline: discussions_2026-07-26.md /
research_2026-07-26.md + standing exclusion set. Read-only; no pushes, no submissions, no spend.

Our state: **1.33 @ #70** (was #51 on 07-28 — pure drift, banked draw intact). Head KOJIMA 1.86.
**Gold cutoff has moved: 1.49 (07-28) → 1.54 (08-01) → ≈1.56 (top-13, 08-04).** 46 teams ≥1.40,
19 teams ≥1.49. The "gap to 1.49" framing is stale; the real gold target is ~1.56 and drifting
+~0.01/day.

---

## RANKED FINDINGS

### 1. HOST SCORING MECHANICS CONFIRMED (thread #729985, Greg Kamradt, ~Jul 27) — ADOPT (strategy)
New-to-set thread. Three host-confirmed facts:
1. **Private-LB scores are computed at each submission's ORIGINAL run time and are NEVER re-run**
   at selection.
2. **Every scored run plays BOTH datasets** (public + private) in the same ≤9h run; the public LB
   shows only the ~50% public-task slice.
3. Wall-clock is exactly 9h ("<12h" language elsewhere applies to the Verified Leaderboard only).

**Why this matters to us more than anyone**: our entire order-stats machine (filler lottery,
E[max@k]) is denominated in PUBLIC max. But the prize outcome = the already-banked PRIVATE score
of the 2 submissions we select on Nov 2. Each daily draw is a **joint (public, private) pair**;
public is only a noisy proxy for the private twin (correlated through same-run luck, ρ<1). Two
consequences: (a) daily cadence still pays — every filler also banks a private draw; (b) **capability
lifts pay double** (they shift the whole joint distribution), while public luck pays nothing at
selection if its private twin is mediocre. Selection risk is real: our banked 1.33's private twin
is fixed and unknowable. **Verdict: ADOPT — re-denominate window pricing in joint-draw terms; keep
grinding cadence; weight capability lanes (EWM, compaction) above further luck-harvesting.**
Impact on gap-to-gold: indirect but foundational — it says the gap must be closed by capability,
not draws. (Route to: main, rules-check, eval-cycle-0728.)

### 2. TYCHO IS PUBLIC (arXiv 2607.28287, Jul 30 + github.com/NIMI-research/Tycho, Apache-2.0) — ADAPT (high)
Lehmann/Aioanei/Vahdati (NIMI research). **Strongest public ARC-AGI-3 result ever: 100.0 RHAE,
all 183/183 levels** with Opus 5 (6,641 actions, 61% fewer than human baseline) and GPT-5.6 Sol —
official competition-mode scorecards linked in repo. Repo created Jul 29 (9 stars, 1 fork).
Architecture (all liftable ideas):
- **Typed frame roles**: every grid tagged decision / transient(animation) / completion-terminal /
  fatal-terminal / reset-init / next-level-init. Separates actionable observations from animation
  noise before the model reasons.
- **EWM loop**: executable Python hypothesis (`State`, `transition`, `render`, `outcome`) →
  replay-verify against recorded history → repair on falsification.
- **KEY ABLATION (Opus 4.8, fixed model): no-WM 79.07 → single actor model 85.36 →
  actor-controlled builder 88.49; falsification-AUTO-triggered builder is WORSE (83.07).**
  The ACTOR deciding when to invoke the builder beats automatic triggering. Direct design input
  for our EWM Stage-1 (due today): make builder invocation an actor tool-call, not a harness rule.
- **Level-boundary compaction**: conversation cleared at each level boundary; a consolidation pass
  writes level summaries; persistent per-game workspace (typed records, actor notes, programs)
  survives. **This is our compaction+retained-reasoning mechanism — 4th independent external
  validation, and the first at 100-RHAE tier.**
NOT liftable wholesale: $97–231/game frontier-API cost, billions of cache-read tokens, Docker
sandbox for agent code (impossible in Kaggle notebook — replace with our in-process exec_wm),
Anthropic/OpenAI transports only (but `tycho/serving/llm_client.py` + `public_backends.py` suggest
an OpenAI-compatible client → likely pointable at localhost vLLM/Qwen3.6-27B; verify).
**Threat side: it is public.** Everyone in the 1.4+ band can mine it. Andy liu's jump to 1.69
landed 08-02, 3 days after release (timing suggestive, unproven). **Verdict: ADAPT — fold typed
frame roles + actor-controlled builder + boundary consolidation into EWM Stage-1 and the
compaction lane NOW; also eval whether Tycho's harness (vLLM transport, exec sandbox swapped)
outright beats duck at 27B on the local bench.** Expected impact: largest capability lever on the
board; their ablation is worth +9.4 RHAE at fixed model — even a fraction closes most of
1.33→1.56. (Route to: exec-reset-day1, compaction-ab, main.)

### 3. LB FORENSICS — gold is a moving target; top-10 are daily grinders
Movement since 07-28 baseline (narrative deltas from runs/lb_ground_truth.md + today's CSV; no
full 07-28 CSV archived — start archiving daily, see Process note):
- **Movers >0.1 / new to top band**: Andy liu → **1.69 #2** (new 08-02); GeniusYY → **1.64 #3**;
  FOYSAL → 1.61 (new 08-01); **cstl 1.59** (new name); **Nkosi Ndwandwe 1.58** (new entrant,
  08-03); Yuchen20 1.58; paul/Seok 1.54 (new 08-01). Head KOJIMA 1.86 unchanged but still
  submitting daily (last 08-04 00:07).
- **Cadence**: 8/10 of top-10 submitted within the last 36h (daily grinders). Occasional-jumpers
  holding stale draws: anngle 1.56 (idle since 07-25), boristown/暗黑AGI 1.47 (07-26),
  **Tshithihi 1.44 idle since 07-03** — a month-old draw still at #34, i.e. the 1.44 "wall" is
  partly fossilized luck, not all active capability.
- **Band math**: ≥1.40 = 46 teams; ≥1.49 = 19; gold ≈ top-13 ≈ **1.56**. Our 1.33 slid #51→#70 in
  7 days (~2-3 ranks/day) with zero change to our score. Extrapolated Nov 2 gold: 1.6+ if drift
  persists.
- Tufa Labs themselves sit #30 @1.45 (their own harness tier). yw8837's public 1.17 fork → many
  sub-1.2 clones churning the low band.
**Verdict: ADOPT (posture) — plan against gold ≈1.56-1.65 by deadline, not 1.49.**

### 4. yw8837 FULL LEDGER RELEASE (thread #731522 + notebook/dataset, Jul 27) — ADAPT (calibration)
Same-family duck fork (Qwen3.6-27B-FP8, concurrency 28, 7920s/game): **11 official submissions:
1.29, 1.05, 0.71, 0.73, 0.68, 0.75, 0.55, 1.11, 1.17, 0.90, 0.94** (mean 0.899, s≈0.23) — an
independent replication of LB-as-lottery, with WIDER σ than our frozen ledger (0.155). His
explicit conclusion: "more local actions, tokens, or public-game levels did not reliably improve
the hidden score." Patches in his 1.17: (1) repeated-no-effect cardinal-direction guard within a
batch → stop and re-observe (≈ our (c)/(d) suppression, already adopted); (2) analyzer yield
window 60→90s. Dataset: 300 per-game diagnostic rows + SHA-256-pinned hypotheses
(kaggle.com/datasets/yw8837/arc-agi-3-run-history-300-game-diagnostics).
**Verdict: ADAPT — pull the diagnostics dataset into the LB process model as an external σ
datapoint; ignore the patches (already covered).** Impact: sharpens gate pricing, no direct lift.

### 5. GPU-QUOTA SAVER (thread #731290) — ADOPT (immediate, free)
Community-confirmed: the second "save & run" execution that auto-starts on submission **consumes
GPU quota and can be cancelled without affecting the submission, its score, or logs**.
Under the zero-cloud-spend rule (30 GPU-h/wk), our daily submit daemon may be burning ~2× quota.
**Verdict: ADOPT — add a cancel step for the shadow execution to scripts/queue.py / ARCDailySubmit.**
Impact: frees up to ~9 GPU-h/day of headroom → more free build-rail evals for the gates.
(Route to: main, a17-unblock — more screen budget.)

### 6. MODEL LANDSCAPE — no new in-window release beats Qwen3.6-27B, but ONE unscreened wall-closer candidate
- **GLM-4.6V (zai-org)**: 106B MoE (~12B active), **native tool-calling VLM**, vLLM-native recipe,
  128K ctx. AWQ/4-bit ≈55-60GB → **fits the 96GB rail with KV headroom**. Released Dec 2025 —
  not new, but never screened by us; it is the natural post-mortem successor to the dead
  Qwen2.5-VL-72B line (A17). GLM-4.6V-Flash (9B) = free local pre-screen. **ADAPT — add
  GLM-4.6V-AWQ to the A17 model-tier screen queue** (multimodal + native FC addresses both the
  harness-is-multimodal constraint and the hermes-parser failure class from research_2026-07-26).
- **Gemma 4** (Apr 2026, in-set): 26B MoE (3.8B active) and 31B dense, natively multimodal,
  128-256K ctx; Jun/Jul MTP drafters ~3× decode speedup. A Kaggle notebook already runs
  Gemma 4 26B NVFP4 ("xCaliber AA3") — scored **0.17**, i.e. no public evidence it works in-harness.
  Keep as screen candidate, prior lowered.
- **Qwen3.6-35B-A3B** (Apr, Apache-2.0): text-only MoE, 73.4 SWE-bench-V, ~3B active → very fast;
  only viable as the reasoning half of a split-stack (small-VL front-end + text reasoner), echoed
  by Yakunin's forum suggestion (Qwen2.5-VL-7B + Qwen3.5-9B). PARK — split-stack is a new
  architecture lane we haven't priced.
- Greg Kamradt published `gregkamradt/arc-agi-3-gpt-oss-120b` (gpt-oss-120b = 117B MoE, 5.1B
  active, fits 96GB in MXFP4, strong reasoning). No public score attached. **WATCH — cheap to
  screen on the free rail.**
**Verdict: A17 successor queue = GLM-4.6V-AWQ (first), gpt-oss-120b (second), Gemma-4-31B (third).**
Impact: this is the declared wall-closer lane; a working 100B-class multimodal/native-FC model is
the only known mechanism for +0.2-class jumps at fixed harness.

### 7. COMPACTION RESEARCH (in-window) — one ADAPT-low, one IGNORE
- **Addressable Recall Compaction** (arXiv 2607.25066, Jul 27): append-only ID-addressable log of
  tool observations; older observations replaced by citation stubs; agent recalls by ID without
  re-executing tools. 99.40% NIAH vs 88.12% best baseline; tested on Qwen3-8B/32B (our tier).
  Complements retained-reasoning: "recall-by-address" instead of lossy summary for observations.
  **ADAPT-low — candidate mechanism for the compaction A/B's observation channel** (route to:
  compaction-ab).
- CompactionRL (2607.05378): train-time RL for summary generation — **IGNORE** (no training budget,
  zero-spend rule).
- Tycho's boundary-consolidation (finding 2) = the 4th and highest-tier external validation of the
  compaction+retained-reasoning mechanism. The lane is externally de-risked; only our eval is
  pending.

### 8. LOW-PRIORITY / HYGIENE
- **Milestone 2 = Sep 30, judged on PUBLIC LB, top-3, open-source deadline same day 23:59 UTC**
  (pinned #713634). At 1.33 vs top-3 ≈1.64+ we are not in reach; do not divert.
- ARC Prize org: **no blog post since Jul 6**, no new model-eval or harness guidance in window.
- ARC-AGI-3-Agents PR#74 merged (host repo QoL, triggered by the RL toolkit thread). RL toolkit
  `InexperiencedMe/ARC-AGI-3-Playground` — clean numpy env wrapper, IGNORE for score.
- GitHub scan (sorted by update): nothing public beats duck. Curiosities, all ≤1 star:
  `Alexyskoutnev/TWIN-ARC-AGI-3` "Test-Time World-Model Inference" (08-02, WATCH),
  `vyomakesh0728/arc-agi-3-schema-harness` (Gemma 4 E4B), JEPA-style repo (that lane is dead for
  us — 3 strikes). Public notebook ceiling unchanged: boristown 1.47; new publics: Tough Guard V2
  1.18 (08-03), Action7 Shadow 0.67, world-model GPU v1 @1.1 (obirdy, 07-25).
- Scott Le Grand (thread #727119 tail, 3d ago): still no-visibility on submission failures; his
  proposed "simulate 110 private games from 25 public" notebook = essentially our duck_eval — we
  are ahead of the public on this.
- Thread #732706 (100% on task r11l, human recruiting) — IGNORE.

## PROCESS NOTE (self-inflicted gap found during sweep)
We do **not archive daily full-LB CSVs** — today's movement forensics had to lean on narrative
snapshots in lb_ground_truth.md. Fix: `kaggle competitions leaderboard -d` is one call; have
ARCDailyIterate save the zip to `runs/lb_snapshots/YYYY-MM-DD.csv` (~65KB/day). Enables exact
per-team deltas, band-velocity tracking, and new-entrant detection.

---

## TOP 5 (for the war room wall)
1. **Tycho is public** (Jul 30): 183/183 levels @100 RHAE with frontier models, Apache-2.0 —
   lift typed frame roles + actor-controlled builder + boundary consolidation into EWM Stage-1 /
   compaction lanes today; assume the 1.4+ band is reading the same repo.
2. **Host confirmed: private scores bank at original run time, both sets play every run** — the
   prize is the private twin of the two submissions we select; capability lifts pay double,
   public luck pays nothing at selection.
3. **Gold is drifting**: 1.49 → ≈1.56 in 7 days; 46 teams ≥1.40; plan for ≈1.6 by Nov, and our
   1.33 slides ~2-3 ranks/day while we hold.
4. **Wall-closer model queue refreshed**: GLM-4.6V-AWQ (106B MoE, native tool-calling VLM, fits
   96GB) is the strongest unscreened successor to the dead 72B line; gpt-oss-120b (host-published
   notebook) second.
5. **Free quota back**: cancel the auto-spawned save&run twin of each submission — recovers up to
   ~9 GPU-h/day for build-rail evals.

## THE SINGLE MOST LIKELY THING WE'RE MISSING
**We are optimizing the wrong random variable.** The whole grinder/order-stats apparatus prices
PUBLIC E[max@k], but per the host, the payout is the PRIVATE score already banked inside whichever
two runs we select — and private is the larger set, so it compresses luck toward true capability.
Meanwhile the strongest capability blueprint ever published for this benchmark (Tycho) went
open-source five days ago and matches our two live lanes (EWM + compaction) almost
component-for-component, with an ablation telling us exactly which variant to build
(actor-controlled builder, typed frames, boundary consolidation). The miss isn't a secret
technique in the 1.4+ band — it's that the fastest route to a better private twin is sitting in a
public Apache-2.0 repo while we spend windows on draw mechanics.
