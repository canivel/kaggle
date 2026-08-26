# ATTRIBUTION DEEP-DIVE — 2026-08-21: what moved 1.33 → 1.59, and what stands between 1.59 and the 2.5 tier
**Order (principal):** "dive deep on what improved and why, and push harder."
**Subject:** submission **55656892** (`canivel/arc3-q38-field-eval` v1 — byte-faithful FOYSAL rebase: Qwen3.8-27B-FP8 repacked Kaggle Model × effort unpinned ⇒ template `xhigh` × Brüggen 08-07 TAAF anim bundle (animation NOT enabled) × no grafts) → **1.59 public, rank 205/2450, campaign best** [V: `kaggle competitions submissions` + fresh LB pull this session].
**Method:** read-only. Kernel output + logs pulled and parsed (benchmark.json per-game); compared against all 23 informative 25-game local runs in `runs/kernel_pulls/`; fresh LB download 08-21; targeted kernel/discussion reads only (no re-sweep). Tags: **[V]** verified this session · **[V-doc]** verbatim claim in a verified artifact · **[INF]** inference · **[UNK]** unknown.

---

## §1 — WHAT MOVED, MECHANICALLY [V]

Full-census comparison (every `runs/kernel_pulls/*/benchmark.json`, 25-game runs only):

| run (date) | lc_total | max per-game lc | total actions | actions/level | mean_score |
|---|---|---|---|---|---|
| **q38_field_v1 (08-20)** | **28** | **7** (sb26) | **1639** | **58.5** | **6.173** |
| q38_v2 = Q38-at-medium, June-30 harness (08-16) | 21 | 4 | 2857 | 136 | 2.795 |
| q38_engine_v3 (08-17) | 17 | 4 | 3127 | 184 | 2.906 |
| war_eval_v1 — prior lc record | 22 | 2 | 3638 | 165 | 1.579 |
| Q3.6 gate family (gate_eval v1/v2) | 18/19 | 2 | 4757/4033 | 212–264 | 1.427/1.939 |
| graft_floor_v1 (08-19) | 18 | 3 | 3257 | 181 | 2.303 |

Prior-campaign bands across all 23 runs: lc_total 10–22, max per-game lc 2–4, actions 2857–5162, mean_score 0.85–3.42. The field-floor run breaks every band simultaneously [V].

**Three separable mechanical changes, in order of size:**

1. **One deep run: sb26 went 4 → 7 of 8 levels (77.78 pts vs prior best 23.65).** sb26 alone contributes **2.17 of the +3.38** local mean-score gain over the best comparator (64%) [V]. This is the "xhigh completes multiple levels in one run" signature Ya Xu described in public [V-doc, topic 735243]. **sb26 is now ONE LEVEL from the campaign's first won game** — relevant because banking-transfer's trigger (a win) has been unreachable field-wide (0/25, PRASHANT 08-19 [V-doc]).
2. **Action economy: total actions 1639 = 57% of the prior minimum; actions-per-completed-level 58.5 vs 136–264.** Under the per-level quadratic `min(115,(baseline/actions)^2*100)`, halving actions quadruples the per-level score. This shows up in every scoring game, not just sb26 (e.g. sc25: 2 levels in 37 actions = 14.29, vs 2 levels in 355 actions = 6.25 for the Q3.6 gate) [V]. Note the same 2h13m wallclock produced HALF the actions — xhigh thinks longer per action and acts less, and the actions it takes are better (28 levels from 1639 actions vs 21 from 2857) [V for the counts; INF on the "thinks longer" mechanism].
3. **Breadth: 17 of 25 games scored ≥1 level vs 14 in the comparators.** New unlocks vs Q38-at-medium: cn04, ft09, lp85 (1→3), m0r0, tn36, tu93, wa30. Regressions: bp35, cd82, r11l went 1→0 [V]. The persistent zeros stay zero across every config: dc22, g50t, sk48, sp80, tr87 [V].

**Runtime certification held** [V]: served model `Qwen/Qwen3.8-27B-FP8` (repacked Kaggle Model path in the vLLM banner), `reasoning_effort` occurs 0 times in all three logs (⇒ template xhigh), anim-bundle serve config (mml 65536 / ctx 32768 / temp 0.6 / preserve_thinking), 25 games, COMPLETE.

**Attribution caveat (stands from the prereg):** this arm is compound by design — engine × effort × harness-generation × packaging moved together. Against Q38-at-medium-on-June-30-harness (lc 21/17), the isolated delta is effort × harness-generation only; engine is held constant in that comparison. Cleanest single-sentence read: **Q38 at xhigh on the 08-07 harness plays FEWER, BETTER actions — one deep run (sb26), plus a field-wide halving of actions-per-level that the quadratic then pays quadratically** [V for the pattern; INF for the effort-vs-harness split, untested].

## §2 — DECOMPOSING 1.33 → 1.59 ON THE PUBLIC BOARD

- 1.33 was the **max of 37 frozen-fork draws** (ledger: μ=0.9316, σ=0.1771) [V]. Sanity check: simulated E[max of 37] at those parameters = **1.31** — our 1.33 is exactly what best-of-37 of that distribution should produce [V, simulation this session]. The old number was draw-count, fully.
- 1.59 is a **single draw** of the new config. Two independent anchors cohere on the new config's single-draw distribution: (a) our 1.59 = 1 draw; (b) Kunal Desale's near-byte replicate: **1.91 = max over 10 lifetime subs** [V board]. If σ_new ≈ σ_old (0.18–0.21; same rerun rail and nightly noise), Kunal's max-of-10 implies **μ_new ≈ 1.59–1.63** — landing exactly on our observed draw [INF, two-point fit]. Working model: **new-config draws ≈ N(μ≈1.6, σ≈0.2)**.
- Consistency with the local read: μ_new/μ_old = 1.72×. Local lc ratio 28/19.5 = 1.44×; local mean-score ratio 1.8–2.2×. The LB gain sits between the lc ratio and the score ratio — consistent with the quadratic LB metric plus partial (not full) reproduction of the sb26-class deep run on hidden games [INF]. **Residual: ≈ none.** The +0.26 over the old max is a +3.7σ single-draw jump vs the old distribution — the same signature as the board's risers, now reproduced on our own row and explained by config, not luck [V for the jump; INF for the mechanism]. This CONFIRMS the standing thesis: **the >1.70 band is configuration, not draw count.**

## §3 — THE 0.64 GAP TO FOYSAL'S 2.23

- FOYSAL: **2.23, rank 28 today, 91 subs** [V]. How many subs were THIS config is [UNK]; the kernel's run window (08-11→08-18) suggests ~5–15 config-draws [INF]. "The notebook scored 2.23" is [V-doc] (their claim) — board Score is best-over-all-subs and cannot date/attribute a run (proven previously on our own row).
- Expected max of k nightly redraws at N(1.59, 0.20) [V, simulation]: k=5 → **1.82** · k=10 → **1.90** · k=14 → **1.93** · k=30 → **2.00** · P(≥2.23) ≈ 2% by k=30. Under a heavier multiplicative-noise tail (lognormal cv 0.2 — motivated by the field's reported 30–50% nightly swings), k=10 → E[max] 2.13 with P(≥2.23) ≈ 30%, k=30 → ≈ 66%. Our 37-draw frozen history fits both tails, so the truth is bracketed:
  - **Redraw alone reaches ~1.9 in 1–2 weeks with high confidence** [INF, robust across tail models].
  - **2.23 by redraw alone: between "essentially never" and "coin-flip in a month"** — real but not bankable [INF].
  - **2.5+ by redraw alone: unreachable** (z ≈ +4.5 normal; even the heavy tail needs ~a year) [INF, strong]. FOYSAL's own 2.23 may itself be μ≈1.6 + tail draws + possible private deltas above the public v12 [UNK].

## §4 — WHAT THE 2.5+ TIER HAS THAT THE 1.59 FLOOR DOESN'T

Board census 08-21 [V]: 12 teams ≥2.5; 64 teams ≥2.0; we are 205th at 1.59. Decisive new fact: **the 2.5 band contains fresh, low-sub teams — AbeLincoln1865 2.72/12 subs, Jonathan Wang2022 2.59/10, rellik13 2.53/8, Pathetic384 2.47/5, Cyrus 2.43/3, WENJIE 2.39/3 — all scored in the same 08-20/08-21 window as our 1.59, and NONE has a public ARC kernel** [V: per-user kernel listings]. A 3-sub team at 2.4+ means its config's single-draw mean is ≈2.3–2.6, not a lucky max [INF, near-arithmetic]. Meanwhile the best public kernel by score is still FOYSAL's 2.23-class artifact [V: scoreDescending census] — **the 2.5 recipe is not on the public kernel graph.**

Candidate mechanisms for the 1.59 → 2.5 gap, ranked by evidence:

1. **Newer harness generation / agent-code iteration (STRONGEST).** Tufa Labs — the harness authors — jumped **2.07 → 2.97 on 08-20** [V, LB diff 08-20→08-21]. Jakob's **08-15 Q38 bundle** (`jakobbrggen/taaf-model-20260815-q38-p1`, 90 votes) is public and NEWER than the 08-07 bundle we ran [V]; Kunal's 1.91 ran a newer docker sha [V, rethink]. tool_agent grew 89KB → 108KB from June-30 to 08-07; the 08-15 generation is unmeasured by us [UNK → testable].
2. **Private tuning on top of the Q38-xhigh floor (STRONG but dark).** The low-sub 2.5 teams plus Scott's "3.8 locally, 50% >3" [V-doc] show the config family has headroom above 1.6 that private prompt/agent deltas are reaching. Nothing liftable [UNK].
3. **More draws (REAL but capped).** Explains 1.59 → ~1.9–2.0 only (§3). NOT the tier mechanism — refuted directly by the 3-to-12-sub teams at 2.4–2.7 [V].
4. **Grafts + clickmap + searchmap compound (PUBLIC, MEASURABLE, UNVALIDATED on the board).** tennant **v22 shipped TODAY 00:23** [V, pulled]: adds `searchmap` (combination-lock reframe of switch-like games; prequential 113/113, precision 1.0000 on tn36, available from action 27) to v21's {efficiency, retry_guard, shortcircuit, goalkeep, hudmask, clickmap}. But the distributor's own team sits at **1.42 — BELOW our floor draw** [V]. The grafts' quantified wins are calibration metrics (Brier 0.1414→0.0627), not levels. Moderate-weak as the tier explanation; still the best public candidate for action-economy gains the quadratic pays.
5. **Banking-transfer (DEAD as tier explanation).** Trigger = a won game; 0/25 field-wide [V-doc]. Footnote: our sb26 at 7/8 is the closest any run of ours has come to arming it [V].
6. **cstl 3.57 (DARK).** "Cstl had already scored 2.70 BEFORE Qwen3.8-27B was released" [V-doc, Ravindra 08-15, topic 735243]; members have no relevant public kernels [V]. Their edge predates the engine wave; nothing liftable [UNK].

Discussion delta since 08-19: **zero new topics** [V, topics list]; nothing material added to the effort thread since the rethink's capture [V, full re-read of 735243].

## §5 — PUSH-HARDER LIST (ordered; free-Kaggle rail, ~2h13m GPU per 25-game eval, 2 pushes/day)

- **ARM 0 (policy, zero GPU): make `arc3-q38-field-eval` the DEFAULT nightly queue head — retire the frozen-fork filler.** Every filler night now costs an expected ~−0.65 vs a field-floor redraw (draw means 0.94 vs ~1.6), and redraws are the certain path to ~1.9 within ~2 weeks (§3). The frozen fork keeps its eternal-fallback role only if the field artifact ever ERRORs. EV: +0.3 LB, near-certain. Owner: coordinator/daemon (this session is read-only; nothing queued by me).
- **ARM A (next build slot): harness-generation isolation — rebase onto Jakob's 08-15 bundle class** (`taaf-model-20260815-q38-p1`), engine/effort/packaging held at field-floor values. Single variable vs the 08-20 run; bands re-baselined vs the NEW floor (lc 28 / 6.173 — the old 17–21 bands are obsolete for this lane). Rationale: §4 mechanism #1; the Tufa +0.90 jump on 08-20 is the strongest recent board move with a plausible public artifact behind it. Fold in the docker-sha question by matching the source kernel's metadata exactly (`feedback_kaggle_env_match`). EV: unknown-but-highest; one slot to test.
- **ARM 3 (ALREADY BUILDING — attribution note only, per order).** Tennant v21 compound (Q38 + grafts + clickmap). A SIGNAL vs the new lc-28/6.173 floor attributes graft-mechanics gain on top of the field floor (its shortcircuit/efficiency channel is exactly the actions-per-level economy the quadratic pays). What it CANNOT attribute: v21/v22 attach the **share-fork bundle, not the 08-07 anim bundle** [V, metadata] — a NULL-or-below-floor result is confounded harness-generation × grafts and must NOT be read as "grafts dead". It also carries zero board validation above 1.42 going in.
- **ARM B (behind Arm 3): v22 searchmap increment.** Only if Arm 3 reads at-or-above floor; one-flag delta on the same lineage; source pulled this session. EV small-moderate (tn36-class games), cheap.
- **ARM C (science slot, when free): xhigh-vs-medium isolation on the 08-07 harness.** Completes the REFUTE-2× re-scope; splits §1's action-economy shift between effort and harness; informs every future engine decision. LB EV ~0 short-term.
- **ARM D (staged): visible-memory capture contract, strengthened for Q38** ("you might need to yell at Qwen3.8" — Jason Feng [V-doc]). The field-wide hidden-channel amnesia is untouched by the xhigh wave; a working contract is a genuine differentiator vs the 2.5 tier's private tweaks. Needs the strengthening pass before it earns a slot.
- **WATCH (no slot): sb26 win adjacency.** First config within one level of a won game. If any future draw completes sb26 8/8, the banking-transfer treatment's trigger exists for the first time — re-open that shelved lane THEN, not before (`feedback_verify_treatment_can_fire`).

## Appendix — evidence pulled this session
| artifact | content |
|---|---|
| scratchpad `q38out/` | full kernel output 55656892: benchmark.json (per-game), summary.txt, vLLM log (served-model banner), stdout, transcripts |
| `runs/kernel_pulls/*/benchmark.json` × 23 | full local-run census (table §1) |
| fresh LB zip (scratchpad `lb21/`) | 08-21 board: cstl 3.57/31 subs · Tufa 2.97/113 · 2.5-band sub counts · FOYSAL 2.23/91 · Kunal 1.91/10 · us 1.59 rank 205/2450 |
| scratchpad `pulls/tennant_v22/` | v22 notebook + metadata: searchmap graft, share-fork bundle attach, full cell-12 rationale text |
| kernels censuses | scoreDescending (public ceiling = FOYSAL) · dateRun (v22 at 00:23 today) · per-user for six 2.5-band teams (no public ARC kernels) |
| topic 735243 re-read (`topic_735243.csv`) | no new comments since 08-18; Ravindra "cstl 2.70 pre-Q38" [V-doc] |
