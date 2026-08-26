# Discussion Sweep — 2026-07-16

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (last ~3 days). No rule changes.

## 1. YUTO KOJIMA (LB #1, 1.86) — still zero footprint
- Profile (kaggle.com/kojimatech): joined 6mo ago, no bio, no notebooks/datasets tabs, 2 comps, 4 followers, last seen <1 day. 40 entries, last sub 11h ago — actively iterating.
- Zero discussion posts; web search finds nothing (no GitHub/blog under kojimatech).
- **LB shift above the 1.44 wall**: Tecnod8.AI 1.61 (#2), Mathurin Ache / anngle / NoOneAhead 1.56, paul 1.54, Dinesh 1.50, hiranorm 1.48. The wall is broken by ~8 teams — consistent with per-draw-mean gains existing, not just order-stats luck.
- **Verdict: WATCH.** Nothing to adopt; reinforces that real headroom above 1.44 exists.

## 2. Milestone Prize #1 official announcement (Greg Kamradt, disc 725002, 3d ago)
- 1st Tufa Labs "The Duck" — notebook now PUBLIC: `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`. REPL over game state, multimodal perception (image + ASCII + segmentation zoom), infinite play via message eviction. Gains from "multimodality + better base models, not hand-built tools".
- 2nd Reki (`ruichardliu/milestone1-2nd-solution`): Gemma-4-31B vision-policy, JSON plan, reflection memory every ~10 steps, numpy click heuristic (small/rare-colored/button-like shapes) + **dead-signature tracker** (stop clicking object types that never change anything, per level).
- 3rd forge (`mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone`): top run **turned OFF** all extra machinery (generator/arbiter/confidence) — simplicity wins again.
- **Milestone 2 deadline: Sept 30** (final milestone).
- **Verdict: ADAPT.** We already run duck+warpack; diff the published Duck notebook vs our fork for drift. Reki's dead-signature per-level click suppression is cheap and orthogonal — candidate warpack tweak.

## 3. Run-to-run variance thread (Alvaro, disc 726552, 17h)
- Byte-identical logic scored 0.20 vs 0.03 — cause: unseeded random ACTION6 fallback. Open question posed to hosts: is the ENV seeded per run (can repeated submits be averaged)?
- Reki (in 725002): his v5=0.86 vs identical v7=0.00 was "infra/timeout failure, not randomness".
- **Verdict: ADOPT (evidence).** Matches our build-rail sd 0.572 finding; also flags 0.00s as infra deaths, not draws — exclude from ledger draws. Watch for a host answer on env seeding (would validate our Δlevels_completed gate + averaging).

## 4. Bill Ma: RLIMIT_AS kills reruns (disc 724841, 3d)
- `resource.setrlimit(RLIMIT_AS,...)` → mmap of .so fails ~30min → silent Submit Error (not real OOM). Fix: never set RLIMIT_AS; log RSS only. It also masked a second missing-bundled-file crash.
- **Verdict: ADOPT (checklist).** Grep our agent/notebook for setrlimit; add to preflight.py forbidden patterns.

## 5. Minor / IGNORE
- GPT-5.6 Sol SOTA 7.8% on arcprize.org leaderboard (disc 726340) — frontier context only.
- RTX PRO 6000 thread (724890), AGI-timeline thread, "Need Team" — no signal.

## Actions
1. Diff public Duck milestone notebook vs our frozen fork (drift check).
2. Preflight: forbid RLIMIT_AS; treat 0.00 as infra, not a draw.
3. Evaluate Reki dead-signature click suppression as warpack variant.
4. Monitor 726552 for host reply on env seeding.
