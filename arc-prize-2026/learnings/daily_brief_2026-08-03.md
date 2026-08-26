# Daily Brief — 2026-08-03 (Monday; weekday cadence, no panel)

## 1. Results deep-dive

### 1a. Overnight scored draw
- **0.99** (frozen-fork filler, 08-03 00:07Z, API COMPLETE). Interior draw (z ≈ +0.31 vs
  n=19 stats): no watch-rule fire, no band change. The 0.65/0.68 dip did NOT continue —
  consistent with the NC-15 verdict (n₂=2 tail artifact, record stationary). Ledger now
  **n=20, mean 0.9430, s 0.1552** (`runs/lb_ground_truth.md` refreshed).
- Paired harm-pause (trailing-4 −1.5s) applies to gated arms only — not evaluated on filler.
- LB: KOJIMA 1.86 frozen; **Andy liu 1.69 NEW #2**; gold band rose again — 1.54 ≈ #13–14,
  **top-10 cutoff ≈ 1.56–1.58** (07-28: 1.49). Our 1.33 slides on pure drift. The static-band
  problem worsens ~0.02–0.03/week; only a mechanism win moves us.

### 1b. A22 compaction v1 — seed-1 prereg screen (THE result of the day)
Build COMPLETE (2h12m, 25 games, mean 1.45 raw). Canaries ALL PASS → run VALID as evidence
(K1/K2 not void): banner `compaction v1: ACTIVE`, `COMPACTION=1`, graft applied=True,
**1449 COMPACTION events**, RETAIN canary 1449/1449.

**Screen verdict (full report `learnings/sweeps/a22_seed1_screen_2026-08-03.md`, JSON
`runs/a22_compaction_v1/m1m2m3_screen.json`; paired vs war-eval seed 1 with full
benchmark+transcripts at `runs/kernel_pulls/war_eval_v1/` — M2/M3 fully paired, no gaps):**

- **M1 FAIL**: mean Δlc **−0.200** (sd 0.646), 2W/6L/17T, exact sign-flip p=0.2344,
  worst game **sc25 −2.0**; lc totals 17 vs 22. Breaches BOTH legs (worst cap −1.0;
  mean precedent −0.128).
- **M2 FAIL (wrong sign)**: tokens/action 0.99× (flat); **tokens/level 1.68× WORSE**
  (119.8k vs 71.3k). The OpenAI ÷6 did not reproduce: +30% tokens, +31% actions, −23% levels.
- **M3 FAIL (wrong sign, NS)**: refuted-re-proposal +1.25 pp; su15 improved (−3.1 pp) but
  **lp85 +21.2 pp, sb26 +10.7 pp worse** — the exact games the mechanism was built for.
- **K3 FIRES → arm PAUSED.** K4 "on track to kill, not yet indicated" (needs both seeds).

**Pre-registered expectation NOT met.** This was a real test (mechanism fully engaged) and
the mechanism produced harm, not noise:
1. **Toxic digest**: HeuristicExtractor promotes hedged, mid-sentence-truncated musings
   ("Actually, I think…", mutually contradictory) into "do NOT re-verify FACT" directives;
   collapses the actual refuted list to "+77 more"; self-ingests via the model's echoed
   world-model (sc25 FACT F5 quotes FACT F3; sc25 logged refuted=0 across all 42 events).
   Net effect: rolling-cut-with-authoritative-misinformation.
2. **Retained reasoning changed the policy, not just the memory**: sc25 ran 49 turns @ 5.9
   actions/turn (vs 101 @ 1.8 paired) — blind action batching replaced observe-act cycles.

### Decision (sealed rules + judgment)
- **NO v1 seed-2 push.** K3 pauses the arm; both root causes are design flaws we already
  know how to fix, and the research sweep (below) independently prescribes the redesign.
  Spending a push to confirm a superseded design kills a day for no decision-relevant bit.
- **A22 pivots to v2 (new intent-file, sealed pre-build, weekday build-rail):** see §4.

## 2. Discussions sweep (`learnings/sweeps/discussions_sweep_2026-08-03.md`)
1 new post: "Minimalistic All-in-One Toolkit for ARC-AGI-3" (harness rewrite, no scores) →
**IGNORE** (fork-never-build rule; evidence-free; orthogonal to A22). All else ≥2d old.
Absence-signal: no public disclosure above 1.17-duck since 07-31 while gold climbs — the
top of the LB is not sharing; forum is low-yield.

## 3. Research sweep (`learnings/sweeps/research_sweep_2026-08-03.md`)
11 relevant items, **5 ADAPT / 0 ADOPT**; arXiv 2608.* not yet indexed; ARC blog unchanged.
The ADAPTs converge on one reframe that the seed-1 failure now empirically corroborates:
- **The duck harness already IS the rolling-cut recency baseline** — so "compaction vs
  rolling cut" was the wrong axis. The literature axis: **region-aware eviction with
  pinning vs recency** (CWL 2606.11213: evict action episodes whose effects persist in env
  state; MemDecay 2607.10582: pin system prompt + active reasoning, fast-decay scratchpad).
- **LightMem repro 2607.29104**: compaction wins are mostly **budget relief**, raw turns >
  constructed summaries → measure/attribute to budget relief; don't build prose summaries.
- **SelfCompact 2606.23525**: deterministic trigger rubric — **suppress eviction
  mid-derivation / while stuck**.
- **Zero-Mem 2607.29377**: zero LLM calls in the eviction path (v1 already complied; keep).
- Scope guard (2607.26637): NO model-maintained memory store — narrows the Living-Harness
  graph-state idea to conceptual only. Its scored-arm prereg amendment (yesterday's handoff
  item 3) is **superseded** by the v2 intent below; R24 sees both.

## 4. Today's development target (single item)
**A22 compaction v2** — same one-flag graft discipline, new mechanism, per root causes + ADAPTs:
1. **Region-aware eviction (replaces digest-of-evicted-span):** pin system prompt +
   scientist-note + most recent reasoning; evict stale action-episode blocks FIRST
   (oldest tool/act cycles whose board effects are visible in the current frame), before
   any reasoning text is touched.
2. **Digest demoted, hygiene-gated:** facts only from complete declarative sentences, no
   hedge-prefixes ("actually/wait/maybe/I think"), no truncation-tail promotion; refuted
   list NEVER elided; digest content marked non-quotable and stripped from model echoes
   (no self-ingestion). If the hygiene gate empties the digest, inject nothing (budget
   relief is the win per LightMem, not summary prose).
3. **RETAIN decoupled and OFF by default** (COMPACTION_RETAIN=0): retained reasoning is
   implicated in the blind-batching harm; it becomes a separate sub-arm, not a rider.
4. **Suppress-cut-while-stuck:** deterministic rubric — no eviction event while the last
   K actions produced no board change (mid-derivation guard).
5. Kill rules inherit K1-K5 verbatim; M1/M2/M3 unchanged; M2 gains a budget-relief
   attribution split (evicted-chars vs digest-tokens vs outcome).
Build path: patch v2 in `duck_eval/warpack/_kaggle_dataset/compaction_patch.py`, extend
`compaction_smoke.py`, byte-audit, dataset version push + kernel push (2 slots free today)
IF smoke is green; banner read tomorrow.

## 5. Open questions
- Does region-aware eviction alone (RETAIN off, digest empty-allowed) recover the war-eval
  seed-1 baseline (M1 ≥ −0.128, worst ≥ −1.0)? That is the ONLY question v2 seed-1 answers.
- Is sc25's collapse digest-poisoning or batching? (v2 removes both — if sc25 recovers we
  can't attribute; acceptable, we need the lane moving, attribution via sub-arms later.)
- A17 72B pin: C4 deadline TODAY, route DEAD since 07-30 (B2a) → pin formally dissolves;
  no successor bench planned (zero-budget rule stands).
- Boristown A/B stays DORMANT (NC-14); nothing today changes that.
