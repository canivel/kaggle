# v14+ Iteration Roadmap — Post Chronos v13

**Date:** 2026-04-20
**Status:** v13 (Chronos v85) submitted at 8pm EDT, scores tomorrow morning

---

## Decision tree based on v13 score

### Path A: v13 = 0.35-0.45 (Chronos works — IDEAL)
**Meaning:** Chronos's BFS improvements (pickle deepcopy, IDDFS, animation drain, action-type filter, scalar transition-table A*) closed much of the 0.27→0.46 gap. Confirmed path forward.

**v14 next-move options** (pick one):
- **v14a: Chronos + back-label** — replay winning paths into CNN buffer with distance-weighted reward. Our v9 neutral addition. Simple port (~50 lines). Expected +0.01-0.03.
- **v14b: Chronos + WorldModel entity tracker** (ashvin FORGE v30, LB=0.42) — learn (action → dx/dy per entity color) from observations, use for A* planning without game copies. ~200 lines. Expected +0.02-0.05.
- **v14c: Chronos + cross-game type memory** (Redpill v8) — keep `TransitionMemory` across game instances of same type. By game #2 agent already knows what works. ~100 lines. Expected +0.03-0.05.

Pick by: whichever has highest EV given v13's failure modes (per-game breakdown).

### Path B: v13 = 0.25-0.32 (marginal — some Chronos features helped)
**Meaning:** Only SOME of Chronos's improvements pay off in our env. Need to isolate what works.

**v14 approach:** Port Chronos features ONE AT A TIME into our forge_v35 v7 baseline (proven 0.27):
1. v14a = v7 + pickle deepcopy only (~10 lines) — measure
2. v14b = v14a + animation drain (~10 lines) — measure
3. v14c = v14b + IDDFS fallback (~50 lines) — measure
4. v14d = v14c + action-type filter (~20 lines) — measure
5. v14e = v14d + scalar transition-table (~200 lines) — measure

One feature per day. Slow but unambiguous attribution.

### Path C: v13 = 0.15-0.22 (env-specific regression, like v11/v12)
**Meaning:** Same pattern as FORGE v16 forks — public code at 0.30+ scores 0.15-0.20 for us. Something account-specific.

**v14 approach:** Investigate account issue:
- Try pushing to a NEW kernel slug (e.g., `canivel/arc3-chronos-fresh`) — does the SAME code score differently?
- Compare our kernel logs to upstream via Kaggle's metadata
- Try ALT account if we have one
- As fallback: revert to forge_v35 v7 (known 0.27) and port Chronos features manually

---

## Parallel research (run during v14 days)

While iterating on v14+:
- Spawn KAOS research agent to investigate **cross-game memory architectures** used by 0.50+ scorers
- Look for recent public 0.40+ notebooks (refresh the search daily)
- Check Kaggle discussions for any leaked technique hints

---

## What we NOT doing (strategic discipline)

- No more multi-feature submissions (v8, v10 pattern — multiple changes hard to attribute)
- No more copying ENTIRE upstream notebooks without isolated testing
- No more "graph explorer" style architectural departures (v10 cost us a day)
- No more over-testing locally (local eval doesn't match Kaggle)

---

## Submission schedule discipline

- 1 submission/day at 8pm EDT sharp
- Each submission = ONE clearly-scoped change
- Record per-game breakdown if possible
- Update memory after EACH submission result

---

## Files for v14 prep (ready on disk)

- `f:/kaggle/arc-prize-2026/notebooks/forge_agent/chronos_v85_base.py` — clean copy of Chronos v85
- `f:/kaggle/arc-prize-2026/notebooks/forge_agent/chronos_plus_backlabel.py` — copy for v14a work
- `f:/kaggle/arc-prize-2026/learnings/v13_deep_research.md` — research findings
- `f:/kaggle/arc-prize-2026/learnings/v14_roadmap.md` — this doc
