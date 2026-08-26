# Daily Brief — 2026-08-06

## 1. Results deep-dive

### 1a. Overnight LB draw: 0.77 (frozen-fork filler, ledger n=23)

- z = **−1.18** vs the frozen ledger (n=22 mean 0.9564, sd 0.1582) — a low but
  interior draw from a stationary process (NC-15: change-point p=0.0117 NS;
  the ledger has seen 0.65 and 0.68). Updated mean ≈ 0.948.
- Trailing-4 (0.99, 0.97, 1.21, 0.77) mean **0.985**, +0.18 sd vs ledger mean —
  nowhere near the paired harm-pause condition.
- One-day swing 1.21 → 0.77 re-confirms the ~0.6-LB spread of the identical
  artifact. **Watch-rule ARMED** (06:00 agent): second consecutive sub-0.80
  draw tonight fires it. No other action; the draw is noise, not signal.
- borro1980's measurement paper (below) now *quantifies* why: 84.7% of
  repetition variance is the binary level-clear term and 2/25 games carry 65%
  of variance. Our no-single-draw discipline is externally validated.

### 1b. A22 compaction v2 seed-1 screen — K3 FAIL; v2 PAUSED; lane one FAIL from DEAD

Full screen: `learnings/sweeps/a22_v2_seed1_screen_2026-08-06.md`,
JSON: `runs/a22_v2_seed1/m1m2m3_screen.json`. Prereg expectation (§0: does
region-aware eviction alone recover the war-eval baseline?) — **NOT met.**

- **Canary/K1/K2 PASS, run VALID**: `compaction v2: ACTIVE` (mirroring=OFF),
  COMPACTION=1, graft applied=True, 2,617 events across all 25 games,
  `retained_reasoning_msgs=0` in 2617/2617 (RETAIN-OFF canary clean), no
  PATCH FAILED. The mechanism ran exactly as sealed.
- **M1 FAIL — worse than v1**: lc total 14 vs war 22 (v1: 17). Mean Δlc
  **−0.320** (breaches −0.128), worst **sc25 −2.0** (breaches −1.0 cap),
  2W/9L, sign-flip p=0.0557 (near-significant harm; v1 was 0.234).
- **sc25 did NOT recover** (0 vs 2, again −2) — and this time at normal action
  economics (2.5 acts/turn vs v1's blind-batching 5.9). With both v1 harm
  channels removed, neither toxic-digest nor batching explains sc25; **harm
  tracks compaction pressure itself** (or paired-seed variance — see caveat).
- **M2 FAIL, attribution split is the day's key mechanism fact**: tokens/action
  1.24× war, tokens/lc 1.87×. ~3.2M tokens of eviction relief vs **1.65M
  digest tokens re-injected — the digest pays back ~51% of relief**; only
  19.1% of events were zero-digest; reserve applied on 80.9%; digest saturates
  its 960-token cap (never-elided refuted list). Eviction classes: episode
  60.6%, user 39.4%, reasoning 0, fallback 0 (pins never yielded);
  stuck-suppress (K=5) never fired. **The LightMem regime (relief with
  digest≈0) was never entered — so the prereg's central bet went untested at
  the operating point we actually hit.**
- **M3 PASS-direction — first genuine win in the lane**: refuted re-proposal
  **−4.57pp (p=0.012 exact)**, v2 worse in only 7/25; all refrep thresholds
  reduced. Reverses v1's +2.24pp. The hygiene-gated, never-elided refuted list
  does what it was designed to do. K4 is NOT on track to kill.
- **Screen-power caveat (recorded, not gate-relevant)**: borro1980's variance
  decomposition implies paired single-seed Δlc screens are noisy on this
  suite. The sealed thresholds stand and the verdict is FAIL; but 2-seed
  screens for any future arm entry (already the gate-eval standard) matter
  even more than we thought.

### 1c. Verdict chain

v1 K3 FAIL (toxic digest + batching) → v2 K3 FAIL (both channels removed,
harm persists) ⇒ **one more independent K3 FAIL kills the A22 lane.** The
recommended highest-information use of that last life (screen §recommendation):
**v2.1 = pure eviction, digest-OFF** — directly tests the M2 attribution
(remove the 51% payback + the 80.9% reserve shrink; keep the eviction
machinery and the M3-winning extraction/logging for measurement only).

## 2. Discussions sweep (new since 08-05)

| Item | Verdict | Why |
|---|---|---|
| borro1980 paper-track post: 500-run Milestone-1 variance decomposition (84.7% binary level-clear; 2 games = 65% of var; half of runs = exactly 0; duck n_passes=20 does NOT fit 9h — ~4 passes max on RTX 6000-class) | **ADAPT** | Validates sigma discipline quantitatively; variance map to cross-check vs our per-game logs; pass-budget flag for reading the 500-run artifact; paper-track EV argument applies to us (write-ups exist, 118 paper teams) |
| borro1980 comment publicly naming the 1.58 pack (@nkosindwandwe, @yuchen2066, @anngle, @vansher, @nileshsarkarra) as merge candidates | **MONITOR** | Independent corroboration of the shared-artifact signature; if any merge with a paper, config details may surface |
| Jason Feng "A lot of Kaggle errors" (host replied: no systemic issue; errors track his own risky changes) | IGNORE | Self-inflicted; host confirms rerun infra healthy |
| Jason Feng open-sourced Sandwich/Gorilla/wles-wltd-mrps notebooks | IGNORE | 167th place, no LB-verified score; adapter-lane idea already in R24 pile |

No rule/deadline/infra changes. No new 1.5×+ public artifacts.

## 3. Research sweep (2026-08-02+)

| Item | Verdict | Why |
|---|---|---|
| **arXiv:2608.00902** — online KV compaction study: *immediate* compaction of fresh agent turns hurts; *delayed* compaction recovers most of the gap; eviction beats attention-matching under imperfect proxies | **ADAPT** | Independent support for pinning latest-reasoning/latest-episode; motivates a **minimum-age (delayed-eviction) gate** as a recorded v3 option — NOT folded into v2.1 (one-change discipline) |
| ScrubJay-MEM 2608.04746 (type-conditioned decay) | IGNORE | LLM-classified coefficients violate zero-LLM-call; conversational QA domain |
| LiveMem 2608.02515 | IGNORE | Requires model-architecture change |
| AgentMemBench 2608.00009 | IGNORE | Conversational setting, no interactive-env tasks |
| RoMeRL 2608.02508 | IGNORE | RL training loop incompatible with fixed-wallclock kernel |

Nothing citing/superseding CWL/MemDecay/LightMem/SelfCompact/Zero-Mem; no new
ARC-AGI-3 or TTT-for-agents work since the July items. (Third-party note,
unverified: benchlm.ai lists Opus 5 at 30.2% on ARC-AGI-3 vs Opus 4.8 at
1.5% — frontier-ceiling context only.)

## 4. Today's development decision (weekday rail — no panel)

**Build v2.1 (digest-OFF pure-eviction sub-arm), seal intent pre-build, one
kernel push.** Rationale: it is the screen's recommended highest-information
push; it tests the only major untested cell of the v2 design matrix (eviction
without digest); M3's win is preserved in the store/logging for measurement
without the injection channel. Risk acknowledged and accepted per the sealed
lane rules: a v2.1 seed-1 K3 FAIL ⇒ **A22 lane DEAD** — which is itself a
decisive, publishable answer (compaction-as-tested harms in this regime; keep
the M3 refuted-list mechanism for future memory-lane designs).

One-change discipline: v2.1 = v2 with digest injection disabled (extraction,
sidecar logging, and event schema unchanged). The 2608.00902 minimum-age gate
is explicitly recorded as OUT of this arm.

## 5. Open questions

- Is sc25 harm eviction-caused or paired-seed variance? (v2.1 gives one more
  paired observation; borro's variance map says treat single-seed sc25 deltas
  cautiously.)
- If the lane dies: does the M3 refuted-list result justify a standalone
  "pin-refuted-list into the scientist note" micro-arm (R24 pile item), which
  injects ~100 tokens instead of 960 and touches no eviction machinery?
- Paper-track: with 118 teams and our measurement corpus (ledger n=23, two
  sealed K3 screens, variance work), is a paper-track entry EV-positive?
  → Sunday panel agenda.
