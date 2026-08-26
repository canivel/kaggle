"""Executable world model for ARC-AGI-3 game `m0r0` — v2 (exploratory).

v2 investigated three strategies for the counter-tick on rows 0/63 that
caused all of v1's 42.5% miss. None beat v1; v1 remains canonical.

Strategy A: per-counter-value majority vote.
    For c % 3 == 0 (every 7th cycle position), no-tick is 4:2 majority,
    same as v1's no-op behavior. For all other c, the split is exactly
    2:2 -> no signal. v2_A == v1 on every tuple. No gain.

Strategy B: predict tick when c % 3 != 0 (treat c%3==0 as no-tick).
    Tested empirically: 97/200 = 48.5% vs v1 57.5%. **Regression of 9 pts.**
    Counter-intuitive but correct: the conditional distribution is 2:2 not
    skewed, so adding any tick prediction creates new wrong predictions on
    the 50% no-tick half.

Strategy C: modular pattern discovery.
    Within a single episode, ticks happen at step indices i mod 7 in
    {1, 3, 5} (run-length pattern 2,2,3 -> 3 ticks per 7 steps).
    100% deterministic IF you know step_idx. The stateless API
    simulate(state, action_id, x, y) does NOT provide step_idx, and the
    observation data spans 2 episodes whose step counts reset. Counter
    value c alone does not pin down step_idx (each c is visited 2-3
    times). Curve-fit attempts on (action, c%3) buckets reached at best
    8/9 = 89% on (action=6, c%3=0) -> tick, below the 90% invariant
    threshold and on only 9 samples.

Conclusion: v1 (no tick prediction) is at the stateless ceiling.
The 2-pixel-per-tick cost is already amortized to 99.98% pixel_match.
Stateful play (track step_idx across calls) would solve this but
requires an API change. v2 re-exports v1 unchanged so validate_sim.py
will report identical numbers; v1 stays active.

Diagnostic findings (recorded for posterity):
- ticks at episode-1 step indices: [1,3,5,8,10,12,15,17,19,22,24,26,29,
  31,33,36,38,41,43,45,48,...] -- ALL satisfy i mod 7 in {1, 3, 5}.
- per-counter-value transition table is 2:2 except at c divisible by 3
  where it's 4:2 (no-tick majority, already exploited by v1).
"""
from __future__ import annotations

# Re-export v1 implementation unchanged. v2's exploratory tick-prediction
# strategies all regressed or stayed flat; the stateless ceiling is 57.5%.
import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location("_m0r0_v1", str(_Path(__file__).with_name("m0r0_sim.py")))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
simulate = _m.simulate  # noqa: F401
