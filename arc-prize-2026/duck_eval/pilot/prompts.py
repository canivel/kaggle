"""Prompt templates for the 3 synthesis scaffold arms (Phase-0c pilot).

Kept clean and mechanical on purpose: the MDL pressure lives in the
ACCEPTANCE RULE (acceptance = held-out pp - 2.0 * gzip-KB of source),
not in prompt exhortations.

Scaffolds:
  skeleton  - fill-in a provided def simulate(...) skeleton with TODO regions
  freeform  - write the function from scratch
  diff      - start from the nearest template sim (from ANOTHER game, LOGO)
              and patch it

Pod-side dependencies: stdlib only.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a program-synthesis engine. You write a single self-contained "
    "Python transition model for a 64x64 grid game from recorded transition "
    "evidence. You reply with exactly one Python code block and nothing else. "
    "The code block must define:\n"
    "  def simulate(state, action_id, x, y):\n"
    "      # state: 64x64 list of lists of ints (colors 0-15)\n"
    "      # returns (next_state, reward_class, done)\n"
    "Allowed imports: numpy (as np), math, typing, collections, itertools, "
    "functools, copy. No file or network access. The function must be pure "
    "given its arguments (module-level state is allowed only if it is "
    "reconstructed from `state` on every call)."
)

_COMMON_HEADER = """GAME: {game_id}
AVAILABLE ACTIONS: {actions}
GRID: 64x64, integer colors 0-15. action_id 6 is a click at (x, y); for all
other actions x=y=0 in the recordings.

EVIDENCE FORMAT: each line is one recorded transition:
  #<step> a<action_id>[@(x,y)] rc<reward_class><|done> :: <cell diffs>
Cell diffs are `row,col:old>new` entries (semicolon-separated); `no-change`
means the frame did not change. A truncated diff list ends with `(+N more)`.

REFERENCE FRAME (state_t of the earliest train transition, run-length encoded
per row as color*count):
{reference_frame}

TRANSITION EVIDENCE ({n_evidence} of {n_train} train transitions shown;
selection: changed-frame transitions first, then most recent):
{evidence}

ACCEPTANCE RULE (how your model is scored):
- Score = held-out 5-step open-loop exact-match percentage
          minus 2.0 points for every gzip-compressed KILOBYTE of your source
          (data literals count). Shorter general rules beat long lookup tables.
- Your model must beat identity-frame and train-lookup baselines by >=10 points.
- reward_class and done are also checked on one-step replay.
"""

SKELETON_TEMPLATE = _COMMON_HEADER + """
TASK: fill in the TODO regions of this skeleton. Keep the signature and the
return contract. You may add helper functions above simulate.

```python
def simulate(state, action_id, x, y):
    import numpy as np
    s = np.array(state, dtype=np.int64)
    ns = s.copy()
    reward_class = 0
    done = False
    # TODO 1: decode persistent structures from `s`
    #         (walls, counters, score bars, cursors, movable objects)
    # TODO 2: implement the effect of each available action_id,
    #         exactly as the evidence shows
    # TODO 3: implement passive dynamics (timers/ticks) if the evidence
    #         shows frames changing under repeated identical actions
    # TODO 4: set reward_class / done when the evidence shows them changing
    return ns.tolist(), int(reward_class), bool(done)
```

Reply with the completed code block only.
"""

FREEFORM_TEMPLATE = _COMMON_HEADER + """
TASK: write `def simulate(state, action_id, x, y)` from scratch. Return
(next_state, reward_class, done) where next_state is a 64x64 list of lists of
ints. Model the mechanics that explain the evidence; default to returning the
input state unchanged for actions with no observed effect.

Reply with one code block only.
"""

DIFF_TEMPLATE = _COMMON_HEADER + """
TEMPLATE MODEL (a working transition model for a DIFFERENT game, `{template_game}`;
its mechanics are related but NOT identical to {game_id}):

```python
{template_source}
```

TASK: refactor/patch the template into a transition model for {game_id}.
Reuse whatever structure transfers (grid decoding, action dispatch, counters);
delete mechanics the evidence for {game_id} does not support; add the ones it
does. Output the FULL patched source (not a diff), defining
simulate(state, action_id, x, y) -> (next_state, reward_class, done).

Reply with one code block only.
"""

REFINE_TEMPLATE = """Your model was replayed against the TRAIN transitions.

RESULT: {train_exact_pct:.1f}% one-step exact ({n_mismatch} mismatching
transitions, {n_error} raised errors).

MISMATCH REPORT (format: #step a<action> -> cells row,col: predicted>truth):
{mismatch_report}

Revise the model to fix these mismatches without breaking the transitions it
already gets right. Remember the acceptance rule: every gzip-KB of source
costs 2.0 points, so prefer fixing the RULE over adding special cases.
Reply with the full corrected code block only.
"""

TEMPLATES = {
    "skeleton": SKELETON_TEMPLATE,
    "freeform": FREEFORM_TEMPLATE,
    "diff": DIFF_TEMPLATE,
}
