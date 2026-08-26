# lf52 sim v1 notes

## Game summary

lf52 is (in the 200-tuple random-exploration sample we have) a pure
**row-0 step counter** game. There is no playfield interaction at all
in the observation set: every action, regardless of action_id or click
coords, advances a single counter on row 0 and nothing else.

## Observations

### Top-level
- 6 available actions: {1, 2, 3, 4, 6, 7}
- 200 random-exploration tuples. Distribution:
  1=34, 2=26, 3=31, 4=34, 6=41, 7=34.
- `reward_class` is **1** for every tuple. `done` is always False.
  `level` stays 0. So `(reward=1, done=False)` is optimal for every
  observed tuple.
- `n_changed == 1` for **all 200** tuples. There are no large diffs
  to chase. This is unusual relative to bp35 and is the signal that
  the entire game is captured by one rule.

### The single rule

For every transition (action_id ignored, x/y ignored):

```
v = min(state_t[0])               # current "tier" (0, 1, 2, ...)
c = leftmost column where state_t[0, c] == v
state_t1 = state_t
state_t1[0, c] = v + 1
```

This is a counter that fills row 0 with 1s (overwriting the initial
row-of-zeros), then on overflow restarts at col 0 overwriting 1s with 2s,
and so on. Across the 200 observed tuples we saw transitions from
(0 -> 1) 134 times and (1 -> 2) 66 times, confirming the tier-bump
behaviour.

### What we explicitly verified
- `state_t1[0, c] == v + 1` for 200/200 tuples.
- All other rows untouched for 200/200 tuples (`n_changed == 1`).
- Holds for every action in {1, 2, 3, 4, 6, 7}.

## Simulator v1 behaviour
1. Find min value `v` on row 0.
2. Set the leftmost cell whose value equals `v` to `v + 1`.
3. `reward_class = 1`, `done = False`.

## v1 result on full 200-tuple set

```
{
  "game": "lf52",
  "split": "all",
  "n": 200,
  "errors": 0,
  "state_exact_pct": 100.0,
  "pixel_match_pct": 100.0,
  "reward_acc_pct": 100.0,
  "done_acc_pct": 100.0,
  "by_action": {
    "1": {"n": 34, "exact_pct": 100.0, "pixel_pct": 100.0},
    "2": {"n": 26, "exact_pct": 100.0, "pixel_pct": 100.0},
    "3": {"n": 31, "exact_pct": 100.0, "pixel_pct": 100.0},
    "4": {"n": 34, "exact_pct": 100.0, "pixel_pct": 100.0},
    "6": {"n": 41, "exact_pct": 100.0, "pixel_pct": 100.0},
    "7": {"n": 34, "exact_pct": 100.0, "pixel_pct": 100.0}
  }
}
```

## Honest caveats

1. The 200 tuples are pre-interaction in a vacuum. Once a playfield
   appears (sprite, score panel, level-up event), this minimal rule
   will be incomplete. We will know this is happening when reward_class
   becomes 0 or 2, when level increases, or when n_changed > 1.
2. The counter probably wraps after row 0 is full of 15s. We do not
   have observation data past the (1 -> 2) tier, but the same
   "leftmost-min -> min+1" rule extrapolates naturally and we set no
   ceiling on `v`.

## v2 plan (only if more obs are collected)

1. Collect a longer trajectory (at least 600+ steps) to observe
   playfield wake-up, level transitions, and reward_class != 1.
2. Look for "carry" / wrap behaviour when row 0 saturates at value 15.
3. Re-bucket by action only once non-counter transitions appear.

For now v1 is provably perfect on the available observations
(state_exact 100%, pixel_match 100%, reward 100%, done 100%), so there
is no v2 to do without new data.
