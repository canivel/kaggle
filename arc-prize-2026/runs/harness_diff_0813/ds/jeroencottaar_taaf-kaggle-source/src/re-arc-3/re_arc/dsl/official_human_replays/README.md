# Official-game human replay data

One JSON per official game: per-level action segments assembled from the
fastest public human replays of ARC-AGI-3 (the human-study public demo set,
https://arcprize.org/blog/arc-agi-3-human-dataset). Each file's `source`
block records the session GUID(s), the hosted game version they were played
on, and the extraction mode (`spliced` / `death-spliced` / `frame-spliced` /
`loop-cut` / `raw` for a single session, `stitched` with `per_level_sources`
when levels come from different sessions).

`OfficialHumanReplayAgent` (re_arc/dsl/agents/_official_human_replay.py)
plays these lists back verbatim; `make precompute_dsl_actions` turns them
into the canonical traces under re_arc/dsl/precomputed_actions/.

Regenerate or extend with:

```bash
python scripts/fetch_official_human_replays.py [--game <base>] [--all]
```

Every file was verified by a local replay at seed 0: it must reach WIN
without any GAME_OVER and use at most `baseline_actions[i]` actions on every
level `i` (the arc_agi scorecard then reports exactly 100.0, which CI
asserts). The officials' `baseline_actions` metadata itself is never
modified.
