# C7 amendment note (2026-07-23)

- Prereg amendment 2026-07-23 C7 originally ruled `SENTINEL_BUDGET=140`
  (null10 mean actions/run). A day-1-execution build of this notebook with
  seed 1 + budget 140 was assembled and smoke-tested (32/32 PASS) but NEVER
  pushed: the morning loop had independently pushed seed 1 (07-22, COMPLETE
  14:59Z) and seed 2 (07-23 ~09:00 local, "W2 confirmatory null") with
  `SENTINEL_BUDGET=150` on this same slug.
- Coordinator ruling 09:57 EDT 2026-07-23: **C7 AMENDED to 150** — the live
  two-seed 150-ledger is authoritative (cross-seed uniformity beats the 140
  provenance; no 140-run exists, so no mixed ledger). The 140 build was NOT
  pushed (push discipline preserved: loop's seed-2 = push 1/2; push 2/2
  reserved for the A17 bench).
- `arc3-duck-sentinel-eval.b140-archived-2026-07-23.ipynb` is the archived,
  never-pushed 140 build (audit reference only — do not push).
- `arc3-duck-sentinel-eval.ipynb` is the canonical staged reference, rebuilt
  by `duck_eval/warpack/build_eval_notebook.py --sentinel --sentinel-budget
  150` (seed-1 banner lines; the loop stamps per-push seed ordinals at push
  time). Guarded by `duck_eval/sentinel/sentinel_smoke.py` S2c/I1c
  (assert the SENTINEL_BUDGET=150 export + banner — the C7 inert-sentinel
  check, #1 pre-seal risk).
- Note: the sentinel is **v2** (game-envelope unit, R16 Q2 re-key) — build-log
  verification greps for `SENTINEL v=2`, not the stale `v=1` in older docs.
- **(f)-default interaction (added after the step-4 builder change):** since
  2026-07-23 the builders default the game-over-continuation graft ON
  (prereg amendment item 4). The staged `arc3-duck-sentinel-eval.ipynb` here
  was built BEFORE that change and matches the LIVE seed-1/2 composition
  (sentinel-only). To reproduce it with the current builder use:
  `uv run python duck_eval/warpack/build_eval_notebook.py --sentinel
  --sentinel-budget 150 --no-continuation`. A default rebuild would ADD the
  (f) graft — a composition change vs the live seed ledger; the loop/panel
  must rule on that before any seed-3 push (seed convention = identical
  notebook per seed).
