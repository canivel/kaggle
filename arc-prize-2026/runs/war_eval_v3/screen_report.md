# war-v1 build-rail SCREEN vs null10 — 1 seed (NOT a gate look)

Scorer validated: max err 0.0e+00 over 1000 checks.

## Seed-3 verification (panel R12 methodology Q3, recorded 2026-07-16)

- **Run identity: VERIFIED as the seed-3 (v3) build.** benchmark start 2026-07-16T12:37:16Z /
  end 14:50:05Z matches the kernel lastRunTime 12:27 UTC today; v1 ran 07-14, v2 ran 07-15
  (three distinct runs, benchmark.json md5s all differ). Log banner
  `RUN_HEAVY=True (fast-submit gate off)` present exactly once (as in v1/v2) — the
  `WARPACK_FORCE_OFFLINE_BENCH=1` eval line took effect and the full 25-game offline bench ran
  (2h12m). Warpack banner: `warpack v1: patches applied from .../canivel/arc-war-kit` +
  config echo `banking=True recovery=True shortcircuit=True retry_guard=True
  bank_min_time=120.0 bank_strict=True recovery_repeats=30` — md5-identical across v1/v2/v3.
- **Seed-only diff: CERTIFIED (code-identical; no explicit seed constant exists on this rail).**
  Kaggle kernel code pulled 2026-07-16 (latest = v3) is cell-identical (17/17 cells,
  sha256/16 `3e22b1c9d6fbb9ff`) to the local deterministic build
  `notebooks/duckwar-eval/arc3-duck-war-eval.ipynb`, which `build_eval_notebook.py` generates
  from `notebooks/duckwar/arc3-duck-war.ipynb` (source unchanged since Jul 13, i.e. before all
  three pushes) by prepending exactly one line: the `WARPACK_FORCE_OFFLINE_BENCH=1` env set.
  The notebook contains no seed constant — "seed N" = independent stochastic replicate N
  (kernel version N); runtime randomness is unseeded LLM sampling (vLLM `seed=0` default,
  temperature 0.6). Cross-run fingerprints identical across v1/v2/v3: `taaf_setup_env.json`
  md5 `2a3cf03c091a6fab...`, `git_status.txt` md5 `5bca49febc467a57...`, warpack banner md5
  `b5e17460d96b1b01...`, summary header (label/solver/games/passes), and benchmark
  `game_weights`. Verdict: v1/v2/v3 are the same code and config; the only difference is the
  replicate draw.

- **PRIMARY paired Δlc: mean -0.088** (sd 0.544, 6W/12L, exact sign-flip p = 0.7957)
- Secondary Δlog1p(RHAE): mean -0.202 (p = 0.8721)
- lc totals: war 13 vs null 15.2
- RHAE run-mean: war 1.162 vs null 1.636

| game | war lc | null lc | Δlc | war RHAE | null RHAE | Δlog1p | flags |
|---|---|---|---|---|---|---|---|
| ar25 | 1 | 1.10 | -0.10 | 2.08 | 3.26 | -0.33 | - |
| bp35 | 1 | 0.80 | +0.20 | 0.44 | 0.25 | +0.14 | - |
| cd82 | 0 | 0.20 | -0.20 | 0.00 | 0.18 | -0.16 | - |
| cn04 | 0 | 0.50 | -0.50 | 0.00 | 1.62 | -0.96 | - |
| dc22 | 0 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | - |
| ft09 | 0 | 1.40 | -1.40 | 0.00 | 10.20 | -2.42 | - |
| g50t | 1 | 0.00 | +1.00 | 2.94 | 0.00 | +1.37 | - |
| ka59 | 1 | 0.40 | +0.60 | 3.57 | 0.49 | +1.12 | - |
| lf52 | 0 | 0.90 | -0.90 | 0.00 | 1.39 | -0.87 | - |
| lp85 | 1 | 1.00 | +0.00 | 2.78 | 2.48 | +0.08 | - |
| ls20 | 0 | 0.30 | -0.30 | 0.00 | 0.62 | -0.48 | - |
| m0r0 | 0 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | - |
| r11l | 1 | 1.00 | +0.00 | 4.76 | 3.52 | +0.24 | - |
| re86 | 1 | 1.30 | -0.30 | 1.53 | 2.57 | -0.34 | - |
| s5i5 | 1 | 0.10 | +0.90 | 2.78 | 0.02 | +1.31 | - |
| sb26 | 1 | 1.00 | +0.00 | 2.78 | 2.78 | +0.00 | - |
| sc25 | 0 | 0.20 | -0.20 | 0.00 | 0.16 | -0.14 | - |
| sk48 | 0 | 0.10 | -0.10 | 0.00 | 0.28 | -0.25 | - |
| sp80 | 1 | 0.70 | +0.30 | 4.76 | 1.09 | +1.02 | - |
| su15 | 0 | 1.00 | -1.00 | 0.00 | 2.20 | -1.16 | - |
| tn36 | 0 | 0.50 | -0.50 | 0.00 | 2.05 | -1.12 | - |
| tr87 | 0 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | - |
| tu93 | 1 | 1.30 | -0.30 | 0.01 | 1.54 | -0.92 | - |
| vc33 | 2 | 1.40 | +0.60 | 0.62 | 4.21 | -1.17 | - |
| wa30 | 0 | 0.00 | +0.00 | 0.00 | 0.00 | +0.00 | - |
