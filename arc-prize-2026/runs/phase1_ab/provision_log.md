# Phase-1 A/B eval — RunPod H100 provision log
Started: 2026-07-10
Pod: root@103.207.149.71 -p 10418 (H100 SXM 80GB, /workspace 100G, CUDA 12.8, $2.99/hr, ~30h failsafe)

GOAL: 3 seeded passes of duck harness + phase1 patches, 25 public games, local
competition-arcade simulator. Null = runs/tufa_example_run/score.json (20 vanilla passes).

## Key facts from recon
- Bundle PYSETUP vLLM flags (Kaggle reference): vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6;
  serve args: --served-model-name vrfai/Qwen3.6-27B-FP8 --tensor-parallel-size 1
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --generation-config vllm
  --enable-prefix-caching --default-chat-template-kwargs '{"preserve_thinking": true}'
  --reasoning-parser qwen3 --max-model-len 65536  (NOTE: bundle uses 65536, not 32768;
  following bundle per feedback_vllm_params — match ALL params. Analyzer ctx stays 32768 via env.)
- Solver env (from PYSETUP setup_env): LOCAL_ANALYZER_* block, temp 0.6 top_p 0.95 top_k 20,
  thinking on, MULTIMODAL_CONTEXT=current_grid UPSCALE=4.
- CLI runner (inference.framework.run) builds its own Benchmark — cannot patch bm via hook.
  => driver script /workspace/run_ab.py mirroring notebook cells 6-14:
  sys.path bundle srcs -> unpickle deploy_target.pkl + benchmark_initial.pkl ->
  phase1_patch.apply(bm) -> CompetitionArcadeServer(game_ids=25, environments_dir=env_files)
  -> bm.games = GameAPI(...arcade_spec) -> bm.n_passes=1 -> await bm.run(target).
- CompetitionArcadeServer(game_ids=..., environments_dir=...) avoids re_arc dep for id list.
- arc_agi>=0.9.8 required by taaf (local wheel is 0.9.6 — use PyPI on pod).
- Preamble: HarnessSolver concurrency=28, max_runtime_s_per_game=7920 (2.2h cap/pass).
- Seeds = independent stochastic passes (temp 0.6); no true seed plumb in HarnessSolver.
  Label seeds 1-3, set PYTHONHASHSEED per pass.

## Timeline
- 08:30 Pod verified: H100 SXM 80GB, /workspace 100G empty, python 3.12.3 (system, PEP668-managed).
- 08:36 Shipped: taaf_bundle.tar.gz (418KB), phase1.tar.gz (22KB), env_files.tar.gz (874KB from
  kaggle-data/environment_files, 25 games), ~/.kaggle/access_token (no kaggle.json on this box —
  CLI uses OAuth access_token; moved to /root/.kaggle/access_token, chmod 600, contents never printed).
- 08:40 venv at /workspace/venv (uv, py3.12.3). Model download launched:
  kaggle datasets download driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot -p /workspace/models --unzip
  (27.1GB @ ~17MB/s, ETA ~27min; log /workspace/model_download.log). OAuth token worked.
- 08:45 Dep install issue 1: re-arc-3 git dep (Tufalabs private repo) unfetchable -> installed
  editables with --no-deps + manual dep list. Issue 2: taaf pyproject pins Python==3.12.12 -> skipped
  editable installs entirely; run_ab.py sys.path-inserts bundle srcs (same as notebook cell 8).
- 08:50 Installed: vllm==0.19.0 torch==2.10.0+cu128 (EXACT wheelhouse match), flashinfer-python,
  arc_agi 0.9.9 (PyPI; taaf needs >=0.9.8), arcengine, matplotlib==3.10.6 etc.
- 08:55 Driver /workspace/run_ab.py shipped (also at duck_eval/phase1/run_ab.py). Mirrors notebook
  cells 6-14: setup_env (PYSETUP LOCAL_ANALYZER_* block, :8000) -> unpickle deploy_target+bm ->
  phase1_patch.apply(bm) -> CompetitionArcadeServer(25 games, /workspace/environment_files) ->
  GameAPI list -> bm.n_passes=1 -> await bm.run() -> per-game JSON to /workspace/ab_results/.
- 08:58 DRY RUN OK on pod: bm unpickled (HarnessSolver), phase1 cfg applied (explore_after_turns=10
  budget=8 max_explores=6 evict_low_frac=0.5, all enables true), arcade served 25 games, GameAPI x25.
  NOTE: env-file game id suffixes differ from tufa run_config for some games (e.g. ar25-e3c63847 vs
  ar25-0c556536) — join A/B vs null on 4-char prefix.
- 09:05 Confirmed solver reads LOCAL_ANALYZER_BASE_URL/MODEL_ID at tool_agent import time;
  run_ab.py sets env before any inference import — safe. Staged /workspace/probe_tps.py
  (single-stream + 8-way concurrent decode probe). No Kaggle RTX PRO 6000 tokens/s reference
  exists in learnings yet — parity table gets the H100 number now; Kaggle side to be pulled
  from a duck kernel log later.
