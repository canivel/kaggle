#!/usr/bin/env python
"""Phase-1 A/B driver — duck harness + phase1 patches, 25 public games,
taaf local competition-arcade simulator. One pass per invocation.

Mirrors duckfork notebook cells 6-14 (env -> sys.path -> unpickle bm ->
customization hook (phase1) -> competition arcade -> bm.run).

Usage (on the pod, vLLM already serving):
    /workspace/venv/bin/python /workspace/run_ab.py --seed 1
Writes /workspace/ab_results/phase1_seed{N}.json + full taaf outputs in
/workspace/ab_results/seed{N}/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from pathlib import Path

BUNDLE_DIR = Path("/workspace/taaf_bundle")
ENV_FILES = "/workspace/environment_files"
PHASE1_DIR = Path("/workspace/phase1")
OUT_DIR = Path("/workspace/ab_results")
VLLM_BASE_URL = os.environ.get("AB_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
SERVED_MODEL_NAME = "vrfai/Qwen3.6-27B-FP8"


def setup_env() -> None:
    """Cell 2 + PYSETUP setup_env, adapted for the pod (vLLM on :8000)."""
    env = {
        "MPLBACKEND": "Agg",
        "TAAF_RUN_AS_SUBMISSION": "0",
        "TAAF_MINIMAL_DIAGNOSTICS": "0",
        "ONLY_RESET_LEVELS": "true",
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "LOCAL_ANALYZER_BASE_URL": VLLM_BASE_URL,
        "OPENAI_BASE_URL": VLLM_BASE_URL,
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
        "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
        "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
        "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "LOCAL_ANALYZER_TEMPERATURE": "0.6",
        "LOCAL_ANALYZER_TOP_P": "0.95",
        "LOCAL_ANALYZER_TOP_K": "20",
        "LOCAL_ANALYZER_ENABLE_THINKING": "true",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "local"),
        "ARC_API_KEY": "test-key-123",
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    # Phase-1 pre-registered v2 arm (hook_cell.py defaults)
    os.environ.setdefault("PHASE1_EXPLORE_AFTER_TURNS", "10")
    os.environ.setdefault("PHASE1_EXPLORE_BUDGET", "6")
    os.environ.setdefault("PHASE1_MAX_EXPLORES", "3")
    os.environ.setdefault("PHASE1_EXPLORE_MIN_LEVEL_ACTIONS", "90")
    os.environ.setdefault("PHASE1_EXPLORE_LEVELUP_COOLDOWN", "20")
    os.environ.setdefault("PHASE1_EVICT_LOW_FRAC", "0.5")


def dump_results(bm, out_path: Path, seed: int, elapsed_s: float) -> dict:
    games: dict = {}
    for run in bm.game_runs:
        score = run.final_score
        if score is None:
            try:
                score = run._compute_final_score()
            except Exception:
                score = None
        games[run.game_id] = {
            "score": score,
            "state": run.state,
            "levels_completed": run.levels_completed,
            "number_of_levels": run.number_of_levels,
            "actions": sum(run.actions_per_level or []),
            "generated_tokens": sum(
                rec.generated_tokens for rec in (run.history or [])
            )
            + (run.final_generated_tokens or 0),
        }
    payload = {
        "label": bm.label,
        "seed": seed,
        "n_passes": bm.n_passes,
        "elapsed_s": elapsed_s,
        "vllm_base_url": VLLM_BASE_URL,
        "phase1_env": {
            k: v for k, v in os.environ.items() if k.startswith("PHASE1_")
        },
        "games": games,
        "mean_score": (
            sum(g["score"] or 0.0 for g in games.values()) / len(games)
            if games
            else None
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", default="phase1")
    parser.add_argument("--no-phase1", action="store_true", help="vanilla control run")
    args = parser.parse_args()

    setup_env()
    run_dir = OUT_DIR / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RECORDINGS_DIR", str(run_dir / "server_recording"))

    # Cell 8 equivalent: bundle sources importable (pip -e already did this,
    # but keep explicit path inserts so the unpickle never depends on it).
    for entry in (
        BUNDLE_DIR / "src" / "ARC3-Inference",
        BUNDLE_DIR / "src" / "tufa-arc-agi-framework" / "src",
    ):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    if str(PHASE1_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE1_DIR))

    import taaf.competition_arcade as competition_arcade
    import taaf.game_api

    # Cell 10: unpickle target + benchmark.
    with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as fh:
        target = pickle.load(fh)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    bm.job_dir = run_dir

    # Cell 12: customization hook — phase1 patches BEFORE the run.
    if not args.no_phase1:
        import phase1_patch

        cfg = phase1_patch.apply(bm)
        print(f"phase1: applied, cfg={cfg}", flush=True)
    bm.label = f"duck-{args.label}-seed{args.seed}"

    # Cell 14 (offline branch) but via the competition-arcade simulator:
    # enumerate the bundled env files, then serve them through the local
    # competition-mode REST arcade (same interface as the Kaggle gateway).
    import arc_agi

    offline = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV_FILES
    )
    game_ids = [env.game_id for env in offline.available_environments]
    print(f"offline env files expose {len(game_ids)} games: {game_ids}", flush=True)
    assert len(game_ids) == 25, f"expected 25 games, got {len(game_ids)}"

    server = competition_arcade.CompetitionArcadeServer(
        game_ids=tuple(game_ids), environments_dir=ENV_FILES
    )
    server.start()
    try:
        spec = server.arcade_spec
        bm.games = [
            taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec)
            for gid in server.exposed_game_ids
        ]
        bm.n_passes = 1
        bm.game_weights = None
        print(
            f"competition arcade at {server.base_url}; "
            f"{len(bm.games)} games, 1 pass, seed tag {args.seed}",
            flush=True,
        )
        t0 = time.monotonic()
        await bm.run(
            soft_end_time=None,
            runtime_environment=target,
            minimal_diagnostics=False,
        )
        elapsed = time.monotonic() - t0
    finally:
        server.stop()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "vanilla" if args.no_phase1 else "phase1"
    payload = dump_results(
        bm, OUT_DIR / f"{suffix}_seed{args.seed}.json", args.seed, elapsed
    )
    print(
        f"SEED {args.seed} DONE in {elapsed/3600:.2f}h; "
        f"mean_score={payload['mean_score']}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
