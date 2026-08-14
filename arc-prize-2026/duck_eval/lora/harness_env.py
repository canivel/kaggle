"""Shared bootstrap for the LoRA-lane local rig (CPU only, no LLM, no network).

Puts the duck harness packages on ``sys.path`` exactly like
``duck_eval/warpack/smoke_test.py`` and ``duck_eval/gpt56_probe/run_probe.py``
do, and resolves the two environment pools we train/hold out on:

- ``PUBLIC_ENVS``   -- the 25 scored games (``kaggle-data/environment_files``).
                      **Never used as a training source.** Evaluation only.
- ``INTERNAL_ENVS`` -- 190 re-arc-3 environments shipped inside Tufa's own
                      public bundle ``jeroencottaar/taaf-kaggle-source``
                      (pulled 08-13 into ``runs/harness_diff_0813/ds/``).
                      165 of these families do not exist in the public 25.

The split between the two pools is what makes the adapter's generalization
claim testable: train only on families that are not scored, measure on
families the adapter has never seen.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
PUBLIC_ENVS = REPO / "kaggle-data" / "environment_files"
INTERNAL_ENVS = (
    REPO
    / "runs"
    / "harness_diff_0813"
    / "ds"
    / "jeroencottaar_taaf-kaggle-source"
    / "src"
    / "re-arc-3"
    / "re_arc"
    / "environment_files"
)
OUT_DIR = REPO / "runs" / "lora_lane"

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Idempotent: harness packages on sys.path + the env vars the agent reads."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for path in (
        BUNDLE / "ARC3-Inference",
        BUNDLE / "tufa-arc-agi-framework" / "src",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # GameAPI._start_game sets ONLY_RESET_LEVELS itself after arcade.make; a
    # leftover value from a previous game in-process breaks play registration.
    os.environ.pop("ONLY_RESET_LEVELS", None)
    # ToolAgent freezes LOCAL_ANALYZER_* into module globals at import time, so
    # these must be set before `inference.agent.tool_agent` is first imported.
    # Values are copied verbatim from the duck's scored setup_commands.json.
    defaults = {
        "LOCAL_ANALYZER_MODEL_ID": "vrfai/Qwen3.6-27B-FP8",
        "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:9/v1",  # never dialled
        "LOCAL_ANALYZER_PROVIDER": "vllm",
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
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    _BOOTSTRAPPED = True


def arcade_spec(environments_dir: Path):
    bootstrap()
    import arc_agi
    from taaf.game_api import ArcadeSpec

    return ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )


def list_environments(environments_dir: Path) -> list[str]:
    bootstrap()
    import arc_agi

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    return sorted(env.game_id for env in arcade.available_environments)


def family(game_id: str) -> str:
    """`breakout-0001` -> `breakout`; `ar25-0c556536` -> `ar25`."""
    return str(game_id).rsplit("-", 1)[0]
