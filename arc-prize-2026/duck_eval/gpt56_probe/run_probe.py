"""GPT-5.6-sol probe: run the duck harness locally with OpenAI's gpt-5.6-sol
as the analyzer, against the bundled offline engines (competition_arcade).

Purpose (2026-07-16, user-provided OpenAI key): gold-standard transcripts on
our failure games. GPT-5.6-sol publicly solved ft09 at 87%; watching where it
diverges from Qwen3.6-27B *inside our own harness* decomposes model-capability
vs harness-bottleneck and gives the distillation target for war-v3.

LEGALITY: local development only. Nothing GPT-5.6 produces ships to Kaggle
except game-agnostic harness/prompt changes we author ourselves.

Cost control: hard ceiling via GPT56_PROBE_BUDGET_USD (tool_agent.py guard:
pre-call sticky check + per-response accounting to the shared usage file).

Usage:
  uv run python duck_eval/gpt56_probe/run_probe.py --smoke        # ft09, 10 actions, $3 cap
  uv run python duck_eval/gpt56_probe/run_probe.py                # full probe, $40 cap
  uv run python duck_eval/gpt56_probe/run_probe.py --list-games   # resolve exact IDs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # f:/kaggle/arc-prize-2026
HARNESS = ROOT / "duck_eval" / "taaf_bundle" / "src" / "ARC3-Inference"
OUT_DIR = ROOT / "runs" / "gpt56_probe"
# Repo-root .venv: has arc_agi/arcengine + .pth entries for the harness
# packages and the local re_arc shim (kaggle-data/re_arc; the real repo is
# private/unreachable). No uv sync needed.
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
ENV_FILES = ROOT / "kaggle-data" / "environment_files"

# A13 su15 re-probe: apply the game-over-continuation fix (A12 hygiene) in the
# harness child process. run_probe subprocesses `python -m inference.framework.
# run`, so the patch fires via duck_eval/continuation/usercustomize.py once its
# dir is on the child PYTHONPATH (see env setup below). Default ON; CONT_FIX=0
# disables it (translated to the CONTINUATION_DISABLE kill switch). The literal
# in-process apply() below is a no-op guard/marker for this control point.
CONT_DIR = ROOT / "duck_eval" / "continuation"
CONT_FIX_ON = os.environ.get("CONT_FIX", "1") != "0"
if CONT_FIX_ON:
    sys.path.insert(0, str(CONT_DIR))
    try:
        import continuation_patch  # noqa: E402

        continuation_patch.apply()
    except Exception:  # noqa: BLE001 - vanilla fallback, never block the probe
        pass

# Failure games (war-room grinders + the variance monster) + one control the
# duck harness handles adequately (divergence baseline).
PROBE_GAMES = ["ft09", "sb26", "su15", "lp85", "vc33"]

MODEL_ID = "gpt-5.6-sol"


def load_openai_key() -> str:
    env_file = ROOT / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for name in ("OPENAI_API_KEY", "OPENA_API_KEY"):  # user's .env has a typo'd name
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    sys.exit("no OpenAI key found in arc-prize-2026/.env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="ft09 only, 10 actions, $3 cap")
    ap.add_argument("--list-games", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="USD ceiling override")
    ap.add_argument("--max-actions", type=int, default=None)
    ap.add_argument("--games", default=None, help="comma list override")
    ap.add_argument("--tag", default=None,
                    help="experiment/usage-file tag override (fresh spend guard)")
    args = ap.parse_args()

    budget = args.budget if args.budget is not None else (3.0 if args.smoke else 40.0)
    max_actions = args.max_actions if args.max_actions is not None else (10 if args.smoke else 100)
    games = (args.games.split(",") if args.games
             else (PROBE_GAMES[:1] if args.smoke else PROBE_GAMES))
    tag = args.tag or ("smoke" if args.smoke else "full")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    usage_file = OUT_DIR / f"usage_{tag}.json"

    key = load_openai_key()
    # gpt-5.6-sol rejects function tools on chat/completions while reasoning
    # is active; proxy56.py translates chat -> /v1/responses on localhost.
    proxy_port = 8056
    proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    env = os.environ.copy()
    env.update({
        "LOCAL_ANALYZER_MODEL_ID": MODEL_ID,
        "LOCAL_ANALYZER_BASE_URL": proxy_url,
        "OPENAI_BASE_URL": proxy_url,
        "LOCAL_ANALYZER_PROVIDER": "openai-api",
        "OPENAI_PROVIDER": "openai-api",
        "LOCAL_ANALYZER_API_KEY": key,
        "OPENAI_API_KEY": key,
        # Match duck's scored conditions (setup_commands.json setup_env block):
        # same context window/eviction, tool-step policy, yield, multimodal.
        "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
        "LOCAL_ANALYZER_MAX_OUTPUT": "0",
        "LOCAL_ANALYZER_TOOL_STEPS": "0",
        "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
        "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
        "LOCAL_ANALYZER_YIELD_SECONDS": "60",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
        "RE_ARC_ENVIRONMENTS_DIR": str(ENV_FILES),
        # Spend guard (tool_agent.py; sticky + cross-process via usage file).
        "GPT56_PROBE_BUDGET_USD": str(budget),
        "GPT56_PROBE_USAGE_FILE": str(usage_file),
        "GPT56_PROBE_PRICE_IN": "1.25",   # $/M input — adjust if dashboard differs
        "GPT56_PROBE_PRICE_OUT": "10.0",  # $/M output
    })
    # Never inherit a stale vLLM pointer.
    env.pop("VIRTUAL_ENV", None)

    # Game-over-continuation (A12) into the harness child: prepend the patch dir
    # to PYTHONPATH so its usercustomize hook applies the fix at child startup.
    # CONT_FIX=0 -> flip the CONTINUATION_DISABLE kill switch instead.
    if CONT_FIX_ON:
        _pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(CONT_DIR) + (os.pathsep + _pp if _pp else "")
    else:
        env["CONTINUATION_DISABLE"] = "1"

    if args.list_games:
        cmd = [str(VENV_PY), "-m", "inference.framework.run", "--list-games",
               "--include-tags", "official",
               "--re-arc-environments-dir", str(ENV_FILES)]
        return subprocess.run(cmd, cwd=HARNESS, env=env).returncode

    exp_dir = OUT_DIR / f"experiment_{tag}"
    if exp_dir.exists():
        n = 1
        while (exp_dir.parent / f"{exp_dir.name}.bak{n}").exists():
            n += 1
        exp_dir.rename(exp_dir.parent / f"{exp_dir.name}.bak{n}")
    cmd = [
        str(VENV_PY), "-m", "inference.framework.run",
        "--model", MODEL_ID,
        "--re-arc-environments-dir", str(ENV_FILES),
        "--deployment-target", "inline",
        "--simulate-competition-arcade",
        "--game", ",".join(games),
        "--n-passes", "1",
        "--max-actions", str(max_actions),
        "--max-runtime-minutes", "20" if args.smoke else "60",
        "--concurrent-jobs", "1" if args.smoke else "5",
        "--timeout", "600",
        "--experiment-dir", str(exp_dir),
        "--run-name", f"gpt56_probe_{tag}",
    ]
    print(f"probe[{tag}]: games={games} max_actions={max_actions} budget=${budget}")
    print("cmd:", " ".join(cmd))
    proxy = subprocess.Popen(
        [str(VENV_PY), str(Path(__file__).resolve().parent / "proxy56.py"),
         "--port", str(proxy_port)],
        env=env,
    )
    try:
        import time
        import urllib.request
        for _ in range(40):
            try:
                urllib.request.urlopen(f"{proxy_url}/models", timeout=5)
                break
            except Exception:
                time.sleep(0.5)
        else:
            proxy.kill()
            sys.exit("proxy56 never became ready")
        rc = subprocess.run(cmd, cwd=HARNESS, env=env).returncode
    finally:
        proxy.terminate()
    if usage_file.exists():
        print("usage:", json.dumps(json.loads(usage_file.read_text()), indent=2))
    print(f"exit={rc}; transcripts under {exp_dir}")
    return rc


if __name__ == "__main__":
    main()
