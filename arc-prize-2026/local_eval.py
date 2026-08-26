"""
Local ARC-AGI-3 agent evaluator.

Adapted from Duc-Cuong Le's "ARC3 Agent Evaluation and Recording Viewer" (Kaggle
discussion 687648). Runs ANY agent through the OFFICIAL main.py command path on
local data — closer to real evaluation than custom simulators.

Usage:
    uv run python local_eval.py <agent_path.py> <agent_class> [--game <id>] [--desc <slug>]

Examples:
    # Run our v19 agent on all games (full eval)
    uv run python local_eval.py notebooks/forge_agent/forge_v35_tips.py MyAgent --desc v19-test

    # Run on a single game for fast iteration
    uv run python local_eval.py notebooks/forge_agent/forge_v35_tips.py MyAgent --game ft09 --desc v19-ft09

Outputs (per run dir under runs/):
    runs-YYYYMMDD-HHMMSS-<desc>/
        recordings/         (JSONL frame recordings per game)
        summary.csv         (per-game stats)
        summary.txt         (human-readable summary)
        scorecard.json      (final official scorecard)
        run.log             (full subprocess stdout/stderr)
        <agent>.py          (immutable copy of the agent that was tested)
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------- LOCAL PATHS (Windows / arc-prize-2026) ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
KAGGLE_DATA = PROJECT_ROOT / "kaggle-data"
ENV_DIR = KAGGLE_DATA / "environment_files"
REPO_SRC = KAGGLE_DATA / "ARC-AGI-3-Agents"
RUNS_BASE = PROJECT_ROOT / "runs"


def _slugify(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _make_run_dir(base: Path, description: str | None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slugify(description)
    name = f"runs-{ts}" + (f"-{slug}" if slug else "")
    p = base / name
    if not p.exists():
        p.mkdir(parents=True)
        return p
    for i in range(1, 1000):
        c = base / f"{name}-{i:02d}"
        if not c.exists():
            c.mkdir(parents=True)
            return c
    raise RuntimeError("could not allocate run dir")


def _free_port(host: str = "127.0.0.1", start: int = 8765) -> int:
    p = start
    for _ in range(200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                p += 1
    raise RuntimeError("no free port")


def _start_gateway(host: str, port: int, env_dir: Path) -> ThreadingHTTPServer:
    """Tiny HTTP server providing /api/games — the agent framework expects this."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/games":
                games = []
                if env_dir.exists():
                    for p in sorted(env_dir.iterdir()):
                        if p.is_dir():
                            games.append({"game_id": p.name})
                payload = json.dumps(games).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_):
            return

    srv = ThreadingHTTPServer((host, port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _extract_scorecard(log_lines: list[str]) -> dict | None:
    """Find FINAL SCORECARD REPORT JSON block in logs."""
    started = False
    balance = 0
    chunks = []
    capture = False
    for raw in log_lines:
        line = re.sub(r"^\d{4}-\d{2}-\d{2}.*?\|\s*INFO\s*\|\s*", "", raw.rstrip("\n"))
        if "FINAL SCORECARD REPORT" in line:
            capture = True
            continue
        if not capture:
            continue
        if not started:
            if "{" in line:
                idx = line.find("{")
                piece = line[idx:]
                chunks.append(piece)
                balance += piece.count("{") - piece.count("}")
                started = True
                if balance == 0:
                    break
        else:
            chunks.append(line)
            balance += line.count("{") - line.count("}")
            if balance == 0:
                break
    if not chunks:
        return None
    try:
        return json.loads("\n".join(chunks))
    except Exception:
        return None


def run_eval(
    agent_src: Path,
    agent_class: str,
    agent_cli_name: str = "myagent",
    run_game: str = "all",
    description: str | None = None,
    timeout_s: int | None = None,
) -> dict:
    if not ENV_DIR.exists():
        raise FileNotFoundError(f"ENV_DIR missing: {ENV_DIR}")
    if not REPO_SRC.exists():
        raise FileNotFoundError(f"REPO_SRC missing: {REPO_SRC}")
    if not agent_src.exists():
        raise FileNotFoundError(f"agent_src missing: {agent_src}")

    host = "127.0.0.1"
    port = _free_port()
    root_url = f"http://{host}:{port}"

    run_dir = _make_run_dir(RUNS_BASE, description)
    rec_dir = run_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)

    # immutable copy of the agent we're testing
    agent_copy = run_dir / agent_src.name
    shutil.copy(agent_src, agent_copy)

    # writable repo clone
    repo_dst = run_dir / "ARC-AGI-3-Agents"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(REPO_SRC, repo_dst)

    # install agent into templates
    mod_name = agent_src.stem
    dst_agent = repo_dst / "agents" / "templates" / f"{mod_name}.py"
    shutil.copy(agent_src, dst_agent)

    # rewrite agents/__init__.py minimally
    init_text = textwrap.dedent(f"""
    from typing import Type
    from dotenv import load_dotenv

    from .agent import Agent, Playback
    from .swarm import Swarm
    from .templates.random_agent import Random
    from .templates.{mod_name} import {agent_class}

    load_dotenv()

    AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{
        "random": Random,
        "{agent_cli_name}": {agent_class},
    }}
    """).strip() + "\n"
    (repo_dst / "agents" / "__init__.py").write_text(init_text, encoding="utf-8")

    # offline .env
    env_text = textwrap.dedent(f"""
    OPERATION_MODE=OFFLINE
    ENVIRONMENTS_DIR={ENV_DIR}
    RECORDINGS_DIR={rec_dir}

    SCHEME=http
    HOST={host}
    PORT={port}
    ARC_BASE_URL={root_url}
    ARC_API_KEY=offline
    """).strip() + "\n"
    (repo_dst / ".env").write_text(env_text, encoding="utf-8")

    # CRITICAL: agents may look for game source under multiple paths:
    # 1. relative `environment_files/<gid>/...` from CWD (forge_v35 style)
    # 2. `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files/<gid>/...`
    #    (ashvin/Chronos style — hardcoded Kaggle-only paths)
    # We materialize BOTH locally so any agent code finds the game source.

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(str(src), str(dst), target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            pass
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           check=True, capture_output=True)
            return
        except Exception:
            shutil.copytree(src, dst)

    # 1. CWD-relative path inside repo
    _link_or_copy(ENV_DIR, repo_dst / "environment_files")

    # 2. Kaggle-style absolute path. On Windows, "/kaggle/..." resolves to the
    #    current drive's root (F:/kaggle/...). Create the structure there.
    kaggle_competition_root = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3")
    _link_or_copy(ENV_DIR, kaggle_competition_root / "environment_files")
    # Some agents also reference the wheels/agents dirs under /kaggle/input
    _link_or_copy(REPO_SRC, kaggle_competition_root / "ARC-AGI-3-Agents")
    wheels_src = KAGGLE_DATA / "arc_agi_3_wheels"
    if wheels_src.exists():
        _link_or_copy(wheels_src, kaggle_competition_root / "arc_agi_3_wheels")

    print(f"[eval] env paths materialized: {repo_dst}/environment_files + {kaggle_competition_root}")

    # spawn gateway
    srv = _start_gateway(host, port, ENV_DIR)
    print(f"[eval] gateway: {root_url}/api/games")

    # build subprocess command
    cmd = [sys.executable, "main.py", "--agent", agent_cli_name]
    if run_game != "all":
        cmd += ["--game", run_game]

    log_path = run_dir / "run.log"
    print(f"[eval] running: {' '.join(cmd)} (cwd={repo_dst})")
    print(f"[eval] log -> {log_path}")
    print(f"[eval] recordings -> {rec_dir}")

    t0 = time.time()
    log_lines: list[str] = []
    proc = None

    def _stream_output():
        """Background thread: read subprocess stdout into log_lines + run.log."""
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace")
                logf.write(line)
                logf.flush()
                log_lines.append(line)
                # Mirror to our stdout (bytes mode survives Windows cp1252)
                try:
                    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
                    sys.stdout.buffer.flush()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        logf = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_dst),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            )
            stream_t = threading.Thread(target=_stream_output, daemon=True)
            stream_t.start()

            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                print(f"[eval] TIMEOUT after {timeout_s}s — killing subprocess")
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            # Give the streamer a moment to drain remaining output
            stream_t.join(timeout=5)
        finally:
            logf.close()
    finally:
        try:
            srv.shutdown()
        except Exception:
            pass
        try:
            srv.server_close()
        except Exception:
            pass

    duration = time.time() - t0
    rc = proc.returncode

    # parse scorecard from logs
    sc = _extract_scorecard(log_lines)
    if sc:
        sc_path = run_dir / "scorecard.json"
        sc_path.write_text(json.dumps(sc, indent=2), encoding="utf-8")
        print(f"[eval] scorecard -> {sc_path}")
    else:
        print("[eval] WARNING: no scorecard found in logs")

    # write a brief human summary
    summary = run_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Agent: {agent_src.name} ({agent_class})\n")
        f.write(f"Game(s): {run_game}\n")
        f.write(f"Description: {description or '-'}\n")
        f.write(f"Duration: {duration:.1f}s\n")
        f.write(f"Subprocess exit: {rc}\n\n")
        if sc:
            score = sc.get("score") or sc.get("total_score")
            f.write(f"FINAL SCORE: {score}\n\n")
            f.write(json.dumps(sc, indent=2))
        else:
            f.write("(no scorecard parsed)\n")
    print(f"[eval] summary  -> {summary}")

    return {
        "run_dir": str(run_dir),
        "duration": duration,
        "exit": rc,
        "scorecard": sc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("agent_src", help="Path to agent .py file")
    p.add_argument("agent_class", help="Class name to register (e.g. MyAgent)")
    p.add_argument("--game", default="all", help="Game id (e.g. ft09) or 'all'")
    p.add_argument("--desc", default=None, help="Slug for run dir")
    p.add_argument("--cli-name", default="myagent")
    p.add_argument("--timeout", type=int, default=None,
                   help="Subprocess timeout (s); None = no limit")
    args = p.parse_args()

    res = run_eval(
        agent_src=Path(args.agent_src),
        agent_class=args.agent_class,
        agent_cli_name=args.cli_name,
        run_game=args.game,
        description=args.desc,
        timeout_s=args.timeout,
    )

    print()
    print(f"[done] run_dir = {res['run_dir']}")
    print(f"[done] duration = {res['duration']:.1f}s")
    print(f"[done] exit = {res['exit']}")
    if res["scorecard"]:
        print(f"[done] scorecard summary keys = {list(res['scorecard'].keys())}")


if __name__ == "__main__":
    main()
