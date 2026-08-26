"""exec-WM CPU smoke: the ENTIRE loop proven end-to-end on the real machinery.

Runs the REAL competition wheels (arc_agi + arcengine + taaf) with the REAL
HarnessSolver/ToolAgent out of a patched COPY of the anim-20260807 vehicle
bundle, a stdlib stub LLM server (no litellm, no openai SDK), and the REAL
environment files. Nothing here touches the staged notebook, the queue, or any
lane's artifacts; all work happens in a temp dir.

Checks
  S1  patch applies to a vehicle copy (anchors count==1); double-apply and
      drifted-bundle both raise ExecWMFatalDrift (drift dies loudly)
  S2  GRACEFUL DEGRADATION: the scripted ExampleGame still completes (WON 2/2)
      under the WRAPPED harness, with the stock stub agent doing the clearing
  S3  kill-switch ARC3_EXECWM=0 leaves the stock analyzer untouched
  S4  REAL GAME: ls20 level 1 is CLEARED VIA PLAN with ZERO LLM tokens
      (PHASE E -> I(mined) -> V -> P on the real simulator)
  S5  fallback on a non-mover game (bp35): no verified model -> stock agent
      runs (stub tokens observed), run report says fallback
  S6  PHASE I LLM half: llm_fill parses a constrained JSON reply from a stub
      server; garbage replies return {} (never raises)
  S7  the sealed scorer reads the S4 artifact's delivery instruments
  S8  PHASE V CAN REFUSE: a deliberately wrong rule is rejected by
      prequential verification on real recorded ls20 transitions

Usage:  uv run python duck_eval/execwm/ewm_smoke.py [--fast]
        (--fast skips S4/S5/S7, the real-game minutes)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import ewm_patch  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""),
          flush=True)


# ---------------------------------------------------------------------------
# stub LLM server (OpenAI-shaped /chat/completions; harness uses raw requests)
# ---------------------------------------------------------------------------
class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    script: list[str] = []
    reply_json: str | None = None
    seen = {"n": 0}

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        i = _StubHandler.seen["n"]
        _StubHandler.seen["n"] += 1
        if _StubHandler.reply_json is not None:
            msg = {"role": "assistant", "content": _StubHandler.reply_json}
        else:
            code = (_StubHandler.script[i] if i < len(_StubHandler.script)
                    else 'print("IDLE")')
            msg = {"role": "assistant", "content": "stub",
                   "tool_calls": [{"id": f"c{i}", "type": "function",
                                   "function": {"name": "python",
                                                "arguments": json.dumps({"code": code})}}]}
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": "stub",
               "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def _start_stub(script=None, reply_json=None):
    _StubHandler.script = script or []
    _StubHandler.reply_json = reply_json
    _StubHandler.seen = {"n": 0}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


# ---------------------------------------------------------------------------
# child runner: each harness run happens in a subprocess so sys.modules stays
# clean between the wrapped, unwrapped, and real-game scenarios.
# ---------------------------------------------------------------------------
_CHILD = r'''
import asyncio, io, json, os, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["bundle"])
os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
os.environ["OPENAI_API_KEY"] = "stub"
os.environ["MULTIMODAL_CONTEXT"] = "0"
for k, v in cfg.get("env", {}).items():
    os.environ[k] = v
os.chdir(cfg["cwd"])
import taaf.benchmark
from inference.framework.solver import HarnessSolver
solver = HarnessSolver(label="ewm-smoke", model="stub-model", concurrency=1,
                       max_actions_per_game=cfg["max_actions"],
                       kaggle_enable_vllm=False, start_local_server=False,
                       analyzer_timeout=20.0, animation_awareness=True,
                       hard_noop_guard=True, save_request_logs=False,
                       max_runtime_s_per_game=cfg["max_runtime"])
if cfg["game"] == "example":
    import taaf.game_examples
    game = taaf.game_examples.ExampleGame(label="ewm_scripted")
else:
    import taaf.game_api
    game = taaf.game_api.GameAPI(env_name=cfg["game"])
job = Path(cfg["job"])
bm = taaf.benchmark.Benchmark(label="ewm-smoke", games=[game], solver=solver,
                              n_passes=1, job_dir=job, periodic_save_interval_s=1e9)
try:
    asyncio.run(bm.run())
except UnicodeEncodeError:
    pass  # taaf diagnostics.html writer under a cp1252 locale; benchmark.json is already written
analyzer_types = []
try:
    analyzer_types.append(type(solver)._make_analyzer.__qualname__)
except Exception:
    pass
bench = json.loads((job / "benchmark.json").read_text(encoding="utf-8"))
run = bench["game_runs"][0]
print("EWM_SMOKE_RESULT " + json.dumps({
    "state": run.get("state"),
    "levels_completed": run.get("levels_completed"),
    "actions_per_level": run.get("actions_per_level"),
    "solver_note": run.get("solver_note"),
}))
'''


def _run_child(bundle: Path, job: Path, game: str, base_url: str, *,
               max_actions=40, max_runtime=120.0, env=None, timeout=420):
    cfg = {"bundle": str(bundle), "job": str(job), "game": game,
           "base_url": base_url, "max_actions": max_actions,
           "max_runtime": max_runtime, "cwd": str(REPO), "env": env or {}}
    child = job.parent / f"child_{job.name}.py"
    child.write_text(_CHILD, encoding="utf-8")
    r = subprocess.run([sys.executable, str(child), json.dumps(cfg)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout,
                       env={**os.environ, "PYTHONUTF8": "1"})
    out = (r.stdout or "") + (r.stderr or "")
    for line in (r.stdout or "").splitlines():
        if line.startswith("EWM_SMOKE_RESULT "):
            return json.loads(line[len("EWM_SMOKE_RESULT "):]), out
    raise RuntimeError(f"smoke child produced no result (rc={r.returncode})\n{out[-3000:]}")


def main() -> int:
    fast = "--fast" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="ewmsmoke-"))
    bundle = work / "ARC3-Inference"
    shutil.copytree(ewm_patch.VEHICLE_BUNDLE, bundle)

    # ---- S1 patch application + drift negative controls ----
    try:
        info = ewm_patch.apply_execwm_patch(bundle)
        check("S1.apply", True, f"exec_wm sha {info['exec_wm_sha']}")
    except Exception as e:
        check("S1.apply", False, str(e))
        return _finish(work)
    try:
        ewm_patch.apply_execwm_patch(bundle)
        check("S1b.double_apply_raises", False)
    except ewm_patch.ExecWMFatalDrift:
        check("S1b.double_apply_raises", True)
    drift = work / "drifted" / "ARC3-Inference"
    shutil.copytree(ewm_patch.VEHICLE_BUNDLE, drift)
    sp = drift / "inference" / "framework" / "solver.py"
    sp.write_text(sp.read_text(encoding="utf-8").replace(
        "    def _make_analyzer(\n", "    def _make_analyzer_x(\n"), encoding="utf-8")
    try:
        ewm_patch.apply_execwm_patch(drift)
        check("S1c.drift_raises", False)
    except ewm_patch.ExecWMFatalDrift:
        check("S1c.drift_raises", True)

    # ---- S2 graceful degradation on the scripted game ----
    # The stub floor agent is STATE-ADAPTIVE (like the real 27B), because
    # exec-WM's E-phase probes legitimately move the game state before the
    # fallback hands over.
    adaptive = (
        'for _ in range(9):\n'
        '    r = action("ACTION1")\n'
        '    if r.get("level_completed") or r.get("run_complete"): break\n'
        'for _ in range(9):\n'
        '    r = action("ACTION2")\n'
        '    if r.get("run_complete"): break\n'
    )
    srv, base_url = _start_stub(script=[adaptive, adaptive, adaptive])
    try:
        res, out = _run_child(bundle, work / "job_s2", "example", base_url,
                              max_actions=40, max_runtime=90.0)
        check("S2.wrapped_still_wins",
              res["state"] == "won" and res["levels_completed"] == 2,
              f"state={res['state']} lc={res['levels_completed']}")
        check("S2.armed_marker", "[execwm] armed" in out)
        check("S2.fallback_logged", "fallback" in out,
              "exec-WM handed the scripted game to the stock agent")
    finally:
        srv.shutdown()

    # ---- S3 kill-switch ----
    srv, base_url = _start_stub(script=[
        'action(["ACTION1","ACTION1","ACTION1"])',
        'action(["ACTION2","ACTION2","ACTION2"])',
    ])
    try:
        res, out = _run_child(bundle, work / "job_s3", "example", base_url,
                              max_actions=40, max_runtime=90.0,
                              env={"ARC3_EXECWM": "0"})
        check("S3.killswitch_wins",
              res["state"] == "won" and res["levels_completed"] == 2)
        check("S3.killswitch_silent", "[execwm] armed" not in out,
              "no exec-WM marker with ARC3_EXECWM=0")
    finally:
        srv.shutdown()

    s4_job = work / "job_s4"
    if not fast:
        # ---- S4 the real game: E -> I -> V -> P clears ls20 level 1 ----
        srv, base_url = _start_stub(script=[])
        try:
            res, out = _run_child(bundle, s4_job, "ls20-9607627b", base_url,
                                  max_actions=400, max_runtime=210.0,
                                  env={"ARC3_EXECWM_LLM": "0"}, timeout=600)
            check("S4.level1_cleared_via_plan", "CLEARED via=plan" in out,
                  f"lc={res['levels_completed']}")
            check("S4.lc_ge_1", int(res["levels_completed"] or 0) >= 1)
            check("S4.verified_model", "VERIFIED moves=4" in out)
            reports = list((s4_job / "execwm").glob("*.json"))
            rep = json.loads(reports[0].read_text(encoding="utf-8")) if reports else {}
            check("S4.report_written", bool(rep.get("armed")),
                  f"{len(reports)} report file(s)")
            check("S4.zero_llm", int(rep.get("llm_calls") or 0) == 0)
        finally:
            srv.shutdown()

        # ---- S5 fallback on a non-mover game ----
        srv, base_url = _start_stub(script=[])
        try:
            res, out = _run_child(bundle, work / "job_s5", "bp35-0a0ad940",
                                  base_url, max_actions=60, max_runtime=45.0,
                                  env={"ARC3_EXECWM_LLM": "0"}, timeout=420)
            check("S5.fallback_latched", "fallback reason=no-verified-model" in out)
            check("S5.stock_agent_ran", _StubHandler.seen["n"] > 0,
                  f"{_StubHandler.seen['n']} stub LLM round-trips")
        finally:
            srv.shutdown()

    # ---- S6 PHASE I LLM half ----
    sys.path.insert(0, str(bundle / "inference" / "agent"))
    import importlib
    import exec_wm as ewm
    importlib.reload(ewm)
    srv, base_url = _start_stub(
        reply_json='{"ACTION3": {"type": "translate", "dr": 0, "dc": 1}}')
    try:
        hints, tokens = ewm.llm_fill({"ACTION3"}, ["ACTION3: 4 cells changed"],
                                     {"base_url": base_url, "model": "m", "api_key": "k"})
        check("S6.llm_parse", hints.get("ACTION3", {}).get("dc") == 1, str(hints))
    finally:
        srv.shutdown()
    srv, base_url = _start_stub(reply_json="I have no idea, sorry!")
    try:
        hints, _ = ewm.llm_fill({"ACTION3"}, ["evidence"],
                                {"base_url": base_url, "model": "m", "api_key": "k"})
        check("S6b.llm_garbage_safe", hints == {})
    finally:
        srv.shutdown()
    hints, _ = ewm.llm_fill({"ACTION3"}, ["evidence"],
                            {"base_url": "http://127.0.0.1:9", "model": "m",
                             "api_key": "k"})
    check("S6c.llm_dead_server_safe", hints == {})

    # ---- S7 scorer reads the S4 artifact ----
    if not fast and s4_job.is_dir():
        sys.path.insert(0, str(HERE))
        import execwm_score
        deliv = execwm_score.delivery(s4_job)
        check("S7.delivery_reads_reports",
              deliv["source"] == "reports"
              and deliv["levels_cleared_by_plan"] >= 1
              and deliv["rules_verified"] >= 2,
              json.dumps({k: deliv[k] for k in
                          ("levels_cleared_by_plan", "rules_verified",
                           "levels_fallback")}))
        st = subprocess.run([sys.executable, str(HERE / "execwm_score.py"),
                             "--selftest"], capture_output=True, text=True)
        check("S7b.scorer_selftest", st.returncode == 0,
              (st.stdout or "").strip().splitlines()[-1] if st.stdout else "")

    # ---- S8 verification can refuse a wrong rule (real engine transitions) ----
    try:
        import numpy as np
        import arc_agi, arcengine
        os.chdir(REPO)
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        env = arc.make("ls20-9607627b")

        def settled():
            f = np.array(env.observation_space.frame)
            g = f[-1] if f.ndim == 3 else f
            return tuple(tuple(int(x) for x in row) for row in g)

        history = [{"action": "", "grid": settled(), "level": 1}]
        for a in [1, 2, 3, 4] * 4:
            env.step(arcengine.GameAction.from_name(f"ACTION{a}"))
            history.append({"action": f"ACTION{a}", "grid": settled(), "level": 1})
        mask = ewm.HudMask()
        for i in range(1, len(history)):
            mask.observe(history[i - 1]["grid"], history[i]["grid"])
        transitions = ewm.transitions_from_history(history)
        wm = ewm.WorldModel()
        wm.mine(transitions, mask)
        wm.verify(transitions, mask)
        right = {a: r.verified for a, r in wm.rules.items()}
        for r in wm.rules.values():
            if r.kind == "move" and r.delta:
                r.delta = (-r.delta[0], -r.delta[1])
        wm.verify(transitions, mask)
        wrong_any = any(r.verified for r in wm.rules.values() if r.kind == "move")
        check("S8.verifier_accepts_right", any(right.values()), str(right))
        check("S8.verifier_refuses_wrong", not wrong_any,
              "flipped deltas all rejected")
    except Exception as e:
        check("S8.verifier_refuses_wrong", False, f"{type(e).__name__}: {e}")

    return _finish(work)


def _finish(work: Path) -> int:
    ok = sum(1 for _, c, _ in RESULTS if c)
    fail = sum(1 for _, c, _ in RESULTS if not c)
    print(f"\newm_smoke: {ok} pass / {fail} fail")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.exit(main())
