"""A17 72B-VL canary — build smoke. CPU only, no GPU/LLM, no network.

Runtime-tests the ASSEMBLED notebook notebooks/a17-canary/arc3-a17-72b-canary
.ipynb (feedback_test_before_submit: always runtime-test the exact artifact).
The canary is the rho_action denominator run (a17_72b_screen_scope_v2.md sec3):
4 screen games, full 7920 s window, Qwen2.5-VL-72B-Instruct-AWQ, W0 composition
(duck + (f) continuation, NO warpack).

  S*  structural: 17 cells; ONLY cells 2/8/12/14 differ from the raw duckwar
      source and each equals the builder's own reconstruction byte-for-byte;
      metadata byte-parity with the duckwar family except id/title/code_file
      + the single added model_sources entry; no submission-queue contact.
  F*  functional: the cell-8 serve-config rewrite applied to the REAL
      duck_eval/taaf_bundle/setup_commands.json (anchors unique, patched
      PYSETUP compiles, 27B artifacts vetoed, 72B tokens present, reset/window
      constants untouched); tamper -> A17-CANARY FATAL (fail-LOUD, the
      inverted policy); model finder (finds VL-AWQ, REFUSES 27B-only, refuses
      ambiguity); serve asserts (tool-call round-trip, silent-27B guard, MM
      probe) against a scripted server; game filter (4-of-4, 1-drop DRIFT,
      2-drop VOID); post-run report (per-game N banners, denominator, MM-cache
      evidence, window-drift warning); heartbeat (healthy beat, post-window
      disarm).
  K*  liveness-gate escalation in a SUBPROCESS: stall -> ONE restart ->
      second stall -> LIVENESS-FAIL + os._exit(70).

Run:  uv run python duck_eval/a17/a17_canary_smoke.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB_PATH = REPO / "notebooks" / "a17-canary" / "arc3-a17-72b-canary.ipynb"
META_PATH = REPO / "notebooks" / "a17-canary" / "kernel-metadata.json"
SRC_NB_PATH = REPO / "notebooks" / "duckwar" / "arc3-duck-war.ipynb"
SRC_META_PATH = REPO / "notebooks" / "duckwar" / "kernel-metadata.json"
BUNDLE_SETUP = REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json"
BUILDER = REPO / "duck_eval" / "warpack" / "build_eval_notebook.py"

spec = importlib.util.spec_from_file_location("build_eval_notebook", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail[:90]}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail[:400]}")


def cell_src(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def extract_between(src: str, begin_token: str, end_token: str) -> str:
    lines = src.splitlines(keepends=True)
    i0 = next(i for i, l in enumerate(lines) if begin_token in l)
    i1 = next(i for i, l in enumerate(lines) if end_token in l)
    return textwrap.dedent("".join(lines[i0:i1]))


def expect_fatal(fn, *args, **kwargs) -> tuple[bool, str]:
    try:
        fn(*args, **kwargs)
        return False, "no exception raised"
    except RuntimeError as exc:
        return "A17-CANARY FATAL" in str(exc), str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"wrong exception type: {exc!r}"


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
raw = json.loads(SRC_NB_PATH.read_text(encoding="utf-8"))

# ---------------------------------------------------------------- structural
print("== structural ==")
check("S1 cell count 17", len(nb["cells"]) == 17, f"got {len(nb['cells'])}")
check("S1b cell types match source",
      [c["cell_type"] for c in nb["cells"]] == [c["cell_type"] for c in raw["cells"]])

diff_cells = [i for i in range(17) if cell_src(nb, i) != cell_src(raw, i)]
check("S2 only cells 2/8/12/14 differ from raw duckwar source",
      diff_cells == [2, 8, 12, 14], f"diff={diff_cells}")

expected_c2 = builder.EVAL_LINE + builder.EVAL_SEED_LINES_A17 + cell_src(raw, 2)
check("S3 cell 2 == eval line + A17 seed lines + raw cell 2 (byte-exact)",
      cell_src(nb, 2) == expected_c2)
check("S3b ONLY_RESET_LEVELS pin intact in cell 2 (risk A)",
      'os.environ["ONLY_RESET_LEVELS"] = "true"' in cell_src(nb, 2))

expected_c8 = cell_src(raw, 8).replace(builder.CELL8_ANCHOR, builder._cell8_a17_graft())
check("S4 cell 8 == raw cell 8 with the serve-config graft (byte-exact)",
      cell_src(nb, 8) == expected_c8)
check("S4b cell 8 embeds the builder's A17_SETUP_REWRITES verbatim",
      repr(builder.A17_SETUP_REWRITES) in cell_src(nb, 8))
check("S4c cell 8 has NO vanilla-fallback path (policy inversion)",
      "PATCH FAILED - continuing" not in cell_src(nb, 8)
      and "A17-CANARY FATAL" in cell_src(nb, 8))

check("S5 cell 12 == CELL12_W0 continuation-only graft (W0 composition)",
      cell_src(nb, 12) == builder.CELL12_W0)
check("S5b cell 12 imports neither warpack nor ledger",
      "import warpack_patch" not in cell_src(nb, 12)
      and "import ledger_patch" not in cell_src(nb, 12)
      and "import continuation_patch" in cell_src(nb, 12))

src14_expected = cell_src(raw, 14)
src14_expected = src14_expected.replace(
    builder.CELL14_DEFS_ANCHOR, builder.CELL14_A17_DEFS + builder.CELL14_DEFS_ANCHOR)
src14_expected = src14_expected.replace(
    builder.CELL14_GAMES_ANCHOR, builder.CELL14_GAMES_ANCHOR + builder.CELL14_A17_FILTER_BLOCK)
src14_expected = src14_expected.replace(builder.CELL14_TRY_ANCHOR, builder.CELL14_TRY_NEW)
src14_expected = src14_expected.replace(
    builder.CELL14_POSTRUN_ANCHOR,
    builder.CELL14_POSTRUN_ANCHOR + "            _a17_post_run_report()\n")
check("S6 cell 14 == raw cell 14 with exactly the 4 grafts (byte-exact)",
      cell_src(nb, 14) == src14_expected)
check("S6b cell 14 keeps the fast-submit dummy path",
      "_write_dummy_submission(WORKING_DIR)" in cell_src(nb, 14)
      and "FAST-SUBMIT" in cell_src(nb, 14))
check("S6c cell 14 lists the 4 versioned screen games exactly",
      '["ft09-0d8bbf25", "sb26-7fbdac44", "lp85-305b61c3", "vc33-5430563c"]'
      in cell_src(nb, 14))

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
src_meta = json.loads(SRC_META_PATH.read_text(encoding="utf-8"))
check("S7 metadata id/title/code_file",
      meta["id"] == "canivel/arc3-a17-72b-canary"
      and meta["title"] == "arc3-a17-72b-canary"
      and meta["code_file"] == "arc3-a17-72b-canary.ipynb")
check("S7b metadata model_sources == [72B Kaggle Model]",
      meta["model_sources"] == ["qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1"])
other = {k: v for k, v in meta.items() if k not in {"id", "title", "code_file", "model_sources"}}
src_other = {k: v for k, v in src_meta.items() if k not in {"id", "title", "code_file", "model_sources"}}
check("S7c all other metadata fields byte-match the duckwar family "
      "(docker_image, dataset_sources, enable_gpu, machine_shape, ...)",
      other == src_other and set(meta) == set(src_meta))

nb_text = json.dumps(nb)
check("S8 no submission-queue contact",
      "submission_queue" not in nb_text and "competitions submit" not in nb_text
      and meta["id"] != "canivel/arc3-duck-war")

required_banners = [
    "A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ",
    "A17-CANARY N(",
    "A17-CANARY rho_action_denominator=",
    "A17-CANARY tool-call-roundtrip=OK parser=hermes",
    "A17-CANARY mm-image-roundtrip=OK",
    "A17-CANARY mm_cache=NONZERO",
    "A17-CANARY gpu=",
    "A17-CANARY HEARTBEAT t=",
    "A17-CANARY LIVENESS-FAIL t=",
    "A17-CANARY games=",
    "A17-CANARY seed=1",
]
missing_banners = [b for b in required_banners if b not in nb_text]
check("S9 all required greppable banners present", not missing_banners,
      f"missing: {missing_banners}")

# ---------------------------------------------------------- cell 2 execution
print("== cell 2 execution ==")
import os  # noqa: E402  (after the structural block on purpose)

for key in ("A17_CANARY_SEED", "WARPACK_FORCE_OFFLINE_BENCH", "KAGGLE_IS_COMPETITION_RERUN"):
    os.environ.pop(key, None)
buf = io.StringIO()
ns2: dict = {}
with contextlib.redirect_stdout(buf):
    exec(cell_src(nb, 2), ns2)  # noqa: S102
out2 = buf.getvalue()
check("I1 cell 2 execs; RUN_HEAVY forced by the eval line", ns2.get("RUN_HEAVY") is True)
check("I1b A17_CANARY_SEED=1 stamped + banner printed",
      os.environ.get("A17_CANARY_SEED") == "1" and "A17-CANARY seed=1" in out2)
check("I1c ONLY_RESET_LEVELS env pinned true", os.environ.get("ONLY_RESET_LEVELS") == "true")

# ------------------------------------------------- serve-config rewrite (F*)
print("== serve-config rewrite vs REAL bundle ==")
graft = extract_between(cell_src(nb, 8),
                        "A17-CANARY BEGIN serve-config rewrite",
                        "A17-CANARY END serve-config rewrite")
gns: dict = {}
exec(graft, gns)  # noqa: S102
patch_fn = gns["_a17_patch_setup_commands"]

real_cmds = json.loads(BUNDLE_SETUP.read_text(encoding="utf-8"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    patched = patch_fn(list(real_cmds))
check("F1 rewrite applies to the REAL setup_commands.json (all 10 anchors unique)",
      "rewrite OK (10 anchors replaced" in buf.getvalue())
ptext = patched[0]
vetoes = ["qwen3_coder", "--reasoning-parser", "preserve_thinking",
          "enable_thinking", "'vrfai/Qwen3.6-27B-FP8'"]
check("F2 27B serve artifacts all vetoed", not [v for v in vetoes if v in ptext])
needs = ["'--tool-call-parser',\n        'hermes',", "'--quantization',\n        'awq_marlin',",
         "SERVED_MODEL_NAME = 'Qwen2.5-VL-72B-Instruct-AWQ'", "VLLM_MAX_MODEL_LEN = 32768",
         "_a17_serve_asserts()", "A17-CANARY gpu=", "a17_vllm_cmd.json", "_a17_find_72b_model"]
check("F2b 72B serve tokens all present", not [n for n in needs if n not in ptext])
body = ptext.split("<<'PYSETUP'\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
try:
    compile(body, "pysetup", "exec")
    compiled = True
except SyntaxError as exc:
    compiled = False
    print("   ", exc)
check("F3 patched PYSETUP script compiles", compiled)
untouched = ["ANALYZER_CONTEXT_WINDOW = 32768", "VLLM_TENSOR_PARALLEL_SIZE = 1",
             "'--enable-prefix-caching',", "'--max-model-len',",
             "WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)",
             "'--enable-auto-tool-choice',"]
check("F4 non-serve constants untouched (risk A: model + serve flags ONLY)",
      not [u for u in untouched if u not in ptext])

tampered = [real_cmds[0].replace("qwen3_coder", "qwen3_coderX")]
ok, msg = expect_fatal(patch_fn, tampered)
check("F5 tampered anchor -> A17-CANARY FATAL (fail-LOUD, no silent 27B)", ok, msg)
ok, msg = expect_fatal(patch_fn, real_cmds + real_cmds)
check("F6 wrong command-list shape -> FATAL", ok, msg)
ok, msg = expect_fatal(patch_fn, [])
check("F6b empty command list -> FATAL", ok, msg)

# ------------------------------------------------------- serve asserts (F7+)
print("== boot serve asserts (scripted server) ==")
sns: dict = {"json": json, "SERVED_MODEL_NAME": "Qwen2.5-VL-72B-Instruct-AWQ",
             "VLLM_BASE_URL": "http://x/v1"}
exec(builder.A17_SERVE_DEFS, sns)  # noqa: S102
png = __import__("base64").b64decode(sns["_a17_png_b64"]())
check("F7 _a17_png_b64 emits a real PNG (magic + 64x64 IHDR)",
      png[:8] == b"\x89PNG\r\n\x1a\n" and png[16:24] == (64).to_bytes(4, "big") * 2)


def scripted_request_json(good_model=True, tool_calls=True, mm_content=True):
    def rj(url, payload=None, timeout=0):
        if url.endswith("/models"):
            mid = "Qwen2.5-VL-72B-Instruct-AWQ" if good_model else "vrfai/Qwen3.6-27B-FP8"
            return {"data": [{"id": mid}]}
        if payload and "tools" in payload:
            if not tool_calls:
                return {"choices": [{"message": {"content": "no tools here"}}]}
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "submit_action",
                "arguments": json.dumps({"action": "ACTION6", "x": 3, "y": 7})}}]}}]}
        return {"choices": [{"message": {"content": "red" if mm_content else ""}}]}
    return rj


sns["request_json"] = scripted_request_json()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sns["_a17_serve_asserts"]()
out = buf.getvalue()
check("F8 serve asserts happy path prints all 3 banners",
      "A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ" in out
      and "A17-CANARY tool-call-roundtrip=OK parser=hermes" in out
      and "A17-CANARY mm-image-roundtrip=OK" in out)
sns["request_json"] = scripted_request_json(good_model=False)
ok, msg = expect_fatal(sns["_a17_serve_asserts"])
check("F9 wrong served model id -> FATAL (the silent-27B guard)", ok, msg)
sns["request_json"] = scripted_request_json(tool_calls=False)
ok, msg = expect_fatal(sns["_a17_serve_asserts"])
check("F10 no tool_calls in response -> FATAL (risk D)", ok, msg)
sns["request_json"] = scripted_request_json(mm_content=False)
ok, msg = expect_fatal(sns["_a17_serve_asserts"])
check("F11 empty MM probe content -> FATAL (risk E)", ok, msg)

# --------------------------------------------------------- model finder (F12+)
print("== 72B model finder ==")
fns: dict = {"os": os, "Path": Path}
finder_src = builder.A17_MODEL_FIND_BLOCK.split("\n\n\nMODEL_PATH")[0]
exec(finder_src, fns)  # noqa: S102
find = fns["_a17_find_72b_model"]

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    decoy = root / "vrfai-qwen3-6-27b-fp8-hf-snapshot"
    decoy.mkdir()
    (decoy / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3_5ForConditionalGeneration"]}), encoding="utf-8")
    (decoy / "model-00001.safetensors").write_bytes(b"x")
    os.environ["A17_INPUT_ROOT"] = str(root)

    ok, msg = expect_fatal(find)
    check("F12 27B-only inputs -> FATAL, REFUSES fallback (the poison guard)", ok, msg)

    vl = root / "qwen2.5-vl" / "transformers" / "72b-instruct-awq" / "1"
    vl.mkdir(parents=True)
    (vl / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen2_5_VLForConditionalGeneration"],
         "quantization_config": {"quant_method": "awq", "bits": 4}}), encoding="utf-8")
    (vl / "model-00001-of-00011.safetensors").write_bytes(b"x")
    check("F13 finds the VL-AWQ model dir among decoys", find() == vl, str(find()))

    vl2 = root / "dup"
    vl2.mkdir()
    (vl2 / "config.json").write_text((vl / "config.json").read_text(), encoding="utf-8")
    (vl2 / "w.safetensors").write_bytes(b"x")
    ok, msg = expect_fatal(find)
    check("F14 ambiguous (two VL-AWQ dirs) -> FATAL", ok, msg)
os.environ.pop("A17_INPUT_ROOT", None)

# ---------------------------------------------------------- game filter (F15+)
print("== 4-game screen filter ==")
GAMES4 = ["ft09-0d8bbf25", "sb26-7fbdac44", "lp85-305b61c3", "vc33-5430563c"]
filter_src = extract_between(cell_src(nb, 14),
                             "A17-CANARY BEGIN 4-game screen filter",
                             "A17-CANARY END 4-game screen filter")


def run_filter(game_ids):
    ns = {"bm": SimpleNamespace(games=[SimpleNamespace(env_name=g) for g in game_ids]),
          "A17_SCREEN_GAMES": list(GAMES4), "A17_WINDOW_S": 7920.0}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(filter_src, ns)  # noqa: S102
    return ns["bm"], buf.getvalue()


bm_out, out = run_filter(GAMES4 + ["sk48-d8078629", "tu93-cc9cbc63"])
check("F15 keeps exactly the 4 screen games, drops the other 21-game tail",
      sorted(g.env_name for g in bm_out.games) == sorted(GAMES4)
      and "A17-CANARY games=" in out and "(n=4 of 4" in out)
bm_out, out = run_filter([g for g in GAMES4 if not g.startswith("vc33")] + ["sk48-d8078629"])
check("F16 one drifted game -> DRIFT banner, run continues with 3",
      "A17-CANARY DRIFT game=vc33-5430563c MISSING" in out and len(bm_out.games) == 3)
ok, msg = expect_fatal(run_filter, GAMES4[:2] + ["sk48-d8078629"])
check("F17 two drifted games -> FATAL (>=2 drops VOID the screen)", ok, msg)

# ------------------------------------------------ heartbeat + report defs
print("== heartbeat / liveness / post-run report ==")
defs_src = extract_between(cell_src(nb, 14),
                           "A17-CANARY BEGIN heartbeat/liveness/report defs",
                           "A17-CANARY END heartbeat/liveness/report defs")


def make_defs_ns(workdir: Path) -> dict:
    ns = {"json": json, "os": os, "sys": sys, "time": time, "subprocess": subprocess,
          "Path": Path, "urlopen": None, "WORKING_DIR": workdir,
          "bm": SimpleNamespace(games=[1, 2, 3, 4])}
    exec(defs_src, ns)  # noqa: S102
    return ns


def write_bench(workdir: Path, lens=(12, 34, 56, 78), window=7920.3) -> None:
    game_runs = []
    for gid, n in zip(GAMES4, lens):
        game_runs.append({"game_id": gid, "history": [{"action": {"id": "ACTION1"}}] * n,
                          "actions_per_level": [n - 1, 1], "levels_completed": 1,
                          "final_wallclock_seconds": window})
    game_runs.append({"game_id": "sk48-d8078629", "history": [{}] * 999,
                      "actions_per_level": [999], "levels_completed": 0,
                      "final_wallclock_seconds": window})
    (workdir / "benchmark.json").write_text(
        json.dumps({"game_runs": game_runs}), encoding="utf-8")


with tempfile.TemporaryDirectory() as td:
    wd = Path(td)
    ns = make_defs_ns(wd)
    ok, msg = expect_fatal(ns["_a17_post_run_report"], wd)
    check("F18 report with no benchmark.json -> FATAL", ok, msg)

    write_bench(wd)
    (wd / "vllm-openai-server.log").write_text(
        "Engine 000: ... MM cache hit rate: 0.0%\nEngine 000: ... MM cache hit rate: 84.2%\n",
        encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["_a17_post_run_report"](wd)
    out = buf.getvalue()
    check("F19 per-game N banners exact",
          "A17-CANARY N(ft09-0d8bbf25)=12" in out and "A17-CANARY N(sb26-7fbdac44)=34" in out
          and "A17-CANARY N(lp85-305b61c3)=56" in out and "A17-CANARY N(vc33-5430563c)=78" in out)
    check("F19b rho_action denominator = sum over the 4 screen games only (extra game ignored)",
          "A17-CANARY rho_action_denominator=180 (games_present=4/4" in out)
    check("F19c MM-cache non-zero banner", "A17-CANARY mm_cache=NONZERO max_hit_rate=84.2%" in out)
    check("F19d concurrency banner reports effective concurrent games",
          "effective concurrent games this run = 4" in out)
    check("F19e no window-drift warning at ~7920 s", "window_drift" not in out)

    (wd / "vllm-openai-server.log").write_text(
        "MM cache hit rate: 0.0%\nMM cache hit rate: 0.0%\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["_a17_post_run_report"](wd)
    check("F20 all-zero MM cache -> WARN mm_cache=ZERO (discard-grade signal)",
          "A17-CANARY WARN mm_cache=ZERO" in buf.getvalue())

    write_bench(wd, window=7000.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["_a17_post_run_report"](wd)
    check("F21 window 7000 s -> WARN window_drift (null-void flag, amendment sec7.2)",
          "A17-CANARY WARN window_drift" in buf.getvalue())

    # healthy heartbeat, in-process (no stall escalation possible: huge stall)
    write_bench(wd, lens=(25, 25, 25, 25))
    (wd / "vllm-openai-server.log").write_text(
        "INFO [loggers.py:259] Engine 000: Avg prompt throughput: 100.0 tokens/s, "
        "Avg generation throughput: 123.4 tokens/s, Running: 4 reqs, Waiting: 0 reqs, "
        "GPU KV cache usage: 20.0%, Prefix cache hit rate: 1.0%, MM cache hit rate: 7.4%\n",
        encoding="utf-8")
    ns["A17_HEARTBEAT_INTERVAL_S"] = 0.05
    ns["A17_STALL_S"] = 1e9
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["_a17_start_heartbeat"](wd)
        time.sleep(0.4)
        ns["A17_HEARTBEAT_INTERVAL_S"] = 1e6
        time.sleep(0.15)
    out = buf.getvalue()
    # actions_total sums ALL game_runs (whole-bench progress observable):
    # 4 x 25 screen games + the 999-action extra game = 1099.
    check("F22 heartbeat line with elapsed/actions/gen_tps/reqs fields",
          "A17-CANARY HEARTBEAT t=" in out and "actions_total=1099" in out
          and "gen_tps=123.4" in out and "running_reqs=4" in out and "restarts=0" in out)
    check("F22b healthy beat never escalates", "LIVENESS" not in out)

    # post-window disarm, in-process (stall always, disarm threshold at -1)
    ns2 = make_defs_ns(wd)
    ns2["A17_HEARTBEAT_INTERVAL_S"] = 0.05
    ns2["A17_STALL_S"] = 0.0
    ns2["A17_KILL_DISARM_S"] = -1.0
    (wd / "vllm-openai-server.log").write_text("no engine lines\n", encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns2["_a17_start_heartbeat"](wd)
        time.sleep(0.4)
        ns2["A17_HEARTBEAT_INTERVAL_S"] = 1e6
        time.sleep(0.15)
    out = buf.getvalue()
    check("F23 stalled-but-post-window -> POSTWINDOW banner, NO kill (artifact protection)",
          "A17-CANARY LIVENESS-STALL-POSTWINDOW" in out and "LIVENESS-FAIL" not in out)

# ------------------------------------------- liveness escalation (subprocess)
print("== liveness-gate escalation (subprocess) ==")
driver = r"""
import contextlib, json, os, subprocess, sys, textwrap, time
from pathlib import Path
from types import SimpleNamespace

nb_path, workdir = Path(sys.argv[1]), Path(sys.argv[2])
nb = json.loads(nb_path.read_text(encoding="utf-8"))
src14 = "".join(nb["cells"][14]["source"])
lines = src14.splitlines(keepends=True)
i0 = next(i for i, l in enumerate(lines) if "A17-CANARY BEGIN heartbeat/liveness/report defs" in l)
i1 = next(i for i, l in enumerate(lines) if "A17-CANARY END heartbeat/liveness/report defs" in l)
defs = textwrap.dedent("".join(lines[i0:i1]))
ns = {"json": json, "os": os, "sys": sys, "time": time, "subprocess": subprocess,
      "Path": Path, "urlopen": None, "WORKING_DIR": workdir,
      "bm": SimpleNamespace(games=[1, 2, 3, 4])}
exec(defs, ns)
ns["A17_HEARTBEAT_INTERVAL_S"] = 0.05
ns["A17_STALL_S"] = 0.0
ns["A17_KILL_DISARM_S"] = 1e9
def stub_restart():
    print("STUB-RESTART-CALLED", flush=True)
ns["_a17_restart_vllm"] = stub_restart
ns["_a17_start_heartbeat"](workdir)
time.sleep(10)
print("SHOULD-NOT-REACH", flush=True)
os._exit(1)
"""
with tempfile.TemporaryDirectory() as td:
    drv = Path(td) / "driver.py"
    drv.write_text(driver, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(drv), str(NB_PATH), td],
                          capture_output=True, text=True, timeout=120)
out = proc.stdout
check("K1 first stall -> LIVENESS-STALL + ONE restart attempt",
      "A17-CANARY LIVENESS-STALL t=" in out and "STUB-RESTART-CALLED" in out
      and "A17-CANARY LIVENESS-RESTART t=" in out and "restarts=1 OK" in out)
check("K2 second stall -> LIVENESS-FAIL banner",
      "A17-CANARY LIVENESS-FAIL t=" in out and "second stall" in out)
check("K3 kernel dies LOUDLY with exit code 70 (never a silent burn)",
      proc.returncode == 70 and "SHOULD-NOT-REACH" not in out,
      f"rc={proc.returncode}")

# ----------------------------------------------------------- builder rebuild
print("== builder idempotence ==")
before_nb = NB_PATH.read_bytes()
before_meta = META_PATH.read_bytes()
proc = subprocess.run([sys.executable, str(BUILDER), "--a17-canary"],
                      capture_output=True, text=True, cwd=str(REPO))
check("B1 builder re-run exits 0", proc.returncode == 0, proc.stderr[-300:])
check("B2 rebuild is byte-identical (artifact in sync with builder)",
      NB_PATH.read_bytes() == before_nb and META_PATH.read_bytes() == before_meta)

print(f"\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
