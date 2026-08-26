"""Deep-test suite for `arc3-q38-private-eval` — run BEFORE any push is even proposed.

Tests the REAL artifacts, not stand-ins (feedback_test_before_submit,
feedback_audit_the_instrument):
  * the sha-pinned 08-15 bundle vendored at duck_eval/private/bundle_20260815/
    (manifest md5 798246d79122856ca1806c9445a7e57b, 75 files)
  * the REAL competition wheels in the repo .venv (arc_agi + arcengine)
  * the ACTUAL cell sources of the built notebook (extracted by anchor, exec'd)

Run with the repo venv so taaf/arcengine imports work:
    .venv/Scripts/python.exe duck_eval/private/private_smoke.py

Groups:
  S1  builder gates green for all four arm variants (temp output dir)
  S2  staged notebook: 12 cells, all code cells compile, base variant stamped
  S3  bundle pins: manifest md5 + setup_commands sha + tool_agent sha
  S4  setup-command patch vs the REAL setup_commands.json, per arm
      (model retarget always 1x each; edge1 ON patches both ints; OFF touches neither)
  S5  bundle-label assert block (real bundle passes; tampered label refuses)
  S6  banner block: correct env passes AND its output matches the SCORER's regex
      (instrument alignment); wrong env refuses
  S7  edge-2 contract cell vs the REAL 08-15 tool_agent: flag ON installs, preflight
      passes, a solver-style ToolAgent constructed AFTER the patch carries the
      contract, audit json written; flag OFF is zero-touch (module not even imported,
      checked in a subprocess)
  S8  unpickle benchmark_initial.pkl + deploy_target.pkl with real wheels; cell-9
      arcade_spec recovery path works
  S9  scorer selftest (32 checks) passes as a subprocess
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = HERE / "bundle_20260815"
NB_PATH = REPO / "notebooks" / "q38-private-eval" / "arc3-q38-private-eval.ipynb"

BUNDLE_MANIFEST_MD5 = "798246d79122856ca1806c9445a7e57b"
SETUP_SHA = "7ca43b0b700cc4c3"
TOOL_AGENT_SHA = "c53df973c3378337"

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))


def cell_src(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def extract_defs(source: str, names: set[str]) -> str:
    """Extract exact function defs from a cell source by AST (test the REAL code)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out.append("".join(lines[node.lineno - 1: node.end_lineno]))
    return "\n".join(out)


def extract_span(source: str, start_anchor: str, end_anchor: str) -> str:
    i = source.index(start_anchor)
    j = source.index(end_anchor, i)
    return source[i: j + len(end_anchor)]


# ---------------------------------------------------------------------------
print("S1: builder gates, all four arm variants")
import importlib.util

spec = importlib.util.spec_from_file_location("build_private_eval", HERE / "build_private_eval.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    real_out_dir, real_nb, real_meta = builder.OUT_DIR, builder.OUT_NB_PATH, builder.OUT_META_PATH
    try:
        for e1, e2 in ((False, False), (True, False), (False, True), (True, True)):
            builder.OUT_DIR = tmp / f"out_{e1}_{e2}"
            builder.OUT_NB_PATH = builder.OUT_DIR / "arc3-q38-private-eval.ipynb"
            builder.OUT_META_PATH = builder.OUT_DIR / "kernel-metadata.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                builder.build(edge1=e1, edge2=e2)
            built = json.loads(builder.OUT_NB_PATH.read_text(encoding="utf-8"))
            c3 = cell_src(built, 3)
            ok = (f"PRIVATE_EDGE1_CTX_RAISE = {e1!r}" in c3
                  and f"PRIVATE_EDGE2_VISIBLE_CONTRACT = {e2!r}" in c3)
            check(f"builder gates green + stamps correct (edge1={e1}, edge2={e2})", ok)
    finally:
        builder.OUT_DIR, builder.OUT_NB_PATH, builder.OUT_META_PATH = real_out_dir, real_nb, real_meta

# ---------------------------------------------------------------------------
print("S2: staged notebook structure + compile + base stamp")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
check("12 cells", len(nb["cells"]) == 12, f"got {len(nb['cells'])}")
compile_ok = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as exc:
        compile_ok = False
        check(f"cell {i} compiles", False, str(exc))
check("all code cells compile", compile_ok)
c3_staged = cell_src(nb, 3)
check("staged variant is pure base (both flags False)",
      "PRIVATE_EDGE1_CTX_RAISE = False" in c3_staged
      and "PRIVATE_EDGE2_VISIBLE_CONTRACT = False" in c3_staged)
meta = json.loads((NB_PATH.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
check("metadata: slug/bundle/model/docker as designed",
      meta["id"] == "canivel/arc3-q38-private-eval"
      and meta["dataset_sources"] == ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                       "jakobbrggen/taaf-kaggle-source"]
      and meta["model_sources"] == ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]
      and "57e612b" in meta["docker_image"]
      and meta["machine_shape"] == "NvidiaRtxPro6000"
      and meta["enable_internet"] is False)

# ---------------------------------------------------------------------------
print("S3: bundle sha pins")
lines = [f"{f.relative_to(BUNDLE).as_posix()}:{f.stat().st_size}:{hashlib.md5(f.read_bytes()).hexdigest()}"
         for f in sorted(BUNDLE.rglob("*"))
         if f.is_file() and "__pycache__" not in f.parts]  # bytecode from local imports is not bundle content
check("bundle manifest md5", hashlib.md5("\n".join(lines).encode()).hexdigest() == BUNDLE_MANIFEST_MD5)
setup_raw = (BUNDLE / "setup_commands.json").read_bytes()
check("setup_commands sha", hashlib.sha256(setup_raw).hexdigest()[:16] == SETUP_SHA)
ta_raw = (BUNDLE / "src/ARC3-Inference/inference/agent/tool_agent.py").read_bytes()
check("tool_agent sha", hashlib.sha256(ta_raw).hexdigest()[:16] == TOOL_AGENT_SHA)
check("bundle label", json.loads((BUNDLE / "taaf-kaggle-bundle.json").read_text())
      ["benchmark_label"] == "model-20260815-q38-p1")

# ---------------------------------------------------------------------------
print("S4: setup-command patching vs the REAL setup_commands.json, per arm")
c5 = cell_src(nb, 5)
patch_code = extract_defs(c5, {"_replace_python_assignment",
                               "_replace_python_int_assignment",
                               "_patch_qwen38_setup_commands"})
real_commands = json.loads(setup_raw.decode("utf-8"))


def run_patch(edge1: bool) -> list[str]:
    ns = {
        "re": re,
        "QWEN_MODEL_OWNER": "foysalemonshanto",
        "QWEN_MODEL_SLUG": "qwen3-8-27b-fp8-repacked-v1",
        "QWEN_SERVED_MODEL_NAME": "Qwen/Qwen3.8-27B-FP8",
        "PRIVATE_EDGE1_CTX_RAISE": edge1,
        "EDGE1_ANALYZER_CONTEXT_WINDOW": 65536,
        "EDGE1_VLLM_MAX_MODEL_LEN": 131072,
        "print": lambda *a, **k: None,
    }
    exec(compile(patch_code, "<cell5-patch>", "exec"), ns)
    return ns["_patch_qwen38_setup_commands"](list(real_commands))


patched_off = run_patch(edge1=False)
joined_off = "\n".join(patched_off)
check("OFF: model retarget applied",
      "MODEL_OWNER = 'foysalemonshanto'" in joined_off
      and "MODEL_SLUG = 'qwen3-8-27b-fp8-repacked-v1'" in joined_off)
check("OFF: ints untouched (base ctx preserved)",
      "VLLM_MAX_MODEL_LEN = 65536" in joined_off
      and "ANALYZER_CONTEXT_WINDOW = 32768" in joined_off)
check("OFF: offline env inserted once", joined_off.count("'HF_HUB_OFFLINE': '1'") == 1)

patched_on = run_patch(edge1=True)
joined_on = "\n".join(patched_on)
check("ON: VLLM_MAX_MODEL_LEN -> 131072",
      "VLLM_MAX_MODEL_LEN = 131072" in joined_on and "VLLM_MAX_MODEL_LEN = 65536" not in joined_on)
check("ON: ANALYZER_CONTEXT_WINDOW -> 65536",
      "ANALYZER_CONTEXT_WINDOW = 65536" in joined_on
      and "ANALYZER_CONTEXT_WINDOW = 32768" not in joined_on)
check("ON: model retarget still applied", "MODEL_OWNER = 'foysalemonshanto'" in joined_on)
check("ON: patched heredoc body still parses as Python",
      (lambda s: (ast.parse(s) or True))(joined_on.split("<<'PYSETUP'\n", 1)[1].rsplit("\nPYSETUP", 1)[0]))

# fail-loud proof: remove the int line -> patch must RAISE, not silently no-op
broken = [c.replace("VLLM_MAX_MODEL_LEN = 65536", "VLLM_MAX_MODEL_LEN='65536'") for c in real_commands]
ns2 = {"re": re, "QWEN_MODEL_OWNER": "foysalemonshanto", "QWEN_MODEL_SLUG": "x",
       "QWEN_SERVED_MODEL_NAME": "y", "PRIVATE_EDGE1_CTX_RAISE": True,
       "EDGE1_ANALYZER_CONTEXT_WINDOW": 65536, "EDGE1_VLLM_MAX_MODEL_LEN": 131072,
       "print": lambda *a, **k: None}
exec(compile(patch_code, "<cell5-patch>", "exec"), ns2)
try:
    ns2["_patch_qwen38_setup_commands"](broken)
    check("ON: missing int assignment REFUSES", False, "patch silently passed")
except RuntimeError as exc:
    check("ON: missing int assignment REFUSES", "VLLM_MAX_MODEL_LEN" in str(exc))

# ---------------------------------------------------------------------------
print("S5: bundle-label assert block (real notebook code, real bundle)")
c3 = cell_src(nb, 3)
label_block = extract_span(c3, "_bundle_meta = json.loads(",
                           'print(f"TAAF bundle generation: {_bundle_label}")')
ns = {"json": json, "BUNDLE_DIR": BUNDLE, "DATASET_BUNDLE_MARKER": "taaf-kaggle-bundle.json",
      "EXPECTED_BUNDLE_LABEL": "model-20260815-q38-p1", "print": lambda *a, **k: None}
try:
    exec(compile(label_block, "<cell3-label>", "exec"), ns)
    check("real bundle passes label assert", True)
except RuntimeError as exc:
    check("real bundle passes label assert", False, str(exc))
ns["EXPECTED_BUNDLE_LABEL"] = "model-20260807-anim"
try:
    exec(compile(label_block, "<cell3-label>", "exec"), ns)
    check("drifted label REFUSES", False, "assert did not fire")
except RuntimeError:
    check("drifted label REFUSES", True)

# ---------------------------------------------------------------------------
print("S6: banner block vs the scorer's regex (instrument alignment)")
banner_block = extract_span(c5, "# ---- PRIVATE-ARM BANNER",
                            "f\"edge2_contract={PRIVATE_EDGE2_VISIBLE_CONTRACT}\"\n)")
sys.path.insert(0, str(HERE))
import private_score  # noqa: E402

for e1 in (False, True):
    fake_env = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "65536" if e1 else "32768"}
    ns = {
        "os": types.SimpleNamespace(environ=fake_env),
        "PRIVATE_EDGE1_CTX_RAISE": e1, "PRIVATE_EDGE2_VISIBLE_CONTRACT": False,
        "BASE_ANALYZER_CONTEXT_WINDOW": 32768, "BASE_VLLM_MAX_MODEL_LEN": 65536,
        "EDGE1_ANALYZER_CONTEXT_WINDOW": 65536, "EDGE1_VLLM_MAX_MODEL_LEN": 131072,
        "_bundle_label": "model-20260815-q38-p1",
        "_actual_model_id": "Qwen/Qwen3.8-27B-FP8",
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(banner_block, "<cell5-banner>", "exec"), ns)
    m = private_score.BANNER_RE.search(buf.getvalue())
    arm = "edge1" if e1 else "base"
    spec_row = private_score.ARMS[arm]
    check(f"banner (edge1={e1}) matches scorer regex", m is not None)
    if m:
        check(f"banner (edge1={e1}) values match arm table",
              int(m.group("ctx")) == spec_row.analyzer_ctx
              and int(m.group("mml")) == spec_row.vllm_mml
              and int(m.group("budget")) == spec_row.effective_budget)
# wrong env must refuse
ns_bad = {
    "os": types.SimpleNamespace(environ={"LOCAL_ANALYZER_CONTEXT_WINDOW": "32768"}),
    "PRIVATE_EDGE1_CTX_RAISE": True, "PRIVATE_EDGE2_VISIBLE_CONTRACT": False,
    "BASE_ANALYZER_CONTEXT_WINDOW": 32768, "BASE_VLLM_MAX_MODEL_LEN": 65536,
    "EDGE1_ANALYZER_CONTEXT_WINDOW": 65536, "EDGE1_VLLM_MAX_MODEL_LEN": 131072,
    "_bundle_label": "model-20260815-q38-p1", "_actual_model_id": "Qwen/Qwen3.8-27B-FP8",
}
try:
    with redirect_stdout(io.StringIO()):
        exec(compile(banner_block, "<cell5-banner>", "exec"), ns_bad)
    check("banner REFUSES on ctx mismatch", False, "no raise")
except RuntimeError:
    check("banner REFUSES on ctx mismatch", True)

# ---------------------------------------------------------------------------
print("S7: edge-2 contract cell vs the REAL 08-15 tool_agent")
sys.path.insert(0, str(BUNDLE / "src" / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "src" / "tufa-arc-agi-framework" / "src"))
# Precondition parity with the kernel: the contract cell runs AFTER cell 5's setup,
# which exports the local-analyzer env (verified in bundle setup_commands.json).
import os as _os

_os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")
_os.environ.setdefault("INFERENCE_ANALYZER_MODEL", "Qwen/Qwen3.8-27B-FP8")
_os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
_os.environ.setdefault("LOCAL_ANALYZER_PROVIDER", "vllm")
edge2_src = cell_src(nb, 8)
with tempfile.TemporaryDirectory() as td:
    wd = Path(td)
    bm_stub = types.SimpleNamespace(solver=types.SimpleNamespace(model="local"))
    ns = {"PRIVATE_EDGE2_VISIBLE_CONTRACT": True, "bm": bm_stub,
          "WORKING_DIR": wd, "json": json}
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(edge2_src, "<cell8-edge2>", "exec"), ns)
        out = buf.getvalue()
        check("edge2 ON: preflight passes against real agent",
              "PRIVATE EDGE 2 (visible-updates contract): ACTIVE (Q38-strengthened)" in out)
        check("edge2 ON: audit json written",
              (wd / "private_visible_updates_contract.json").is_file())
        # a solver-style construction AFTER the patch must carry the contract
        from inference.agent.tool_agent import ToolAgent
        later = ToolAgent(model="local")
        check("edge2 ON: later solver-style ToolAgent carries the contract",
              "PRIVATE VISIBLE WORLD-MODEL UPDATE CONTRACT" in later._system_prompt)
        check("edge2 ON: base-prompt override target still present",
              "If you include assistant text before a tool call" in later._system_prompt)
        # module-level parser accepts the demanded shape
        import inference.agent.tool_agent as ta_mod
        parsed = ta_mod._extract_scientist_note("World model: x\nPlan: y")
        check("edge2 ON: parser accepts demanded field names",
              parsed.get("world_model") == "x" and parsed.get("current_plan") == "y")
    except Exception as exc:  # noqa: BLE001
        check("edge2 ON: preflight passes against real agent", False, repr(exc))

# flag OFF must be zero-touch — subprocess so module-import state is clean
off_probe = subprocess.run(
    [sys.executable, "-c", (
        "import sys, json, types\n"
        f"nb=json.load(open(r'{NB_PATH}', encoding='utf-8'))\n"
        "src=''.join(nb['cells'][8]['source'])\n"
        "ns={'PRIVATE_EDGE2_VISIBLE_CONTRACT': False}\n"
        "exec(compile(src,'<c8>','exec'), ns)\n"
        "assert 'inference.agent.tool_agent' not in sys.modules, 'imported despite OFF'\n"
        "print('ZERO-TOUCH-OK')\n"
    )],
    capture_output=True, text=True,
)
check("edge2 OFF: zero-touch (no import, skip line printed)",
      "ZERO-TOUCH-OK" in off_probe.stdout and "OFF - zero-touch" in off_probe.stdout,
      off_probe.stderr[-300:])

# ---------------------------------------------------------------------------
print("S8: unpickle real bundle pkls with real wheels")
import pathlib as _pl

_orig_posix = _pl.PosixPath
if sys.platform == "win32":
    _pl.PosixPath = _pl.WindowsPath  # local-only shim; Kaggle is Linux
try:
    import pickle

    bm_real = pickle.loads((BUNDLE / "benchmark_initial.pkl").read_bytes())
    check("benchmark_initial.pkl unpickles",
          bm_real.label == "model-20260815-q38-p1" and len(bm_real.games) == 25
          and bm_real.n_passes == 1 and bm_real.solver.concurrency == 28)
    tgt = pickle.loads((BUNDLE / "deploy_target.pkl").read_bytes())
    check("deploy_target.pkl unpickles", type(tgt).__name__ == "KaggleTarget")
    import taaf.game_api

    spec0 = getattr(bm_real.games[0], "arcade_spec", None) or getattr(bm_real.games[0], "_arcade_spec", None)
    g = taaf.game_api.GameAPI(env_name="sb26-7fbdac44", arcade_spec=spec0)
    check("cell-9 arcade_spec recovery path works", g is not None)
except Exception as exc:  # noqa: BLE001
    check("benchmark_initial.pkl unpickles", False, repr(exc))
finally:
    _pl.PosixPath = _orig_posix

# ---------------------------------------------------------------------------
print("S9: scorer selftest as subprocess")
st = subprocess.run([sys.executable, str(HERE / "private_score.py"), "--selftest"],
                    capture_output=True, text=True)
check("private_score --selftest 32/32", "SELFTEST: 32/32" in st.stdout, st.stdout[-200:])

# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"\nPRIVATE SMOKE: {n_pass}/{len(RESULTS)}")
if n_pass != len(RESULTS):
    for label, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {label} {detail}")
    raise SystemExit(1)
