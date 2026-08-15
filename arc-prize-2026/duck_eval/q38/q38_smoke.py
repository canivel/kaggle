"""Q38 engine-swap eval — BUILD GATE. Runtime-tests the artifact before a slot is spent.

`feedback_test_before_submit` (v38 scored 0.00 on a missing import) and the 2026-08-14 LoRA v1
death (a NameError inside the setup command, because `compile()` catches syntax and not scope)
are the two failures this file exists to prevent. It therefore does not merely inspect the
notebook — it EXECUTES the rewrite against the real bundled `setup_commands.json`, walks the
rewritten command's AST for unresolved names, and then actually RUNS the injected pre-serve
asserts against a staged copy of the real Qwen3.8 snapshot AND against a staged Qwen3.6
snapshot as a NEGATIVE CONTROL.

A gate that only ever passes is not a gate. Every assert here has a paired negative control.

    python duck_eval/q38/q38_smoke.py
"""
from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "duck_eval" / "q38"))

import build_q38_eval as B  # noqa: E402

FIX = REPO / "duck_eval" / "q38" / "_fixtures"
BUNDLE_SETUP = REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - older interpreters
    pass

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(f"{name}: {detail}" if detail else name)
    print(("  ok   " if ok else "  FAIL ") + name + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------
# 1. Build determinism and shape
# ---------------------------------------------------------------------------
section("1. build")
nb_path, meta_path = B.build()
before = nb_path.read_bytes()
B.build()
check("builder is idempotent (deterministic from the pristine fork)",
      nb_path.read_bytes() == before)

nb = json.loads(nb_path.read_text(encoding="utf-8"))
fork = json.loads(B.SRC_NB.read_text(encoding="utf-8"))
check("17 cells", len(nb["cells"]) == 17, str(len(nb["cells"])))
check("cell_type sequence matches the frozen fork",
      [c["cell_type"] for c in nb["cells"]] == [c["cell_type"] for c in fork["cells"]])
check("no metadata.kaggle block (preflight D2 compares 'both ABSENT')",
      "kaggle" not in (nb.get("metadata") or {}))
diff_cells = [i for i, (a, b) in enumerate(zip(fork["cells"], nb["cells"]))
              if "".join(a["source"]) != "".join(b["source"])]
check("exactly cells [2, 6, 8] differ from the frozen fork", diff_cells == [2, 6, 8],
      str(diff_cells))

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if i == 14:
        continue  # top-level await; not compilable standalone
    try:
        compile(src, f"cell{i}", "exec")
        ok, why = True, ""
    except SyntaxError as exc:
        ok, why = False, str(exc)
    check(f"cell {i} compiles", ok, why)

# MEASURED ON THIS KERNEL'S OWN v1 (2026-08-15): the Kaggle CLI push path re-encodes non-ASCII
# UTF-8 -> cp1252 mojibake. The frozen fork's own em-dashes (cells 7/11/13/16) arrive mangled
# and that is cosmetic, but ANY non-ASCII we inject into a CODE cell is a live hazard: v1's
# literal U+0120 in the /tokenize normaliser arrived corrupted and blinded the server-side
# effort probe. Everything WE author must be 7-bit; use \\uXXXX escapes instead.
for idx in (2, 6, 8):
    src = "".join(nb["cells"][idx]["source"])
    bad = sorted({c for c in src if ord(c) > 127})
    check(f"cell {idx} (authored by us) is pure ASCII — the Kaggle push path mangles non-ASCII",
          not bad, str([(c, hex(ord(c))) for c in bad]))

check("cell 14 (the run cell) is byte-identical to the frozen fork — the measurement surface "
      "is untouched",
      "".join(nb["cells"][14]["source"]) == "".join(fork["cells"][14]["source"]))
check("cell 12 (customization hook) is byte-identical — no graft rides along",
      "".join(nb["cells"][12]["source"]) == "".join(fork["cells"][12]["source"]))

# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------
section("2. kernel metadata")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
ref = json.loads(B.SRC_META.read_text(encoding="utf-8"))
check("fresh slug", meta["id"] == B.KERNEL_ID, meta["id"])
check("slug is NOT the scored/frozen-fork slug (feedback_fresh_kernel_slug)",
      meta["id"] != ref["id"])
check("dataset_sources = the frozen fork's list with ONLY the engine entry substituted, "
      "order preserved",
      meta["dataset_sources"] == [B.NEW_ENGINE_DS if s == B.OLD_ENGINE_DS else s
                                  for s in ref["dataset_sources"]],
      str(meta["dataset_sources"]))
check("exactly 3 dataset_sources (no extra attachment rides along)",
      len(meta["dataset_sources"]) == 3)
check("the incumbent engine is NOT attached", B.OLD_ENGINE_DS not in meta["dataset_sources"])
for key in ("enable_gpu", "enable_internet", "machine_shape", "docker_image",
            "competition_sources", "model_sources", "kernel_sources"):
    check(f"env field {key} byte-identical to the frozen fork (feedback_kaggle_env_match)",
          meta.get(key) == ref.get(key), repr(meta.get(key))[:80])
check("cell-6 DATASET_SOURCES matches kernel-metadata (they are two separate declarations)",
      json.dumps(meta["dataset_sources"]).replace(", ", ", ") in "".join(nb["cells"][6]["source"])
      or all(s in "".join(nb["cells"][6]["source"]) for s in meta["dataset_sources"]))

# ---------------------------------------------------------------------------
# 3. Execute the rewrite against the REAL bundled setup command
# ---------------------------------------------------------------------------
section("3. serve-config rewrite, executed against the real bundle")
cell8 = "".join(nb["cells"][8]["source"])
begin = cell8.index("# --- Q38-EVAL BEGIN")
end = cell8.index("# --- Q38-EVAL END")
q38_block = cell8[begin:end]
check("the Q38 block in cell 8 is delimited by BEGIN/END markers", bool(q38_block))
ns: dict = {}
exec(compile(q38_block, "cell8-q38-block", "exec"), ns)
patch = ns["_q38_patch_setup_commands"]

pristine = json.loads(BUNDLE_SETUP.read_text(encoding="utf-8"))
check("bundled setup_commands.json has exactly 1 command", len(pristine) == 1)
rewritten = patch(list(pristine))[0]
original = pristine[0]

check("MODEL_OWNER swapped", "MODEL_OWNER = 'saltb0x'" in rewritten)
check("MODEL_SLUG swapped", "MODEL_SLUG = 'qwen3-8-27b-fp8'" in rewritten)
check("SERVED_MODEL_NAME swapped",
      "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'" in rewritten)
check("reasoning_effort pinned to medium in --default-chat-template-kwargs",
      '\'{"preserve_thinking": true, "reasoning_effort": "medium"}\'' in rewritten)
check("WHEELHOUSE owner/slug untouched",
      "WHEELHOUSE_OWNER = 'driessmit1'" in rewritten
      and "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'" in rewritten)
for veto in B.Q38_VETO:
    check(f"veto {veto!r} absent after rewrite", veto not in rewritten)
for inv in B.Q38_INVARIANTS:
    check(f"invariant survives: {inv[:46]!r}", inv in rewritten)

# The one-variable proof, stated as a line diff of the two setup commands.
import difflib  # noqa: E402

removed = [l for l in difflib.unified_diff(original.splitlines(), rewritten.splitlines(), n=0)
           if l.startswith("-") and not l.startswith("---")]
added = [l for l in difflib.unified_diff(original.splitlines(), rewritten.splitlines(), n=0)
         if l.startswith("+") and not l.startswith("+++")]
print("    removed lines (the entire footprint of the change on the pristine command):")
for line in removed:
    print("      " + line[:150])
expect_removed = {
    "-MODEL_OWNER = 'driessmit1'",
    "-MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
    "-SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
    "-        '{\"preserve_thinking\": true}',",
}
check("** the ONLY lines removed are the 3 engine constants and the chat-template-kwargs "
      "line - nothing else in the serve config is touched",
      set(removed) == expect_removed,
      str(sorted(set(removed) ^ expect_removed))[:220])

defs_lines = set(B.Q38_SERVE_DEFS.splitlines())
non_instrumentation = [l[1:] for l in added if l[1:] not in defs_lines]
expect_added = {
    "MODEL_OWNER = 'saltb0x'",
    "MODEL_SLUG = 'qwen3-8-27b-fp8'",
    "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'",
    '        \'{"preserve_thinking": true, "reasoning_effort": "medium"}\',',
    "_q38_pre_serve_asserts()",
    "_q38_boot_asserts()",
    "",
}
check("** every ADDED line is either the Q38 instrumentation block verbatim or one of the "
      "4 swapped constants / 2 assert calls",
      set(non_instrumentation) <= expect_added,
      str(sorted(set(non_instrumentation) - expect_added))[:220])

# ---------------------------------------------------------------------------
# 4. AST name resolution of the rewritten command (the LoRA v1 gate)
# ---------------------------------------------------------------------------
section("4. name resolution of the rewritten setup command (LoRA v1 NameError class)")


def _payload(command: str) -> str:
    start = command.index("<<'PYSETUP'\n") + len("<<'PYSETUP'\n")
    return command[start:command.rindex("\nPYSETUP")]


def unresolved_names(source: str) -> list[str]:
    tree = ast.parse(source)
    bound: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                    bound.add(arg.arg)
                if a.vararg:
                    bound.add(a.vararg.arg)
                if a.kwarg:
                    bound.add(a.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    bound.add(sub.id)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                bound.add(arg.arg)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for sub in ast.walk(item.optional_vars):
                        if isinstance(sub, ast.Name):
                            bound.add(sub.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - bound)


base_payload = _payload(original)
new_payload = _payload(rewritten)
check("rewritten setup payload compiles", bool(compile(new_payload, "PYSETUP", "exec")) or True)
base_unresolved = set(unresolved_names(base_payload))
new_unresolved = set(unresolved_names(new_payload))
check("the rewrite introduces NO unresolved name (LoRA v1 died exactly here)",
      new_unresolved <= base_unresolved, str(sorted(new_unresolved - base_unresolved)))
print(f"    loaded names checked; pristine-unresolved={sorted(base_unresolved)} "
      f"rewritten-unresolved={sorted(new_unresolved)}")

# Negative control: an out-of-scope name MUST be caught.
poisoned = new_payload + "\n_a_name_that_does_not_exist()\n"
check("NEGATIVE CONTROL: the name gate catches an injected out-of-scope call",
      "_a_name_that_does_not_exist" in unresolved_names(poisoned))

# ---------------------------------------------------------------------------
# 5. The reasoning_effort pin — measured on the real templates
# ---------------------------------------------------------------------------
section("5. reasoning_effort pin (measured, both templates)")
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402


def render(template_path: Path, **extra) -> str:
    def _raise(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        keep_trailing_newline=True)
    env.globals["raise_exception"] = _raise
    env.filters["tojson"] = lambda o, **kw: json.dumps(o, ensure_ascii=False)
    tpl = env.from_string(template_path.read_text(encoding="utf-8"))
    tools = [{"type": "function", "function": {
        "name": "submit_action", "description": "act",
        "parameters": {"type": "object", "properties": {"action": {"type": "string"}},
                       "required": ["action"]}}}]
    messages = [
        {"role": "system", "content": "You are the ARC3 agent."},
        {"role": "user", "content": [{"type": "text", "text": "grid"},
                                     {"type": "image_url",
                                      "image_url": {"url": "data:image/png;base64,AAA"}}]},
        {"role": "assistant", "content": "moving right",
         "reasoning": "harness sets 'reasoning', never 'reasoning_content'",
         "tool_calls": [{"type": "function",
                         "function": {"name": "submit_action",
                                      "arguments": {"action": "ACTION3"}}}]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    kwargs = dict(messages=messages, tools=tools, add_generation_prompt=True,
                  preserve_thinking=True, enable_thinking=True, add_vision_id=False)
    kwargs.update(extra)
    return tpl.render(**kwargs)


q36 = render(FIX / "q36_chat_template.jinja")
q38_medium = render(FIX / "q38_chat_template.jinja", reasoning_effort="medium")
q38_default = render(FIX / "q38_chat_template.jinja")
q38_low = render(FIX / "q38_chat_template.jinja", reasoning_effort="low")
q38_xhigh = render(FIX / "q38_chat_template.jinja", reasoning_effort="xhigh")

check("** reasoning_effort='medium' renders BYTE-IDENTICALLY to the Qwen3.6 template — the pin "
      "isolates the weights", q38_medium == q36,
      f"{len(q38_medium)} vs {len(q36)} chars")
check("the DEFAULT (xhigh) does NOT — this is the live regression risk the pin removes",
      q38_default != q36, f"{len(q38_default)} chars, +{len(q38_default) - len(q36)}")
check("'low' also differs — it is the SEPARATE later arm, not a neutral value",
      q38_low != q36, f"{len(q38_low)} chars")
check("default == xhigh (the template's documented default)", q38_default == q38_xhigh)
check("the pinned render contains NO reasoning-effort instruction",
      "Reasoning effort" not in q38_medium)
check("NEGATIVE CONTROL: the xhigh render DOES contain one (the probe is not blind)",
      "Reasoning effort is set to xhigh" in q38_default)
try:
    render(FIX / "q38_chat_template.jinja", reasoning_effort="ultra")
    raised = False
except Exception:
    raised = True
check("unrecognised reasoning_effort raises (fails loud, not silent)", raised)

check("tool-call block syntax is byte-identical across the two templates (so "
      "--tool-call-parser qwen3_coder carries over)",
      "<tool_call>\n<function=" in q36 and "<tool_call>\n<function=" in q38_medium)

# ---------------------------------------------------------------------------
# 6. Run the injected pre-serve asserts against a staged real snapshot
# ---------------------------------------------------------------------------
section("6. pre-serve asserts, executed (real Qwen3.8 config + Qwen3.6 negative control)")


def stage(config_fixture: str, template_fixture: str, *, shards: int = 64,
          drop: tuple[str, ...] = ()) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "config.json").write_bytes((FIX / config_fixture).read_bytes())
    (root / "chat_template.jinja").write_bytes((FIX / template_fixture).read_bytes())
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "preprocessor_config.json", "model.safetensors.index.json"):
        if name not in drop:
            (root / name).write_text("{}", encoding="utf-8")
    for i in range(shards):
        (root / f"layers-{i}.safetensors").write_bytes(b"")
    return root


def run_pre_serve(model_dir: Path) -> tuple[bool, str, str]:
    """Exec the injected defs in an isolated namespace mimicking the setup process."""
    import io
    import contextlib
    import shutil as _shutil

    scope = {
        "json": json, "MODEL_PATH": model_dir, "shutil": _shutil,
        "subprocess": subprocess, "Path": Path,
    }
    exec(compile(B.Q38_SERVE_DEFS, "q38defs", "exec"), scope)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            scope["_q38_pre_serve_asserts"]()
        return True, buf.getvalue(), ""
    except Exception as exc:  # noqa: BLE001
        return False, buf.getvalue(), f"{type(exc).__name__}: {exc}"


ok38, out38, err38 = run_pre_serve(stage("q38_config.json", "q38_chat_template.jinja"))
check("** pre-serve asserts PASS on the real saltb0x/qwen3-8-27b-fp8 config", ok38, err38[:220])
check("the pin is certified by the local render inside the asserts",
      "effort-pin=medium local-render reasoning_instruction=ABSENT" in out38)
for line in out38.strip().splitlines():
    print("      " + line[:170])

ok36, _, err36 = run_pre_serve(stage("q36_config.json", "q36_chat_template.jinja", shards=0))
check("** NEGATIVE CONTROL: the same asserts REJECT the incumbent Qwen3.6 snapshot "
      "(a silent-3.6 run cannot pass)", not ok36, err36[:200])

okmiss, _, errmiss = run_pre_serve(
    stage("q38_config.json", "q38_chat_template.jinja", drop=("tokenizer_config.json",)))
check("** NEGATIVE CONTROL: a snapshot missing tokenizer_config.json is rejected BEFORE the "
      "load (the 122B lesson)", not okmiss and "INCOMPLETE" in errmiss, errmiss[:160])

okshard, _, errshard = run_pre_serve(
    stage("q38_config.json", "q38_chat_template.jinja", shards=63))
check("NEGATIVE CONTROL: a short shard set is rejected", not okshard, errshard[:120])

# ---------------------------------------------------------------------------
# 7. Boot-assert wiring (static; the server side runs on Kaggle)
# ---------------------------------------------------------------------------
section("7. boot-assert wiring")
scope: dict = {}
exec(compile(B.Q38_SERVE_DEFS, "q38defs", "exec"), scope)
check("_q38_boot_asserts defined", callable(scope.get("_q38_boot_asserts")))
check("_q38_png_b64 produces a valid PNG header",
      scope["_q38_png_b64"]().startswith("iVBORw0KGgo"))
import base64 as _b64  # noqa: E402
png = _b64.b64decode(scope["_q38_png_b64"]())
import struct as _struct  # noqa: E402

_w, _h = _struct.unpack(">II", png[16:24])
check("PNG decodes to a valid 64x64 RGB image",
      png[:8] == b"\x89PNG\r\n\x1a\n" and (_w, _h) == (64, 64),
      f"{_w}x{_h}, {len(png)} bytes (a solid colour compresses small — this is correct)")
check("the pin is FATAL if BOTH instruments fail (no silent unknown-effort run)",
      "UNCERTIFIED by BOTH" in B.Q38_SERVE_DEFS)
check("a served model-id mismatch is FATAL (no silent incumbent)",
      "refusing to continue" in B.Q38_SERVE_DEFS)
check("cell 2 carries a greppable Q38-EVAL banner", "Q38-EVAL seed=1" in "".join(nb["cells"][2]["source"]))

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"Q38 SMOKE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAIL " + f)
    sys.exit(1)
print("ALL GATES GREEN — the artifact is runtime-tested; the slot decision is separate.")
