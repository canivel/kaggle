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
_EFFORT_LITERAL = ('\'{"preserve_thinking": true, "reasoning_effort": "'
                   + B.REASONING_EFFORT + '"}\'')
check(f"reasoning_effort pinned to {B.REASONING_EFFORT} in --default-chat-template-kwargs",
      _EFFORT_LITERAL in rewritten)
check("the OTHER arm's effort value is absent (the two arms cannot be confused)",
      ('"reasoning_effort": "low"' if B.REASONING_EFFORT == "medium"
       else '"reasoning_effort": "medium"') not in rewritten)
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
    '        ' + _EFFORT_LITERAL + ',',
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
# 6b. PAYLOAD LINT — the gate that would have caught the v1 death, with no GPU
# ---------------------------------------------------------------------------
# v1 died because the MM probe sent max_tokens=32 with NO chat_template_kwargs, so
# `enable_thinking` defaulted ON, all 32 tokens were routed to `reasoning_content` by
# --reasoning-parser qwen3, and `content` came back empty — the MODAL behaviour of this stack
# (Jason Feng: 66.8% of tool-call responses have zero visible content). The probe then declared
# the vision path broken. The invariant that was missing: OUR PROBES MUST SEND WHAT THE HARNESS
# SENDS. The harness always sets chat_template_kwargs (openai_compat.py:78), and 32 tokens is
# not a budget any real turn runs under. Both properties are statically checkable.
section("6b. probe payload lint (harness-shaped payloads)")

_defs_tree = ast.parse(B.Q38_SERVE_DEFS)
_payloads: list[tuple[str, dict, int]] = []
for _node in ast.walk(_defs_tree):
    if not isinstance(_node, ast.Dict):
        continue
    keys = [k.value for k in _node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    if "model" in keys and "messages" in keys:
        _payloads.append(("dict", {k: v for k, v in zip(keys, _node.values)}, _node.lineno))
# payloads assembled by mutation (auto = dict(forced); auto['x'] = ...) inherit the base dict,
# so track assigned subscript keys too
_mutated: dict[str, set] = {}
for _node in ast.walk(_defs_tree):
    if (isinstance(_node, ast.Assign) and len(_node.targets) == 1
            and isinstance(_node.targets[0], ast.Subscript)
            and isinstance(_node.targets[0].value, ast.Name)
            and isinstance(_node.targets[0].slice, ast.Constant)):
        _mutated.setdefault(_node.targets[0].value.id, set()).add(_node.targets[0].slice.value)

check("found the chat/completions probe payloads to lint", len(_payloads) >= 2,
      f"{len(_payloads)} payload dict(s) at lines {[p[2] for p in _payloads]}")

_n_gen = 0
for _kind, _pl, _line in _payloads:
    # /tokenize payloads are identified by return_token_strs and legitimately carry no
    # max_tokens — they generate nothing. Only GENERATION payloads get the budget rule.
    is_generation = "return_token_strs" not in _pl
    check(f"payload@line{_line} sends chat_template_kwargs (the harness always does)",
          "chat_template_kwargs" in _pl)
    if not is_generation:
        print(f"    (line{_line} is a /tokenize payload — max_tokens rule N/A, not skipped "
              f"silently)")
        continue
    _n_gen += 1
    mt = _pl.get("max_tokens")
    val = mt.value if isinstance(mt, ast.Constant) else None
    check(f"generation payload@line{_line} has max_tokens >= 256 (v1 used 32 and truncated "
          "inside <think>)", isinstance(val, int) and val >= 256, f"max_tokens={val}")
check("at least 2 generation payloads were linted", _n_gen >= 2, str(_n_gen))

check("the mutated auto-probe payload re-sets chat_template_kwargs explicitly",
      "chat_template_kwargs" in _mutated.get("auto", set()),
      str(sorted(_mutated.get("auto", set()))))

# NEGATIVE CONTROL: reconstruct the v1 payload and prove the lint rejects it.
_v1_like = ast.parse(
    "payload = {'model': M, 'messages': [], 'temperature': 0.0, 'max_tokens': 32}"
).body[0].value
_v1_keys = [k.value for k in _v1_like.keys]
_v1_mt = dict(zip(_v1_keys, _v1_like.values)).get("max_tokens")
check("** NEGATIVE CONTROL: the lint rejects the exact v1 payload that killed the kernel",
      "chat_template_kwargs" not in _v1_keys and _v1_mt.value < 256,
      "no chat_template_kwargs, max_tokens=32")

# ---------------------------------------------------------------------------
# 6c. Gate classification — prereg section 10 must match the shipped code
# ---------------------------------------------------------------------------
section("6c. fatal/report-only classification (prereg section 10)")
_defs = B.Q38_SERVE_DEFS
check("gate K (served model id mismatch) is FATAL — poisoning",
      "raise RuntimeError('Q38-EVAL FATAL: served model ids" in _defs)
check("gate B (quant_method / weight_block_size) is FATAL — poisoning",
      "not Qwen3.8 blockwise fp8" in _defs and "raise RuntimeError" in _defs)
check("gate M (pin uncertified by BOTH instruments) is FATAL — poisoning",
      "UNCERTIFIED by BOTH" in _defs)
check("gate I1 (pinned render still injects) is FATAL — poisoning",
      "STILL '\n                               'injects an instruction" in _defs
      or "STILL " in _defs)
check("** gate N (tool-call) is RECLASSIFIED to REPORT-ONLY",
      "WARN tool-call-roundtrip=FAILED mode=forced" in _defs
      and "FATAL: tool-call round-trip" not in _defs)
check("** gate P (MM image) is RECLASSIFIED to REPORT-ONLY — the assert that killed v1",
      "WARN mm-image-roundtrip=EMPTY-CONTENT" in _defs
      and "FATAL: MM boot probe" not in _defs)
check("gate L (/tokenize) targets the ROOT url, not /v1 — v1's 404 was our URL",
      "root_url + '/tokenize'" in _defs and "VLLM_BASE_URL[:-3]" in _defs)
check("every probe emits OBSERVE (evidence) before any verdict",
      _defs.count("_q38_observe(") >= 4 and "Q38-EVAL OBSERVE" in _defs)
check("the OBSERVE line carries finish_reason, content_chars and reasoning_chars",
      all(t in _defs for t in ("finish_reason=", "content_chars=", "reasoning_chars=")))

# ---------------------------------------------------------------------------
# 6d. REGRESSION: replay v1's actual server behaviour through the NEW boot asserts
# ---------------------------------------------------------------------------
# Proving the reclassification changed BEHAVIOUR, not just strings. The stub reproduces the
# server exactly as v1 observed it: /v1/models correct, /tokenize 404 (v1's URL bug), tool
# calls fine, and the MM request answering 200 with EMPTY content and the output in
# reasoning_content. Under v1 this raised and killed the kernel. It must now complete.
section("6d. replay of the v1 failure through the reclassified boot asserts")


def _replay(mm_content: str, mm_reasoning: str, tokenize_404: bool,
            tool_calls_ok: bool = True) -> tuple[bool, str, str]:
    import contextlib
    import io
    import urllib.error

    calls_log: list[str] = []

    def fake_request_json(url, payload=None, timeout=30):
        calls_log.append(url)
        if url.endswith("/v1/models"):
            return {"data": [{"id": "Qwen/Qwen3.8-27B-FP8"}]}
        if url.endswith("/tokenize"):
            if tokenize_404:
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
            effort = (payload or {}).get("chat_template_kwargs", {}).get("reasoning_effort")
            text = "Q38_SYSTEM_SENTINEL <think>"
            if effort == "xhigh":
                text = "Reasoning effort is set to xhigh. " + text
            return {"token_strs": list(text)}
        # /chat/completions
        if any("image_url" in str(m.get("content")) for m in (payload or {}).get("messages", [])):
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": mm_content, "reasoning_content": mm_reasoning}}],
                "usage": {"completion_tokens": 32}}
        tc = ([{"function": {"name": "submit_action",
                             "arguments": '{"action": "ACTION6", "x": 3, "y": 7}'}}]
              if tool_calls_ok else [])
        return {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "reasoning_content": "thinking", "tool_calls": tc}}],
            "usage": {"completion_tokens": 40}}

    scope = {"json": json, "request_json": fake_request_json,
             "VLLM_BASE_URL": "http://127.0.0.1:1234/v1", "Path": Path,
             "shutil": __import__("shutil"), "subprocess": subprocess}
    exec(compile(B.Q38_SERVE_DEFS, "q38defs", "exec"), scope)
    scope["Q38_PIN_CERTIFIED"].append("local-render")  # as certified pre-serve, exactly as v1
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            scope["_q38_boot_asserts"]()
        return True, buf.getvalue(), ""
    except Exception as exc:  # noqa: BLE001
        return False, buf.getvalue(), f"{type(exc).__name__}: {exc}"


ok_v1, out_v1, err_v1 = _replay("", "the image appears to be a solid red square", True)
check("** the EXACT v1 scenario (MM empty content + /tokenize 404) now COMPLETES instead of "
      "killing the kernel", ok_v1, err_v1[:200])
check("and it reaches the hand-off to the bench",
      "BOOT-ASSERTS PASSED" in out_v1)
check("and it reports the MM result loudly rather than silently",
      "WARN mm-image-roundtrip=EMPTY-CONTENT" in out_v1)
check("and the OBSERVE line now carries the evidence v1 never printed "
      "(reasoning_chars > 0 is what would have closed the question)",
      "OBSERVE mm-image" in out_v1 and "reasoning_chars=42" in out_v1)
for _l in out_v1.strip().splitlines():
    print("      " + _l[:165])

ok_h, out_h, _ = _replay("red", "", False)
check("a healthy server still passes and now certifies the pin by BOTH instruments", ok_h)
check("the /tokenize probe works against the ROOT url once it is not 404",
      "server-probe reasoning_instruction=ABSENT" in out_h)
check("both instruments recorded", "effort-pin-certified-by=local-render,server-tokenize" in out_h)

ok_t, out_t, _ = _replay("red", "", False, tool_calls_ok=False)
check("** a broken tool-call path is now REPORTED, not fatal (it would BE the number)",
      ok_t and "WARN tool-call-roundtrip=FAILED" in out_t)

# The fatal gates must still be fatal.
def _replay_wrong_model() -> tuple[bool, str]:
    import contextlib, io
    def fake(url, payload=None, timeout=30):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "vrfai/Qwen3.6-27B-FP8"}]}
        return {"choices": [{"message": {"content": "x"}}]}
    scope = {"json": json, "request_json": fake, "VLLM_BASE_URL": "http://x/v1", "Path": Path,
             "shutil": __import__("shutil"), "subprocess": subprocess}
    exec(compile(B.Q38_SERVE_DEFS, "q38defs", "exec"), scope)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scope["_q38_boot_asserts"]()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


ok_w, err_w = _replay_wrong_model()
check("** NEGATIVE CONTROL: a silently-served Qwen3.6 is STILL FATAL (poisoning gate intact)",
      not ok_w and "served model ids" in err_w, err_w[:120])

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
check("the boot path can no longer kill the run for anything that would simply BE the number",
      "FATAL: MM boot probe" not in B.Q38_SERVE_DEFS
      and "FATAL: tool-call round-trip" not in B.Q38_SERVE_DEFS)
check("cell 2 carries a greppable Q38-EVAL banner", "Q38-EVAL seed=1" in "".join(nb["cells"][2]["source"]))

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"Q38 SMOKE: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAIL " + f)
    sys.exit(1)
print("ALL GATES GREEN — the artifact is runtime-tested; the slot decision is separate.")

# COVERAGE BOUNDARY. A pass count printed without the shape of what it excludes is a
# half-truth: v1's "81 passed, 0 failed" was TRUE of every check it ran, and every one of those
# checks also passed on the rail — the kernel still died, in the region the suite does not
# reach. State the boundary next to the number, every time.
print("""
COVERAGE BOUNDARY — what these checks do NOT validate:
  VALIDATED HERE   notebook structure and cell provenance; kernel metadata and env fields;
                   the setup-command rewrite EXECUTED against the real bundle (anchors, vetoes,
                   18 invariants); AST name resolution in the rewritten command; the chat
                   templates rendered for real; the pre-serve config asserts EXECUTED against a
                   staged real snapshot, with negative controls; probe payload SHAPE.
  NOT VALIDATED    anything requiring a SERVED MODEL. Specifically: whether an endpoint is
                   mounted at the path we call (v1 lost its second pin instrument to a /v1
                   prefix that does not exist for /tokenize); whether a response arrives in the
                   field we read (v1 died on `content` being empty while `reasoning_content`
                   held the output); token budgets vs thinking; kernel selection and the
                   blockwise-FP8 Triton fallback; throughput; and the 25-game number itself.
  IMPLICATION      a green run here means the artifact is well-formed and self-consistent. It
                   does NOT mean the kernel will survive the rail. The boot probes are now
                   report-only wherever a failure would simply BE the number, so the eval is
                   its own canary for its first ~7 minutes and continues if it survives them.""")
