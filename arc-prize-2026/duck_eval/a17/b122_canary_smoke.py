"""Runtime smoke for the B122 boot canary (2026-08-13).

`feedback_test_before_submit`: v38 scored 0.00 from a missing import. Nothing goes to
Kaggle until the injected code has actually RUN here.

This does NOT need a GPU. It:
  * rebuilds the notebook and checks determinism,
  * runs the structural-diff gate against the frozen fork (preflight has no
    duck-family profile — documented debt, see the war-room note),
  * extracts the REWRITTEN setup command out of the built notebook and
    EXECUTES its definition section against a fake /kaggle/input mount, so
    `_b122_stage_model()` really runs: linking, the three shim files, and every
    FAIL-LOUD post-condition, positive and negative,
  * exercises `_b122_png_b64` (decodes as a real PNG) and the projection arithmetic,
  * asserts the serve args and the veto/require sets.

Usage:  python duck_eval/a17/b122_canary_smoke.py
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILDER = HERE / "build_b122_boot_canary.py"
OUT_NB = REPO / "notebooks" / "b122-canary" / "arc3-b122-boot-canary.ipynb"
OUT_META = REPO / "notebooks" / "b122-canary" / "kernel-metadata.json"
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' — ' + detail) if detail else ''}")
    print(("  PASS " if cond else "  FAIL ") + name + ((" — " + detail) if detail else ""))


def expect_raises(name: str, fn, needle: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        check(name, needle in str(exc), f"raised {type(exc).__name__}: {str(exc)[:120]}")
        return
    check(name, False, "did NOT raise")


# ---------------------------------------------------------------------------
# B — build determinism
# ---------------------------------------------------------------------------
def section_build() -> None:
    print("\n[B] builder")
    r1 = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True)
    check("B1 builder exits 0", r1.returncode == 0, r1.stderr.strip()[-300:])
    first = OUT_NB.read_bytes()
    r2 = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True)
    check("B2 builder is idempotent (rebuild byte-identical)",
          r2.returncode == 0 and OUT_NB.read_bytes() == first)


# ---------------------------------------------------------------------------
# S — structural diff vs the frozen fork (the gate preflight cannot be)
# ---------------------------------------------------------------------------
def section_structure() -> None:
    print("\n[S] structural diff vs the frozen duckfork")
    base = json.loads(SRC_NB.read_text(encoding="utf-8"))
    new = json.loads(OUT_NB.read_text(encoding="utf-8"))
    check("S1 cell count 17 vs 17", len(base["cells"]) == 17 == len(new["cells"]),
          f"{len(base['cells'])} vs {len(new['cells'])}")
    diff = [i for i in range(len(base["cells"]))
            if "".join(base["cells"][i]["source"]) != "".join(new["cells"][i]["source"])]
    check("S2 exactly cells 2, 6, 8, 14 differ", diff == [2, 6, 8, 14], f"differ={diff}")
    types_ok = all(base["cells"][i]["cell_type"] == new["cells"][i]["cell_type"]
                   for i in range(len(base["cells"])))
    check("S3 cell types unchanged", types_ok)
    for i, cell in enumerate(new["cells"]):
        if cell["cell_type"] == "code":
            try:
                compile("".join(cell["source"]), f"cell{i}", "exec",
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as exc:
                check(f"S4.{i} cell {i} compiles", False, str(exc))
                break
    else:
        check("S4 every code cell compiles", True)

    bmeta = json.loads((SRC_NB.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    nmeta = json.loads(OUT_META.read_text(encoding="utf-8"))
    changed = {k for k in set(bmeta) | set(nmeta) if bmeta.get(k) != nmeta.get(k)}
    check("S5 metadata delta is only {id,title,code_file,dataset_sources,model_sources}",
          changed == {"id", "title", "code_file", "dataset_sources", "model_sources"},
          f"changed={sorted(changed)}")
    check("S6 env matched to the family (feedback_kaggle_env_match)",
          nmeta["docker_image"] == bmeta["docker_image"]
          and nmeta["machine_shape"] == bmeta["machine_shape"]
          and nmeta["enable_gpu"] is True and nmeta["enable_internet"] is False)
    check("S7 fresh kernel slug", nmeta["id"] == "canivel/arc3-b122-boot-canary")
    check("S8 datasets = source + jcole75 wheels + ippeiogawa 122B; h100 wheelhouse and 27B GONE",
          nmeta["dataset_sources"] == ["jeroencottaar/taaf-kaggle-source-share",
                                       "jcole75/arc3-qwen36-runtime-wheels",
                                       "ippeiogawa/qwen35-122b-a10b-nvfp4"])
    cell6 = "".join(new["cells"][6]["source"])
    check("S9 cell-6 DATASET_SOURCES matches the metadata",
          all(d in cell6 for d in nmeta["dataset_sources"])
          and "arc3-vllm-h100-wheelhouse-v3" not in cell6
          and "vrfai-qwen3-6-27b-fp8-hf-snapshot" not in cell6)
    cell14 = "".join(new["cells"][14]["source"])
    check("S10 boot-only short-circuit present and rerun-safe",
          "B122_BOOT_ONLY" in cell14 and "not TRUE_SUBMISSION" in cell14
          and "await bm.run(" in cell14)
    check("S11 no graft in cell 12 (this A/Bs the BRAIN, one variable)",
          "".join(new["cells"][12]["source"]) == "".join(base["cells"][12]["source"]))


# ---------------------------------------------------------------------------
# R — the rewritten setup command
# ---------------------------------------------------------------------------
def _rewritten_setup_text() -> str:
    """Reproduce, from the BUILT notebook, exactly what will run on Kaggle."""
    nb = json.loads(OUT_NB.read_text(encoding="utf-8"))
    cell8 = "".join(nb["cells"][8]["source"])
    ns: dict = {}
    block = cell8.split("# --- B122-CANARY BEGIN serve-config rewrite", 1)[1]
    block = block.split("# --- B122-CANARY END serve-config rewrite", 1)[0]
    exec("B122_SETUP_REWRITES" + block.split("B122_SETUP_REWRITES", 1)[1], ns)  # noqa: S102
    original = json.loads((REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json")
                          .read_text(encoding="utf-8"))
    return ns["_b122_patch_setup_commands"](original)[0]


def _python_body(text: str) -> str:
    """The setup command is `"$PYTHON" - <<'PYSETUP' ... PYSETUP` — a QUOTED heredoc, so the
    body is passed to python verbatim with no shell expansion. Return just that body."""
    lines = text.splitlines()
    if lines[0] != '"$PYTHON" - <<\'PYSETUP\'' or lines[-1] != "PYSETUP":
        raise AssertionError(f"unexpected heredoc wrapper: {lines[0]!r} ... {lines[-1]!r}")
    return "\n".join(lines[1:-1])


def section_rewrite(text: str) -> None:
    print("\n[R] rewritten setup command")
    body = _python_body(text)
    check("R1 the heredoc body compiles as Python", _compiles(body))
    check("R1b nothing we inject breaks the heredoc "
          "(no bare PYSETUP sentinel; quoted heredoc so $ is inert)",
          "PYSETUP" not in body and text.count("PYSETUP") == 2)
    for token, want in [
        ("--quantization", True), ("modelopt_fp4", True),
        ("--gpu-memory-utilization", True), ("'0.93'", True),
        ("qwen3_coder", True), ("--reasoning-parser", True), ("preserve_thinking", True),
        ("--enable-prefix-caching", True), ("--enable-auto-tool-choice", True),
        ("--chat-template", True), ("str(B122_CHAT_TEMPLATE)", True),
        ("awq_marlin", False), ("hermes", False),
        ("--kv-cache-dtype", False),        # deliberate omission, documented
        ("--limit-mm-per-prompt", False),   # deliberate omission, documented
    ]:
        check(f"R2 {'has' if want else 'lacks'} {token}", (token in text) is want)
    check("R3 served name is the 122B", "SERVED_MODEL_NAME = 'Qwen/Qwen3.5-122B-A10B'" in text)
    check("R4 max-model-len 32768 (= ANALYZER_CONTEXT_WINDOW, so behaviourally free)",
          "VLLM_MAX_MODEL_LEN = 32768" in text and "ANALYZER_CONTEXT_WINDOW = 32768" in text)
    check("R5 wheelhouse layout patched (wheels/ + runtime requirements)",
          "requirements-runtime.txt" in text and "WHEELHOUSE / 'wheels'" in text
          and "requirements.lock" not in text)
    check("R6 45-min load KILL rule encoded in the boot wait",
          "def wait_for_vllm_server(timeout_seconds: int = 2700)" in text)
    check("R7 no 27B / duck-wheelhouse artifact survives",
          not any(t in text for t in ("driessmit1", "vrfai/Qwen3.6-27B-FP8",
                                      "arc3-vllm-h100-wheelhouse-v3")))
    check("R8 canary entrypoints wired after boot",
          "_b122_serve_asserts()" in text and "_b122_throughput_probe(_b122_boot_s)" in text
          and text.index("start_vllm_server()") < text.index("_b122_serve_asserts()"))
    check("R9 thinking left ON (unlike the a17 72B canary, which disabled it)",
          "'LOCAL_ANALYZER_ENABLE_THINKING': 'true'" in text)
    # --- the v1 INFRA DEATH: ImportError libcudart.so.13 at `import vllm` ---
    check("R10 LD_LIBRARY_PATH is built for the child vLLM process",
          "env['LD_LIBRARY_PATH'] = os.pathsep.join(" in body
          and "_b122_cuda_lib_dirs()" in body)
    check("R11 the CUDA-13 wheel lib dirs are globbed (nvidia/*/lib, torch/lib)",
          "'nvidia/*/lib'" in body and "'torch/lib'" in body)
    check("R12 fail-fast guard runs AFTER the wheelhouse install and BEFORE the weight load",
          body.index("install_vllm_wheelhouse()\n    _b122_assert_cuda_runtime()")
          < body.index("VLLM_SERVER_LOG.parent.mkdir"))
    check("R13 the guard names the v1 death so a rerun is self-explaining",
          "libcudart.so.13" in body and "ImportError" in body)


def _compiles(text: str) -> bool:
    try:
        compile(text, "setup", "exec")
        return True
    except SyntaxError as exc:
        print("   syntax error:", exc)
        return False


# ---------------------------------------------------------------------------
# I — INTEGRATION: really run the injected definitions against a fake mount
# ---------------------------------------------------------------------------
_REAL_122B_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "image_token_id": 248056,
    "quantization_config": {"quant_algo": "NVFP4", "quant_method": "modelopt",
                            "kv_cache_scheme": {"num_bits": 8, "type": "float"}},
    "text_config": {"num_hidden_layers": 48, "num_key_value_heads": 2, "head_dim": 256,
                    "hidden_size": 3072, "num_experts": 256, "num_experts_per_tok": 8,
                    "full_attention_interval": 4, "vocab_size": 248320,
                    "max_position_embeddings": 262144},
    "vision_config": {"depth": 27, "hidden_size": 1152, "patch_size": 16,
                      "spatial_merge_size": 2},
}
# The upstream 122B template: the preserve_thinking branch is MISSING (that is the trap).
_UPSTREAM_TEMPLATE = (
    "{%- for message in messages %}\n"
    "  {%- if loop.index0 > ns.last_query_index %}\n"
    "    {{- '<|im_start|>' }}\n"
    "  {%- endif %}\n"
    "{%- endfor %}\n"
)


def _fake_mount(root: Path, *, shards: int = 9, decoy: bool = False) -> Path:
    model = root / "input" / "qwen35-122b-a10b-nvfp4"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps(_REAL_122B_CONFIG), encoding="utf-8")
    (model / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8",
                                     "group_size": 16}}), encoding="utf-8")
    (model / "chat_template.jinja").write_text(_UPSTREAM_TEMPLATE, encoding="utf-8")
    (model / "preprocessor_config.json").write_text(
        json.dumps({"processor_class": "Qwen3VLProcessor", "patch_size": 16}), encoding="utf-8")
    (model / "tokenizer.json").write_text('{"model": {"type": "BPE"}}', encoding="utf-8")
    (model / "vocab.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    for i in range(1, shards + 1):
        (model / f"model-{i:05d}-of-{shards:05d}.safetensors").write_bytes(b"\0" * 16)
    # Decoys the finder must ignore: the source bundle and the wheelhouse both hold config.json.
    other = root / "input" / "taaf-kaggle-source-share"
    other.mkdir(parents=True, exist_ok=True)
    (other / "config.json").write_text('{"architectures": ["SomethingElse"]}', encoding="utf-8")
    if decoy:
        dup = root / "input" / "qwen35-dup"
        dup.mkdir(parents=True, exist_ok=True)
        (dup / "config.json").write_text(json.dumps(_REAL_122B_CONFIG), encoding="utf-8")
        (dup / "a.safetensors").write_bytes(b"\0")
    return model


def _defs_namespace(text: str, root: Path):
    """Exec the setup command's DEFINITION section (everything before the imperative tail)."""
    head = _python_body(text).split("print(f'vLLM wheelhouse path:", 1)[0]
    working = root / "working"
    working.mkdir(parents=True, exist_ok=True)
    os.environ["B122_INPUT_ROOT"] = str(root / "input")
    os.environ["TAAF_KAGGLE_WORKING_DIR"] = str(working)
    ns: dict = {"__name__": "b122_setup"}
    exec(compile(head, "setup_defs", "exec"), ns)  # noqa: S102
    return ns


def section_cuda(text: str) -> None:
    """The v1 death, reproduced and fixed under test."""
    print("\n[C] CUDA-13 loader path (the v1 INFRA DEATH)")
    root = Path(tempfile.mkdtemp(prefix="b122smoke_cuda_"))
    try:
        _fake_mount(root)
        ns = _defs_namespace(text, root)
        site = Path(ns["SITE_PACKAGES"])
        # No CUDA wheels installed yet -> the guard must refuse, loudly.
        expect_raises("C1 no CUDA runtime present => FATAL before any weight load "
                      "(v1 spent a GPU-hour learning this)",
                      ns["_b122_assert_cuda_runtime"], "libcudart.so* not found")
        # Now lay out the wheelhouse exactly as pip --target does.
        for sub in ("nvidia/cuda_runtime/lib", "nvidia/cublas/lib", "nvidia/curand/lib",
                    "torch/lib"):
            (site / sub).mkdir(parents=True, exist_ok=True)
        (site / "nvidia/cuda_runtime/lib/libcudart.so.13").write_bytes(b"\0")
        dirs = ns["_b122_cuda_lib_dirs"]()
        check("C2 every wheel lib dir is discovered", len(dirs) == 4, f"{len(dirs)} dirs")
        ns["_b122_assert_cuda_runtime"]()
        check("C3 guard passes once libcudart.so.13 is present", True)
        env = ns["vllm_env"]()
        ld = env["LD_LIBRARY_PATH"].split(os.pathsep)
        check("C4 LD_LIBRARY_PATH carries the CUDA-13 runtime dir to the child process",
              str(site / "nvidia" / "cuda_runtime" / "lib") in ld)
        check("C5 torch/lib is on it too", str(site / "torch" / "lib") in ld)
        check("C6 PYTHONPATH still points at the wheelhouse target (unbroken)",
              str(site) in env["PYTHONPATH"].split(os.pathsep))
        os.environ["LD_LIBRARY_PATH"] = "/pre/existing"
        try:
            check("C7 a pre-existing LD_LIBRARY_PATH is preserved, appended not clobbered",
                  ns["vllm_env"]()["LD_LIBRARY_PATH"].split(os.pathsep)[-1] == "/pre/existing")
        finally:
            os.environ.pop("LD_LIBRARY_PATH", None)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop("B122_INPUT_ROOT", None)
        os.environ.pop("TAAF_KAGGLE_WORKING_DIR", None)


def section_integration(text: str) -> None:
    print("\n[I] integration — the injected code actually runs")
    root = Path(tempfile.mkdtemp(prefix="b122smoke_"))
    try:
        _fake_mount(root)
        ns = _defs_namespace(text, root)
        staged = ns["MODEL_PATH"]
        check("I1 _b122_stage_model ran and returned the staging dir",
              staged.name == "b122-model" and staged.is_dir())
        check("I2 all 9 shards linked", len(list(staged.glob("*.safetensors"))) == 9)
        check("I3 shim shipped: tokenizer_config.json (ABSENT upstream)",
              (staged / "tokenizer_config.json").is_file())
        check("I4 shim shipped: processor_config.json (ABSENT upstream)",
              (staged / "processor_config.json").is_file())
        tmpl = (staged / "chat_template.jinja").read_text(encoding="utf-8")
        check("I5 preserve_thinking branch RESTORED over the upstream template",
              "preserve_thinking is defined and preserve_thinking is true" in tmpl)
        check("I6 the upstream (branchless) template did NOT survive",
              tmpl != _UPSTREAM_TEMPLATE and len(tmpl) == 7764)
        check("I7 the 27B tokenizer_config declares a tokenizer class",
              json.loads((staged / "tokenizer_config.json").read_text())["tokenizer_class"]
              == "Qwen2Tokenizer")
        check("I8 processor_class is Qwen3VLProcessor",
              json.loads((staged / "processor_config.json").read_text())["processor_class"]
              == "Qwen3VLProcessor")
        check("I9 mount files are LINKED not copied (no 83.5 GB duplication)",
              (staged / "config.json").stat().st_size
              == (root / "input" / "qwen35-122b-a10b-nvfp4" / "config.json").stat().st_size)
        check("I10 B122_CHAT_TEMPLATE points at the staged (fixed) template",
              ns["B122_CHAT_TEMPLATE"] == staged / "chat_template.jinja")
        check("I11 the decoy config.json in the source bundle was ignored", True)

        # PNG probe really decodes.
        png = base64.b64decode(ns["_b122_png_b64"]())
        check("I12 MM probe emits a real PNG (magic + IHDR 64x64)",
              png[:8] == b"\x89PNG\r\n\x1a\n"
              and struct.unpack(">II", png[16:24]) == (64, 64)
              and zlib.crc32(png[12:16] + png[16:16 + 13]) & 0xFFFFFFFF
              == struct.unpack(">I", png[29:33])[0])

        # --- negative paths: every FAIL-LOUD assert fires ---
        print("  -- negative paths --")
        root2 = Path(tempfile.mkdtemp(prefix="b122smoke_neg_"))
        try:
            (root2 / "input").mkdir(parents=True)
            expect_raises("I13 no weights attached => FATAL, no fallback brain",
                          lambda: _defs_namespace(text, root2), "not found under")
        finally:
            shutil.rmtree(root2, ignore_errors=True)

        root3 = Path(tempfile.mkdtemp(prefix="b122smoke_dup_"))
        try:
            _fake_mount(root3, decoy=True)
            expect_raises("I14 two candidate checkpoints => FATAL",
                          lambda: _defs_namespace(text, root3), "multiple NVFP4 candidate")
        finally:
            shutil.rmtree(root3, ignore_errors=True)

        root4 = Path(tempfile.mkdtemp(prefix="b122smoke_shards_"))
        try:
            _fake_mount(root4, shards=8)
            expect_raises("I15 wrong shard count => FATAL (partial mount caught)",
                          lambda: _defs_namespace(text, root4), "safetensors shards, want 9")
        finally:
            shutil.rmtree(root4, ignore_errors=True)

        root5 = Path(tempfile.mkdtemp(prefix="b122smoke_tmpl_"))
        try:
            _fake_mount(root5)
            broken = text.replace(json.dumps(_shim()["chat_template.jinja"]),
                                  json.dumps(base64.b64encode(b"no branch here").decode()))
            check("I16 tamper harness rewired the shim payload", broken != text)
            expect_raises("I17 a shim template without the branch => FATAL "
                          "(the silent-no-op trap cannot ship)",
                          lambda: _defs_namespace(broken, root5),
                          "no preserve_thinking branch")
        finally:
            shutil.rmtree(root5, ignore_errors=True)

        # Two independent layers reject a wrong-quant checkpoint. Layer 1: the marker-based
        # finder never selects it. Layer 2 (defence in depth): even if a config mentions NVFP4
        # elsewhere — e.g. in the 148-entry `ignore` list — the staged quant is re-asserted.
        root6 = Path(tempfile.mkdtemp(prefix="b122smoke_quant_"))
        try:
            model = _fake_mount(root6)
            cfg = dict(_REAL_122B_CONFIG)
            cfg["quantization_config"] = {"quant_algo": "FP8", "quant_method": "modelopt"}
            (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            expect_raises("I18 layer 1: an FP8 checkpoint is never selected by the finder",
                          lambda: _defs_namespace(text, root6), "not found under")
        finally:
            shutil.rmtree(root6, ignore_errors=True)

        root7 = Path(tempfile.mkdtemp(prefix="b122smoke_quant2_"))
        try:
            model = _fake_mount(root7)
            cfg = dict(_REAL_122B_CONFIG)
            cfg["quantization_config"] = {"quant_algo": "FP8", "quant_method": "modelopt",
                                          "ignore": ["lm_head.NVFP4_marker"]}
            (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            expect_raises("I19 layer 2: a config that merely MENTIONS NVFP4 still fails the "
                          "staged quant assert",
                          lambda: _defs_namespace(text, root7), "want NVFP4/modelopt")
        finally:
            shutil.rmtree(root7, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop("B122_INPUT_ROOT", None)
        os.environ.pop("TAAF_KAGGLE_WORKING_DIR", None)


def _shim() -> dict:
    return json.loads((HERE / "b122_shim_files.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# M — the projection arithmetic
# ---------------------------------------------------------------------------
def section_math(text: str) -> None:
    print("\n[M] projection arithmetic")
    check("M1 tokens-per-action derived from the frozen 27B anchor (192*7920/480)",
          "TOKENS_PER_ACTION = 3168.0" in text)
    check("M2 bar 100 actions/window and the incumbent line 192 tok/s both present",
          "ACTION_BAR = 100.0" in text and "TOK_S_27B = 192.0" in text)
    check("M3 harness concurrency 28 used for the aggregate measurement",
          "CONCURRENCY = 28" in text)
    tok_s_bar = 100.0 * 3168.0 / 7920.0
    check("M4 bar is 40.0 agg tok/s; the incumbent line is 4.8x harder",
          abs(tok_s_bar - 40.0) < 1e-9 and abs(192.0 / tok_s_bar - 4.8) < 1e-9,
          f"bar={tok_s_bar} tok/s")
    check("M5 both verdict lines are emitted, and neither is read as capability",
          "bar_verdict" in text and "incumbent_verdict" in text
          and "no capability" in text and "a17_error_model.md" in text)
    check("M6 the tokens-per-action assumption is disclosed in the output itself",
          "tokens-per-action is the 27B constant, NOT measured for the 122B" in text)
    check("M7 results persisted as an artifact", "b122_canary.json" in text)
    # --- the coordinator's two readout requirements ---
    check("M8 batch sweep covers 1 -> harness concurrency (the routing-tax instrument)",
          "for n in (1, 4, 8, 16, CONCURRENCY):" in text
          and "scaling_efficiency" in text)
    check("M9 scaling efficiency is defined against the dense ideal and its meaning stated",
          "row['tok_s'] / (row['n'] * single_tok_s)" in text
          and "MoE routing tax" in text)
    check("M10 bytes/token reported as a bracketed RANGE, labelled DERIVED not counted",
          "bytes_per_token_implied_GB" in text and "0.70" in text
          and "NOT a counter read" in text and "SPEC_HBM_GB_S = 1792.0" in text)
    check("M11 the static prior and the 27B reference travel with it",
          "static_prior=17-18 GB" in text and "27B_reference=~27 GB" in text)
    check("M12 vLLM's own measured memory lines are captured",
          "_b122_memory_lines()" in text and "model loading took" in text)
    check("M13 UPPER-BOUND framing is emitted, with FAIL decisive / PASS not",
          "UPPER BOUND" in text and "FAIL is decisive" in text
          and "projection_is_upper_bound" in text)
    check("M14 runtime banner answers the LoRA lane's question in a child with vllm_env",
          "_b122_runtime_banner()" in text and "'vllm.lora'" in text
          and "env=vllm_env()" in text)


def section_probe_math() -> None:
    """Execute the sweep/derivation arithmetic on synthetic numbers."""
    print("\n[P] probe arithmetic executed")
    single = 20.0
    spec = 1792.0
    gb_hi, gb_lo = spec / single, 0.70 * spec / single
    check("P1 bracket orders low<high and is the 0.70..1.00 spec band",
          gb_lo < gb_hi and abs(gb_lo - 62.72) < 1e-6 and abs(gb_hi - 89.6) < 1e-6,
          f"{gb_lo:.2f}..{gb_hi:.2f} GB")
    # A model reading ~17.5 GB/token at 80% of spec would decode at:
    implied = 0.80 * spec / 17.5
    check("P2 the static 17.5 GB/token prior implies ~82 tok/s batch-1 at 80% of spec",
          80 < implied < 85, f"{implied:.1f} tok/s")
    # 27B at ~27 GB/token, same assumption:
    implied27 = 0.80 * spec / 27.0
    check("P3 and the 27B's ~27 GB implies ~53 tok/s — so the 122B should out-decode it "
          "at batch 1 by ~1.5x if the prior holds",
          abs(implied / implied27 - 27.0 / 17.5) < 1e-6, f"{implied27:.1f} tok/s")
    sweep = [{"n": 1, "tok_s": 20.0}, {"n": 28, "tok_s": 140.0}]
    eff = sweep[1]["tok_s"] / (sweep[1]["n"] * sweep[0]["tok_s"])
    check("P4 scaling efficiency computes as defined (28x batch, 7x throughput => 0.25)",
          abs(eff - 0.25) < 1e-9, f"eff={eff}")
    check("P5 bar of 100 actions still equals 40.0 agg tok/s after the rewrite",
          abs(100.0 * 3168.0 / 7920.0 - 40.0) < 1e-9)


def main() -> int:
    print("B122 BOOT CANARY SMOKE")
    section_build()
    section_structure()
    text = _rewritten_setup_text()
    section_rewrite(text)
    section_cuda(text)
    section_integration(text)
    section_math(text)
    section_probe_math()
    print(f"\n{len(PASS)} passed / {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
