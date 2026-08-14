"""B122 BOOT CANARY builder (2026-08-13) — screen the 122B MoE brain.

Candidate: ``ippeiogawa/qwen35-122b-a10b-nvfp4`` (Qwen3.5-122B-A10B, NVFP4/ModelOpt,
122B total / 10B active), served on ``jcole75/arc3-qwen36-runtime-wheels``
(vLLM 0.24.0 + transformers 5.13.0 + nvidia-modelopt 0.45.0 + CUDA 13.3).
Both public, both free.  Census: ``learnings/war_room/what_the_field_runs_2026-08-13.md``
sec(c)#1 / sec(d).  Verification of every census claim: ``learnings/war_room/brain122b_2026-08-13.md``.

WHY THIS IS NOT THE 72B REPEAT.  The 72B-dense-AWQ died on decode bandwidth
(~36 GB read/token vs the 27B-FP8's ~27 GB) -> 26-33 actions/window against a
>=100 bar.  The 122B-A10B-NVFP4 activates 10B params at 4 bits ~ 5 GB/token,
i.e. ~5x LESS traffic than the incumbent, on a card with native FP4 tensor
cores.  On arithmetic it should be FASTER than the model we run today.  This
canary measures that and nothing else.

BUILT FROM: ``notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb``
— the frozen upstream fork (17 cells, code-identical to jeroencottaar's 1.62 kernel per
``learnings/war_room/harness_diff_2026-08-13.md``).  NEVER hand-built
(``feedback_arc_kernel_structural_drift``: 5 ERRORs, all from hand-built kernels).
Fresh kernel slug (``feedback_fresh_kernel_slug``).

WHAT THE CANARY DOES (boot only — NO game window, ~1 GPU-h):
  1. installs the jcole75 wheelhouse (``wheels/`` subdir + ``requirements-runtime.txt``,
     NOT ``requirements.lock`` at the root — the install path differs from the duck's),
  2. STAGES the model dir (see below),
  3. boots vLLM with ``--quantization modelopt_fp4`` explicitly,
  4. FAIL-LOUD boot asserts: ``/v1/models`` identity, ONE forced tool-call round-trip
     through ``qwen3_coder``, ONE real image through the vision tower,
  5. measures decode throughput single-stream and at the harness's concurrency 28,
     and projects actions/window against the >=100 bar and against the 27B incumbent,
  6. skips the benchmark entirely and exits.

THE STAGING SHIM — this is a defect in the upstream snapshot, found by auditing the
Kaggle file listing on 2026-08-13 (20 files, enumerated in the war-room note):

  * ``tokenizer_config.json`` is ABSENT.  ``AutoTokenizer.from_pretrained`` would have
    to resolve the tokenizer class from ``config.json``'s ``model_type: qwen3_5_moe``,
    and ``AutoProcessor`` would have no tokenizer at all -> plausible hard boot / MM failure.
  * ``processor_config.json`` is ABSENT (only ``preprocessor_config.json`` ships).
  * ``chat_template.jinja`` has NO ``preserve_thinking`` branch (line 100), so the
    harness's ``--default-chat-template-kwargs '{"preserve_thinking": true}'`` would
    SILENTLY NO-OP and every prior-turn reasoning block would be stripped — a behaviour
    change disguised as a config no-op.  This is the trap named in the census.

All three are fixed by copying the corresponding files from the 27B we already serve
(``driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot``), embedded here as base64 (~10 KB total,
so the 27B does NOT have to be attached — which also keeps the total attached input down).
That is safe because the two tokenizers are the same: vocab_size 248320 on both, and
identical special-token ids (endoftext 248044, im_start 248045, im_end 248046,
vision_start 248053, vision_end 248054, image_pad 248056, video_pad 248057) — verified
against the 122B's own ``tokenizer.json`` added_tokens and ``config.json``.

Using the 27B's chat template also reverts the OTHER differing line (122, tool-arg
serialisation) to the exact behaviour our 100+ submissions were drawn under, so the
canary A/Bs ONE variable: the brain.

DELIBERATE OMISSIONS from the census's step-4 arg list, so a first canary cannot die on
an avoidable flag:
  * ``--kv-cache-dtype fp8`` NOT set.  The checkpoint declares ``kv_cache_quant_algo: FP8``
    and applies it itself; KV is not the binding constraint anyway (~0.4 GB at 32k with
    ``full_attention_interval 4`` -> only 12 of 48 layers cache; 0.81 GB even at fp16,
    against 17.8 GiB of headroom).  Forcing it risks an unsupported-combination abort on
    a hybrid linear-attention model for no memory gain.
  * ``--limit-mm-per-prompt`` NOT set.  ``MULTIMODAL_CONTEXT=current_grid`` is one image;
    the stock 27B serve line does not set it either.  Minimal delta wins.
``--quantization modelopt_fp4`` IS set explicitly (``a17_72b_screen_scope.md:301`` — never
let vLLM guess the quant).  ``--gpu-memory-utilization 0.93`` IS set (77.76 GiB of weights
in 95.6 GiB; vLLM's 0.9 default would leave only ~8.3 GiB).

Every rewrite anchor must match EXACTLY once or this script raises.
Idempotent by construction: it always rebuilds the output notebook from the frozen fork.

Usage:  python duck_eval/a17/build_b122_boot_canary.py
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
SRC_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "b122-canary"
OUT_NB = OUT_DIR / "arc3-b122-boot-canary.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "canivel/arc3-b122-boot-canary"
WHEELS_DS = "jcole75/arc3-qwen36-runtime-wheels"
MODEL_DS = "ippeiogawa/qwen35-122b-a10b-nvfp4"
SOURCE_DS = "jeroencottaar/taaf-kaggle-source-share"

# --- frozen 27B anchor (learnings/war_room/a17_envelope_onepager.md sec"Throughput envelope")
#     "The 27B-FP8 baseline serves the full window at 192 tok/s job-wallclock with
#      480 actions/7920 s pooled over the 4 screen games (w0_eval_s1, frozen numerator)."
#     => tokens per action = 192 * 7920 / 480 = 3168.0
TOK_S_27B = 192.0
ACTIONS_27B = 480.0
WINDOW_S = 7920.0
TOKENS_PER_ACTION = TOK_S_27B * WINDOW_S / ACTIONS_27B  # 3168.0
ACTION_BAR = 100.0
# RTX PRO 6000 Blackwell: 96 GB GDDR7, 512-bit @ 28 Gbps => ~1792 GB/s spec.
# Used ONLY to bracket an implied bytes/token; this rail exposes no HBM traffic counter.
SPEC_HBM_GB_S = 1792.0

# ---------------------------------------------------------------------------
# Shim payloads: the three files the upstream 122B snapshot is missing / has wrong.
# Sourced verbatim from driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot on 2026-08-13.
# sha256[:16] recorded so a future reader can re-derive them.
# ---------------------------------------------------------------------------
SHIM_PATH = Path(__file__).with_name("b122_shim_files.json")
SHIM_EXPECT = {
    "tokenizer_config.json": ("792fa3f0cb88b111", 1165),
    "processor_config.json": ("d89ef49ce9cd37fb", 1191),
    "chat_template.jinja": ("e84f32a23fdda276", 7764),
}


def _load_shim() -> dict[str, str]:
    if not SHIM_PATH.exists():
        raise SystemExit(
            f"FATAL: shim payload {SHIM_PATH} missing. Re-create it with the three files from\n"
            f"  kaggle datasets download driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot "
            f"-f {{tokenizer_config.json,processor_config.json,chat_template.jinja}}"
        )
    blobs = json.loads(SHIM_PATH.read_text(encoding="utf-8"))
    if sorted(blobs) != sorted(SHIM_EXPECT):
        raise SystemExit(f"FATAL: shim payload keys {sorted(blobs)} != {sorted(SHIM_EXPECT)}")
    for name, b64 in blobs.items():
        raw = base64.b64decode(b64)
        want_sha, want_len = SHIM_EXPECT[name]
        got_sha = hashlib.sha256(raw).hexdigest()[:16]
        if len(raw) != want_len or got_sha != want_sha:
            raise SystemExit(
                f"FATAL: shim {name} is {len(raw)} B sha {got_sha}, want {want_len} B sha {want_sha}"
            )
    tmpl = base64.b64decode(blobs["chat_template.jinja"]).decode("utf-8")
    if "preserve_thinking is defined and preserve_thinking is true" not in tmpl:
        raise SystemExit("FATAL: the 27B chat template has no preserve_thinking branch — wrong file")
    return blobs


# ---------------------------------------------------------------------------
# Cell 2 — banner
# ---------------------------------------------------------------------------
CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
CELL2_NEW = '''os.environ["B122_BOOT_ONLY"] = "1"  # boot canary: measure the serve path, SKIP the game window
print(
    "B122-CANARY mode=boot-only "
    "brain=ippeiogawa/qwen35-122b-a10b-nvfp4 (Qwen3.5-122B-A10B NVFP4, 10B active) "
    "wheels=jcole75/arc3-qwen36-runtime-wheels (vLLM 0.24.0 / modelopt 0.45.0 / CUDA 13.3) "
    "base=frozen duckfork (17 cells, code-identical to the 1.62 kernel) "
    "measurement=decode throughput ONLY; k=1 => NO capability reading "
    "(learnings/a17_error_model.md: k=1 false-NO-GO = 1.0)",
    flush=True,
)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'''

# ---------------------------------------------------------------------------
# Cell 6 — the attached-dataset list is hardcoded here as well as in the metadata
# ---------------------------------------------------------------------------
CELL6_ANCHOR = (
    'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
    '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
    '"driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]'
)
CELL6_NEW = (
    "# B122-CANARY: the duck's pinned wheelhouse (vLLM 0.19.0, H100 build, NO ModelOpt) cannot\n"
    "# load an NVFP4/ModelOpt checkpoint. Swapped for jcole75's SM120 build. The 27B is NOT\n"
    "# attached: the three files we need from it are embedded in the setup command (~10 KB).\n"
    f'DATASET_SOURCES = ["{SOURCE_DS}", "{WHEELS_DS}", "{MODEL_DS}"]'
)

# ---------------------------------------------------------------------------
# Cell 8 — the serve-config rewrite (applied to the single rendered setup command)
# ---------------------------------------------------------------------------
CELL8_ANCHOR = (
    "# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
    "env = _command_env()\n"
    'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
)

# ---------------------------------------------------------------------------
# Cell 14 — boot-only short circuit
# ---------------------------------------------------------------------------
CELL14_ANCHOR_HEAD = "# Build the live competition game list from the gateway's available environments.\n"
CELL14_NEW_HEAD = (
    "# B122-CANARY: boot-only. Everything decisive already ran in the setup command\n"
    "# (cell 8): vLLM boot, /v1/models identity, tool-call round-trip, MM round-trip,\n"
    "# throughput probe. The game window is skipped so the kernel costs ~1 GPU-h.\n"
    'B122_BOOT_ONLY = os.environ.get("B122_BOOT_ONLY") == "1" and not TRUE_SUBMISSION\n'
    "# Build the live competition game list from the gateway's available environments.\n"
)

CELL14_ANCHOR_RUN = (
    "try:\n"
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
)
CELL14_NEW_RUN = (
    "try:\n"
    "    if B122_BOOT_ONLY:\n"
    '        print("B122-CANARY: boot-only — SKIPPING the game window on purpose. '
    'The measurement is in the setup-command output above.", flush=True)\n'
    "    else:\n"
    "        await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
)


def _setup_rewrites(shim: dict[str, str]) -> list[tuple[str, str]]:
    """(anchor, replacement) pairs applied to the rendered setup command text."""
    shim_literal = json.dumps(shim, sort_keys=True)
    return [
        # --- 1. wheelhouse + model identity -------------------------------------
        ("WHEELHOUSE_OWNER = 'driessmit1'", "WHEELHOUSE_OWNER = 'jcole75'"),
        (
            "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
            "WHEELHOUSE_SLUG = 'arc3-qwen36-runtime-wheels'",
        ),
        ("MODEL_OWNER = 'driessmit1'", "MODEL_OWNER = 'ippeiogawa'"),
        (
            "MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
            "MODEL_SLUG = 'qwen35-122b-a10b-nvfp4'",
        ),
        (
            "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
            "SERVED_MODEL_NAME = 'Qwen/Qwen3.5-122B-A10B'",
        ),
        ("VLLM_MAX_MODEL_LEN = 65536", "VLLM_MAX_MODEL_LEN = 32768"),
        (
            "STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
            "STAMP_TEXT = 'vllm==0.24.0 transformers==5.13.0 nvidia-modelopt==0.45.0 cuda13.3\\n'",
        ),
        # --- 2. wheelhouse layout differs: wheels/ subdir + requirements-runtime.txt
        (
            "    requirements = WHEELHOUSE / 'requirements.lock'",
            "    requirements = WHEELHOUSE / 'requirements-runtime.txt'\n"
            "    # B122-CANARY: jcole75's layout is a wheels/ subdir plus a runtime requirements\n"
            "    # file, not a lock file at the wheelhouse root (the duck wheelhouse's layout).",
        ),
        (
            "        '--find-links',\n        str(WHEELHOUSE),\n",
            "        '--find-links',\n        str(WHEELHOUSE / 'wheels'),\n",
        ),
        # --- 3. model staging (missing tokenizer/processor cfg + preserve_thinking fix)
        (
            "MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)",
            _STAGER.replace("__SHIM_JSON__", shim_literal),
        ),
        # --- 4. serve args: explicit NVFP4 quant + memory util ---------------------
        (
            "        '--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
            "        '--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
            "        # B122-CANARY: never let vLLM guess the quant (a17_72b_screen_scope.md:301).\n"
            "        '--quantization',\n        'modelopt_fp4',\n"
            "        # 77.76 GiB of weights in 95.6 GiB: vLLM's 0.9 default leaves only ~8.3 GiB.\n"
            "        '--gpu-memory-utilization',\n        '0.93',\n"
            "        # Point vLLM at the STAGED (preserve_thinking-restored) template explicitly.\n"
            "        # Implicit model-dir pickup would probably work, but the 27B tokenizer_config\n"
            "        # carries no chat_template key, so the .jinja file is the ONLY source — and a\n"
            "        # silent fallback here is precisely the no-op trap this canary exists to avoid.\n"
            "        '--chat-template',\n        str(B122_CHAT_TEMPLATE),\n    ]",
        ),
        # --- 4b. CUDA 13 loader path (v1 died here) --------------------------------
        # v1 INFRA DEATH, 2026-08-13: `import vllm` aborted at
        # `vllm/platforms/cuda.py: import vllm._C_stable_libtorch` with
        # `ImportError: libcudart.so.13: cannot open shared object file`.
        # vLLM 0.24.0 links CUDA 13; the Kaggle image ships CUDA 12, so the runtime
        # arrives only as pip wheels under <SITE_PACKAGES>/nvidia/*/lib — a path the
        # dynamic loader knows nothing about. The duck's vllm_env() sets PYTHONPATH and
        # nothing else, which was sufficient for its own CUDA-12.8 build because the
        # image already provided those .so files. It is NOT sufficient here.
        (
            "            'VLLM_NO_USAGE_STATS': '1',\n"
            "        }\n"
            "    )\n"
            "    return env",
            "            'VLLM_NO_USAGE_STATS': '1',\n"
            "        }\n"
            "    )\n"
            "    env['LD_LIBRARY_PATH'] = os.pathsep.join(\n"
            "        _b122_cuda_lib_dirs() + [p for p in [env.get('LD_LIBRARY_PATH', '')] if p]\n"
            "    )\n"
            "    return env",
        ),
        (
            "def vllm_env() -> dict[str, str]:",
            "def _b122_cuda_lib_dirs() -> list:\n"
            "    # Every shared-library dir the pip-installed CUDA 13 toolchain and torch ship.\n"
            "    dirs = []\n"
            "    for pattern in ('nvidia/*/lib', 'nvidia/*/lib64', 'torch/lib', 'cusparselt/lib'):\n"
            "        dirs.extend(str(p) for p in sorted(SITE_PACKAGES.glob(pattern)) if p.is_dir())\n"
            "    return dirs\n"
            "\n"
            "\n"
            "def _b122_assert_cuda_runtime() -> None:\n"
            "    # FAIL FAST and LEGIBLY, before a 45-minute weight load, if the CUDA 13\n"
            "    # runtime is still not resolvable. v1 spent a whole GPU-hour to learn this.\n"
            "    dirs = _b122_cuda_lib_dirs()\n"
            "    found = []\n"
            "    for d in dirs:\n"
            "        found.extend(str(p) for p in sorted(Path(d).glob('libcudart.so*')))\n"
            "    print('B122-CANARY cuda_lib_dirs=%d libcudart=%s'\n"
            "          % (len(dirs), found or 'NONE'), flush=True)\n"
            "    if not found:\n"
            "        raise RuntimeError(\n"
            "            'B122-CANARY FATAL: libcudart.so* not found under any of %r. vLLM 0.24.0 '\n"
            "            'links CUDA 13 and the Kaggle image ships CUDA 12, so the runtime must come '\n"
            "            'from the wheelhouse. This is the v1 death (ImportError: libcudart.so.13).'\n"
            "            % (dirs,)\n"
            "        )\n"
            "\n"
            "\n"
            "def vllm_env() -> dict[str, str]:",
        ),
        (
            "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()",
            "def start_vllm_server() -> None:\n"
            "    install_vllm_wheelhouse()\n"
            "    _b122_assert_cuda_runtime()",
        ),
        # --- 5. boot budget: the 45-min load KILL rule, encoded --------------------
        (
            "def wait_for_vllm_server(timeout_seconds: int = 900) -> None:",
            "def wait_for_vllm_server(timeout_seconds: int = 2700) -> None:\n"
            "    # B122-CANARY: 45 min = the census's 'weights load > 45 min => KILL' rule,\n"
            "    # encoded mechanically. 83.5 GB off a Kaggle mount cannot fit the 900 s default.",
        ),
        # --- 6. GPU identity, printed loudly --------------------------------------
        (
            "assert_expected_cuda_gpu()\nmissing",
            "assert_expected_cuda_gpu()\n"
            "_b122_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', "
            "'--format=csv,noheader'], capture_output=True, text=True).stdout.strip()\n"
            "print('B122-CANARY gpu=' + _b122_gpu, flush=True)\n"
            "if 'rtx pro 6000' not in _b122_gpu.lower():\n"
            "    raise RuntimeError('B122-CANARY FATAL: GPU ' + repr(_b122_gpu) + "
            "' is not RTX PRO 6000 — VOID (wrong SKU, measurement not comparable)')\n"
            "missing",
        ),
        # --- 7. the canary body ----------------------------------------------------
        ("\n\ndef run_vllm_api_smoke_test() -> None:", _CANARY_BODY + "\n\ndef run_vllm_api_smoke_test() -> None:"),
        (
            "start_vllm_server()\nrun_vllm_api_smoke_test()\nsetup_env = {",
            "_b122_t0 = time.monotonic()\n"
            "start_vllm_server()\n"
            "_b122_boot_s = time.monotonic() - _b122_t0\n"
            "print('B122-CANARY boot_seconds=%.1f (install + weights load + engine init)' % _b122_boot_s, flush=True)\n"
            "run_vllm_api_smoke_test()\n"
            "_b122_serve_asserts()\n"
            "_b122_throughput_probe(_b122_boot_s)\n"
            "setup_env = {",
        ),
    ]


# The staged-model builder, injected in place of the one-line MODEL_PATH resolve.
_STAGER = r"""_B122_SHIM_FILES = json.loads(r'''__SHIM_JSON__''')


def _b122_find_model() -> Path:
    # Marker-based and mount-path agnostic (same discipline as the a17 canary).
    # The ONLY acceptable artifact is the Qwen3.5-122B-A10B NVFP4 checkpoint; if it is
    # not attached we FAIL LOUDLY rather than silently serving something else.
    input_root = Path(os.environ.get('B122_INPUT_ROOT', '/kaggle/input'))
    hits = []
    for cfg in sorted(input_root.rglob('config.json')):
        try:
            text = cfg.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if 'Qwen3_5MoeForConditionalGeneration' in text and 'NVFP4' in text:
            if any(cfg.parent.glob('*.safetensors')):
                hits.append(cfg.parent)
    if not hits:
        raise RuntimeError(
            'B122-CANARY FATAL: Qwen3.5-122B-A10B-NVFP4 not found under ' + str(input_root)
            + ' - attach dataset ippeiogawa/qwen35-122b-a10b-nvfp4. REFUSING any fallback brain.'
        )
    if len(hits) > 1:
        raise RuntimeError('B122-CANARY FATAL: multiple NVFP4 candidate dirs: '
                           + ', '.join(str(h) for h in hits))
    return hits[0]


def _b122_stage_model() -> Path:
    # The upstream snapshot is missing tokenizer_config.json and processor_config.json,
    # and its chat_template.jinja has NO preserve_thinking branch (verified against the
    # Kaggle file listing, 2026-08-13). Symlink the read-only mount into a writable dir
    # and drop the three files in from the 27B we already serve (same vocab 248320, same
    # special-token ids). Restoring the 27B template ALSO reverts the tool-arg
    # serialisation line, so exactly ONE variable changes: the brain.
    import base64 as _b64

    src = _b122_find_model()
    staged = WORKING_DIR / 'b122-model'
    if staged.exists():
        shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True, exist_ok=True)
    # Link, never copy: the checkpoint is 83.5 GB and /kaggle/working is not that big.
    linked = 0
    mode = 'symlink'
    for entry in sorted(src.iterdir()):
        if not entry.is_file():
            continue
        target = staged / entry.name
        try:
            os.symlink(entry, target)
        except OSError:
            try:
                os.link(entry, target)
                mode = 'hardlink'
            except OSError as exc:
                raise RuntimeError(
                    'B122-CANARY FATAL: cannot link %s into the staging dir (%r). Refusing to '
                    'COPY an 83.5 GB checkpoint into /kaggle/working.' % (entry.name, exc)
                ) from exc
        linked += 1
    shipped = []
    for name, blob in sorted(_B122_SHIM_FILES.items()):
        target = staged / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.write_bytes(_b64.b64decode(blob))
        shipped.append(name)
    # FAIL-LOUD post-conditions.
    shards = sorted(staged.glob('*.safetensors'))
    if len(shards) != 9:
        raise RuntimeError('B122-CANARY FATAL: staged %d safetensors shards, want 9' % len(shards))
    template = (staged / 'chat_template.jinja').read_text(encoding='utf-8')
    if 'preserve_thinking is defined and preserve_thinking is true' not in template:
        raise RuntimeError(
            'B122-CANARY FATAL: staged chat template has no preserve_thinking branch - the '
            '--default-chat-template-kwargs flag would SILENTLY NO-OP and strip prior-turn reasoning'
        )
    tok_cfg = json.loads((staged / 'tokenizer_config.json').read_text(encoding='utf-8'))
    proc_cfg = json.loads((staged / 'processor_config.json').read_text(encoding='utf-8'))
    if not tok_cfg.get('tokenizer_class'):
        raise RuntimeError('B122-CANARY FATAL: staged tokenizer_config.json has no tokenizer_class')
    if proc_cfg.get('processor_class') != 'Qwen3VLProcessor':
        raise RuntimeError('B122-CANARY FATAL: staged processor_class != Qwen3VLProcessor')
    cfg = json.loads((staged / 'config.json').read_text(encoding='utf-8'))
    quant = cfg.get('quantization_config', {})
    if quant.get('quant_algo') != 'NVFP4' or quant.get('quant_method') != 'modelopt':
        raise RuntimeError('B122-CANARY FATAL: staged config quant is %r, want NVFP4/modelopt'
                           % (quant,))
    print('B122-CANARY staged_model=%s src=%s link_mode=%s linked=%d shim=%s shards=%d'
          % (staged, src, mode, linked, ','.join(shipped), len(shards)), flush=True)
    print('B122-CANARY preserve_thinking_branch=RESTORED (upstream 122B template lacks it)', flush=True)
    return staged


MODEL_PATH = _b122_stage_model()
B122_CHAT_TEMPLATE = MODEL_PATH / 'chat_template.jinja'"""


# The boot asserts + throughput probe, injected before run_vllm_api_smoke_test.
_CANARY_BODY = r"""

def _b122_png_b64(size: int = 64) -> str:
    # Dependency-free solid-colour PNG for the MM boot probe.
    import base64
    import struct
    import zlib

    raw = (b'\x00' + b'\xc8\x32\x32' * size) * size

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    png = (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr)
           + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b''))
    return base64.b64encode(png).decode('ascii')


def _b122_kv_from_log() -> str:
    text = tail_server_log(4000)
    for line in text.splitlines():
        low = line.lower()
        if 'kv cache size' in low or 'gpu kv cache' in low or 'available kv cache memory' in low:
            return line.strip()[:200]
    return 'not-reported'


def _b122_memory_lines() -> dict:
    # vLLM prints the MEASURED weights footprint and KV allocation at boot. This is the
    # only real byte-count the rail exposes; HBM traffic counters are not available.
    out = {}
    for line in tail_server_log(6000).splitlines():
        low = line.lower()
        if 'model loading took' in low and 'weights' not in out:
            out['weights'] = line.strip()[:220]
        elif 'kv cache size' in low or 'gpu kv cache' in low:
            out.setdefault('kv', line.strip()[:220])
        elif 'memory profiling' in low or 'non_torch_memory' in low or 'peak_memory' in low:
            out.setdefault('profile', line.strip()[:220])
        elif 'graph capturing finished' in low:
            out.setdefault('graphs', line.strip()[:220])
    return out


_B122_RUNTIME_PROBE = (
    "import json\n"
    "out = {}\n"
    "for name in ('vllm', 'transformers', 'torch'):\n"
    "    try:\n"
    "        out[name] = __import__(name).__version__\n"
    "    except Exception as exc:\n"
    "        out[name] = 'IMPORT-FAILED ' + repr(exc)[:120]\n"
    "for name in ('vllm.lora', 'modelopt'):\n"
    "    try:\n"
    "        __import__(name)\n"
    "        out[name] = 'OK'\n"
    "    except Exception as exc:\n"
    "        out[name] = 'FAILED ' + repr(exc)[:120]\n"
    "print(json.dumps(out))\n"
)


def _b122_runtime_banner() -> dict:
    # Cheap runtime facts the OTHER lanes will ask the moment this rail changes.
    # MUST run in a child with vllm_env(): the wheelhouse is on the CHILD's PYTHONPATH
    # (and LD_LIBRARY_PATH), never on this process's.
    proc = subprocess.run([sys.executable, '-c', _B122_RUNTIME_PROBE],
                          env=vllm_env(), capture_output=True, text=True)
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        info = {'error': (proc.stdout + proc.stderr)[-400:]}
    print('B122-CANARY runtime ' + json.dumps(info, sort_keys=True), flush=True)
    return info


def _b122_serve_asserts() -> None:
    # FAIL-LOUD boot gate. Any miss raises -> kernel ERROR. A dead canary is a retry;
    # a silently-wrong serve path is a poisoned measurement.
    models = request_json(VLLM_BASE_URL + '/models', timeout=60)
    ids = sorted(m.get('id', '') for m in models.get('data', []))
    if ids != ['Qwen/Qwen3.5-122B-A10B']:
        raise RuntimeError('B122-CANARY FATAL: served model ids ' + repr(ids)
                           + " != ['Qwen/Qwen3.5-122B-A10B']")
    print('B122-CANARY models_endpoint=OK id=Qwen/Qwen3.5-122B-A10B', flush=True)
    print('B122-CANARY kv_cache_line=' + _b122_kv_from_log(), flush=True)

    # (1) tool-call round-trip through the harness's own parser (qwen3_coder).
    tools = [{'type': 'function', 'function': {
        'name': 'submit_action',
        'description': 'Submit the next ARC action.',
        'parameters': {'type': 'object',
                       'properties': {'action': {'type': 'string'},
                                      'x': {'type': 'integer'},
                                      'y': {'type': 'integer'}},
                       'required': ['action']}}}]
    payload = {'model': SERVED_MODEL_NAME,
               'messages': [{'role': 'user',
                             'content': 'Call the submit_action tool with action ACTION6, x 3, y 7.'}],
               'tools': tools,
               'tool_choice': {'type': 'function', 'function': {'name': 'submit_action'}},
               'temperature': 0.0, 'max_tokens': 512}
    response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
    calls = response['choices'][0]['message'].get('tool_calls') or []
    if not calls or calls[0].get('function', {}).get('name') != 'submit_action':
        raise RuntimeError('B122-CANARY FATAL: tool-call round-trip FAILED under qwen3_coder '
                           '(silent-zero class): ' + json.dumps(response)[:2000])
    args = json.loads(calls[0]['function'].get('arguments') or '{}')
    if 'action' not in args:
        raise RuntimeError('B122-CANARY FATAL: tool-call arguments missing key action: ' + repr(args))
    print('B122-CANARY tool_call_roundtrip=OK parser=qwen3_coder args='
          + json.dumps(args, sort_keys=True), flush=True)

    # (2) one real image through the vision tower.
    image_url = 'data:image/png;base64,' + _b122_png_b64()
    payload = {'model': SERVED_MODEL_NAME,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': image_url}},
                   {'type': 'text', 'text': 'Answer with one word: what colour is this image?'}]}],
               'temperature': 0.0, 'max_tokens': 64}
    response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
    content = (response['choices'][0]['message'].get('content') or '').strip()
    if not content:
        raise RuntimeError('B122-CANARY FATAL: MM boot probe returned empty content - vision path broken')
    print('B122-CANARY mm_image_roundtrip=OK reply=' + repr(content[:60]), flush=True)

    # (3) preserve_thinking actually binds. REPORTED, not fatal: /tokenize is a vLLM
    #     extension and an endpoint rename must not cost a GPU-hour. The hard guarantee
    #     is the staged-template assert in _b122_stage_model().
    try:
        tok_url = VLLM_BASE_URL.rsplit('/v1', 1)[0] + '/tokenize'
        convo = [{'role': 'user', 'content': 'first question'},
                 {'role': 'assistant',
                  'content': '<think>\nA long private reasoning block that should be kept.\n</think>\n\nanswer'},
                 {'role': 'user', 'content': 'second question'}]
        counts = {}
        for flag in (True, False):
            got = request_json(tok_url, payload={'model': SERVED_MODEL_NAME, 'messages': convo,
                                                 'chat_template_kwargs': {'preserve_thinking': flag}},
                               timeout=120)
            counts[flag] = int(got.get('count') or len(got.get('tokens') or []))
        verdict = 'BINDS' if counts[True] > counts[False] else 'NO-OP'
        print('B122-CANARY preserve_thinking_check=%s tokens_true=%d tokens_false=%d'
              % (verdict, counts[True], counts[False]), flush=True)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print('B122-CANARY preserve_thinking_check=UNAVAILABLE (' + repr(exc)[:200] + ')', flush=True)


def _b122_throughput_probe(boot_seconds: float) -> None:
    # THE DECISIVE MEASUREMENT. The 72B died here: 26-33 actions/window against a >=100 bar,
    # because a 72B dense AWQ reads ~36 GB/token vs the 27B-FP8's ~27 GB. The 122B-A10B-NVFP4
    # reads ~5 GB/token, so the census predicts it beats the INCUMBENT, not merely the bar.
    import concurrent.futures
    import random
    import time as _t

    # Frozen 27B anchor, learnings/war_room/a17_envelope_onepager.md:
    # 192 tok/s job-wallclock, 480 actions / 7920 s pooled over the 4 screen games.
    TOK_S_27B = __TOK_S_27B__
    ACTIONS_27B = __ACTIONS_27B__
    WINDOW_S = __WINDOW_S__
    TOKENS_PER_ACTION = __TOKENS_PER_ACTION__
    ACTION_BAR = __ACTION_BAR__
    CONCURRENCY = 28          # the harness's own concurrency (benchmark_initial.pkl)
    MAX_TOKENS = 384
    # ~6k-token prompt: a realistic mid-game analysis turn under ANALYZER_CONTEXT_WINDOW=32768.
    FILLER = ('The ARC-AGI-3 board is a 64x64 grid of colour indices; the current frame and the '
              'previous frame are given below, followed by the action history for this level. ') * 190

    runtime_info = _b122_runtime_banner()
    ignore_eos_ok = True
    try:
        request_json(VLLM_BASE_URL + '/chat/completions', timeout=300, payload={
            'model': SERVED_MODEL_NAME, 'messages': [{'role': 'user', 'content': 'hi'}],
            'max_tokens': 8, 'temperature': 0.0, 'ignore_eos': True})
    except Exception:
        ignore_eos_ok = False
    print('B122-CANARY ignore_eos_supported=%s' % ignore_eos_ok, flush=True)

    def _one(tag, max_tokens):
        payload = {'model': SERVED_MODEL_NAME,
                   'messages': [{'role': 'user',
                                 'content': 'REQ-%s-%d %s Now describe, in detail, the next '
                                            'action plan and why.' % (tag, random.getrandbits(48), FILLER)}],
                   'temperature': 0.0, 'max_tokens': max_tokens}
        if ignore_eos_ok:
            payload['ignore_eos'] = True
        started = _t.monotonic()
        got = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=3600)
        usage = got.get('usage', {}) or {}
        return (int(usage.get('completion_tokens') or 0),
                int(usage.get('prompt_tokens') or 0),
                _t.monotonic() - started)

    _one('warmup', 16)  # pay the first-call JIT / graph cost outside the timed window

    # --- BATCH SWEEP. This is the instrument that answers "does a ~1.5x bandwidth edge
    # --- actually convert, or does MoE routing + the BF16 attention path eat it?"
    # At batch 1 decode is bandwidth-bound and reads the whole active weight set per token.
    # As batch grows, a DENSE model amortises that read across the batch (near-linear tok/s).
    # An MoE does NOT: distinct sequences route to distinct experts, so the number of experts
    # touched per step grows with batch. The scaling curve therefore measures the routing tax
    # directly — no HBM counter needed (this rail exposes none).
    sweep = []
    for n in (1, 4, 8, 16, CONCURRENCY):
        t0 = _t.monotonic()
        # Identical per-request work at every point, so scaling_efficiency is a clean ratio.
        if n == 1:
            res = [_one('s1_0', MAX_TOKENS)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                res = list(pool.map(lambda i: _one('s%d_%d' % (n, i), MAX_TOKENS), range(n)))
        wall = _t.monotonic() - t0
        g = sum(r[0] for r in res)
        p = sum(r[1] for r in res)
        tps = g / wall if wall > 0 else 0.0
        sweep.append({'n': n, 'gen_tokens': g, 'prompt_tokens': p,
                      'wall_s': wall, 'tok_s': tps})
        print('B122-CANARY sweep n=%-3d gen=%-6d wall_s=%7.2f tok_s=%8.2f'
              % (n, g, wall, tps), flush=True)

    single_tok_s = sweep[0]['tok_s']
    agg_tok_s = sweep[-1]['tok_s']
    gen = sweep[-1]['gen_tokens']
    prompt = sweep[-1]['prompt_tokens']
    wall = sweep[-1]['wall_s']
    for row in sweep:
        row['scaling_efficiency'] = (
            row['tok_s'] / (row['n'] * single_tok_s) if single_tok_s > 0 else 0.0)
    print('B122-CANARY scaling ' + ' '.join(
        'n%d=%.3f' % (r['n'], r['scaling_efficiency']) for r in sweep), flush=True)
    print('B122-CANARY scaling_note 1.0 = perfect weight-read amortisation (dense ideal); '
          'a low value at high n is the MoE routing tax + the BF16 attention path, i.e. the '
          'bandwidth edge NOT converting.', flush=True)

    # --- BYTES PER TOKEN. The rail exposes no HBM traffic counter, so this is DERIVED from
    # --- measured batch-1 decode against the card's spec bandwidth, and reported as a RANGE
    # --- (real achieved bandwidth is typically 70-100% of spec). Static prior from
    # --- model.safetensors.index.json is ~17-18 GB/token; the 27B-FP8 reads ~27 GB.
    SPEC_HBM_GB_S = __SPEC_HBM_GB_S__
    if single_tok_s > 0:
        gb_hi = SPEC_HBM_GB_S / single_tok_s          # if the card hit 100% of spec
        gb_lo = 0.70 * SPEC_HBM_GB_S / single_tok_s   # if it hit 70%
    else:
        gb_hi = gb_lo = 0.0
    print('B122-CANARY bytes_per_token_implied_GB=%.1f..%.1f (DERIVED: %.0f GB/s spec x '
          '0.70..1.00 / batch-1 decode %.2f tok/s; NOT a counter read). static_prior=17-18 GB, '
          '27B_reference=~27 GB' % (gb_lo, gb_hi, SPEC_HBM_GB_S, single_tok_s), flush=True)
    mem = _b122_memory_lines()
    for k, v in sorted(mem.items()):
        print('B122-CANARY vllm_mem_%s=%s' % (k, v), flush=True)

    projected = agg_tok_s * WINDOW_S / TOKENS_PER_ACTION
    speedup = agg_tok_s / TOK_S_27B if TOK_S_27B else 0.0
    print('B122-CANARY tokens_per_action_27B=%.1f (frozen: %.0f tok/s * %.0f s / %.0f actions)'
          % (TOKENS_PER_ACTION, TOK_S_27B, WINDOW_S, ACTIONS_27B), flush=True)
    print('B122-CANARY actions_projected_%.0fs=%.1f bar=%.0f incumbent_27B=%.0f speedup_vs_27B=%.3fx'
          % (WINDOW_S, projected, ACTION_BAR, ACTIONS_27B, speedup), flush=True)
    # Two pre-registered lines. The bar is the 72B's grave; the incumbent line is the
    # census's own prediction ("should be faster than the incumbent, not slower").
    bar_verdict = 'PASS' if projected >= ACTION_BAR else 'FAIL'
    inc_verdict = 'PASS' if agg_tok_s >= TOK_S_27B else 'FAIL'
    print('B122-CANARY VERDICT bar_>=%.0f_actions=%s incumbent_>=%.0f_tok_s=%s boot_seconds=%.1f'
          % (ACTION_BAR, bar_verdict, TOK_S_27B, inc_verdict, boot_seconds), flush=True)
    print('B122-CANARY NOTE tokens-per-action is the 27B constant, NOT measured for the 122B; '
          'a verbosity difference moves the projection. k=1 => MEASUREMENT ONLY, no capability '
          'reading (learnings/a17_error_model.md).', flush=True)
    # ---- THE ASYMMETRY, stated so the readout cannot be over-claimed ----
    # The 27B anchor (192 tok/s, 480 actions/7920 s) is JOB-WALLCLOCK over a real agentic run:
    # it includes prefill, tool execution, env stepping and idle. THIS probe is pure synthetic
    # generation, so it necessarily reports a HIGHER tok/s than the same model would show
    # job-wallclock. Converting it through the 27B's tokens-per-action therefore yields an
    # UPPER BOUND on actions/window. Consequence, pre-registered:
    #   * projected < 100  => DECISIVE ENVELOPE-FAIL (even the optimistic instrument misses).
    #   * projected >= 100 => NOT decisive; it only licenses the full screen, which measures
    #                         actions directly on the 25 offline environments.
    print('B122-CANARY FRAMING projection is an UPPER BOUND (synthetic generation vs the 27B '
          "anchor's job-wallclock, which includes prefill/tool/env/idle). FAIL is decisive; "
          'PASS only licenses the full screen. The clean apples-to-apples control is this same '
          'probe against the 27B — not yet run.', flush=True)
    summary = {'boot_seconds': boot_seconds, 'single_tok_s': single_tok_s,
               'agg_tok_s': agg_tok_s, 'concurrency': CONCURRENCY,
               'gen_tokens': gen, 'prompt_tokens': prompt, 'wall_s': wall,
               'sweep': sweep,
               'bytes_per_token_implied_GB': [gb_lo, gb_hi],
               'spec_hbm_gb_s': SPEC_HBM_GB_S,
               'static_prior_gb_per_token': 17.5,
               'gb_per_token_27B_reference': 27.0,
               'vllm_memory_lines': mem,
               'runtime': runtime_info,
               'projection_is_upper_bound': True,
               'tokens_per_action_27B': TOKENS_PER_ACTION, 'window_s': WINDOW_S,
               'actions_projected': projected, 'action_bar': ACTION_BAR,
               'tok_s_27B': TOK_S_27B, 'speedup_vs_27B': speedup,
               'bar_verdict': bar_verdict, 'incumbent_verdict': inc_verdict,
               'kv_cache_line': _b122_kv_from_log(),
               'ignore_eos_supported': ignore_eos_ok}
    (WORKING_DIR / 'b122_canary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('B122-CANARY wrote b122_canary.json', flush=True)
"""

_CANARY_BODY = (
    _CANARY_BODY.replace("__TOK_S_27B__", repr(TOK_S_27B))
    .replace("__ACTIONS_27B__", repr(ACTIONS_27B))
    .replace("__WINDOW_S__", repr(WINDOW_S))
    .replace("__TOKENS_PER_ACTION__", repr(TOKENS_PER_ACTION))
    .replace("__ACTION_BAR__", repr(ACTION_BAR))
    .replace("__SPEC_HBM_GB_S__", repr(SPEC_HBM_GB_S))
)

# Tokens that must be GONE from the rewritten setup command (27B / duck-wheelhouse artifacts).
VETO = (
    "'driessmit1'",
    "arc3-vllm-h100-wheelhouse-v3",
    "vrfai-qwen3-6-27b-fp8-hf-snapshot",
    "vrfai/Qwen3.6-27B-FP8",
    "requirements.lock",
    "VLLM_MAX_MODEL_LEN = 65536",
)
# Tokens that MUST be present after the rewrite.
REQUIRE = (
    "jcole75",
    "arc3-qwen36-runtime-wheels",
    "ippeiogawa",
    "qwen35-122b-a10b-nvfp4",
    "Qwen/Qwen3.5-122B-A10B",
    "requirements-runtime.txt",
    "WHEELHOUSE / 'wheels'",
    "modelopt_fp4",
    "'--gpu-memory-utilization'",
    "VLLM_MAX_MODEL_LEN = 32768",
    "_b122_stage_model()",
    "_b122_serve_asserts()",
    "_b122_throughput_probe(_b122_boot_s)",
    "preserve_thinking is defined and preserve_thinking is true",
    "timeout_seconds: int = 2700",
    "B122-CANARY gpu=",
    "LD_LIBRARY_PATH",
    "_b122_assert_cuda_runtime()",
    "libcudart.so",
    "B122-CANARY sweep n=",
    "bytes_per_token_implied_GB",
    "B122-CANARY FRAMING projection is an UPPER BOUND",
    "_b122_runtime_banner()",
    "vllm.lora",
)


def _build_cell8_block(rewrites: list[tuple[str, str]]) -> str:
    """The literal source injected into cell 8, mirroring the a17 loud-fail pattern."""
    return (
        "# --- B122-CANARY BEGIN serve-config rewrite (loud-fail; no fallback brain) ---\n"
        "# POLICY INVERSION, as in the a17 canary: eval grafts normally fall back to vanilla\n"
        "# duck on failure. Here a vanilla run would SILENTLY SERVE THE 27B and report the\n"
        "# INCUMBENT's throughput as the 122B's, so ANY rewrite failure raises -> kernel ERROR.\n"
        "B122_SETUP_REWRITES = " + repr(rewrites) + "\n"
        "B122_VETO = " + repr(VETO) + "\n"
        "B122_REQUIRE = " + repr(REQUIRE) + "\n"
        "\n"
        "\n"
        "def _b122_patch_setup_commands(commands):\n"
        "    if not isinstance(commands, list) or len(commands) != 1:\n"
        "        raise RuntimeError('B122-CANARY FATAL: expected exactly 1 setup command, got '\n"
        "                           + repr(commands)[:200])\n"
        "    text = commands[0]\n"
        "    for old, new in B122_SETUP_REWRITES:\n"
        "        found = text.count(old)\n"
        "        if found != 1:\n"
        "            raise RuntimeError('B122-CANARY FATAL: serve-config anchor matched %d times '\n"
        "                               '(want 1): %r' % (found, old[:110]))\n"
        "        text = text.replace(old, new)\n"
        "    for veto in B122_VETO:\n"
        "        if veto in text:\n"
        "            raise RuntimeError('B122-CANARY FATAL: 27B/duck-wheelhouse artifact %r survived '\n"
        "                               'the rewrite' % veto)\n"
        "    for need in B122_REQUIRE:\n"
        "        if need not in text:\n"
        "            raise RuntimeError('B122-CANARY FATAL: required 122B serve token %r missing '\n"
        "                               'after rewrite' % need)\n"
        "    print('B122-CANARY setup-commands rewrite OK (%d anchors replaced; loud-fail, no 27B '\n"
        "          'fallback)' % len(B122_SETUP_REWRITES), flush=True)\n"
        "    return [text]\n"
        "\n"
        "\n"
        "# --- B122-CANARY END serve-config rewrite ---\n"
        "# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
        "env = _command_env()\n"
        "for command in _b122_patch_setup_commands("
        'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())):'
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        raise SystemExit(f"FATAL {label}: anchor matched {found} times (want 1): {old[:110]!r}")
    return text.replace(old, new)


def main() -> int:
    shim = _load_shim()
    rewrites = _setup_rewrites(shim)

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    if len(nb["cells"]) != 17:
        raise SystemExit(f"FATAL: base notebook has {len(nb['cells'])} cells, want 17 (frozen fork)")

    edits = {
        2: [(CELL2_ANCHOR, CELL2_NEW)],
        6: [(CELL6_ANCHOR, CELL6_NEW)],
        8: [(CELL8_ANCHOR, _build_cell8_block(rewrites))],
        14: [(CELL14_ANCHOR_HEAD, CELL14_NEW_HEAD), (CELL14_ANCHOR_RUN, CELL14_NEW_RUN)],
    }
    for idx, pairs in edits.items():
        cell = nb["cells"][idx]
        if cell["cell_type"] != "code":
            raise SystemExit(f"FATAL: cell {idx} is {cell['cell_type']}, want code")
        src = "".join(cell["source"])
        for old, new in pairs:
            src = _replace_once(src, old, new, f"cell {idx}")
        compile(src, f"cell{idx}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        cell["source"] = src.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    # The rewritten setup command must itself be valid Python (it is exec'd by python -c).
    setup_src = json.loads((REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json").read_text())
    if len(setup_src) != 1:
        raise SystemExit("FATAL: reference taaf bundle does not hold exactly 1 setup command")
    text = setup_src[0]
    for old, new in rewrites:
        text = _replace_once(text, old, new, "setup command")
    for veto in VETO:
        if veto in text:
            raise SystemExit(f"FATAL: veto token {veto!r} survived the setup rewrite")
    for need in REQUIRE:
        if need not in text:
            raise SystemExit(f"FATAL: required token {need!r} missing from the setup rewrite")
    # The setup command is `"$PYTHON" - <<'PYSETUP' ... PYSETUP`: a QUOTED heredoc, so the body
    # reaches python verbatim (no shell expansion) — but it must still be valid Python, and
    # nothing we inject may contain the sentinel.
    lines = text.splitlines()
    if lines[0] != '"$PYTHON" - <<\'PYSETUP\'' or lines[-1] != "PYSETUP":
        raise SystemExit(f"FATAL: unexpected heredoc wrapper {lines[0]!r} ... {lines[-1]!r}")
    body = "\n".join(lines[1:-1])
    if "PYSETUP" in body:
        raise SystemExit("FATAL: injected text contains the heredoc sentinel PYSETUP")
    compile(body, "setup_command", "exec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/", 1)[1]
    meta["code_file"] = OUT_NB.name
    meta["dataset_sources"] = [SOURCE_DS, WHEELS_DS, MODEL_DS]
    meta.pop("model_sources", None)  # dropped silently at push (root-caused 07-25/07-26)
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"b122 boot canary written: {OUT_NB}")
    print(f"  cells: {len(nb['cells'])} (edited 2, 6, 8, 14)")
    print(f"  setup rewrites: {len(rewrites)} anchors, all matched exactly once")
    print(f"  datasets: {meta['dataset_sources']}")
    print(f"  docker_image: {meta['docker_image']}")
    print(f"  tokens_per_action(27B) = {TOKENS_PER_ACTION}  bar = {ACTION_BAR} actions/{WINDOW_S:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
