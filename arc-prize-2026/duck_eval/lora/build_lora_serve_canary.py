"""LORA SERVE CANARY builder (2026-08-13) — prove the adapter serve path, before training.

WHY THIS EXISTS. Nobody in this competition has been *observed* serving a LoRA on the duck
stack. `auxentr` (#163) specified it completely and correctly — their `setup_commands.json`
adds `--enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules nca-ppt=<path>` — but their
public kernel ERRORed at t=12 s in cell 2 (`uv pip install ... --no-warn-conflicts`, an
unexpected argument, because they ran the public image with `enable_gpu:false`), and their two
kernels that DO complete never attach the adapter. So the public record contains a recipe and
zero executions. Our own verdict (`learnings/war_room/lora_lane_2026-08-13.md` §2) is a read of
the pinned wheel, not a run. This canary converts both into a measurement for ~1 GPU-h, and it
does so BEFORE a single training token is spent.

THE DESIGN POINT — two adapters, not one. A standard LoRA init has `B = 0`, so its delta is
exactly zero and its output is token-identical to the base. That is perfect for measuring the
*cost* of `--enable-lora`, and useless for proving the adapter is actually applied: a silently
ignored adapter looks identical. So we serve two:

    arc3-noop   B = 0        -> MUST be token-identical to the base
    arc3-probe  B ~ 1e-3     -> MUST differ from the base

    | noop==base | probe!=base | verdict                                                    |
    |------------|-------------|------------------------------------------------------------|
    | yes        | yes         | loaded AND applied. PASS.                                   |
    | yes        | no          | SILENTLY IGNORED — the exact failure Tufa's own              |
    |            |             | `vllm_runtime_lora_guard` exists to catch. FAIL.            |
    | no         | yes         | a zero-delta adapter changed the output => numerically unsound. FAIL. |

Both adapters are built locally by `make_probe_adapters.py`, with key names and shapes taken as
GROUND TRUTH from auxentr's shipped `adapter_model.safetensors` (128 tensors, F32, 10,485,760
params; `q_proj.lora_B` is [12288,16] because Qwen3.5 uses gated attention — this is the number
that corrected our whole parameter table).

BUILT FROM: `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` — the
frozen upstream fork, code-identical to jeroencottaar's 1.62 kernel. NEVER hand-built
(`feedback_arc_kernel_structural_drift`: 5 ERRORs, all hand-built). Fresh slug
(`feedback_fresh_kernel_slug`).

WHAT IT CHANGES vs the scored config — deliberately minimal, ONE variable:
  * `--enable-lora --max-loras 2 --max-lora-rank 16 --lora-dtype auto --lora-modules ...`
  * nothing else. Same wheelhouse (vLLM 0.19.0), same model, same tool/reasoning parser, same
    prefix caching, same max-model-len, same BYOD image, same machine shape.

WHAT IT MEASURES:
  1. `/v1/models` lists the base AND both adapters
  2. noop == base and probe != base at temperature 0 (the differential above)
  3. tool-call round-trip through `qwen3_coder`, addressed to the ADAPTER name
  4. one real image through the vision tower, addressed to the ADAPTER name
  5. `preserve_thinking` still binds with LoRA enabled (reported, not fatal)
  6. throughput base vs adapter at the harness's own concurrency 28, projected to
     actions/window against the >=100 bar and against the 27B incumbent — i.e. the LoRA tax

Usage:  python duck_eval/lora/build_lora_serve_canary.py
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
SRC_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "lora-serve-canary"
OUT_NB = OUT_DIR / "arc3-lora-serve-canary.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"
ADAPTER_DIR = REPO / "runs" / "lora_lane" / "probe_adapters"

KERNEL_ID = "canivel/arc3-lora-serve-canary"
ADAPTER_DS = "canivel/arc3-lora-probe-adapters"
WHEELS_DS = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
SOURCE_DS = "jeroencottaar/taaf-kaggle-source-share"
MODEL_DS = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"

# Frozen 27B throughput anchor (learnings/war_room/a17_envelope_onepager.md):
# 192 tok/s job-wallclock, 480 actions / 7920 s pooled over the 4 screen games.
TOK_S_27B = 192.0
ACTIONS_27B = 480.0
WINDOW_S = 7920.0
TOKENS_PER_ACTION = TOK_S_27B * WINDOW_S / ACTIONS_27B  # 3168.0
ACTION_BAR = 100.0
LORA_RANK = 16

# ---------------------------------------------------------------------------
# cell 2 — banner
# ---------------------------------------------------------------------------
CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
CELL2_NEW = '''os.environ["LORA_BOOT_ONLY"] = "1"  # serve canary: measure the LoRA serve path, SKIP the game window
print(
    "LORA-SERVE-CANARY mode=boot-only "
    "brain=driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot (UNCHANGED, the scored brain) "
    "wheels=driessmit1/arc3-vllm-h100-wheelhouse-v3 (UNCHANGED, vLLM 0.19.0) "
    "delta=--enable-lora + two RANDOM r16 adapters (noop B=0, probe B~1e-3) AND NOTHING ELSE "
    "measures=adapter load + differential apply + tool-call + MM + LoRA throughput tax; "
    "NO training tokens were spent to produce these adapters",
    flush=True,
)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'''

# ---------------------------------------------------------------------------
# cell 6 — attached datasets are hardcoded here as well as in the metadata
# ---------------------------------------------------------------------------
CELL6_ANCHOR = (
    'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
    '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
    '"driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]'
)
CELL6_NEW = (
    "# LORA-SERVE-CANARY: the scored triple is UNCHANGED; the probe adapters are the only\n"
    "# addition (two 40 MB directories, built locally, never trained).\n"
    f'DATASET_SOURCES = ["{SOURCE_DS}", "{WHEELS_DS}", "{MODEL_DS}", "{ADAPTER_DS}"]'
)

# ---------------------------------------------------------------------------
# cell 14 — boot-only short circuit
# ---------------------------------------------------------------------------
CELL14_ANCHOR_HEAD = "# Build the live competition game list from the gateway's available environments.\n"
CELL14_NEW_HEAD = (
    "# LORA-SERVE-CANARY: boot-only. Everything decisive already ran in the setup command\n"
    "# (cell 8): adapter load, noop/probe differential, tool-call + MM round-trips, LoRA\n"
    "# throughput tax. The game window is skipped so the kernel costs ~1 GPU-h.\n"
    'LORA_BOOT_ONLY = os.environ.get("LORA_BOOT_ONLY") == "1" and not TRUE_SUBMISSION\n'
    "# Build the live competition game list from the gateway's available environments.\n"
)
CELL14_ANCHOR_RUN = (
    "try:\n"
    "    await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
)
CELL14_NEW_RUN = (
    "try:\n"
    "    if LORA_BOOT_ONLY:\n"
    '        print("LORA-SERVE-CANARY: boot-only \\u2014 SKIPPING the game window on purpose. '
    'The measurement is in the setup-command output above.", flush=True)\n'
    "    else:\n"
    "        await bm.run(soft_end_time=soft_end, runtime_environment=target, "
    "minimal_diagnostics=TRUE_SUBMISSION)\n"
)

# ---------------------------------------------------------------------------
# The canary body, injected into the rendered setup command.
# ---------------------------------------------------------------------------
_CANARY_BODY = r'''

# ===================== LORA-SERVE-CANARY =====================
_LORA_ADAPTERS = {'arc3-noop': None, 'arc3-probe': None}


def _lora_find_adapters() -> None:
    root = resolve_kaggle_dataset_path('__ADAPTER_OWNER__', '__ADAPTER_SLUG__')
    if not root.exists():
        raise FileNotFoundError('LORA-CANARY FATAL: adapter dataset not mounted at ' + str(root))
    for name, sub in (('arc3-noop', 'lora-noop'), ('arc3-probe', 'lora-probe')):
        hits = [p.parent for p in root.rglob('adapter_config.json') if p.parent.name == sub]
        if not hits:
            listing = sorted(str(p.relative_to(root)) for p in root.rglob('*'))[:40]
            raise FileNotFoundError(
                'LORA-CANARY FATAL: no ' + sub + '/adapter_config.json under ' + str(root)
                + ' contents=' + repr(listing))
        path = hits[0]
        weights = path / 'adapter_model.safetensors'
        raw = weights.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        want = __ADAPTER_SHA__[sub]
        if digest != want:
            raise RuntimeError('LORA-CANARY FATAL: ' + sub + ' sha ' + digest + ' != ' + want
                               + ' (the dataset push did not ship what we built)')
        cfg = json.loads((path / 'adapter_config.json').read_text())
        if int(cfg.get('r', 0)) != __LORA_RANK__:
            raise RuntimeError('LORA-CANARY FATAL: ' + sub + ' rank ' + repr(cfg.get('r')))
        _LORA_ADAPTERS[name] = path
        print('LORA-CANARY adapter %s path=%s bytes=%d sha=%s r=%d rslora=%s'
              % (name, path, len(raw), digest, cfg['r'], cfg.get('use_rslora')), flush=True)


def _lora_install_guard() -> None:
    """Wire Tufa's own, never-called `inference/tools/vllm_runtime_lora_guard.py` into the
    vLLM SERVER process. It turns "vLLM would silently ignore this LoRA module" warnings into
    hard errors -- exactly the failure class this canary is here to detect. The server is a
    subprocess, so the hook is delivered by a `sitecustomize.py` on its PYTHONPATH that defers
    the patch until `vllm.lora.worker_manager` is actually imported. Best-effort and REPORTED:
    the primary evidence is the noop/probe differential, which detects a silent ignore
    behaviourally. A guard that fails to install must not cost a GPU-hour."""
    guard_src = None
    for base in _source_path_entries(BUNDLE_DIR):
        candidate = Path(base) / 'inference' / 'tools' / 'vllm_runtime_lora_guard.py'
        if candidate.is_file():
            guard_src = candidate
            break
    if guard_src is None:
        print('LORA-CANARY guard=NOT-FOUND (vllm_runtime_lora_guard.py absent from the bundle)', flush=True)
        return
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    (SITE_PACKAGES / '_arc3_lora_guard.py').write_text(guard_src.read_text(), encoding='utf-8')
    (SITE_PACKAGES / 'sitecustomize.py').write_text(
        'import importlib.abc, importlib.machinery, sys\n'
        'class _H(importlib.abc.MetaPathFinder):\n'
        '    def find_module(self, *a, **k):\n'
        '        return None\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        '        if name != "vllm.lora.worker_manager":\n'
        '            return None\n'
        '        sys.meta_path.remove(self)\n'
        '        spec = importlib.machinery.PathFinder.find_spec(name, path, target)\n'
        '        if spec is None:\n'
        '            return None\n'
        '        orig = spec.loader.exec_module\n'
        '        def exec_module(module, _orig=orig):\n'
        '            _orig(module)\n'
        '            try:\n'
        '                import _arc3_lora_guard\n'
        '                _arc3_lora_guard.install_runtime_lora_warning_guard()\n'
        '                print("LORA-CANARY guard=INSTALLED (in vLLM server process)", flush=True)\n'
        '            except Exception as exc:\n'
        '                print("LORA-CANARY guard=FAILED " + repr(exc)[:200], flush=True)\n'
        '        spec.loader.exec_module = exec_module\n'
        '        return spec\n'
        'sys.meta_path.insert(0, _H())\n',
        encoding='utf-8')
    print('LORA-CANARY guard=STAGED at ' + str(SITE_PACKAGES / 'sitecustomize.py'), flush=True)


def _lora_png_b64(size: int = 64) -> str:
    # Dependency-free solid-colour PNG for the MM probe.
    import struct
    import zlib
    raw = (b'\x00' + b'\xc8\x32\x32' * size) * size

    def _chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    png = (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr)
           + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b''))
    return base64.b64encode(png).decode('ascii')


_LORA_DIFF_PROMPT = ('You are analysing a 64x64 ARC grid. In exactly one sentence, describe a '
                     'strategy for finding the shortest action sequence that clears a level.')


def _lora_greedy(model_name: str, n: int = 48) -> dict:
    payload = {'model': model_name,
               'messages': [{'role': 'user', 'content': _LORA_DIFF_PROMPT}],
               'temperature': 0.0, 'top_p': 1.0, 'seed': 20260813,
               'max_tokens': n, 'logprobs': True, 'top_logprobs': 1,
               'chat_template_kwargs': {'enable_thinking': False}}
    response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
    choice = response['choices'][0]
    content = choice['message'].get('content') or ''
    lp = choice.get('logprobs') or {}
    toks = [t.get('token') for t in (lp.get('content') or [])]
    vals = [t.get('logprob') for t in (lp.get('content') or [])]
    return {'text': content, 'tokens': toks, 'logprobs': vals}


def _lora_serve_asserts() -> None:
    # FAIL-LOUD. A dead canary is a retry; a silently-wrong serve path is a poisoned verdict.
    models = request_json(VLLM_BASE_URL + '/models', timeout=60)
    ids = sorted(m.get('id', '') for m in models.get('data', []))
    want = sorted([SERVED_MODEL_NAME, 'arc3-noop', 'arc3-probe'])
    if ids != want:
        raise RuntimeError('LORA-CANARY FATAL: /v1/models = ' + repr(ids) + ' != ' + repr(want)
                           + ' -- vLLM did not register both adapters')
    print('LORA-CANARY models_endpoint=OK ids=' + repr(ids), flush=True)

    # (1) THE DIFFERENTIAL. This is the whole canary.
    base = _lora_greedy(SERVED_MODEL_NAME)
    noop = _lora_greedy('arc3-noop')
    probe = _lora_greedy('arc3-probe')
    noop_same = noop['tokens'] == base['tokens']
    probe_diff = probe['tokens'] != base['tokens']
    first_div = next((i for i, (a, b) in enumerate(zip(base['tokens'], probe['tokens'])) if a != b),
                     None)
    print('LORA-CANARY differential noop_identical_to_base=%s probe_differs_from_base=%s '
          'first_divergent_token_index=%s' % (noop_same, probe_diff, first_div), flush=True)
    print('LORA-CANARY base_text=' + repr(base['text'][:160]), flush=True)
    print('LORA-CANARY probe_text=' + repr(probe['text'][:160]), flush=True)
    if not noop_same:
        raise RuntimeError('LORA-CANARY FATAL: a ZERO-delta adapter changed the output. The LoRA '
                           'path is numerically unsound, not merely inactive. base=%r noop=%r'
                           % (base['tokens'][:12], noop['tokens'][:12]))
    if not probe_diff:
        raise RuntimeError('LORA-CANARY FATAL: a NON-ZERO adapter did not change the output. The '
                           'adapter is being SILENTLY IGNORED -- this is the exact failure class '
                           'Tufa built vllm_runtime_lora_guard for, and it would have read as '
                           '"LoRA did not help" after a full training run.')
    print('LORA-CANARY differential=PASS (adapter is loaded AND applied)', flush=True)

    # (2) tool-call round-trip through the harness's own parser, ADDRESSED TO THE ADAPTER.
    tools = [{'type': 'function', 'function': {
        'name': 'python',
        'description': 'Run python against the live frame.',
        'parameters': {'type': 'object',
                       'properties': {'code': {'type': 'string'}},
                       'required': ['code']}}}]
    payload = {'model': 'arc3-probe',
               'messages': [{'role': 'user',
                             'content': "Call the python tool with code exactly: action(['UP'])"}],
               'tools': tools,
               'tool_choice': {'type': 'function', 'function': {'name': 'python'}},
               'temperature': 0.0, 'max_tokens': 512}
    response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
    calls = response['choices'][0]['message'].get('tool_calls') or []
    if not calls or calls[0].get('function', {}).get('name') != 'python':
        raise RuntimeError('LORA-CANARY FATAL: tool-call round-trip FAILED under qwen3_coder WITH '
                           'LoRA enabled: ' + json.dumps(response)[:2000])
    args = json.loads(calls[0]['function'].get('arguments') or '{}')
    if 'code' not in args:
        raise RuntimeError('LORA-CANARY FATAL: tool-call arguments missing key code: ' + repr(args))
    print('LORA-CANARY tool_call_roundtrip=OK parser=qwen3_coder on=arc3-probe code='
          + repr(str(args['code'])[:60]), flush=True)

    # (3) one real image through the vision tower, ADDRESSED TO THE ADAPTER.
    payload = {'model': 'arc3-probe',
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url',
                    'image_url': {'url': 'data:image/png;base64,' + _lora_png_b64()}},
                   {'type': 'text', 'text': 'Answer with one word: what colour is this image?'}]}],
               'temperature': 0.0, 'max_tokens': 64}
    response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
    content = (response['choices'][0]['message'].get('content') or '').strip()
    if not content:
        raise RuntimeError('LORA-CANARY FATAL: MM probe returned empty content with LoRA enabled '
                           '-- the vision path does not survive --enable-lora')
    print('LORA-CANARY mm_image_roundtrip=OK on=arc3-probe reply=' + repr(content[:60]), flush=True)

    # (4) preserve_thinking still binds with LoRA on. REPORTED, never fatal.
    try:
        tok_url = VLLM_BASE_URL.rsplit('/v1', 1)[0] + '/tokenize'
        convo = [{'role': 'user', 'content': 'first question'},
                 {'role': 'assistant',
                  'content': '<think>\nA long private reasoning block that should be kept.\n</think>\n\nanswer'},
                 {'role': 'user', 'content': 'second question'}]
        counts = {}
        for flag in (True, False):
            got = request_json(tok_url, payload={'model': 'arc3-probe', 'messages': convo,
                                                 'chat_template_kwargs': {'preserve_thinking': flag}},
                               timeout=120)
            counts[flag] = int(got.get('count') or len(got.get('tokens') or []))
        print('LORA-CANARY preserve_thinking_check=%s tokens_true=%d tokens_false=%d'
              % ('BINDS' if counts[True] > counts[False] else 'NO-OP', counts[True], counts[False]),
              flush=True)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print('LORA-CANARY preserve_thinking_check=UNAVAILABLE (' + repr(exc)[:200] + ')', flush=True)


def _lora_throughput(boot_seconds: float) -> None:
    """THE COST MEASUREMENT. `--enable-lora` is not free: vLLM adds a shrink/expand pair per
    targeted layer. If the tax pushes actions/window under the >=100 bar that killed the 72B,
    the lane is dead on serving grounds regardless of how good the adapter is."""
    import concurrent.futures
    import time as _t

    TOKENS_PER_ACTION = __TOKENS_PER_ACTION__
    ACTION_BAR = __ACTION_BAR__
    WINDOW_S = __WINDOW_S__
    TOK_S_27B = __TOK_S_27B__
    CONCURRENCY = 28          # the harness's own concurrency (benchmark_initial.pkl)
    MAX_TOKENS = 384
    FILLER = ('The ARC-AGI-3 board is a 64x64 grid of colour indices; the current frame and the '
              'previous frame are given below, followed by the action history for this level. ') * 190

    def one(model_name, index):
        payload = {'model': model_name,
                   'messages': [{'role': 'user', 'content': FILLER + ' Request %d. Reply at length.' % index}],
                   'temperature': 0.7, 'max_tokens': MAX_TOKENS,
                   'chat_template_kwargs': {'enable_thinking': False}}
        got = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=1800)
        usage = got.get('usage') or {}
        return int(usage.get('completion_tokens') or 0)

    results = {}
    for label, model_name in (('base', SERVED_MODEL_NAME), ('adapter', 'arc3-probe')):
        t0 = _t.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            counts = list(pool.map(lambda i: one(model_name, i), range(CONCURRENCY)))
        elapsed = max(1e-6, _t.monotonic() - t0)
        tok_s = sum(counts) / elapsed
        actions = tok_s * WINDOW_S / TOKENS_PER_ACTION
        results[label] = {'tok_s': tok_s, 'actions_per_window': actions,
                          'completion_tokens': sum(counts), 'elapsed_s': elapsed}
        print('LORA-CANARY throughput[%s] concurrency=%d tokens=%d elapsed=%.1fs tok_s=%.1f '
              'actions_per_window=%.0f (bar %.0f, 27B incumbent %.0f)'
              % (label, CONCURRENCY, sum(counts), elapsed, tok_s, actions, ACTION_BAR,
                 TOK_S_27B * WINDOW_S / TOKENS_PER_ACTION), flush=True)

    tax = 1.0 - results['adapter']['tok_s'] / max(1e-9, results['base']['tok_s'])
    verdict = 'PASS' if results['adapter']['actions_per_window'] >= ACTION_BAR else 'FAIL'
    print('LORA-CANARY lora_throughput_tax=%.1f%% adapter_actions_per_window=%.0f verdict=%s'
          % (100 * tax, results['adapter']['actions_per_window'], verdict), flush=True)
    summary = {'boot_seconds': boot_seconds, 'lora_tax_fraction': tax,
               'action_bar': ACTION_BAR, 'verdict': verdict, 'throughput': results}
    out = Path(os.environ.get('TAAF_KAGGLE_WORKING_DIR', '/kaggle/working')) / 'lora_canary.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('LORA-CANARY summary written to ' + str(out), flush=True)
'''


def _setup_rewrites(shas: dict[str, str]) -> list[tuple[str, str]]:
    body = (
        _CANARY_BODY
        .replace("__ADAPTER_OWNER__", ADAPTER_DS.split("/")[0])
        .replace("__ADAPTER_SLUG__", ADAPTER_DS.split("/")[1])
        .replace("__ADAPTER_SHA__", json.dumps(shas, sort_keys=True))
        .replace("__LORA_RANK__", str(LORA_RANK))
        .replace("__TOKENS_PER_ACTION__", repr(TOKENS_PER_ACTION))
        .replace("__ACTION_BAR__", repr(ACTION_BAR))
        .replace("__WINDOW_S__", repr(WINDOW_S))
        .replace("__TOK_S_27B__", repr(TOK_S_27B))
    )
    return [
        # hashlib/base64 are needed by the canary body and are not imported by the stock setup.
        ("import json\nimport os\n", "import base64\nimport hashlib\nimport json\nimport os\n"),
        # the canary body
        ("\n\ndef run_vllm_api_smoke_test() -> None:", body + "\n\ndef run_vllm_api_smoke_test() -> None:"),
        # serve args: the ONLY functional change to the scored serve line
        (
            "        '--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n    ]",
            "        '--max-model-len',\n        str(VLLM_MAX_MODEL_LEN),\n"
            "        # LORA-SERVE-CANARY: the entire delta vs the scored config.\n"
            "        '--enable-lora',\n"
            "        '--max-loras',\n        '2',\n"
            f"        '--max-lora-rank',\n        '{LORA_RANK}',\n"
            "        '--lora-dtype',\n        'auto',\n"
            "        '--lora-modules',\n"
            "        'arc3-noop=' + str(_LORA_ADAPTERS['arc3-noop']),\n"
            "        'arc3-probe=' + str(_LORA_ADAPTERS['arc3-probe']),\n    ]",
        ),
        # resolve + verify the adapters and stage the guard before the server starts
        (
            "print(f'vLLM wheelhouse path: {WHEELHOUSE}', flush=True)",
            "_lora_find_adapters()\n"
            "print(f'vLLM wheelhouse path: {WHEELHOUSE}', flush=True)",
        ),
        (
            "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()",
            "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()\n"
            "    _lora_install_guard()",
        ),
        # run the canary after the stock smoke test
        (
            "start_vllm_server()\nrun_vllm_api_smoke_test()\nsetup_env = {",
            "_lora_t0 = time.monotonic()\n"
            "start_vllm_server()\n"
            "_lora_boot_s = time.monotonic() - _lora_t0\n"
            "print('LORA-CANARY boot_seconds=%.1f (install + weights load + adapter load + engine init)'\n"
            "      % _lora_boot_s, flush=True)\n"
            "run_vllm_api_smoke_test()\n"
            "_lora_serve_asserts()\n"
            "_lora_throughput(_lora_boot_s)\n"
            "setup_env = {",
        ),
    ]


# Tokens that MUST survive / MUST NOT appear after the rewrite. A vanilla fallback here
# would serve the base with no adapter and report it as "LoRA works" -- the exact silent-null
# this canary exists to prevent -- so every check is loud-fail.
REQUIRE = ["--enable-lora", "--max-lora-rank", "arc3-noop=", "arc3-probe=",
           "_lora_serve_asserts()", "_lora_throughput(", "_lora_find_adapters()"]
VETO: list[str] = []

CELL8_ANCHOR = (
    "# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
    "env = _command_env()\n"
    'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
)


def _build_cell8_block(rewrites: list[tuple[str, str]]) -> str:
    """Source injected into cell 8. The setup command lives in the ATTACHED BUNDLE and is only
    materialized at runtime, so the rewrites must be applied there, not here. Loud-fail: any
    anchor miss raises and ERRORs the kernel rather than silently serving an adapter-free base."""
    return (
        "# --- LORA-SERVE-CANARY BEGIN serve-config rewrite (loud-fail; no adapter-free fallback) ---\n"
        "# A vanilla fallback would serve the base with no adapter and report it as a LoRA run.\n"
        "# That is the silent null this canary exists to prevent, so ANY miss raises.\n"
        "LORA_SETUP_REWRITES = " + repr(rewrites) + "\n"
        "LORA_REQUIRE = " + repr(REQUIRE) + "\n"
        "\n"
        "\n"
        "def _lora_patch_setup_commands(commands):\n"
        "    if not isinstance(commands, list) or len(commands) != 1:\n"
        "        raise RuntimeError('LORA-CANARY FATAL: expected exactly 1 setup command, got '\n"
        "                           + repr(commands)[:200])\n"
        "    text = commands[0]\n"
        "    for old, new in LORA_SETUP_REWRITES:\n"
        "        found = text.count(old)\n"
        "        if found != 1:\n"
        "            raise RuntimeError('LORA-CANARY FATAL: anchor matched %d times (want 1): %r'\n"
        "                               % (found, old[:110]))\n"
        "        text = text.replace(old, new)\n"
        "    for need in LORA_REQUIRE:\n"
        "        if need not in text:\n"
        "            raise RuntimeError('LORA-CANARY FATAL: required token %r missing after rewrite'\n"
        "                               % need)\n"
        "    print('LORA-CANARY setup-commands rewrite OK (%d anchors replaced; loud-fail)'\n"
        "          % len(LORA_SETUP_REWRITES), flush=True)\n"
        "    return [text]\n"
        "\n"
        "\n"
        "# --- LORA-SERVE-CANARY END serve-config rewrite ---\n"
        "# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.\n"
        "env = _command_env()\n"
        "for command in _lora_patch_setup_commands("
        'json.loads((BUNDLE_DIR / "setup_commands.json").read_text())):'
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"FATAL {label}: anchor matched {count} times (want 1): {old[:110]!r}")
    return text.replace(old, new)


def main() -> int:
    import ast

    if not SRC_NB.is_file():
        raise SystemExit(f"FATAL: frozen fork missing at {SRC_NB}")
    shas: dict[str, str] = {}
    for sub in ("lora-noop", "lora-probe"):
        weights = ADAPTER_DIR / sub / "adapter_model.safetensors"
        if not weights.is_file():
            raise SystemExit(f"FATAL: {weights} missing — run make_probe_adapters.py first")
        shas[sub] = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]

    rewrites = _setup_rewrites(shas)

    notebook = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 17:
        raise SystemExit(f"FATAL: frozen fork has {len(cells)} cells, expected 17")

    edits = {
        2: [(CELL2_ANCHOR, CELL2_NEW)],
        6: [(CELL6_ANCHOR, CELL6_NEW)],
        8: [(CELL8_ANCHOR, _build_cell8_block(rewrites))],
        14: [(CELL14_ANCHOR_HEAD, CELL14_NEW_HEAD), (CELL14_ANCHOR_RUN, CELL14_NEW_RUN)],
    }
    for index, pairs in edits.items():
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise SystemExit(f"FATAL: cell {index} is {cell['cell_type']}, want code")
        source = "".join(cell["source"])
        for old, new in pairs:
            source = _replace_once(source, old, new, f"cell {index}")
        compile(source, f"cell{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    # The rewritten setup command must itself be valid Python: it is a QUOTED heredoc body
    # handed to `python -`, so it reaches the interpreter verbatim.
    reference = json.loads((REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json").read_text(encoding="utf-8"))
    if len(reference) != 1:
        raise SystemExit("FATAL: reference taaf bundle does not hold exactly 1 setup command")
    text = reference[0]
    for old, new in rewrites:
        text = _replace_once(text, old, new, "setup command")
    for need in REQUIRE:
        if need not in text:
            raise SystemExit(f"FATAL: required token {need!r} missing from the setup rewrite")
    for veto in VETO:
        if veto in text:
            raise SystemExit(f"FATAL: veto token {veto!r} survived the setup rewrite")
    lines = text.splitlines()
    if lines[0] != '"$PYTHON" - <<\'PYSETUP\'' or lines[-1] != "PYSETUP":
        raise SystemExit("FATAL: rewritten setup command is no longer a quoted PYSETUP heredoc")
    if "PYSETUP" in "\n".join(lines[1:-1]):
        raise SystemExit("FATAL: injected code contains the PYSETUP heredoc sentinel")
    compile("\n".join(lines[1:-1]), "setup_command", "exec")
    print(f"setup-command rewrite validated locally: {len(rewrites)} anchors, "
          f"{len(text):,} B, compiles clean")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/", 1)[1]
    meta["code_file"] = OUT_NB.name
    meta["dataset_sources"] = [SOURCE_DS, WHEELS_DS, MODEL_DS, ADAPTER_DS]
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT_NB} ({OUT_NB.stat().st_size:,} B, {len(cells)} cells)")
    print(f"wrote {OUT_META}")
    print("adapter shas:", json.dumps(shas, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
