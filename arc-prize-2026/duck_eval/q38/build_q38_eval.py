"""Q38 ENGINE-SWAP EVAL builder (2026-08-15) — ONE variable: the weights.

WHY THIS EXISTS. `learnings/war_room/research_restart_2026-08-15.md` argues, at ~85 %
confidence, that the 2.5+ leaderboard regime that appeared on 08-14/08-15 is an ENGINE
GENERATION, not a technique: Alibaba shipped Qwen3.8-27B (Apache 2.0) at 15:00 UTC on
2026-08-14, three complete FP8 Kaggle mirrors were public within 8 h, every jumper except
cstl has a best-submission timestamp after the first mirror, and one participant reports
"a consistent 2x score on the local 25". The named PRIMARY FALSIFIER of that whole report
is one free build: screen the engine on our own local-25 harness. This is that build.

BUILT FROM: `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` —
the frozen upstream fork, the exact bytes the `duck-harness-kaggle` baseline family ran.
NEVER hand-built (`feedback_arc_kernel_structural_drift`: 5 ERRORs, all hand-built).
Fresh slug (`feedback_fresh_kernel_slug`).

A PLAIN BUILD *IS* THE EVAL. Unlike the war/sentinel eval builds (which fork the *duckwar*
baseline and need WARPACK_FORCE_OFFLINE_BENCH=1), the frozen fork's run cell branches on
TRUE_SUBMISSION = KAGGLE_IS_COMPETITION_RERUN: unset in any ordinary kernel BUILD, so it
plays the 25 bundled competition environments OFFLINE via `_offline_games()`, writes a dummy
submission.parquet, and is never scored. No eval-mode graft is needed or added.

WHAT CHANGES — exactly three code cells, and nothing else:
  cell 2  identity banner only (no behavioural change)
  cell 6  DATASET_SOURCES: the engine entry only
            driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot -> saltb0x/qwen3-8-27b-fp8
  cell 8  FAIL-LOUD, anchor-exact rewrite of the bundle's single setup command:
            MODEL_OWNER / MODEL_SLUG / SERVED_MODEL_NAME  (the engine)
            --default-chat-template-kwargs gains "reasoning_effort": "medium"  (the PIN)
            + pre-serve config asserts and post-serve boot asserts

WHY THE PIN IS "medium" AND WHY THAT IS THE NEUTRAL CHOICE — this is measured, not argued.
Qwen3.8's chat template adds a `reasoning_effort` knob defaulting to `xhigh`, which injects
an instruction paragraph into the system block. Reading the template (lines 45-56 of the
attached `chat_template.jinja`): `xhigh` and `low` each set `reasoning_instructions` to a
sentence, and `medium` is validated-but-silent — it leaves `reasoning_instructions` EMPTY.
Rendering both templates on a representative harness payload (system + tools + user +
assistant-with-tool_call + tool + user, `preserve_thinking=true`, `enable_thinking=true`):

    Qwen3.6 (incumbent)                       1495 chars  --
    reasoning_effort='medium'                 1495 chars  identical to Qwen3.6: YES (0 diff)
    reasoning_effort=None (default -> xhigh)  1704 chars  identical to Qwen3.6: NO
    reasoning_effort='xhigh'                  1704 chars  identical to Qwen3.6: NO
    reasoning_effort='low'                    1633 chars  identical to Qwen3.6: NO

(numbers of record: duck_eval/q38/q38_smoke.py section 5, which re-measures them every build)

So `medium` reproduces the Qwen3.6 prompt BYTE-FOR-BYTE. It is the only value that isolates
the weights. `xhigh` (the default) and `low` are the SEPARATE LATER ARM and are not touched
here — shipping the swap and the knob together is exactly how this campaign has burned draws.

HOW THE PIN REACHES THE TEMPLATE (verified in the pinned wheel, not assumed). The harness
sends per-request `chat_template_kwargs={"enable_thinking": bool}`
(`ARC3-Inference/inference/utils/openai_compat.py:78`). vLLM 0.19.0 merges server defaults
UNDER request kwargs — `vllm/entrypoints/openai/engine/serving.py:807`:
`return default_chat_template_kwargs | request_chat_template_kwargs`. The request never sends
`reasoning_effort`, so the server default survives; `enable_thinking` keeps coming from the
request exactly as today.

WHAT IS *NOT* CHANGED, and is asserted to survive the rewrite (the one-variable proof):
wheelhouse (vLLM 0.19.0), `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`,
`--enable-prefix-caching`, `--generation-config vllm`, preserve_thinking, max-model-len
65536, ANALYZER_CONTEXT_WINDOW 32768, temperature/top_p/top_k, MULTIMODAL_UPSCALE 4,
tensor-parallel 1, the BYOD image sha, machine shape, the competition source, cell 14.

FAILURE POLICY IS INVERTED vs the graft builds (they fall back to VANILLA duck, never 0).
Here a fallback would SILENTLY SERVE QWEN3.6 and produce a number we would read as an engine
result. Any config-, rewrite- or boot-assert failure RAISES. A dead canary is a retry; a
silently-3.6 canary is a poisoned measurement. (Same inversion the A17 72B canary used.)

Usage:  python duck_eval/q38/build_q38_eval.py
NO kernel push. NO submission-queue change. $0 cloud. Build-rail only.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
SRC_META = REPO / "notebooks" / "duckfork" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "q38-eval"
OUT_NB = OUT_DIR / "arc3-q38-engine-eval.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

# ARM SELECTION. The engine arm (effort=medium) is COMPLETE and its result is sealed in the
# prereg section 12 (REFUTE-2x). The low arm changes exactly one thing: the effort value.
# Q38_ARM=low -> canivel/arc3-q38-low-eval, reasoning_effort="low".
import os as _os

_ARM = _os.environ.get("Q38_ARM", "medium").strip().lower()
if _ARM not in ("medium", "low"):
    raise SystemExit(f"BUILD FAIL: Q38_ARM must be 'medium' or 'low', got {_ARM!r}")

if _ARM == "low":
    KERNEL_ID = "canivel/arc3-q38-low-eval"
    KERNEL_TITLE = "arc3-q38-low-eval"
    OUT_DIR = REPO / "notebooks" / "q38-low-eval"
    OUT_NB = OUT_DIR / "arc3-q38-low-eval.ipynb"
    OUT_META = OUT_DIR / "kernel-metadata.json"
else:
    KERNEL_ID = "canivel/arc3-q38-engine-eval"
    KERNEL_TITLE = "arc3-q38-engine-eval"

OLD_ENGINE_DS = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
NEW_ENGINE_DS = "saltb0x/qwen3-8-27b-fp8"
NEW_ENGINE_OWNER, NEW_ENGINE_SLUG = NEW_ENGINE_DS.split("/", 1)
OLD_SERVED = "vrfai/Qwen3.6-27B-FP8"
NEW_SERVED = "Qwen/Qwen3.8-27B-FP8"
REASONING_EFFORT = _ARM  # "medium" = the engine arm (COMPLETE); "low" = the token-cost arm

SOURCE_DS = "jeroencottaar/taaf-kaggle-source-share"
WHEELS_DS = "driessmit1/arc3-vllm-h100-wheelhouse-v3"

# ---------------------------------------------------------------------------
# cell 2 — identity banner (appended after the fork's own last print)
# ---------------------------------------------------------------------------
CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
CELL2_NEW = f'''print(f"taaf.kaggle: TRUE_SUBMISSION={{TRUE_SUBMISSION}}")
print(
    "Q38-EVAL seed=1 mode=engine-swap-local25 "
    "engine={NEW_ENGINE_DS} (Qwen3.8-27B-FP8, 25.3 GB, blockwise fp8) "
    "REPLACES {OLD_ENGINE_DS} (Qwen3.6-27B-FP8, 35.9 GB) "
    "wheels={WHEELS_DS} (UNCHANGED, vLLM 0.19.0) "
    "reasoning_effort=PINNED-{REASONING_EFFORT} "
    "arm={_ARM} "
    "delta(medium)=THE WEIGHTS AND NOTHING ELSE vs duck-harness-kaggle m=3 lc 18/19/21; "
    "delta(low)=THE EFFORT KNOB AND NOTHING ELSE vs the medium arm (21 levels, 2857 actions) "
    "primary=ACTIONS and LEVELS, never job-wallclock tokens/s",
    flush=True,
)'''

# ---------------------------------------------------------------------------
# cell 6 — the attached datasets are hardcoded here as well as in the metadata
# ---------------------------------------------------------------------------
CELL6_ANCHOR = (
    'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", '
    '"driessmit1/arc3-vllm-h100-wheelhouse-v3", '
    '"driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]'
)
CELL6_NEW = (
    "# Q38-EVAL: the source bundle (index 0) and the wheelhouse are UNCHANGED; the engine\n"
    "# entry is the ONE substitution. Order is preserved so index 0 stays the bundle.\n"
    f'DATASET_SOURCES = ["{SOURCE_DS}", "{WHEELS_DS}", "{NEW_ENGINE_DS}"]'
)

# ---------------------------------------------------------------------------
# cell 8 — the serve-config rewrite (defs injected INTO the setup command text)
# ---------------------------------------------------------------------------
# These run inside the setup command's own `python -` process, so they may reference ONLY
# names defined in that process (MODEL_PATH, VLLM_BASE_URL, request_json, json, os, Path,
# subprocess, ...). `_assert_names_resolve` in q38_smoke.py walks the rewritten command's
# AST and fails the BUILD on any unresolved name — the gate the LoRA v1 NameError needed.
_SERVE_DEFS_TEMPLATE = r'''Q38_EXPECT_SERVED = 'Qwen/Qwen3.8-27B-FP8'
Q38_EFFORT = '__Q38_EFFORT__'
# What the ATTACHED template injects for each effort value (measured, q38_smoke.py s5):
#   medium -> NOTHING (validated-but-silent)   low/xhigh -> an instruction sentence
# So certifying the pin means asserting the prompt matches the REQUESTED value's
# signature, NOT asserting silence. v1 of the low arm died because this constant was
# stale ('medium') and the server probe treated the arm's own intended instruction as
# a failure to bind. The kernel died BECAUSE the pin worked.
Q38_EFFORT_MARKERS = {'medium': None, 'low': 'Reasoning effort is set to low',
                      'xhigh': 'Reasoning effort is set to xhigh'}
Q38_EXPECT_MARKER = Q38_EFFORT_MARKERS[Q38_EFFORT]
Q38_REQUIRED_FILES = (
    'config.json', 'chat_template.jinja', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'preprocessor_config.json', 'model.safetensors.index.json',
)
Q38_PIN_CERTIFIED = []


def _q38_probe_messages():
    # The shape the harness actually sends: a system message plus a tool list.
    messages = [{'role': 'system', 'content': 'Q38_SYSTEM_SENTINEL'},
                {'role': 'user', 'content': 'hi'}]
    tools = [{'type': 'function', 'function': {
        'name': 'submit_action', 'description': 'Submit the next ARC action.',
        'parameters': {'type': 'object',
                       'properties': {'action': {'type': 'string'}, 'x': {'type': 'integer'},
                                      'y': {'type': 'integer'}},
                       'required': ['action']}}}]
    return messages, tools


def _q38_render_local(effort):
    # Effort-pin instrument #1: render the ATTACHED template ourselves. No server, no
    # transport, so this cannot fail for infrastructure reasons.
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        keep_trailing_newline=True)
    env.globals['raise_exception'] = _raise
    env.filters['tojson'] = lambda obj, **kw: json.dumps(obj, ensure_ascii=False)
    template = env.from_string((MODEL_PATH / 'chat_template.jinja').read_text(encoding='utf-8'))
    messages, tools = _q38_probe_messages()
    kwargs = dict(messages=messages, tools=tools, add_generation_prompt=True,
                  preserve_thinking=True, enable_thinking=True, add_vision_id=False)
    if effort is not None:
        kwargs['reasoning_effort'] = effort
    return template.render(**kwargs)


def _q38_png_b64(size: int = 64) -> str:
    # Dependency-free solid-colour PNG for the boot MM probe.
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


def _q38_pre_serve_asserts() -> None:
    # Runs BEFORE the 25 GB load. The 122B lane loaded 78 GB and only THEN found its
    # snapshot was missing tokenizer_config/processor_config; every cheap check is here.
    print('Q38-EVAL model_path=' + str(MODEL_PATH), flush=True)
    missing = [name for name in Q38_REQUIRED_FILES if not (MODEL_PATH / name).is_file()]
    if missing:
        raise RuntimeError('Q38-EVAL FATAL: engine snapshot is INCOMPLETE, missing '
                           + ', '.join(missing) + ' under ' + str(MODEL_PATH)
                           + ' - refusing to start a load that would die later')
    shards = sorted(MODEL_PATH.glob('layers-*.safetensors'))
    cfg = json.loads((MODEL_PATH / 'config.json').read_text(encoding='utf-8'))
    quant = cfg.get('quantization_config') or {}
    text = cfg.get('text_config') or {}
    facts = {
        'architectures': cfg.get('architectures'),
        'model_type': cfg.get('model_type'),
        'transformers_version': cfg.get('transformers_version'),
        'image_token_id': cfg.get('image_token_id'),
        'quant_method': quant.get('quant_method'),
        'weight_block_size': quant.get('weight_block_size'),
        'activation_scheme': quant.get('activation_scheme'),
        'vocab_size': text.get('vocab_size'),
        'num_hidden_layers': text.get('num_hidden_layers'),
        'hidden_size': text.get('hidden_size'),
        'num_key_value_heads': text.get('num_key_value_heads'),
        'head_dim': text.get('head_dim'),
        'full_attention_interval': text.get('full_attention_interval'),
        'layer_shards': len(shards),
    }
    print('Q38-EVAL engine-config ' + json.dumps(facts, sort_keys=True), flush=True)
    # quant_method is the one field that CANNOT be true of Qwen3.6 (compressed-tensors W8A8
    # per-tensor). If this passes, the attached weights are not the incumbent.
    if quant.get('quant_method') != 'fp8' or quant.get('weight_block_size') != [128, 128]:
        raise RuntimeError('Q38-EVAL FATAL: quantization_config is not Qwen3.8 blockwise fp8: '
                           + json.dumps(quant, sort_keys=True)[:400]
                           + ' - a silent Qwen3.6 run would poison the whole screen')
    if cfg.get('architectures') != ['Qwen3_5ForConditionalGeneration']:
        raise RuntimeError('Q38-EVAL FATAL: unexpected architectures '
                           + repr(cfg.get('architectures')))
    if not str(cfg.get('transformers_version', '')).startswith('5.8'):
        raise RuntimeError('Q38-EVAL FATAL: transformers_version '
                           + repr(cfg.get('transformers_version'))
                           + ' is not the 5.8.x that wrote the Qwen3.8 release configs')
    for key, want in (('vocab_size', 248320), ('num_hidden_layers', 64), ('hidden_size', 5120),
                      ('num_key_value_heads', 4), ('head_dim', 256),
                      ('full_attention_interval', 4)):
        if text.get(key) != want:
            raise RuntimeError('Q38-EVAL FATAL: text_config.' + key + ' = ' + repr(text.get(key))
                               + ' != ' + repr(want) + ' - NOT a drop-in for the incumbent')
    if cfg.get('image_token_id') != 248056:
        raise RuntimeError('Q38-EVAL FATAL: image_token_id drifted; the vision path would break')
    if len(shards) != 64:
        raise RuntimeError('Q38-EVAL FATAL: expected 64 layer shards, found ' + str(len(shards)))

    template_src = (MODEL_PATH / 'chat_template.jinja').read_text(encoding='utf-8')
    if 'reasoning_effort' not in template_src:
        raise RuntimeError('Q38-EVAL FATAL: the attached chat template has no reasoning_effort '
                           'knob - our pin would be a SILENT NO-OP and the arm would run at an '
                           'unknown effort')
    try:
        pinned = _q38_render_local(Q38_EFFORT)
        control = _q38_render_local(None)      # positive control: the xhigh default
    except Exception as exc:
        print('Q38-EVAL WARN effort-pin local-render UNAVAILABLE (' + repr(exc)[:200]
              + ') - falling back to the server probe at boot', flush=True)
    else:
        # ARM-AWARE certification: assert the REQUESTED value's signature, never silence.
        expect = Q38_EXPECT_MARKER
        ok_pin = (('easoning effort' not in pinned) if expect is None
                  else (expect in pinned))
        wrong = [m for e, m in Q38_EFFORT_MARKERS.items()
                 if m and e != Q38_EFFORT and m in pinned]
        if 'easoning effort' not in control:
            print('Q38-EVAL WARN effort-pin local probe is BLIND (the xhigh default injected '
                  'nothing), so it proves nothing - auditing the instrument, not trusting it',
                  flush=True)
        elif not ok_pin or wrong:
            raise RuntimeError('Q38-EVAL FATAL: local render does not match effort='
                               + Q38_EFFORT + ' (expected marker ' + repr(expect)
                               + ', wrong-arm markers ' + repr(wrong)
                               + ') - the prompt does not carry the requested effort')
        elif 'Q38_SYSTEM_SENTINEL' not in pinned:
            raise RuntimeError('Q38-EVAL FATAL: the system message did not render')
        else:
            Q38_PIN_CERTIFIED.append('local-render')
            print('Q38-EVAL effort-pin=' + Q38_EFFORT + ' local-render marker='
                  + ('ABSENT-as-expected' if expect is None else 'PRESENT-as-expected')
                  + ' control(default=xhigh)=PRESENT pinned_chars=' + str(len(pinned))
                  + ' control_chars=' + str(len(control)), flush=True)

    if shutil.which('nvidia-smi'):
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
        print('Q38-EVAL gpu=' + repr(gpu), flush=True)


def _q38_observe(tag, response, extra=''):
    # OBSERVATION BEFORE VERDICT. v1's MM assert printed the conclusion "the vision path is
    # broken" and none of the evidence, so the question could not be closed from the log
    # afterwards. Every probe now emits its raw observation first, pass or fail.
    try:
        message = response['choices'][0]['message']
        finish = response['choices'][0].get('finish_reason')
    except Exception:
        print('Q38-EVAL OBSERVE ' + tag + ' UNPARSEABLE body=' + json.dumps(response)[:400],
              flush=True)
        return '', ''
    content = (message.get('content') or '')
    reasoning = (message.get('reasoning_content') or message.get('reasoning') or '')
    calls = message.get('tool_calls') or []
    usage = response.get('usage') or {}
    print('Q38-EVAL OBSERVE ' + tag
          + ' finish_reason=' + repr(finish)
          + ' content_chars=' + str(len(content))
          + ' reasoning_chars=' + str(len(reasoning))
          + ' tool_calls=' + str(len(calls))
          + ' completion_tokens=' + str(usage.get('completion_tokens'))
          + (' ' + extra if extra else '')
          + ' content_head=' + repr(content[:120])
          + ' reasoning_head=' + repr(reasoning[:120]), flush=True)
    return content.strip(), reasoning


def _q38_decode_rate() -> None:
    # THE DECODE-RATE PROBE (prereg section 16). Three consecutive model-level lanes ended with
    # no clean tokens/s because `generated tokens/sec (job wallclock)` is total tokens over a
    # fixed job duration - it cannot separate a fast engine from a verbose one. vLLM 0.19.0
    # supports ignore_eos + min_tokens, so pinning both to max_tokens makes every request emit
    # EXACTLY that many tokens and the rate becomes arithmetic. Runs BEFORE the bench, so it
    # costs kernel time and ZERO measurement window. REPORT-ONLY: a slow engine IS the number.
    import threading
    import time

    ntok = 256   # kept in sync with the payload literals below; asserted at run time
    prompt = ('Write a detailed technical description of a sorting algorithm. '
              'Continue until you are told to stop.')

    def one(out, idx):
        payload = {'model': Q38_EXPECT_SERVED,
                   'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0.0, 'max_tokens': 256, 'min_tokens': 256,
                   'ignore_eos': True,
                   'chat_template_kwargs': {'enable_thinking': False}}
        try:
            r = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=600)
            out[idx] = ((r.get('usage') or {}).get('completion_tokens') or 0)
        except Exception as exc:
            out[idx] = -1
            print('Q38-EVAL WARN decode-probe request failed: ' + repr(exc)[:160], flush=True)

    if ntok != 256:
        raise RuntimeError('Q38-EVAL decode probe: ntok is out of sync with the payload')
    for concurrency in (1, 8):
        out = [0] * concurrency
        threads = [threading.Thread(target=one, args=(out, i)) for i in range(concurrency)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0
        got = sum(v for v in out if v > 0)
        exact = all(v == ntok for v in out)
        if elapsed <= 0 or got <= 0:
            print('Q38-EVAL WARN decode-probe concurrency=%d produced no usable timing'
                  % concurrency, flush=True)
            continue
        print('Q38-EVAL DECODE concurrency=%d requested_tokens=%d generated_tokens=%d '
              'exact_token_count=%s elapsed_s=%.2f tok_s=%.1f'
              % (concurrency, ntok * concurrency, got, exact, elapsed, got / elapsed),
              flush=True)
    print('Q38-EVAL DECODE note=synthetic fixed-concurrency rate; NOT comparable to the '
          'job-wallclock tokens/sec in summary.txt, and there is no Qwen3.6 point in this '
          'series', flush=True)


def _q38_boot_asserts() -> None:
    # Runs AFTER the server is up and the stock smoke test passed.
    #
    # CLASSIFICATION (prereg section 10, written before this code): a gate is FATAL only if its
    # failure would make the number MEAN something other than what we would read it as. If the
    # failure would simply BE the number, it is REPORT-ONLY. Two gates were reclassified after
    # v1: the tool-call and MM probes. A broken tool-call path or a dead vision path produces a
    # genuine low score for Qwen3.8-in-this-harness - that is a true measurement, not a
    # poisoned one, and killing the kernel for it destroys the very number we came for.
    models = request_json(VLLM_BASE_URL + '/models', timeout=60)
    ids = sorted(entry.get('id', '') for entry in models.get('data', []))
    if ids != [Q38_EXPECT_SERVED]:
        raise RuntimeError('Q38-EVAL FATAL: served model ids ' + repr(ids) + ' != '
                           + repr([Q38_EXPECT_SERVED]) + ' - refusing to continue')
    print('Q38-EVAL served=' + Q38_EXPECT_SERVED, flush=True)

    # Effort-pin instrument #2: ask the SERVER what it renders. /tokenize is mounted at the
    # ROOT, not under /v1 - v1 of this kernel appended it to VLLM_BASE_URL and got a 404. The
    # endpoint existed; my URL was wrong. Same class as the payload bug: existence verified,
    # call shape not. Transport failure stays non-fatal; a positive detection is fatal.
    root_url = VLLM_BASE_URL[:-3] if VLLM_BASE_URL.endswith('/v1') else VLLM_BASE_URL
    root_url = root_url.rstrip('/')
    messages, tools = _q38_probe_messages()

    def _rendered(extra):
        kwargs = {'enable_thinking': True}
        kwargs.update(extra)
        payload = {'model': Q38_EXPECT_SERVED, 'messages': messages, 'tools': tools,
                   'add_generation_prompt': True, 'return_token_strs': True,
                   'chat_template_kwargs': kwargs}
        response = request_json(root_url + '/tokenize', payload=payload, timeout=120)
        joined = ''.join(response.get('token_strs') or [])
        return joined.replace('\u0120', ' ').replace('\u010a', '\n')

    try:
        served_prompt = _rendered({})
        control_prompt = _rendered({'reasoning_effort': 'xhigh'})
    except Exception as exc:
        served_prompt = ''
        print('Q38-EVAL WARN effort-pin server-probe UNAVAILABLE at ' + root_url
              + '/tokenize (' + repr(exc)[:200] + ')', flush=True)
    else:
        print('Q38-EVAL OBSERVE tokenize served_chars=' + str(len(served_prompt))
              + ' control_chars=' + str(len(control_prompt))
              + ' served_head=' + repr(served_prompt[:100]), flush=True)
        expect = Q38_EXPECT_MARKER
        ok_pin = (('easoning effort' not in served_prompt) if expect is None
                  else (expect in served_prompt))
        wrong = [m for e, m in Q38_EFFORT_MARKERS.items()
                 if m and e != Q38_EFFORT and m in served_prompt]
        if 'easoning effort' not in control_prompt:
            print('Q38-EVAL WARN effort-pin server-probe is BLIND (xhigh control produced no '
                  'instruction) - not treating it as evidence', flush=True)
        elif not ok_pin or wrong:
            raise RuntimeError('Q38-EVAL FATAL: the SERVED prompt does not match effort='
                               + Q38_EFFORT + ' (expected marker ' + repr(expect)
                               + ', wrong-arm markers ' + repr(wrong)
                               + ') - --default-chat-template-kwargs did not bind to the '
                               'requested value')
        else:
            Q38_PIN_CERTIFIED.append('server-tokenize')
            print('Q38-EVAL effort-pin=' + Q38_EFFORT + ' server-probe marker='
                  + ('ABSENT-as-expected' if expect is None else 'PRESENT-as-expected')
                  + ' control_xhigh=PRESENT served_prompt_chars=' + str(len(served_prompt)),
                  flush=True)
        if 'Q38_SYSTEM_SENTINEL' in served_prompt and '<think>' in served_prompt:
            print('Q38-EVAL preserve_thinking=BOUND (think block present in the served prompt)',
                  flush=True)

    if not Q38_PIN_CERTIFIED:
        raise RuntimeError('Q38-EVAL FATAL: the reasoning_effort pin is UNCERTIFIED by BOTH '
                           'independent instruments. The arm would run at an unknown effort and '
                           'the result would not isolate the weights. Dying at boot costs '
                           'minutes; an uninterpretable 2h run costs the slot.')
    print('Q38-EVAL effort-pin-certified-by=' + ','.join(Q38_PIN_CERTIFIED), flush=True)

    # --- REPORT-ONLY from here down (prereg section 10, gates N/O/P) ---------------------
    # Every payload below is HARNESS-SHAPED: chat_template_kwargs always present and
    # max_tokens >= 256. v1 died because the MM probe omitted chat_template_kwargs and used
    # max_tokens=32, so thinking stayed on, all 32 tokens went to reasoning_content, and
    # `content` was empty - the MODAL behaviour of this stack, asserted on as if it were a
    # fault. A static lint in q38_smoke.py now enforces both properties at build time.
    forced = {'model': Q38_EXPECT_SERVED,
              'messages': [{'role': 'user',
                            'content': 'Call the submit_action tool with action ACTION6, '
                                       'x 3, y 7.'}],
              'tools': tools,
              'tool_choice': {'type': 'function', 'function': {'name': 'submit_action'}},
              'temperature': 0.0, 'max_tokens': 512,
              'chat_template_kwargs': {'enable_thinking': False}}
    try:
        response = request_json(VLLM_BASE_URL + '/chat/completions', payload=forced, timeout=300)
        _q38_observe('tool-call-forced', response)
        calls = response['choices'][0]['message'].get('tool_calls') or []
    except Exception as exc:
        calls = []
        print('Q38-EVAL WARN forced tool-call probe errored: ' + repr(exc)[:300], flush=True)
    if calls and calls[0].get('function', {}).get('name') == 'submit_action':
        args = json.loads(calls[0]['function'].get('arguments') or '{}')
        print('Q38-EVAL tool-call-roundtrip=OK mode=forced args='
              + json.dumps(args, sort_keys=True), flush=True)
    else:
        print('Q38-EVAL WARN tool-call-roundtrip=FAILED mode=forced - REPORT-ONLY: a broken '
              'tool-call path yields a genuine low score, which IS the measurement. Watch the '
              'bench for a zero-action signature.', flush=True)

    auto = dict(forced)
    auto['tool_choice'] = 'auto'
    auto['chat_template_kwargs'] = {'enable_thinking': True}
    auto['messages'] = [{'role': 'user', 'content': 'Use the submit_action tool to submit '
                                                    'ACTION6 at x 3, y 7. Reply with the tool '
                                                    'call only.'}]
    try:
        response = request_json(VLLM_BASE_URL + '/chat/completions', payload=auto, timeout=300)
        _q38_observe('tool-call-auto', response)
        calls = response['choices'][0]['message'].get('tool_calls') or []
    except Exception as exc:
        calls = []
        print('Q38-EVAL WARN auto tool-call probe errored: ' + repr(exc)[:300], flush=True)
    if calls:
        print('Q38-EVAL tool-call-roundtrip=OK mode=auto parser=qwen3_coder name='
              + str(calls[0].get('function', {}).get('name')), flush=True)
    else:
        print('Q38-EVAL WARN auto tool-call probe produced NO parsed call under qwen3_coder - '
              'REPORT-ONLY', flush=True)

    # One real image through the vision tower. REPORT-ONLY (gate P): this is the assert that
    # killed v1. Thinking is OFF and the budget is 512 so that an empty `content` means the
    # vision path really is dead rather than that the model is still thinking.
    payload = {'model': Q38_EXPECT_SERVED,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url',
                    'image_url': {'url': 'data:image/png;base64,' + _q38_png_b64()}},
                   {'type': 'text', 'text': 'Answer with one word: what colour is this image?'}]}],
               'temperature': 0.0, 'max_tokens': 512,
               'chat_template_kwargs': {'enable_thinking': False}}
    try:
        response = request_json(VLLM_BASE_URL + '/chat/completions', payload=payload, timeout=300)
        content, reasoning = _q38_observe('mm-image', response)
    except Exception as exc:
        content, reasoning = '', ''
        print('Q38-EVAL WARN MM probe errored (HTTP-level, i.e. the request was REJECTED rather '
              'than answered - that is the signature of a genuinely broken vision path): '
              + repr(exc)[:300], flush=True)
    if content:
        print('Q38-EVAL mm-image-roundtrip=OK reply=' + repr(content[:60]), flush=True)
    else:
        print('Q38-EVAL WARN mm-image-roundtrip=EMPTY-CONTENT reasoning_chars='
              + str(len(reasoning)) + ' - REPORT-ONLY: a dead vision path would produce a low '
              'score, which IS the number. Read this against the per-game traces, not as a '
              'reason to kill the run.', flush=True)

    try:
        _q38_decode_rate()
    except Exception as exc:
        print('Q38-EVAL WARN decode-rate probe errored (REPORT-ONLY): '
              + repr(exc)[:200], flush=True)

    print('Q38-EVAL BOOT-ASSERTS PASSED - handing off to the 25-game offline bench', flush=True)'''

Q38_SERVE_DEFS = _SERVE_DEFS_TEMPLATE.replace('__Q38_EFFORT__', REASONING_EFFORT)

# (old, new) — every anchor must match EXACTLY once against the pristine setup command.
Q38_SETUP_REWRITES: list[tuple[str, str]] = [
    ("MODEL_OWNER = 'driessmit1'", f"MODEL_OWNER = '{NEW_ENGINE_OWNER}'"),
    ("MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'", f"MODEL_SLUG = '{NEW_ENGINE_SLUG}'"),
    (f"SERVED_MODEL_NAME = '{OLD_SERVED}'", f"SERVED_MODEL_NAME = '{NEW_SERVED}'"),
    ('\'{"preserve_thinking": true}\',',
     '\'{"preserve_thinking": true, "reasoning_effort": "' + REASONING_EFFORT + '"}\','),
    ("\n\ndef run_vllm_api_smoke_test() -> None:",
     "\n\n" + Q38_SERVE_DEFS + "\n\n\ndef run_vllm_api_smoke_test() -> None:"),
    ("\nstart_vllm_server()\nrun_vllm_api_smoke_test()\n",
     "\n_q38_pre_serve_asserts()\nstart_vllm_server()\nrun_vllm_api_smoke_test()\n"
     "_q38_boot_asserts()\n"),
]

# Strings that MUST be gone after the rewrite (a survivor means a silent-3.6 route). Only the
# two identifiers that actually ROUTE are vetoed; the prose "Qwen3.6" appears legitimately in
# the injected error messages and vetoing it would fire on our own diagnostics.
Q38_VETO = ("vrfai-qwen3-6-27b-fp8-hf-snapshot", OLD_SERVED)
# Strings the rewrite MUST have introduced.
Q38_REQUIRE = (
    NEW_ENGINE_OWNER, NEW_ENGINE_SLUG, NEW_SERVED,
    '"reasoning_effort": "' + REASONING_EFFORT + '"',
    "_q38_pre_serve_asserts()", "_q38_boot_asserts()", "Q38-EVAL",
)
# Strings that MUST SURVIVE untouched — the one-variable proof, mechanised.
Q38_INVARIANTS = (
    "WHEELHOUSE_OWNER = 'driessmit1'",
    "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'",
    "'--tool-call-parser',\n        'qwen3_coder',",
    "'--reasoning-parser',\n        'qwen3',",
    "'--enable-prefix-caching',",
    "'--generation-config',\n        'vllm',",
    '"preserve_thinking": true',
    "VLLM_MAX_MODEL_LEN = 65536",
    "ANALYZER_CONTEXT_WINDOW = 32768",
    "VLLM_TENSOR_PARALLEL_SIZE = 1",
    "'LOCAL_ANALYZER_TEMPERATURE': '0.6'",
    "'LOCAL_ANALYZER_TOP_P': '0.95'",
    "'LOCAL_ANALYZER_TOP_K': '20'",
    "'LOCAL_ANALYZER_ENABLE_THINKING': 'true'",
    "'MULTIMODAL_CONTEXT': 'current_grid'",
    "'MULTIMODAL_UPSCALE': '4'",
    "'LOCAL_ANALYZER_YIELD_SECONDS': '60'",
    "STAMP_TEXT = 'vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\\n'",
)

CELL8_ANCHOR = (
    'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):\n'
)
CELL8_NEW = (
    "# --- Q38-EVAL BEGIN serve-config rewrite (FAIL-LOUD; no Qwen3.6 fallback) ---\n"
    "# Anchored, counted, vetoed and invariant-checked. Any miss RAISES: a fallback here would\n"
    "# silently serve the incumbent and produce a number we would read as an engine result.\n"
    "# Failure policy is inverted vs the graft builds on purpose (same call the A17 canary made).\n"
    f"Q38_SETUP_REWRITES = {Q38_SETUP_REWRITES!r}\n"
    f"Q38_VETO = {Q38_VETO!r}\n"
    f"Q38_REQUIRE = {Q38_REQUIRE!r}\n"
    f"Q38_INVARIANTS = {Q38_INVARIANTS!r}\n"
    "\n"
    "\n"
    "def _q38_patch_setup_commands(commands):\n"
    "    if not isinstance(commands, list) or len(commands) != 1:\n"
    "        raise RuntimeError('Q38-EVAL FATAL: expected exactly 1 setup command, got '\n"
    "                           + repr(commands)[:200])\n"
    "    command = commands[0]\n"
    "    for invariant in Q38_INVARIANTS:\n"
    "        if command.count(invariant) < 1:\n"
    "            raise RuntimeError('Q38-EVAL FATAL: pristine setup command is missing the '\n"
    "                               'invariant %r - the bundle drifted and this arm is no longer '\n"
    "                               'a one-variable change' % (invariant[:80],))\n"
    "    for old, new in Q38_SETUP_REWRITES:\n"
    "        found = command.count(old)\n"
    "        if found != 1:\n"
    "            raise RuntimeError('Q38-EVAL FATAL: serve-config anchor matched %d times '\n"
    "                               '(want 1): %r' % (found, old[:100]))\n"
    "        command = command.replace(old, new)\n"
    "    for veto in Q38_VETO:\n"
    "        if veto in command:\n"
    "            raise RuntimeError('Q38-EVAL FATAL: incumbent serve artifact %r survived the '\n"
    "                               'rewrite' % (veto,))\n"
    "    for need in Q38_REQUIRE:\n"
    "        if need not in command:\n"
    "            raise RuntimeError('Q38-EVAL FATAL: required Qwen3.8 serve token %r missing '\n"
    "                               'after rewrite' % (need,))\n"
    "    for invariant in Q38_INVARIANTS:\n"
    "        if invariant not in command:\n"
    "            raise RuntimeError('Q38-EVAL FATAL: the rewrite destroyed the invariant %r - '\n"
    "                               'more than one variable changed' % (invariant[:80],))\n"
    "    print('Q38-EVAL setup-commands rewrite OK (%d anchors replaced, %d invariants held; '\n"
    "          'loud-fail mode, no incumbent fallback)'\n"
    "          % (len(Q38_SETUP_REWRITES), len(Q38_INVARIANTS)), flush=True)\n"
    "    return [command]\n"
    "\n"
    "\n"
    "# --- Q38-EVAL END serve-config rewrite ---\n"
    'for command in _q38_patch_setup_commands(\n'
    '        json.loads((BUNDLE_DIR / "setup_commands.json").read_text())):\n'
)


def _cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"BUILD FAIL: anchor in {where} matched {count} times (want 1): {old[:90]!r}")
    return text.replace(old, new)


def build() -> tuple[Path, Path]:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if len(cells) != 17:
        raise SystemExit(f"BUILD FAIL: frozen fork has {len(cells)} cells, expected 17 (drift)")
    if "kaggle" in (nb.get("metadata") or {}):
        raise SystemExit("BUILD FAIL: frozen fork gained a metadata.kaggle block (preflight D2)")

    src2 = _cell_source(cells[2])
    _set_source(cells[2], _replace_once(src2, CELL2_ANCHOR, CELL2_NEW, "cell 2"))

    src6 = _cell_source(cells[6])
    _set_source(cells[6], _replace_once(src6, CELL6_ANCHOR, CELL6_NEW, "cell 6"))

    src8 = _cell_source(cells[8])
    _set_source(cells[8], _replace_once(src8, CELL8_ANCHOR, CELL8_NEW, "cell 8"))

    # Every other cell must be byte-identical to the frozen fork.
    pristine = json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"]
    changed = [i for i, (a, b) in enumerate(zip(pristine, cells))
               if _cell_source(a) != _cell_source(b)]
    if changed != [2, 6, 8]:
        raise SystemExit(f"BUILD FAIL: differing cells {changed}, expected [2, 6, 8]")

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_TITLE
    meta["code_file"] = OUT_NB.name
    sources = list(meta["dataset_sources"])
    if OLD_ENGINE_DS not in sources:
        raise SystemExit("BUILD FAIL: frozen-fork metadata no longer attaches the incumbent engine")
    meta["dataset_sources"] = [NEW_ENGINE_DS if s == OLD_ENGINE_DS else s for s in sources]

    # Env fields must be byte-identical to the frozen fork (feedback_kaggle_env_match).
    ref = json.loads(SRC_META.read_text(encoding="utf-8"))
    for key in ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
                "competition_sources", "kernel_sources", "model_sources", "language",
                "kernel_type", "is_private", "keywords"):
        if meta.get(key) != ref.get(key):
            raise SystemExit(f"BUILD FAIL: env field {key} drifted from the frozen fork")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_text = json.dumps(nb, ensure_ascii=False)
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"

    # Idempotence: deterministic-from-pristine-base. Re-running must reproduce byte-for-byte.
    for path, text in ((OUT_NB, nb_text), (OUT_META, meta_text)):
        if path.exists() and path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        elif not path.exists():
            path.write_text(text, encoding="utf-8")

    return OUT_NB, OUT_META


if __name__ == "__main__":
    nb_path, meta_path = build()
    import hashlib

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print(f"built {nb_path}")
    print(f"built {meta_path}")
    print(f"cells={len(nb['cells'])} code_sha256={hashlib.sha256(code.encode()).hexdigest()[:16]}")
    print("differing cells vs frozen fork: [2, 6, 8]")
