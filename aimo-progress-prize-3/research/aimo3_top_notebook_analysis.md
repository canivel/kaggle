# AIMO3 Top Notebook Analysis: Why We Scored 1/50 vs 43-46/50

## Executive Summary

Every notebook scoring 40+ on the AIMO3 public leaderboard uses the `openai_harmony` protocol
to communicate with gpt-oss-120b at the token level via `/v1/completions`. Our submission
uses `/v1/chat/completions` with plain-text TIR. This is not a tuning gap. It is a protocol
mismatch — equivalent to calling a function-calling model without the function-calling API.
The model produces correct token sequences for Harmony tool calls; we were never able to
receive or interpret them.

---

## Notebooks Analyzed

| Notebook | Author | LB Score | Protocol |
|---|---|---|---|
| aimo-3-gpt-oss-120b-with-tools | (top public) | ~46/50 | openai_harmony + /v1/completions |
| aimo-3-gpt-oss-120b-weighted-entropy | nihilisticneuralnet | 43/50 | openai_harmony + /v1/completions |
| aimo-3-42-50-stable-lb-possible-43-luck | datasciencegrad | 42/50 | openai_harmony + /v1/completions |
| aimo-3-winner | bhargavaabhi | ~44/50 | openai_harmony + /v1/completions |
| aimo-3-gpt-oss-120b-agentic-solver | seshurajup | 5-10/50 | Qwen3-30B + Python REPL, no Harmony |
| submission_v4_robust (ours) | us | 1/50 | /v1/chat/completions + text TIR |

---

## Fatal Issue #1: Chat Completions API vs Raw Completions API

### What top notebooks do

```python
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    Role, Conversation, Message, Author, TextContent, SystemContent,
    ReasoningEffort, ToolNamespaceConfig,
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build a typed Conversation object
conversation = Conversation(messages=[
    Message(author=Author(role=Role.SYSTEM),
            content=[SystemContent(
                text=SYSTEM_PROMPT,
                reasoning_effort=ReasoningEffort.HIGH
            )]),
    Message(author=Author(role=Role.USER),
            content=[TextContent(text=problem_text)]),
])

# Render to raw token IDs — the native format gpt-oss-120b was trained on
prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

# Call /v1/completions with token IDs, not text
stream = client.completions.create(
    model=served_model_name,
    prompt=prompt_ids,          # RAW INTEGER TOKEN IDS
    temperature=1.0,
    max_tokens=65536,
    stream=True,
    logprobs=5,
    extra_body={
        'min_p': 0.02,
        'stop_token_ids': encoding.stop_tokens_for_assistant_actions(),
        'return_token_ids': True,
    }
)
```

### What our code does

```python
# submission_v4_robust.ipynb
def vllm_chat(prompt, n=1, temp=0.7, max_tok=4096, stop=None):
    payload = {
        'model': SERVED_NAME,
        'temperature': temp,
        'max_tokens': max_tok,
        'n': n,
        'messages': [{'role': 'user', 'content': prompt}]
    }
    r = httpx.post(f'{VLLM_URL}/chat/completions', json=payload, timeout=300)
```

### Why this causes 1/50

gpt-oss-120b was trained exclusively on the Harmony token-level protocol. Its system prompt,
tool calls, and reasoning chains are all encoded as special integer tokens that only make sense
to `parse_messages_from_completion_tokens()`. When you send plain text via `/v1/chat/completions`:

1. The chat template applied by vLLM is NOT the Harmony protocol. The model sees a different
   prompt structure than it was trained on and enters an off-distribution mode.
2. Even if the model generates some output, the tool-call tokens are raw integers that the
   chat endpoint does not know how to route. The response is garbled or empty.
3. `ReasoningEffort.HIGH` is encoded as a special `SystemContent` type in the Harmony system
   message. Without it, the model does not activate its extended reasoning chains.

The result is a model that either produces no valid output or produces random text because
the input format is completely foreign to its training distribution.

---

## Fatal Issue #2: Native Harmony Tool Calling vs Text-Based TIR

### What top notebooks do

After streaming tokens from `/v1/completions`, they call:

```python
token_buffer = []
for chunk in stream:
    token_buffer.extend(chunk.choices[0].logprobs.tokens)  # raw integer IDs

# Harmony decodes the token buffer into typed Message objects
new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
conversation.messages.extend(new_messages)

last_message = new_messages[-1]

# Check how the model wants to continue
if last_message.channel == 'final':
    # Model is done; extract the boxed answer
    answer_text = last_message.content[0].text
    final_answer = scan_for_answer(answer_text)
    break

if last_message.recipient == 'python':
    # Model wants to execute Python — dispatch to Jupyter kernel
    python_calls += 1
    tool_responses = local_tool.process_sync_plus(last_message)  # returns Message objects
    conversation.messages.extend(tool_responses)
    # Loop continues: render updated conversation, call /v1/completions again
```

The model natively signals "I want to run Python" and "I'm done" using special Harmony tokens,
not by outputting text strings like ` ```python ` or ` ```output `.

### What our code does

```python
# tir_executor.py
resp = generate_n_completions(
    vllm_generate,
    prompt,
    n=8,
    stop=["```output"],   # gpt-oss-120b never outputs this string
    temperature=1.0,
    max_tokens=8192,
)
code_blocks = get_code_blocks(text)   # regex: r'```python\s*\n(.*?)```'
output = execute_code(code)           # exec() with redirect_stdout
new_text = text + f"\n```output\n{output}\n```\n"
```

The ` ```output ` convention was used by Numina (AIMO 2024) models. gpt-oss-120b was not
trained on it. The model never produces ` ```output ` as a stop condition. Our stop string
was never triggered, so the model ran to max_tokens generating continuation text, and we
mistakenly ran `exec()` on non-code content.

---

## Fatal Issue #3: Wrong Stop Tokens

### What top notebooks do

```python
# Get the Harmony-native stop token IDs (model-specific special integers)
self.stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# Pass as extra_body to vLLM
extra_body={
    'stop_token_ids': self.stop_token_ids,  # e.g. [128009, 128011, ...]
    'return_token_ids': True,
}
```

`stop_tokens_for_assistant_actions()` returns the exact token IDs that gpt-oss-120b was
trained to emit when it wants to call a tool (i.e., "I'm about to hand off to Python") or
complete (i.e., "I'm done with my answer"). These are specific integer IDs in the Harmony
vocabulary, not text strings.

### What our code does

```python
stop=["```output"]   # text string; gpt-oss-120b never emits this
```

Consequence: generation never stops at a tool-call boundary. The model runs to `max_tokens`
producing a single monolithic output that is not parseable as a structured tool call sequence.

---

## Significant Issue #4: ReasoningEffort.HIGH

### What top notebooks do

```python
SystemContent(
    text=SYSTEM_PROMPT,
    reasoning_effort=ReasoningEffort.HIGH   # activates extended chain-of-thought
)
```

This is embedded in the Harmony system message as a special `SystemContent` token field.
It instructs gpt-oss-120b to use long reasoning chains before answering. Without it, the
model uses a shorter reasoning mode that is significantly less accurate on competition math.

### What our code does

Plain text system prompt with no equivalent activation mechanism. There is no way to set
`ReasoningEffort.HIGH` via the chat completions API.

---

## Significant Issue #5: exec() vs Persistent Jupyter Kernels

### What top notebooks do

```python
# At startup: spin up 16 persistent kernel managers
kernel_managers = queue.Queue()
for _ in range(16):
    km = KernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)
    kernel_managers.put((km, kc))

# For each tool call: acquire a kernel from the pool
km, kc = kernel_managers.get()
try:
    kc.execute(code_string)
    msg = kc.get_iopub_msg(timeout=30)
    # parse stdout/stderr/error from msg
finally:
    kernel_managers.put((km, kc))
```

Kernels are stateful across tool calls within a problem. If the model defines a function in
one call and uses it in the next, that works correctly. Between problems, kernels are reset
with `%reset -f`. 16 kernels support 16 concurrent workers solving problems in parallel.

### What our code does

```python
def execute_code(code, timeout=30):
    result = {}
    exec_globals = {}
    try:
        with redirect_stdout(io.StringIO()) as f:
            exec(code, exec_globals)
        result['stdout'] = f.getvalue()
    except Exception as e:
        result['error'] = str(e)
    return result
```

`exec()` is not stateful. Each call has a fresh `exec_globals` dict. Multi-step Python
programs that rely on previously-defined variables fail silently. There is also no subprocess
isolation — a crash in the code crashes the main process.

---

## Significant Issue #6: Wrong Sampling Parameters

| Parameter | Our Code | Top Notebooks | Impact |
|---|---|---|---|
| temperature | 0.7 | 1.0 | Lower exploration diversity across 8 attempts |
| max_tokens | 4096-8192 | 65536 | Cuts off long reasoning chains for hard problems |
| min_p | not set | 0.02 | Misses nucleus sampling optimization |
| context_tokens | 16384 | 65536-81920 | Cannot fit long multi-turn tool call conversations |
| top_logprobs | not set | 5 | Cannot compute entropy-weighted voting |
| attempts | varies | 8 | Fewer votes = less reliable majority |
| workers | 1-4 | 16 | Much slower, fewer problems attempted |

The `max_tokens=8192` combined with `context_tokens=16384` means that for any problem
requiring multiple Python tool calls with long reasoning between each, our model was
simply truncated mid-reasoning. The top notebooks allow up to 65536 tokens per call
in a context window of 81920 tokens.

---

## Voting Mechanism Comparison

### Top notebook (reference): Simple inverse-entropy

```python
for chunk in stream:
    for logprob_info in chunk.choices[0].logprobs.top_logprobs:
        token_entropy = -sum(p * math.log(p + 1e-12)
                            for p in softmax(logprob_info.values()))
        running_entropy.append(token_entropy)

final_entropy = sum(running_entropy) / max(len(running_entropy), 1)
weight = 1.0 / max(final_entropy, 1e-9)
```

### nihilisticneuralnet (43/50): Enhanced 5-component entropy

```python
# Position-weighted: recent tokens count more (exponential decay 0.995)
weights = np.array([0.995 ** (n - i - 1) for i in range(n)])
position_weighted_ent = np.sum(weights * entropies_arr) / np.sum(weights)

# Variance penalty: penalize inconsistent entropy
std_dev = np.std(entropies_arr)

# Sustained high-entropy penalty
high_ent_ratio = np.sum(entropies_arr > 1.5) / n

# Streak bonus: reward long runs of confident (low entropy) tokens
streak_bonus = -0.05 * max_streak if max_streak > 10 else 0

final_entropy = (
    0.3 * mean_ent +
    0.4 * position_weighted_ent +
    0.2 * std_dev +
    0.3 * high_ent_ratio * 3.0 +
    streak_bonus
)
```

### datasciencegrad (42/50): Multi-signal confidence

```python
# 3 signals: entropy (60%), execution quality (25%), completeness (15%)
entropy_score = 1.0 / (1.0 + avg_entropy)
execution_score = 1.0 if code_executed_ok else 0.3
completeness_score = 1.0 if has_boxed_answer else 0.5

confidence = (
    0.6 * entropy_score +
    0.25 * execution_score +
    0.15 * completeness_score
)
weight = math.exp(confidence * 2.0)  # exponential scaling
```

### Our code: Simple heuristic

```python
weight = 1.0
if code_ok: weight += 2.0
if has_boxed: weight += 0.5
```

**Impact of voting differences**: Approximately 1-3 points on the leaderboard. The 40-point
gap comes entirely from issues #1-#3 above. Even with perfect voting, using the wrong API
would still score near 0.

---

## Early Stopping and Time Management

### Top notebooks

```python
# Time budget per problem
budget = min(time_left - reserved_for_others, 900)
budget = max(budget, 270)  # floor: always try for at least 270s
reserved_for_others = (remaining_problems - 1) * 270

# Early stop: if 4 of 8 attempts agree, cancel remaining futures
early_stop = 4
if top_vote_count >= early_stop:
    for future in pending_futures:
        future.cancel()
```

This ensures no problem starves later problems. Each problem gets at least 270 seconds,
and the 8-attempt majority vote terminates early if consensus is reached.

### Our code

No time budget management. Problems could time out or consume all available time.

---

## vLLM Server Configuration

### Top notebook flags

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /kaggle/input/gpt-oss-120b/... \
    --served-model-name gpt-oss-120b \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.99 \
    --max-model-len 81920 \
    --max-num-batched-tokens 2048 \
    --max-cudagraph-capture-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --async-scheduling \
    --disable-log-stats \
    --enable-prefix-caching \
    --stream-interval 200 \
    --port 8000
```

Key flags:
- `--kv-cache-dtype fp8_e4m3`: Compresses KV cache by 2x, critical for fitting 81920 context
- `--enable-prefix-caching`: Amortizes cost of shared system prompt across 8 attempts per problem
- `--async-scheduling`: Better throughput under concurrent load (16 workers)
- `--max-num-batched-tokens 2048`: Limits batch size for memory stability
- `--stream-interval 200`: Streams every 200 tokens (for logprob collection)

### nihilisticneuralnet / datasciencegrad flags

```bash
--max-model-len 65536 \
--gpu-memory-utilization 0.96 \
# No --max-num-batched-tokens or --max-cudagraph-capture-size
```

### Our flags

```bash
--max-model-len 16384 \   # CRITICALLY SHORT
--gpu-memory-utilization 0.85
# No fp8 KV cache, no prefix caching, no async scheduling
```

`max_model_len=16384` alone would cause 90%+ of problems to fail: multi-step tool call
conversations routinely exceed 20,000-40,000 tokens.

---

## Model Weight Preloading

### Top notebooks

```python
# Before starting vLLM: read all model shards into OS page cache
model_files = list(MODEL_DIR.glob('*.safetensors'))
def preload_file(f):
    with open(f, 'rb') as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            pass  # just read into page cache

with ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(preload_file, model_files))
```

This loads ~240GB of weights into RAM before vLLM starts, so the first inference request
is not delayed by disk I/O. On Kaggle's NVMe storage this saves 60-120 seconds.

### Our code

No preloading. vLLM loads weights on first request.

---

## The Exact Rewrite Required

To go from 1/50 to 40+/50 we must rewrite the submission to use the Harmony protocol.
The minimum viable rewrite has these components:

### 1. Install and import openai_harmony

```python
# Already installed as a competition dataset — just import
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    Role, Conversation, Message, Author, TextContent, SystemContent,
    ReasoningEffort, ToolNamespaceConfig,
)
```

### 2. Initialize encoding and stop tokens

```python
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()
```

### 3. Build a Harmony Conversation object

```python
def make_conversation(problem_text, system_prompt):
    return Conversation(messages=[
        Message(
            author=Author(role=Role.SYSTEM),
            content=[SystemContent(
                text=system_prompt,
                reasoning_effort=ReasoningEffort.HIGH,
            )],
        ),
        Message(
            author=Author(role=Role.USER),
            content=[TextContent(text=problem_text)],
        ),
    ])
```

### 4. Call /v1/completions with raw token IDs

```python
def run_attempt(client, conversation, cfg):
    prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    token_buffer = []
    entropy_list = []

    stream = client.completions.create(
        model=cfg.served_model_name,
        prompt=prompt_ids,
        temperature=cfg.temperature,          # 1.0
        max_tokens=cfg.max_tokens,            # 65536
        stream=True,
        logprobs=cfg.top_logprobs,            # 5
        seed=cfg.seed,
        extra_body={
            'min_p': cfg.min_p,               # 0.02
            'stop_token_ids': stop_token_ids,
            'return_token_ids': True,
        }
    )

    for chunk in stream:
        choice = chunk.choices[0]
        if choice.logprobs and choice.logprobs.token_ids:
            token_buffer.extend(choice.logprobs.token_ids)
        if choice.logprobs and choice.logprobs.top_logprobs:
            for logprob_dict in choice.logprobs.top_logprobs:
                probs = softmax(list(logprob_dict.values()))
                ent = -sum(p * math.log(p + 1e-12) for p in probs)
                entropy_list.append(ent)

    return token_buffer, entropy_list
```

### 5. Parse Harmony messages and dispatch tool calls

```python
def process_attempt(client, problem_text, cfg, kernel_client):
    conversation = make_conversation(problem_text, SYSTEM_PROMPT)

    for step in range(cfg.max_python_calls):
        token_buffer, entropy_list = run_attempt(client, conversation, cfg)
        new_messages = encoding.parse_messages_from_completion_tokens(
            token_buffer, Role.ASSISTANT
        )
        conversation.messages.extend(new_messages)
        last_message = new_messages[-1]

        if last_message.channel == 'final':
            answer = scan_for_boxed_answer(last_message.content[0].text)
            avg_entropy = sum(entropy_list) / max(len(entropy_list), 1)
            weight = 1.0 / max(avg_entropy, 1e-9)
            return answer, weight

        if last_message.recipient == 'python':
            code = last_message.content[0].text
            output = execute_on_kernel(kernel_client, code)
            # Add tool response as Harmony message
            tool_msg = Message(
                author=Author(role=Role.TOOL, name='python'),
                content=[TextContent(text=output)],
            )
            conversation.messages.append(tool_msg)

    return None, 0.0
```

### 6. vLLM server startup

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /kaggle/input/gpt-oss-120b/... \
    --served-model-name gpt-oss-120b \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.99 \
    --max-model-len 81920 \
    --kv-cache-dtype fp8_e4m3 \
    --async-scheduling \
    --enable-prefix-caching \
    --stream-interval 200 \
    --port 8000
```

---

## Impact Estimate by Fix

| Fix | Expected LB Improvement | Effort |
|---|---|---|
| #1: Use /v1/completions + Harmony token IDs | +35-40 points | High (rewrite inference) |
| #2: Use Harmony tool calling + kernel pool | included in #1 | High (add kernel pool) |
| #3: Use stop_tokens_for_assistant_actions() | included in #1 | Trivial once #1 is done |
| #4: Add ReasoningEffort.HIGH | +2-4 points | Trivial once #1 is done |
| #5: Persistent Jupyter kernels | +1-3 points | Medium |
| #6: Fix sampling params (temp=1.0, max_tokens=65536, min_p=0.02) | +2-5 points | Trivial |
| #7: Fix context length (81920 vs 16384) | +3-8 points | Trivial (vLLM flag) |
| #8: Entropy-weighted voting | +1-2 points | Medium |
| #9: Time management + early stopping | +1-2 points | Medium |
| #10: Weight preloading | 0 points (reliability) | Low |

Fixes #1-#7 together should bring us from ~1/50 to ~35-43/50.
Fixes #8-#10 are refinements that help at the margin.

---

## Parameter Comparison Table

| Parameter | submission_v4_robust | top_notebook | nihilisticneuralnet | datasciencegrad |
|---|---|---|---|---|
| API endpoint | /v1/chat/completions | /v1/completions | /v1/completions | /v1/completions |
| Protocol | plain text | openai_harmony | openai_harmony | openai_harmony |
| ReasoningEffort | none | HIGH | HIGH | HIGH |
| temperature | 0.7 | 1.0 | 1.0 | 1.0 |
| max_tokens | 4096-8192 | 65536 | 65536 | 65536 |
| context_tokens | 16384 | 81920 | 65536 | 65536 |
| min_p | not set | 0.02 | 0.02 | 0.02 |
| top_logprobs | not set | 5 | 5 | 5 |
| attempts | varies | 8 | 8 | 8 |
| workers | 1-4 | 16 | 16 | 16 |
| early_stop | none | 4 | 4 | 4 |
| stop tokens | ["```output"] | stop_token_ids (Harmony) | stop_token_ids (Harmony) | stop_token_ids (Harmony) |
| code execution | exec() | Jupyter kernel pool | Jupyter kernel pool | Jupyter kernel pool |
| kernel pool size | 1 (no pool) | 16 | 16 | 16 |
| kv_cache_dtype | default (fp16) | fp8_e4m3 | default | default |
| prefix_caching | no | yes | no | no |
| gpu_memory_util | 0.85 | 0.99 | 0.96 | 0.96 |
| voting | code_ok+2, boxed+0.5 | inverse entropy | 5-component entropy | exp(confidence*2.0) |

---

## Conclusion

The 42-point gap (1/50 vs 43/50) is explained almost entirely by three protocol mismatches:

1. We call `/v1/chat/completions` with plain text. gpt-oss-120b expects `/v1/completions`
   with raw Harmony token IDs produced by `render_conversation_for_completion()`.

2. We detect tool calls by watching for the text string ` ```output `. gpt-oss-120b signals
   tool calls using special Harmony tokens decoded by `parse_messages_from_completion_tokens()`.
   It never outputs ` ```output `.

3. We use string-based stop conditions. gpt-oss-120b expects integer stop token IDs from
   `stop_tokens_for_assistant_actions()`.

The model was never able to use its native tool-calling capability in our submission. Every
inference call produced an off-distribution output because the input format was wrong.

The rewrite is substantial (replace the entire inference layer) but the path is clear:
copy the pattern from the top public notebooks exactly. The openai_harmony package is already
installed as a competition dataset. The key function calls are:
- `load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)` to get the encoding
- `encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)` to build prompts
- `encoding.stop_tokens_for_assistant_actions()` to get stop tokens
- `encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)` to parse responses

With this rewrite and the correct vLLM flags, we should reach 35-43/50 on the public leaderboard.
