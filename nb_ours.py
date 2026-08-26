# === CELL 0 ===
# Uninstall conflicting packages (if present)\nimport subprocess, sys\nfor pkg in ['keras', 'matplotlib', 'scikit-learn', 'tensorflow']:\n    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', pkg], capture_output=True)

# === CELL 1 ===
import warnings
warnings.simplefilter('ignore')

# === CELL 2 ===
import os
import sys
import subprocess

# === CELL 3 ===
import glob

def set_env(input_archive, temp_dir):
    # Find the archive - path varies by Kaggle runtime
    archive = input_archive
    if not os.path.exists(archive):
        candidates = glob.glob('/kaggle/input/**/wheels.tar.gz', recursive=True)
        if candidates:
            archive = candidates[0]
            print(f'Found archive at: {archive}')
        else:
            print('No wheels archive found, using pre-installed packages')
            return

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(['tar', '-xzf', archive, '-C', temp_dir], check=True)

    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--no-index', '--find-links', f'{temp_dir}/wheels',
        'unsloth', 'trl', 'vllm', 'openai_harmony'
    ], check=True)

    # Set tiktoken path
    tk = os.path.join(temp_dir, 'tiktoken_encodings')
    if os.path.exists(tk):
        os.environ['TIKTOKEN_ENCODINGS_BASE'] = tk

# === CELL 4 ===
set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz',
    temp_dir='/kaggle/tmp/setup'
)

# Find tiktoken encodings
for tk in glob.glob('/kaggle/input/**/tiktoken_encodings', recursive=True) + glob.glob('/kaggle/tmp/**/tiktoken_encodings', recursive=True):
    os.environ['TIKTOKEN_ENCODINGS_BASE'] = tk
    print(f'TIKTOKEN: {tk}')
    break

# USE BASE MODEL â€” 50-experiment study shows huikang with wrong pipeline = -6 points
# Only use huikang with full 131K + VOI pipeline (not implemented yet)
model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
if not os.path.exists(model_path):
    for candidate in glob.glob('/kaggle/input/**/config.json', recursive=True):
        d = os.path.dirname(candidate)
        if 'gpt-oss' in d.lower():
            model_path = d
            break
    print(f'Model path (fallback): {model_path}')
else:
    print(f'Model path: {model_path}')

# === CELL 5 ===
subprocess.run(['ls', '/kaggle/tmp/setup/tiktoken_encodings'])

# === CELL 6 ===
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'

# === CELL 7 ===
import gc
import re
import math
import time
import glob
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation
)

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

# === CELL 8 ===
# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up
# + phase split + failure retry + follow-up
# NO domain routing (Classify Then Solve = -3.7 pts)
# NO huikang (wrong pipeline = -6 pts)
# NO complex voting (worse than 1/entropy)
CFG_model_path = model_path

class CFG:
    # 5-step prompt from BOTH 44/50 notebooks (kaanyorgun + nihilisticneuralnet)
    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\n\n'
        '# Problem-Solving Approach:\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\'t commit to one approach immediately.\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\n\n'
        '# Mathematical Reasoning Principles:\n'
        '- Break complex problems into smaller, manageable sub-problems\n'
        '- Look for patterns, symmetries, and special cases that provide insight\n'
        '- Use concrete examples to build intuition before generalizing\n'
        '- Consider extreme cases and boundary conditions\n'
        '- If stuck, try working backwards from the desired result\n'
        '- Be willing to restart with a different approach if needed\n\n'
        '# Verification Requirements:\n'
        '- Cross-check arithmetic and algebraic manipulations\n'
        '- Verify that your solution satisfies all problem constraints\n'
        '- Test your answer with simple cases or special values when possible\n'
        '- Ensure dimensional consistency and reasonableness of the result\n\n'
        '# Output Format:\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )
    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )
    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n\n'
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations\n\n'
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions (trig, log, exp)\n'
        '- Constants like pi and e\n\n'
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification\n'
        '- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n'
        '- Validate computational results against known cases or theoretical bounds'
    )

    served_model_name = 'gpt-oss'
    model_path = CFG_model_path
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    # EXACT 44/50 vLLM params (kaanyorgun + nihilisticneuralnet)
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit = 17400
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3
    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256
    early_stop = 4
    attempts = 8
    workers = 16
    turns = 128
    seed = 42
    gpu_memory_utilization = 0.96
    temperature = 0.8  # arxiv 2603.27844: T=0.8 best mean (+0.3 over T=1.0)
    min_p = 0.02

print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')

# === CELL 9 ===
set_seed(CFG.seed)

# === CELL 10 ===
class AIMO3Template:
    def __init__(self): pass

    def get_system_content(self, system_prompt, tool_config):
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, system_prompt, user_prompt, tool_config):
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]

# === CELL 11 ===
class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'
        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]
        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True
        self.execute('import math\nimport numpy\nimport sympy\nimport mpmath\nimport itertools\nimport collections\nmpmath.mp.dps = 64\n')

    def _format_error(self, traceback):
        clean = []
        for frame in traceback:
            cf = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in cf and 'ipython-input' not in cf: continue
            clean.append(cf)
        return ''.join(clean)

    def execute(self, code, timeout=None):
        client = self._client
        eff_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        stdout_parts, stderr_parts = [], []
        start = time.time()
        while True:
            if time.time() - start > eff_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {eff_timeout} seconds'
            try: msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty: continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id: continue
            mt = msg.get('msg_type')
            ct = msg.get('content', {})
            if mt == 'stream':
                (stdout_parts if ct.get('name') == 'stdout' else stderr_parts).append(ct.get('text', ''))
            elif mt == 'error': stderr_parts.append(self._format_error(ct.get('traceback', [])))
            elif mt in {'execute_result', 'display_data'}:
                t = ct.get('data', {}).get('text/plain')
                if t: stdout_parts.append(t if t.endswith('\n') else f'{t}\n')
            elif mt == 'status' and ct.get('execution_state') == 'idle': break
        so = ''.join(stdout_parts)
        se = ''.join(stderr_parts)
        if se: return f'{so.rstrip()}\n{se}' if so else se
        return so if so.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        with contextlib.suppress(Exception):
            if self._client: self._client.stop_channels()
        if self._owns_kernel and self._km:
            with contextlib.suppress(Exception): self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception): self._km.cleanup_resources()

    def reset(self):
        self.execute('%reset -f\nimport math\nimport numpy\nimport sympy\nimport mpmath\nimport itertools\nimport collections\nmpmath.mp.dps = 64\n')

    def __del__(self): self.close()

# === CELL 12 ===
class AIMO3Tool:
    def __init__(self, local_jupyter_timeout, tool_prompt, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code):
        lines = code.strip().split('\n')
        if not lines: return code
        last = lines[-1].strip()
        if not last or 'print' in last or 'import' in last or last.startswith('#'): return code
        lines[-1] = 'print(' + last + ')'
        return '\n'.join(lines)

    @property
    def instruction(self): return self._tool_prompt

    @property
    def tool_config(self): return ToolNamespaceConfig(name='python', description=self.instruction, tools=[])

    def _make_response(self, output, channel=None):
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel: message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message):
        self._ensure_session()
        raw = message.content[0].text
        final = self._ensure_last_print(raw)
        with self._execution_lock:
            try: output = self._jupyter_session.execute(final)
            except TimeoutError as exc: output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]

# === CELL 13 ===

# Strategy-diverse preference prompts
PREF_CODE_FIRST = (
    'Solve this by writing a complete Python program. Go directly to code. '
    'Available: math, numpy, sympy. '
    'Your program must: 1) Compute the answer, '
    '2) Verify constraints, 3) Print the final answer. '
    'Prefer exact computation with sympy over floating point.'
)
PREF_SMALL_CASES = (
    'Start by testing small cases to find a pattern. '
    'If the problem involves n, try n=1,2,3,...,10. '
    'Write Python code to: 1) Compute results for small cases, '
    '2) Identify the pattern, 3) Verify for larger cases, '
    '4) Compute the final answer. Available: math, numpy, sympy'
)

FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\boxed{}.'
)

class AIMO3Solver:
    def __init__(self, cfg, port=8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self._preload_model_weights()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)
        self._wait_for_server()
        self._initialize_kernels()
        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self):
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        t0 = time.time()
        files, total = [], 0
        for root, _, fnames in os.walk(self.cfg.model_path):
            for fn in fnames:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp): files.append(fp); total += os.path.getsize(fp)
        def _read(p):
            with open(p, 'rb') as f:
                while f.read(1024*1024*1024): pass
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex: list(ex.map(_read, files))
        print(f'Processed {len(files)} files ({total/1e9:.2f} GB) in {time.time()-t0:.2f} seconds.\n')

    def _start_server(self):
        # EXACT 43/50 vLLM flags â€” including batched_tokens and capture_size
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0', '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        t0 = time.time()
        for _ in range(self.cfg.server_timeout):
            rc = self.server_process.poll()
            if rc is not None:
                self.log_file.flush()
                with open('vllm_server.log') as f: logs = f.read()
                raise RuntimeError(f'Server died with code {rc}. Last 3000 chars:\n{logs[-3000:]}')
            try:
                self.client.models.list()
                print(f'Server is ready (took {time.time()-t0:.2f} seconds).\n')
                return
            except: time.sleep(1)
        raise RuntimeError('Server failed to start (timeout).')

    def _initialize_kernels(self):
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        t0 = time.time()
        self.sandbox_pool = queue.Queue()
        def _mk(): return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(_mk) for _ in range(self.cfg.workers)]
            for f in as_completed(futs): self.sandbox_pool.put(f.result())
        print(f'Kernels initialized in {time.time()-t0:.2f} seconds.\n')

    def _scan_for_answer(self, text):
        for pat in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}']:
            ms = re.findall(pat, text)
            if ms:
                try:
                    v = int(ms[-1].replace(',',''))
                    if 0 <= v <= 99999: return v
                except: pass
        ms = re.findall(r'\\boxed\s*\{\s*(-[0-9,]+)\s*\}', text)
        if ms:
            try:
                v = int(ms[-1].replace(',','')) % 100000
                if 0 <= v <= 99999: return v
            except: pass
        ms = re.findall(r'final\s+answer\s+is\s*([0-9,]+)', text, re.IGNORECASE)
        if ms:
            try:
                v = int(ms[-1].replace(',',''))
                if 0 <= v <= 99999: return v
            except: pass
        return None

    def _compute_mean_entropy(self, logprobs_buffer):
        if not logprobs_buffer: return float('inf')
        total, cnt = 0.0, 0
        for d in logprobs_buffer:
            if not isinstance(d, dict) or not d: continue
            e = 0.0
            for _, lp in d.items():
                p = math.exp(lp)
                if p > 0: e -= p * math.log2(p)
            total += e; cnt += 1
        return total / cnt if cnt else float('inf')

    def _process_attempt(self, problem, system_prompt, attempt_index, stop_event, deadline):
        if stop_event.is_set() or time.time() > deadline:
            return {'Attempt': attempt_index+1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0, 'Response Length': 0, 'Entropy': float('inf')}
        sandbox = None
        python_calls = python_errors = total_tokens = 0
        final_answer = None
        logprobs_buffer = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(local_jupyter_timeout=self.cfg.jupyter_timeout, tool_prompt=self.cfg.tool_prompt, sandbox=sandbox)
            encoding = self.encoding
            messages = self.template.apply_chat_template(system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)
            for turn_idx in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline: break
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens: break
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs, max_tokens=max_tokens,
                    prompt=prompt_ids, seed=attempt_seed, stream=True,
                    extra_body={'min_p': self.cfg.min_p, 'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                )
                try:
                    token_buffer, text_chunks = [], []
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline: break
                        nt = chunk.choices[0].token_ids
                        nx = chunk.choices[0].text
                        if nt:
                            token_buffer.extend(nt); total_tokens += len(nt); text_chunks.append(nx)
                            clp = chunk.choices[0].logprobs
                            if clp and clp.top_logprobs: logprobs_buffer.extend(clp.top_logprobs)
                        if '}' in (nx or ''):
                            st = ''.join(text_chunks[-self.cfg.search_tokens:])
                            a = self._scan_for_answer(st)
                            if a is not None: final_answer = a; break
                finally: stream.close()
                if final_answer is not None: break
                if not token_buffer: break
                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last = new_messages[-1]
                if last.channel == 'final':
                    final_answer = self._scan_for_answer(last.content[0].text); break
                if last.recipient == 'python':
                    python_calls += 1
                    tool_resp = local_tool.process_sync_plus(last)
                    rt = tool_resp[0].content[0].text
                    if rt.startswith('[ERROR]') or 'Traceback' in rt or 'Error:' in rt: python_errors += 1
                    conversation.messages.extend(tool_resp)

            # ONLY ADDITION over 43/50 base: follow-up when no answer
            if final_answer is None and not stop_event.is_set() and time.time() < deadline:
                followup_msg = Message.from_role_and_content(Role.USER, FOLLOWUP_PROMPT)
                conversation.messages.append(followup_msg)
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens >= self.cfg.buffer_tokens:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name, temperature=0.0,
                        max_tokens=min(max_tokens, 512),
                        prompt=prompt_ids, seed=attempt_seed, stream=True,
                        extra_body={'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                    )
                    try:
                        text_chunks = []
                        for chunk in stream:
                            if stop_event.is_set() or time.time() > deadline: break
                            nx = chunk.choices[0].text
                            if nx: text_chunks.append(nx)
                            if '}' in (nx or ''):
                                st = ''.join(text_chunks[-16:])
                                a = self._scan_for_answer(st)
                                if a is not None: final_answer = a; break
                    finally: stream.close()

        except Exception as exc:
            python_errors += 1
        finally:
            if sandbox: sandbox.reset(); self.sandbox_pool.put(sandbox)
        return {'Attempt': attempt_index+1, 'Response Length': total_tokens, 'Python Calls': python_calls,
                'Python Errors': python_errors, 'Entropy': self._compute_mean_entropy(logprobs_buffer), 'Answer': final_answer}

    def _select_answer(self, results):
        # Plain 1/entropy â€” same as 43/50 base
        aw, av = defaultdict(float), defaultdict(int)
        for r in results:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                w = 1.0 / max(e, 1e-9)
                aw[a] += w; av[a] += 1
        scored = sorted([{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw], key=lambda x: x['score'], reverse=True)
        df = pd.DataFrame([(s['answer'], s['votes'], round(s['score'],3)) for s in scored], columns=['Answer','Votes','Score'])
        display(df)
        if not scored: print('\nFinal Answer: 0\n'); return 0
        print(f'\nFinal Answer: {scored[0]["answer"]}\n')
        return scored[0]['answer']

    def solve_problem(self, problem):
        print(f'\nProblem: {problem}\n')
        strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2
        user_inputs = [f'{problem} {p}' for p in strategy_prefs[:self.cfg.attempts]]
        user_input = user_inputs[0]
        elapsed = time.time() - self.notebook_start_time
        left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        deadline = time.time() + budget
        print(f'Budget: {budget:.0f}s | Problems left: {self.problems_remaining}\n')
        tasks = [(self.cfg.system_prompt, i) for i in range(self.cfg.attempts)]
        detailed, valid = [], []
        stop = threading.Event()
        ex = ThreadPoolExecutor(max_workers=self.cfg.workers)
        try:
            futs = [ex.submit(self._process_attempt, user_inputs[ai], sp, ai, stop, deadline) for sp, ai in tasks]
            for f in as_completed(futs):
                try:
                    r = f.result(); detailed.append(r)
                    if r['Answer'] is not None: valid.append(r['Answer'])
                    c = Counter(valid).most_common(1)
                    if c and c[0][1] >= self.cfg.early_stop:
                        stop.set()
                        for ff in futs: ff.cancel()
                        break
                except Exception as e: print(f'Future failed: {e}')
        finally:
            stop.set(); ex.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)
        if detailed:
            df = pd.DataFrame(detailed)
            df['Entropy'] = df['Entropy'].round(3)
            df['Answer'] = df['Answer'].astype('Int64')
            display(df)
        if not valid: print('\nResult: 0\n'); return 0
        return self._select_answer(detailed)

    def __del__(self):
        if hasattr(self, 'server_process'): self.server_process.terminate(); self.server_process.wait()
        if hasattr(self, 'log_file'): self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try: self.sandbox_pool.get_nowait().close()
                except: pass

# === CELL 14 ===
solver = AIMO3Solver(CFG)

# === CELL 15 ===
def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_value = id_.item(0)
    question_text = question.item(0)
    gc.disable()
    final_answer = solver.solve_problem(question_text)
    gc.enable()
    gc.collect()
    return pl.DataFrame({'id': id_value, 'answer': final_answer})

# === CELL 16 ===
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # Find test.csv
    candidates = (
        glob.glob('/kaggle/input/competitions/*/test.csv') +
        glob.glob('/kaggle/input/*/test.csv') +
        glob.glob('/kaggle/input/*/*/test.csv')
    )
    test_path = candidates[0] if candidates else '/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/test.csv'
    print(f'Test: {test_path}')
    inference_server.run_local_gateway((test_path,))