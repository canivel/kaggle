"""Build a standalone Kaggle validation notebook.

Runs GPT-OSS-120B on the 10 AIMO3 reference problems.
Compares baseline (1/entropy) vs EV voting side by side.
NOT a competition submission — runs free on Kaggle H100.
"""

import json
import io
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_NB = NOTEBOOKS_DIR / "validation_ev_vs_baseline.ipynb"
PUSH_DIR = NOTEBOOKS_DIR / "push_validation"

# Load reference problems to embed inline
with open(DATA_DIR / "aimo3_reference_bench.json") as f:
    BENCH = json.load(f)

BENCH_INLINE = json.dumps(BENCH, indent=2)

# ── Notebook cells ─────────────────────────────────────────────────────────────

CELLS = []

def code(src): CELLS.append({"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None})
def md(src):   CELLS.append({"cell_type": "markdown", "source": src, "metadata": {}})

md("# AIMO3 Validation: Baseline vs EV Voting\n\nRuns GPT-OSS-120B on 10 reference problems. Compares plain 1/entropy voting (v27) vs execution-verified voting (v31).\n\n**NOT a competition submission — free H100 validation run.**")

code("""\
import subprocess, sys
for pkg in ['keras', 'matplotlib', 'scikit-learn', 'tensorflow']:
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '--yes', pkg], capture_output=True)
""")

code("import warnings\nwarnings.simplefilter('ignore')")

code("""\
import os, sys, subprocess, glob

def set_env(input_archive, temp_dir):
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
    tk = os.path.join(temp_dir, 'tiktoken_encodings')
    if os.path.exists(tk):
        os.environ['TIKTOKEN_ENCODINGS_BASE'] = tk

set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz',
    temp_dir='/kaggle/tmp/setup'
)
for tk in glob.glob('/kaggle/input/**/tiktoken_encodings', recursive=True) + glob.glob('/kaggle/tmp/**/tiktoken_encodings', recursive=True):
    os.environ['TIKTOKEN_ENCODINGS_BASE'] = tk
    print(f'TIKTOKEN: {tk}')
    break

model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
if not os.path.exists(model_path):
    for c in glob.glob('/kaggle/input/**/config.json', recursive=True):
        d = os.path.dirname(c)
        if 'gpt-oss' in d.lower():
            model_path = d
            break
print(f'Model: {model_path}')
""")

code("""\
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'
""")

code("""\
import gc, re, math, time, queue, threading, contextlib, json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from jupyter_client import KernelManager
from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    SystemContent, ReasoningEffort, ToolNamespaceConfig,
    Author, Message, Role, TextContent, Conversation
)
from transformers import set_seed
""")

code("""\
CFG_model_path = model_path

class CFG:
    system_prompt = (
        'You are an elite mathematical problem solver with expertise at the International '
        'Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through '
        'rigorous mathematical reasoning.\\n\\n'
        '# Problem-Solving Approach:\\n'
        '1. UNDERSTAND: Carefully read and rephrase the problem in your own words. '
        'Identify what is given, what needs to be found, and any constraints.\\n'
        '2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, '
        'techniques, patterns, or analogous problems. Don\\'t commit to one approach immediately.\\n'
        '3. PLAN: Select the most promising approach and outline key steps before executing.\\n'
        '4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\\n'
        '5. VERIFY: Check your answer by substituting back, testing edge cases, or using '
        'alternative methods. Ensure logical consistency throughout.\\n\\n'
        '# Mathematical Reasoning Principles:\\n'
        '- Break complex problems into smaller, manageable sub-problems\\n'
        '- Look for patterns, symmetries, and special cases that provide insight\\n'
        '- Use concrete examples to build intuition before generalizing\\n'
        '- Consider extreme cases and boundary conditions\\n'
        '- If stuck, try working backwards from the desired result\\n'
        '- Be willing to restart with a different approach if needed\\n\\n'
        '# Verification Requirements:\\n'
        '- Cross-check arithmetic and algebraic manipulations\\n'
        '- Verify that your solution satisfies all problem constraints\\n'
        '- Test your answer with simple cases or special values when possible\\n'
        '- Ensure dimensional consistency and reasonableness of the result\\n\\n'
        '# Output Format:\\n'
        'The final answer must be a non-negative integer between 0 and 99999.\\n'
        'Place your final numerical answer inside \\\\boxed{}, e.g., \\\\boxed{42}\\n\\n'
        'Think step-by-step and show your complete reasoning process. Quality of reasoning '
        'is as important as the final answer.'
    )
    tool_prompt = (
        'Use this tool to execute Python code for:\\n'
        '- Complex calculations that would be error-prone by hand\\n'
        '- Numerical verification of analytical results\\n'
        '- Generating examples or testing conjectures\\n'
        '- Brute-force verification for small cases\\n\\n'
        'The environment is a stateful Jupyter notebook. Always use print() to display results.\\n'
        'Code should support your mathematical reasoning, not replace it.'
    )
    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for symbolic and numerical computation. '
        'Use sympy for exact answers, numpy for numerical verification. '
        'Combine symbolic and numerical approaches: derive symbolically, verify numerically.'
    )
    served_model_name  = 'gpt-oss'
    model_path         = CFG_model_path
    kv_cache_dtype     = 'fp8_e4m3'
    dtype              = 'auto'
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit     = 17400
    server_timeout     = 1200  # 20 min — model has 15 shards, ~19 min to load without preload
    session_timeout    = 960
    jupyter_timeout    = 6
    sandbox_timeout    = 3
    stream_interval    = 200
    context_tokens     = 65536
    buffer_tokens      = 512
    search_tokens      = 32
    top_logprobs       = 5
    batch_size         = 256
    early_stop         = 4
    attempts           = 8
    workers            = 16
    turns              = 128
    seed               = 42
    gpu_memory_utilization = 0.96
    temperature        = 0.8
    min_p              = 0.02

set_seed(CFG.seed)
print('CFG loaded')
""")

code("""\
PREF_CODE_FIRST = (
    'Solve this by writing a complete Python program. Go directly to code. '
    'Available: math, numpy, sympy. '
    'Your program must: 1) Compute the answer, 2) Verify constraints, 3) Print the final answer. '
    'Prefer exact computation with sympy over floating point.'
)
PREF_SMALL_CASES = (
    'Start by testing small cases to find a pattern. '
    'If the problem involves n, try n=1,2,3,...,10. '
    'Write Python code to: 1) Compute results for small cases, 2) Identify the pattern, '
    '3) Verify for larger cases, 4) Compute the final answer. Available: math, numpy, sympy'
)
FOLLOWUP_PROMPT = (
    'You have been working on this problem. Based on your analysis so far, '
    'what is the final integer answer? The answer must be between 0 and 99999. '
    'Please state your answer inside \\\\boxed{}.'
)
""")

code("""\
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
        self._km = None
        self._client = None
        ports = self._get_next_ports(5)
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'
        self._km = KernelManager()
        self._km.shell_port   = ports[0]
        self._km.iopub_port   = ports[1]
        self._km.stdin_port   = ports[2]
        self._km.hb_port      = ports[3]
        self._km.control_port = ports[4]
        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=timeout)
        self.execute('import math\\nimport numpy\\nimport sympy\\nimport mpmath\\nimport itertools\\nimport collections\\nmpmath.mp.dps = 64\\n')

    def _fmt_error(self, tb):
        clean = []
        for frame in tb:
            cf = re.sub(r'\\x1b\\[[0-9;]*m', '', frame)
            if 'File "' in cf and 'ipython-input' not in cf: continue
            clean.append(cf)
        return ''.join(clean)

    def execute(self, code, timeout=None):
        eff = timeout or self._default_timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        out, err = [], []
        start = time.time()
        while True:
            if time.time() - start > eff:
                self._km.interrupt_kernel()
                return f'[ERROR] Timeout after {eff}s'
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id: continue
            mt = msg.get('msg_type'); ct = msg.get('content', {})
            if mt == 'stream':
                (out if ct.get('name') == 'stdout' else err).append(ct.get('text', ''))
            elif mt == 'error':
                err.append(self._fmt_error(ct.get('traceback', [])))
            elif mt in {'execute_result', 'display_data'}:
                t = ct.get('data', {}).get('text/plain')
                if t: out.append(t if t.endswith('\\n') else t + '\\n')
            elif mt == 'status' and ct.get('execution_state') == 'idle': break
        so, se = ''.join(out), ''.join(err)
        if se: return f'{so.rstrip()}\\n{se}' if so else se
        return so if so.strip() else '[WARN] No output. Use print().'

    def reset(self):
        self.execute('%reset -f\\nimport math\\nimport numpy\\nimport sympy\\nimport mpmath\\nimport itertools\\nimport collections\\nmpmath.mp.dps = 64\\n')

    def close(self):
        with contextlib.suppress(Exception):
            if self._client: self._client.stop_channels()
        with contextlib.suppress(Exception):
            if self._km: self._km.shutdown_kernel(now=True)

    def __del__(self): self.close()
""")

code("""\
class AIMO3Tool:
    def __init__(self, jupyter_timeout, tool_prompt, sandbox):
        self._jupyter_timeout = jupyter_timeout
        self._tool_prompt = tool_prompt
        self._sandbox = sandbox
        self._lock = threading.Lock()

    @property
    def tool_config(self):
        return ToolNamespaceConfig(name='python', description=self._tool_prompt, tools=[])

    def _ensure_print(self, code):
        lines = code.strip().split('\\n')
        if not lines: return code
        last = lines[-1].strip()
        if not last or 'print' in last or 'import' in last or last.startswith('#'): return code
        lines[-1] = 'print(' + last + ')'
        return '\\n'.join(lines)

    def process_sync_plus(self, message):
        code = self._ensure_print(message.content[0].text)
        with self._lock:
            output = self._sandbox.execute(code)
        author = Author(role=Role.TOOL, name='python')
        resp = Message(author=author, content=[TextContent(text=output)]).with_recipient('assistant')
        if message.channel: resp = resp.with_channel(message.channel)
        return [resp]
""")

code("""\
class AIMO3Template:
    def apply_chat_template(self, system_prompt, user_prompt, tool_config):
        sc = (SystemContent.new()
              .with_model_identity(system_prompt)
              .with_reasoning_effort(ReasoningEffort.HIGH)
              .with_tools(tool_config))
        return [
            Message.from_role_and_content(Role.SYSTEM, sc),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]

class AIMO3Solver:
    def __init__(self, cfg, port=8000):
        self.cfg = cfg
        self.port = port
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self._preload_weights()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=f'http://0.0.0.0:{port}/v1', api_key='sk-local', timeout=cfg.session_timeout)
        self._wait_for_server()
        self._init_kernels()
        self.notebook_start = time.time()
        self.problems_remaining = 10  # validation: 10 problems

    def _preload_weights(self):
        print(f'Preloading model weights into OS page cache...')
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
        print(f'Preloaded {len(files)} files ({total/1e9:.1f} GB) in {time.time()-t0:.1f}s')

    def _start_server(self):
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
            '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching',
        ]
        log = open('/kaggle/working/vllm.log', 'w')
        print(f'Starting vLLM server...')
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM...')
        for _ in range(self.cfg.server_timeout):
            if self.server_process.poll() is not None:
                with open('/kaggle/working/vllm.log') as f: logs = f.read()
                raise RuntimeError(f'vLLM died.\\n{logs[-2000:]}')
            try:
                self.client.models.list()
                print('vLLM ready.')
                return
            except: time.sleep(1)
        raise RuntimeError('vLLM timeout')

    def _init_kernels(self):
        print(f'Starting {self.cfg.workers} kernels...')
        self.pool = queue.Queue()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(AIMO3Sandbox, self.cfg.jupyter_timeout) for _ in range(self.cfg.workers)]
            for f in as_completed(futs): self.pool.put(f.result())
        print('Kernels ready.')

    def _scan_answer(self, text):
        for pat in [r'\\\\boxed\\s*\\{\\s*([0-9,]+)\\s*\\}']:
            ms = re.findall(pat, text)
            if ms:
                try:
                    v = int(ms[-1].replace(',', ''))
                    if 0 <= v <= 99999: return v
                except: pass
        ms = re.findall(r'final\\s+answer\\s+is\\s*([0-9,]+)', text, re.IGNORECASE)
        if ms:
            try:
                v = int(ms[-1].replace(',', ''))
                if 0 <= v <= 99999: return v
            except: pass
        return None

    def _mean_entropy(self, buf):
        if not buf: return float('inf')
        total, cnt = 0.0, 0
        for d in buf:
            if not isinstance(d, dict): continue
            e = sum(-math.exp(lp)*math.log2(math.exp(lp)) for lp in d.values() if math.exp(lp) > 0)
            total += e; cnt += 1
        return total / cnt if cnt else float('inf')

    def _process_attempt(self, problem, system_prompt, attempt_idx, stop_evt, deadline):
        if stop_evt.is_set() or time.time() > deadline:
            return {'Attempt': attempt_idx+1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0, 'Entropy': float('inf')}
        python_calls = python_errors = total_tokens = 0
        final_answer = None
        logprobs_buf = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_idx, 2))
        sandbox = None
        try:
            sandbox = self.pool.get(timeout=self.cfg.sandbox_timeout)
            tool = AIMO3Tool(self.cfg.jupyter_timeout, self.cfg.tool_prompt, sandbox)
            msgs = self.template.apply_chat_template(system_prompt, problem, tool.tool_config)
            conv = Conversation.from_messages(msgs)
            for _ in range(self.cfg.turns):
                if stop_evt.is_set() or time.time() > deadline: break
                prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
                max_tok = self.cfg.context_tokens - len(prompt_ids)
                if max_tok < self.cfg.buffer_tokens: break
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs, max_tokens=max_tok,
                    prompt=prompt_ids, seed=attempt_seed, stream=True,
                    extra_body={'min_p': self.cfg.min_p, 'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                )
                try:
                    tok_buf, txt = [], []
                    for chunk in stream:
                        if stop_evt.is_set() or time.time() > deadline: break
                        nt = chunk.choices[0].token_ids
                        nx = chunk.choices[0].text
                        if nt:
                            tok_buf.extend(nt); total_tokens += len(nt); txt.append(nx)
                            clp = chunk.choices[0].logprobs
                            if clp and clp.top_logprobs: logprobs_buf.extend(clp.top_logprobs)
                        if '}' in (nx or ''):
                            a = self._scan_answer(''.join(txt[-self.cfg.search_tokens:]))
                            if a is not None: final_answer = a; break
                finally: stream.close()
                if final_answer is not None: break
                if not tok_buf: break
                new_msgs = self.encoding.parse_messages_from_completion_tokens(tok_buf, Role.ASSISTANT)
                conv.messages.extend(new_msgs)
                last = new_msgs[-1]
                if last.channel == 'final':
                    final_answer = self._scan_answer(last.content[0].text); break
                if last.recipient == 'python':
                    python_calls += 1
                    resp = tool.process_sync_plus(last)
                    rt = resp[0].content[0].text
                    if rt.startswith('[ERROR]') or 'Traceback' in rt or 'Error:' in rt: python_errors += 1
                    conv.messages.extend(resp)
            # follow-up
            if final_answer is None and not stop_evt.is_set() and time.time() < deadline:
                fu = Message.from_role_and_content(Role.USER, FOLLOWUP_PROMPT)
                conv.messages.append(fu)
                prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
                max_tok = self.cfg.context_tokens - len(prompt_ids)
                if max_tok >= self.cfg.buffer_tokens:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name, temperature=0.0,
                        max_tokens=min(max_tok, 512), prompt=prompt_ids, seed=attempt_seed, stream=True,
                        extra_body={'stop_token_ids': self.stop_token_ids, 'return_token_ids': True}
                    )
                    try:
                        txt = []
                        for chunk in stream:
                            nx = chunk.choices[0].text
                            if nx: txt.append(nx)
                            if '}' in (nx or ''):
                                a = self._scan_answer(''.join(txt[-16:]))
                                if a is not None: final_answer = a; break
                    finally: stream.close()
        except Exception as e:
            print(f'Attempt {attempt_idx} exception: {e}')
            python_errors += 1
        finally:
            if sandbox: sandbox.reset(); self.pool.put(sandbox)
        return {
            'Attempt': attempt_idx+1, 'Answer': final_answer,
            'Python Calls': python_calls, 'Python Errors': python_errors,
            'Entropy': self._mean_entropy(logprobs_buf),
        }

    def solve(self, problem):
        strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2
        user_inputs = [f'{problem} {p}' for p in strategy_prefs[:self.cfg.attempts]]
        elapsed = time.time() - self.notebook_start
        left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        deadline = time.time() + budget
        print(f'Budget: {budget:.0f}s | Problems remaining: {self.problems_remaining}')
        results = []
        valid = []
        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(self._process_attempt, user_inputs[ai], self.cfg.system_prompt, ai, stop, deadline) for ai in range(self.cfg.attempts)]
            for f in as_completed(futs):
                try:
                    r = f.result(); results.append(r)
                    if r['Answer'] is not None: valid.append(r['Answer'])
                    c = Counter(valid).most_common(1)
                    if c and c[0][1] >= self.cfg.early_stop:
                        stop.set()
                        for ff in futs: ff.cancel()
                        break
                except Exception as e: print(f'Future error: {e}')
        self.problems_remaining = max(0, self.problems_remaining - 1)
        return results

    def shutdown(self):
        self.server_process.terminate()
        self.server_process.wait()
        while not self.pool.empty():
            try: self.pool.get_nowait().close()
            except: pass

print('Classes loaded')
""")

code("""\
def select_baseline(results):
    \"\"\"Plain 1/entropy voting (v27 baseline)\"\"\"
    aw = defaultdict(float)
    for r in results:
        a, e = r['Answer'], r['Entropy']
        if a is not None:
            aw[a] += 1.0 / max(e, 1e-9)
    return max(aw, key=aw.get) if aw else 0

def select_ev(results):
    \"\"\"EV voting (v31): execution-verified entropy weights\"\"\"
    aw = defaultdict(float)
    for r in results:
        a, e = r['Answer'], r['Entropy']
        if a is not None:
            base_w = 1.0 / max(e, 1e-9)
            pc, pe = r.get('Python Calls', 0), r.get('Python Errors', 0)
            if pc > 0 and pe == 0:
                mult = 10.0   # clean code execution
            elif pc > 0:
                mult = 0.1    # code ran with errors
            else:
                mult = 0.2    # no code execution
            aw[a] += base_w * mult
    return max(aw, key=aw.get) if aw else 0
""")

# Embed reference problems inline
code(f"""\
# 10 AIMO3 reference problems (from reference.csv + AIMO3_Reference_Problems.pdf)
BENCH = {BENCH_INLINE}
print(f'Loaded {{len(BENCH)}} reference problems')
""")

code("""\
solver = AIMO3Solver(CFG)
print('Solver ready')
""")

code("""\
# ── Run validation ─────────────────────────────────────────────────────────────
results_log = []
baseline_correct = ev_correct = 0

for i, item in enumerate(BENCH):
    pid      = item['id']
    problem  = item['problem']
    expected = item['answer']
    source   = item.get('source', '')
    print(f'\\n{"="*60}')
    print(f'Problem {i+1}/10 | {source}')
    print(f'Expected: {expected}')
    print(f'{"="*60}')

    attempt_results = solver.solve(problem)

    ans_b = select_baseline(attempt_results)
    ans_e = select_ev(attempt_results)

    b_ok = int(ans_b) == int(expected)
    e_ok = int(ans_e) == int(expected)
    if b_ok: baseline_correct += 1
    if e_ok: ev_correct += 1

    # Per-attempt table
    df = pd.DataFrame(attempt_results)
    df['Entropy'] = df['Entropy'].apply(lambda x: round(x, 3) if x != float('inf') else 'inf')
    df['Answer'] = df['Answer'].astype('Int64')
    display(df)

    print(f'  Baseline (1/entropy): {ans_b}  {"CORRECT" if b_ok else "WRONG"}')
    print(f'  EV voting:            {ans_e}  {"CORRECT" if e_ok else "WRONG"}')
    print(f'  Running: Baseline={baseline_correct}/{i+1}  EV={ev_correct}/{i+1}')

    results_log.append({
        'id': pid, 'source': source, 'expected': expected,
        'ans_baseline': ans_b, 'ans_ev': ans_e,
        'baseline_correct': b_ok, 'ev_correct': e_ok,
        'attempts': attempt_results,
    })

solver.shutdown()
""")

code("""\
# ── Summary ────────────────────────────────────────────────────────────────────
print(f'\\n{"="*60}')
print(f'VALIDATION RESULTS — 10 AIMO3 Reference Problems')
print(f'{"="*60}')
print(f'  Baseline (v27 strategy diversity, 1/entropy):  {baseline_correct}/10')
print(f'  EV voting (v31 = v27 + EV multiplier):         {ev_correct}/10')
print(f'  Delta:                                         {ev_correct - baseline_correct:+d}')
print(f'{"="*60}')

summary_df = pd.DataFrame([{
    'Problem': r['source'],
    'Expected': r['expected'],
    'Baseline': r['ans_baseline'],
    'B_ok': 'Y' if r['baseline_correct'] else 'N',
    'EV': r['ans_ev'],
    'EV_ok': 'Y' if r['ev_correct'] else 'N',
    'Same': 'Y' if r['ans_baseline'] == r['ans_ev'] else 'DIFF',
} for r in results_log])
display(summary_df)

import json
with open('/kaggle/working/validation_results.json', 'w') as f:
    json.dump(results_log, f, indent=2)
print('Results saved to /kaggle/working/validation_results.json')
""")

# ── Build notebook ─────────────────────────────────────────────────────────────

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": CELLS
}

PUSH_DIR.mkdir(exist_ok=True)
with io.open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"Written: {OUT_NB}")

import shutil
shutil.copy(OUT_NB, PUSH_DIR / "validation_ev_vs_baseline.ipynb")

meta = {
    "id": "canivel/aimo3-validation-ev-vs-baseline",
    "title": "AIMO3 Validation EV vs Baseline",
    "code_file": "validation_ev_vs_baseline.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "model_sources": ["danielhanchen/gpt-oss-120b/Transformers/default/1"],
    "kernel_sources": ["andreasbis/aimo-3-utils"],
    "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
    "keywords": [],
    "machine_shape": "NvidiaH100",
}
with io.open(PUSH_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Push dir: {PUSH_DIR}")
print(f"Slug: aimo3-validation-ev-vs-baseline")
print(f"To push: cd {PUSH_DIR} && kaggle kernels push")
