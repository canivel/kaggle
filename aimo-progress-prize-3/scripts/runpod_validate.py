"""
RunPod validation script for AIMO3 submission logic.

Setup:
  1. RunPod: 3x A40 (or 2x A100 80GB) pod, PyTorch image
  2. On the pod:
     pip install vllm openai openai_harmony pandas
  3. Download model:
     huggingface-cli download danielhanchen/gpt-oss-120b --local-dir /workspace/gpt-oss-120b
  4. Run:
     python runpod_validate.py

This validates:
  A) Baseline v27: strategy diversity only (plain 1/entropy voting)
  B) v31: strategy diversity + EV voting (10x/0.1x/0.2x multipliers)

Expected result on 10 reference problems:
  - Baseline Claude Opus: 9/10 (for reference)
  - GPT-OSS-120B baseline: 4/10 (official PDF)
  - Our v27 config: ~4-6/10 expected
  - v31 EV voting: should match or beat v27 (never worse by design if MC is right)

Cost estimate: ~$3-5 on RunPod (3x A40 @ $0.44/hr x ~2hrs)
"""

import os, sys, re, math, time, json, queue, threading, subprocess, contextlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding,
    SystemContent, ReasoningEffort, ToolNamespaceConfig,
    Author, Message, Role, TextContent, Conversation
)
from jupyter_client import KernelManager
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/gpt-oss-120b")
BENCH_PATH = os.environ.get("BENCH_PATH", str(Path(__file__).parent.parent / "data/aimo3_reference_bench.json"))
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/workspace/validation_results.json")

class CFG:
    served_model_name = "gpt-oss"
    model_path        = MODEL_PATH
    kv_cache_dtype    = "fp8_e4m3"
    dtype             = "auto"
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit    = 17400
    server_timeout    = 180
    session_timeout   = 960
    jupyter_timeout   = 6
    sandbox_timeout   = 3
    stream_interval   = 200
    context_tokens    = 65536
    buffer_tokens     = 512
    search_tokens     = 32
    top_logprobs      = 5
    batch_size        = 256
    early_stop        = 4
    attempts          = 8
    workers           = 16
    turns             = 128
    seed              = 42
    gpu_memory_utilization = 0.96
    temperature       = 0.8
    min_p             = 0.02

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
        '- Brute-force verification for small cases\n\n'
        'The environment is a stateful Jupyter notebook. Always use print() to display results.\n'
        'Code should support your mathematical reasoning, not replace it.'
    )
    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for symbolic and numerical computation. '
        'Use sympy for exact answers, numpy for numerical verification. '
        'Combine symbolic and numerical approaches: derive symbolically, verify numerically.'
    )

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
    'Please state your answer inside \\boxed{}.'
)

# ── Sandbox ─────────────────────────────────────────────────────────────────────

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
        self._km.shell_port, self._km.iopub_port = ports[0], ports[1]
        self._km.stdin_port, self._km.hb_port = ports[2], ports[3]
        self._km.control_port = ports[4]
        self._km.start_kernel(env=env)
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=timeout)
        self.execute('import math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n')

    def execute(self, code, timeout=None):
        t = timeout or self._default_timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
        out, err = [], []
        start = time.time()
        while True:
            if time.time() - start > t:
                self._km.interrupt_kernel()
                return f'[ERROR] Timeout after {t}s'
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue
            mt = msg.get('msg_type')
            ct = msg.get('content', {})
            if mt == 'stream':
                (out if ct.get('name') == 'stdout' else err).append(ct.get('text', ''))
            elif mt == 'error':
                err.append('\n'.join(ct.get('traceback', [])))
            elif mt in {'execute_result', 'display_data'}:
                t2 = ct.get('data', {}).get('text/plain')
                if t2: out.append(t2 + '\n')
            elif mt == 'status' and ct.get('execution_state') == 'idle':
                break
        so, se = ''.join(out), ''.join(err)
        if se: return f'{so.rstrip()}\n{se}' if so else se
        return so if so.strip() else '[WARN] No output.'

    def reset(self):
        self.execute('%reset -f\nimport math, numpy, sympy, mpmath, itertools, collections\nmpmath.mp.dps = 64\n')

    def close(self):
        with contextlib.suppress(Exception):
            if self._client: self._client.stop_channels()
        with contextlib.suppress(Exception):
            if self._km: self._km.shutdown_kernel(now=True)

    def __del__(self): self.close()


# ── Tool ────────────────────────────────────────────────────────────────────────

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
        lines = code.strip().split('\n')
        if not lines: return code
        last = lines[-1].strip()
        if not last or 'print' in last or 'import' in last or last.startswith('#'): return code
        lines[-1] = 'print(' + last + ')'
        return '\n'.join(lines)

    def process_sync_plus(self, message):
        raw = message.content[0].text
        code = self._ensure_print(raw)
        with self._lock:
            output = self._sandbox.execute(code)
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        resp = Message(author=author, content=[content]).with_recipient('assistant')
        if message.channel: resp = resp.with_channel(message.channel)
        return [resp]


# ── Solver ───────────────────────────────────────────────────────────────────────

class AIMO3Solver:
    def __init__(self, cfg, port=8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.template_obj = self._make_template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key='sk-local', timeout=cfg.session_timeout)
        self._wait_for_server()
        self._init_kernels()
        self.notebook_start = time.time()
        self.problems_remaining = 50

    def _make_template(self):
        class T:
            def apply_chat_template(self, sys_p, user_p, tool_config):
                sc = (SystemContent.new()
                      .with_model_identity(sys_p)
                      .with_reasoning_effort(ReasoningEffort.HIGH)
                      .with_tools(tool_config))
                sm = Message.from_role_and_content(Role.SYSTEM, sc)
                um = Message.from_role_and_content(Role.USER, user_p)
                return [sm, um]
        return T()

    def _start_server(self):
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', str(max(1, len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')))),
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0', '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching',
        ]
        log = open('/workspace/vllm.log', 'w')
        print(f'Starting vLLM: {" ".join(cmd)}')
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        for _ in range(self.cfg.server_timeout):
            if self.server_process.poll() is not None:
                raise RuntimeError(f'vLLM died. Check /workspace/vllm.log')
            try:
                self.client.models.list()
                print('vLLM ready.')
                return
            except:
                time.sleep(1)
        raise RuntimeError('vLLM timeout')

    def _init_kernels(self):
        print(f'Starting {self.cfg.workers} Jupyter kernels...')
        self.pool = queue.Queue()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(AIMO3Sandbox, self.cfg.jupyter_timeout) for _ in range(self.cfg.workers)]
            for f in as_completed(futs): self.pool.put(f.result())
        print('Kernels ready.')

    def _scan_answer(self, text):
        for pat in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}']:
            ms = re.findall(pat, text)
            if ms:
                try:
                    v = int(ms[-1].replace(',', ''))
                    if 0 <= v <= 99999: return v
                except: pass
        ms = re.findall(r'final\s+answer\s+is\s*([0-9,]+)', text, re.IGNORECASE)
        if ms:
            try:
                v = int(ms[-1].replace(',', ''))
                if 0 <= v <= 99999: return v
            except: pass
        return None

    def _mean_entropy(self, logprobs_buf):
        if not logprobs_buf: return float('inf')
        total, cnt = 0.0, 0
        for d in logprobs_buf:
            if not isinstance(d, dict): continue
            e = sum(-math.exp(lp) * math.log2(math.exp(lp)) for lp in d.values() if math.exp(lp) > 0)
            total += e; cnt += 1
        return total / cnt if cnt else float('inf')

    def _process_attempt(self, problem, system_prompt, attempt_idx, stop_evt, deadline):
        if stop_evt.is_set() or time.time() > deadline:
            return {'Attempt': attempt_idx+1, 'Answer': None, 'Python Calls': 0, 'Python Errors': 0, 'Entropy': float('inf')}
        sandbox = python_calls = python_errors = total_tokens = 0
        final_answer = None
        logprobs_buf = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_idx, 2))
        sandbox = None
        try:
            sandbox = self.pool.get(timeout=self.cfg.sandbox_timeout)
            tool = AIMO3Tool(self.cfg.jupyter_timeout, self.cfg.tool_prompt, sandbox)
            msgs = self.template_obj.apply_chat_template(system_prompt, problem, tool.tool_config)
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
                    if rt.startswith('[ERROR]') or 'Traceback' in rt: python_errors += 1
                    conv.messages.extend(resp)

            # follow-up if no answer
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
            print(f'Attempt {attempt_idx} error: {e}')
            python_errors += 1
        finally:
            if sandbox: sandbox.reset(); self.pool.put(sandbox)
        return {
            'Attempt': attempt_idx+1, 'Answer': final_answer,
            'Python Calls': python_calls, 'Python Errors': python_errors,
            'Entropy': self._mean_entropy(logprobs_buf),
        }

    def _select_baseline(self, results):
        """Plain 1/entropy (v27 baseline)"""
        aw, av = defaultdict(float), defaultdict(int)
        for r in results:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                aw[a] += 1.0 / max(e, 1e-9)
                av[a] += 1
        if not aw: return 0
        return max(aw, key=aw.get)

    def _select_ev(self, results):
        """EV voting (v31): execution-verified entropy weights"""
        aw, av = defaultdict(float), defaultdict(int)
        for r in results:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                base_w = 1.0 / max(e, 1e-9)
                pc, pe = r.get('Python Calls', 0), r.get('Python Errors', 0)
                if pc > 0 and pe == 0:
                    mult = 10.0
                elif pc > 0:
                    mult = 0.1
                else:
                    mult = 0.2
                aw[a] += base_w * mult
                av[a] += 1
        if not aw: return 0
        return max(aw, key=aw.get)

    def solve(self, problem):
        strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2
        user_inputs = [f'{problem} {p}' for p in strategy_prefs[:self.cfg.attempts]]
        elapsed = time.time() - self.notebook_start
        left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        deadline = time.time() + budget
        tasks = [(self.cfg.system_prompt, i) for i in range(self.cfg.attempts)]
        results = []
        valid = []
        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(self._process_attempt, user_inputs[ai], sp, ai, stop, deadline) for sp, ai in tasks]
            for f in as_completed(futs):
                try:
                    r = f.result(); results.append(r)
                    if r['Answer'] is not None: valid.append(r['Answer'])
                    c = Counter(valid).most_common(1)
                    if c and c[0][1] >= self.cfg.early_stop:
                        stop.set()
                        for ff in futs: ff.cancel()
                        break
                except Exception as e:
                    print(f'Future error: {e}')
        self.problems_remaining = max(0, self.problems_remaining - 1)
        return results

    def shutdown(self):
        self.server_process.terminate()
        self.server_process.wait()
        while not self.pool.empty():
            try: self.pool.get_nowait().close()
            except: pass


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading benchmark from {BENCH_PATH}')
    with open(BENCH_PATH) as f:
        bench = json.load(f)

    solver = AIMO3Solver(CFG)

    results_out = []
    baseline_correct = ev_correct = 0

    for i, item in enumerate(bench):
        pid = item['id']
        problem = item['problem']
        expected = item['answer']
        print(f'\n{"="*60}')
        print(f'Problem {i+1}/10: {pid} (expected={expected})')
        print(f'{"="*60}')

        attempt_results = solver.solve(problem)

        ans_baseline = solver._select_baseline(attempt_results)
        ans_ev       = solver._select_ev(attempt_results)

        b_ok = (ans_baseline == expected)
        e_ok = (ans_ev == expected)
        if b_ok: baseline_correct += 1
        if e_ok: ev_correct += 1

        print(f'  Baseline: {ans_baseline} {"✓" if b_ok else "✗"}')
        print(f'  EV:       {ans_ev}       {"✓" if e_ok else "✗"}')
        print(f'  Scores so far -> Baseline: {baseline_correct}/{i+1}, EV: {ev_correct}/{i+1}')

        results_out.append({
            'id': pid,
            'expected': expected,
            'attempts': attempt_results,
            'ans_baseline': ans_baseline,
            'ans_ev': ans_ev,
            'baseline_correct': b_ok,
            'ev_correct': e_ok,
        })

    solver.shutdown()

    print(f'\n{"="*60}')
    print(f'FINAL RESULTS (10 reference problems)')
    print(f'  Baseline (v27 strategy diversity): {baseline_correct}/10')
    print(f'  EV voting (v31):                   {ev_correct}/10')
    print(f'  Delta:                             {ev_correct - baseline_correct:+d}')
    print(f'{"="*60}')

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f'Results saved to {RESULTS_PATH}')


if __name__ == '__main__':
    main()
