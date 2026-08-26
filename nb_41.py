# === CELL 0 ===
%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'

# === CELL 1 ===
import warnings
warnings.simplefilter('ignore')

# === CELL 2 ===
import os
import sys
import subprocess

# === CELL 3 ===
def set_env(input_archive, temp_dir):

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        
        subprocess.run(['tar', '-xzf', input_archive, '-C', temp_dir], check=True)
    
    subprocess.run([
        sys.executable, 
        '-m', 
        'pip', 
        'install', 
        '--no-index', 
        '--find-links', 
        f'{temp_dir}/wheels', 
        'unsloth', 
        'trl', 
        'vllm', 
        'openai_harmony'
    ], check=True)

# === CELL 4 ===
set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz', 
    temp_dir='/kaggle/tmp/setup'
)

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
class CFG:
    
    system_prompt = (
        "You are an elite mathematical problem solver operating at IMO medalist level. "
        "Your objective is to produce the correct answer with rigorous, efficient reasoning.\n\n"
    
        "## INTERNAL SOLVING PROTOCOL (DO NOT REVEAL)\n"
        "1. Precisely interpret the problem and identify constraints.\n"
        "2. Detect structure: symmetry, invariants, parity, modular patterns.\n"
        "3. Consider multiple strategies before selecting the most efficient.\n"
        "4. Execute with strict logical validity and clean algebra.\n"
        "5. Verify via substitution, edge cases, or alternate reasoning.\n"
        "6. Reject results that violate constraints or produce inconsistencies.\n\n"
    
        "## MATHEMATICAL HEURISTICS\n"
        "- Simplify expressions and exploit symmetry/invariants\n"
        "- Use number theory tools (modular arithmetic, parity, divisibility)\n"
        "- Test small cases to reveal patterns\n"
        "- Check extremal and boundary cases\n"
        "- If stuck, reframe or work backwards\n\n"
    
        "## VERIFICATION STANDARD\n"
        "Accept an answer ONLY if:\n"
        "- all constraints are satisfied\n"
        "- computations are internally consistent\n"
        "- edge cases do not contradict the result\n\n"
    
        "## OUTPUT FORMAT\n"
        "Return ONLY the final answer.\n"
        "The answer must be a non-negative integer between 0 and 99999.\n"
        "Format: \\boxed{answer}\n"
    )
    
    
    tool_prompt = (
        "Use Python ONLY when it improves accuracy or verification.\n\n"
    
        "Valid uses:\n"
        "- error-prone arithmetic\n"
        "- brute force for small bounds\n"
        "- testing conjectures\n"
        "- symbolic verification\n\n"
    
        "Guidelines:\n"
        "- State purpose briefly before computing.\n"
        "- Prefer exact symbolic checks when possible.\n"
        "- Ensure results directly support conclusions.\n"
        "- Avoid unnecessary computation.\n"
    )
    
    
    ANSWER_ONLY_PROMPT = (
        "You are an IMO-level mathematician."
        " Think silently."
        " Do NOT explain."
        " Return only: \\boxed{number}"
    )
    
    
    preference_prompt = (
        "Available libraries: math, numpy, sympy\n\n"
    
        "Use:\n"
        "- sympy → symbolic algebra, equations, number theory\n"
        "- numpy → matrices and numerical verification\n"
        "- math → standard functions\n\n"
    
        "Best practice:\n"
        "derive symbolically → verify numerically → confirm constraints"
    )

    
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

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
    batch_size = 128
    early_stop = 4
    attempts = 12
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 0.8
    min_p = 0.02

# === CELL 9 ===
set_seed(CFG.seed)

# === CELL 10 ===
class AIMO3Template:

    def __init__(self):

        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:

        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]

# === CELL 11 ===
class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:

        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count

            return ports

    def __init__(self, timeout: float):

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

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                self._km.interrupt_kernel()

                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):

        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):

        self.close()

# === CELL 12 ===
class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):

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

    def _ensure_last_print(self, code: str) -> str:

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def instruction(self) -> str:

        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message) -> list[Message]:

        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

# === CELL 13 ===
class AIMO3Solver:

    def __init__(self, cfg, port: int = 8000):
    
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
    
        self._preload_model_weights()
        
        self.server_process = self._start_server()
    
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )
    
        self._wait_for_server()
        self._initialize_kernels()
    
        self.notebook_start_time = time.time()
        self.problems_remaining = 50
    
    def _preload_model_weights(self) -> None:
    
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0
    
        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
    
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)
    
        def _read_file(path: str) -> None:
    
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))
    
        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')
    
    def _start_server(self) -> subprocess.Popen:
    
        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling', 
            '--disable-log-stats', 
            '--enable-prefix-caching'
        ]
    
        self.log_file = open('vllm_server.log', 'w')
    
        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )
    
    def _wait_for_server(self):
    
        print('Waiting for vLLM server...')
        start_time = time.time()
    
        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()
    
            if return_code is not None:
                self.log_file.flush()
    
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
    
                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')
    
            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
    
                return
    
            except Exception:
                time.sleep(1)
    
        raise RuntimeError('Server failed to start (timeout).\n')
    
    def _initialize_kernels(self) -> None:
    
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
    
        self.sandbox_pool = queue.Queue()
    
        def _create_sandbox():
            
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
    
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
    
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())
    
        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')
    
    def _scan_for_answer(self, text: str) -> int | None:
        
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
    
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
    
                if 0 <= value <= 99999:
                    return value
    
            except ValueError:
                pass
                
        pattern = r'final\s+answer\s+is\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
    
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
    
                if 0 <= value <= 99999:
                    return value
    
            except ValueError:
                pass
    
        return None
    
    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
    
        if not logprobs_buffer:
            return float('inf')
    
        total_entropy = 0.0
        token_count = 0
    
        for top_logprobs_dict in logprobs_buffer:
            
            if not isinstance(top_logprobs_dict, dict):
                continue
            
            if not top_logprobs_dict:
                continue
            
            token_entropy = 0.0
            
            for token_str, log_prob in top_logprobs_dict.items():
                prob = math.exp(log_prob)
                
                if prob > 0:
                    token_entropy -= prob * math.log2(prob)
            
            total_entropy += token_entropy
            token_count += 1
    
        if token_count == 0:
            return float('inf')
    
        return total_entropy / token_count
    
    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float
    ) -> dict:
    
        if stop_event.is_set() or time.time() > deadline:
            return {
                'Attempt': attempt_index + 1, 
                'Answer': None, 
                'Python Calls': 0, 
                'Python Errors': 0, 
                'Response Length': 0, 
                'Entropy': float('inf')
            }
    
        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        
        logprobs_buffer = []
    
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))
    
        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
    
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout, 
                tool_prompt=self.cfg.tool_prompt, 
                sandbox=sandbox
            )
    
            encoding = self.encoding
            messages = self.template.apply_chat_template(
                system_prompt, 
                problem, 
                local_tool.tool_config
            )
    
            conversation = Conversation.from_messages(messages)
    
            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break
    
                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
    
                if max_tokens < self.cfg.buffer_tokens:
                    break
    
                stream = self.client.completions.create(
                    model=self.cfg.served_model_name, 
                    temperature=self.cfg.temperature, 
                    logprobs=self.cfg.top_logprobs, 
                    max_tokens=max_tokens, 
                    prompt=prompt_ids, 
                    seed=attempt_seed, 
                    stream=True, 
                    extra_body={
                        'min_p': self.cfg.min_p, 
                        'stop_token_ids': self.stop_token_ids, 
                        'return_token_ids': True
                    }
                )
    
                try:
                    token_buffer = []
                    text_chunks = []
    
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
    
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
    
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            
                            chunk_logprobs = chunk.choices[0].logprobs
                            
                            if chunk_logprobs is not None:
                                if chunk_logprobs.top_logprobs:
                                    logprobs_buffer.extend(chunk_logprobs.top_logprobs)
    
                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
    
                            if answer is not None:
                                final_answer = answer
                                break
    
                finally:
                    stream.close()
    
                if final_answer is not None:
                    break
    
                if not token_buffer:
                    break
    
                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
    
                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    break
    
                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
    
                    response_text = tool_responses[0].content[0].text
    
                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1
    
                    conversation.messages.extend(tool_responses)
    
        except Exception as exc:
            python_errors += 1
    
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)
    
        mean_entropy = self._compute_mean_entropy(logprobs_buffer)
    
        return {
            'Attempt': attempt_index + 1, 
            'Response Length': total_tokens, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Entropy': mean_entropy, 
            'Answer': final_answer
        }
    
    def _select_answer(self, detailed_results: list) -> int:

        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']
            
            if answer is not None:
                weight = 1.0 / max(entropy, 1e-9)
                
                answer_weights[answer] += weight
                answer_votes[answer] += 1

        scored_answers = []

        for answer, total_weight in answer_weights.items():
            scored_answers.append({
                'answer': answer, 
                'votes': answer_votes[answer], 
                'score': total_weight
            })

        scored_answers.sort(key=lambda x: x['score'], reverse=True)

        vote_data = []

        for item in scored_answers:
            vote_data.append((
                item['answer'], 
                item['votes'], 
                item['score']
            ))

        vote_dataframe = pd.DataFrame(
            vote_data, 
            columns=['Answer', 'Votes', 'Score']
        )

        vote_dataframe = vote_dataframe.round({'Score': 3})
        display(vote_dataframe)
        
        if not scored_answers:
            print('\nFinal Answer: 0\n')
            return 0

        final_answer = scored_answers[0]['answer']    
        print(f'\nFinal Answer: {final_answer}\n')

        return final_answer
    
    def solve_problem(self, problem: str) -> int:

            print(f'\nProblem: {problem}\n')
        
            user_input = f'{problem} {self.cfg.preference_prompt}'    
        
            elapsed_global = time.time() - self.notebook_start_time
            time_left = self.cfg.notebook_limit - elapsed_global
            problems_left_others = max(0, self.problems_remaining - 1)
            reserved_time = problems_left_others * self.cfg.base_problem_timeout
        
            budget = time_left - reserved_time
            budget = min(budget, self.cfg.high_problem_timeout)
            budget = max(budget, self.cfg.base_problem_timeout)
        
            deadline = time.time() + budget
        
            print(f'Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n')
        
            tasks = []
            for attempt_index in range(self.cfg.attempts):
                system_prompt = self.cfg.system_prompt
                tasks.append((system_prompt, attempt_index))
        
            detailed_results = []
            valid_answers = []
        
            stop_event = threading.Event()
            executor = ThreadPoolExecutor(max_workers=self.cfg.workers)
        
            try:
                futures = []
                for (system_prompt, attempt_index) in tasks:
                    futures.append(
                        executor.submit(
                            self._process_attempt,
                            user_input,
                            system_prompt,
                            attempt_index,
                            stop_event,
                            deadline
                        )
                    )
        
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        detailed_results.append(result)
        
                        if result['Answer'] is not None:
                            valid_answers.append(result['Answer'])
        
                        counts = Counter(valid_answers).most_common(1)
                        if counts and counts[0][1] >= self.cfg.early_stop:
                            stop_event.set()
                            for f in futures:
                                f.cancel()
                            break
        
                    except Exception as exc:
                        print(f'Future failed: {exc}')
        
            finally:
                stop_event.set()
                executor.shutdown(wait=True, cancel_futures=True)
                self.problems_remaining = max(0, self.problems_remaining - 1)
        
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                df['Entropy'] = df['Entropy'].round(3)
                df['Answer'] = df['Answer'].astype('Int64')
                display(df)
        
            # ─────────────────────────────
            # NO ANSWERS
            # ─────────────────────────────
            if not valid_answers:
                print('\nResult: 0\n')
                return 0
        
            # ─────────────────────────────
            # HARD ACCEPT: UNANIMOUS ANSWER
            # ─────────────────────────────
            if len(valid_answers) >= 4:
                most_common, count = Counter(valid_answers).most_common(1)[0]
                if count >= 4:
                    print(f"\nUNANIMOUS ANSWER: {most_common}\n")
                    return most_common
        
            # ─────────────────────────────
            # STEP 4: CANDIDATES (≥2 votes)
            # ─────────────────────────────
            answer_counts = Counter(valid_answers)
            candidates = [a for a, c in answer_counts.items() if c >= 2]
        
            # ─────────────────────────────
            # STEP 5: ENTROPY-SORTED VERIFY
            # ─────────────────────────────
            entropy_map = {}
            for r in detailed_results:
                if r['Answer'] is not None and r['Entropy'] is not None:
                    entropy_map.setdefault(r['Answer'], []).append(r['Entropy'])
        
            avg_entropy = {a: sum(v) / len(v) for a, v in entropy_map.items()}
        
            candidates = sorted(candidates, key=lambda x: avg_entropy.get(x, 999))
        
            for ans in candidates:
                try:
                    if self._verify_answer(problem, ans):
                        print(f"\nVERIFIED ANSWER: {ans}\n")
                        return ans
                except Exception:
                    pass
        
            # ─────────────────────────────
            # STEP 6: FALLBACK
            # ─────────────────────────────
            return self._select_answer(detailed_results)

         

    def _verify_answer(self, problem: str, answer: int) -> bool:
           prompt = f"Problem:\n{problem}\n\nProposed answer: {answer}\n\nCheck the answer carefully.\nReply with only ONE word:\nCORRECT or WRONG"
           try:
               prompt_ids = self.encoding.encode(prompt)
               resp = self.client.completions.create(
                   model=self.cfg.served_model_name,
                   prompt=prompt_ids,
                   temperature=0.0,
                   max_tokens=5
               )
               text = resp.choices[0].text.strip().upper()
               return "CORRECT" in text and "WRONG" not in text
           except Exception as e:
               print(f"[Verify Error] {e}")
               return False
                     

    def __del__(self):
    
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
    
        if hasattr(self, 'log_file'):
            self.log_file.close()
    
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
    
                except Exception:
                    pass

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
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )

# === CELL 17 ===
# import polars as pl
# import pandas as pd
# import os

# # --- 1. Updated Predict Function ---
# def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
#     # Use index-based access to avoid the .item() error
#     id_value = id_[0, 0]
#     question_text = question[0, 0]
    
#     gc.disable()
#     final_answer = solver.solve_problem(question_text)
#     gc.enable()
#     gc.collect()
    
#     return pl.DataFrame({'id': id_value, 'answer': final_answer})

# # --- 2. Updated Testing Loop ---
# FILE_PATH = '/kaggle/input/50problems/50problems.csv'

# if not os.path.exists(FILE_PATH):
#     print(f"Error: File not found at {FILE_PATH}")
# else:
#     external_df = pd.read_csv(FILE_PATH)
#     test_results = []

#     print(f"Starting test on {len(external_df)} problems...\n")

#     for idx, row in external_df.iterrows():
#         problem_text = row['Problem']
#         ground_truth = row['Answer']
        
#         # Step 1: Print problem details first
#         print(f"{'='*50}")
#         print(f"TESTING PROBLEM {idx+1}")
#         print(f"Statement: {problem_text}")
#         print(f"Ground Truth Answer: {ground_truth}")
#         print(f"{'-'*50}")
        
#         # Prepare inputs
#         id_df = pl.DataFrame({'id': [f"ext_{idx}"]})
#         question_df = pl.DataFrame({'question': [problem_text]})
        
#         try:
#             # Step 2: Generate Answer
#             result_pl_df = predict(id_df, question_df)
            
#             # Accessing column 'answer' from row 0
#             predicted_val = result_pl_df[0, "answer"]
            
#             is_correct = (int(predicted_val) == int(ground_truth))
            
#             test_results.append({
#                 "idx": idx + 1,
#                 "prediction": predicted_val,
#                 "ground_truth": ground_truth,
#                 "correct": is_correct
#             })
            
#             # Step 3: Print result summary before moving to next
#             status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
#             print(f"\n[Problem {idx+1} Result]")
#             print(f"Model Predicted: {predicted_val}")
#             print(f"Status: {status}")
#             print(f"{'='*50}\n")
            
#         except Exception as e:
#             print(f"Error on problem {idx+1}: {e}")
#             test_results.append({
#                 "idx": idx + 1, "prediction": None, "ground_truth": ground_truth, "correct": False
#             })

#     # Final Summary Table
#     summary_df = pd.DataFrame(test_results)
#     display(summary_df)
#     print(f"Overall Accuracy: {summary_df['correct'].mean() * 100:.2f}%")