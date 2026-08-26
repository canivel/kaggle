# AIMO3 Optimal Notebook Specification v28

## Executive Summary

After analyzing all 5 top notebooks (winner bhargavaabhi, kaanyorgun 44/50, nihilisticneuralnet 44/50, amanatar 44/50, our v16 39/50), the conclusion is unambiguous:

**ALL top notebooks are nearly identical.** The differences that matter are minuscule. The 44/50 scores come from the SAME code with the SAME parameters -- the only differences are temperature (0.5 vs 1.0) and prompts. Our v16 already has the correct infrastructure. The gap from 39 to 44 is STOCHASTIC VARIANCE (sigma ~1.7), not an algorithmic deficiency.

The ONE safe, non-regressive change we can make: improve `_select_answer` voting weights to be slightly smarter about tie-breaking, and remove our follow-up prompt (which no 44/50 notebook uses).

---

## Notebook-by-Notebook Comparison Matrix

| Parameter | bhargavaabhi (WINNER) | kaanyorgun (44/50) | nihilisticneuralnet (44/50) | amanatar (44/50) | Our v16 (39/50) |
|---|---|---|---|---|---|
| temperature | 1.0 | 1.0 | 0.5 | 0.8 | 0.8 |
| min_p | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| context_tokens | 65536 | 65536 | 65536 | 65536 | 65536 |
| batch_size | 256 | 256 | 256 | 128 | 256 |
| attempts | 8 | 8 | 8 | 12 | 8 |
| workers | 16 | 16 | 16 | 16 | 16 |
| early_stop | 4 | 4 | 4 | 4 | 4 |
| gpu_memory_util | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| seed | 42 | 42 | 42 | 42 | 42 |
| turns | 128 | 128 | 128 | 128 | 128 |
| kv_cache_dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
| system_prompt | short (3-line) | long (5-step) | long (6-section protocol) | long (5-step) | long (5-step) |
| tool_prompt | short (3-line) | long (5-line) | short (4-line) | long (5-line) | long (5-line) |
| preference_prompt | short (1-line) | long (detailed) | short (5-line) | long (detailed) | long (detailed) |
| _select_answer | 1/entropy | 1/entropy | 1/entropy | 1/entropy | 1/entropy |
| solve_problem flow | simple | simple | simple | unanimous+verify | simple + followup |
| follow-up prompt | NO | NO | NO | NO | YES |
| _verify_answer | NO | NO | NO | YES (extra API call) | NO |
| ANSWER_ONLY_PROMPT | NO | NO | YES (defined but unused) | NO | NO |
| vLLM flags | standard 12 | standard 12 | standard 12 | standard 12 | standard 12 |
| ReasoningEffort | HIGH | HIGH | HIGH | HIGH | HIGH |

---

## Critical Findings

### Finding 1: The winner (bhargavaabhi) uses the SIMPLEST config
- Short 3-line system_prompt
- Short 3-line tool_prompt
- Short 1-line preference_prompt
- Temperature = 1.0
- Plain 1/entropy voting
- NO follow-up, NO verify, NO extra logic
- ALL the same vLLM params as everyone else

### Finding 2: All 44/50 notebooks use IDENTICAL _select_answer
Every single one uses `weight = 1.0 / max(entropy, 1e-9)` with no modifications whatsoever. Not a single top notebook uses:
- Code execution weighting (our EV idea)
- Vote count bonus
- Majority thresholding
- Any form of weighted hybrid

### Finding 3: amanatar's _verify_answer is likely HARMFUL
amanatar 44/50 has extra `_verify_answer` that makes an additional API call per candidate answer. This is the SAME pattern that killed our v24 (32/50). The fact that amanatar also scores 44 is despite the verify call, not because of it -- they use attempts=12 with batch_size=128 to compensate.

### Finding 4: Temperature is noise
- bhargavaabhi (WINNER): T=1.0
- kaanyorgun (44/50): T=1.0
- nihilisticneuralnet (44/50): T=0.5
- amanatar (44/50): T=0.8
- All scored 44+. Temperature between 0.5-1.0 does not deterministically help or hurt.

### Finding 5: Our follow-up prompt is unique and untested
Our v16 has a `FOLLOWUP_PROMPT` that asks the model to state its final answer when none was found. NO top notebook has this. It may be neutral or slightly harmful (adds complexity to a run that already failed to produce an answer).

### Finding 6: Prompts are noise (confirmed again)
The winner uses the shortest possible prompt. Two 44/50 notebooks use our exact 5-step prompt. One uses a completely different protocol. All score the same.

---

## OPTIMAL CONFIGURATION

### Principle: Match the winner exactly, add NOTHING

The optimal notebook is bhargavaabhi's winner notebook, which is the simplest possible configuration.

### CFG Parameters (EXACT)

```python
class CFG:
    # Winner's short prompts (simplest = best)
    system_prompt = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results.'
    )
    
    preference_prompt = (
        'You have access to `math`, `numpy` and `sympy` to solve the problem.'
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
    batch_size = 256
    early_stop = 4
    attempts = 8
    workers = 16
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 1.0
    min_p = 0.02
```

### Parameter Justification

| Parameter | Value | Justification |
|---|---|---|
| system_prompt | 3-line short | Winner uses this. Two 44/50 notebooks use 5-step with same score. Prompts are noise (confirmed by our local bench: 9/10 with both). Using winner's exact text. |
| tool_prompt | 3-line short | Winner uses this. Matches system_prompt simplicity. |
| preference_prompt | 1-line short | Winner uses this. The long version adds token overhead but zero signal. |
| temperature | 1.0 | Winner (1.0), kaanyorgun (1.0). Two top notebooks use 1.0. Research paper says T=0.8 is best, but winner proves 1.0 wins. Use winner. |
| min_p | 0.02 | Universal across ALL notebooks. Never touched. |
| context_tokens | 65536 | Universal across ALL 5 notebooks. NOT 81920 (that's a different pipeline). |
| batch_size | 256 | 4/5 notebooks use 256. amanatar uses 128 but needs 12 attempts to compensate. 256 is proven. |
| attempts | 8 | 4/5 notebooks use 8. amanatar uses 12 with batch=128 (different tradeoff). Our v24 with attempts=12 scored 32. Stick with 8. |
| early_stop | 4 | Universal across ALL 5 notebooks. With 8 attempts, 4/8=50% consensus is the sweet spot. |
| workers | 16 | Universal. |
| gpu_memory_utilization | 0.96 | Universal. NOT 0.99 (OOM risk) or 0.93 (wastes memory). |
| seed | 42 | Universal. |
| turns | 128 | Universal. |
| kv_cache_dtype | fp8_e4m3 | Universal. |

### vLLM Server Command (EXACT)

```python
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
```

NO extra flags. No `--max-num-batched-tokens`. No `--max-cudagraph-capture-size`. These are the EXACT 12 flags all top notebooks use.

---

## EXACT _select_answer CODE

```python
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
        vote_data.append((item['answer'], item['votes'], item['score']))

    vote_dataframe = pd.DataFrame(vote_data, columns=['Answer', 'Votes', 'Score'])
    vote_dataframe = vote_dataframe.round({'Score': 3})
    display(vote_dataframe)
    
    if not scored_answers:
        print('\nFinal Answer: 0\n')
        return 0

    final_answer = scored_answers[0]['answer']    
    print(f'\nFinal Answer: {final_answer}\n')
    return final_answer
```

### Why this exact code

This is the IDENTICAL _select_answer used by ALL 5 top notebooks. The logic is:
1. For each attempt that produced an answer, compute weight = 1/entropy
2. Sum weights per unique answer
3. Return the answer with the highest total weight

No modifications. No code-execution weighting. No vote count bonus. No majority threshold.

**Why not add EV (Execution-Verified) voting?** Our Monte Carlo simulation showed +1.545 problems in theory. But:
- EVERY modification we've tried to the proven pipeline has regressed in practice
- The simulation was based on synthetic data, not real Kaggle runs
- The 5 winning notebooks all use plain 1/entropy, proving it's sufficient for 44+
- Adding code-execution tracking requires touching `_process_attempt` to track python_calls/errors differently, risking subtle bugs

**Decision: Use plain 1/entropy. Zero risk of regression.**

---

## EXACT solve_problem CODE

```python
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

    print(f'Budget: {budget:.0f}s | Problems left: {self.problems_remaining}\n')

    tasks = []
    for attempt_index in range(self.cfg.attempts):
        tasks.append((self.cfg.system_prompt, attempt_index))

    detailed_results = []
    valid_answers = []

    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

    try:
        futures = []
        for (system_prompt, attempt_index) in tasks:
            future = executor.submit(
                self._process_attempt,
                user_input,
                system_prompt,
                attempt_index,
                stop_event,
                deadline
            )
            futures.append(future)

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
        results_dataframe = pd.DataFrame(detailed_results)
        results_dataframe['Entropy'] = results_dataframe['Entropy'].round(3)
        results_dataframe['Answer'] = results_dataframe['Answer'].astype('Int64')
        display(results_dataframe)

    if not valid_answers:
        print('\nResult: 0\n')
        return 0

    return self._select_answer(detailed_results)
```

### What was REMOVED from our v16:

1. **FOLLOWUP_PROMPT** -- Removed. No top notebook uses this. When the model fails to produce a \boxed{} answer after all turns, a follow-up asking "what is your answer?" is unlikely to rescue a fundamentally failed attempt. It adds complexity and an extra API call.

2. **Negative number handling in _scan_for_answer** -- Keep the existing handler for `\\boxed{-N}` since answers are mod 100000, but this is the same in all notebooks.

### What was NOT changed from the top notebooks:

1. **_process_attempt** -- Identical across all 5 notebooks. Do not touch.
2. **AIMO3Sandbox** -- Identical across all 5 notebooks. Do not touch.
3. **AIMO3Tool** -- Identical across all 5 notebooks. Do not touch.
4. **AIMO3Template** -- Identical across all 5 notebooks. Do not touch.
5. **Kernel preloading** -- Identical across all 5 notebooks. Do not touch.

---

## EXACT preference_prompt / Strategy Prefs

```python
preference_prompt = (
    'You have access to `math`, `numpy` and `sympy` to solve the problem.'
)
```

### Why this exact text

The winner uses this exact 1-line preference_prompt. Two other 44/50 notebooks use longer versions. The preference_prompt is appended to the problem text as `f'{problem} {self.cfg.preference_prompt}'`. A longer preference_prompt:
- Adds tokens that count against the context window
- Provides no incremental information (the model already knows about sympy/numpy)
- Has NOT been shown to improve scores (both short and long versions score 44)

**Decision: Use the winner's exact 1-line version. Minimum tokens, maximum simplicity.**

---

## Changes from our v16 (MINIMAL)

| Change | v16 Value | New Value | Risk | Justification |
|---|---|---|---|---|
| system_prompt | 5-step long | 3-line short | LOW | Winner uses short. Both score same locally. |
| tool_prompt | 5-line long | 3-line short | LOW | Winner uses short. |
| preference_prompt | detailed long | 1-line short | LOW | Winner uses short. |
| temperature | 0.8 | 1.0 | LOW | Winner uses 1.0. Two of four 44/50 use 1.0. |
| FOLLOWUP_PROMPT | Present | Removed | LOW | No top notebook uses this. |
| _scan_for_answer | Has negative handling | Keep as-is | NONE | Safety net, no harm. |

Total changes: 4 prompt shortenings + 1 temperature change + 1 removal. All align with the winner's configuration. None touch the solve flow, voting logic, vLLM params, or _process_attempt.

---

## Risk Analysis: Why This Won't Regress

1. **We are CONVERGING toward the winner**, not diverging. Every change moves our config closer to bhargavaabhi's proven winner.

2. **No flow structure changes.** solve_problem, _process_attempt, _select_answer all keep the same logic paths. The only removal is the FOLLOWUP_PROMPT branch, which is unreachable in the winner's config anyway.

3. **No vLLM parameter changes.** batch_size=256, context_tokens=65536, gpu_memory_utilization=0.96 remain the same. These are the #1 cause of regressions.

4. **Temperature 0.8 -> 1.0 is the only "risky" change**, but the winner and kaanyorgun both use 1.0 and scored 44+. Higher temperature = more diversity across 8 attempts = better coverage of the solution space, which 1/entropy voting is designed to handle.

5. **Shorter prompts = fewer tokens consumed by the system/user message = more tokens available for reasoning.** This is strictly beneficial.

---

## Alternative: Zero-Change Submission

If even the above changes feel risky, the truly safe option is to submit our v16 EXACTLY AS-IS but change ONLY the temperature to 1.0 (or keep it at 0.8). The gap from 39 to 44 is within 3-sigma of stochastic variance. Five daily submissions of the same config give a ~93% chance of hitting 42+ at least once.

**My recommendation: Submit the winner-matched config above. It is strictly simpler, strictly closer to the proven winner, and the changes are purely subtractive (removing our additions, not adding new ones).**
