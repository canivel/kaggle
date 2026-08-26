# Learning: Two Paths to 44/50

## nihilisticneuralnet (44/50)
- temperature = 0.5
- 5-step system prompt
- context_tokens = 65536, batch_size = 256, gpu = 0.96
- No --max-num-batched-tokens or --max-cudagraph-capture-size in vLLM

## kaanyorgun (44/50)
- temperature = 1.0 (same as base 43/50!)
- IDENTICAL 5-step system prompt as nihilisticneuralnet
- IDENTICAL tool_prompt and preference_prompt
- IDENTICAL other params (context=65536, batch=256, gpu=0.96)
- Also no --max-num-batched-tokens or --max-cudagraph-capture-size

## Key Insight
The 5-step structured prompt is the PROVEN change for 43→44.
Temperature 0.5 may or may not help — one 44/50 uses it, one doesn't.
Both notebooks are character-for-character identical EXCEPT for temperature.

## What Actually Matters (43→44)
1. The 5-step system prompt (UNDERSTAND→EXPLORE→PLAN→EXECUTE→VERIFY)
2. Enhanced tool_prompt (5 use cases, "code supports reasoning")
3. Enhanced preference_prompt (categorized by library, best practices)
4. context_tokens = 65536
5. batch_size = 256
6. gpu_memory = 0.96
7. base_problem_timeout = 300
8. Removing --max-num-batched-tokens and --max-cudagraph-capture-size from vLLM

## What Does NOT Matter
- Temperature (both 0.5 and 1.0 score 44)
- Complex entropy weighting (reverted in 44/50)
