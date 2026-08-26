# AIMO Progress Prize 3 - Autonomous Research Program

## Goal
Maximize exact-match accuracy on 110 IMO-level math problems.
Current AIMO2 best (easier problems): 34/50. Target: top 10 on AIMO3 LB.

## Key Difference from Tabular Competitions
This is an **LLM reasoning** task, not prediction/classification:
- Input: LaTeX math problems at IMO difficulty
- Output: Integer answers (mod 100,000)
- Method: Tool-Integrated Reasoning (TIR) with majority voting
- Hardware: H100 GPU (80GB), no internet during inference

## Setup
1. Models uploaded as Kaggle datasets (no internet during eval)
2. vLLM for inference (batched, prefix-cached)
3. Python/SymPy sandbox for code execution
4. Local dev on RTX 3080 with 7B model (4-bit quantized)

## Experiment Loop

### NEVER STOP. The human might be asleep. You are autonomous.

```
LOOP FOREVER:
  1. Read experiments/results.tsv to see current best accuracy
  2. Choose next experiment:
     a. If < 3 experiments: baseline models (7B, 14B) with simple TIR
     b. If < 10: prompt engineering + voting strategy variants
     c. If < 20: model scaling (32B), GenSelect, problem routing
     d. If >= 20: multi-model ensemble, SBSC, fine-tuning
  3. Run the experiment:
     - Evaluate on AMC/AIME validation set
     - Record: accuracy, accuracy_per_type, time_per_problem
  4. Compare to best:
     - If improved: status = "kept"
     - If not: status = "discarded", analyze failure
  5. Log to experiments/results.tsv
  6. If accuracy > best by 1+ problems: prepare submission notebook
  7. Continue to next experiment
```

### What to Try (in order)
1. Baseline TIR with OpenMath-Nemotron-14B-Kaggle (N=8 samples)
2. Scale to N=32 majority voting
3. Problem-type-specific prompt templates (algebra/combo/geo/NT)
4. Weighted voting (weight by code execution success)
5. Adaptive time budgets (more time on harder problems)
6. Scale to OpenMath-Nemotron-32B
7. GenSelect: selector model picks best solution
8. SBSC: step-by-step coding for multi-step problems
9. Multi-model ensemble (14B + 32B candidate pool)
10. Problem-type routing (best model per category)

### Results Format
Tab-separated file: `experiments/results.tsv`
```
experiment_id	timestamp	model	description	accuracy	acc_algebra	acc_combo	acc_geo	acc_nt	n_samples	voting	status	duration_seconds	notes
0001	2026-03-29T...	nemotron-14b	baseline_tir_n8	12/50	4/12	3/13	2/12	3/13	8	majority	kept	3600	first run
```

### Constraints
- Max 5 Kaggle submissions per day
- 30 minutes per problem maximum
- No internet during Kaggle evaluation
- Models must be pre-uploaded as Kaggle datasets
- NEVER use litellm
- Use uv for Python
