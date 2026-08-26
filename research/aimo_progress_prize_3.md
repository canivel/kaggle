# AI Mathematical Olympiad - Progress Prize 3: Research Report

**Research date**: 2026-03-29
**Competition URL**: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3

---

## 1. What the Competition Is About

The AI Mathematical Olympiad (AIMO) Progress Prize 3 challenges teams to build open-source AI systems that can solve olympiad-level mathematics problems. This is the third in a series of progress prizes toward a $10M grand prize for an AI that can win an IMO gold medal.

**Problem difficulty**: IMO and near-IMO level (a step up from AIMO2's National Olympiad level).

**Problem set**: 110 original math problems spanning:
- Algebra
- Combinatorics
- Geometry
- Number theory

Problems are written in LaTeX. All problems are original (created specifically for the competition) to prevent data contamination. A reference subset of 10 problems is publicly available; the remaining 100 are split between a public leaderboard (~50) and a private leaderboard (~50).

**Key design change from AIMO2**: Answers now require five digits instead of three, making random guessing virtually impossible. This forces genuine mathematical reasoning.

**Grand Vision**: Winners will showcase at AI Day at the 2026 IMO in Shanghai, China.

---

## 2. Evaluation Metric

- **Primary metric**: Accuracy -- fraction of predicted answers that exactly match the ground-truth integer answers.
- Answers are integers. In AIMO2 the answers were reduced mod 1000 (last 3 digits); AIMO3 extends this to 5 digits (last 5 digits / mod 100000), making the space much larger.
- Scoring is binary per problem: correct or incorrect.
- Test problems are served one-by-one in random order via a Python evaluation API.

---

## 3. Submission Format

This is a **Code Competition** (Kaggle notebook-based submission). Key requirements:

- Submit a Python notebook (Kaggle kernel) containing inference code.
- The notebook is called via a Kaggle evaluation API:
  ```python
  env, iter_test = get_kaggle_env(config)
  # For each problem:
  env.predict(submission)  # submit integer answer
  ```
- The prediction server must be called within **15 minutes** of the notebook starting.
- Each prediction (per problem) must be returned within **30 minutes**.
- Problems are served one at a time in random order.
- The answer field should contain an integer (the final answer, reduced to the specified digit count).

**Data columns** (from prior AIMO competitions and community notebooks):
- `problem_id` -- unique identifier
- `question` -- LaTeX-formatted math problem text
- Answer is submitted as an integer via the evaluation API (not as a CSV row).

**Reference files** available locally in the notebook:
- `reference.csv` -- the 10 public reference problems with known answers
- `test.csv` -- the actual test problems (no answers)
- `sample_submission.csv` -- submission format template

---

## 4. Deadlines

| Event | Date |
|-------|------|
| Competition launch | November 20, 2025 |
| Entry deadline | April 8, 2026 |
| Team merger deadline | April 8, 2026 |
| Final submission deadline | April 15, 2026 |

---

## 5. Prize Structure

**Main prize pool**: $2,207,152

Prizes are distributed by final leaderboard rank (prize-doubling structure from AIMO2 as reference):

| Placement | Prize (AIMO2 reference) |
|-----------|------------------------|
| 1st | ~$524K |
| 2nd | ~$262K |
| 3rd | ~$131K |
| 4th | ~$65K |
| 5th | ~$32K |

*(Exact AIMO3 per-rank amounts not confirmed; total pool is $2.207M)*

**Grand threshold prize**: A minimum of ~$1.5M+ goes to the first team achieving a score equivalent to IMO gold medal performance (exact threshold TBD for AIMO3).

**Extra Prizes** (new in AIMO3, total ~$110,000):

1. **Longest Leader Prize** -- team whose model stays on top of the public leaderboard the longest.
2. **Write-up Prizes** -- best technical explanation of approach (encourages knowledge sharing).
3. **MathCorpus Prize** -- awarded for publishing novel, high-quality math datasets.
4. **Hardest Problem Prize** -- best model that solves the single problem least solved by all AIMO3 models.

**Requirement**: Top-5 teams must publicly release their models (open-source requirement).

---

## 6. Hardware and Compute Constraints

### Submission (Inference) Environment

- **GPU**: H100 GPU(s) available on Kaggle for submission notebooks (major upgrade from AIMO2's 4x L4 GPUs).
- AIMO3 provides roughly **double the compute of AIMO2**.
- H100s are used for both training and testing (same architecture end-to-end).
- **No internet access** during notebook execution (standard Kaggle code competition rule; models must be pre-loaded or attached as datasets).

### Training (Fine-tuning) Environment

- **Up to 128 H100 GPUs** available to selected participants for fine-tuning via partnerships with Fields Model Initiative and Thinking Machines (Tinker credits).
- This is available to qualifying teams, not all participants.

### Historical context (AIMO2 for comparison):
- AIMO2 used 4x L4 GPUs with a 5-hour inference window.
- AIMO1 used 2x T4 GPUs with a 9-hour limit for 50 problems.
- AIMO3's exact total time limit is not yet publicly confirmed, but with H100s and 110 problems, expect at least 5-9 hours.

---

## 7. Key Community Discussions and Winning Strategies

### What Won AIMO1 (NuminaMath / Project Numina, 29/50)

**Model**: DeepSeekMath-Base 7B fine-tuned in two stages:
1. **Chain of Thought (CoT)** fine-tuning on hundreds of thousands of competition math problems.
2. **Tool-Integrated Reasoning (TIR)** -- model generates Python code, executes it, feeds output back, self-corrects.

**Inference**: Self-Consistency TIR (SC-TIR):
- Generate N=48 completions per problem.
- Execute code blocks, feed tracebacks back for self-correction.
- Repeat M=4 times.
- **Majority vote** on final answers.

**Hardware workaround**: 8-bit quantization (AutoGPTQ) to fit within 2x T4 VRAM, plus vLLM for batched inference.

**Open-source release**: `AI-MO/NuminaMath-7B-TIR` on Hugging Face.

### What Won AIMO2 (NemoSkills / NVIDIA, 34/50)

**Three core pillars**:
1. **OpenMathReasoning dataset**: 540K unique high-quality math problems with 3.2M long-reasoning solutions; 1.7M Tool-Integrated Reasoning solutions.
2. **TIR with long-reasoning models**: Iterative training, generation, and quality filtering to integrate code execution.
3. **GenSelect (Generative Solution Selection)**: Instead of majority voting, trained a model to pick the best solution from many candidates -- significantly outperforms majority voting.

**Result**: 34/50 on Kaggle (4x L4 GPUs, 5 hours); 35/50 on unrestricted 8x H100 hardware.

**Models released**: OpenMath-Nemotron-32B and 14B-Kaggle under commercially permissive license.

### Key Techniques That Work

1. **Tool-Integrated Reasoning (TIR)** -- models that write Python/SymPy code and execute it, then reason from the output. This is the single biggest technique jump (doubles performance over pure CoT).

2. **Self-consistency / majority voting** -- sample many solutions (N=32-64), majority vote on final integer answer. Critical for hard problems where single-pass accuracy is low.

3. **Generative Solution Selection (GenSelect)** -- train a "selector" model to pick the best candidate solution rather than using simple majority vote. Outperforms majority voting, especially on IMO-level problems.

4. **Step-By-Step Coding (SBSC)** -- decompose a problem into sub-tasks, generate a program for each, integrate outputs. Reported +6-12% absolute accuracy improvement over state-of-the-art.

5. **Long-chain reasoning** -- use models trained for extended reasoning traces (DeepSeek-R1 style, o1 style). The longer the reasoning chain, the better for hard problems.

6. **Problem classification** -- automatically categorize problems (number theory, algebra, combinatorics, geometry) to route to specialized strategies or prompts.

7. **Data quality over model size** -- a well-trained 7B model beats poorly-fine-tuned 70B+ models. The OpenMathReasoning dataset (540K problems, 3.2M solutions) is the key differentiator.

8. **Self-correction loops** -- feed execution tracebacks and intermediate errors back into the model context for iterative refinement.

---

## 8. Candidate Models for AIMO3

Based on state-of-the-art as of early 2026:

| Model | Size | Notes |
|-------|------|-------|
| OpenMath-Nemotron-32B | 32B | AIMO2 winner's model, open-source |
| OpenMath-Nemotron-14B-Kaggle | 14B | Kaggle-optimized version |
| DeepSeek-R1 family | 7B-671B | Strong reasoning, TIR-compatible |
| Qwen3-Next | ~70B+ | Mentioned as a target for H100 scale |
| GPT-OSS-120B | 120B | Mentioned as runnable on H100 at AIMO3 scale |
| NuminaMath-7B-TIR | 7B | AIMO1 winner, still a strong baseline |

The H100 upgrade (vs L4 in AIMO2) means competitors can now run 30B-70B parameter models at inference time, a major shift from AIMO1/2 where 7B was the practical ceiling.

---

## 9. Commercial vs Open-Source Gap

As of September 2025:
- **OpenAI o3-preview**: 47-50/50 on AIMO2 problems (high-compute setting).
- **Best open-source (NemoSkills)**: 35/50 on AIMO2 problems.
- Gap is closing but still ~12 problems behind.

AIMO3 is harder (IMO-level), so top scores will likely start lower and scale as teams improve over the competition period.

---

## 10. Common Pitfalls to Avoid

1. **Model too large for GPU memory** -- with H100 (80GB), models up to ~70B at FP16 fit in one GPU; for 120B+ need multi-GPU or quantization.

2. **Insufficient sampling** -- with only 1 sample per problem, performance is low. Budget for N=32+ samples, especially on hard problems.

3. **Pure text CoT without code execution** -- consistently underperforms TIR approaches on olympiad math.

4. **Majority voting at small N** -- majority vote needs N>=16 to be meaningful; small N loses to GenSelect.

5. **Overfitting to public leaderboard** -- public LB is only ~50/110 problems; monitor with private validation sets (AMC, AIME benchmarks).

6. **Large models without quantization** -- at 8-bit or 4-bit quantization, even 70B models fit and run fast enough for the inference budget.

7. **Not exporting successful solutions** -- save correct solution traces to JSONL for iterative fine-tuning.

8. **Ignoring the five-digit answer constraint** -- unlike AIMO2 (mod 1000), AIMO3 answers are in a larger space. Answer extraction and normalization logic must be updated.

---

## 11. Recommended Strategy for Participation

**Priority order based on expected impact**:

1. **Start with OpenMath-Nemotron-14B-Kaggle** (AIMO2 winner's Kaggle-optimized model) as baseline -- it's already proven on this exact competition format.

2. **Implement TIR + majority voting** (N=32+) as the inference engine -- this is table stakes for competitive performance.

3. **Upgrade to GenSelect** over majority voting -- NVIDIA's paper shows this is the next significant gain beyond simple majority vote.

4. **Scale up the model** -- H100 access means 32B models are feasible; try OpenMath-Nemotron-32B.

5. **Fine-tune on OpenMathReasoning dataset** if compute budget allows (128 H100 GPUs for top teams).

6. **Problem-specific strategies**: Geometry often benefits from coordinate bashing via code; combinatorics from brute-force enumeration; number theory from modular arithmetic libraries.

7. **Build a private validation suite**: Use AMC/AIME problems as a proxy benchmark to avoid leaderboard overfitting.

---

## 12. Sources

- [AIMO Prize - Third $2.2M Prize Launched](https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [AIMO2 Winning Solution Paper (arXiv)](https://arxiv.org/abs/2504.16891)
- [How NuminaMath Won AIMO1 (Hugging Face Blog)](https://huggingface.co/blog/winning-aimo-progress-prize)
- [NemoSkills 1st Place Solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills)
- [The Gap Is Shrinking (AIMO Prize Blog)](https://aimoprize.com/updates/2025-09-05-the-gap-is-shrinking)
- [Second Progress Prize Closed](https://aimoprize.com/updates/2025-04-15-second-progress-prize-closed)
- [AIMO3 GPU Submission Notebook](https://www.kaggle.com/code/solokop/aimo3-gpu-submission)
- [AIMO3 Submission Demo](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo)
- [GitHub: project-numina/aimo-progress-prize](https://github.com/project-numina/aimo-progress-prize)
- [GitHub: AIMO3-Kaggle (abonvalle)](https://github.com/abonvalle/AIMO3-Kaggle)
- [NVIDIA OpenMath-Nemotron Release](https://www.marktechpost.com/2025/04/24/nvidia-ai-releases-openmath-nemotron-32b-and-14b-kaggle-advanced-ai-models-for-mathematical-reasoning-that-secured-first-place-in-the-aimo-2-competition-and-set-new-benchmark-records/)
