# Meta-Harness: End-to-End Optimization of Model Harnesses

## Paper Details

- **Title:** Meta-Harness: End-to-End Optimization of Model Harnesses
- **ArXiv:** https://arxiv.org/abs/2603.28052
- **Submitted:** March 30, 2026
- **Project page:** https://yoonholee.com/meta-harness/
- **GitHub artifact (TBench2):** https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact

## Authors and Affiliations

- Yoonho Lee (Stanford)
- Roshen Nair (Stanford)
- Qizheng Zhang (Stanford)
- Chelsea Finn (Stanford)
- Kangwook Lee (KRAFTON)
- Omar Khattab (MIT)

## Abstract (Verbatim)

"The performance of large language model (LLM) systems depends not only on model weights, but also on their harness: the code that determines what information to store, retrieve, and present to the model. Yet harnesses are still designed largely by hand, and existing text optimizers are poorly matched to this setting because they compress feedback too aggressively. We introduce Meta-Harness, an outer-loop system that searches over harness code for LLM applications. It uses an agentic proposer that accesses the source code, scores, and execution traces of all prior candidates through a filesystem. On online text classification, Meta-Harness improves over a state-of-the-art context management system by 7.7 points while using 4x fewer context tokens. On retrieval-augmented math reasoning, a single discovered harness improves accuracy on 200 IMO-level problems by 4.7 points on average across five held-out models. On agentic coding, discovered harnesses surpass the best hand-engineered baselines on TerminalBench-2."

## Core Idea

Meta-Harness is an **outer-loop optimizer** that searches over harness *code* rather than model weights or text prompts. A "harness" is the scaffolding code that controls what information gets stored, retrieved, and presented to the LLM -- system prompts, tool definitions, retrieval logic, context management.

The key differentiator from prior optimizers (like DSPy, TextGrad): instead of compressing feedback into a short text summary, the Meta-Harness proposer is given **unrestricted filesystem access to the full source code, scores, and execution traces of every prior candidate** -- up to 10M tokens of diagnostic context per step vs. 26K for prior methods. This lets the proposer trace failures back to specific harness decisions rather than guessing from aggregate scores.

## Three Benchmarks

### 1. Online Text Classification
- +7.7 points over ACE (state-of-the-art context management) using 4x fewer tokens
- +16 points on LawBench (215-class dataset) with GPT-OSS-120B

### 2. Retrieval-Augmented Math Reasoning (IMO-level) -- AIMO Connection

**Setup:**
- 200 held-out IMO-level problems as test set (from multiple sources)
- 250 search/training problems for harness optimization
- Corpus of ~500,000 deduplicated, decontaminated problems for retrieval
- Evaluated on 5 different held-out models (never seen during optimization)

**Key insight:** Math solutions share reusable proof patterns, so previous reasoning traces are exploitable at inference time -- but the right *retrieval policy* matters more than retrieval itself.

**What the discovered harness does:**
A "lexical router" assigns each problem to one of four routes: combinatorics, geometry, number theory, or default. Each route uses a different BM25 retrieval strategy:
- Combinatorics: fetch 20 candidates, deduplicate and rerank to 8
- Geometry: 1 fixed reference + 2 BM25 neighbors
- Number theory: fetch 12 with technique-based reranking
- Default: adaptive retrieval of 10

**Results (Table 6 -- accuracy on 200 IMO-level problems):**

| Method                 | GPT-5.4-nano | GPT-5.4-mini | Gemini-3.1-Flash-Lite | Gemini-3-Flash | GPT-OSS-20B | Average |
|------------------------|--------------|--------------|----------------------|----------------|-------------|---------|
| No Retriever           | 23.0         | 28.8         | 28.6                 | 42.6           | 47.6        | 34.1    |
| Dense Retrieval (k=1)  | 27.1         | 24.5         | 31.3                 | 42.3           | 46.9        | 34.4    |
| Dense Retrieval (k=5)  | 31.1         | 28.3         | 37.1                 | 47.2           | 46.7        | 38.1    |
| Random Few-shot        | 23.1         | 24.5         | 31.0                 | 40.4           | 41.8        | 32.2    |
| BM25 Retrieval         | 30.2         | 29.2         | 32.8                 | 46.6           | 48.9        | 37.5    |
| **Meta-Harness**       | **31.7**     | **30.4**     | **34.9**             | **46.3**       | **50.6**    | **38.8** |

+4.7 points over no-retrieval baseline, +1.3 over BM25.
Crucially, the harness was discovered on 250 problems but generalizes to 5 unseen models.

### 3. Agentic Coding (TerminalBench-2)
- Discovered harnesses surpass best hand-engineered baselines
- The meta-harness-tbench2-artifact GitHub repo is the artifact for this: 76.4% with Claude Opus 4.6

## Connection to AIMO

The paper does **not** specifically target the Kaggle AIMO competition. The "AIMO benchmark" connection is:

1. The math reasoning benchmark uses **200 IMO-level problems** -- these are IMO-difficulty but not necessarily the exact Kaggle AIMO dataset.
2. The approach (automated harness discovery for retrieval-augmented math reasoning) is **directly applicable** to AIMO: replace their corpus with AIMO training problems, let Meta-Harness discover the optimal retrieval strategy.
3. The paper demonstrates the key insight: **hand-engineered retrieval consistently underperforms Meta-Harness-discovered retrieval** across all 5 models tested.

## Relationship to TIR (Tool-Integrated Reasoning)

Meta-Harness is **orthogonal to TIR** (the NVIDIA/NovaSky approach of letting the model write and execute Python code). TIR changes what the model *can do* at inference time. Meta-Harness changes how the scaffolding *wraps* the model. The two can be combined:
- TIR gives the model a code execution tool
- Meta-Harness auto-discovers the optimal system prompt, retrieval logic, context management, and tool definitions for that TIR setup

The companion paper on the coding side is arxiv 2603.05344: "Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned."

## Actionable Takeaways for AIMO-3

1. **Retrieval-augmented generation with discovered routing** beats hand-crafted retrieval. For AIMO-3, build a corpus from the training/public problems and let an outer loop discover the retrieval policy per problem type.

2. **Problem-type routing is crucial.** The discovered harness used topic-specific retrieval strategies -- this is something we can engineer manually: different retrieval k, reranking, and prompt strategies for algebra vs. combinatorics vs. geometry vs. number theory.

3. **The outer loop approach**: Rather than hand-tuning, run Meta-Harness's proposer pattern on a validation split -- give an LLM read access to all prior run logs, scores, and harness code, and let it propose the next harness to try.

4. **Generalization across models**: The harness found on smaller models generalizes to larger ones, suggesting meta-level scaffold optimization is transferable and not overfit to a specific model.
