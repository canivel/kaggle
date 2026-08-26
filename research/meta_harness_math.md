# Meta-Harness: Math Benchmark Deep Dive

Paper: arxiv 2603.28052
Project page: https://yoonholee.com/meta-harness/
Authors: Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn
Date: March 30, 2026

---

## 1. Paper Overview

Meta-Harness is an outer-loop system that searches over LLM harness code using an
agentic proposer that accesses the full history of prior candidates' source code,
execution traces, and scores via a filesystem. The proposer is Claude Code (with
Opus 4.6) acting as a coding agent that reads prior results using standard tools
(grep, cat, etc.) - giving it "up to 10M tokens of diagnostic context per step, vs.
at most 26K for all prior methods."

Three benchmarks are reported: text classification, math reasoning (retrieval-augmented),
and agentic coding (TerminalBench-2). This document focuses exclusively on math.

---

## 2. Math Problem Retrieval Corpus

### Corpus Dataset: `yoonholee/math-corpus-combined` (HuggingFace)

**Total size: 535,356 problems**

This is described in the paper as ">=500,000 solved problems from eight open-source datasets."
The actual HuggingFace dataset lists exactly eight source datasets:

| Dataset | Row Count | HuggingFace ID |
|---------|-----------|----------------|
| OpenMathReasoning | 281,743 | nvidia/OpenMathReasoning |
| NuminaMath-1.5 | 129,520 | AI-MO/NuminaMath-1.5 |
| DeepMath-103K | 103,021 | zwhe99/DeepMath-103K |
| PolyMath | 11,083 | AIMO-Corpus/PolyMath |
| Omni-MATH | 4,289 | KbsdJames/Omni-MATH |
| FineProofs-SFT | 4,275 | SPIderman5/FineProofs-SFT |
| AIME 1983-2024 | 933 | gneubig/aime-1983-2024 |
| Putnam-AXIOM | 492 | Putnam-AXIOM/putnam-axiom-dataset-v1 |

### Schema Fields (per row)
- `problem` (Text) - the problem statement
- `solution` (Text) - worked solution
- `answer` (Text) - final answer (null for proofs)
- `dataset` (Text) - source dataset name
- `source` (Text)
- `topic` (Text) - e.g., "Geometry", "Number Theory"
- `difficulty` (Float64) - numeric difficulty score
- `problem_type` (Text) - e.g., "proof", "converted_proof"
- `domain` (Text)
- `category` (Text)
- `competition` (Text)
- `grade` (Int64)

### Supplementary Books Corpus: `yoonholee/olympiad-books-open-source`
- 3,510 chunks from 12 open-source math textbooks
- Includes "An Infinitely Large Napkin" (Evan Chen), combinatorics books, etc.
- Fields: text, book, subject, level, part, chapter, section, chunk_id, tokens_est
- Used alongside the main corpus in some retrievers

---

## 3. Evaluation Set (Test Problems)

**200 held-out IMO-level problems** from three sources (paper Table 6 context):
- IMO-AnswerBench
- IMO-ProofBench
- ArXivMath (referenced in the paper)

The eval repo (`yoonholee/math-frontier-eval`) actually uses **four test benchmarks**
totaling 506 problems (the paper reports a 200-problem subset for Table 6):

| Dataset | Problems | Type |
|---------|----------|------|
| CMIMC | 40 | Answer (competition) |
| USAMO | 6 | Proof (olympiad) |
| IMO-AnswerBench | 400 | Answer (100 x 4 domains) |
| IMO-ProofBench | 60 | Proof (4 levels x 4 domains) |

**Search/validation set (used during harness optimization):** 250 problems from
OlympiadBench + Omni-MATH hard difficulty tier (also available as the `val_v4` split
of `yoonholee/math-corpus-combined` with 250 rows).

**Decontamination:** "confirmed that held-out problems have no exact prefix matches
under our string-based filter, and manually inspected top BM25 retrievals for
held-out examples."

---

## 4. The Math Retriever Base Infrastructure

All harnesses inherit from `MathRetriever` (defined in `math_retriever.py`).

### `math_retriever.py` - Complete Base Class

```python
"""Base class and retrieval primitives for math retrieval systems."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenized
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parent


class MathRetriever(ABC):
    """Base class. Only contract: build_prompt(problem) -> str."""

    def __init__(self, test_problems: list[dict] = None):
        BOOKS_REPO = "yoonholee/olympiad-books-open-source"
        CORPUS_REPO = "yoonholee/math-corpus-combined"
        self.books = load_dataset(BOOKS_REPO, split="train")
        self.corpus = load_dataset(CORPUS_REPO, split="train")
        self.test_problems = test_problems or []

    @abstractmethod
    def build_prompt(self, problem: str) -> str:
        """Return the complete prompt for this problem."""
```

### Text Normalization

```python
_RE_DISPLAY = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle)\s*")
_RE_DELIM = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)([\[(). \[\]{}])")
_RE_DELIM2 = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*")
_RE_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize LaTeX to canonical forms. ~0.07ms/doc."""
    s = _RE_DISPLAY.sub("", text)
    s = _RE_DELIM.sub(r"\1", s)
    s = _RE_DELIM2.sub("", s)
    s = s.replace("\\leqslant", "\\le").replace("\\geqslant", "\\ge")
    s = s.replace("\\leq", "\\le").replace("\\geq", "\\ge")
    s = s.replace("\\neq", "\\ne")
    s = s.replace("\\lvert", "|").replace("\\rvert", "|")
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    s = s.replace("\\operatorname", "\\mathrm")
    s = s.replace("\\mathbb", "\\mathbb").replace("\\mathbf", "\\mathbf")
    s = _RE_WS.sub(" ", s).strip()
    return s
```

### Math-Aware Tokenizer

The tokenizer is the critical piece - it preserves LaTeX commands as atomic units:

```python
# Captures: \commands, ^{...}, _{...}, words, numbers, single chars
_MATH_TOKEN = re.compile(r"\\[a-zA-Z]+|[_^]\{[^}]*\}|[a-zA-Z]+|[0-9]+|\S")


def math_tokenize(text: str) -> list[str]:
    r"""Tokenize preserving LaTeX commands (\frac, \sum), superscripts, subscripts."""
    return _MATH_TOKEN.findall(text.lower())
```

### `MathBM25` Class (Complete Implementation)

```python
def _make_tokenized(docs_tokens: list[list[str]]) -> Tokenized:
    """Build bm25s Tokenized from pre-tokenized docs. Preserves backslashes."""
    vocab = {}
    ids = []
    for doc in docs_tokens:
        doc_ids = []
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab)


class MathBM25:
    """BM25 index with math-aware tokenizer."""

    def __init__(self, documents, *, doc_ids=None, text_fn=None):
        if text_fn is not None:
            texts = [text_fn(d) for d in documents]
        else:
            texts = list(documents)

        self.n = len(texts)
        self.doc_ids = doc_ids if doc_ids is not None else list(range(self.n))

        tokenized = [math_tokenize(normalize(t)) for t in texts]
        self._toks = _make_tokenized(tokenized)
        self._bm25 = bm25s.BM25()
        self._bm25.index(self._toks)

    def query(self, text: str, k: int = 3) -> list[tuple[float, int]]:
        """Return [(score, doc_id), ...] sorted by score descending."""
        q_toks = math_tokenize(normalize(text))
        vocab = self._toks.vocab
        q_ids = [vocab[t] for t in q_toks if t in vocab]
        if not q_ids:
            return []
        qt = Tokenized(ids=[q_ids], vocab=vocab)
        doc_indices, scores = self._bm25.retrieve(qt, k=min(k, self.n))
        return [
            (float(scores[0, r]), self.doc_ids[int(doc_indices[0, r])])
            for r in range(scores.shape[1])
        ]
```

The BM25 implementation uses the `bm25s` library. Index documents are normalized
and tokenized at build time; queries are normalized and tokenized at query time.
No explicit k1/b parameter tuning is visible - relies on `bm25s.BM25()` defaults.

---

## 5. The Lexical Router: Exact Classification Logic

The discovered harness uses lightweight lexical predicates. Here are the exact
regex patterns from the top-performing retriever (`evo_geo_solution_indexed.py`):

### Proof Detection

```python
_PROOF_RE = re.compile(
    r'\bprove\b|\bshow\s+that\b|\bdemonstrate\b|\bverify\s+that\b|\bestablish\s+that\b|'
    r'find\s+all\b.{0,60}(and\s+)?(prove|show)\b|\bif\s+and\s+only\s+if\b|\biff\b',
    re.IGNORECASE
)
```

### Domain Detection Regexes

```python
_COMB_RE = re.compile(
    r"\b(combinatorics?|counting|arrangement|pigeonhole|permutation|combination"
    r"|bijection|path|lattice|tournament|graph|coloring|tiling|choose|binomial"
    r"|committee|select|arrange|ways)\b",
    re.IGNORECASE,
)

_GEO_RE = re.compile(
    r"\b(triangle|circle|polygon|angle|perpendicular|parallel|circumscribed"
    r"|inscribed|tangent|chord|arc|radius|diameter|midpoint|centroid|incircle"
    r"|circumcircle|orthocenter|altitude|median)\b",
    re.IGNORECASE,
)

_NT_RE = re.compile(
    r"\b(prime|divisible|divisor|modulo|mod|congruent|gcd|lcm|floor|ceiling"
    r"|digit|integer|remainder|factor|coprime|euler|phi|fermat"
    r"|quadratic residue)\b",
    re.IGNORECASE,
)
```

### Domain Priority Order

```python
def _detect_domain(problem: str) -> str:
    if _COMB_RE.search(problem):
        return "COMB"
    if _GEO_RE.search(problem):
        return "GEO"
    if _NT_RE.search(problem):
        return "NT"
    return "ALGEBRA"  # default
```

### Difficulty Estimation (Keyword Heuristic)

```python
def _estimate_difficulty(problem: str) -> float:
    if re.search(r"\bIMO\b|USAMO|Putnam|EGMO|Olympiad", p):
        return 8.0
    if re.search(r"\bAIME\b|HMMT|AMC 12|Harvard-MIT", p):
        return 7.0
    if re.search(r"AMC 10|AMC 8|\bSMT\b", p):
        return 5.5
    return 6.5  # corpus median fallback
```

---

## 6. BM25 Retrieval Parameters Per Route

This section documents the best-performing harness (`evo_geo_solution_indexed`),
which is the "overall leader" in the eval repo.

### Route Architecture (3-branch, not 4)

The paper's Figure 8 shows a 4-route harness (combinatorics / geometry / number
theory / algebra+other). The actual evolved best harness (`evo_geo_solution_indexed`)
uses **3 branches** based on proof vs. computation and geometry specialization:

| Condition | Branch | Index Built On |
|-----------|--------|----------------|
| IS_PROOF AND IS_GEO | geo_solution_index | Solution text (cross-field) |
| IS_PROOF AND NOT GEO | proof_index | Problem text, difficulty >= 6.0 |
| NOT IS_PROOF | answer_index | Full corpus problem text |

### Hyperparameters

```python
PRE_RETRIEVE_K = 20          # BM25 candidates fetched per query
TOP_K = 3                    # final examples in prompt
DIFF_WINDOW = 2.0            # difficulty band filter: abs(corpus_diff - query_diff) <= 2.0
FALLBACK_THRESHOLD = 2       # if < 2 pass difficulty filter, use all BM25 results
DIVERSITY_THRESHOLD = 0.5    # Jaccard similarity threshold for deduplication
MIN_SECONDARY_TERMS = 3      # minimum terms for secondary BM25 query
PROOF_MIN_DIFFICULTY = 6.0   # minimum difficulty for proof corpus filter
PROOF_MIN_ROWS = 1000        # minimum proof corpus size before supplementing
GEO_PROOF_MIN_ROWS = 100     # minimum geo proof corpus size before fallback
```

### Domain-Adaptive Solution Length Truncation

```python
_LEN = {
    "COMB": 800,
    "GEO": 300,
    "NT": 400,
    "ALGEBRA": 400,
    "PROOF": 600,
    "GEO_PROOF": 300,
}
```

### OR-Max Dual Query Retrieval

The advanced harnesses use a dual-query OR-max approach for secondary term expansion:

```python
def _or_max_retrieve(problem: str, index: MathBM25, difficulties, diff_window: float) -> list:
    """Run dual-query OR-max BM25 then difficulty-filter."""
    primary = {idx: score for score, idx in index.query(problem, k=PRE_RETRIEVE_K)}

    secondary: dict = {}
    phrase_query = _extract_math_phrases(problem)
    if phrase_query and len(phrase_query.split()) >= MIN_SECONDARY_TERMS:
        secondary = {idx: score for score, idx in index.query(phrase_query, k=PRE_RETRIEVE_K)}

    all_idx = set(primary) | set(secondary)
    fused = [
        (max(primary.get(idx, 0.0), secondary.get(idx, 0.0)), idx)
        for idx in all_idx
    ]
    fused.sort(key=lambda x: x[0], reverse=True)

    est_diff = _estimate_difficulty(problem)
    filtered = [
        (score, idx)
        for score, idx in fused
        if abs(difficulties[idx] - est_diff) <= diff_window
    ]
    if len(filtered) < FALLBACK_THRESHOLD:
        filtered = fused
    return filtered
```

### Named Math Terms for Secondary Query

```python
_NAMED_TERMS = re.compile(
    r"\b("
    r"Fermat|Euler|Cauchy|Pigeonhole|Polya|P[oó]lya|Lagrange|Bezout|B[eé]zout"
    r"|Vieta|AM-GM|AM.GM|Cauchy-Schwarz|Cauchy.Schwarz|Chebyshev|Stirling"
    r"|Ramsey|Wilson|Lucas|Hensel|Dirichlet|Gauss|Legendre|Jacobi"
    r"|incenter|circumcenter|orthocenter|centroid"
    r"|incircle|circumcircle|circumradius|inradius|excircle"
    r"|angle\s+bisector|altitude|symmedian|radical\s+axis|power\s+of\s+a\s+point"
    r"|Ptolemy|Menelaus|Ceva|Simson|Euler\s+line"
    r"|arithmetic\s+progression|geometric\s+progression|Fibonacci|recurrence"
    r"|polynomial|quadratic|cubic|binomial|multinomial"
    r"|totient|quadratic\s+residue|Legendre\s+symbol|primitive\s+root"
    r")\b",
    re.IGNORECASE,
)
```

### Geometry Sub-Index: Solution-Indexed Cross-Field Retrieval

The key innovation in `evo_geo_solution_indexed` is building the BM25 index on
**solution text** rather than problem text for geometry proofs:

```python
# KEY DIFFERENCE: index solution text, not problem text
self.geo_solution_index = MathBM25(self.geo_corpus["solution"])
```

At query time, the problem statement is used as the BM25 query against an index
of solution bodies. This shifts IDF weights from problem-setup vocabulary
("triangle", "circle") toward proof-technique vocabulary ("angle chasing",
"power of a point", "spiral similarity", "inversion", "radical axis").

The geo corpus is curated from:
- Omni-MATH geometry problems
- FineProofs geometry problems (FineProofs-SFT, topic in ("Geometry", None))
- NuminaMath geometry proofs (where answer is proof-like)

---

## 7. Few-Shot Prompt Format

### Prompt Structure

All prompts follow this structure (from `bm25_retrieval.py` and evolved harnesses):

```
{preamble}

{problem}

Here are some similar solved examples for reference:

Example 1:
Problem: {retrieved_problem_1}
Solution: {solution_1}
Answer: {answer_1}   ← only included if answer field is non-empty

Example 2:
Problem: {retrieved_problem_2}
Solution: {solution_2}
Answer: {answer_2}

Example 3:
Problem: {retrieved_problem_3}
Solution: {solution_3}
Answer: {answer_3}

{reminder}  ← only for computation (answer) problems
```

### Preamble Variants

**Computation/answer problems:**
```
Solve the following math problem step by step. Put your answer inside \boxed{}.
```

**Proof problems:**
```
Solve the following math problem with a rigorous proof or complete justification.
Show all steps clearly, including any lemmas, base cases, or key structural arguments.
```

**Geometry proof problems:**
```
Solve the following geometry problem with a rigorous proof.
Clearly identify key geometric relationships, use precise angle/length arguments,
and justify each step.
```

**Reminder (computation only, appended at end):**
```
Remember to put your answer inside \boxed{}.
```

### Final Prompt Assembly (from `benchmark.py`)

The evaluation harness wraps the retriever context with an official template:

```python
ANSWER_TEMPLATE = "{problem_statement}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
PROOF_TEMPLATE = "Generate a rigorous proof to the following question:\n\n{problem_statement}"

def _make_prompt(context: str, problem: str, ds_tag: str) -> str:
    """Compose final prompt: retriever context + official template."""
    template = PROOF_TEMPLATE if ds_tag in PROOF_DATASETS else ANSWER_TEMPLATE
    official = template.format(problem_statement=problem)
    if context:
        return context.rstrip() + "\n\n" + official
    return official
```

So the final prompt for an answer problem with retrieval context is:
```
{retriever preamble + examples block}

{problem_statement}

Please reason step by step, and put your final answer within \boxed{}.
```

### Solution Truncation
Solutions are truncated at domain-adaptive character limits with `\n[... truncated]` suffix.

---

## 8. Greedy Jaccard Diversity Selection

After BM25 retrieval and difficulty filtering, examples are selected via greedy
Jaccard deduplication on problem text token sets:

```python
def _greedy_diversity_corpus(self, candidates: list, corpus, max_len: int) -> str:
    selected: list = []
    selected_texts: list = []

    for score, idx in candidates:
        text = corpus[idx]["problem"]
        if not any(_jaccard(text, t) >= DIVERSITY_THRESHOLD for t in selected_texts):
            selected.append((score, idx))
            selected_texts.append(text)
        if len(selected) >= TOP_K:
            break
    ...

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
```

Note: Jaccard here is computed on whitespace-split tokens (not math_tokenize).
Threshold is 0.5 - examples with >= 50% token overlap are skipped.

---

## 9. Evaluation Setup

### Configuration (`config.yaml`)

```yaml
eval:
  n_samples: 3              # pass@1 averaged over 3 samples
  temperature: 1.0          # sampling temperature
  max_tokens: 16384         # generation budget
  concurrency: 48
  judge_model: openrouter/openai/gpt-oss-20b
  datasets:
    - cmimc
    - usamo
    - imo_answerbench
    - imo_proofbench
```

### Models Evaluated (5 held-out)

From the paper (Table 6) and README:

| Alias | Full model name | Notes |
|-------|----------------|-------|
| GPT-5.4n / gpt5.4nano | GPT-5.4-nano | OpenAI API |
| GPT-5.4m / gpt5.4mini | GPT-5.4-mini | OpenAI API |
| Gem-3.1FL | Gemini-3.1-Flash-Lite | Google |
| Gem-3F | Gemini-3-Flash | Google |
| GPT-20B | local/openai/gpt-oss-20b | vLLM local |

The search/optimization used GPT-OSS-20B only; the other four are held-out.

### Grading

**Answer problems (CMIMC, IMO-AnswerBench):** Symbolic equivalence check
(`grading.verify()`)

**Proof problems (USAMO, IMO-ProofBench):** LLM judge (GPT-OSS-20B) on 0-7 scale;
pass threshold is score >= 6.0/7.0

**Metric:** pass@1 averaged over 3 samples per problem

### Results (Table 6 verbatim)

| Method | GPT-5.4n | GPT-5.4m | Gem-3.1FL | Gem-3F | GPT-20B | Avg |
|--------|----------|----------|-----------|--------|---------|-----|
| No Retrieval | 23.0 | 28.8 | 28.6 | 42.6 | 47.6 | 34.1 |
| Dense k=1 | 27.1 (+4.1) | 24.5 (-4.3) | 31.3 (+2.7) | 42.3 (-0.3) | 46.9 (-0.7) | 34.4 (+0.3) |
| Dense k=5 | 31.1 (+8.1) | 28.3 (-0.5) | 37.1 (+8.5) | 47.2 (+4.6) | 46.7 (-0.9) | 38.1 (+4.0) |
| Random Few-shot | 23.1 (+0.1) | 24.5 (-4.3) | 31.0 (+2.4) | 40.4 (-2.2) | 41.8 (-5.8) | 32.2 (-1.9) |
| BM25 Retrieval | 30.2 (+7.2) | 29.2 (+0.4) | 32.8 (+4.2) | 46.6 (+4.0) | 48.9 (+1.3) | 37.5 (+3.4) |
| Meta-Harness | 31.7 (+8.7) | 30.4 (+1.6) | 34.9 (+6.3) | 46.3 (+3.7) | 50.6 (+3.0) | 38.8 (+4.7) |

Key finding: Meta-Harness beats BM25 by +1.3 points avg. Random few-shot hurts.
Dense retrieval with k=1 can hurt (GPT-5.4m: -4.3). The Meta-Harness operates
entirely on the same BM25 retrieval stack - no dense encoder added.

---

## 10. The Optimization Loop

### Algorithm 1 (verbatim from paper)

```
Algorithm 1 Meta-Harness outer loop over harnesses
1: Input: tasks X, LLM M, proposer P, iterations N
2: Initialize: population H  ⊳ Initial set of valid harnesses
3: Initialize: filesystem D <- empty  ⊳ stores code, scores, traces
4: for H in H do
5:   E_H <- Evaluate(H, M, X)
6:   D <- D union {(H, E_H)}
7: for t = 1...N do
8:   Proposer P queries filesystem D  ⊳ inspects prior harnesses and scores
9:   Proposer P proposes k new harnesses {H_1, ..., H_k}
10:  for H in {H_1, ..., H_k} do
11:    if H passes interface validation then
12:      D <- D union {(H, Evaluate(H, M, X))}
13: return Pareto frontier of harnesses stored in D
```

### Math Experiment Parameters

- **Total iterations:** 40 (producing 109 candidate retrieval harnesses)
- **Candidates per iteration (k):** not exact, but 109 total / 40 iters ~ 2-3 per iter
- **Proposer model:** Claude Code with Opus 4.6
- **Search model (for scoring candidates):** GPT-OSS-20B
- **Search set size:** 250 problems (OlympiadBench + Omni-MATH hard)
- **Selection:** Single harness selected based on search-set performance with GPT-OSS-20B
- **Starting population:** BM25 baseline + a small number of hand-written variants

### Filesystem Context Available to Proposer

The proposer (Claude Code) sees a filesystem containing for each prior candidate:
- Full Python source code of the harness
- Accuracy scores on the 250-problem search set
- Per-problem execution traces (which examples were retrieved, what prompt was built)

The discovered harness is described as "a merge of two successful search lineages,
autonomously combined by the proposer during search."

---

## 11. All Discovered Harnesses (Named in Eval Repo)

The 9 "frontier" evolved retrievers and their specializations:

| Name | Description |
|------|-------------|
| `evo_geo_solution_indexed` | **Overall leader.** Solution-indexed geo sub-index (cross-field retrieval). 3-branch routing. |
| `evo_proof_split_or_max_diversity` | Proof/answer split + OR-max dual query + diversity |
| `evo_geo_proof_curated_index` | Curated geo-proof sub-corpus (Omni-MATH + FineProofs + NuminaMath) |
| `evo_openmath_geo_proof_branch` | Branch using OpenMathReasoning data specifically |
| `evo_domain_conditional_secondary` | Domain-conditional secondary query expansion |
| `evo_deepmath_hard_augment` | **Geometry champion.** Uses DeepMath hard problems |
| `evo_proof_answer_split` | **Number theory champion.** Proof vs answer split |
| `evo_combined_routing_diversity` | **Algebra champion.** Merged routing + Jaccard diversity |
| `evo_algebra_hard_fusion` | **Combinatorics champion.** Algebra/hard problem fusion |

---

## 12. Baseline Retrievers for Comparison

### `no_memory.py` (minimal baseline)

```python
class NoMemory(MathRetriever):
    def build_prompt(self, problem: str) -> str:
        return f"{problem}\n\nPut your answer inside \\boxed{{}}."
```

### `bm25_retrieval.py` (BM25 baseline, k=3)

```python
class BM25Retrieval(MathRetriever):
    def __init__(self, test_problems=None, k: int = 3):
        super().__init__(test_problems)
        self.k = k
        self.corpus = self.corpus.filter(lambda x: x["solution"] is not None)
        self.index = MathBM25(self.corpus["problem"])

    def build_prompt(self, problem: str) -> str:
        preamble = "Solve the following math problem step by step. Put your answer inside \\boxed{}."
        reminder = "Remember to put your answer inside \\boxed{}."
        context = self._retrieve(problem)
        if context:
            return f"{preamble}\n\n{problem}\n\n{context}\n\n{reminder}"
        return f"{preamble}\n\n{problem}\n\n{reminder}"
```

---

## 13. Key Code Repositories and Artifacts

| Resource | URL | Access |
|----------|-----|--------|
| Paper | https://arxiv.org/abs/2603.28052 | Public |
| Paper HTML | https://arxiv.org/html/2603.28052v1 | Public |
| Project page | https://yoonholee.com/meta-harness/ | Public |
| Eval code | https://github.com/yoonholee/math-frontier-eval | Public |
| Math corpus | https://huggingface.co/datasets/yoonholee/math-corpus-combined | Public |
| Prompts dataset | https://huggingface.co/datasets/yoonholee/math-frontier-prompts | Private (HF_TOKEN needed) |
| Books corpus | https://huggingface.co/datasets/yoonholee/olympiad-books-open-source | Public |
| TerminalBench artifact | https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact | Public |

---

## 14. Key Takeaways and Design Principles

1. **BM25 beats dense retrieval** for this task when properly tuned. Dense k=1 can
   actually hurt performance (-4.3 on GPT-5.4m). Meta-Harness never adds a dense
   encoder - all gains come from better BM25 routing and reranking.

2. **Lexical routing works.** Simple regex-based domain classification (COMB/GEO/NT/ALGEBRA)
   enables specialized retrieval without any learned components.

3. **Cross-field indexing is novel.** Indexing solution text but querying with problem
   text shifts IDF weights toward proof-technique vocabulary. This is the key
   innovation in the best harness.

4. **Difficulty filtering is essential.** A ±2.0 difficulty band filter is applied
   universally. Without it, easy problems retrieve easy examples, losing discriminative
   value.

5. **Dual-query OR-max.** A secondary query using extracted named mathematical terms
   (Fermat, Cauchy-Schwarz, etc.) is run and OR-maxed with the primary query. This
   expands recall for problems with recognizable technique names.

6. **Jaccard diversity.** Greedy deduplication at 0.5 threshold on whitespace tokens
   ensures 3 structurally distinct examples rather than near-duplicates.

7. **3 examples is optimal (TOP_K=3).** Across all evolved harnesses, the final
   selection is always 3 examples. Pre-retrieval fetches 20, filters to ~8, diversity-
   selects to 3.

8. **Random few-shot hurts** (-1.9 avg vs no retrieval). Retrieval quality matters.

9. **The optimization finds merges.** The best harness was autonomously created by
   the proposer merging two separately evolved lineages - something hand-engineering
   would likely not discover.
