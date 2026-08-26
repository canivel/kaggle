"""BM25-based math problem retriever with lexical routing.

Implements the Meta-Harness retrieval architecture:
1. Classify problem into domain (combinatorics/geometry/number_theory/algebra)
2. Detect if problem is a proof or computation
3. Route to domain-specific BM25 index with tuned parameters
4. Retrieve top-k similar solved problems as few-shot examples
5. Apply difficulty filtering + Jaccard diversity dedup

Corpus: NuminaMath-TIR + MATH-Hard + AIME + IMO problems with solutions.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass


# ============================================================
# Math-aware tokenizer (from Meta-Harness paper)
# ============================================================

_MATH_TOKEN = re.compile(r"\\[a-zA-Z]+|[_^]\{[^}]*\}|[a-zA-Z]+|[0-9]+|\S")

# LaTeX normalization
_LATEX_SUBS = [
    (r"\\leqslant", r"\\le"),
    (r"\\geqslant", r"\\ge"),
    (r"\\tfrac", r"\\frac"),
    (r"\\dfrac", r"\\frac"),
    (r"\\operatorname", r"\\mathrm"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\displaystyle", ""),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\!", ""),
]


def normalize_latex(text: str) -> str:
    """Normalize LaTeX for consistent tokenization."""
    for old, new in _LATEX_SUBS:
        text = text.replace(old, new)
    return text


def math_tokenize(text: str) -> list[str]:
    """Tokenize text preserving LaTeX math tokens."""
    text = normalize_latex(text.lower())
    return _MATH_TOKEN.findall(text)


# ============================================================
# Domain classifier (lexical router from Meta-Harness)
# ============================================================

_PROOF_RE = re.compile(
    r"\bprove\b|\bshow\s+that\b|\bdemonstrate\b|\bverify\s+that\b|\bestablish\s+that\b|"
    r"find\s+all\b.{0,60}(and\s+)?(prove|show)\b|\bif\s+and\s+only\s+if\b|\biff\b",
    re.IGNORECASE,
)

_DOMAIN_KEYWORDS = {
    "combinatorics": [
        "combinatorics", "counting", "pigeonhole", "permutation", "bijection",
        "graph", "coloring", "choose", "binomial", "partition", "how many",
        "number of ways", "arrangement", "subset", "tournament", "matching",
        "probability", "expected value",
    ],
    "geometry": [
        "triangle", "circle", "polygon", "angle", "perpendicular", "inscribed",
        "tangent", "chord", "incircle", "circumcircle", "orthocenter", "altitude",
        "midpoint", "bisect", "diameter", "radius", "collinear", "concyclic",
        "area", "perimeter", "\\sin", "\\cos", "\\tan",
    ],
    "number_theory": [
        "prime", "divisible", "modulo", "congruent", "gcd", "lcm", "digit",
        "remainder", "factor", "coprime", "euler", "phi", "fermat", "\\bmod",
        "\\pmod", "diophantine", "totient", "residue",
    ],
}

# Named math terms for secondary BM25 query
_NAMED_TERMS = re.compile(
    r"(?:cauchy.schwarz|am.gm|pigeonhole|vieta|fermat|euler|chinese remainder|"
    r"power of a point|radical axis|inversion|spiral similarity|"
    r"angle chasing|ptolemy|menelaus|ceva|stewart|heron|"
    r"generating function|inclusion.exclusion|burnside|polya)",
    re.IGNORECASE,
)


def classify_domain(problem: str) -> str:
    """Classify problem domain. Priority: COMB > GEO > NT > ALGEBRA."""
    text = problem.lower()
    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw.lower() in text)

    # Priority order
    for domain in ["combinatorics", "geometry", "number_theory"]:
        if scores.get(domain, 0) >= 2:
            return domain

    best = max(scores, key=scores.get) if scores else "algebra"
    return best if scores.get(best, 0) >= 1 else "algebra"


def is_proof_problem(problem: str) -> bool:
    """Detect if problem requires a proof."""
    return bool(_PROOF_RE.search(problem))


def extract_named_terms(problem: str) -> list[str]:
    """Extract named mathematical theorems/techniques."""
    return _NAMED_TERMS.findall(problem)


# ============================================================
# Retrieval corpus entry
# ============================================================

@dataclass
class MathProblem:
    """A solved math problem in the retrieval corpus."""
    problem: str
    solution: str
    answer: str = ""
    source: str = ""
    domain: str = ""
    difficulty: float = 6.0
    has_code: bool = False

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "solution": self.solution,
            "answer": self.answer,
            "source": self.source,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "has_code": self.has_code,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MathProblem:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ============================================================
# BM25 Retriever
# ============================================================

# Domain-adaptive solution truncation (chars) from Meta-Harness
_SOLUTION_MAX_CHARS = {
    "combinatorics": 800,
    "geometry": 300,
    "number_theory": 400,
    "algebra": 400,
}

# Retrieval parameters from Meta-Harness
PRE_RETRIEVE_K = 20
TOP_K = 3
DIFF_WINDOW = 2.0
FALLBACK_THRESHOLD = 2
DIVERSITY_THRESHOLD = 0.5


def jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard similarity between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class MathRetriever:
    """BM25 retriever for math problems with domain routing.

    Usage:
        retriever = MathRetriever()
        retriever.load_corpus("data/math_corpus.jsonl")
        retriever.build_index()

        examples = retriever.retrieve(problem_text, k=3)
    """

    def __init__(self):
        self.corpus: list[MathProblem] = []
        self._index = None
        self._corpus_tokens: list[list[str]] = []

    def load_corpus(self, path: str | Path):
        """Load corpus from JSONL file."""
        path = Path(path)
        self.corpus = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.corpus.append(MathProblem.from_dict(d))
        print(f"Loaded {len(self.corpus)} problems from {path}")

    def build_index(self):
        """Build BM25 index over the corpus."""
        try:
            import bm25s
        except ImportError:
            # Fallback: simple TF-IDF style matching
            print("bm25s not available, using fallback retrieval")
            self._index = None
            self._corpus_tokens = [math_tokenize(p.problem) for p in self.corpus]
            return

        self._corpus_tokens = [math_tokenize(p.problem) for p in self.corpus]
        corpus_text = [" ".join(tokens) for tokens in self._corpus_tokens]

        self._index = bm25s.BM25()
        self._index.index(bm25s.tokenize(corpus_text))
        print(f"BM25 index built over {len(self.corpus)} problems")

    def retrieve(
        self,
        problem: str,
        k: int = TOP_K,
        target_difficulty: float = 7.0,
    ) -> list[MathProblem]:
        """Retrieve top-k similar solved problems.

        Pipeline: BM25@20 → difficulty filter → Jaccard diversity → top-k

        Args:
            problem: The query problem text.
            k: Number of examples to return.
            target_difficulty: Target difficulty for filtering.

        Returns:
            List of MathProblem objects, most relevant first.
        """
        if not self.corpus:
            return []

        query_tokens = math_tokenize(problem)
        query_domain = classify_domain(problem)

        # Step 1: BM25 retrieval
        candidates = self._bm25_search(query_tokens, PRE_RETRIEVE_K)

        # Step 2: Difficulty filter
        filtered = []
        for idx, score in candidates:
            p = self.corpus[idx]
            if abs(p.difficulty - target_difficulty) <= DIFF_WINDOW:
                filtered.append((idx, score))

        # Fallback if too few pass difficulty filter
        if len(filtered) < FALLBACK_THRESHOLD:
            filtered = candidates[:PRE_RETRIEVE_K]

        # Step 3: Prefer problems with code solutions
        filtered.sort(key=lambda x: (self.corpus[x[0]].has_code, x[1]), reverse=True)

        # Step 4: Jaccard diversity dedup
        selected = []
        selected_tokens = []
        for idx, score in filtered:
            if len(selected) >= k:
                break
            p_tokens = self._corpus_tokens[idx]
            # Check diversity against already selected
            too_similar = False
            for sel_tokens in selected_tokens:
                if jaccard_similarity(p_tokens, sel_tokens) > DIVERSITY_THRESHOLD:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(self.corpus[idx])
                selected_tokens.append(p_tokens)

        return selected

    def _bm25_search(
        self, query_tokens: list[str], k: int
    ) -> list[tuple[int, float]]:
        """Search the BM25 index."""
        if self._index is not None:
            import bm25s

            query_text = " ".join(query_tokens)
            results, scores = self._index.retrieve(
                bm25s.tokenize([query_text]), k=k
            )
            return [(int(results[0][i]), float(scores[0][i])) for i in range(len(results[0]))]

        # Fallback: simple token overlap scoring
        query_set = set(query_tokens)
        scored = []
        for i, tokens in enumerate(self._corpus_tokens):
            overlap = len(query_set & set(tokens))
            if overlap > 0:
                scored.append((i, overlap / max(len(query_set), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def format_examples(
        self, examples: list[MathProblem], domain: str = "algebra"
    ) -> str:
        """Format retrieved examples as few-shot prompt text."""
        if not examples:
            return ""

        max_chars = _SOLUTION_MAX_CHARS.get(domain, 400)
        parts = ["Here are some similar solved examples for reference:\n"]

        for i, ex in enumerate(examples, 1):
            sol = ex.solution[:max_chars]
            if len(ex.solution) > max_chars:
                sol += "..."

            part = f"Example {i}:\nProblem: {ex.problem}\nSolution: {sol}"
            if ex.answer:
                part += f"\nAnswer: {ex.answer}"
            parts.append(part)

        return "\n\n".join(parts)
