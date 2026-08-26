"""Build the math problem retrieval corpus for AIMO3.

Downloads and combines problems from multiple sources:
1. NuminaMath-TIR (olympiad subset) - problems WITH code solutions
2. MATH-Hard (Level 5) - competition math
3. AIME 1983-2024 - historical AIME
4. NuminaMath-CoT (olympiad subset) - problems with NL solutions

Output: data/math_corpus.jsonl (~15-20K problems)

Usage:
    uv run python scripts/build_corpus.py
    uv run python scripts/build_corpus.py --max-problems 5000  # smaller corpus
"""

import argparse
import json
import hashlib
from pathlib import Path


def problem_hash(text: str) -> str:
    """Hash problem text for deduplication."""
    # Normalize whitespace and case
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def classify_domain(problem: str) -> str:
    """Simple domain classification."""
    text = problem.lower()
    geo_kw = ["triangle", "circle", "angle", "perpendicular", "inscribed", "polygon"]
    nt_kw = ["prime", "divisible", "modulo", "gcd", "remainder", "congruent"]
    combo_kw = ["how many", "number of ways", "permutation", "combinat", "probability"]

    geo = sum(1 for k in geo_kw if k in text)
    nt = sum(1 for k in nt_kw if k in text)
    combo = sum(1 for k in combo_kw if k in text)

    if combo >= 2:
        return "combinatorics"
    if geo >= 2:
        return "geometry"
    if nt >= 2:
        return "number_theory"
    return "algebra"


def estimate_difficulty(source: str) -> float:
    """Estimate difficulty from source name."""
    source_lower = source.lower()
    if any(k in source_lower for k in ["imo", "usamo", "putnam"]):
        return 8.0
    if any(k in source_lower for k in ["aime", "hmmt"]):
        return 7.0
    if any(k in source_lower for k in ["amc_12", "amc12"]):
        return 6.0
    if any(k in source_lower for k in ["amc_10", "amc10", "mathcounts"]):
        return 5.5
    if "olympiad" in source_lower:
        return 7.5
    return 6.5


def load_numinamath_tir(max_problems: int = -1) -> list[dict]:
    """Load NuminaMath-TIR dataset (has Python code solutions)."""
    from datasets import load_dataset

    print("Loading NuminaMath-TIR...")
    ds = load_dataset("AI-MO/NuminaMath-TIR", split="train", streaming=True)

    problems = []
    seen = set()

    for row in ds:
        problem_text = row.get("problem", "")
        solution_text = row.get("solution", "")

        if not problem_text or len(problem_text) < 20:
            continue

        h = problem_hash(problem_text)
        if h in seen:
            continue
        seen.add(h)

        has_code = "```python" in solution_text or "```" in solution_text
        domain = classify_domain(problem_text)

        problems.append({
            "problem": problem_text,
            "solution": solution_text,
            "answer": "",
            "source": "numinamath_tir",
            "domain": domain,
            "difficulty": 6.5,
            "has_code": has_code,
        })

        if max_problems > 0 and len(problems) >= max_problems:
            break

    print(f"  NuminaMath-TIR: {len(problems)} problems ({sum(1 for p in problems if p['has_code'])} with code)")
    return problems


def load_math_hard() -> list[dict]:
    """Load MATH-Hard (Level 5) dataset."""
    from datasets import load_dataset

    print("Loading MATH-Hard...")
    ds = load_dataset("lighteval/MATH-Hard", split="test")

    problems = []
    for row in ds:
        problems.append({
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "answer": str(row.get("answer", "")),
            "source": f"math_hard_{row.get('type', 'unknown')}",
            "domain": classify_domain(row.get("problem", "")),
            "difficulty": 7.0,
            "has_code": False,
        })

    print(f"  MATH-Hard: {len(problems)} problems")
    return problems


def load_aime() -> list[dict]:
    """Load AIME 1983-2024 dataset."""
    from datasets import load_dataset

    print("Loading AIME 1983-2024...")
    try:
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    except Exception:
        # Try alternative source
        try:
            ds = load_dataset("gneubig/aime-1983-2024", split="train")
        except Exception:
            print("  AIME dataset not available, skipping")
            return []

    problems = []
    for row in ds:
        problems.append({
            "problem": row.get("problem", row.get("question", "")),
            "solution": row.get("solution", ""),
            "answer": str(row.get("answer", "")),
            "source": "aime",
            "domain": classify_domain(row.get("problem", row.get("question", ""))),
            "difficulty": 7.0,
            "has_code": False,
        })

    print(f"  AIME: {len(problems)} problems")
    return problems


def load_aimo_validation() -> list[dict]:
    """Load AIMO validation set (90 AIME problems with solutions)."""
    from datasets import load_dataset

    print("Loading AIMO validation...")
    try:
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    except Exception:
        print("  AIMO validation not available, skipping")
        return []

    problems = []
    for row in ds:
        problems.append({
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "answer": str(row.get("answer", "")),
            "source": "aimo_validation",
            "domain": classify_domain(row.get("problem", "")),
            "difficulty": 7.0,
            "has_code": False,
        })

    print(f"  AIMO validation: {len(problems)} problems")
    return problems


def build_corpus(output_path: str, max_problems: int = -1):
    """Build the combined corpus."""
    all_problems = []

    # Load all sources
    all_problems.extend(load_numinamath_tir(max_problems=max_problems))
    all_problems.extend(load_math_hard())
    all_problems.extend(load_aime())
    all_problems.extend(load_aimo_validation())

    # Deduplicate
    seen = set()
    deduped = []
    for p in all_problems:
        h = problem_hash(p["problem"])
        if h not in seen and len(p["problem"]) > 20:
            seen.add(h)
            deduped.append(p)

    # Sort: problems with code first, then by difficulty descending
    deduped.sort(key=lambda x: (x["has_code"], x["difficulty"]), reverse=True)

    # Write
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for p in deduped:
            f.write(json.dumps(p) + "\n")

    # Stats
    n_code = sum(1 for p in deduped if p["has_code"])
    domains = {}
    for p in deduped:
        d = p["domain"]
        domains[d] = domains.get(d, 0) + 1

    print(f"\n=== Corpus built: {len(deduped)} problems ===")
    print(f"  With code solutions: {n_code}")
    print(f"  Domains: {domains}")
    print(f"  Output: {output}")


def main():
    parser = argparse.ArgumentParser(description="Build math retrieval corpus")
    parser.add_argument(
        "--output", default="data/math_corpus.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--max-problems", type=int, default=-1, help="Max problems from NuminaMath"
    )
    args = parser.parse_args()
    build_corpus(args.output, args.max_problems)


if __name__ == "__main__":
    main()
