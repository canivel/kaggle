"""Classify math problems by type using keyword/pattern matching.

Categories: algebra, combinatorics, geometry, number_theory.
Used to select problem-type-specific prompt templates.
"""

import re


# Keyword patterns for each problem type
GEOMETRY_PATTERNS = [
    r"\\triangle", r"\\angle", r"\\circ", r"circle", r"circumscribe",
    r"inscribe", r"polygon", r"quadrilateral", r"rectangle", r"square",
    r"parallelogram", r"trapezoid", r"rhombus", r"pentagon", r"hexagon",
    r"diameter", r"radius", r"tangent", r"chord", r"perpendicular",
    r"bisect", r"midpoint", r"centroid", r"circumcenter", r"incircle",
    r"excircle", r"altitude", r"median", r"\\overline", r"\\overrightarrow",
    r"collinear", r"concyclic", r"area of", r"perimeter",
    r"right triangle", r"isosceles", r"equilateral", r"hypotenuse",
    r"\\sin", r"\\cos", r"\\tan", r"coordinate",
]

NUMBER_THEORY_PATTERNS = [
    r"\\bmod", r"\\pmod", r"modulo", r"remainder when",
    r"divisible", r"divides", r"divisor", r"\\gcd", r"\\text{gcd}",
    r"greatest common", r"least common", r"\\text{lcm}",
    r"prime", r"composite", r"coprime", r"relatively prime",
    r"Euler.*totient", r"\\phi", r"Fermat", r"Wilson",
    r"congruent", r"residue", r"Diophantine",
    r"perfect square", r"perfect cube", r"factorial",
    r"digit sum", r"sum of digits", r"number of digits",
    r"base \d+", r"binary", r"representation",
]

COMBINATORICS_PATTERNS = [
    r"how many", r"number of ways", r"in how many",
    r"probability", r"expected value", r"expectation",
    r"permutation", r"combination", r"\\binom", r"choose",
    r"arrange", r"distribute", r"partition",
    r"pigeonhole", r"inclusion.exclusion",
    r"sequence.*satisfy", r"recurrence", r"recursive",
    r"subset", r"\\cup", r"\\cap", r"graph", r"vertex", r"edge",
    r"coloring", r"tournament", r"matching",
    r"chess", r"board", r"grid",
    r"distinct", r"ordered", r"unordered",
]

ALGEBRA_PATTERNS = [
    r"polynomial", r"equation", r"inequality", r"system of",
    r"\\sum", r"\\prod", r"series", r"sequence",
    r"maximum", r"minimum", r"minimize", r"maximize",
    r"function.*satisf", r"functional equation",
    r"real number", r"positive integer", r"complex number",
    r"root", r"zero", r"coefficient",
    r"matrix", r"determinant", r"eigenvalue",
    r"limit", r"converge", r"floor", r"ceiling",
    r"\\lfloor", r"\\rfloor", r"\\lceil", r"\\rceil",
]


def classify_problem(problem: str) -> str:
    """Classify a math problem into a category.

    Args:
        problem: The problem text (may contain LaTeX).

    Returns:
        One of: 'algebra', 'combinatorics', 'geometry', 'number_theory', 'default'.
    """
    problem_lower = problem.lower()

    scores = {
        "geometry": _score_patterns(problem_lower, GEOMETRY_PATTERNS),
        "number_theory": _score_patterns(problem_lower, NUMBER_THEORY_PATTERNS),
        "combinatorics": _score_patterns(problem_lower, COMBINATORICS_PATTERNS),
        "algebra": _score_patterns(problem_lower, ALGEBRA_PATTERNS),
    }

    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best

    # Single match - still use it if confident
    if scores[best] >= 1:
        return best

    return "default"


def _score_patterns(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    score = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    return score
