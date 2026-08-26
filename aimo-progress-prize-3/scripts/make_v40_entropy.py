"""
Build v40: v39 (diversity + binary verify) + sophisticated 5-component weighted entropy.

Source: 43/50 competitor notebook (43-50-aimo-3-gpt-oss-120b-weighted-entropy.ipynb)
Their entropy beats plain mean because:
  1. Position weighting (decay=0.995): tokens near final answer matter MORE
  2. Variance penalty: inconsistent confidence = uncertain reasoning
  3. Sustained high entropy penalty: if model stays confused, it's lost
  4. Low entropy streak bonus: confident reasoning chains rewarded
  5. Weighted combination calibrated for math reasoning

Stacking on v39 (diversity + binary verify) should give both improvements.
"""
import json, io, ast, pathlib

ROOT    = pathlib.Path(__file__).parent.parent
SRC_NB  = ROOT / "notebooks/push_v39/submission_v39.ipynb"
OUT_DIR = ROOT / "notebooks/push_v40"
OUT_NB  = OUT_DIR / "submission_v40.ipynb"

NEW_ENTROPY = '''\
    def _compute_mean_entropy(self, logprobs_buffer):
        """
        5-component weighted entropy (43/50 competitor, significantly better than plain mean).
        Lower = more confident, focused reasoning.
        1. Position weighting (decay=0.995): tokens near final answer matter more
        2. Variance penalty: inconsistent confidence = uncertain reasoning chain
        3. Sustained high entropy penalty: long confusion stretches are bad
        4. Low entropy streak bonus: reward confident reasoning chains
        5. Weighted combination calibrated for math reasoning
        """
        if not logprobs_buffer:
            return float('inf')

        entropies = []
        for top_lp in logprobs_buffer:
            if isinstance(top_lp, dict) and top_lp:
                ent = sum(-math.exp(lp) * math.log2(math.exp(lp)) for lp in top_lp.values() if math.exp(lp) > 0)
                entropies.append(ent)

        if not entropies:
            return float('inf')

        n = len(entropies)

        # Component 1: Base mean entropy
        mean_ent = sum(entropies) / n

        # Component 2: Variance penalty (inconsistent confidence = bad reasoning)
        variance = sum((e - mean_ent) ** 2 for e in entropies) / n
        std_dev = math.sqrt(variance)

        # Component 3: Position-weighted entropy (recent tokens near answer matter most)
        decay_factor = 0.995
        weighted_sum = sum(e * (decay_factor ** (n - i - 1)) for i, e in enumerate(entropies))
        weighted_count = sum(decay_factor ** (n - i - 1) for i in range(n))
        position_weighted_ent = weighted_sum / weighted_count if weighted_count > 0 else mean_ent

        # Component 4: Sustained high entropy penalty (model is lost)
        high_ent_threshold = 2.0
        high_ent_ratio = sum(1 for e in entropies if e > high_ent_threshold) / n

        # Component 5: Low entropy streak bonus (confident reasoning chains)
        low_ent_threshold = 0.5
        max_streak = cur = 0
        for e in entropies:
            if e < low_ent_threshold:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        streak_bonus = -0.1 * (max_streak / n)

        return (
            0.3 * mean_ent
            + 0.4 * position_weighted_ent
            + 0.2 * std_dev
            + 0.3 * high_ent_ratio * 3.0
            + streak_bonus
        )

'''

OLD_ENTROPY_MARKER = "    def _compute_mean_entropy(self, logprobs_buffer):"
NEXT_DEF_MARKER = "    def _verify_answer"

with io.open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

applied = []

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    changed = False

    # Patch 1: Replace _compute_mean_entropy (index-based, markers include indentation)
    if OLD_ENTROPY_MARKER in src and NEXT_DEF_MARKER in src and "5-component" not in src:
        old_start = src.index(OLD_ENTROPY_MARKER)
        old_end   = src.index(NEXT_DEF_MARKER, old_start + len(OLD_ENTROPY_MARKER))
        old_block = src[old_start:old_end]
        # Safety check: make sure we're replacing the right thing
        if "_compute_mean_entropy" in old_block and "_verify_answer" not in old_block:
            src = src[:old_start] + NEW_ENTROPY + src[old_end:]
            applied.append("entropy"); changed = True

    # Patch 2: version comment
    if "print(f'CFG: v39" in src:
        src = src.replace(
            "print(f'CFG: v39 | strategy diversity + 1/entropy + binary verify tiebreaker')",
            "print(f'CFG: v40 | diversity + 5-component entropy (43/50 proven) + binary verify')"
        )
        applied.append("version"); changed = True

    if changed:
        cell["source"] = src.splitlines(keepends=True)

print(f"Applied: {applied}")
missing = [p for p in ["entropy", "version"] if p not in applied]
if missing:
    print(f"ERROR — not applied: {missing}")
    exit(1)

# Syntax check
errors = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        s = "".join(cell["source"])
        if s.strip():
            try:
                ast.parse(s)
            except SyntaxError as e:
                errors.append(f"Cell {i}: {e}")
if errors:
    print(f"SYNTAX ERRORS: {errors}"); exit(1)
print("Syntax: OK")

OUT_DIR.mkdir(exist_ok=True)
with io.open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"Written: {OUT_NB}")

meta = {
    "id": "canivel/aimo3-v40-entropy",
    "title": "AIMO3 v40 entropy",
    "code_file": "submission_v40.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
    "model_sources": ["danielhanchen/gpt-oss-120b/Transformers/default/1"],
    "dataset_sources": [],
    "kernel_sources": ["andreasbis/aimo-3-utils"],
    "keywords": [],
    "machine_shape": "NvidiaH100",
}
with io.open(OUT_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Slug: aimo3-v40-entropy")
