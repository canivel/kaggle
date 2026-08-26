"""Build v28: Exact winner config.

Changes from v16:
1. System prompt: 1804 chars -> 163 chars (winner's minimal 3-sentence prompt)
2. Tool prompt: 569 chars -> 125 chars (winner's minimal)
3. Preference prompt: 768 chars -> 68 chars (winner's minimal)
4. Temperature: 0.8 -> 1.0 (winner uses T=1.0 with min_p=0.02)
5. Remove strategy diversity (winner uses identical prompt for all 8 attempts)
6. Push to FRESH kernel slug (not tir-rag-baseline which never scored above 33)
"""

import json
import io
import ast
import shutil
import re
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v16_exact44.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v28_winner.ipynb"

# Winner's exact prompts
WINNER_SYSTEM = (
    "'You are a world-class International Mathematical Olympiad (IMO) competitor. '"
    "'The final answer must be a non-negative integer between 0 and 99999. '"
    r"'You must place the final integer answer inside \\boxed{}.'"
)

WINNER_TOOL = (
    "'Use this tool to execute Python code. '"
    "'The environment is a stateful Jupyter notebook. '"
    "'You must use print() to output results.'"
)

WINNER_PREF = "'You have access to `math`, `numpy` and `sympy` to solve the problem.'"


def main():
    print("Building v28 winner-config from v16...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        if "class CFG" in new_src and "system_prompt" in new_src:
            # Replace system_prompt
            # Find it by matching the opening and counting parens
            m = re.search(r"(    system_prompt = \([\s\S]*?\n    \))", new_src)
            if not m:
                # Try alternate format
                m = re.search(r"(    system_prompt = \([^)]+\))", new_src)
            if m:
                old = m.group(1)
                new = f"    system_prompt = (\n        {WINNER_SYSTEM})"
                new_src = new_src.replace(old, new)
                print(f"Cell {i}: system_prompt -> winner minimal (163 chars)")

            # Replace tool_prompt
            m = re.search(r"(    tool_prompt = \([\s\S]*?\n    \))", new_src)
            if not m:
                m = re.search(r"(    tool_prompt = \([^)]+\))", new_src)
            if m:
                old = m.group(1)
                new = f"    tool_prompt = (\n        {WINNER_TOOL})"
                new_src = new_src.replace(old, new)
                print(f"Cell {i}: tool_prompt -> winner minimal (125 chars)")

            # Replace preference_prompt
            m = re.search(r"(    preference_prompt = \([\s\S]*?\n    \))", new_src)
            if not m:
                m = re.search(r"(    preference_prompt = \([^)]+\))", new_src)
            if m:
                old = m.group(1)
                new = f"    preference_prompt = {WINNER_PREF}"
                new_src = new_src.replace(old, new)
                print(f"Cell {i}: preference_prompt -> winner minimal (68 chars)")

            # Temperature 0.8 -> 1.0
            new_src = new_src.replace(
                "temperature = 0.8  # arxiv 2603.27844: T=0.8 best mean (+0.3 over T=1.0)",
                "temperature = 1.0  # winner config"
            )
            print(f"Cell {i}: T=0.8 -> T=1.0")

        # Remove strategy diversity if present
        if "PREF_CODE_FIRST" in new_src:
            # Remove the constants
            for const in ["PREF_CODE_FIRST", "PREF_SMALL_CASES"]:
                pattern = re.compile(rf"\n{const} = \([\s\S]*?\)\n", re.MULTILINE)
                new_src = pattern.sub("\n", new_src)
            new_src = new_src.replace("\n# Strategy-diverse preference prompts\n", "\n")

        if "strategy_prefs" in new_src:
            new_src = new_src.replace(
                "strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2\n"
                "        user_inputs = [f'{problem} {p}' for p in strategy_prefs[:self.cfg.attempts]]\n"
                "        user_input = user_inputs[0]",
                "user_input = f'{problem} {self.cfg.preference_prompt}'"
            )
            new_src = new_src.replace(
                "futs = [ex.submit(self._process_attempt, user_inputs[ai], sp, ai, stop, deadline) for sp, ai in tasks]",
                "futs = [ex.submit(self._process_attempt, user_input, sp, ai, stop, deadline) for sp, ai in tasks]"
            )
            print(f"Cell {i}: removed strategy diversity")

        if new_src != src:
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)

    # Syntax check
    errors = []
    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            s = "".join(cell["source"])
            if s.strip():
                try:
                    ast.parse(s)
                except SyntaxError as e:
                    errors.append(f"Cell {ci}: {e}")

    if errors:
        print(f"SYNTAX ERRORS: {errors}")
        return

    print("Syntax: OK")

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Written: {OUT_NB}")

    # Push dir with FRESH kernel slug
    push = NOTEBOOKS_DIR / "push_v28"
    push.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push / "submission_v28_winner.ipynb")
    meta = {
        "id": "canivel/aimo3-v28-winner-config",
        "title": "AIMO3 v28 winner config",
        "code_file": "submission_v28_winner.ipynb",
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
    with io.open(push / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Push dir: {push}")
    print(f"FRESH kernel slug: aimo3-v28-winner-config")
    print(f"Submit: cd {push} && kaggle kernels push")


if __name__ == "__main__":
    main()
