"""
Build v38: Remove strategy diversity — exact 44/50 notebook config.

Finding: Both 44/50 notebooks (nihilisticneuralnet + kaanyorgun) use
IDENTICAL prompt for all 8 attempts. No CODE_FIRST, no SMALL_CASES.

WHY removing diversity could help:
  - early_stop=4 fires when ANY answer appears 4 times
  - With diversity: 4 standard attempts may all agree on wrong answer,
    early_stop fires, kills the code-first/small-cases attempts
  - With no diversity: all 8 attempts independent samples of same approach,
    true self-consistency voting, no premature termination

Changes:
  - solve_problem: remove strategy_prefs split, use preference_prompt for ALL 8
  - Remove PREF_CODE_FIRST and PREF_SMALL_CASES entirely
  - Version comment updated
  - Everything else unchanged (vLLM params, voting, temperature, attempts)
"""

import json, io, ast, pathlib

ROOT    = pathlib.Path(__file__).parent.parent
SRC_NB  = ROOT / "notebooks/submission_v27_diverse.ipynb"
OUT_DIR = ROOT / "notebooks/push_v38"
OUT_NB  = OUT_DIR / "submission_v38.ipynb"

# Patch 1: Remove strategy diversity in solve_problem
OLD_STRATEGY = (
    "        strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2\n"
    "        user_inputs = [f'{problem} {p}' for p in strategy_prefs[:self.cfg.attempts]]\n"
    "        user_input = user_inputs[0]"
)
NEW_STRATEGY = (
    "        # No strategy diversity: identical prompt for all attempts (exact 44/50 config)\n"
    "        user_input = f'{problem}\\n\\n{self.cfg.preference_prompt}'"
)

# Patch 2: Use user_input (not user_inputs[ai]) when submitting futures
OLD_FUTURES = "            futs = [ex.submit(self._process_attempt, user_inputs[ai], sp, ai, stop, deadline) for sp, ai in tasks]"
NEW_FUTURES = "            futs = [ex.submit(self._process_attempt, user_input, sp, ai, stop, deadline) for sp, ai in tasks]"

# Patch 3: Version comment
OLD_VER = "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')"
NEW_VER = "print(f'CFG: v38 | NO strategy diversity + identical prompt all 8 attempts (exact 44/50 config)')"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with io.open(SRC_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patches = {
        OLD_STRATEGY: NEW_STRATEGY,
        OLD_FUTURES: NEW_FUTURES,
        OLD_VER: NEW_VER,
    }
    applied = {k: False for k in patches}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        changed = False
        for old, new in patches.items():
            if old in src:
                src = src.replace(old, new)
                applied[old] = True
                changed = True
        if changed:
            cell["source"] = src.splitlines(keepends=True)

    missing = [k for k, v in applied.items() if not v]
    if missing:
        print("ERROR: patches not applied:")
        for m in missing:
            print(f"  {repr(m[:100])}")
        return

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
        print(f"SYNTAX ERRORS: {errors}")
        return

    print("All patches applied. Syntax: OK")

    # Verify the change looks right
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "user_input = f'{problem}" in src and "preference_prompt" in src:
                for line in src.split("\n"):
                    if "user_input" in line and "preference_prompt" in line:
                        print(f"  Strategy fix: {line.strip()}")
            if "futs = [ex.submit" in src:
                for line in src.split("\n"):
                    if "futs = [ex.submit" in line:
                        print(f"  Futures fix:  {line.strip()}")

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Written: {OUT_NB}")

    meta = {
        "id": "canivel/aimo3-v38-nodiversity",
        "title": "AIMO3 v38 no diversity",
        "code_file": "submission_v38.ipynb",
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
    print(f"Slug: aimo3-v38-nodiversity")


if __name__ == "__main__":
    main()
