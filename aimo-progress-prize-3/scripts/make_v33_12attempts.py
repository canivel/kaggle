"""
Build v33: strategy diversity + 12 attempts + batch=128

Why 12 attempts:
  Hard problems have ~20% solve rate per attempt.
  P(>=1 correct | 8 attempts) = 83%
  P(>=1 correct | 12 attempts) = 93%  <- +10% coverage on hard problems

Why batch=128:
  12 attempts x larger sequences fits H100 with batch=128.
  Proven by amanatar (644 votes, author confirmed at 44 LB).

What stays the same:
  - strategy diversity (4 standard + 2 code-first + 2 small-cases)
    Extended to 12: 5 standard + 4 code-first + 3 small-cases
  - plain 1/entropy voting (validated: EV voting hurts)
  - exact same vLLM flags, T=0.8, ctx=65536
  - early_stop=4 (same, so easy problems still fast)
"""

import json, io, shutil, ast, re, pathlib

ROOT    = pathlib.Path(__file__).parent.parent
SRC_NB  = ROOT / "notebooks/submission_v27_diverse.ipynb"
OUT_DIR = ROOT / "notebooks/push_v33"
OUT_NB  = OUT_DIR / "submission_v33.ipynb"

# ── Patches ────────────────────────────────────────────────────────────────────

# 1. CFG: attempts=12, batch_size=128
OLD_CFG = "    batch_size = 256\n    early_stop = 4\n    attempts = 8"
NEW_CFG = "    batch_size = 128\n    early_stop = 4\n    attempts = 12"

# 2. solve_problem: extend strategy_prefs to 12
#    was: [preference]*4 + [CODE_FIRST]*2 + [SMALL_CASES]*2  (8 total)
#    now: [preference]*5 + [CODE_FIRST]*4 + [SMALL_CASES]*3  (12 total)
OLD_PREFS = "        strategy_prefs = [self.cfg.preference_prompt]*4 + [PREF_CODE_FIRST]*2 + [PREF_SMALL_CASES]*2"
NEW_PREFS = "        strategy_prefs = [self.cfg.preference_prompt]*5 + [PREF_CODE_FIRST]*4 + [PREF_SMALL_CASES]*3"

# 3. Version comment
OLD_VER = "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')"
NEW_VER = "print(f'CFG: v33 | strategy diversity (12 attempts, batch=128) + 1/entropy voting')"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with io.open(SRC_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    patches = {
        OLD_CFG: NEW_CFG,
        OLD_PREFS: NEW_PREFS,
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
            print(f"  {repr(m[:80])}")
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

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Written: {OUT_NB}")

    meta = {
        "id": "canivel/aimo3-v33-12attempts",
        "title": "AIMO3 v33 12 attempts",
        "code_file": "submission_v33.ipynb",
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

    print(f"Slug: aimo3-v33-12attempts")


if __name__ == "__main__":
    main()
