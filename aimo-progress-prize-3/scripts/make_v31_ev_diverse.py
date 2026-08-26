"""Build v31: Strategy diversity (v27) + EV voting.

Only change from v27: _select_answer uses execution-verified entropy weights.
- Clean code execution (python_calls > 0, python_errors == 0): 10x weight
- Code ran with errors: 0.1x weight
- No code execution: 0.2x weight

Monte Carlo validated: +1.545 expected problems (95% CI [1.515, 1.575]).
v27 scored 40/50 (+1 over v16 baseline). This stacks EV on top.
"""

import json
import io
import ast
import shutil
import re
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v27_diverse.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v31_ev_diverse.ipynb"

OLD_SELECT = '''    def _select_answer(self, results):
        # Plain 1/entropy â€" same as 43/50 base
        aw, av = defaultdict(float), defaultdict(int)
        for r in results:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                w = 1.0 / max(e, 1e-9)
                aw[a] += w; av[a] += 1
        scored = sorted([{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw], key=lambda x: x['score'], reverse=True)
        df = pd.DataFrame([(s['answer'], s['votes'], round(s['score'],3)) for s in scored], columns=['Answer','Votes','Score'])
        display(df)
        if not scored: print('\\nFinal Answer: 0\\n'); return 0
        print(f'\\nFinal Answer: {scored[0]["answer"]}\\n')
        return scored[0]['answer']'''

NEW_SELECT = '''    def _select_answer(self, results):
        # EV voting: execution-verified entropy weighting
        # Monte Carlo validated: +1.545 expected problems (95% CI [1.515, 1.575])
        aw, av = defaultdict(float), defaultdict(int)
        for r in results:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                base_w = 1.0 / max(e, 1e-9)
                pc, pe = r.get('Python Calls', 0), r.get('Python Errors', 0)
                if pc > 0 and pe == 0:
                    ev_mult = 10.0   # clean code execution
                elif pc > 0:
                    ev_mult = 0.1    # code ran with errors
                else:
                    ev_mult = 0.2    # no code execution
                aw[a] += base_w * ev_mult
                av[a] += 1
        scored = sorted([{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw], key=lambda x: x['score'], reverse=True)
        df = pd.DataFrame([(s['answer'], s['votes'], round(s['score'],3)) for s in scored], columns=['Answer','Votes','Score'])
        display(df)
        if not scored: print('\\nFinal Answer: 0\\n'); return 0
        print(f'\\nFinal Answer: {scored[0]["answer"]}\\n')
        return scored[0]['answer']'''


def main():
    print("Building v31 (strategy diversity + EV voting) from v27...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    replaced = False
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "_select_answer" in src and "1/entropy" in src:
            if OLD_SELECT not in src:
                print(f"WARNING: Cell {i} has _select_answer but exact match failed. Trying flexible match...")
                # Try to replace just the body with a regex
                pattern = re.compile(
                    r'(    def _select_answer\(self, results\):.*?return scored\[0\]\[.answer.\]\n)',
                    re.DOTALL
                )
                m = pattern.search(src)
                if m:
                    new_src = src[:m.start()] + NEW_SELECT + "\n" + src[m.end():]
                    nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
                    print(f"Cell {i}: _select_answer -> EV voting (regex match)")
                    replaced = True
            else:
                new_src = src.replace(OLD_SELECT, NEW_SELECT)
                nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
                print(f"Cell {i}: _select_answer -> EV voting (exact match)")
                replaced = True
            break

    if not replaced:
        print("ERROR: Could not find _select_answer to replace!")
        return

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

    # Push dir with fresh slug
    push = NOTEBOOKS_DIR / "push_v31"
    push.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push / "submission_v31_ev_diverse.ipynb")
    meta = {
        "id": "canivel/aimo3-v31-ev-diverse",
        "title": "AIMO3 v31 EV diverse",
        "code_file": "submission_v31_ev_diverse.ipynb",
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
    print(f"Slug: aimo3-v31-ev-diverse")
    print(f"Submit: cd {push} && kaggle kernels push")


if __name__ == "__main__":
    main()
