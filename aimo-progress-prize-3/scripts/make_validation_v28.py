"""
Build validation notebook for v28_mathtools.
Takes v28_mathtools notebook, replaces the competition inference cell
with validation against 10 reference problems.
Compares v27 baseline (1/entropy, no math tools) vs v28 (math tools + domain prompts).
"""
import json, io, pathlib, shutil

ROOT    = pathlib.Path(__file__).parent.parent
SRC_NB  = ROOT / "notebooks/submission_v28_mathtools.ipynb"
OUT_DIR = ROOT / "notebooks/push_validation_v28"
OUT_NB  = OUT_DIR / "validation_v28_mathtools.ipynb"

with io.open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Load reference bench inline
with open(ROOT / "data/aimo3_reference_bench.json") as f:
    bench = json.load(f)
BENCH_INLINE = json.dumps(bench, indent=2)

# Remove the last cell (competition inference server) and replace with validation
# Find and remove the inference_server cell
cells = nb["cells"]
new_cells = []
for cell in cells:
    src = "".join(cell.get("source", []))
    if "inference_server" in src or "KAGGLE_IS_COMPETITION_RERUN" in src:
        continue  # drop competition cell
    new_cells.append(cell)

def code(src):
    return {"cell_type": "code", "source": src, "metadata": {}, "outputs": [], "execution_count": None}

def md(src):
    return {"cell_type": "markdown", "source": src, "metadata": {}}

# Add validation cells
new_cells.append(code("from transformers import set_seed\nset_seed(CFG.seed)"))
new_cells.append(md("## Validation: v27 baseline vs v28 math tools\n\n10 AIMO3 reference problems."))

new_cells.append(code(f"""\
import json
BENCH = {BENCH_INLINE}
print(f'Loaded {{len(BENCH)}} reference problems')
"""))

new_cells.append(code("""\
def select_baseline(results):
    \"\"\"Plain 1/entropy — v27 baseline (no math tools, no domain prefs)\"\"\"
    from collections import defaultdict
    aw = defaultdict(float)
    for r in results:
        a, e = r['Answer'], r['Entropy']
        if a is not None:
            aw[a] += 1.0 / max(e, 1e-9)
    return max(aw, key=aw.get) if aw else 0

def select_v28(results):
    \"\"\"v28: same 1/entropy but with math tools + domain prompts active\"\"\"
    from collections import defaultdict
    aw = defaultdict(float)
    for r in results:
        a, e = r['Answer'], r['Entropy']
        if a is not None:
            aw[a] += 1.0 / max(e, 1e-9)
    return max(aw, key=aw.get) if aw else 0
"""))

new_cells.append(code("""\
solver = AIMO3Solver(CFG)
print('Solver ready')
"""))

new_cells.append(code("""\
import pandas as pd
results_log = []
baseline_correct = v28_correct = 0

for i, item in enumerate(BENCH):
    pid      = item['id']
    problem  = item['problem']
    expected = item['answer']
    source   = item.get('source', f'P{i+1}')
    print(f'\\n{"="*60}')
    print(f'Problem {i+1}/10 | {source}')
    print(f'Expected: {expected}')
    print(f'{"="*60}')

    attempt_results = solver.solve(problem)

    ans_b  = select_baseline(attempt_results)
    ans_v28 = select_v28(attempt_results)

    b_ok  = int(ans_b)   == int(expected)
    v_ok  = int(ans_v28) == int(expected)
    if b_ok: baseline_correct += 1
    if v_ok: v28_correct += 1

    df = pd.DataFrame(attempt_results)
    df['Entropy'] = df['Entropy'].apply(lambda x: round(x,3) if x != float('inf') else 'inf')
    df['Answer']  = df['Answer'].astype('Int64')
    display(df)

    print(f'  Baseline (v27 1/entropy):   {ans_b}   {"CORRECT" if b_ok else "WRONG"}')
    print(f'  v28 (math tools + domain):  {ans_v28}  {"CORRECT" if v_ok else "WRONG"}')
    print(f'  Running: baseline={baseline_correct}/{i+1}  v28={v28_correct}/{i+1}')

    results_log.append({
        'id': pid, 'source': source, 'expected': expected,
        'ans_baseline': ans_b, 'ans_v28': ans_v28,
        'baseline_correct': b_ok, 'v28_correct': v_ok,
        'attempts': attempt_results,
    })

solver.shutdown()
"""))

new_cells.append(code("""\
print(f'\\n{"="*60}')
print(f'VALIDATION: v27 baseline vs v28 math tools')
print(f'{"="*60}')
print(f'  v27 baseline (1/entropy, no math tools): {baseline_correct}/10')
print(f'  v28 math tools + domain prompts:          {v28_correct}/10')
print(f'  Delta:                                    {v28_correct - baseline_correct:+d}')
print(f'{"="*60}')

summary = pd.DataFrame([{
    'Problem': r['source'],
    'Expected': r['expected'],
    'Baseline': r['ans_baseline'],
    'B_ok': 'Y' if r['baseline_correct'] else 'N',
    'v28': r['ans_v28'],
    'V28_ok': 'Y' if r['v28_correct'] else 'N',
    'Same': 'Y' if r['ans_baseline'] == r['ans_v28'] else 'DIFF',
} for r in results_log])
display(summary)

import json
with open('/kaggle/working/validation_v28_results.json', 'w') as f:
    json.dump(results_log, f, indent=2)
print('Saved to /kaggle/working/validation_v28_results.json')
"""))

nb["cells"] = new_cells

OUT_DIR.mkdir(exist_ok=True)
with io.open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"Written: {OUT_NB}")

# Validate syntax
import ast
errors = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        s = "".join(cell["source"])
        if s.strip():
            try: ast.parse(s)
            except SyntaxError as e: errors.append(f"Cell {i}: {e}")
print(f"Syntax: {'OK' if not errors else errors}")

meta = {
    "id": "canivel/aimo3-validation-v28-mathtools",
    "title": "AIMO3 Validation v28 MathTools",
    "code_file": "validation_v28_mathtools.ipynb",
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
print(f"Slug: aimo3-validation-v28-mathtools")
