"""
Build v39: v27 diversity + binary answer verification tiebreaker.
Proven by amanatar (44/50): after entropy voting, verify top-2 candidates
at T=0.0 with 'CORRECT or WRONG'. Only override if #2 CORRECT and #1 WRONG.
"""

import json, io, ast, pathlib

ROOT    = pathlib.Path(__file__).parent.parent
SRC_NB  = ROOT / "notebooks/submission_v27_diverse.ipynb"
OUT_DIR = ROOT / "notebooks/push_v39"
OUT_NB  = OUT_DIR / "submission_v39.ipynb"

with io.open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

applied = []

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    changed = False

    # Patch 1: version comment
    if "print(f'CFG: ULTIMATE v35" in src:
        src = src.replace(
            "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
            "print(f'CFG: v39 | strategy diversity + 1/entropy + binary verify tiebreaker')"
        )
        applied.append("version"); changed = True

    # Patch 2: inject _verify_answer before _process_attempt
    VERIFY_METHOD = '''
    def _verify_answer(self, problem: str, answer: int) -> bool:
        """T=0.0 binary verify: ask model CORRECT or WRONG. Used as tiebreaker only."""
        try:
            verify_text = (
                f"Problem:\\n{problem}\\n\\n"
                f"Proposed answer: {answer}\\n\\n"
                "Is this the correct answer? Reply with ONLY one word: CORRECT or WRONG."
            )
            tool_cfg = ToolNamespaceConfig(name='python', description='', tools=[])
            msgs = self.template.apply_chat_template(self.cfg.system_prompt, verify_text, tool_cfg)
            conv = Conversation.from_messages(msgs)
            prompt_ids = self.encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
            max_tok = min(10, self.cfg.context_tokens - len(prompt_ids))
            if max_tok < 2:
                return True
            resp = self.client.completions.create(
                model=self.cfg.served_model_name, temperature=0.0,
                max_tokens=max_tok, prompt=prompt_ids, seed=self.cfg.seed,
                extra_body={'stop_token_ids': self.stop_token_ids}
            )
            text = resp.choices[0].text.strip().upper()
            return 'CORRECT' in text and 'WRONG' not in text
        except Exception:
            return True  # safe fallback

'''
    if "def _process_attempt(self, problem, system_prompt" in src and "_verify_answer" not in src:
        src = src.replace(
            "    def _process_attempt(self, problem, system_prompt",
            VERIFY_METHOD + "    def _process_attempt(self, problem, system_prompt"
        )
        applied.append("verify_method"); changed = True

    # Patch 3: replace _select_answer with verified version (index-based to avoid em dash matching issues)
    NEW_SELECT = (
        "def _select_answer(self, results, problem=None):\n"
        "        # Plain 1/entropy base\n"
        "        aw, av = defaultdict(float), defaultdict(int)\n"
        "        for r in results:\n"
        "            a, e = r['Answer'], r['Entropy']\n"
        "            if a is not None:\n"
        "                w = 1.0 / max(e, 1e-9)\n"
        "                aw[a] += w; av[a] += 1\n"
        "        scored = sorted([{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw], key=lambda x: x['score'], reverse=True)\n"
        "        df = pd.DataFrame([(s['answer'], s['votes'], round(s['score'],3)) for s in scored], columns=['Answer','Votes','Score'])\n"
        "        display(df)\n"
        "        if not scored: print('\\nFinal Answer: 0\\n'); return 0\n"
        "        # Binary verify tiebreaker (amanatar-proven, +1-2 pts)\n"
        "        # Only override when top-2 both have >=2 votes\n"
        "        if problem and len(scored) >= 2 and scored[1]['votes'] >= 2:\n"
        "            try:\n"
        "                v1 = self._verify_answer(problem, scored[0]['answer'])\n"
        "                v2 = self._verify_answer(problem, scored[1]['answer'])\n"
        "                if v2 and not v1:\n"
        "                    print(f'  [verify] override: {scored[0][\"answer\"]} WRONG -> {scored[1][\"answer\"]} CORRECT')\n"
        "                    print(f'\\nFinal Answer: {scored[1][\"answer\"]}\\n')\n"
        "                    return scored[1]['answer']\n"
        "                elif v1:\n"
        "                    print(f'  [verify] confirmed: {scored[0][\"answer\"]} CORRECT')\n"
        "            except Exception as exc:\n"
        "                print(f'  [verify] error (fallback to entropy): {exc}')\n"
        "        print(f'\\nFinal Answer: {scored[0][\"answer\"]}\\n')\n"
        "        return scored[0]['answer']"
    )
    SEL_MARKER = "def _select_answer(self, results):"
    SOL_MARKER = "def solve_problem(self, problem):"
    if SEL_MARKER in src and SOL_MARKER in src and "_select_answer" not in NEW_SELECT.split("\n")[0].replace("def _select_answer", ""):
        old_start = src.index(SEL_MARKER)
        old_end = src.index(SOL_MARKER)
        old_block = src[old_start:old_end]
        if "problem=None" not in old_block:  # not already patched
            src = src[:old_start] + NEW_SELECT + "\n\n    " + src[old_end:]
            applied.append("select_answer"); changed = True

    # Patch 4: pass problem to _select_answer
    if "return self._select_answer(detailed)" in src:
        src = src.replace(
            "return self._select_answer(detailed)",
            "return self._select_answer(detailed, problem=problem)"
        )
        applied.append("select_call"); changed = True

    if changed:
        cell["source"] = src.splitlines(keepends=True)

print(f"Applied: {applied}")
missing = [p for p in ["version","verify_method","select_answer","select_call"] if p not in applied]
if missing:
    print(f"ERROR — not applied: {missing}")
    exit(1)

# Syntax check
errors = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        s = "".join(cell["source"])
        if s.strip():
            try: ast.parse(s)
            except SyntaxError as e: errors.append(f"Cell {i}: {e}")
if errors:
    print(f"SYNTAX ERRORS: {errors}"); exit(1)
print("Syntax: OK")

OUT_DIR.mkdir(exist_ok=True)
with io.open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"Written: {OUT_NB}")

meta = {
    "id": "canivel/aimo3-v39-verify",
    "title": "AIMO3 v39 verify",
    "code_file": "submission_v39.ipynb",
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
print("Slug: aimo3-v39-verify")
