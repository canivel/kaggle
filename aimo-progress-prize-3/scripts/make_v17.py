"""Create v17 notebook with binary answer verification."""
import json, io

with io.open('f:/kaggle/aimo-progress-prize-3/notebooks/submission_v16_exact44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The _verify_answer method + updated _select_answer
# Note: \n inside strings must be literal backslash+n for Python source
bsn = chr(92) + 'n'  # literal \n (2 chars)

verify_code = (
    '    def _verify_answer(self, problem_text, answer):\n'
    '        """Binary verification at T=0.0 (amanatar 44/50 approach)."""\n'
    '        prompt = (\n'
    f'            f"Problem:{bsn}{{problem_text}}{bsn}{bsn}"\n'
    f'            f"Proposed answer: {{answer}}{bsn}{bsn}"\n'
    f'            "Check the answer carefully.{bsn}Reply with only ONE word:{bsn}CORRECT or WRONG"\n'
    '        )\n'
    '        try:\n'
    '            prompt_ids = self.encoding.encode(prompt)\n'
    '            resp = self.client.completions.create(\n'
    '                model=self.cfg.served_model_name,\n'
    '                prompt=prompt_ids,\n'
    '                temperature=0.0,\n'
    '                max_tokens=5\n'
    '            )\n'
    '            text = resp.choices[0].text.strip().upper()\n'
    '            return "CORRECT" in text and "WRONG" not in text\n'
    '        except Exception:\n'
    '            return False\n'
    '\n'
    '    def _select_answer(self, results, problem_text=None):\n'
    '        # 1/entropy weighted vote\n'
    '        aw, av = defaultdict(float), defaultdict(int)\n'
    '        for r in results:\n'
    '            a, e = r[\'Answer\'], r[\'Entropy\']\n'
    '            if a is not None:\n'
    '                w = 1.0 / max(e, 1e-9)\n'
    '                aw[a] += w; av[a] += 1\n'
    '        scored = sorted([{\'answer\': a, \'votes\': av[a], \'score\': aw[a]} for a in aw], key=lambda x: x[\'score\'], reverse=True)\n'
    '        df = pd.DataFrame([(s[\'answer\'], s[\'votes\'], round(s[\'score\'],3)) for s in scored], columns=[\'Answer\',\'Votes\',\'Score\'])\n'
    '        display(df)\n'
    f'        if not scored: print(\'{bsn}Final Answer: 0{bsn}\'); return 0\n'
    '        # Binary verification when vote is split (< 4 of 8 agree on top answer)\n'
    '        if problem_text and len(scored) > 1 and scored[0][\'votes\'] < 4:\n'
    '            for s in scored[:3]:\n'
    '                a = s[\'answer\']\n'
    '                print(f\'Verifying {a}...\')\n'
    '                if self._verify_answer(problem_text, a):\n'
    f'                    print(f\'{bsn}Final Answer (verified): {{a}}{bsn}\')\n'
    '                    return a\n'
    '            print(\'No candidate verified, using entropy vote\')\n'
    f'        print(f\'{bsn}Final Answer: {{scored[0]["answer"]}}{bsn}\')\n'
    '        return scored[0][\'answer\']\n'
)

changed = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Must have both _select_answer AND solve_problem (preserve solve_problem!)
        if '    def _select_answer' in src and '    def solve_problem' in src:
            idx_select = src.find('    def _select_answer')
            idx_solve = src.find('    def solve_problem')
            idx_del = src.find('\n    def __del__', idx_select)
            # Update solve_problem to pass problem_text to _select_answer
            old_solve = src[idx_solve:idx_del]
            updated_solve = old_solve.replace(
                'return self._select_answer(detailed)',
                'return self._select_answer(detailed, problem_text=user_input)'
            )
            new_src = src[:idx_select] + verify_code + updated_solve + src[idx_del:]
            nb['cells'][i]['source'] = new_src.splitlines(keepends=True)
            print(f'Cell {i}: inserted _verify_answer + new _select_answer, preserved solve_problem')
            changed = True

# Update version tag
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'ULTIMATE v35' in src:
            new_src = src.replace(
                '# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up',
                '# ULTIMATE v36: exact 44/50 params + T=0.8 + follow-up + binary verification cascade'
            ).replace(
                "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
                "print(f'CFG: ULTIMATE v36 | exact 44/50 + T=0.8 + binary verification (amanatar 44/50)')"
            )
            nb['cells'][i]['source'] = new_src.splitlines(keepends=True)
            print(f'Cell {i}: version updated to v36')

if not changed:
    print('ERROR: _select_answer method not found!')

with io.open('f:/kaggle/aimo-progress-prize-3/notebooks/submission_v17_verify.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved submission_v17_verify.ipynb')
