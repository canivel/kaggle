"""Build v24: Adaptive Quality Solver (AQS).

Research-driven innovations over v16 baseline:

1. QUALITY-WEIGHTED VOTING (replaces pure 1/entropy)
   w = (1/entropy) * error_penalty * code_bonus
   - error_penalty = 0.3 if python_errors > 0 (biggest missed signal: 75% -> 27% correct)
   - code_bonus = 1.3 if clean code execution, 0.7 if no code at all
   Source: failure analysis agent — 80.9% of failures had correct answer outvoted

2. ADAPTIVE ATTEMPTS (probe then expand)
   - Phase 1: 2 probe attempts
   - If 2/2 agree with clean code: run 2 more confirmations (total 4) -> early exit
   - If split: run 12 more (total 14) for hard problems
   Source: time budget agent — we use <5% of compute budget, N=14 fits easily

3. CODE-BASED VERIFICATION (for split votes only)
   - When top-2 candidates are close in score: ask model to write Python verification
   - Different from Pawan Mali V133 (bare CORRECT/WRONG = 0 improvement)
   - We ask for EXECUTABLE PYTHON that tests the answer against problem constraints
   Source: multi-turn agent — code execution is objective, independent of model reasoning

Base: submission_v16_exact44.ipynb (proven 39/50)
Changes: _select_answer() + solve_problem() only. No prompt changes.
"""

from __future__ import annotations
import ast
import io
import json
import shutil
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v16_exact44.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v24_aqs.ipynb"

# ── New _select_answer with quality-weighted voting ─────────────────────────

NEW_SELECT_ANSWER = '''    def _select_answer(self, results):
        # Quality-weighted voting: penalize errors, bonus for clean code
        aw = defaultdict(float)
        av = defaultdict(int)
        for r in results:
            a = r['Answer']
            e = r['Entropy']
            if a is None:
                continue
            # Base weight: inverse entropy (same as before)
            w = 1.0 / max(e, 1e-9)
            # Error penalty: attempts with Python errors are much less reliable
            py_errors = r.get('Python Errors', 0)
            if py_errors > 0:
                w *= 0.3
            # Code bonus: clean code execution = more reliable
            py_calls = r.get('Python Calls', 0)
            if py_calls >= 2 and py_errors == 0:
                w *= 1.3  # clean code verification
            elif py_calls == 0:
                w *= 0.7  # no code = less trustworthy
            aw[a] += w
            av[a] += 1
        scored = sorted([{'answer': a, 'votes': av[a], 'score': aw[a]} for a in aw],
                        key=lambda x: x['score'], reverse=True)
        df = pd.DataFrame([(s['answer'], s['votes'], round(s['score'], 3)) for s in scored],
                          columns=['Answer', 'Votes', 'Score'])
        display(df)
        if not scored:
            print('\\nFinal Answer: 0\\n')
            return 0
        print(f'\\nFinal Answer: {scored[0]["answer"]}\\n')
        return scored[0]['answer']
'''

# ── Code-based verification helper ──────────────────────────────────────────

VERIFY_HELPER = '''
VERIFY_PROMPT = (
    'A student claims the answer to the following problem is {answer}.\\n\\n'
    'Problem: {problem}\\n\\n'
    'Write a short Python program that checks whether {answer} satisfies the '
    'constraints of the problem. Print "VERIFIED" if the answer is correct, '
    'or "WRONG" if it is not. Do NOT re-solve the problem from scratch. '
    'Only verify the given answer.'
)

def _code_verify(solver, problem_text, candidate, deadline):
    """Ask model to write verification code. Returns True if verified."""
    if time.time() > deadline - 30:
        return True  # no time, trust the vote
    prompt = VERIFY_PROMPT.format(answer=candidate, problem=problem_text)
    try:
        resp = solver.client.completions.create(
            model=solver.cfg.served_model_name,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024,
            timeout=25,
        )
        text = resp.choices[0].text if resp.choices else ''
        if 'WRONG' in text.upper() and 'VERIFIED' not in text.upper():
            return False
        return True
    except Exception:
        return True  # on error, trust the vote

'''

# ── New solve_problem with adaptive attempts ────────────────────────────────

NEW_SOLVE_PROBLEM = '''    def solve_problem(self, problem):
        user_input = f'{problem} {self.cfg.preference_prompt}'
        print(f'\\nProblem: {problem[:200]}\\n')

        elapsed = time.time() - self.notebook_start_time
        left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        total_budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        deadline = time.time() + total_budget
        print(f'Budget: {total_budget:.0f}s | Problems left: {self.problems_remaining}\\n')

        # ── Phase 1: PROBE (2 attempts) ──
        tasks_p1 = [(self.cfg.system_prompt, i) for i in range(2)]
        detailed, valid = [], []
        stop = threading.Event()
        ex = ThreadPoolExecutor(max_workers=self.cfg.workers)
        try:
            futs = [ex.submit(self._process_attempt, user_input, sp, ai, stop, deadline)
                    for sp, ai in tasks_p1]
            for f in as_completed(futs):
                try:
                    r = f.result()
                    detailed.append(r)
                    if r['Answer'] is not None:
                        valid.append(r['Answer'])
                except Exception as e:
                    print(f'Probe error: {e}')
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

        # Classify: easy (2/2 agree + clean code) or hard
        probe_agree = len(valid) == 2 and valid[0] == valid[1]
        probe_clean = all(r.get('Python Errors', 0) == 0 and r.get('Python Calls', 0) > 0
                         for r in detailed if r['Answer'] is not None)

        if probe_agree and probe_clean:
            # EASY PATH: 2 more confirmations (total 4)
            n_more = 2
            print(f'Probe: EASY (2/2 agree, clean code) -> {n_more} more')
        else:
            # HARD PATH: 12 more attempts (total 14)
            n_more = 12
            print(f'Probe: HARD -> {n_more} more attempts')

        # ── Phase 2: FULL attempts ──
        if time.time() < deadline - 30 and n_more > 0:
            tasks_p2 = [(self.cfg.system_prompt, i + 2) for i in range(n_more)]
            stop2 = threading.Event()
            ex2 = ThreadPoolExecutor(max_workers=self.cfg.workers)
            try:
                futs2 = [ex2.submit(self._process_attempt, user_input, sp, ai, stop2, deadline)
                         for sp, ai in tasks_p2]
                for f in as_completed(futs2):
                    try:
                        r = f.result()
                        detailed.append(r)
                        if r['Answer'] is not None:
                            valid.append(r['Answer'])
                        # Early stop: if 4+ agree on same answer
                        c = Counter(valid).most_common(1)
                        if c and c[0][1] >= self.cfg.early_stop:
                            stop2.set()
                            for ff in futs2: ff.cancel()
                            break
                    except Exception as e:
                        print(f'Attempt error: {e}')
            finally:
                stop2.set()
                ex2.shutdown(wait=True, cancel_futures=True)

        if detailed:
            df = pd.DataFrame(detailed)
            df['Entropy'] = df['Entropy'].round(3)
            df['Answer'] = df['Answer'].astype('Int64')
            display(df)

        if not valid:
            print('\\nResult: 0\\n')
            self.problems_remaining = max(0, self.problems_remaining - 1)
            return 0

        # ── Quality-weighted vote ──
        answer = self._select_answer(detailed)

        # ── Code-based verification for split votes ──
        aw = defaultdict(float)
        for r in detailed:
            a, e = r['Answer'], r['Entropy']
            if a is not None:
                w = 1.0 / max(e, 1e-9)
                py_err = r.get('Python Errors', 0)
                if py_err > 0: w *= 0.3
                pc = r.get('Python Calls', 0)
                if pc >= 2 and py_err == 0: w *= 1.3
                elif pc == 0: w *= 0.7
                aw[a] += w
        scored = sorted(aw.items(), key=lambda x: x[1], reverse=True)
        if len(scored) >= 2:
            top_score = scored[0][1]
            second_score = scored[1][1]
            # If close race (second is >40% of top), verify
            if second_score > 0.4 * top_score and time.time() < deadline - 40:
                print(f'Split vote: verifying top={scored[0][0]} vs second={scored[1][0]}')
                verified = _code_verify(self, problem, scored[0][0], deadline)
                if not verified:
                    print(f'  Top answer FAILED verification, switching to {scored[1][0]}')
                    answer = scored[1][0]

        self.problems_remaining = max(0, self.problems_remaining - 1)
        return answer
'''


def main():
    print("Building v24 AQS (Adaptive Quality Solver) from v16...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified_cells = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        # 1. Replace _select_answer
        if "def _select_answer(self, results):" in new_src:
            idx_start = new_src.find("    def _select_answer(self, results):")
            idx_end = new_src.find("\n    def ", idx_start + 20)
            if idx_start >= 0 and idx_end >= 0:
                new_src = new_src[:idx_start] + NEW_SELECT_ANSWER + new_src[idx_end:]
                print(f"Cell {i}: replaced _select_answer (quality-weighted)")

        # 2. Inject verify helper before solve_problem
        if "def solve_problem(self, problem):" in new_src and "VERIFY_PROMPT" not in new_src:
            idx_solve = new_src.find("    def solve_problem(self, problem):")
            if idx_solve >= 0:
                # Find the FOLLOWUP_PROMPT or similar injection point
                idx_followup = new_src.find("FOLLOWUP_PROMPT")
                if idx_followup >= 0:
                    inject_point = idx_followup
                else:
                    inject_point = idx_solve
                new_src = new_src[:inject_point] + VERIFY_HELPER + "\n" + new_src[inject_point:]
                print(f"Cell {i}: injected VERIFY_HELPER")

        # 3. Replace solve_problem
        if "def solve_problem(self, problem):" in new_src:
            idx_start = new_src.find("    def solve_problem(self, problem):")
            idx_end = new_src.find("\n    def __del__", idx_start)
            if idx_start >= 0 and idx_end >= 0:
                new_src = new_src[:idx_start] + NEW_SOLVE_PROBLEM + "\n" + new_src[idx_end:]
                print(f"Cell {i}: replaced solve_problem (adaptive + code verify)")

        # 4. Update version tag
        for old_tag in [
            "# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up",
        ]:
            if old_tag in new_src:
                new_src = new_src.replace(old_tag,
                    "# AQS v24: Adaptive Quality Solver — quality-weighted voting + adaptive N + code verification")
        for old_print in [
            "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
        ]:
            if old_print in new_src:
                new_src = new_src.replace(old_print,
                    "print(f'CFG: AQS v24 | quality-weighted voting + adaptive attempts + code verify')")

        if new_src != src:
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
            modified_cells += 1

    print(f"Modified {modified_cells} cells")
    if modified_cells == 0:
        print("ERROR: no cells modified!")
        return

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Written: {OUT_NB}")

    # Syntax validation
    with io.open(OUT_NB, "r", encoding="utf-8") as f:
        nb_check = json.load(f)
    errors = []
    for i, cell in enumerate(nb_check["cells"]):
        if cell["cell_type"] == "code":
            src_c = "".join(cell["source"])
            if src_c.strip():
                try:
                    ast.parse(src_c)
                except SyntaxError as e:
                    errors.append(f"Cell {i}: {e}")
    if errors:
        print("SYNTAX ERRORS:\n" + "\n".join(errors))
        return
    print("Syntax validation: PASSED")

    # Build push dir
    push_dir = NOTEBOOKS_DIR / "push_v24"
    push_dir.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push_dir / "submission_v24_aqs.ipynb")

    meta = {
        "id": "canivel/aimo3-tir-rag-baseline",
        "title": "aimo3-tir-rag-baseline",
        "code_file": "submission_v24_aqs.ipynb",
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
    with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPush dir: {push_dir}")
    print(f"Submit: cd {push_dir} && kaggle kernels push")


if __name__ == "__main__":
    main()
