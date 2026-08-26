"""Build v19: Two-phase split WITHOUT domain hints.

Based on v17 (verify cascade), adds two-phase solving:
- Phase 1: 4 quick attempts (25% budget max 3min)
  → 3+/4 agree → early exit (saves time for harder problems)
  → split → inject DISAGREE_CONTEXT into Phase 2
- Phase 2: 4 deep attempts (remaining budget)
- Failure retry: up to 2 retries for no-code attempts
- NO domain classification (CRITICAL_REVERSAL_2: classify-then-solve = -3.7 pts)
- Binary verify cascade carried over from v17

Differences from v37 (push_v18):
- v37 had domain hints in user_input (risky per CRITICAL_REVERSAL_2)
- v19 uses the same prompt for all attempts (no domain routing)
- Safer because all other changes from v17 are pure additions
"""

from __future__ import annotations
import io
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v17_verify.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v19_twophase_nodomain.ipynb"


# Two-phase helpers — no domain routing
TWOPHASE_HELPERS = '''
DISAGREE_CONTEXT = (
    'NOTE: Initial analysis produced conflicting results with answers: {answers}. '
    'At least some are wrong. Be extra thorough — verify every step with Python code '
    'and check your answer with a second independent method.'
)

PYTHON_MANDATORY = (
    'You MUST execute Python code to verify your answer before giving it. '
    'Do not provide a final boxed answer without code verification.'
)

def _run_batch(solver, user_input, seed_offset, n, deadline, consensus):
    """Run n attempts. Returns (detailed, valid)."""
    tasks = [(solver.cfg.system_prompt, i + seed_offset) for i in range(n)]
    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=solver.cfg.workers)
    try:
        futs = [ex.submit(solver._process_attempt, user_input, sp, ai, stop, deadline)
                for sp, ai in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
                detailed.append(r)
                if r['Answer'] is not None:
                    valid.append(r['Answer'])
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= consensus:
                    stop.set()
                    for ff in futs: ff.cancel()
                    break
            except Exception as e:
                print(f'Batch attempt error: {e}')
    finally:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
    return detailed, valid

'''


NEW_SOLVE_PROBLEM = '''    def solve_problem(self, problem):
        user_input = f'{problem} {self.cfg.preference_prompt}'
        print(f'\\nProblem: {problem[:200]}\\n')

        # Time budget
        elapsed = time.time() - self.notebook_start_time
        left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        total_budget = min(max(left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        overall_deadline = time.time() + total_budget
        print(f'Budget: {total_budget:.0f}s | Problems left: {self.problems_remaining}\\n')

        # ─── Phase 1: Quick triage (4 attempts, 25% budget max 3min) ───
        phase1_budget = min(180, total_budget * 0.25)
        phase1_deadline = time.time() + phase1_budget
        detailed_p1, valid_p1 = _run_batch(self, user_input, 0, 4, phase1_deadline, consensus=3)

        if valid_p1:
            counter_p1 = Counter(valid_p1)
            top, count = counter_p1.most_common(1)[0]
            if count >= 3:
                # Strong consensus — early exit
                print(f'Phase 1: CONSENSUS ({count}/4 → {top})')
                if detailed_p1:
                    df = pd.DataFrame(detailed_p1)
                    df['Entropy'] = df['Entropy'].round(3)
                    df['Answer'] = df['Answer'].astype('Int64')
                    display(df)
                print(f'\\nFinal Answer: {top}\\n')
                self.problems_remaining = max(0, self.problems_remaining - 1)
                return top
            print(f'Phase 1: SPLIT — {dict(counter_p1)}')
        else:
            print('Phase 1: No valid answers')

        # ─── Phase 2: Deep solving, with disagreement context if split ───
        if valid_p1 and len(set(valid_p1)) > 1:
            ans_list = ', '.join(str(a) for a in sorted(set(valid_p1)))
            context = DISAGREE_CONTEXT.format(answers=ans_list)
            user_p2 = f'{user_input} {context}'
        else:
            user_p2 = user_input

        detailed_p2, valid_p2 = _run_batch(self, user_p2, 4, 4, overall_deadline, consensus=4)

        all_detailed = detailed_p1 + detailed_p2
        all_valid = valid_p1 + valid_p2

        # ─── Failure-aware retry (up to 2 for no-code/no-answer) ───
        if time.time() < overall_deadline - 60:
            failed = [r for r in all_detailed if r['Answer'] is None or r.get('Python Calls', 0) == 0]
            n_retry = min(len(failed), 2)
            if n_retry > 0:
                print(f'Retrying {n_retry} low-quality attempts (no code/no answer)')
                retry_input = f'{user_input} {PYTHON_MANDATORY}'
                retry_d, retry_v = _run_batch(self, retry_input, 8, n_retry, overall_deadline, consensus=4)
                all_detailed.extend(retry_d)
                all_valid.extend(retry_v)

        if all_detailed:
            df = pd.DataFrame(all_detailed)
            df['Entropy'] = df['Entropy'].round(3)
            df['Answer'] = df['Answer'].astype('Int64')
            display(df)

        if not all_valid:
            print('\\nResult: 0\\n')
            self.problems_remaining = max(0, self.problems_remaining - 1)
            return 0

        self.problems_remaining = max(0, self.problems_remaining - 1)
        return self._select_answer(all_detailed, problem_text=user_input)
'''


def main():
    print("Building v19 two-phase (no domain) notebook from v17...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified_cells = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        # Replace solve_problem, injecting helpers before FOLLOWUP_PROMPT
        if "def solve_problem(self, problem):" in src:
            idx_start = src.find("    def solve_problem(self, problem):")
            idx_end = src.find("\n    def __del__", idx_start)
            if idx_start >= 0 and idx_end >= 0:
                idx_followup = src.find("FOLLOWUP_PROMPT")
                if idx_followup >= 0:
                    new_src = (
                        src[:idx_followup]
                        + TWOPHASE_HELPERS
                        + src[idx_followup:idx_start]
                        + NEW_SOLVE_PROBLEM
                        + "\n"
                        + src[idx_end:]
                    )
                else:
                    new_src = (
                        TWOPHASE_HELPERS
                        + src[:idx_start]
                        + NEW_SOLVE_PROBLEM
                        + "\n"
                        + src[idx_end:]
                    )
                nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
                print(f"Cell {i}: injected two-phase helpers + replaced solve_problem (no domain)")
                modified_cells += 1

        # Update version tag
        if "ULTIMATE v36" in src or "ULTIMATE v35" in src:
            new_src = src.replace(
                "# ULTIMATE v36: exact 44/50 params + T=0.8 + follow-up + binary verification cascade",
                "# ULTIMATE v38: exact 44/50 params + T=0.8 + two-phase (no domain) + disagree ctx + retry + verify"
            ).replace(
                "# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up",
                "# ULTIMATE v38: exact 44/50 params + T=0.8 + two-phase (no domain) + disagree ctx + retry + verify"
            ).replace(
                "print(f'CFG: ULTIMATE v36 | exact 44/50 + T=0.8 + binary verification (amanatar 44/50)')",
                "print(f'CFG: ULTIMATE v38 | exact 44/50 + T=0.8 + 2-phase (no domain) + disagree + retry + verify')"
            ).replace(
                "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
                "print(f'CFG: ULTIMATE v38 | exact 44/50 + T=0.8 + 2-phase (no domain) + disagree + retry + verify')"
            )
            if new_src != src:
                nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
                print(f"Cell {i}: version tag → ULTIMATE v38")
                modified_cells += 1

    print(f"Modified {modified_cells} cells")
    if modified_cells == 0:
        print("ERROR: no cells modified!")
        return

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {OUT_NB}")

    # Syntax validation
    import ast
    with io.open(OUT_NB, "r", encoding="utf-8") as f:
        nb_check = json.load(f)

    errors = []
    for i, cell in enumerate(nb_check["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if src.strip():
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    errors.append(f"Cell {i}: {e}")

    if errors:
        print(f"SYNTAX ERRORS:\n" + "\n".join(errors))
        return

    print("Syntax validation: PASSED")

    # Build push dir
    import shutil
    push_dir = NOTEBOOKS_DIR / "push_v19"
    push_dir.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push_dir / "submission_v19_twophase_nodomain.ipynb")

    # Kernel metadata
    with io.open(NOTEBOOKS_DIR / "push_v17" / "kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["code_file"] = "submission_v19_twophase_nodomain.ipynb"
    meta["title"] = "AIMO3 v38 two-phase no-domain"
    with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Push dir ready: {push_dir}")
    print(f"Submit: cd {push_dir} && kaggle kernels push")


if __name__ == "__main__":
    main()
