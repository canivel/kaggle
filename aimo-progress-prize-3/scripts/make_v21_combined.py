"""Build v21: Combined BEST notebook.

Everything together in one submission:
  1. Exact 44/50 vLLM params (ctx=65536, batch=256, gpu=0.93 for Eagle-3)
  2. T=0.8, 5-step prompt (base 44/50 config)
  3. Eagle-3 speculative decoding (+36% speed → more tokens per budget)
  4. Two-phase solving (no domain routing): 4 quick → consensus check → 4 deep
  5. DISAGREE_CONTEXT injected in Phase 2 when Phase 1 splits
  6. Failure-aware retry (up to 2 retries for no-code attempts)
  7. Binary verification cascade from v17 (amanatar 44/50 proven)

Base: submission_v17_verify.ipynb (has binary verify cascade already)
Add: Eagle-3 + two-phase (no domain)

Risk: Medium (Eagle-3 untested on Kaggle). Fallback: push_v17 (verify only).
"""

from __future__ import annotations
import io
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v17_verify.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v21_combined.ipynb"

EAGLE3_PATH = "/kaggle/input/gpt-oss-120b-eagle3-throughput"
EAGLE3_DATASET = "eliork/gpt-oss-120b-eagle3-throughput"


TWOPHASE_HELPERS = '''
DISAGREE_CONTEXT = (
    'NOTE: Initial analysis produced conflicting results with answers: {answers}. '
    'At least some are wrong. Be extra thorough -- verify every step with Python code '
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

        # Phase 1: Quick triage (4 attempts, 25% budget max 3min)
        phase1_budget = min(180, total_budget * 0.25)
        phase1_deadline = time.time() + phase1_budget
        detailed_p1, valid_p1 = _run_batch(self, user_input, 0, 4, phase1_deadline, consensus=3)

        if valid_p1:
            counter_p1 = Counter(valid_p1)
            top, count = counter_p1.most_common(1)[0]
            if count >= 3:
                # Strong consensus -- early exit saves time for harder problems
                print(f'Phase 1: CONSENSUS ({count}/4 -> {top})')
                if detailed_p1:
                    df = pd.DataFrame(detailed_p1)
                    df['Entropy'] = df['Entropy'].round(3)
                    df['Answer'] = df['Answer'].astype('Int64')
                    display(df)
                print(f'\\nFinal Answer: {top}\\n')
                self.problems_remaining = max(0, self.problems_remaining - 1)
                return top
            print(f'Phase 1: SPLIT -- {dict(counter_p1)}')
        else:
            print('Phase 1: No valid answers')

        # Phase 2: Deep solving with disagreement context when split
        if valid_p1 and len(set(valid_p1)) > 1:
            ans_list = ', '.join(str(a) for a in sorted(set(valid_p1)))
            context = DISAGREE_CONTEXT.format(answers=ans_list)
            user_p2 = f'{user_input} {context}'
        else:
            user_p2 = user_input

        detailed_p2, valid_p2 = _run_batch(self, user_p2, 4, 4, overall_deadline, consensus=4)

        all_detailed = detailed_p1 + detailed_p2
        all_valid = valid_p1 + valid_p2

        # Failure-aware retry (up to 2 for no-code/no-answer attempts)
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
    print("Building v21 COMBINED (Eagle-3 + two-phase + verify) from v17...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified_cells = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        # 1. gpu_memory_utilization 0.96 -> 0.93 (Eagle-3 needs ~0.6GB)
        if "gpu_memory_utilization = 0.96" in new_src:
            new_src = new_src.replace(
                "gpu_memory_utilization = 0.96",
                "gpu_memory_utilization = 0.93  # Eagle-3 draft model needs ~0.6GB extra VRAM"
            )

        # 2. Add Eagle-3 speculative-config to vLLM cmd
        if "'--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'" in new_src:
            eagle3_snippet = (
                "'--async-scheduling', '--disable-log-stats', '--enable-prefix-caching',\n"
                "            # Eagle-3 speculative decoding (+36-42% speed, same output distribution)\n"
                f"            '--speculative-config', json.dumps({{\n"
                f"                'method': 'eagle3',\n"
                f"                'model': '{EAGLE3_PATH}',\n"
                f"                'num_speculative_tokens': 3,\n"
                f"                'draft_tensor_parallel_size': 1,\n"
                f"            }})"
            )
            new_src = new_src.replace(
                "'--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'",
                eagle3_snippet
            )

        # 3. Inject two-phase helpers + replace solve_problem (no domain hints)
        if "def solve_problem(self, problem):" in new_src:
            idx_start = new_src.find("    def solve_problem(self, problem):")
            idx_end = new_src.find("\n    def __del__", idx_start)
            idx_followup = new_src.find("FOLLOWUP_PROMPT")
            if idx_start >= 0 and idx_end >= 0 and idx_followup >= 0:
                new_src = (
                    new_src[:idx_followup]
                    + TWOPHASE_HELPERS
                    + new_src[idx_followup:idx_start]
                    + NEW_SOLVE_PROBLEM
                    + "\n"
                    + new_src[idx_end:]
                )

        # 4. Update version tag
        for old_tag in [
            "# ULTIMATE v36: exact 44/50 params + T=0.8 + follow-up + binary verification cascade",
            "# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up",
        ]:
            if old_tag in new_src:
                new_src = new_src.replace(
                    old_tag,
                    "# ULTIMATE v40: exact 44/50 + T=0.8 + Eagle-3 + two-phase (no domain) + disagree ctx + retry + verify"
                )
        for old_print in [
            "print(f'CFG: ULTIMATE v36 | exact 44/50 + T=0.8 + binary verification (amanatar 44/50)')",
            "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
        ]:
            if old_print in new_src:
                new_src = new_src.replace(
                    old_print,
                    "print(f'CFG: ULTIMATE v40 | exact 44/50 + T=0.8 + Eagle-3 + 2-phase + disagree + retry + verify')"
                )

        if new_src != src:
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
            modified_cells += 1
            print(f"Cell {i}: modified")

    print(f"Total modified: {modified_cells} cells")
    if modified_cells == 0:
        print("ERROR: no cells modified!")
        return

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {OUT_NB}")

    # Syntax check
    import ast
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
    import shutil
    push_dir = NOTEBOOKS_DIR / "push_v21"
    push_dir.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push_dir / "submission_v21_combined.ipynb")

    with io.open(NOTEBOOKS_DIR / "push_v17" / "kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["code_file"] = "submission_v21_combined.ipynb"
    meta["title"] = "AIMO3 v40 combined best"
    meta["dataset_sources"] = [EAGLE3_DATASET]
    with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nPush dir ready: {push_dir}")
    print(f"Submit: cd {push_dir} && kaggle kernels push")


if __name__ == "__main__":
    main()
