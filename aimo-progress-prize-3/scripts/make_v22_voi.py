"""Build v22: Best combined notebook.

Based on v19 (two-phase no-domain + verify cascade), adds:
  - VOI (Value of Information) stopping rule replaces consensus=3 in Phase 1
  - 3-factor Bayesian posterior replaces pure entropy voting
  - (Eagle-3 SKIPPED: vLLM 0.11 MXFP4 crash bug)

VOI source: jonathanchan/aimo3-gpt-oss-120b-with-bayesian (185 votes, Kaggle)
Formula: stop when posterior_entropy <= 0.0667 nats
  = entropy * 0.6 <= 0.04
  = submit_utility >= continue_utility

3-factor Bayesian posterior:
  weight_i = (1/(1+H_i)) * (1/(1+errors_i)) * tool_bonus_i

Changes from v19:
- _compute_bayesian_posterior() replaces simple Counter(valid)
- _should_stop_voi() drives Phase 1 early exit instead of count >= 3
- _run_batch_voi() replaces _run_batch()
- _select_answer() still uses verify cascade (unchanged from v17)
"""

from __future__ import annotations
import ast
import io
import json
import shutil
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v19_twophase_nodomain.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v22_voi.ipynb"


# ── Helpers injected before FOLLOWUP_PROMPT ──────────────────────────────────
VOI_HELPERS = '''
DISAGREE_CONTEXT = (
    'NOTE: Initial analysis produced conflicting results with answers: {answers}. '
    'At least some are wrong. Be extra thorough -- verify every step with Python code '
    'and check your answer with a second independent method.'
)

PYTHON_MANDATORY = (
    'You MUST execute Python code to verify your answer before giving it. '
    'Do not provide a final boxed answer without code verification.'
)

# ── Bayesian posterior + VOI (Jonathan Chan / aimo3-gpt-oss-120b-with-bayesian) ──

def _compute_bayesian_posterior(detailed_results):
    """3-factor Bayesian posterior over candidate answers."""
    import math
    from collections import defaultdict
    posterior = defaultdict(float)
    for r in detailed_results:
        answer = r.get('Answer')
        if answer is None:
            continue
        entropy_weight = 1.0 / (1.0 + r.get('Entropy', 1.0))
        errors = r.get('Python Errors', 0)
        reliability = 1.0 / (1.0 + errors)
        pc = r.get('Python Calls', 0)
        tool_bonus = 1.2 if (pc > 0 and errors == 0) else (0.8 if errors > 0 else 1.0)
        posterior[answer] += entropy_weight * reliability * tool_bonus
    total = sum(posterior.values())
    if not total:
        return {}
    return {k: v / total for k, v in posterior.items()}


def _should_stop_voi(detailed_results, min_attempts=3):
    """VOI stopping rule: stop when posterior_entropy <= 0.0667 nats."""
    import math
    if len(detailed_results) < min_attempts:
        return False
    posterior = _compute_bayesian_posterior(detailed_results)
    if not posterior:
        return False
    entropy = -sum(p * math.log(p + 1e-9) for p in posterior.values())
    # stop when entropy * 0.6 <= 0.04  (i.e., <= 0.0667 nats)
    return entropy * 0.6 <= 0.04


def _run_batch_voi(solver, user_input, seed_offset, n, deadline, use_voi=True, consensus=4):
    """Run up to n attempts. Stops early on VOI or consensus."""
    detailed, valid = [], []
    stop = threading.Event()
    ex = ThreadPoolExecutor(max_workers=solver.cfg.workers)
    try:
        futs = [ex.submit(solver._process_attempt,
                          user_input, solver.cfg.system_prompt, i + seed_offset,
                          stop, deadline)
                for i in range(n)]
        for f in as_completed(futs):
            if stop.is_set():
                break
            try:
                r = f.result()
                detailed.append(r)
                if r['Answer'] is not None:
                    valid.append(r['Answer'])
                # Check stopping
                if use_voi and _should_stop_voi(detailed):
                    stop.set()
                    break
                c = Counter(valid).most_common(1)
                if c and c[0][1] >= consensus:
                    stop.set()
                    break
            except Exception as e:
                print(f'Attempt error: {e}')
    finally:
        stop.set()
        ex.shutdown(wait=True, cancel_futures=True)
    return detailed, valid

'''


# ── New solve_problem using VOI ───────────────────────────────────────────────
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

        # ── Phase 1: Quick triage (4 attempts, 25% budget max 3min, VOI stopping) ──
        phase1_budget = min(180, total_budget * 0.25)
        phase1_deadline = time.time() + phase1_budget
        detailed_p1, valid_p1 = _run_batch_voi(self, user_input, 0, 4, phase1_deadline,
                                               use_voi=True, consensus=3)

        # Check if VOI already gave us strong consensus
        if valid_p1:
            posterior = _compute_bayesian_posterior(detailed_p1)
            if posterior:
                top_ans = max(posterior, key=posterior.get)
                top_prob = posterior[top_ans]
                if top_prob >= 0.95 or _should_stop_voi(detailed_p1):
                    print(f'Phase 1 VOI: CONFIDENT ({top_prob:.2f} -> {top_ans})')
                    if detailed_p1:
                        df = pd.DataFrame(detailed_p1)
                        df['Entropy'] = df['Entropy'].round(3)
                        df['Answer'] = df['Answer'].astype('Int64')
                        display(df)
                    print(f'\\nFinal Answer: {top_ans}\\n')
                    self.problems_remaining = max(0, self.problems_remaining - 1)
                    return top_ans
            print(f'Phase 1: SPLIT -- {dict(Counter(valid_p1))}')
        else:
            print('Phase 1: No valid answers')

        # ── Phase 2: Deep solving with disagreement context when split ──
        if valid_p1 and len(set(valid_p1)) > 1:
            ans_list = ', '.join(str(a) for a in sorted(set(valid_p1)))
            context = DISAGREE_CONTEXT.format(answers=ans_list)
            user_p2 = f'{user_input} {context}'
        else:
            user_p2 = user_input

        detailed_p2, valid_p2 = _run_batch_voi(self, user_p2, 4, 4, overall_deadline,
                                               use_voi=True, consensus=4)

        all_detailed = detailed_p1 + detailed_p2
        all_valid = valid_p1 + valid_p2

        # ── Failure-aware retry (up to 2 for no-code/no-answer) ──
        if time.time() < overall_deadline - 60:
            failed = [r for r in all_detailed if r['Answer'] is None or r.get('Python Calls', 0) == 0]
            n_retry = min(len(failed), 2)
            if n_retry > 0:
                print(f'Retrying {n_retry} low-quality attempts')
                retry_input = f'{user_input} {PYTHON_MANDATORY}'
                retry_d, retry_v = _run_batch_voi(self, retry_input, 8, n_retry, overall_deadline,
                                                  use_voi=False, consensus=4)
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
    print("Building v22 (VOI + 3-factor Bayesian + two-phase + verify) from v19...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified_cells = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        # Replace the helpers + solve_problem in the main solver cell
        if "DISAGREE_CONTEXT" in src and "def solve_problem" in src:
            # Find where old DISAGREE_CONTEXT starts (begin of old helpers)
            idx_helpers = new_src.find("\nDISAGREE_CONTEXT")
            if idx_helpers < 0:
                idx_helpers = new_src.find("DISAGREE_CONTEXT")

            # Find solve_problem start and end
            idx_start = new_src.find("    def solve_problem(self, problem):")
            idx_end = new_src.find("\n    def __del__", idx_start)

            if idx_helpers >= 0 and idx_start >= 0 and idx_end >= 0:
                # Keep everything before helpers, inject new helpers + solve_problem
                new_src = (
                    new_src[:idx_helpers]
                    + "\n" + VOI_HELPERS
                    + new_src[idx_helpers:idx_start]  # FOLLOWUP_PROMPT etc between helpers and solve_problem
                    + NEW_SOLVE_PROBLEM
                    + "\n"
                    + new_src[idx_end:]
                )
                print(f"Cell {i}: replaced helpers (VOI + 3-factor) + solve_problem")

        # Update version tag
        for old in [
            "# ULTIMATE v38: exact 44/50 params + T=0.8 + two-phase (no domain) + disagree ctx + retry + verify",
            "# ULTIMATE v36: exact 44/50 params + T=0.8 + follow-up + binary verification cascade",
            "# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up",
        ]:
            if old in new_src:
                new_src = new_src.replace(old,
                    "# ULTIMATE v41: exact 44/50 + T=0.8 + VOI stopping + 3-factor Bayesian + 2-phase + verify")

        for old in [
            "print(f'CFG: ULTIMATE v38 | exact 44/50 + T=0.8 + 2-phase (no domain) + disagree + retry + verify')",
            "print(f'CFG: ULTIMATE v36 | exact 44/50 + T=0.8 + binary verification (amanatar 44/50)')",
            "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
        ]:
            if old in new_src:
                new_src = new_src.replace(old,
                    "print(f'CFG: ULTIMATE v41 | exact 44/50 + T=0.8 + VOI + Bayesian + 2-phase + verify')")

        if new_src != src:
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
            modified_cells += 1

    print(f"Modified {modified_cells} cells")
    if modified_cells == 0:
        print("ERROR: no cells modified!")
        return

    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
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
    push_dir = NOTEBOOKS_DIR / "push_v22"
    push_dir.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push_dir / "submission_v22_voi.ipynb")

    with io.open(NOTEBOOKS_DIR / "push_v17" / "kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["code_file"] = "submission_v22_voi.ipynb"
    meta["title"] = "AIMO3 v41 VOI Bayesian two-phase verify"
    with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nPush dir: {push_dir}")
    print("Submit: cd notebooks/push_v22 && kaggle kernels push")


if __name__ == "__main__":
    main()
