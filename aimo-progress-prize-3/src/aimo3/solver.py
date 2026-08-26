"""Main solver orchestrator for AIMO3.

Ties together all components:
  Problem → Classify → Prompt → TIR (N samples) → Vote → Answer

This is the top-level module that the submission notebook calls.
"""

from __future__ import annotations

import time
import traceback

from .inference import InferenceEngine, InferenceConfig
from .tir_executor import run_tir_batch, TIRResult
from .voting import majority_vote, vote_with_quality, confidence_score
from .answer_extraction import extract_answer, ANSWER_MOD
from .prompt_templates import format_problem_prompt, build_chat_messages
from .problem_classifier import classify_problem
from .time_manager import TimeManager


class AIMOSolver:
    """End-to-end solver for AIMO math problems.

    Usage:
        solver = AIMOSolver(model_path="/kaggle/input/model/")
        solver.setup()

        for problem_text in problems:
            answer = solver.solve(problem_text)
    """

    def __init__(
        self,
        model_path: str = "nvidia/OpenMath-Nemotron-14B-Kaggle",
        n_samples: int = 32,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        tir_max_retries: int = 3,
        code_timeout: int = 30,
        use_chat_template: bool = True,
        total_time_limit: float = 32400.0,
        per_problem_limit: float = 1700.0,
        n_problems: int = 110,
    ):
        self.config = InferenceConfig(
            model_path=model_path,
            n_samples=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.n_samples = n_samples
        self.tir_max_retries = tir_max_retries
        self.code_timeout = code_timeout
        self.use_chat_template = use_chat_template
        self.engine: InferenceEngine | None = None
        self.time_manager = TimeManager(
            total_time_limit=total_time_limit,
            per_problem_limit=per_problem_limit,
            n_problems=n_problems,
        )
        self.solve_log: list[dict] = []

    def setup(self):
        """Initialize the inference engine. Call once before solving."""
        print(f"Loading model: {self.config.model_path}")
        t0 = time.time()
        self.engine = InferenceEngine(self.config)
        # Trigger lazy load
        _ = self.engine.llm
        print(f"Model loaded in {time.time() - t0:.1f}s")

    def solve(self, problem: str) -> int:
        """Solve a single math problem.

        Args:
            problem: LaTeX math problem text.

        Returns:
            Integer answer (mod 100000). Returns 0 on failure.
        """
        if self.engine is None:
            raise RuntimeError("Call setup() before solve()")

        if self.time_manager.should_skip():
            print("WARNING: Time critical, returning default answer")
            return 0

        t0 = time.time()

        try:
            answer = self._solve_inner(problem)
        except Exception as e:
            print(f"ERROR solving problem: {e}")
            traceback.print_exc()
            answer = 0

        elapsed = time.time() - t0
        self.time_manager.record_problem(elapsed)
        print(f"  -> Answer: {answer} ({elapsed:.1f}s) | {self.time_manager.status()}")

        return answer

    def _solve_inner(self, problem: str) -> int:
        """Core solve logic with TIR and voting."""
        # Step 1: Classify problem type
        problem_type = classify_problem(problem)
        print(f"  Problem type: {problem_type}")

        # Step 2: Build prompt
        if self.use_chat_template:
            messages = build_chat_messages(problem, problem_type)
            prompt = self.engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = format_problem_prompt(problem, problem_type)

        # Step 3: Adaptive N based on time budget
        n_samples = self.time_manager.get_n_samples(self.n_samples)

        # Step 4: Run TIR batch
        tir_results = run_tir_batch(
            engine=self.engine,
            prompt=prompt,
            n_samples=n_samples,
            max_retries=self.tir_max_retries,
            code_timeout=self.code_timeout,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Step 5: Vote on answers
        answers = [r.answer for r in tir_results]
        code_executed = [r.code_executed for r in tir_results]
        code_succeeded = [r.code_succeeded for r in tir_results]
        has_boxed = [r.has_boxed for r in tir_results]

        answer, conf = vote_with_quality(
            answers, code_executed, code_succeeded, has_boxed
        )

        # Log
        valid_answers = [a for a in answers if a is not None]
        n_with_code = sum(code_executed)
        n_code_ok = sum(code_succeeded)
        print(
            f"  Samples: {n_samples} | Valid answers: {len(valid_answers)} | "
            f"Code executed: {n_with_code} | Code OK: {n_code_ok} | "
            f"Confidence: {conf:.2f}"
        )

        self.solve_log.append({
            "problem_type": problem_type,
            "n_samples": n_samples,
            "n_valid": len(valid_answers),
            "n_code_executed": n_with_code,
            "n_code_succeeded": n_code_ok,
            "confidence": conf,
            "answer": answer,
        })

        return answer % ANSWER_MOD


def create_solver(
    model_path: str = "/kaggle/input/openmath-nemotron-14b-kaggle/",
    n_samples: int = 32,
    **kwargs,
) -> AIMOSolver:
    """Create and initialize an AIMOSolver.

    Convenience function for the submission notebook.
    """
    solver = AIMOSolver(model_path=model_path, n_samples=n_samples, **kwargs)
    solver.setup()
    return solver
