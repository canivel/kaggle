"""Adaptive time budget manager for AIMO3 inference.

Manages per-problem time allocation to ensure we solve as many
problems as possible within the total time limit.
"""

from __future__ import annotations

import time


class TimeManager:
    """Manages time budgets across problems.

    Tracks elapsed time and adaptively adjusts per-problem budgets
    based on remaining problems and total time limit.
    """

    def __init__(
        self,
        total_time_limit: float = 32400.0,  # 9 hours
        per_problem_limit: float = 1700.0,  # ~28 min (margin from 30)
        n_problems: int = 110,
    ):
        self.total_time_limit = total_time_limit
        self.per_problem_limit = per_problem_limit
        self.n_problems = n_problems
        self.start_time = time.time()
        self.problems_solved = 0
        self.problem_times: list[float] = []

    def elapsed(self) -> float:
        """Total elapsed time since start."""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """Total remaining time."""
        return max(0, self.total_time_limit - self.elapsed())

    def problems_remaining(self) -> int:
        """Number of unsolved problems."""
        return max(0, self.n_problems - self.problems_solved)

    def budget_for_next_problem(self) -> float:
        """Get the time budget for the next problem.

        Uses adaptive allocation: distributes remaining time evenly
        across remaining problems, capped by per_problem_limit.
        """
        remaining_problems = self.problems_remaining()
        if remaining_problems <= 0:
            return 0.0

        # Distribute remaining time evenly
        adaptive_budget = self.remaining() / remaining_problems

        # Cap at per-problem limit
        budget = min(adaptive_budget, self.per_problem_limit)

        # Minimum budget: at least 60 seconds to try something
        return max(60.0, budget)

    def get_n_samples(self, base_n: int = 32) -> int:
        """Get recommended N samples based on available time.

        Scales down N when time is tight.
        """
        budget = self.budget_for_next_problem()

        if budget >= 1500:  # 25+ min: full budget
            return base_n
        elif budget >= 900:  # 15+ min: reduced
            return max(16, base_n // 2)
        elif budget >= 300:  # 5+ min: minimal
            return max(8, base_n // 4)
        else:  # very tight: bare minimum
            return 4

    def record_problem(self, elapsed: float):
        """Record time spent on a problem."""
        self.problems_solved += 1
        self.problem_times.append(elapsed)

    def avg_time_per_problem(self) -> float:
        """Average time per solved problem."""
        if not self.problem_times:
            return 0.0
        return sum(self.problem_times) / len(self.problem_times)

    def should_skip(self) -> bool:
        """Whether we should skip remaining problems (time critical)."""
        return self.remaining() < 30.0  # less than 30 seconds left

    def status(self) -> str:
        """Human-readable status string."""
        return (
            f"Solved {self.problems_solved}/{self.n_problems} | "
            f"Elapsed {self.elapsed():.0f}s | "
            f"Remaining {self.remaining():.0f}s | "
            f"Avg {self.avg_time_per_problem():.1f}s/problem | "
            f"Budget {self.budget_for_next_problem():.0f}s next"
        )
