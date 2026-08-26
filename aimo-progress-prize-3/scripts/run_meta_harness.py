"""Run Meta-Harness search to discover optimal math retrieval harness.

Uses kaos + our math corpus to evolve retrieval strategies that maximize
accuracy on math competition problems.

Usage:
    # Quick test (synthetic data):
    uv run python scripts/run_meta_harness.py

    # With real corpus:
    uv run python scripts/run_meta_harness.py \
        --problems data/math_corpus.jsonl \
        --corpus data/math_corpus.jsonl \
        --iterations 10

    # Inspect results:
    uv run kaos mh frontier <search-agent-id> --db aimo3-search.db
"""

from __future__ import annotations

import sys
import argparse
import asyncio

sys.path.insert(0, "f:/kaggle/kaos")

from kaos import Kaos
from kaos.metaharness.harness import SearchConfig
from kaos.metaharness.search import MetaHarnessSearch
from kaos.metaharness.benchmarks import get_benchmark
import kaos.metaharness.benchmarks.math_rag  # noqa: register
from kaos.router.gepa import GEPARouter


async def main(
    problems_path: str | None,
    corpus_path: str | None,
    iterations: int,
    candidates: int,
):
    db = Kaos("aimo3-search.db")

    kwargs = {}
    if problems_path:
        kwargs["problems_path"] = problems_path
    if corpus_path:
        kwargs["corpus_path"] = corpus_path

    bench = get_benchmark("math_rag", **kwargs)

    config = SearchConfig(
        benchmark="math_rag",
        max_iterations=iterations,
        candidates_per_iteration=candidates,
        objectives=["+accuracy", "-context_cost"],
        max_parallel_evals=2,
    )

    search_set = bench.get_search_set()
    seeds = bench.get_seed_harnesses()

    print("=" * 60)
    print("Meta-Harness: Math Retrieval Optimization for AIMO3")
    print("=" * 60)
    print(f"  Search set: {len(search_set)} problems")
    print(f"  Seed harnesses: {len(seeds)}")
    print(f"  Iterations: {iterations}")
    print(f"  Candidates per iteration: {candidates}")
    print(f"  Objectives: accuracy (max), context_cost (min)")
    print()

    router = GEPARouter.from_config("kaos.yaml")
    search = MetaHarnessSearch(db, router, bench, config)
    result = await search.run()

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    print("\nPareto Frontier:")
    for point in result.frontier.points:
        scores = ", ".join(f"{k}={v:.4f}" for k, v in point.scores.items())
        print(f"  {point.harness_id[:12]}... (iter {point.iteration}): {scores}")

    # Show all agents
    agents = db.query("""
        SELECT a.name, a.status, COUNT(tc.call_id) as calls,
               COALESCE(SUM(tc.token_count), 0) as tokens
        FROM agents a
        LEFT JOIN tool_calls tc ON a.agent_id = tc.agent_id
        GROUP BY a.agent_id ORDER BY tokens DESC
    """)
    print(f"\nAll agents ({len(agents)} total):")
    for a in agents[:15]:
        print(f"  {a['name']:35s} {a['status']:12s} {a['tokens']:>8,} tokens")

    # Export the best harness
    if result.frontier.points:
        best = result.frontier.best_by_objective.get("+accuracy")
        if best:
            hid = best.harness_id
            source = db.read(result.search_agent_id, f"/harnesses/{hid}/source.py")
            out_path = "data/best_harness.py"
            with open(out_path, "wb") as f:
                f.write(source)
            print(f"\nBest harness exported to {out_path}")
            print(f"  Accuracy: {best.scores.get('accuracy', 0):.4f}")

    db.close()
    print(f"\nAll data saved in aimo3-search.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default=None, help="JSONL of eval problems")
    parser.add_argument("--corpus", default=None, help="JSONL of retrieval corpus")
    parser.add_argument("--iterations", type=int, default=5, help="Search iterations")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates per iter")
    args = parser.parse_args()
    asyncio.run(main(args.problems, args.corpus, args.iterations, args.candidates))
