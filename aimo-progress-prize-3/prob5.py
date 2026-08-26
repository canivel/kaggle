"""
Problem 5: Tournament with 2^20 runners, each with different speed.
20 rounds. In each round, runners paired by same score (Swiss system).
Winner of round i gets 2^(20-i) points.
N = number of possible final orderings.
k = largest integer where 10^k divides N.
Find k mod 10^5.

Let me think about this carefully.

There are 2^20 = 1,048,576 runners with distinct speeds.
Round i (i=1,...,20): runners with the same score are paired.
Winner gets 2^(20-i) points.

After round 1: winner gets 2^19, loser gets 0. Scores: {2^19, 0}.
After round 2: among 2^19 winners (score 2^19), pair them up.
  Winner gets 2^18 (total 2^19+2^18), loser stays at 2^19.
  Among 2^19 losers (score 0), pair them up.
  Winner gets 2^18, loser stays at 0.
  Scores: {2^19+2^18, 2^19, 2^18, 0}.

After round k: scores are sums of subsets of {2^19, 2^18, ..., 2^(20-k)}.
Each score corresponds to a binary string of length k.

After all 20 rounds: scores are sums of subsets of {2^19, 2^18, ..., 2^0}.
So final scores are 0, 1, 2, ..., 2^20 - 1. All distinct!
Each runner ends with a unique score.
The final ordering is the ordering by score.

Wait, but in each round, within each score group, the pairing is chosen
(presumably we can choose the pairing?), and then the faster runner wins.

The "final ordering" refers to the ordering of runners by their final scores.
The question is: how many distinct permutations of the runners (ordered by final score)
are achievable?

Since final scores are all distinct (0 to 2^20-1), the ordering is determined
by which runner gets which score. So we're counting the number of possible
score-to-runner assignments.

Each runner's score is their win/loss record encoded in binary:
score = sum_{i where they won round i} 2^(20-i).

In round i, runners with the same (i-1)-round record are paired, and the faster
runner in each pair wins.

The key insight: in each round, within each group of runners with the same score,
we can choose ANY perfect matching. Then the faster runner in each pair wins.

So the number of possible orderings is the number of possible outcomes, which
depends on how many different matchings lead to different final orderings.

Let me think recursively.

Actually, this is equivalent to a Swiss-system tournament where:
- 2^n players
- n rounds
- In each round, players with equal scores are paired
- We choose the pairing
- Stronger player always wins

The final result is a permutation of players ranked 1st through 2^n th.

Let me think about this for small n first.

n=1: 2 runners. 1 round. They are paired. Faster wins.
Score: faster gets 1, slower gets 0. Only 1 possible ordering.
N = 1.

n=2: 4 runners, speeds 1<2<3<4. 2 rounds.
Round 1 (2^1 = 2 points for winner):
We choose a perfect matching of {1,2,3,4}.
3 possible matchings: {12,34}, {13,24}, {14,23}.

Matching {1-2, 3-4}: 2 beats 1, 4 beats 3.
Winners (score 2): {2, 4}. Losers (score 0): {1, 3}.

Round 2 (1 point for winner):
Pair winners: 2 vs 4. 4 beats 2. Scores: 4->3, 2->2.
Pair losers: 1 vs 3. 3 beats 1. Scores: 3->1, 1->0.
Final ranking: 4(3), 2(2), 3(1), 1(0).

Matching {1-3, 2-4}: 3 beats 1, 4 beats 2.
Winners: {3, 4}. Losers: {1, 2}.
Round 2: 3 vs 4 -> 4 wins. 1 vs 2 -> 2 wins.
Final: 4(3), 3(2), 2(1), 1(0).

Matching {1-4, 2-3}: 4 beats 1, 3 beats 2.
Winners: {4, 3}. Losers: {1, 2}.
Round 2: 4 vs 3 -> 4 wins. 1 vs 2 -> 2 wins.
Final: 4(3), 3(2), 2(1), 1(0).

So for n=2: two distinct orderings: [4,2,3,1] and [4,3,2,1].
N = 2.

Let me verify with n=3 (8 runners).
"""

from itertools import combinations
from math import factorial
import sys

def solve_tournament(n):
    """Count distinct final orderings for 2^n runners in Swiss tournament."""
    num_runners = 2**n
    # Runners labeled 0 to 2^n - 1 by speed (0 = slowest, 2^n-1 = fastest)

    def get_matchings(group):
        """Generate all perfect matchings of a group."""
        if len(group) == 0:
            yield []
            return
        if len(group) == 2:
            yield [(group[0], group[1])]
            return
        first = group[0]
        rest = group[1:]
        for i, partner in enumerate(rest):
            remaining = rest[:i] + rest[i+1:]
            for matching in get_matchings(remaining):
                yield [(first, partner)] + matching

    def run_tournament(runners):
        """Run tournament, return set of possible final score tuples."""
        scores = {r: 0 for r in runners}
        results = set()

        def recurse(round_num, scores):
            if round_num > n:
                # Final ordering: tuple of runners sorted by score (descending)
                ordering = tuple(r for r, s in sorted(scores.items(), key=lambda x: -x[1]))
                results.add(ordering)
                return

            points = 2**(n - round_num)
            # Group by score
            groups = {}
            for r, s in scores.items():
                groups.setdefault(s, []).append(r)

            # For each group, generate all matchings
            group_matchings = {}
            for s, group in sorted(groups.items()):
                group.sort()
                gm = list(get_matchings(group))
                group_matchings[s] = gm

            # Take Cartesian product of matchings across groups
            def combine(group_keys, idx, new_scores):
                if idx == len(group_keys):
                    recurse(round_num + 1, new_scores)
                    return
                s = group_keys[idx]
                for matching in group_matchings[s]:
                    ns = dict(new_scores)
                    for a, b in matching:
                        winner = max(a, b)  # faster runner wins
                        ns[winner] = ns[winner] + points
                    combine(group_keys, idx + 1, ns)

            combine(sorted(groups.keys()), 0, dict(scores))

        recurse(1, scores)
        return results

    runners = list(range(num_runners))
    results = run_tournament(runners)
    return len(results)

for n in range(1, 5):
    N = solve_tournament(n)
    print(f"n={n}: 2^{n}={2**n} runners, N={N} distinct orderings")

# This is too slow for n=20. Need a smarter approach.
