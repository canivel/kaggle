"""
Verify the tournament problem by brute force for small n.
"""
from itertools import combinations, permutations
from math import comb

def simulate_tournament(n, players):
    """
    Simulate tournament with 2^n players.
    players is a list of player speeds (lower = faster).
    Returns the final ranking (list of players sorted by score, ties broken...
    actually all scores should be distinct).
    """
    num_players = 2**n
    assert len(players) == num_players

    scores = {p: 0 for p in players}

    # We need to try all possible pairings and collect all possible orderings
    # This is exponential, so only for small n

    def get_all_orderings(round_num, scores):
        if round_num > n:
            # Sort by score (descending)
            ranking = sorted(players, key=lambda p: -scores[p])
            return {tuple(ranking)}

        points = 2**(n - round_num)

        # Group players by score
        from collections import defaultdict
        groups = defaultdict(list)
        for p in players:
            groups[scores[p]].append(p)

        # For each group, enumerate all perfect matchings
        # Then combine across groups
        group_matchings_list = []
        for score_val in sorted(groups.keys()):
            group = groups[score_val]
            assert len(group) % 2 == 0
            matchings = list(all_perfect_matchings(group))
            group_matchings_list.append(matchings)

        # Take cartesian product of matchings across groups
        from itertools import product
        all_orderings = set()
        for combo in product(*group_matchings_list):
            # Apply all matchings
            new_scores = dict(scores)
            for matching in combo:
                for (a, b) in matching:
                    # faster (lower number) wins
                    winner = min(a, b)
                    new_scores[winner] += points

            sub_orderings = get_all_orderings(round_num + 1, new_scores)
            all_orderings.update(sub_orderings)

        return all_orderings

    return get_all_orderings(1, scores)

def all_perfect_matchings(lst):
    """Generate all perfect matchings of a list."""
    if len(lst) == 0:
        yield []
        return
    if len(lst) == 2:
        yield [(lst[0], lst[1])]
        return

    first = lst[0]
    rest = lst[1:]
    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i+1:]
        for matching in all_perfect_matchings(remaining):
            yield [(first, partner)] + matching

# Test for small n
for n in range(1, 5):
    num_players = 2**n
    players = list(range(1, num_players + 1))  # 1 = fastest
    orderings = simulate_tournament(n, players)
    N = len(orderings)

    # Compute expected N using Catalan formula
    N_expected = 1
    for j in range(n):
        m = 2**j
        cat = comb(2*m, m) // (m+1)
        exp = 2**(n-1-j)
        N_expected *= cat**exp

    print(f"n={n}: N={N}, expected={N_expected}, match={N==N_expected}")
    if n <= 3:
        for o in sorted(orderings):
            print(f"  {o}")
