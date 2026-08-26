"""Verify tournament problem for n=1,2,3 only (brute force)."""
from itertools import combinations
from math import comb
from collections import defaultdict

def all_perfect_matchings(lst):
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

def simulate(n, players, round_num, scores):
    if round_num > n:
        ranking = tuple(sorted(players, key=lambda p: -scores[p]))
        return {ranking}

    points = 2**(n - round_num)
    groups = defaultdict(list)
    for p in players:
        groups[scores[p]].append(p)

    # Get matchings for each group
    group_keys = sorted(groups.keys())
    group_matchings = []
    for key in group_keys:
        group = groups[key]
        matchings = list(all_perfect_matchings(group))
        group_matchings.append(matchings)

    from itertools import product
    all_orderings = set()
    for combo in product(*group_matchings):
        new_scores = dict(scores)
        for matching in combo:
            for (a, b) in matching:
                winner = min(a, b)  # lower number = faster
                new_scores[winner] += points
        sub = simulate(n, players, round_num + 1, new_scores)
        all_orderings.update(sub)

    return all_orderings

for n in range(1, 4):
    num_players = 2**n
    players = list(range(1, num_players + 1))
    scores = {p: 0 for p in players}
    orderings = simulate(n, players, 1, scores)
    N = len(orderings)

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
