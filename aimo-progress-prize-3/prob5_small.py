"""Cross-verify for small n only."""
from math import comb

def catalan(n):
    if n <= 0:
        return 1
    return comb(2*n, n) // (n + 1)

def N_formula(n):
    result = 1
    for j in range(n):
        c = catalan(2**j)
        exp = 2**(n-1-j)
        result *= c ** exp
    return result

def all_matchings(lst):
    if len(lst) == 0:
        yield []
        return
    if len(lst) == 2:
        yield [(lst[0], lst[1])]
        return
    first = lst[0]
    for i in range(1, len(lst)):
        pair = (first, lst[i])
        remaining = lst[1:i] + lst[i+1:]
        for rest in all_matchings(remaining):
            yield [pair] + rest

def solve(players, n):
    if n == 0:
        return {(players[0],)}
    results = set()
    for matching in all_matchings(sorted(players)):
        winners = []
        losers = []
        for a, b in matching:
            winners.append(max(a, b))
            losers.append(min(a, b))
        w_orderings = solve(winners, n-1)
        l_orderings = solve(losers, n-1)
        for wo in w_orderings:
            for lo in l_orderings:
                results.add(wo + lo)
    return results

for n in range(1, 4):
    players = list(range(1, 2**n + 1))
    bf = len(solve(players, n))
    fm = N_formula(n)
    match = "OK" if bf == fm else "MISMATCH!"
    print(f"n={n}: brute_force={bf}, formula={fm} {match}")
