"""Cross-verify N(n) formula against brute force for small n."""

from math import comb

def catalan(n):
    if n <= 0:
        return 1
    return comb(2*n, n) // (n + 1)

# Formula: N(n) = prod_{j=0}^{n-1} C_{2^j}^{2^{n-1-j}}
def N_formula(n):
    result = 1
    for j in range(n):
        c = catalan(2**j)
        exp = 2**(n-1-j)
        result *= c ** exp
    return result

for n in range(1, 7):
    print(f"N({n}) = {N_formula(n)}")

# Verify N(2) = 2
print(f"\nN(2) should be 2: {N_formula(2)}")

# Verify N(3) = 56
print(f"N(3) should be 56: {N_formula(3)}")

# Brute force for N(3):
def brute_force_orderings(n):
    """Brute force: enumerate all possible final orderings."""
    from itertools import combinations

    num = 2**n
    players = list(range(1, num+1))

    def run_round(groups, round_num, n):
        """groups: dict score -> list of players. Returns set of outcome tuples."""
        if round_num > n:
            # Create ordering tuple sorted by score (descending), break ties? No ties possible.
            all_items = []
            for score, plist in groups.items():
                for p in plist:
                    all_items.append((score, p))
            all_items.sort(key=lambda x: -x[0])
            return {tuple(p for _, p in all_items)}

        points = 2**(n - round_num)
        results = set()

        # For each group, generate all matchings
        sorted_groups = sorted(groups.items())
        group_matchings = []
        for score, plist in sorted_groups:
            plist = sorted(plist)
            matchings = list(all_matchings(plist))
            group_matchings.append((score, matchings))

        # Enumerate all combinations of matchings
        def enumerate_matchings(idx, new_groups):
            if idx == len(group_matchings):
                results.update(run_round(new_groups, round_num + 1, n))
                return
            score, matchings = group_matchings[idx]
            for matching in matchings:
                ng = {s: list(ps) for s, ps in new_groups.items()}
                for a, b in matching:
                    winner = max(a, b)
                    loser = min(a, b)
                    new_score = score + points
                    if new_score not in ng:
                        ng[new_score] = []
                    ng[new_score].append(winner)
                    if score not in ng:
                        ng[score] = []
                    ng[score].append(loser)
                # Remove old group
                if score in ng:
                    # Remove the original players
                    for a, b in matching:
                        ng[score].remove(max(a,b))
                        ng[score].remove(min(a,b))
                    if not ng[score]:
                        del ng[score]
                enumerate_matchings(idx + 1, ng)

        # Actually this is getting complicated. Let me do it differently.
        # Process all groups at once.

        return results

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

    return len(solve(players, n))

for n in range(1, 5):
    bf = brute_force_orderings(n)
    fm = N_formula(n)
    match = "OK" if bf == fm else "MISMATCH!"
    print(f"n={n}: brute_force={bf}, formula={fm} {match}")

# Check factorization
print(f"\nN(4) = {N_formula(4)}")
n4 = N_formula(4)
v2 = 0
t = n4
while t % 2 == 0:
    v2 += 1
    t //= 2
v5 = 0
t = n4
while t % 5 == 0:
    v5 += 1
    t //= 5
print(f"v_2(N(4)) = {v2}, v_5(N(4)) = {v5}, v_10(N(4)) = {min(v2,v5)}")

# Verify using formula
def legendre(n, p):
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def v_catalan(m, p):
    return legendre(2*m, p) - legendre(m, p) - legendre(m+1, p)

v2_f = sum(2**(3-j) * v_catalan(2**j, 2) for j in range(4))
v5_f = sum(2**(3-j) * v_catalan(2**j, 5) for j in range(4))
print(f"Formula: v_2(N(4)) = {v2_f}, v_5(N(4)) = {v5_f}")
