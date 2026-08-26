"""
Let me verify with small cases by brute force, then re-examine.
Also, let me reconsider: maybe the problem does NOT require rectangles to fit
in the square (i.e., maybe it is just about the area sum and integer sides,
not about a physical tiling). Competition problems sometimes mean "partition
the area" in a way that allows any rectangles.

Actually no, "divided into k rectangles" definitely means a tiling/partition.
Each rectangle must fit inside the 500x500 square.

Let me brute force small cases to verify my formula.
"""

from itertools import combinations

def brute_force(n, max_k=20):
    """Brute force: find max k for n x n square."""
    area = n * n

    # All possible rectangles: (a, b) with 1 <= a <= b <= n
    # Perimeter = 2(a+b)
    # Group by perimeter (equivalently by a+b = s)
    # For each s, list possible (a, b) pairs
    rects_by_s = {}
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            s = a + b
            if s not in rects_by_s:
                rects_by_s[s] = []
            rects_by_s[s].append((a, b, a * b))

    all_s_values = sorted(rects_by_s.keys())

    # For each s, we can pick any one rectangle from that group.
    # The possible areas for perimeter-class s:
    areas_by_s = {}
    for s in all_s_values:
        areas_by_s[s] = sorted(set(a * b for a, b, ab in rects_by_s[s]))

    # Now: find max k distinct s values where we can pick areas summing to n^2.
    # Try from largest k downward.
    best_k = 0

    for k in range(min(len(all_s_values), max_k), 0, -1):
        if best_k >= k:
            break
        # Try all combinations of k s-values
        found = False
        for combo in combinations(all_s_values, k):
            # Check if we can pick areas summing to area
            # This is a subset-sum variant. Use DP or backtracking.
            # For small cases, use recursive search.
            target = area
            areas_lists = [areas_by_s[s] for s in combo]

            # Quick bounds check
            min_sum = sum(min(al) for al in areas_lists)
            max_sum = sum(max(al) for al in areas_lists)
            if min_sum > target or max_sum < target:
                continue

            # DP to check if target is achievable
            # States: remaining target after choosing first i rectangles
            possible = {target}
            ok = True
            for al in areas_lists:
                new_possible = set()
                for p in possible:
                    for a in al:
                        if p - a >= 0:
                            new_possible.add(p - a)
                possible = new_possible
                if not possible:
                    ok = False
                    break

            if ok and 0 in possible:
                best_k = k
                print(f"  n={n}: k={k} achievable with s={combo}")
                found = True
                break

        if not found and k <= best_k:
            break

    return best_k

# Test small cases
for n in range(2, 12):
    k = brute_force(n)
    print(f"n={n}: max k = {k}")

print()

# Now compare with formula
def formula_k(n):
    area = n * n
    def min_area(s):
        if s <= n + 1:
            return s - 1
        else:
            return (s - n) * n

    all_s = list(range(2, 2*n + 1))
    all_s.sort(key=min_area)
    total = 0
    count = 0
    for s in all_s:
        c = min_area(s)
        if total + c <= area:
            total += c
            count += 1
    return count

print("Comparison:")
for n in range(2, 12):
    bf = brute_force(n, max_k=2*n)
    fm = formula_k(n)
    match = "OK" if bf == fm else "MISMATCH"
    print(f"n={n}: brute_force={bf}, formula={fm} {match}")
