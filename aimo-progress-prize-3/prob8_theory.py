
# Verify the brute force results match the cyclotomic theory.
# From brute force:
# D=2: 14, D=3: 24, D=4: 40

# Theory: A(x) = x^s * A1(x), deg(A) <= D, A1 | x^N+1 for some N >= 1
# Valid A1 are: +/- products of subsets of Phi_d with all d having same v_2 >= 1

# Identify the brute force A1 polynomials:
# D=4 found:
# (1,) = 1
# (-1,) = -1
# (1, 1) = 1 + x = Phi_2 (since Phi_2 = x + 1... wait)
# Actually Phi_2(x) = x + 1, coefficients (low to high) = (1, 1). Yes.
# (-1, -1) = -(x + 1) = -Phi_2
# (1, -1, 1) = 1 - x + x^2 = Phi_6
# (-1, 1, -1) = -(1 - x + x^2) = -Phi_6
# (1, 0, 1) = 1 + x^2 = Phi_4
# (-1, 0, -1) = -(1 + x^2) = -Phi_4
# (1, 0, 0, 1) = 1 + x^3 = Phi_2 * Phi_6 = (x+1)(x^2-x+1) = x^3+1 ✓
# (-1, 0, 0, -1) = -(1 + x^3)
# (1, 0, 0, 0, 1) = 1 + x^4 = Phi_8 (since Phi_8 = x^4+1)? Let me check.
# Phi_8(x) = x^4 + 1. Yes! Coefficients: (1,0,0,0,1). ✓
# (-1, 0, 0, 0, -1) = -Phi_8
# (1, -1, 1, -1, 1) = 1 - x + x^2 - x^3 + x^4 = Phi_10
# Phi_10(x) = x^4 - x^3 + x^2 - x + 1. Yes!
# (-1, 1, -1, 1, -1) = -Phi_10
# (1, 0, -1, 0, 1) = 1 - x^2 + x^4 = Phi_12
# Phi_12(x) = x^4 - x^2 + 1. Yes!
# (-1, 0, 1, 0, -1) = -Phi_12

print("All brute force A1 match cyclotomic products!")
print()

# Now verify counts:
# D=2:
# A1 options and their contributions:
# deg 0: {1, -1} -> 2 * (2+1) = 6  (s in {0,1,2})
# deg 1: {Phi_2, -Phi_2} -> 2 * (1+1) = 4  (s in {0,1})
# deg 2: {Phi_6, -Phi_6, Phi_4, -Phi_4} -> 4 * 1 = 4  (s in {0})
# Total = 6 + 4 + 4 = 14 ✓

# D=3:
# deg 0: 2 * 4 = 8
# deg 1: 2 * 3 = 6
# deg 2: 4 * 2 = 8
# deg 3: {Phi_2*Phi_6, -Phi_2*Phi_6} -> 2 * 1 = 2
# Total = 8 + 6 + 8 + 2 = 24 ✓

# D=4:
# deg 0: 2 * 5 = 10
# deg 1: 2 * 4 = 8
# deg 2: 4 * 3 = 12
# deg 3: 2 * 2 = 4
# deg 4: {Phi_8, -Phi_8, Phi_10, -Phi_10, Phi_12, -Phi_12} -> 6 * 1 = 6
# Total = 10 + 8 + 12 + 4 + 6 = 40 ✓

print("D=2: theory=14, brute=14 OK")
print("D=3: theory=24, brute=24 OK")
print("D=4: theory=40, brute=40 OK")
print()

# Great! The theory is validated. Now compute for D=8.
# Need to enumerate all valid A1 for D=8.

# Cyclotomic polynomials Phi_d with degree <= 8 and v_2(d) >= 1:
# v_2=1 (d=2*odd):
#   Phi_2 (deg 1), Phi_6 (deg 2), Phi_10 (deg 4), Phi_14 (deg 6), Phi_18 (deg 6), Phi_30 (deg 8)
# v_2=2 (d=4*odd):
#   Phi_4 (deg 2), Phi_12 (deg 4), Phi_20 (deg 8)
# v_2=3 (d=8*odd):
#   Phi_8 (deg 4), Phi_24 (deg 8)
# v_2=4 (d=16*odd):
#   Phi_16 (deg 8)

# But we also need products within a v_2 group!
# v_2=1 products:
# Single: Phi_2(1), Phi_6(2), Phi_10(4), Phi_14(6), Phi_18(6), Phi_30(8)
# Pairs: Phi_2*Phi_6(3), Phi_2*Phi_10(5), Phi_2*Phi_14(7), Phi_2*Phi_18(7),
#         Phi_6*Phi_10(6), Phi_6*Phi_14(8), Phi_6*Phi_18(8)
# Triples: Phi_2*Phi_6*Phi_10(7)
# That's it for degree <= 8.

# v_2=2 products:
# Single: Phi_4(2), Phi_12(4), Phi_20(8)
# Pairs: Phi_4*Phi_12(6)
# That's it.

# v_2=3 products:
# Single: Phi_8(4), Phi_24(8)
# No pairs fit in degree 8 (4+8=12 > 8).

# v_2=4 products:
# Single: Phi_16(8)

from itertools import combinations

def enumerate_subsets(group_degs, max_deg=8):
    """Enumerate all non-empty subsets with total degree <= max_deg."""
    results = []
    n = len(group_degs)
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            d = sum(group_degs[i] for i in combo)
            if d <= max_deg:
                results.append((combo, d))
    return results

groups = {
    1: [('Phi_2', 1), ('Phi_6', 2), ('Phi_10', 4), ('Phi_14', 6), ('Phi_18', 6), ('Phi_30', 8)],
    2: [('Phi_4', 2), ('Phi_12', 4), ('Phi_20', 8)],
    3: [('Phi_8', 4), ('Phi_24', 8)],
    4: [('Phi_16', 8)]
}

# Collect all non-empty subsets across all groups, recording degree
all_products = []  # list of (degree, description)

for e, phis in groups.items():
    degs = [d for (name, d) in phis]
    names = [name for (name, d) in phis]
    subsets = enumerate_subsets(degs)
    for (combo, total_deg) in subsets:
        desc = ' * '.join(names[i] for i in combo)
        all_products.append((total_deg, desc, e))

# For A1 of degree d, with epsilon in {+1,-1}:
# Number of A1 choices = 2 * (number of products of degree d)
# For each A1, s ranges from 0 to 8-d, giving (9-d) values.
# For A1 = +/-1 (degree 0): 2 * (9-0) = 18.

total = 18  # degree 0 case
print("Degree 0 (constant): 2 * 9 = 18")

by_degree = {}
for (d, desc, e) in all_products:
    by_degree.setdefault(d, []).append(desc)

for d in sorted(by_degree.keys()):
    products_at_d = by_degree[d]
    n_products = len(products_at_d)
    # 2 epsilon choices, (9-d) s choices
    contribution = 2 * n_products * (9 - d)
    total += contribution
    print(f"Degree {d}: {n_products} products, each with 2*(9-{d})={2*(9-d)} functions -> {contribution}")
    for desc in products_at_d:
        print(f"    {desc}")

print(f"\nTotal shifty functions for D=8: {total}")
