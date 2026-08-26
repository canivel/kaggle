"""
Shifty functions solver.

P(x) = sum alpha(k)*x^k, degree <= 8, supported on {0,...,8}.
Condition: P(x) * R(x) = x^k + x^l for some Laurent polynomial R with integer
coefficients and integers k != l.

Clearing denominators: P(x) * S(x) = x^a + x^b for polynomial S in Z[x], a != b.
So P(x) | x^a + x^b = x^min(a,b) * (1 + x^{|a-b|}).

Write P(x) = x^j * P0(x) where P0(0) != 0.
Then P0(x) | (1 + x^d) for some d >= 1.

1 + x^d factors into cyclotomic polynomials Phi_n where n | 2d but n does not divide d.

We enumerate all products of cyclotomic polynomials (with degree <= 8) that divide
some 1 + x^d, then count shifty functions.
"""

from sympy import divisors, cyclotomic_poly, Symbol, ZZ, expand, totient
from sympy.polys.polytools import Poly
from itertools import combinations

x = Symbol('x')


def get_cyclotomic_factors_of_xd_plus_1(d):
    """Return list of cyclotomic polynomial indices n such that Phi_n | x^d+1"""
    divs_2d = set(divisors(2 * d))
    divs_d = set(divisors(d))
    return sorted(divs_2d - divs_d)


def poly_to_tuple(p):
    """Convert sympy polynomial to tuple of coefficients (constant first)"""
    poly = Poly(p, x, domain=ZZ)
    d = poly.degree()
    coeffs = [0] * (d + 1)
    for monom, coeff in poly.as_dict().items():
        coeffs[monom[0]] = int(coeff)
    return tuple(coeffs)


# Find all cyclotomic polys with degree <= 8
small_cyclo = []
for n in range(1, 200):
    if totient(n) <= 8:
        small_cyclo.append(n)

print("Cyclotomic indices with phi(n) <= 8:", small_cyclo)

max_deg = 8
found_polys = set()

for d in range(1, 1000):
    # Find cyclotomic factors of x^d+1 with degree <= 8
    factors_n = get_cyclotomic_factors_of_xd_plus_1(d)
    small_factors = [n for n in factors_n if totient(n) <= 8]

    if not small_factors:
        continue

    # Get the actual polynomials
    factor_polys = []
    factor_degs = []
    for n in small_factors:
        cp = cyclotomic_poly(n, x)
        deg = totient(n)
        factor_polys.append(cp)
        factor_degs.append(deg)

    # Enumerate all nonempty subsets with total degree <= 8
    nf = len(factor_polys)
    for r in range(1, nf + 1):
        for subset in combinations(range(nf), r):
            total_deg = sum(factor_degs[i] for i in subset)
            if total_deg > max_deg:
                continue
            # Compute product
            prod = 1
            for i in subset:
                prod = prod * factor_polys[i]
            prod = expand(prod)
            tup = poly_to_tuple(prod)
            found_polys.add(tup)

print(f"Found {len(found_polys)} distinct P0 polynomials (monic-style)")

# Include the constant polynomial 1 (empty product)
found_polys.add((1,))

# Now add negatives: if P0 | (1+x^d), then (-P0) | (1+x^d) too
all_P0 = set()
for tup in found_polys:
    all_P0.add(tup)
    neg_tup = tuple(-c for c in tup)
    all_P0.add(neg_tup)

print(f"With signs: {len(all_P0)} P0 polynomials")

# Verify P0(0) != 0 for all
for tup in all_P0:
    assert tup[0] != 0, f"P0 with zero constant term: {tup}"

# Count total shifty functions:
# P(x) = x^j * P0(x), j >= 0, j + deg(P0) <= 8
# Each gives a unique alpha since P0(0) != 0 determines j uniquely.
total = 0
from collections import Counter
deg_count = Counter()
for tup in all_P0:
    d0 = len(tup) - 1  # degree of P0
    num_shifts = 9 - d0  # j from 0 to 8-d0
    deg_count[d0] += 1
    total += num_shifts

print(f"\nTotal shifty functions: {total}")

for d in sorted(deg_count):
    print(f"  Degree {d}: {deg_count[d]} P0 polynomials, each with {9-d} shifts = {deg_count[d] * (9-d)}")

# Print all P0 polynomials for verification
print("\nAll P0 polynomials (coefficient tuples, constant first):")
for tup in sorted(all_P0, key=lambda t: (len(t), t)):
    print(f"  deg {len(tup)-1}: {tup}")
