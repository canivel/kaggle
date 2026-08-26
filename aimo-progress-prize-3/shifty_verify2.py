"""
Additional verification for higher degrees.
Key insight: if P0(x) | (1+x^d) in Z[x], then by Gauss's lemma, every root of P0
is a root of unity (specifically a primitive nth root for some n dividing 2d but not d).
So P0 must be a product of cyclotomic polynomials (up to sign).

This means our enumeration of products of cyclotomic polynomials is COMPLETE.
We just need to verify we checked enough values of d.

Actually, the question is: did we find ALL cyclotomic polynomial products of degree <= 8
that divide SOME 1+x^d? The answer is yes if we checked enough d values.

A product of cyclotomic polys Phi_{n1} * ... * Phi_{nr} divides 1+x^d iff
each Phi_{ni} divides 1+x^d, which means each ni | 2d and ni does not divide d.

For each ni, the set of valid d is: d such that ni | 2d and ni does not divide d.
If ni = 2^a * m (m odd), then we need 2d divisible by ni, so d divisible by ni/2
if ni is even, or d divisible by ni if ni is odd... Actually let me think differently.

ni | 2d means 2d = ni * q for some integer q, so d = ni*q/2.
ni does not divide d means d/ni is not an integer.

For a set {n1,...,nr}, we need a single d that works for ALL of them simultaneously.
Such d always exists (by CRT-like argument) as long as the conditions are compatible.

Let me verify: for each pair of cyclotomic indices from our small_cyclo list,
check if they can co-occur in some x^d+1.
"""

from sympy import totient, divisors, gcd, lcm
from itertools import combinations

def valid_d_exists(n_list, max_d=10000):
    """Check if there exists d such that all n in n_list divide x^d+1"""
    for d in range(1, max_d):
        all_ok = True
        for n in n_list:
            if (2*d) % n != 0 or d % n == 0:
                all_ok = False
                break
        if all_ok:
            return d
    return None

# Get our small cyclotomic indices
small_cyclo = []
for n in range(1, 200):
    if totient(n) <= 8:
        small_cyclo.append(n)

print("Small cyclotomic indices:", small_cyclo)
print("Their degrees:", [totient(n) for n in small_cyclo])

# For each subset of small_cyclo with total degree <= 8, check if a valid d exists
print("\nChecking all subsets of cyclotomic polys with degree <= 8...")

all_valid_subsets = []
small_degs = {n: totient(n) for n in small_cyclo}

# We need to be smart about enumeration.
# Max number of factors: degree 8 can be achieved with up to 8 factors of degree 1.
# But phi(n)=1 only for n=1,2. So at most 2 factors of degree 1.
# Let's enumerate systematically.

count_subsets = 0
count_valid = 0

for r in range(1, len(small_cyclo) + 1):
    any_found = False
    for subset in combinations(small_cyclo, r):
        total_deg = sum(small_degs[n] for n in subset)
        if total_deg > 8:
            continue
        count_subsets += 1
        d = valid_d_exists(subset, max_d=5000)
        if d is not None:
            count_valid += 1
            all_valid_subsets.append((subset, d))
            any_found = True
    if r >= 4 and not any_found:
        # Check if we should stop
        pass

print(f"Total subsets checked: {count_subsets}")
print(f"Valid subsets (where compatible d exists): {count_valid}")

# Now count unique polynomials
from sympy import Symbol, cyclotomic_poly, expand, Poly, ZZ

x = Symbol('x')

unique_polys = set()
for subset, d in all_valid_subsets:
    prod = 1
    for n in subset:
        prod *= cyclotomic_poly(n, x)
    prod = expand(prod)
    # Convert to tuple
    p = Poly(prod, x, domain=ZZ)
    deg = p.degree()
    coeffs = [0] * (deg + 1)
    for monom, coeff in p.as_dict().items():
        coeffs[monom[0]] = int(coeff)
    tup = tuple(coeffs)
    unique_polys.add(tup)

# Add constant 1
unique_polys.add((1,))

print(f"\nUnique P0 polynomials (positive): {len(unique_polys)}")

# With signs
all_P0 = set()
for tup in unique_polys:
    all_P0.add(tup)
    all_P0.add(tuple(-c for c in tup))

print(f"With signs: {len(all_P0)}")

# Count shifty functions
total = 0
for tup in all_P0:
    d0 = len(tup) - 1
    total += 9 - d0

print(f"\nTotal shifty functions: {total}")
