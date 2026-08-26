from sympy import totient
from collections import defaultdict
from itertools import combinations

even_n_list = []
for n in range(2, 200):
    if n % 2 == 0 and totient(n) <= 8:
        even_n_list.append(n)

groups = defaultdict(list)
for n in even_n_list:
    v2 = 0
    tmp = n
    while tmp % 2 == 0:
        v2 += 1
        tmp //= 2
    groups[v2].append(n)

print("Detailed subsets:\n")
for v2 in sorted(groups.keys()):
    ns = groups[v2]
    degs = [totient(n) for n in ns]
    print(f"Group v_2={v2}: Phi indices = {ns}, degrees = {degs}")
    for r in range(1, len(ns)+1):
        for combo in combinations(range(len(ns)), r):
            total_deg = sum(degs[i] for i in combo)
            if total_deg <= 8:
                names = [ns[i] for i in combo]
                print(f"  Subset: Phi_{names}, total deg = {total_deg}")
    print()

# But wait - I need to reconsider. The problem says alpha is a function Z -> Z.
# Two different polynomials A(x) give two different functions alpha.
# A(x) and -A(x) give different alpha (one is the negative of the other).
# So the count is 2 * 22 = 44... but let me double-check.
#
# Actually, is A(x) = 0 excluded? The problem says alpha has finitely many nonzero values,
# which is satisfied even if all values are 0. But then S_n(alpha)*beta = 0 for all n,
# which can never equal 1. So the zero function is NOT shifty. Good.
# And we already excluded the empty set giving 0 - wait no, the empty product gives 1, not 0.
# So we're fine.

# But wait: can +A and -A both be valid? Let me re-examine.
# If A(x) divides 1 + x^d, then -A(x) divides -(1+x^d). 
# We need (1+x^d)/A(x) to have integer coefficients. If A works, does -A?
# (1+x^d)/(-A(x)) = -(1+x^d)/A(x), which also has integer coefficients.
# And gamma = -(1+x^d)/A(x) corresponds to beta = -original_beta.
# S_n(alpha)*beta where alpha is from -A: S_n(-alpha)*(-beta) = S_n(alpha)*beta. Hmm.
# Actually we need ALPHA to be from A or -A, and then we choose beta.
# For alpha from -A and some beta: S_n(-alpha)*beta = -S_n(alpha)*beta.
# We need this to be 1 for n in {k,l} and 0 otherwise.
# So -S_n(alpha)*beta = 1 for n=k,l and 0 otherwise.
# This means S_n(alpha)*(-beta) = 1, same as before with -beta.
# So yes, if alpha works, -alpha also works (with -beta).
# So both +A and -A give valid shifty functions.

# Actually wait, let me re-examine whether ALL alpha satisfying the conditions are 
# exactly the coefficient vectors of +-products_of_cyclotomic_polys.
# The condition: A(x) | (1+x^d) in Z[x] for some d >= 1.
# (1+x^d) = Prod of certain Phi_n(x).
# A(x) must be a SIGNED product of a SUBSET of these Phi_n.
# But the sign can only be +1 or -1 (leading coeff of 1+x^d is 1).
# A(x)*Q(x) = 1+x^d. Leading coeffs multiply to 1, constant terms multiply to 1.
# So lc(A)*lc(Q)=1 and A(0)*Q(0)=1.
# Since cyclotomic polys are monic with constant term +/-1,
# a product of cyclotomic polys is monic with constant term +/-1.
# So A monic means lc=1, Q monic, lc(Q)=1. Then A(0)*Q(0)=1 is automatically satisfied
# since both constant terms are +/-1.
#
# If A = -Phi_n: lc = -1 (for deg >= 1). Then lc(Q) = -1.
# Q = -((1+x^d)/Phi_n), which has leading coeff -1. And Q(0) = -(1/Phi_n(0)) = -(+/-1) = -/+1.
# A(0)*Q(0) = (-Phi_n(0))*(-(1+0)/Phi_n(0)) = Phi_n(0)*(1/Phi_n(0)) = 1. OK.
#
# So -A also works. Each signed product gives a distinct polynomial (A != -A for nonzero A).

# Actually, I realize I need to reconsider more carefully. The problem is asking about
# alpha: Z -> Z with alpha(m) = 0 for m < 0 and m > 8. So alpha is determined by
# (alpha(0), alpha(1), ..., alpha(8)), a vector of 9 integers.
# Two different polynomials A(x) = sum alpha(m) x^m give different alpha.
# So the answer is the number of valid A(x) polynomials, including sign.

# Let me also check: can two different (v2_group, subset, sign) pairs give the same polynomial?
# Products from different v2 groups involve different cyclotomic factors, so they're different.
# Different subsets within the same group give different factorizations -> different polynomials.
# +A vs -A: always different (for A != 0).
# So the count 2 * 22 = 44 should be correct.

# Wait, but I should also check: is {empty set} from different groups the same?
# The empty product is 1 regardless of which group. But I counted it only once.
# count_subsets = 1 (empty) + 14 + 4 + 2 + 1 = 22.
# With sign: 2 * 22 = 44.

# Hmm, but actually maybe I should double-check by direct computation for small d.
# Let me find all monic divisors of 1+x^d for small d, of degree <= 8.

from sympy import Poly, ZZ, factor, div

def get_divisors_of_xd_plus_1(d, max_deg=8):
    """Find all monic polynomial divisors of 1+x^d in Z[x] with degree <= max_deg."""
    from sympy import Symbol, factor_list
    x = Symbol('x')
    poly = 1 + x**d
    fl = factor_list(poly, x, domain='ZZ')
    # fl = (content, [(factor1, mult1), (factor2, mult2), ...])
    content = fl[0]
    factors = fl[1]
    # Each factor appears with multiplicity (should be 1 for cyclotomic)
    # Generate all subsets
    result = set()
    n = len(factors)
    for mask in range(1 << n):
        p = content if mask == 0 else 1  # hmm, actually for mask=0 it's the empty product = 1
        p = 1
        deg = 0
        ok = True
        for i in range(n):
            if mask & (1 << i):
                fi, mi = factors[i]
                # fi is a sympy expression
                fi_poly = Poly(fi, x, domain='ZZ')
                deg += fi_poly.degree() * mi
                if deg > max_deg:
                    ok = False
                    break
        if ok:
            # Compute the product
            p_poly = Poly(1, x, domain='ZZ')
            for i in range(n):
                if mask & (1 << i):
                    fi, mi = factors[i]
                    fi_poly = Poly(fi**mi, x, domain='ZZ')
                    p_poly = p_poly * fi_poly
            result.add(tuple(p_poly.all_coeffs()))
    return result

all_polys = set()
for d in range(1, 200):
    divs = get_divisors_of_xd_plus_1(d)
    for p in divs:
        all_polys.add(p)

print(f"\nTotal distinct monic divisors of 1+x^d (d=1..199) with deg <= 8: {len(all_polys)}")
# With sign: multiply by 2
print(f"Total shifty functions: {2 * len(all_polys)}")

# Let me also show the polynomials
for p in sorted(all_polys, key=lambda t: (len(t), t)):
    print(f"  {p}")
