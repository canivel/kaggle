"""
Verify the shifty function count by direct computation.

For each candidate P(x) (degree <= 8, integer coefficients, with P(x) = x^j * P0(x)),
verify that P(x) * S(x) = x^a + x^b is solvable for some polynomial S in Z[x].

This means P(x) divides x^a + x^b = x^min(a,b) * (1 + x^{|a-b|}) in Z[x].
Since P(x) = x^j * P0(x) with P0(0) != 0, we need P0(x) | (1 + x^d) for some d >= 1.

Let's verify: for each P0 we found, check that it actually divides 1 + x^d for some d.
"""

from sympy import Symbol, Poly, ZZ, cyclotomic_poly, expand, div, totient, divisors
from itertools import combinations

x = Symbol('x')

all_P0_tuples = [
    (-1,), (1,),
    (-1, -1), (1, 1),
    (-1, 0, -1), (-1, 1, -1), (1, -1, 1), (1, 0, 1),
    (-1, 0, 0, -1), (1, 0, 0, 1),
    (-1, 0, 0, 0, -1), (-1, 0, 1, 0, -1), (-1, 1, -1, 1, -1),
    (1, -1, 1, -1, 1), (1, 0, -1, 0, 1), (1, 0, 0, 0, 1),
    (-1, 0, 0, 0, 0, -1), (1, 0, 0, 0, 0, 1),
    (-1, 0, 0, 0, 0, 0, -1), (-1, 0, 0, 1, 0, 0, -1),
    (-1, 1, -1, 1, -1, 1, -1), (-1, 2, -3, 3, -3, 2, -1),
    (1, -2, 3, -3, 3, -2, 1), (1, -1, 1, -1, 1, -1, 1),
    (1, 0, 0, -1, 0, 0, 1), (1, 0, 0, 0, 0, 0, 1),
    (-1, -1, 0, 1, 1, 0, -1, -1), (-1, 0, 0, 0, 0, 0, 0, -1),
    (-1, 1, -1, 0, 0, -1, 1, -1), (1, -1, 1, 0, 0, 1, -1, 1),
    (1, 0, 0, 0, 0, 0, 0, 1), (1, 1, 0, -1, -1, 0, 1, 1),
    (-1, -1, 0, 1, 1, 1, 0, -1, -1), (-1, 0, 0, 0, 0, 0, 0, 0, -1),
    (-1, 0, 0, 0, 1, 0, 0, 0, -1), (-1, 0, 1, 0, -1, 0, 1, 0, -1),
    (-1, 1, -1, 1, -1, 1, -1, 1, -1), (-1, 2, -3, 3, -3, 3, -3, 2, -1),
    (1, -2, 3, -3, 3, -3, 3, -2, 1), (1, -1, 1, -1, 1, -1, 1, -1, 1),
    (1, 0, -1, 0, 1, 0, -1, 0, 1), (1, 0, 0, 0, -1, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1), (1, 1, 0, -1, -1, -1, 0, 1, 1),
]

def tuple_to_poly(tup):
    """Convert coefficient tuple (constant first) to sympy expression"""
    result = 0
    for i, c in enumerate(tup):
        result += c * x**i
    return result

# Verify each P0 divides 1 + x^d for some d
print("Verifying each P0...")
all_verified = True
for tup in all_P0_tuples:
    p = tuple_to_poly(tup)
    p_poly = Poly(p, x, domain=ZZ)
    found = False
    for d in range(1, 500):
        target = 1 + x**d
        t_poly = Poly(target, x, domain=ZZ)
        q, r = div(t_poly, p_poly, domain=ZZ)
        if r.is_zero:
            found = True
            break
    if not found:
        print(f"  FAILED: {tup} does not divide 1+x^d for d in [1,499]")
        all_verified = False

if all_verified:
    print("All P0 verified!")

# Now let's also verify completeness: are there any other polynomials P0
# of small degree that divide 1+x^d but we missed?
# We'll brute-force for small degrees.

print("\nBrute-force checking for missed P0 polynomials...")

# For degree 0: P0 = c (constant). P0 | (1+x^d) means c | every coefficient of 1+x^d.
# Coefficients of 1+x^d are all 0 or 1. So c | 1, meaning c = ±1. Check.

# For degree 1: P0 = a + bx, P0(0) = a != 0.
# P0 | (1+x^d) for some d.
# The roots of P0 are -a/b. For P0 | (1+x^d), we need (-a/b)^d = -1.
# Since a,b are integers with |a|,|b| dividing the leading/trailing coeff of 1+x^d,
# by Gauss lemma, a | 1 and b | 1. So a,b in {-1,1}.
# P0 = 1+x or P0 = 1-x or P0 = -1+x or P0 = -1-x.
# But P0 = 1+x: root is -1, (-1)^d = -1 for odd d. Works. (1+x) | (1+x^d) for odd d.
# P0 = 1-x: root is 1, 1^d = -1? No, 1+1 = 2 != 0. So (1-x) does NOT divide (1+x^d).
# P0 = -1+x = x-1: same issue. (-1+x) has root 1, but 1+1^d = 2 != 0.
# P0 = -1-x = -(1+x): same as -(1+x), already covered by signs.
# So degree 1: only ±(1+x). Matches what we found.

# For degree 2, let's be more systematic
print("\nChecking degree 2 P0 by brute force...")
# P0 = a0 + a1*x + a2*x^2, a0 != 0, a2 != 0
# By Gauss lemma, a0 | 1 and a2 | 1 (since leading and trailing coeff of 1+x^d are 1)
# So a0, a2 in {-1, 1}. a1 can be any integer.
# But coefficients of P0 * Q = 1 + x^d must all be in {0, 1} (for d >= 3, they're 0 or 1).
# Actually the coefficients of 1+x^d are: 1 at position 0, 1 at position d, 0 elsewhere.
# So |a1| is bounded. Let's just check.

for a0 in [-1, 1]:
    for a2 in [-1, 1]:
        for a1 in range(-5, 6):
            if a1 == 0 and a0 == a2:
                # P0 = a0(1 + x^2)
                pass
            tup = (a0, a1, a2)
            p = tuple_to_poly(tup)
            p_poly = Poly(p, x, domain=ZZ)
            for d in range(2, 200):
                target = 1 + x**d
                t_poly = Poly(target, x, domain=ZZ)
                q, r = div(t_poly, p_poly, domain=ZZ)
                if r.is_zero:
                    if tup not in all_P0_tuples:
                        print(f"  MISSED degree 2: {tup} divides 1+x^{d}")
                    break

print("Degree 2 check done.")

# Check degree 3
print("\nChecking degree 3 P0 by brute force...")
for a0 in [-1, 1]:
    for a3 in [-1, 1]:
        for a1 in range(-3, 4):
            for a2 in range(-3, 4):
                tup = (a0, a1, a2, a3)
                p = tuple_to_poly(tup)
                p_poly = Poly(p, x, domain=ZZ)
                for d in range(3, 200):
                    target = 1 + x**d
                    t_poly = Poly(target, x, domain=ZZ)
                    q, r = div(t_poly, p_poly, domain=ZZ)
                    if r.is_zero:
                        if tup not in all_P0_tuples:
                            print(f"  MISSED degree 3: {tup} divides 1+x^{d}")
                        break

print("Degree 3 check done.")

# Check degree 4
print("\nChecking degree 4 P0 by brute force...")
for a0 in [-1, 1]:
    for a4 in [-1, 1]:
        for a1 in range(-3, 4):
            for a2 in range(-3, 4):
                for a3 in range(-3, 4):
                    tup = (a0, a1, a2, a3, a4)
                    p = tuple_to_poly(tup)
                    p_poly = Poly(p, x, domain=ZZ)
                    for d in range(4, 100):
                        target = 1 + x**d
                        t_poly = Poly(target, x, domain=ZZ)
                        q, r = div(t_poly, p_poly, domain=ZZ)
                        if r.is_zero:
                            if tup not in all_P0_tuples:
                                print(f"  MISSED degree 4: {tup} divides 1+x^{d}")
                            break

print("Degree 4 check done.")

print("\nFinal answer: 160")
