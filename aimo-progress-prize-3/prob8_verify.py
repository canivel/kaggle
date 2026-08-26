
# Let me verify the analysis with brute force for small cases.
# Instead of degree <= 8, let's check for degree <= 2 first and compare.

# A function alpha supported on {0,...,D} (D small).
# We need to find beta in F and k != l such that
# S_n(alpha)*beta = 1 if n in {k,l}, 0 otherwise.
# i.e., sum_t alpha(t+n)*beta(t) = [n=k] + [n=l]

# In polynomial terms: A(x)*C(x) = x^k + x^l where C(x) = B(1/x).
# C is a Laurent polynomial. So A(x) * (Laurent poly) = x^k + x^l.

# Let me do brute force: for each alpha supported on {0,...,D} with integer values,
# check if there exist beta and k != l satisfying the condition.

# For the brute force, we need to work with generating functions.
# A(x) * C(x) = x^k + x^l. We can assume k > l (WLOG).
# Then C(x) = (x^k + x^l) / A(x).
# C must be a Laurent polynomial with INTEGER coefficients.

# Equivalently, we need A(x) | (x^k + x^l) in Z[x, x^{-1}].
# x^k + x^l = x^l(x^{k-l} + 1), so A(x) | x^l(x^{k-l}+1).
# Writing A(x) = x^s * A1(x) with A1(0) != 0,
# we need x^s * A1(x) | x^l * (x^{k-l}+1).
# So x^s | x^l (need s <= l, or adjust l) and A1(x) | (x^{k-l}+1).
# Actually since l can be any integer (not necessarily non-negative!), x^s | x^l is free.
# So the only real constraint is A1(x) | x^N + 1 for some N >= 1 (with N = k-l, and we need k != l so N >= 1).
# And the quotient (x^N+1)/A1(x) must have integer coefficients.

# Wait, but what about A1 = -A1'? If A1(x) = -(x+1), then (x^N+1)/A1(x) = (x^N+1)/(-(x+1)) = -(x^{N-1} - x^{N-2} + ... - 1) when N is odd.
# This has integer coefficients. So yes, A1 can have leading coefficient -1.
# Then C(x) = x^{l-s} * (x^N+1)/A1(x), which has integer coefficients.

# But the SIGN matters. We need A(x)*C(x) = x^k + x^l EXACTLY.
# If A1 has leading coeff -1, then quotient (x^N+1)/A1 starts with leading coeff -1.
# Then C(x) = x^{l-s} * (x^N+1)/A1(x).
# A(x)*C(x) = x^s * A1(x) * x^{l-s} * (x^N+1)/A1(x) = x^l * (x^N + 1) = x^{l+N} + x^l.
# So we always get x^k + x^l with k = l+N. This works regardless of the sign of the quotient.
# The key is: A(x)*C(x) = x^{l+N} + x^l. C just needs to exist as a Laurent poly with integer coeffs.

# So the condition is simply: A1(x) | x^N + 1 in Z[x] for some N >= 1.

# Now: which polynomials with A1(0) != 0 and integer coefficients divide x^N + 1 in Z[x]?
#
# x^N + 1 is a product of cyclotomic polynomials. The divisors of x^N + 1 in Z[x] are:
# {epsilon * prod_{d in S} Phi_d(x) : S subset of cyclotomic factors of x^N+1, epsilon in {+1,-1}}
#
# Wait - not quite. The divisors must also have leading coefficient +1 or -1 and constant term +1 or -1
# for the division to be exact in Z[x]. Actually, since x^N+1 is monic with constant term 1,
# if f*g = x^N+1 with f,g in Z[x], by Gauss's lemma the content of f times content of g = 1,
# so both are primitive. And leading coeff of f times leading coeff of g = 1, so both are +1 or both -1.
# Similarly constant: f(0)*g(0) = 1.
#
# So divisors are: products of subsets of {Phi_d} (which are monic with Phi_d(0) = 1... wait is Phi_d(0)=1?)
# Phi_d(0) = 1 for d > 1. Phi_1(0) = -1 but Phi_1 never divides x^N+1.
# So all Phi_d dividing x^N+1 have Phi_d(0) = 1, and their products are monic with constant term 1.
#
# We can also take -1 times any such product? If A1 = -(Phi_d1 * ... * Phi_dk),
# then A1 * (-D1) = Phi_d1*...*Phi_dk * D1 subset of x^N+1. But then we need A1 * C1 = x^N+1
# where C1 = (x^N+1)/A1 = -(x^N+1)/(prod Phi_di). This has integer coefficients.
# And A1 * C1 = -(prod Phi_di) * (-(x^N+1)/(prod Phi_di)) = x^N+1. Yes!
# So A1 = -(prod of some Phi's) is also a valid divisor.

# Therefore the MONIC products of subsets of cyclotomic factors give valid A1,
# and their negatives also give valid A1.
# Plus A1 = 1 and A1 = -1 (the empty product and its negative).

# Now I need to verify my approach by brute force for small degree bound.
# Let D = 3 (alpha supported on {0,1,2,3}).

import numpy as np
from itertools import product as iproduct

def check_shifty_brute(alpha_coeffs, max_N=100):
    """Check if alpha with given coefficients is shifty by brute force.
    alpha_coeffs = [alpha(0), alpha(1), ..., alpha(D)]
    """
    from numpy.polynomial import polynomial as P

    # A(x) = sum alpha_coeffs[i] * x^i
    A = list(alpha_coeffs)

    # Remove trailing zeros to get actual degree
    while len(A) > 1 and A[-1] == 0:
        A.pop()

    if all(a == 0 for a in A):
        return False  # zero function can't give inner product 1

    # We need A(x) | x^l * (x^N + 1) for some N >= 1 and some integer l
    # Write A(x) = x^s * A1(x) with A1[0] != 0
    s = 0
    while s < len(A) and A[s] == 0:
        s += 1
    A1 = A[s:]

    # Check: does A1(x) divide x^N + 1 for some N in range?
    for N in range(1, max_N + 1):
        # x^N + 1 = [1, 0, 0, ..., 0, 1] (polynomial coefficients, low to high)
        dividend = [0] * (N + 1)
        dividend[0] = 1
        dividend[N] = 1

        # Polynomial division of dividend by A1
        q, r = poly_div(dividend, A1)
        if all(abs(ri) < 1e-10 for ri in r):
            # Check quotient has integer coefficients
            if all(abs(qi - round(qi)) < 1e-10 for qi in q):
                return True

    return False

def poly_div(dividend, divisor):
    """Polynomial long division. Coefficients are low-to-high."""
    import numpy as np
    # Convert to high-to-low for numpy
    d1 = list(reversed(dividend))
    d2 = list(reversed(divisor))
    q, r = np.polydiv(d1, d2)
    return list(reversed(q)), list(reversed(r))

# Test for D = 3 (support on {0,1,2,3})
# Enumerate all alpha with small integer values
D = 3
val_range = range(-3, 4)  # values from -3 to 3

count_brute = 0
shifty_list = []

for coeffs in iproduct(val_range, repeat=D+1):
    if check_shifty_brute(coeffs):
        count_brute += 1
        shifty_list.append(coeffs)

print(f"Brute force count for D={D}, values in {list(val_range)}: {count_brute}")

# Now count using the cyclotomic theory for D=3
# A(x) = x^s * A1(x), deg(A) <= 3, A1 | x^N+1 for some N >= 1
# A1 = epsilon * prod(Phi_d) where all d have same v_2 >= 1, or A1 = epsilon * 1

# Cyclotomic polys with degree <= 3:
# Phi_2 (deg 1): x+1
# Phi_6 (deg 2): x^2-x+1
# Phi_4 (deg 2): x^2+1
# Phi_2*Phi_6 (deg 3): (x+1)(x^2-x+1) = x^3+1... wait that's x^3+1? No.
# (x+1)(x^2-x+1) = x^3 - x^2 + x + x^2 - x + 1 = x^3 + 1. Yes!

# Available cyclotomic products with degree <= 3:
# v_2=1: Phi_2 (deg 1), Phi_6 (deg 2), Phi_2*Phi_6 (deg 3)
# v_2=2: Phi_4 (deg 2)
# v_2=3: nothing with deg <= 3 (Phi_8 has deg 4)
# v_2=4: nothing

# A1 options (with epsilon):
# degree 0: epsilon = +1 or -1 (2 options)
# degree 1: +Phi_2, -Phi_2 (v_2=1)
# degree 2: +Phi_6, -Phi_6 (v_2=1), +Phi_4, -Phi_4 (v_2=2)
# degree 3: +Phi_2*Phi_6, -Phi_2*Phi_6 (v_2=1)

# For each A1 of degree d, s can be 0,...,3-d.
# Total count:
# d=0: 2 * 4 = 8
# d=1: 2 * 3 = 6  (just Phi_2 with +-1)
# d=2: 4 * 2 = 8  (Phi_6 and Phi_4, each with +-1)  -> 4 choices of A1, each with (3-2+1)=2 choices of s
# d=3: 2 * 1 = 2  (Phi_2*Phi_6 with +-1)

theory_count = 2*4 + 2*3 + 4*2 + 2*1  # = 8 + 6 + 8 + 2 = 24

print(f"Theory count for D=3: {8 + 6 + 8 + 2}")

# But wait, the theory counts POLYNOMIALS, not functions.
# Two different polynomials A(x) could give the same alpha function.
# Actually no: A(x) = sum alpha(m) x^m for m=0,...,D uniquely determines alpha.
# But alpha has support on {0,...,D}, so the polynomial A of degree <= D
# uniquely corresponds to the function alpha. (alpha(m) = coefficient of x^m.)
# So the theory count should match the brute force.

# But brute force only checks values -3 to 3! We might be missing functions
# with larger coefficient values. Let me check what coefficients appear.
print(f"\nShifty functions found (D=3, values -3..3):")
for s in shifty_list:
    print(f"  {s}")
