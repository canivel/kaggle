
# Brute force for small D: check which polynomials A(x) with integer coefficients
# and degree <= D divide x^N + 1 for some N >= 1 (after removing the x^s factor).
#
# We enumerate A1(x) monic with A1(0) != 0 and check divisibility.
# Then count functions alpha = epsilon * x^s * A1(x).

import numpy as np

def poly_divides(A1, N):
    """Check if A1(x) divides x^N + 1 over Z.
    A1 is a list of coefficients [a0, a1, ..., ad] (low to high).
    Returns True if division is exact with integer quotient."""
    # Build x^N + 1
    dividend = [0] * (N + 1)
    dividend[0] = 1
    dividend[N] = 1

    # Polynomial long division (high to low)
    d1 = list(reversed(dividend))
    d2 = list(reversed(A1))

    if len(d2) == 0 or d2[0] == 0:
        return False

    q = []
    r = list(d1)
    while len(r) >= len(d2):
        # Leading coeff ratio must be integer
        if r[0] % d2[0] != 0:
            return False
        coeff = r[0] // d2[0]
        q.append(coeff)
        for i in range(len(d2)):
            r[i] -= coeff * d2[i]
        r.pop(0)

    return all(ri == 0 for ri in r)


def count_shifty(D, max_N=200):
    """Count shifty functions supported on {0,...,D}."""
    # For each polynomial A(x) = x^s * A1(x), deg(A) <= D, A1(0) != 0,
    # A1 divides x^N + 1 for some N >= 1.
    # The function alpha is determined by the coefficients of A(x).

    # First find all valid A1 (with A1(0) != 0, integer coefficients)
    # that divide some x^N + 1.
    # Since x^N + 1 has leading coeff 1 and constant term 1,
    # and A1 * Q = x^N + 1, the leading coeff of A1 must divide 1,
    # and A1(0) must divide 1.
    # So A1 has leading coeff +1 or -1, and A1(0) = +1 or -1.

    # We can enumerate: for each degree d from 0 to D,
    # for each A1 of degree d with leading coeff in {+1,-1} and A1(0) in {+1,-1},
    # and all intermediate coefficients are integers (but what range?),
    # check if A1 | x^N + 1 for some N.

    # For cyclotomic polynomials, coefficients are bounded.
    # All Phi_d with degree <= 8 have coefficients in {-1, 0, 1}.
    # Products of such Phi_d also have bounded coefficients.
    # For degree <= 8, let's check coefficients in range [-8, 8] to be safe.

    valid_A1 = set()  # store as tuples of coefficients (low to high)

    coeff_range = range(-4, 5)  # should be enough for small D

    from itertools import product as iproduct

    for d in range(0, D + 1):
        if d == 0:
            # A1 = 1 or -1
            for a0 in [1, -1]:
                # Check: does constant a0 divide x^N + 1? Yes, for any N.
                valid_A1.add((a0,))
        else:
            # leading coeff in {1, -1}, constant in {1, -1}, middle in coeff_range
            for lc in [1, -1]:
                for a0 in [1, -1]:
                    if d == 1:
                        A1 = [a0, lc]
                        found = False
                        for N in range(1, max_N + 1):
                            if poly_divides(A1, N):
                                found = True
                                break
                        if found:
                            valid_A1.add(tuple(A1))
                    else:
                        for mid_coeffs in iproduct(coeff_range, repeat=d - 1):
                            A1 = [a0] + list(mid_coeffs) + [lc]
                            found = False
                            for N in range(1, max_N + 1):
                                if poly_divides(A1, N):
                                    found = True
                                    break
                            if found:
                                valid_A1.add(tuple(A1))

    print(f"D={D}: Found {len(valid_A1)} valid A1 polynomials")
    for a1 in sorted(valid_A1, key=lambda x: (len(x), x)):
        print(f"  {a1}")

    # Count shifty functions
    total = 0
    for a1 in valid_A1:
        d = len(a1) - 1  # degree of A1
        s_choices = D - d + 1  # s in {0, ..., D - d}
        total += s_choices

    print(f"Total shifty functions for D={D}: {total}")
    return total


# Test for D=2
count_shifty(2, max_N=100)
print()

# Test for D=3
count_shifty(3, max_N=100)
print()

# Now for D=4
count_shifty(4, max_N=200)
