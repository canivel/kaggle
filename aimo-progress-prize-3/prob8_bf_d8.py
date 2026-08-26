
# Brute force verification for D=8
# We check all A1(x) with:
# - degree d, 1 <= d <= 8
# - leading coeff and constant term in {+1, -1}
# - intermediate coefficients in {-2, -1, 0, 1, 2} (cyclotomic polys have small coeffs)
# - A1 divides x^N + 1 for some N in [1, 500]

from itertools import product as iproduct

def poly_divides(A1, N):
    """Check if A1(x) divides x^N + 1 over Z. A1 coefficients low-to-high."""
    dividend = [0] * (N + 1)
    dividend[0] = 1
    dividend[N] = 1

    d1 = list(reversed(dividend))
    d2 = list(reversed(A1))

    if len(d2) == 0 or d2[0] == 0:
        return False

    r = list(d1)
    while len(r) >= len(d2):
        if r[0] % d2[0] != 0:
            return False
        coeff = r[0] // d2[0]
        for i in range(len(d2)):
            r[i] -= coeff * d2[i]
        r.pop(0)

    return all(ri == 0 for ri in r)

max_N = 500
coeff_range = range(-2, 3)  # -2 to 2

valid_A1_by_deg = {}

# Degree 0
valid_A1_by_deg[0] = [(1,), (-1,)]

for d in range(1, 9):
    valid = set()
    for lc in [1, -1]:
        for a0 in [1, -1]:
            if d == 1:
                mids_list = [()]
            else:
                mids_list = iproduct(coeff_range, repeat=d-1)

            for mid in mids_list:
                A1 = (a0,) + mid + (lc,) if d > 1 else (a0, lc)
                # Quick check: A1 should have non-zero leading and constant
                assert A1[0] != 0 and A1[-1] != 0

                for N in range(1, max_N + 1):
                    if poly_divides(list(A1), N):
                        valid.add(A1)
                        break

    valid_A1_by_deg[d] = list(valid)
    print(f"Degree {d}: {len(valid)} valid A1 polynomials")
    for a1 in sorted(valid, key=lambda x: x):
        print(f"  {a1}")

# Count total shifty functions
total = 0
for d, polys in valid_A1_by_deg.items():
    s_choices = 9 - d  # s in {0, ..., 8-d}
    contribution = len(polys) * s_choices
    total += contribution
    print(f"Degree {d}: {len(polys)} A1s * {s_choices} s-values = {contribution}")

print(f"\nTotal shifty functions: {total}")
