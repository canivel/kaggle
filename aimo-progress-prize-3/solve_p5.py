"""
Problem 5: Tournament with 2^20 runners, Swiss-system pairing.
Count N = number of possible final orderings.
Find k = largest power of 10 dividing N, then k mod 10^5.

Key insight: N(n) = prod_{j=0}^{n-1} C_{2^j}^{2^{n-1-j}}
where C_m is the m-th Catalan number.
"""

from math import comb, factorial

def vp_factorial(n, p):
    """Compute v_p(n!) using Legendre's formula"""
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def vp_catalan(m, p):
    """Compute v_p(C_m) where C_m = (2m)!/((m+1)!*m!)"""
    return vp_factorial(2*m, p) - vp_factorial(m+1, p) - vp_factorial(m, p)

def compute_vp_N(n, p):
    """Compute v_p(N(n))"""
    total = 0
    for j in range(n):
        m = 2**j
        exponent = 2**(n-1-j)
        vc = vp_catalan(m, p)
        total += exponent * vc
    return total

# For n=20
v2 = compute_vp_N(20, 2)
v5 = compute_vp_N(20, 5)
k = min(v2, v5)
print(f"v_2(N(20)) = {v2}")
print(f"v_5(N(20)) = {v5}")
print(f"k = {k}")
print(f"k mod 10^5 = {k % 100000}")

# Verification for small n
print("\nVerification for small n:")
for n in range(1, 8):
    N = 1
    for j in range(n):
        m = 2**j
        cat = comb(2*m, m) // (m+1)
        exp = 2**(n-1-j)
        N *= cat**exp
    # Count trailing zeros
    tz = 0
    temp = N
    while temp > 0 and temp % 10 == 0:
        tz += 1
        temp //= 10
    print(f"  N({n}) = {N} (trailing zeros: {tz})")

    v2_check = compute_vp_N(n, 2)
    v5_check = compute_vp_N(n, 5)
    k_check = min(v2_check, v5_check)
    print(f"    v2={v2_check}, v5={v5_check}, k={k_check}")
    assert k_check == tz, f"Mismatch! k_check={k_check} but trailing zeros={tz}"

print("\nAll verifications passed!")
