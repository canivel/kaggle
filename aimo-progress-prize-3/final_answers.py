"""
Final answers for both problems.
"""
from math import comb
from decimal import Decimal, getcontext

# =====================
# PROBLEM 5
# =====================
# Tournament with 2^20 runners, Swiss-system.
# N = product_{j=0}^{19} C_{2^j}^{2^{19-j}}
# where C_m = (2m)! / ((m+1)! * m!) is the m-th Catalan number.
# k = v_{10}(N) = min(v_2(N), v_5(N))

def vp_factorial(n, p):
    """v_p(n!) using Legendre's formula"""
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def vp_catalan(m, p):
    """v_p(C_m) where C_m = (2m)!/((m+1)!*m!)"""
    return vp_factorial(2*m, p) - vp_factorial(m+1, p) - vp_factorial(m, p)

def compute_vp_N(n, p):
    total = 0
    for j in range(n):
        m = 2**j
        exponent = 2**(n-1-j)
        vc = vp_catalan(m, p)
        total += exponent * vc
    return total

v2 = compute_vp_N(20, 2)
v5 = compute_vp_N(20, 5)
k5 = min(v2, v5)
ans5 = k5 % 100000
print(f"Problem 5:")
print(f"  v_2(N) = {v2}")
print(f"  v_5(N) = {v5}")
print(f"  k = min(v2, v5) = {k5}")
print(f"  k mod 10^5 = {ans5}")

# =====================
# PROBLEM 6
# =====================
# f(m) = ceil(log2(m)) for m >= 2.
# M = f(10^100000) = ceil(100000 * log2(10))
# log2(10) = 3.32192809488736...
# 100000 * log2(10) = 332192.809...
# ceil = 332193

getcontext().prec = 50
log10_2 = Decimal('0.30102999566398119521373889472449302676818596588058635042586592614252245205728026862065462830968066440')

# Verify: 2^332192 < 10^100000 < 2^332193
v_low = 332192 * log10_2
v_high = 332193 * log10_2
assert v_low < 100000 < v_high, f"Range check failed: {v_low} < 100000 < {v_high}"

M = 332193
ans6 = M % 100000
print(f"\nProblem 6:")
print(f"  M = ceil(log2(10^100000)) = {M}")
print(f"  M mod 10^5 = {ans6}")

print(f"\n=== FINAL ANSWERS ===")
print(f"Problem 5 (id=424e18): {ans5}")
print(f"Problem 6 (id=42d360): {ans6}")
