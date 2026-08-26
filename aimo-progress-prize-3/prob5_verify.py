"""Verify Problem 5 computations."""

def legendre(n, p):
    """Compute v_p(n!)"""
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def v_catalan(m, p):
    """Compute v_p(C_m) where C_m = (2m)! / (m! * (m+1)!)"""
    return legendre(2*m, p) - legendre(m, p) - legendre(m+1, p)

# Verify: C_2 = 2. v_2(2) = 1. ✓
# C_4 = 14 = 2*7. v_2(14) = 1. ✓
# C_8 = 1430 = 2*5*11*13. v_2 = 1, v_5 = 1. ✓
print("Direct verification:")
catalan_vals = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]
for i, c in enumerate(catalan_vals):
    if c > 1:
        v2 = 0
        temp = c
        while temp % 2 == 0:
            v2 += 1
            temp //= 2
        v5 = 0
        temp = c
        while temp % 5 == 0:
            v5 += 1
            temp //= 5
        print(f"  C_{i} = {c}: v_2={v2}, v_5={v5}")

# Verify for C_16
from math import comb
m = 16
c16 = comb(32, 16) // 17
print(f"\nC_16 = {c16}")
v2 = 0
temp = c16
while temp % 2 == 0:
    v2 += 1
    temp //= 2
v5 = 0
temp = c16
while temp % 5 == 0:
    v5 += 1
    temp //= 5
print(f"  v_2 = {v2}, v_5 = {v5}")
print(f"  Formula: v_2 = {v_catalan(16, 2)}, v_5 = {v_catalan(16, 5)}")

# Verify for C_32
m = 32
c32 = comb(64, 32) // 33
print(f"\nC_32 = {c32}")
v2 = 0
temp = c32
while temp % 2 == 0:
    v2 += 1
    temp //= 2
v5 = 0
temp = c32
while temp % 5 == 0:
    v5 += 1
    temp //= 5
print(f"  v_2 = {v2}, v_5 = {v5}")
print(f"  Formula: v_2 = {v_catalan(32, 2)}, v_5 = {v_catalan(32, 5)}")

# Now verify the claim that v_2(C_{2^j}) = 1 for j >= 1
# C_{2^j} = (2^{j+1})! / ((2^j)! * (2^j + 1)!)
# v_2(C_{2^j}) = v_2((2^{j+1})!) - v_2((2^j)!) - v_2((2^j+1)!)
# v_2(n!) = n - s_2(n) where s_2(n) is the digit sum of n in base 2
# (Legendre's formula)
# Actually v_2(n!) = (n - s_2(n)) / (2-1) = n - s_2(n)

# For n = 2^{j+1}: s_2 = 1, so v_2 = 2^{j+1} - 1
# For n = 2^j: s_2 = 1, so v_2 = 2^j - 1
# For n = 2^j + 1: s_2 = 2 (binary is 100...01), so v_2 = 2^j + 1 - 2 = 2^j - 1

# v_2(C_{2^j}) = (2^{j+1}-1) - (2^j-1) - (2^j-1) = 2^{j+1}-1-2^j+1-2^j+1 = 1

print("\n\nProof that v_2(C_{2^j}) = 1 for j >= 1:")
print("v_2(n!) = n - s_2(n)")
print("v_2((2^{j+1})!) = 2^{j+1} - 1")
print("v_2((2^j)!) = 2^j - 1")
print("v_2((2^j+1)!) = 2^j + 1 - 2 = 2^j - 1")
print("v_2(C_{2^j}) = (2^{j+1}-1) - (2^j-1) - (2^j-1) = 1")

# So v_2(N(20)) = sum_{j=0}^{19} 2^{19-j} * v_2(C_{2^j})
# = sum_{j=1}^{19} 2^{19-j} * 1 + 2^19 * 0 (j=0: C_1 has v_2=0)
# = sum_{j=1}^{19} 2^{19-j} = 2^18 + 2^17 + ... + 2^0 = 2^19 - 1 = 524287
print(f"\nv_2(N(20)) = 2^19 - 1 = {2**19 - 1}")

# For v_5, I need to compute v_5(C_{2^j}) for j=0,...,19
# v_5(n!) = (n - s_5(n)) / 4 where s_5(n) is digit sum in base 5

# For n = 2^j, we need s_5(2^j).
# Let me compute these carefully.

print("\n\nv_5 computations:")
for j in range(21):
    m = 2**j
    # C_m = (2m)! / (m! * (m+1)!)
    v5 = legendre(2*m, 5) - legendre(m, 5) - legendre(m+1, 5)
    print(f"j={j}: m=2^{j}={m}, v_5(C_m) = {v5}")

# v_5(N(20)) = sum_{j=0}^{19} 2^{19-j} * v_5(C_{2^j})
total_v5 = 0
for j in range(20):
    m = 2**j
    v5 = legendre(2*m, 5) - legendre(m, 5) - legendre(m+1, 5)
    contrib = (2**(19-j)) * v5
    total_v5 += contrib
    if v5 > 0:
        print(f"  j={j}: 2^{19-j}={2**(19-j)} * {v5} = {contrib}")

print(f"\nTotal v_5(N(20)) = {total_v5}")
print(f"v_2(N(20)) = {2**19 - 1}")
print(f"k = min(v_2, v_5) = {min(2**19 - 1, total_v5)}")
print(f"k mod 10^5 = {min(2**19 - 1, total_v5) % 100000}")
