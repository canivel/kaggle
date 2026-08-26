from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

def vp(n, p):
    """p-adic valuation of n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v

def factor(n):
    """Return dict of prime: exponent."""
    d = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            d[p] = d.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        d[n] = d.get(n, 0) + 1
    return d

def best_ratio_for_n(n_vals, label="", max_k1=2000):
    """Find best f(n)/n ratio.
    n_vals: dict prime -> valuation in n.
    """
    best = Fraction(1, 1)
    best_k = None
    for k1 in range(3, max_k1):
        for k2 in range(2, k1):
            S = k1*k2 + k1 + k2
            ratio = Fraction(k1*k2, S)
            if ratio >= best:
                continue
            # Check: for each prime p | S:
            # v_p(S) <= v_p(n) + v_p(k1) AND v_p(S) <= v_p(n) + v_p(k2)
            sf = factor(S)
            ok = True
            for p, vs in sf.items():
                vn = n_vals.get(p, 0)
                v1 = vp(k1, p)
                v2 = vp(k2, p)
                if vs > vn + v1 or vs > vn + v2:
                    ok = False
                    break
            if ok:
                best = ratio
                best_k = (k1, k2)
    print(f"  {label}: ratio={best}={float(best):.8f}, k={best_k}")
    return best

# c = 0: n = M = 3^K. Only prime: 3 with huge power.
print("=== c = 0 ===")
g0 = best_ratio_for_n({3: 10000}, "c=0", max_k1=500)
print(f"g(0) = {g0}")
print()

# c = 4M: n = 5M = 5*3^K. Primes: 3 (huge), 5 (power 1).
print("=== c = 4M ===")
r4M = best_ratio_for_n({3: 10000, 5: 1}, "c=4M", max_k1=500)
g4M = 5 * r4M
print(f"g(4M) = 5 * {r4M} = {g4M}")
print()

# c = 1848374: c+1 = 1848375 = 3^2 * 5^3 * 31 * 53
# v_3(n) = 0 (since c mod 3 = 2, M ≡ 0 mod 3, n = M + c ≡ c ≡ 2 mod 3)
# Wait: n = M + c. v_3(n): M = 3^K, c = 1848374. c mod 3 = 1848374 mod 3.
# 1848374 / 3 = 616124.666... so c mod 3 = 1848374 - 3*616124 = 1848374 - 1848372 = 2.
# So n = 3^K + c where c ≡ 2 mod 3. v_3(n) = v_3(3^K + c) = v_3(c) = 0 (since 3 does not divide c).
# Primes dividing n (for p != 3): n ≡ 1 + c mod p. So p | n iff p | (c+1) = 1848375.
# v_p(n) = v_p(c+1) for p != 3.
# c+1 = 1848375 = 3^2 * 5^3 * 31 * 53.
# For p=5: v_5(n) = v_5(1848375) = 3. For p=31: v_31(n) = 1. For p=53: v_53(n) = 1.
# For p=3: v_3(n) = 0. For p=2: v_2(n) = v_2(1848375) = 0 (1848375 is odd). So 2 does not divide n.
print("=== c = 1848374 ===")
g1 = best_ratio_for_n({5: 3, 31: 1, 53: 1}, "c=1848374")
print(f"g(1848374) = {g1}")
print()

# c = 10162574: c+1 = 10162575 = 3^2 * 5^2 * 31^2 * 47
# v_3(n) = 0 (c mod 3 = 10162574 mod 3 = ?)
print(f"10162574 mod 3 = {10162574 % 3}")
# v_5(n) = 2, v_31(n) = 2, v_47(n) = 1. Others: 0.
print("=== c = 10162574 ===")
g2 = best_ratio_for_n({5: 2, 31: 2, 47: 1}, "c=10162574")
print(f"g(10162574) = {g2}")
print()

# c = 265710644: c+1 = 265710645 = 3^3 * 5 * 97 * 103 * 197
print(f"265710644 mod 3 = {265710644 % 3}")
# v_5(n) = 1, v_97(n) = 1, v_103(n) = 1, v_197(n) = 1.
print("=== c = 265710644 ===")
g3 = best_ratio_for_n({5: 1, 97: 1, 103: 1, 197: 1}, "c=265710644")
print(f"g(265710644) = {g3}")
print()

# c = 44636594: c+1 = 44636595 = 3 * 5 * 103 * 167 * 173
print(f"44636594 mod 3 = {44636594 % 3}")
# v_5(n) = 1, v_103(n) = 1, v_167(n) = 1, v_173(n) = 1.
print("=== c = 44636594 ===")
g4 = best_ratio_for_n({5: 1, 103: 1, 167: 1, 173: 1}, "c=44636594")
print(f"g(44636594) = {g4}")
print()

# Verify c+1 factorizations
print("=== Verify factorizations ===")
for c in [1848374, 10162574, 265710644, 44636594]:
    c1 = c + 1
    f = factor(c1)
    print(f"c+1 = {c1} = {f}")
print()

# Compute final answer
print("=== FINAL COMPUTATION ===")
total = g0 + g4M + g1 + g2 + g3 + g4
print(f"g(0) = {g0}")
print(f"g(4M) = {g4M}")
print(f"g(1848374) = {g1}")
print(f"g(10162574) = {g2}")
print(f"g(265710644) = {g3}")
print(f"g(44636594) = {g4}")
print(f"Sum = {total} = {float(total):.10f}")
p = total.numerator
q = total.denominator
print(f"p/q = {p}/{q}")
print(f"gcd(p,q) = {gcd(p,q)}")
print(f"p + q = {p + q}")
print(f"(p + q) mod 99991 = {(p + q) % 99991}")
