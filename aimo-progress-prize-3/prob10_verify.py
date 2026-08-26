from fractions import Fraction
import math

# Verify each g(c) value by checking the divisibility conditions.

def pow3_mod_s(s):
    """Compute 3^L mod s where L = 2025!"""
    if s == 1:
        return 0
    factors = {}
    temp = s
    for p in range(2, int(temp**0.5) + 2):
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = 1
    remainders = []
    moduli = []
    for p, a in factors.items():
        pa = p ** a
        if p == 3:
            remainders.append(0)
        else:
            # 3^L = 1 mod p^a when ord_{p^a}(3) | L = 2025!
            # For p <= 2025 and small a: ord divides phi(p^a) = p^{a-1}(p-1)
            # phi(p^a) divides 2025! since p^{a-1} <= p^a <= s < 1000 < 2025
            # and p-1 < 2025.
            remainders.append(1)
        moduli.append(pa)
    if not moduli:
        return 0
    result = 0
    M_total = 1
    for m in moduli:
        M_total *= m
    for i in range(len(moduli)):
        Mi = M_total // moduli[i]
        x = pow(Mi, -1, moduli[i])
        result += remainders[i] * Mi * x
    return result % M_total

# Verify: for c=1848374, sum=25 should divide 3^L + 1848374
c = 1848374
s = 25
r = pow3_mod_s(s)
print(f"3^L mod 25 = {r}")
print(f"(3^L + {c}) mod 25 = {(r + c) % s}")
# 25 = 5^2. 3^L mod 5 = 1 (ord_5(3)=4, 4|L). 3^L mod 25: ord_25(3).
# 3^4 = 81 = 6 mod 25. 3^20 = 6^5 mod 25 = 7776 mod 25 = 1. ord_25(3) = 20.
# 20 | L = 2025!. So 3^L = 1 mod 25.
# c = 1848374 mod 25 = ?
print(f"1848374 mod 25 = {1848374 % 25}")
# So (1 + 24) mod 25 = 0. YES!

print()
c = 10162574
s = 47
r = pow3_mod_s(s)
print(f"3^L mod 47 = {r}")
print(f"(3^L + {c}) mod 47 = {(r + c) % s}")
print(f"c mod 47 = {c % s}")

print()
c = 265710644
s = 97
r = pow3_mod_s(s)
print(f"3^L mod 97 = {r}")
print(f"(3^L + {c}) mod 97 = {(r + c) % s}")
print(f"c mod 97 = {c % s}")

print()
c = 44636594
s = 167
r = pow3_mod_s(s)
print(f"3^L mod 167 = {r}")
print(f"(3^L + {c}) mod 167 = {(r + c) % s}")
print(f"c mod 167 = {c % s}")

# Now verify that there's no better ratio available by checking even more sums.
# The best possible ratios are:
# 6/11 ≈ 0.545 (needs c = 10 mod 11)
# 4/7 ≈ 0.571 (needs c = 6 mod 7)
# 10/17 ≈ 0.588 (needs c = 16 mod 17)
# 3/5 = 0.600 (needs c = 9 mod 10)
# 14/23 ≈ 0.609 (needs c = 22 mod 23)
# 8/13 ≈ 0.615 (needs c = 12 mod 13)
# 18/29 ≈ 0.621 (needs c = 28 mod 29)
# 5/8 = 0.625 (needs c = 15 mod 16)
# 22/35 ≈ 0.629 (needs c = 34 mod 35 or ...)
# 12/19 ≈ 0.632 (needs c = 18 mod 19)
# 26/41 ≈ 0.634 (needs c = 40 mod 41)
# 30/47 ≈ 0.638 (needs c = 46 mod 47)
# 16/25 = 0.640 (needs c = 24 mod 25)

# For these modular conditions, 3^L = 1 mod p for all the primes involved.
# So the condition simplifies to: c + 1 = 0 mod s (i.e., c = s-1 mod s) when s is prime.
# But many "sums" are composite and the condition is more nuanced.

# Let me list the exact conditions needed for each of the best ratios:
# Ratio 6/11: sum 11 (prime). Need 11 | (3^L + c). 3^L = 1 mod 11. Need c = 10 mod 11.
# Ratio 4/7: sum 7 (prime). Need 7 | (3^L + c). 3^L = 1 mod 7. Need c = 6 mod 7.
# Ratio 10/17: sum 17 (prime). 3^L = 1 mod 17. Need c = 16 mod 17.
# Ratio 3/5: sum 10 = 2*5. 3^L mod 10 = 1 (3^L = 1 mod 2, 3^L = 1 mod 5, CRT: 1 mod 10). Need c = 9 mod 10.
# Ratio 14/23: sum 23 (prime). 3^L = 1 mod 23. Need c = 22 mod 23.
# Ratio 8/13: sum 13 (prime). 3^L = 1 mod 13. Need c = 12 mod 13.
# Ratio 18/29: sum 29 (prime). 3^L = 1 mod 29. Need c = 28 mod 29.
# Ratio 5/8: sum 16 = 2^4. 3^L mod 16. 3^L mod 2 = 1. 3^L mod 16: ord_16(3) = 4. 4|L.
#   3^4 = 81 = 1 mod 16. So 3^L = 1 mod 16. Need c = 15 mod 16.
# Ratio 22/35: sum 35 = 5*7. 3^L = 1 mod 5, 3^L = 1 mod 7. 3^L = 1 mod 35. Need c = 34 mod 35.
# Ratio 12/19: sum 19 (prime). 3^L = 1 mod 19. Need c = 18 mod 19.
# Ratio 26/41: sum 41 (prime). 3^L = 1 mod 41. Need c = 40 mod 41.
# Ratio 30/47: sum 47 (prime). 3^L = 1 mod 47. Need c = 46 mod 47.
# Ratio 16/25: sum 25 = 5^2. 3^L mod 25 = 1. Need c = 24 mod 25.

print("\nDetailed check for each c value:")
for c_val in [1848374, 10162574, 265710644, 44636594]:
    print(f"\nc = {c_val}:")
    checks = [
        (Fraction(6,11), 11, 10, "6/11"),
        (Fraction(4,7), 7, 6, "4/7"),
        (Fraction(10,17), 17, 16, "10/17"),
        (Fraction(3,5), 10, 9, "3/5"),
        (Fraction(14,23), 23, 22, "14/23"),
        (Fraction(8,13), 13, 12, "8/13"),
        (Fraction(18,29), 29, 28, "18/29"),
        (Fraction(5,8), 16, 15, "5/8"),
        (Fraction(22,35), 35, 34, "22/35"),
        (Fraction(12,19), 19, 18, "12/19"),
        (Fraction(26,41), 41, 40, "26/41"),
        (Fraction(30,47), 47, 46, "30/47"),
        (Fraction(16,25), 25, 24, "16/25"),
        (Fraction(34,53), 53, 52, "34/53"),
        (Fraction(38,59), 59, 58, "38/59"),
        (Fraction(42,65), 65, 64, "42/65"),
        (Fraction(46,71), 71, 70, "46/71"),
        (Fraction(50,77), 77, 76, "50/77"),
        (Fraction(54,83), 83, 82, "54/83"),
        (Fraction(58,89), 89, 88, "58/89"),
        (Fraction(62,95), 95, 94, "62/95"),
        (Fraction(64,97), 97, 96, "64/97"),
        (Fraction(66,101), 101, 100, "66/101"),
        (Fraction(70,107), 107, 106, "70/107"),
        (Fraction(74,113), 113, 112, "74/113"),
        (Fraction(110,167), 167, 166, "110/167"),
    ]
    best = Fraction(10**9)
    for ratio, s, need_c_mod_s, name in checks:
        if c_val % s == need_c_mod_s:
            if ratio < best:
                best = ratio
                print(f"  {name}: c mod {s} = {c_val % s} = {need_c_mod_s}. MATCH! (ratio = {float(ratio):.6f})")
    print(f"  Best: {best}")

# Now let me also check if there are OTHER families not of the form (2k+1) that might apply.
# Key families:
# (1, 2k, 4k) sum = 6k+1, lcm = 4k, ratio = 4k/(6k+1)
# k=1: (1,2,4) sum=7, 4/7
# k=2: (1,4,8) sum=13, 8/13
# k=4: (1,8,16) sum=25, 16/25
# k=8: (1,16,32) sum=49, 32/49
# k=16: (1,32,64) sum=97, 64/97
# k=32: (1,64,128) sum=193, 128/193

# (2, 2k+1, 2(2k+1)) sum = 4k+5, lcm = 2(2k+1), ratio = 2(2k+1)/(4k+5)
# k=1: (2,3,6) sum=11, 6/11
# k=2: (2,5,10) sum=17, 10/17
# k=3: (2,7,14) sum=23, 14/23
# k=4: (2,9,18) sum=29, 18/29
# ...
# k=27: (2,55,110) sum=167, 110/167

# (1, k, 2k) sum = 3k+1, lcm = 2k, ratio = 2k/(3k+1)
# k=2: (1,2,4) sum=7, 4/7
# k=3: (1,3,6) sum=10, 6/10=3/5
# k=4: (1,4,8) sum=13, 8/13
# etc.

# For c=265710644: need c=96 mod 97.
print(f"\n265710644 mod 97 = {265710644 % 97}")
# And for (2, 2k+1, 2(2k+1)) with k such that 4k+5 = 97: k = 23.
# Triple (2, 47, 94). Sum = 143, not 97. Hmm.
# Actually (1,32,64) sum=97, lcm=64, ratio=64/97.
# What about (2, 31, 62)? Sum = 95, not 97.
# (1, 32, 64): ratio 64/97 ≈ 0.6598.
# Is there a triple with sum 97 giving better ratio?

# Actually I realize there might be better triples for certain sums.
# Let me double-check:
# For sum = 97: best triple?
# (2, 3, 6)*k: sum = 11k. 11*8 = 88, 11*9 = 99. Not 97.
# (1, 2, 4)*k: sum = 7k. 7*13 = 91, 7*14 = 98. Not 97.
# 97 is prime, so we can't use multiples of smaller sums.
# Best triple with sum 97 and coprime elements:
# (1, 32, 64): lcm = 64. ratio = 64/97.
# (2, 31, 64): sum = 97. lcm = lcm(2,31,64) = 64*31 = 1984. ratio = 1984/97 ≈ 20. Bad.
# (1, 2, 94): lcm = 94. ratio ≈ 0.97. Bad.
# Hmm, 64/97 is indeed the best for sum 97.

# But could sum = some number other than 97 divide n = 3^L + 265710644 and give a better ratio?
# We checked sums up to 1000 and 64/97 was the best.

# Let me extend the search for 265710644 specifically.
c_val = 265710644
best_ratio = Fraction(10**9)
best_for_sum = {}
for s in range(3, 2000):
    best = None
    for p in range(1, s//3 + 1):
        for q in range(p+1, (s-p)//2 + 1):
            r = s - p - q
            if r <= q:
                continue
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if best is None or ratio < best[0]:
                best = (ratio, (p, q, r), l)
    if best and best[0] < best_ratio:
        r3L = pow3_mod_s(s)
        if (r3L + c_val) % s == 0:
            if best[0] < best_ratio:
                best_ratio = best[0]
                print(f"c=265710644: New best at sum={s}, triple={best[1]}, ratio={best[0]} = {float(best[0]):.8f}")

print(f"\nBest for c=265710644: {best_ratio}")
