from fractions import Fraction
import math

# Key families of ratios:
# (2, 2k+1, 2(2k+1)): sum = 4k+5, lcm = 2(2k+1), ratio = 2(2k+1)/(4k+5)
# k=1: (2,3,6) sum=11, ratio=6/11
# k=2: (2,5,10) sum=17, ratio=10/17
# k=3: (2,7,14) sum=23, ratio=14/23
# etc.
#
# (1, k, 2k): sum = 3k+1, lcm = 2k, ratio = 2k/(3k+1)
# k=2: (1,2,4) sum=7, ratio=4/7
# k=3: (1,3,6) sum=10, ratio=6/10=3/5
# k=4: (1,4,8) sum=13, ratio=8/13
# etc.
#
# For n = M + c where M = 3^L (L = 2025!):
# We need n to be divisible by the sum of the ratio.
# n = 3^L + c. For the triple (2a, 3a, 6a): need 11 | n, i.e., 11 | (3^L + c).
# 3^L mod 11: ord_11(3) = 5. L = 2025!. 5 | L. So 3^L = 1 mod 11.
# Need 1 + c = 0 mod 11, i.e., c = 10 mod 11.
#
# For triple (a, 2a, 4a): need 7 | n, i.e., 7 | (3^L + c).
# 3^L mod 7: ord_7(3) = 6. 6 | L. 3^L = 1 mod 7. Need c = -1 = 6 mod 7.
#
# For triple (a, 2a, 6a): need 9 | n. 3^L mod 9 = 0. Need c = 0 mod 9.
#
# For triple (2a, 5a, 10a): need 17 | n. 3^L mod 17: ord_17(3) = 16. 16 | L? L = 2025!. 16 | 2025!. Yes.
# 3^L = 1 mod 17. Need c = -1 = 16 mod 17.
#
# So for each triple (p, q, r) with sum s, we need s | n, i.e., s | (3^L + c), i.e., c = -3^L mod s.
# Since 3^L mod s depends on ord_s(3) and whether ord_s(3) | L.
# For s < 2025: all prime factors of s are < 2025, so ord_p(3) | (p-1) < 2025 for each prime p | s.
# By CRT, ord_s(3) | lcm of (p-1) for primes p | s. Since all (p-1) | L = 2025! (for primes <= 2025):
# 3^L = 1 mod p for all primes p | s (p != 3). And 3^L = 0 mod 3^{v_3(s)}.
# So 3^L mod s is determined by CRT.

# For s coprime to 3: 3^L = 1 mod s. So need c = -1 mod s.
# For s = 3^a * t with gcd(t,3)=1: 3^L = 0 mod 3^a (if a <= L, which is true for any reasonable a).
# So 3^L mod s: by CRT, 3^L = 0 mod 3^a and 3^L = 1 mod t. So 3^L = s/t * (t^{-1} mod 3^a) * ...
# Actually: 3^L mod s. s = 3^a * t, gcd(t,3)=1. 3^L mod 3^a = 0. 3^L mod t = 1.
# By CRT: 3^L = 3^a * u where u is chosen so 3^a * u = 1 mod t.
# Hmm, this gets complicated. Let me just compute.

# For our specific c values, let me check which "good" triples are available.
# For each c, find the best available triple.

# Good triple families:
# Family 1: (2, 3, 6) -> sum 11, lcm 6, ratio 6/11. Need c = -1 mod 11 (since 3^L=1 mod 11).
# Wait, need 11 | (3^L + c). 3^L = 1 mod 11. So need c = -1 = 10 mod 11.
# Family 2: (1, 2, 4) -> sum 7, lcm 4, ratio 4/7. Need c = -1 = 6 mod 7.
# Family 3: (1, 2, 6) -> sum 9, lcm 6, ratio 2/3. Need 9 | (3^L + c). 3^L mod 9 = 0. c = 0 mod 9.
# Family 4: (2, 5, 10) -> sum 17, lcm 10, ratio 10/17. Need c = -1 = 16 mod 17.
# Family 5: (1, 3, 6) -> sum 10, lcm 6, ratio 3/5. Need 10 | n. 3^L mod 10: 3^L mod 2 = 1, 3^L mod 5 = 1.
#   3^L mod 10 = 1 (by CRT). Need c = -1 = 9 mod 10.
#   Wait but gcd(1,3,6) = 1. lcm(1,3,6) = 6. ratio = 6/10 = 3/5.
# Family 6: (1, 4, 8) -> sum 13, lcm 8, ratio 8/13. Need c = -1 = 12 mod 13.
# Family 7: (3, 4, 8) -> sum 15, lcm 24, ratio 24/15. Too large.
# Family 8: (2, 7, 14) -> sum 23, lcm 14, ratio 14/23. Need c = -1 = 22 mod 23.

# Also non-"c=-1" types. For sum divisible by 3:
# (1, 2, 6): sum 9. 3^L + c mod 9 = 0 + c mod 9 = c mod 9. Need c = 0 mod 9.
# (3, 5, 10): sum 18. 3^L mod 18: 3^L mod 2 = 1, 3^L mod 9 = 0. CRT: 3^L = 9 mod 18.
#   Need c = -9 = 9 mod 18.
# (3, 4, 6): sum 13. lcm = 12. ratio = 12/13. Need c = 12 mod 13.

# Wait, I should be more careful. Let me compute 3^L mod s for each relevant s.
# For s with gcd(s, 3) = 1: 3^L = 1 mod s.
# For s = 9k' with gcd(k',3)=1: 3^L = 0 mod 9, 3^L = 1 mod k'. CRT.
# For s = 3k' with gcd(k',3)=1 and 9 does not divide s: 3^L = 0 mod 3, 3^L = 1 mod k'. CRT.

# Let me just check for each c value which triples apply.
c_values = [0, 1848374, 10162574, 265710644, 44636594]

# The "best" triples sorted by ratio:
triples = [
    ((2, 3, 6), 11, Fraction(6, 11)),
    ((1, 2, 4), 7, Fraction(4, 7)),
    ((2, 5, 10), 17, Fraction(10, 17)),
    ((1, 3, 6), 10, Fraction(3, 5)),
    ((2, 7, 14), 23, Fraction(14, 23)),
    ((1, 4, 8), 13, Fraction(8, 13)),
    ((2, 9, 18), 29, Fraction(18, 29)),
    ((1, 5, 10), 16, Fraction(5, 8)),
    ((2, 11, 22), 35, Fraction(22, 35)),
    ((1, 6, 12), 19, Fraction(12, 19)),
    ((1, 2, 6), 9, Fraction(6, 9)),  # = 2/3
    ((2, 3, 4), 9, Fraction(12, 9)),  # lcm = 12, ratio = 4/3. Bad.
    ((3, 4, 6), 13, Fraction(12, 13)),  # Bad.
]

# Actually let me recompute properly.
# For triple (p, q, r): ratio = lcm(p,q,r) / (p+q+r).
# We want this ratio to be small.

all_triples = []
for p in range(1, 40):
    for q in range(p+1, 80):
        for r in range(q+1, 150):
            s = p + q + r
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if ratio < Fraction(2, 3) + Fraction(1, 100):
                all_triples.append((ratio, (p, q, r), s, l))

all_triples.sort()
# Keep only unique sums
seen = set()
best_by_sum = {}
for ratio, triple, s, l in all_triples:
    if s not in best_by_sum or ratio < best_by_sum[s][0]:
        best_by_sum[s] = (ratio, triple, l)

print("Best ratio for each sum:")
for s in sorted(best_by_sum.keys()):
    ratio, triple, l = best_by_sum[s]
    if ratio <= Fraction(2, 3):
        print(f"  sum={s}: triple={triple}, lcm={l}, ratio={ratio} = {float(ratio):.6f}")

# Now for each c value, find the best applicable triple.
# 3^L mod s:
def pow3_mod_s(s, L_factorial=True):
    """Compute 3^L mod s where L = 2025!"""
    # Factor s
    if s == 1:
        return 0
    result_mod = {}
    temp = s
    for p in range(2, s+1):
        if p * p > temp:
            break
        if temp % p == 0:
            a = 0
            while temp % p == 0:
                a += 1
                temp //= p
            result_mod[p] = a
    if temp > 1:
        result_mod[temp] = 1

    # 3^L mod p^a for each prime power
    remainders = []
    moduli = []
    for p, a in result_mod.items():
        pa = p ** a
        if p == 3:
            # 3^L mod 3^a = 0 (since L >= a for any reasonable a)
            remainders.append(0)
        else:
            # ord_{p^a}(3) divides phi(p^a) = p^{a-1}(p-1)
            # For L = 2025!, all primes up to 2025 divide L, so
            # phi(p^a) | L (if p < 2025 and a is small)
            # Actually we need ord_{p^a}(3) | L.
            # ord_{p^a}(3) | phi(p^a) = p^{a-1}*(p-1).
            # For p <= 2025: p-1 <= 2024. p^{a-1} divides 2025! if p^{a-1} <= 2025.
            # And (p-1) | 2025!. So phi(p^a) | 2025! if p^{a-1} <= 2025.
            # For our sums s < 150, p^a < 150, so a is small, p^{a-1} < 150 < 2025.
            # So 3^L = 1 mod p^a.
            remainders.append(1)
        moduli.append(pa)

    # CRT
    result = 0
    M_total = 1
    for m in moduli:
        M_total *= m
    for i in range(len(moduli)):
        Mi = M_total // moduli[i]
        # Mi * x = 1 mod moduli[i]
        x = pow(Mi, -1, moduli[i])
        result += remainders[i] * Mi * x
    return result % M_total

print("\n\nFor each c value, best applicable triple:")
for c_val in c_values:
    print(f"\nc = {c_val}:")
    best_ratio = Fraction(10**9)
    best_info = None
    for s in sorted(best_by_sum.keys()):
        ratio, triple, l = best_by_sum[s]
        # Check if s | (3^L + c)
        r3L = pow3_mod_s(s)
        needed_c = (-r3L) % s
        if c_val % s == needed_c:
            if ratio < best_ratio:
                best_ratio = ratio
                best_info = (s, triple, l, ratio)
                print(f"  sum={s}: triple={triple}, lcm={l}, ratio={ratio} = {float(ratio):.6f} <-- APPLICABLE")
                if ratio <= Fraction(6, 11):
                    break  # Can't do better than 6/11
    if best_info:
        print(f"  BEST: sum={best_info[0]}, triple={best_info[1]}, ratio={best_info[3]}")
