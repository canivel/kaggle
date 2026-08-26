from fractions import Fraction
import math

# Summary of findings so far:
# For c = 0: g(0) = 2/3
# For c = 4M: g(4M) = 10/3
# For the fixed c values, we found applicable triples but we're not sure they're optimal.
# The key issue: we only checked triples of the form (pa, qa, ra) where p+q+r is the sum.
# There might be better triples that don't follow this pattern.

# Actually, the approach f(n) = min lcm(a,b,c) over a<b<c with a+b+c=n is correct.
# And the best "ratio triples" (p,q,r) with small lcm/sum give UPPER BOUNDS.
# But for specific n, there might be ad-hoc triples doing better.

# However, for n = 3^L + c where L = 2025! and c is fixed:
# The "ratio triple" approach requires sum | n. This is very restrictive.
# But there are infinitely many sums to check (we only checked up to ~250).

# For the (1, d, n-1-d) family: we showed that this gives f = 10(n-1)/11 for c=1848374.
# But we found a ratio triple giving 16n/25 which is much better.
# 16/25 = 0.64 vs 10/11 = 0.909. So ratio triples are way better.

# But are the ratio triples truly optimal? They could be beaten by non-ratio triples.
# However, for very large n, ratio triples are likely optimal since the "error term"
# from n not being exactly sum*a vanishes.

# Wait, but for n = 3^L + c, the ratio triple (p,q,r) with sum s requires s | n.
# For this to work: s | (3^L + c). With 3^L mod s known, this means c ≡ -3^L (mod s).
# For each c, only certain sums s work.

# The question is: what is the BEST (smallest) ratio achievable for each c?
# We need to search over a wider range of sums.

# Actually, let me reconsider. We need to be more careful. For the triple (pa, qa, ra)
# with a = n/s: we need a to be a positive integer. If s | n, then a = n/s.
# But we also need pa < qa < ra, i.e., p < q < r. And all must be distinct positive integers.
# And lcm(pa, qa, ra) = a * lcm(p,q,r).
# The question is: can we find non-homogeneous triples that do better?

# For n = 3^L + c: consider (a, b, n-a-b) where a and b are chosen freely.
# lcm(a, b, n-a-b). For this to be small, we need a, b, n-a-b to have large common factors.
# If gcd(a,b) = g and g | (n-a-b) too, then all three are multiples of g.
# n-a-b = n - a - b. g | a and g | b implies g | (a+b). g | (n-a-b) iff g | n.
# So if g | n, all three are multiples of g.
# For n = 3^L + c with c not divisible by 3: gcd constraints are limited.

# The bottom line: for each c, we search for the best sum s dividing n = 3^L + c,
# and for each such s, use the best triple (p,q,r) with p+q+r = s.

# But we also need to verify that the best sum isn't very large.
# For a sum s, the best ratio is at most 6/11 ≈ 0.545 (using (2,3,6) when 11|s).
# The question: is there always a small s dividing n = 3^L + c with good ratio?

# For n = 3^L + c: the divisors of n include small primes dividing n.
# n mod 2: 3^L is odd. If c is even, n is odd. If c is odd, n is even.
# All our c values (1848374, 10162574, 265710644, 44636594) are even!
# And M + c = 3^L + c. 3^L is odd. Even c -> odd n. So 2 does NOT divide n.
# Wait: 3^L is odd, c is even, n = odd + even = odd. So n is odd.
# For n odd: 2 does not divide n. So sum s must be odd? No, s divides n, and n is odd, so s is odd.
# All the "good" triples with sum 7, 9, 11... are odd sums. That's fine.

# Let me search more systematically for the best achievable ratio for each c.

def pow3_mod_s(s):
    """Compute 3^L mod s where L = 2025!"""
    if s == 1:
        return 0
    # Factor s
    factors = {}
    temp = s
    for p in range(2, int(temp**0.5) + 2):
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = 1

    # CRT
    remainders = []
    moduli = []
    for p, a in factors.items():
        pa = p ** a
        if p == 3:
            remainders.append(0)
        else:
            # 3^L = 1 mod p^a (assuming ord divides L = 2025!)
            # This is true for p <= 2025 and small a.
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

# Precompute best triples for each sum
best_for_sum = {}
for s in range(3, 500):
    best = None
    for p in range(1, s):
        for q in range(p+1, s):
            r = s - p - q
            if r <= q:
                continue
            if math.gcd(p, math.gcd(q, r)) > 1:
                # Can reduce by common factor, but still valid
                pass
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if best is None or ratio < best[0]:
                best = (ratio, (p, q, r), l)
    if best:
        best_for_sum[s] = best

# Now for each c, find the best s dividing n = 3^L + c
c_values_fixed = [1848374, 10162574, 265710644, 44636594]

for c_val in c_values_fixed:
    print(f"\nc = {c_val}:")
    best_ratio = Fraction(10**9)
    best_info = None
    for s in range(3, 500):
        if s not in best_for_sum:
            continue
        ratio, triple, l = best_for_sum[s]
        if ratio >= best_ratio:
            continue
        # Check if s | (3^L + c)
        r3L = pow3_mod_s(s)
        if (r3L + c_val) % s == 0:
            best_ratio = ratio
            best_info = (s, triple, l, ratio)
    if best_info:
        print(f"  BEST: sum={best_info[0]}, triple={best_info[1]}, lcm={best_info[2]}, ratio={best_info[3]} = {float(best_info[3]):.8f}")

# Also handle c=0 and c=4M
print(f"\nc = 0:")
# n = 3^L. Divisible by 9, 27, 81, etc.
# Best with sum 9: ratio 2/3. But also sum 11 if 11 | 3^L? 3^L mod 11: 3^L = 1 mod 11 != 0. No.
# Sum 7: 7 | 3^L? 3^L mod 7 = 1. No.
# Sum 99: (18,27,54), sum=99, ratio=6/11. 99 | 3^L? 99 = 9*11. 9 | 3^L yes. 11 | 3^L? No.
# Sum 63: (9,18,36), sum=63, ratio=4/7. 63 = 9*7. 7 | 3^L? No.
# Looks like best is 2/3 from sum=9.
best_ratio_0 = Fraction(10**9)
for s in range(3, 500):
    if s not in best_for_sum:
        continue
    ratio, triple, l = best_for_sum[s]
    if ratio >= best_ratio_0:
        continue
    r3L = pow3_mod_s(s)
    # For c=0: check if s | (3^L + 0) = s | 3^L
    if r3L % s == 0:
        best_ratio_0 = ratio
        print(f"  sum={s}: triple={triple}, ratio={ratio} = {float(ratio):.8f}")

print(f"\nc = 4M:")
# n = 5M = 5*3^L.
# sum s must divide 5*3^L.
# s can be any factor of 5*3^L that is >= 3.
# For sum=9: 9 | 5*3^L? Yes (9 | 3^L for L >= 2). Ratio 2/3.
# For sum=45: 45 = 9*5. 45 | 5*3^L? Yes. Best triple with sum 45: ratio 2/3 (same).
# For sum=55: 55 = 5*11. 55 | 5*3^L? Need 11 | 3^L. 3^L mod 11 = 1 != 0. No.
# For sum=15: 15 = 3*5. 15 | 5*3^L? Yes. Best triple sum 15: (1,4,10)->lcm=20, ratio 4/3. Bad.
#   Actually best for sum 15: (1,6,8)->lcm=24, ratio 8/5. Or (2,4,9)->lcm=36. Hmm.
#   Actually (1,4,10): lcm(1,4,10)=20, ratio=20/15=4/3. (3,4,8): lcm=24, ratio=8/5.
#   (1,5,9): lcm=45, ratio=3. (2,5,8): lcm=40, 8/3. (1,6,8): lcm=24, 8/5.
#   (2,4,9): lcm=36, 12/5. Hmm none below 2/3 for sum=15.
#   (3,5,7): lcm=105, ratio 7. (1,2,12): lcm=12, ratio=4/5! 12/15=4/5. Still > 2/3.
#   (1,3,11): lcm=33, ratio 11/5. (2,3,10): lcm=30, ratio 2.
#   (5,4,6): lcm=60, ratio=4. (2,6,7): lcm=42. ratio=42/15.
#   (1,4,10): best is lcm 20, ratio 4/3. All > 2/3.
# So for c=4M, g(4M) = 10/3 as previously determined (from f(5M) = 2*5M/3).
# Actually wait, f(5M)/M = 10/3 from the data, and g(4M) = (1/L)*floor(L*10/3) = 10/3.
best_ratio_4M = Fraction(10**9)
for s in range(3, 500):
    if s not in best_for_sum:
        continue
    ratio, triple, l = best_for_sum[s]
    if ratio >= best_ratio_4M:
        continue
    # Check if s | 5*3^L. Need s | 5*3^L.
    # s must only have prime factors 3 and 5.
    # Check: if s has prime factor p other than 3 and 5: p | 5*3^L? Only if p=3 or p=5.
    # So s = 3^a * 5^b with a,b >= 0.
    temp = s
    while temp % 3 == 0:
        temp //= 3
    while temp % 5 == 0:
        temp //= 5
    if temp != 1:
        continue
    best_ratio_4M = ratio
    print(f"  sum={s}: triple={triple}, ratio={ratio} = {float(ratio):.8f}")

if best_ratio_4M == Fraction(10**9):
    print("  No triple with ratio < inf")

# Hmm, for n = 5*3^L, the sums must be of form 3^a * 5^b.
# Available sums: 3, 5, 9, 15, 25, 27, 45, 75, 81, 125, 135, 225, 243, 375, 405, ...
# For sum=9: ratio 2/3.
# For sum=45: let me check best triple.

for s in [3, 5, 9, 15, 25, 27, 45, 75, 81, 125, 135, 225, 243, 375]:
    if s in best_for_sum:
        ratio, triple, l = best_for_sum[s]
        print(f"  sum={s}: triple={triple}, lcm={l}, ratio={ratio} = {float(ratio):.8f}")
