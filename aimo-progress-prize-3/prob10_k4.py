from fractions import Fraction

# The validation for k=4, L=4 failed because L is too small.
# g(c) = (1/L) * floor(L * f(M+c)/M). When L=4, the floor operation loses precision.
# For L = 2025! (huge), the floor gives the exact ratio.
# Let me verify with larger L.

# k=8, M=3^8=6561, L=8
M = 6561
L = 8

def f_val(n, limit=None):
    if limit is None:
        limit = max(n*3, 100)
    for m in range(1, limit):
        divs = []
        for d in range(1, int(m**0.5)+1):
            if m % d == 0:
                divs.append(d)
                if d != m // d:
                    divs.append(m // d)
        divs.sort()
        nd = len(divs)
        for i in range(nd - 2):
            target = n - divs[i]
            lo, hi = i + 1, nd - 1
            while lo < hi:
                s = divs[lo] + divs[hi]
                if s == target:
                    return m
                elif s < target:
                    lo += 1
                else:
                    hi -= 1
    return None

for c_mult in [0, 1, 2, 3, 4]:
    c = c_mult * M
    n = M + c
    fv = f_val(n, 5*n)
    if fv is not None:
        g_exact = Fraction(1, L) * (L * fv // M)
        ratio_expected = Fraction(2 * (1 + c_mult), 3)
        print(f"c={c_mult}M: n={n}, f={fv}, f/M={Fraction(fv,M)}, g={g_exact}, expected={ratio_expected}, match={g_exact == ratio_expected}")

# Now check small c values for k=8:
# For c=24 (24 mod 25 = 24): n = 6561 + 24 = 6585. 6585/25 = 263.4, not integer!
# Wait: 6585 mod 25 = 6585 - 263*25 = 6585 - 6575 = 10. Not 0.
# For c=24: 3^8 = 6561. 6561 mod 25 = 6561 - 262*25 = 6561 - 6550 = 11.
# n mod 25 = 11 + 24 = 35 = 10 mod 25. Not 0!
# This is because 3^8 mod 25 != 1. ord_25(3) = 20. 8 does not divide 20. So 3^8 != 1 mod 25.

# This is exactly the issue: for small k (k=8, L=8), the modular arithmetic is different from L=2025!.
# For L = 2025!, ord_25(3) = 20 divides L = 2025!, so 3^L = 1 mod 25.
# For L = 8, ord_25(3) = 20 does NOT divide 8, so 3^8 != 1 mod 25.

# Let me try k where ord_25(3) | k. ord_25(3) = 20. So k must be multiple of 20.
# 3^20 mod 25: let me compute.
print(f"\n3^20 mod 25 = {pow(3, 20, 25)}")
# Use k=20: M = 3^20, L = 20.
# But 3^20 is huge. Too large for brute force f computation.

# Instead, let me verify the theory directly.
# For L = 2025! and n = 3^L + c:
# If s | (c+1) and gcd(s, 3) = 1: then s | (3^L + c) since 3^L = 1 mod s (when ord_s(3) | L).
# The triple (p, q, r) with p+q+r = s gives f(n) <= lcm(p,q,r) * n/s.
# f(n)/n <= lcm/s = ratio.
# And this is the BEST if no divisor d of n with d < s gives a better ratio.

# The factorizations of c+1:
# 1848374 + 1 = 1848375 = 3^2 * 5^3 * 31 * 53
# 10162574 + 1 = 10162575 = 3^2 * 5^2 * 31^2 * 47
# 265710644 + 1 = 265710645 = 3^3 * 5 * 97 * 103 * 197
# 44636594 + 1 = 44636595 = 3 * 5 * 103 * 167 * 173

# For s coprime to 3: s must divide c+1 (from the non-3 part).
# The non-3 parts:
# 1848375 / 9 = 205375 = 5^3 * 31 * 53
# Divisors of 205375 (coprime to 3): 1, 5, 25, 31, 53, 125, 155, 265, 775, 1325, 1643, 3875, 6625, 8215, 41075, 205375
# Relevant sums (>= 3): 5, 25, 31, 53, 125, 155, 265, ...
# But we also need s itself coprime to 3, OR s has factor 3 with appropriate handling.

# Actually: s doesn't need to be coprime to 3. We need s | n = 3^L + c.
# If 3 | s: write s = 3^a * t with gcd(t,3)=1. Then s | n means 3^a | n and t | n.
# 3^a | n = 3^L + c. If a <= L: 3^a | 3^L, so 3^a | c must hold too? No: 3^L + c = 0 mod 3^a.
# 3^L mod 3^a = 0 (since L >= a). So c mod 3^a = 0. Need 3^a | c.
# Also t | n: 3^L = 1 mod t (when ord_t(3) | L). And t | (1 + c). Wait: n = 3^L + c.
# t | n: t | (3^L + c) = t | (1 + c) (since 3^L = 1 mod t).

# So s | n iff: 3^{v_3(s)} | c AND (s/3^{v_3(s)}) | (c+1).
# For c coprime to 3 (c mod 3 = 2 for our values): 3^a | c only for a=0.
# So s must be coprime to 3 for s | n to hold (when c is not divisible by 3).

# Our c values all have c mod 3 = 2 (not divisible by 3).
# So s must be coprime to 3, and s | (c+1).

# Corrected: s | n iff s is coprime to 3 and s | (c+1).

# So the relevant divisors of c+1 (coprime to 3):
# 1848375 = 3^2 * 5^3 * 31 * 53
# Coprime-to-3 divisors: divisors of 5^3 * 31 * 53 = 205375

# 10162575 = 3^2 * 5^2 * 31^2 * 47
# Coprime-to-3 divisors: divisors of 5^2 * 31^2 * 47 = 1129175

# 265710645 = 3^3 * 5 * 97 * 103 * 197
# Coprime-to-3 divisors: divisors of 5 * 97 * 103 * 197 = 9845555

# 44636595 = 3 * 5 * 103 * 167 * 173
# Coprime-to-3 divisors: divisors of 5 * 103 * 167 * 173 = 14878865

import math

def get_divisors(n):
    divs = [1]
    temp = n
    factors = {}
    for p in range(2, int(temp**0.5)+2):
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = 1
    for p, a in factors.items():
        new_divs = []
        for d in divs:
            pk = 1
            for j in range(a+1):
                new_divs.append(d * pk)
                pk *= p
        divs = new_divs
    return sorted(divs)

# For each c value, find all coprime-to-3 divisors of c+1, then find best ratio
best_for_sum_cache = {}
def best_ratio_for_sum(s):
    if s in best_for_sum_cache:
        return best_for_sum_cache[s]
    best = None
    for p in range(1, s//3 + 1):
        for q in range(p+1, (s-p)//2 + 1):
            r = s - p - q
            if r <= q:
                continue
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if best is None or ratio < best:
                best = ratio
    best_for_sum_cache[s] = best
    return best

cases = [
    (1848374, 1848375),
    (10162574, 10162575),
    (265710644, 265710645),
    (44636594, 44636595),
]

for c_val, cp1 in cases:
    # Remove factors of 3
    t = cp1
    while t % 3 == 0:
        t //= 3
    divs = get_divisors(t)
    print(f"\nc = {c_val}: c+1 = {cp1}, coprime-to-3 part = {t}")
    print(f"  Number of coprime-to-3 divisors: {len(divs)}")
    # Only check divisors >= 7 (need at least sum 7 for a useful triple)
    best = Fraction(10**9)
    best_s = None
    for d in divs:
        if d < 7:
            continue
        if d > 5000:  # skip very large sums (ratio approaches 2/3)
            continue
        r = best_ratio_for_sum(d)
        if r is not None and r < best:
            best = r
            best_s = d
    print(f"  Best: sum={best_s}, ratio={best} = {float(best):.8f}")

# Result should match what we found earlier
