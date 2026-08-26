from fractions import Fraction
import math

def f_val(n):
    """Find smallest m with 3 distinct divisors summing to n."""
    for m in range(1, max(n*3, 100)):
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

def best_ratio_for_n(n):
    """Compute f(n)/n = min over d|n of best_ratio(d)."""
    # For each divisor d of n, find the best triple with sum d
    best = Fraction(10**9)
    for d in range(3, n+1):
        if n % d != 0:
            continue
        # Best triple with sum d
        for p in range(1, d//3 + 1):
            for q in range(p+1, (d-p)//2 + 1):
                r = d - p - q
                if r <= q:
                    continue
                l = math.lcm(p, math.lcm(q, r))
                ratio = Fraction(l, d)
                if ratio < best:
                    best = ratio
    return best

# Verify for powers of 3 plus small offsets
# Using k=6 (M=729, L=6):
# g(c) = (1/6) * floor(6 * f(729+c) / 729)
M = 729
L = 6

# For c=0:
n = M
fv = f_val(n)
gv = Fraction(1, L) * (L * fv // M)
print(f"c=0: n={n}, f={fv}, f/M={Fraction(fv,M)}, g={gv}")
print(f"  Expected g = 2/3 = {Fraction(2,3)}")

# For c=4M=2916:
# n=5M=3645, but n is too large for brute force.
# Let me try smaller k.

# k=4, M=81, L=4:
M4 = 81
L4 = 4
print(f"\nk=4, M={M4}, L={L4}")
for c_mult in [0, 1, 2, 3, 4]:
    c = c_mult * M4
    n = M4 + c
    fv = f_val(n)
    if fv is not None:
        g_exact = Fraction(1, L4) * (L4 * fv // M4)
        ratio_expected = Fraction(2 * (1 + c_mult), 3)
        print(f"  c={c_mult}M: n={n}, f={fv}, f/M={Fraction(fv,M4)}, g={g_exact}, expected={ratio_expected}, match={g_exact == ratio_expected}")

# Now for c NOT a multiple of M, check the formula g(c) = best_ratio:
# k=5, M=243, L=5:
M5 = 243
L5 = 5
print(f"\nk=5, M={M5}, L={L5}")

# Check c where 25 | (n = 243 + c), i.e., c = 24 mod 25 and c = -243 mod 25
# 243 mod 25 = 18. Need c = (25-18) mod 25 = 7 mod 25.
# Wait: 25 | (243 + c) means (243+c) mod 25 = 0. 243 mod 25 = 18. c mod 25 = 7.
# c = 7: n = 250. 250 / 25 = 10. Triple (1,8,16)*10 = (10, 80, 160). Sum = 250. lcm = 160.
# f(250) = ?
fv = f_val(250)
print(f"  f(250) = {fv}")  # Should be 150 from our earlier data (triple (25,75,150))
# 150 = 3/5 * 250. Sum 10 divides 250. Best ratio for sum 10: 3/5.
# Versus 16/25 for sum 25: 16/25 * 250 = 160. 3/5 * 250 = 150. 150 < 160.
# So f(250) = 150, not 160.
# g(7) for k=5: (1/5) * floor(5 * 150 / 243) = (1/5) * floor(750/243) = (1/5) * 3 = 3/5.
g_val = Fraction(1, L5) * (L5 * fv // M5)
print(f"  g(7) = {g_val}")

# Hmm, but for the REAL problem with L = 2025! and M = 3^L:
# n = 3^L + 7. Does 10 divide n? n mod 10: 3^L mod 10 = 1 (since 4|L). 1+7 = 8 mod 10. NO.
# Does 25 divide n? n mod 25: 3^L mod 25 = 1. 1+7 = 8 mod 25. NO!
# So NEITHER 10 nor 25 divides 3^L + 7 in the real problem.

# The c values in the problem are specifically chosen so that certain sums divide n.
# For the small k case (k=5), the divisibility is different because 3^5 = 243 has different residues.
# We can't directly validate the large-L formula using small k!

# The formula g(c) = best achievable ratio (for sums dividing 3^L + c with L = 2025!)
# is valid as long as:
# 1. The best ratio triple gives the true minimum f(n)/n for the specific n.
# 2. The floor operation in g works out to give exactly the ratio.

# Let me verify condition 2 more carefully.
# g(c) = (1/L) * floor(L * f(M+c) / M) where L = 2025!, M = 3^L.
# f(M+c) = ratio * (M+c) where ratio = p/q with q | L (since q is a sum < 1000 < 2025, q | 2025!).
# f(M+c)/M = ratio * (1 + c/M).
# L * ratio * (1 + c/M) = L*ratio + L*ratio*c/M = L*p/q + L*p*c/(q*M).
# L*p/q is an integer (since q | L).
# L*p*c/(q*M) = L*p*c/(q*3^L). This is extremely tiny (3^L >> L*p*c/q).
# So floor(L * f(M+c)/M) = L*p/q + floor(L*p*c/(q*3^L)) = L*p/q.
# Therefore g(c) = p/q = ratio. Confirmed!

# But wait: we assumed f(M+c) = ratio * (M+c). Is this exact?
# f(n) = lcm(pa, qa, ra) where a = n/s, s = p+q+r.
# For s | n: a = n/s is an integer. lcm(p,q,r)*a = lcm*n/s.
# f(n)/n = lcm/s = ratio. And f(n) = ratio * n exactly.
# But is this the MINIMUM? We need f(n) = min lcm(a,b,c) over a+b+c=n.
# Our search showed this is the minimum for sums dividing n and sum < 1000.
# But what about larger sums?

# For the given c values, the best ratios found are:
# 16/25, 30/47, 64/97, 110/167.
# These are specific to the (1,2k,4k) and (2,2k+1,2(2k+1)) families.
# Could a different family with larger sum give a better ratio?

# Family (1,2k,4k): sum=6k+1, ratio=4k/(6k+1) -> limit 2/3 as k->inf.
# Family (2,2k+1,2(2k+1)): sum=4k+5, ratio=2(2k+1)/(4k+5) -> limit 1/2... wait:
# lim 2(2k+1)/(4k+5) = lim (4k+2)/(4k+5) = 1 as k->inf.
# Actually that's wrong. Let me recompute.
# (2, 2k+1, 2(2k+1)): lcm = 2(2k+1) (since gcd(2, 2k+1)=1).
# ratio = 2(2k+1) / (2 + 2k+1 + 2(2k+1)) = 2(2k+1)/(4k+5).
# lim = (4k+2)/(4k+5) -> 1. So this ratio approaches 1, getting WORSE.
# Hmm, that can't be right for the smaller k values.
# k=1: 6/11 ≈ 0.545. k=2: 10/17 ≈ 0.588. k=3: 14/23 ≈ 0.609. k=27: 110/167 ≈ 0.659.
# Indeed, as k grows, the ratio increases toward 1. So the BEST is k=1 (ratio 6/11).

# Family (1, 2k, 4k): ratio = 4k/(6k+1). lim = 4/6 = 2/3.
# k=1: 4/7 ≈ 0.571. k=2: 8/13 ≈ 0.615. k=4: 16/25 = 0.64. k=8: 32/49 ≈ 0.653.
# k=16: 64/97 ≈ 0.660.
# These also approach 2/3 from below.

# Family (2, 3, 6): best ratio 6/11 but only works for c=10 mod 11.
# Family (1, 2, 4): ratio 4/7 but only works for c=6 mod 7.

# For c values where neither 7 nor 11 divides n, we're stuck with worse ratios.
# The c values were chosen to have c = s-1 mod s for specific primes s:
# c=1848374: c+1 = 1848375 = 3 * 5^4 * 2959 = ... actually 25 | (c+1)?
# 1848374 + 1 = 1848375. 1848375 / 25 = 73935. Yes!
# And 1848375 / 5 = 369675. 369675/5 = 73935. So 25 | 1848375.
# Is 1848375 / 7 = 264053.57... no. So 7 does not divide c+1.
# 1848375 / 11 = 168034.09... no. 11 does not divide c+1.

# c+1 = 1848375 = 3 * 5^4 * ... let me factorize
n = 1848375
temp = n
facts = []
for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
    while temp % p == 0:
        facts.append(p)
        temp //= p
if temp > 1:
    facts.append(temp)
print(f"\n1848374 + 1 = {n} = {'*'.join(map(str,facts))}")

n = 10162575
temp = n
facts = []
for p in range(2, 1000):
    while temp % p == 0:
        facts.append(p)
        temp //= p
if temp > 1:
    facts.append(temp)
print(f"10162574 + 1 = {n} = {'*'.join(map(str,facts))}")

n = 265710645
temp = n
facts = []
for p in range(2, 1000):
    while temp % p == 0:
        facts.append(p)
        temp //= p
if temp > 1:
    facts.append(temp)
print(f"265710644 + 1 = {n} = {'*'.join(map(str,facts))}")

n = 44636595
temp = n
facts = []
for p in range(2, 1000):
    while temp % p == 0:
        facts.append(p)
        temp //= p
if temp > 1:
    facts.append(temp)
print(f"44636594 + 1 = {n} = {'*'.join(map(str,facts))}")

# The key: c+1 = c_val + 1 must be divisible by the sum s.
# Since 3^L = 1 mod s for s coprime to 3, we need s | (1 + c), i.e., s | (c+1).
# So the prime factorization of c+1 determines which sums are available!
