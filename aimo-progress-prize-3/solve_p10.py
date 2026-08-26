from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

def f_norwegian(n):
    best = float('inf')
    best_triple = None
    for d1 in range(1, n//3 + 1):
        for d2 in range(d1+1, (n-d1)//2 + 1):
            d3 = n - d1 - d2
            if d3 <= d2:
                continue
            l = lcm(lcm(d1, d2), d3)
            if l < best:
                best = l
                best_triple = (d1, d2, d3)
    return best, best_triple

# For M = 3^(2025!), which is astronomically large:
# Key properties:
# M ≡ 1 mod p for all primes p != 3 (since ord_p(3) | p-1 | 2025!)
# M ≡ 0 mod 3^k for any k <= 2025!

# For n = M + c:
# n ≡ (1 + c) mod p for p != 3
# n ≡ c mod 3

# Strategy: the triple (d, kd, m) where m = lcm and d, kd, m are divisors of m summing to n.
# Best known patterns (d1 = m/k1, d2 = m/k2, d3 = m where k1 > k2 >= 2):
# Sum = m(1 + 1/k2 + 1/k1) = n. f(n) = m = n / (1 + 1/k2 + 1/k1).
#
# Best ratios f(n)/n = 1/(1 + 1/k2 + 1/k1):
# (k1,k2) = (6,3): ratio = 1/(1+1/3+1/6) = 1/(3/2) = 2/3. Need lcm(6,3)=6 divides m=2n/3. So 9|n.
# (k1,k2) = (3,2): ratio = 1/(1+1/2+1/3) = 1/(11/6) = 6/11. Need lcm(3,2)=6 divides 6n/11. So 11|n.
# (k1,k2) = (4,2): ratio = 1/(1+1/2+1/4) = 1/(7/4) = 4/7. Need lcm(4,2)=4 divides 4n/7. So 7|n.
# (k1,k2) = (5,2): ratio = 1/(1+1/2+1/5) = 1/(17/10) = 10/17. Need lcm(5,2)=10 divides 10n/17. So 17|n.
# (k1,k2) = (6,2): ratio = 1/(1+1/2+1/6) = 1/(5/3) = 3/5. Need lcm(6,2)=6 divides 3n/5. So 10|n (2|n and 5|n).
# (k1,k2) = (4,3): ratio = 1/(1+1/3+1/4) = 1/(19/12) = 12/19. Need lcm(4,3)=12 divides 12n/19. So 19|n.
# (k1,k2) = (5,3): ratio = 1/(1+1/3+1/5) = 1/(23/15) = 15/23. Need lcm(5,3)=15 divides 15n/23. So 23|n.
# (k1,k2) = (8,2): ratio = 1/(1+1/2+1/8) = 1/(13/8) = 8/13. Need lcm(8,2)=8 divides 8n/13. So 13|n.
# (k1,k2) = (10,2): ratio = 1/(1+1/2+1/10) = 1/(8/5) = 5/8. Need lcm(10,2)=10 divides 5n/8. Hmm need 16|n? Let me check.
#   m = 5n/8. Need 10 | 5n/8. 10 | 5n/8 iff 2 | n/8 iff 16|n.
#   Also need 2 | 5n/8 (for k2=2): 5n/8 / 2 = 5n/16 must be integer. So 16 | 5n, i.e., 16|n.
#   Hmm, too restrictive.

# Actually I need to be more careful. The divisors are m/k1, m/k2, m.
# These must be DISTINCT positive integers. k1 != k2 and both != 1.
# And m must be divisible by k1 and k2.
# m = n * k1 * k2 / (k1*k2 + k1 + k2).
# We need m to be a positive integer divisible by both k1 and k2, i.e., by lcm(k1,k2).

# Let me also consider triples where d3 != m (d3 < m, m = lcm(d1,d2,d3) > d3).
# For example: (2, 3, n-5) with lcm(2,3,n-5) = lcm(6, n-5).
# If 6 | (n-5): lcm = n-5. Otherwise lcm could be 2(n-5), 3(n-5), or 6(n-5).
# If 6 | (n-5): f(n) <= n-5. Ratio = (n-5)/n ≈ 1. Bad.
# If n-5 is odd and 3|(n-5): lcm = 2(n-5). Ratio ≈ 2. Very bad.

# So the (m/k1, m/k2, m) pattern seems best. Let me also consider (d1, d2, d3) where
# d1 | d3 and d2 | d3, so m = d3.
# Then d1 + d2 + d3 = n. d1 | d3, d2 | d3. d3 = n - d1 - d2.
# This is equivalent to the above pattern with m = d3.

# But also (d1, d2, d3) where none divides lcm directly through one...
# Actually lcm(d1,d2,d3) is always >= max(d1,d2,d3), and if all three divide d3,
# then lcm = d3. So the pattern where d1|d3 and d2|d3 always gives lcm = d3 (as long as d1|d3, d2|d3).
# This is optimal since lcm >= d3 always.

# So f(n) = min d3 where d3 = n - d1 - d2, d1 | d3, d2 | d3, d1 < d2 < d3.
# Equivalently: d3 has divisors d1 < d2 with d1 + d2 + d3 = n, i.e., d1 + d2 = n - d3.
# sigma_sub(d3) = sum of some pair of divisors of d3 = n - d3.
# We need d3 to have two distinct proper divisors (both < d3) summing to n - d3.

# For the (k1,k2) pattern: d3 = m = n/(1 + 1/k2 + 1/k1).
# For this to work, n - d3 = n - n/(1+1/k2+1/k1) = n * (1/k2 + 1/k1)/(1+1/k2+1/k1)
# And the two divisors are d3/k2 and d3/k1.

# OK let me just enumerate the best possible ratios and their requirements on n.
# I'll collect them and for each c value, find which applies.

ratios = []
# (k1, k2): ratio = k1*k2/(k1*k2+k1+k2), requirement: n must be divisible by (k1*k2+k1+k2)/gcd(...)
for k1 in range(2, 50):
    for k2 in range(2, k1):
        # Wait, I had k1 > k2. d1 = m/k1 < d2 = m/k2. So k1 > k2.
        numer = k1 * k2
        denom = k1 * k2 + k1 + k2
        g = gcd(numer, denom)
        ratio = Fraction(numer, denom)
        # m = n * k1 * k2 / denom. Need m to be integer and k1|m and k2|m.
        # m = n * numer / denom. For m integer: denom/gcd(numer,denom) | n.
        # Also need k1 | m and k2 | m, i.e., lcm(k1,k2) | m.
        # m = n * numer / denom. lcm(k1,k2) | n*numer/denom.
        # This is complex, let me just find what n must be divisible by.
        L = lcm(k1, k2)
        # m = n * numer / denom
        # Need denom | n*numer (for m integer): let d = denom/gcd(numer,denom). Need d | n.
        d_req = denom // gcd(numer, denom)
        # Need L | m = n*numer/denom: L*denom | n*numer.
        # L*denom/gcd(L*denom, numer) | n.
        full_req = L * denom // gcd(L * denom, numer)
        # But I also need d1 < d2 < d3: k1 > k2 > 1 ensures m/k1 < m/k2 < m.
        # And all three distinct.
        ratios.append((ratio, k1, k2, full_req))

# Sort by ratio (ascending = better/smaller f(n))
ratios.sort(key=lambda x: x[0])

# Print best 30
print("Top 30 ratios (smallest f(n)/n):")
seen = set()
for ratio, k1, k2, req in ratios[:50]:
    key = (ratio, req)
    if key not in seen:
        seen.add(key)
        print(f"  (k1={k1}, k2={k2}): f/n = {ratio} = {float(ratio):.6f}, need {req} | n")
    if len(seen) >= 30:
        break

print()

# Now for each c, determine which primes divide n = M+c.
# For p != 3: p | n iff p | (1+c)
# For p = 3: 3 | n iff 3 | c; 9 | n iff 9 | c; etc.

c_values = {
    'c=0': 0,
    'c=4M': None,  # special: n = 5M
    'c=1848374': 1848374,
    'c=10162574': 10162574,
    'c=265710644': 265710644,
    'c=44636594': 44636594,
}

print("Prime divisibility of n = M+c:")
for label, c in c_values.items():
    if c is None:
        # n = 5M = 5 * 3^K
        # For p != 3: 5M mod p = 5 mod p (since M ≡ 1 mod p, but 5M ≡ 5 mod p)
        # Wait: 5M mod p. M ≡ 1 mod p (for p != 3), so 5M ≡ 5 mod p. So p|n iff p|5, i.e., p=5.
        # For p = 3: 3 | 5M iff 3|5M. Since 3|M, yes. 9|5M iff 9|5M. Since 9|M (for K>=2), yes.
        # So n = 5M is divisible by 3, 5, 9, 27, ..., and 5 only among primes != 3.
        divs_3 = "all powers of 3"
        divs_other = [5]
        print(f"  {label}: n = 5M, divisible by 3^k (all), and 5. NOT by 2,7,11,13,...")
        print(f"    9|n: True. Best ratio with 9|n: 2/3")
        # Check: can we do better?
        # n = 5M. 5|n yes. 11|n? 5M mod 11 = 5 mod 11 = 5. No.
        # 7|n? 5M mod 7 = 5. No.
        # So among the nice ratios, 9|n gives 2/3 as the best.
        # But wait, 5|n too. Can we use both?
        # Need a ratio requiring 5|n and/or 9|n.
        # (k1=6, k2=3) requires 9|n: ratio 2/3.
        # (k1=6, k2=2) requires 10|n (2|n and 5|n). 2|5M: M=3^K is odd, 5M is odd. So 2 does NOT divide n. So can't use.
        # So best for c=4M: ratio = 2/3.
    else:
        divs = []
        c_plus_1 = c + 1
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p == 3:
                if c % 3 == 0:
                    divs.append(3)
                if c % 9 == 0:
                    divs.append(9)
                if c % 27 == 0:
                    divs.append(27)
            else:
                if c_plus_1 % p == 0:
                    divs.append(p)
        print(f"  {label}: c={c}, c+1={c_plus_1}")
        print(f"    Primes (!=3) dividing n: p|n iff p|(c+1)={c_plus_1}")

        # Factor c+1 for small primes
        temp = c_plus_1
        factors = []
        for p in range(2, 200):
            while temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        print(f"    c+1 = {c_plus_1} = {'*'.join(map(str, factors))}")

        # Factor c for mod 3
        c3 = c % 9
        print(f"    c mod 9 = {c3}, so {'9|n' if c3==0 else ('3|n' if c%3==0 else '3 does not divide n')}")
