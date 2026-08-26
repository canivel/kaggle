from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# For n = M + c where M = 3^(2025!):
# M ≡ 1 mod p for all primes p != 3 (since p-1 | 2025! for any p <= 2025)
# M ≡ 0 mod 3^k for k <= 2025!
# n = M + c ≡ 1+c mod p (p != 3), n ≡ c mod 3

# f(n) = smallest m such that m has 3 distinct divisors summing to n.
# Best approach: m = d3 where d1 | d3, d2 | d3, d1+d2+d3 = n.
# Pattern: d3 = m, d2 = m/k2, d1 = m/k1 (k1 > k2 >= 2).
# m = n * k1*k2 / (k1*k2 + k1 + k2).
# Ratio f(n)/n = k1*k2/(k1*k2+k1+k2).
# Need m to be a positive integer with k1|m and k2|m.

# For each c value, I need to find the best (smallest) achievable ratio.
# The requirement is that (k1*k2+k1+k2) divides n*k1*k2,
# AND lcm(k1,k2) divides m = n*k1*k2/(k1*k2+k1+k2).

# Let me think about this more carefully with specific c values.

# Also: we can have triples where d3 is NOT m (i.e., lcm(d1,d2,d3) > d3).
# But then f(n) = lcm(d1,d2,d3) >= 2*d3 > d3, which is worse than any same-m approach.
# Wait, not necessarily. If d1|d2 and d2|d3, then lcm = d3. So that's covered.
# If d1 doesn't divide d3 or d2 doesn't divide d3, then lcm > d3.
# For example: d1=2, d2=3, d3=n-5. lcm(2,3,n-5) = lcm(6, n-5).
# If 6 | (n-5): lcm = n-5. This is the d3=m case.
# If not: lcm = 2(n-5) or 3(n-5) or 6(n-5). These are worse.

# So the d3=m approach (where all divisors divide m) is always best.
# And within that, the (k1,k2) parameterization is complete.

# But wait, there's another approach: d1 and d2 don't have to be m/k1 and m/k2 exactly.
# They can be ANY two distinct proper divisors of m. So the constraint is just:
# m has two distinct proper divisors d1, d2 with d1 + d2 = n - m.
# And d1 < d2 < m.

# For this to work: m must have at least 2 proper divisors summing to n-m.
# The set of proper divisors of m is {1, ..., m/2, ...}.
# We need two of them summing to n-m.
# If n-m < 3 (i.e., m > n-3): impossible (smallest sum of 2 distinct proper divisors is 1+2=3).
# If n-m >= 3: we need d1 + d2 = n-m where d1, d2 are proper divisors of m, d1 != d2, d1 < d2 < m.

# The question is: for the best m (smallest), what are the proper divisors?
# We want the SMALLEST m >= n/3 (since d3 = m >= n/3 as d1+d2 >= 3 and d1+d2+d3=n, well d1>=1,d2>=2 so d3<=n-3, and d3=m).
# Actually m must be at least ceil(n/3) since d1+d2 >= 3 but with d1 < d2 < m, actually d1 >= 1, d2 >= 2.

# I think for large n, f(n) is always given by the (k1,k2) approach, specifically:
# f(n) = n/sigma where sigma = 1 + 1/k2 + 1/k1 for the best valid (k1,k2).

# Let me verify this with brute force for some values.

def f_norwegian(n):
    best = float('inf')
    for d1 in range(1, n//3 + 1):
        for d2 in range(d1+1, (n-d1)//2 + 1):
            d3 = n - d1 - d2
            if d3 <= d2:
                continue
            l = lcm(lcm(d1, d2), d3)
            if l < best:
                best = l
    return best

# Verify for n = 3^k:
# Predicted: f = 2n/3 (ratio 2/3), using (k1,k2)=(6,3).
for k in range(3, 8):
    n = 3**k
    fn = f_norwegian(n)
    print(f"f({n}) = {fn}, predicted = {2*n//3}, match = {fn == 2*n//3}")

print()

# Now let me figure out g(c) for each c.
# g(c) = floor(2025! * f(M+c) / M) / 2025!
# Let's denote n = M+c, and f(n) = m = n * k1*k2/(k1*k2+k1+k2) = n * R where R = ratio.
# f(M+c)/M = (M+c) * R / M = R + c*R/M.
# Since M is astronomical, c*R/M ≈ 0 (for c a fixed integer, but c=4M is different).
# 2025! * f(M+c)/M = 2025! * R + 2025! * c * R / M.
# For fixed c (not 4M): 2025! * c * R / M is extremely small (essentially 0).
# So g(c) = floor(2025! * R) / 2025!.

# But wait: f(M+c) might not be EXACTLY n*R because of divisibility requirements.
# The (k1,k2) approach gives f(n) = n*R only when the divisibility conditions are met.
# For n = M+c with M having the special properties, we need to check if
# the conditions are met exactly.

# Let me reconsider. For (k1,k2) with n = M+c:
# m = n * k1*k2 / S where S = k1*k2+k1+k2.
# Need S | n*k1*k2, i.e., (S/gcd(S, k1*k2)) | n.
# Let D = S / gcd(S, k1*k2). Need D | n.
# Also need lcm(k1,k2) | m = n*k1*k2/S.

# These are divisibility conditions on n = M+c.
# For p != 3: n ≡ 1+c mod p.
# For p = 3: n ≡ c mod 3 (and mod 9, etc.).

# Now: if D | n means that for each prime power p^a dividing D:
# p^a | n. For p != 3: p^a | (1+c). For p = 3: 3^a | c.

# Case c = 0: n = M. n ≡ 1 mod p (p!=3), n ≡ 0 mod 3^k.
# So n is divisible by: all 3^k, but NOT by any p != 3 (since n ≡ 1 mod p).
# Except: 1 mod p means p does NOT divide n.
# So for c=0, the ONLY primes dividing n are powers of 3.
# Best ratio requiring only 3-divisibility:
# (k1=6,k2=3): need 9|n. 9|M, yes! f/n = 2/3.
# Can we do better? Need a ratio < 2/3.
# From the table: 6/11, 4/7, 10/17, etc. All need primes != 3 to divide n. Not available for c=0.
# The (k1,k2) = (6,3) gives ratio 2/3 requiring only 9|n. Check what other (k1,k2) need only 3-powers:
# (k1=6,k2=3): S=18+6+3=27. k1*k2=18. gcd(27,18)=9. D=27/9=3. Need 3|n. Yes.
#   Also need lcm(6,3)=6 | m. m = n*18/27 = 2n/3. 6|2n/3 iff 9|n. Yes (M = 3^K, K >= 2).
# (k1=15,k2=3): S=45+15+3=63. k1*k2=45. gcd(63,45)=9. D=7. Need 7|n. No (c=0 means n≡1 mod 7).
# So for c=0, best is 2/3.
# g(0) = floor(2025! * 2/3) / 2025! = (2*2025!/3) / 2025! = 2/3.
# (Since 3 | 2025!, 2*2025!/3 is an integer, and the tiny correction from c*R/M is negative, so floor = 2*2025!/3 - 1?
# Wait: actually f(M) = 2M/3 exactly (verified above). So f(M)/M = 2/3 exactly.
# 2025! * 2/3 is an integer. floor of integer = itself.
# g(0) = 2/3.

print("g(0) = 2/3")
print()

# Case c = 4M: n = 5M = 5*3^K.
# n is divisible by 3, 5, 9, 15, 25? No, 25 does not divide 5*3^K (only one factor of 5).
# n ≡ 0 mod 5, n ≡ 0 mod 3^k for all k. n ≡ 5 mod p for p != 3,5. So NOT divisible by 2,7,11,etc.
# Best ratio requiring divisibility by 5 and/or 3:
# (k1=6,k2=3): need 9|n. Yes. f/n = 2/3.
# Can we do better using 5|n?
# Looking for (k1,k2) requiring only {3,5}-divisibility and ratio < 2/3:
# (k1=6,k2=2): need 10|n. 10 = 2*5. But 2 does not divide n (n is odd). Fail.
# (k1=12,k2=3): S=36+12+3=51. k1k2=36. gcd(51,36)=3. D=17. Need 17|n. n≡5 mod 17≠0. Fail.
# (k1=9,k2=3): S=27+9+3=39. k1k2=27. gcd(39,27)=3. D=13. Need 13|n. n≡5 mod 13≠0. Fail.
# (k1=24,k2=3): S=72+24+3=99. k1k2=72. gcd(99,72)=9. D=11. Need 11|n. n≡5 mod 11≠0. Fail.
# Hmm, all need extra primes. What about (k1,k2) with k2=5?
# (k1=5,k2=3): S=15+5+3=23. k1k2=15. gcd(23,15)=1. D=23. Need 23|n. n≡5 mod 23≠0. Fail.
# (k1=5,k2=4): S=20+5+4=29. k1k2=20. gcd(29,20)=1. D=29. Need 29|n. n≡5 mod 29≠0. Fail.
#
# What about using 5 explicitly in the triple?
# Triple (5, d2, n-5-d2): need 5|m and d2|m where m = lcm(5, d2, n-5-d2).
# If we want m = n-5-d2 (the d3=m approach): need 5|m and d2|m.
# 5|m means 5|(n-5-d2) i.e., 5|d2 (since 5|n because n=5M).
# Let d2 = 5j. Then m = n - 5 - 5j = 5(M-1-j). Need d2|m: 5j | 5(M-1-j), i.e., j|(M-1-j), i.e., j|(M-1).
# Also need 5 < 5j < m: j >= 2 and 5j < 5(M-1-j), j < (M-1)/2.
# m = 5(M-1-j). Minimize m: maximize j. j | (M-1) and j < (M-1)/2.
# Largest j dividing M-1 with j < (M-1)/2.
# M-1 = 3^K - 1. For K = 2025!, this is even but not div by 3.
# As before, M-1 ≡ 0 mod p for all primes p != 3 (where p <= 2026).
# Same analysis: largest divisor of M-1 less than (M-1)/2.
# Smallest prime of M-1 is 2, but (M-1)/2 is not < (M-1)/2.
# Since 3 does not divide M-1, and 4 | M-1 (as shown), largest divisor < (M-1)/2 is (M-1)/4.
# Wait, but also check: does M-1 have a prime factor of 3? No.
# So j = (M-1)/4. Then m = 5(M-1-(M-1)/4) = 5*3(M-1)/4 = 15(M-1)/4.
# f(n) <= 15(M-1)/4. f(n)/n = 15(M-1)/(4*5M) = 3(M-1)/(4M) ≈ 3/4 > 2/3. Worse!

# So for c=4M, best is still 2/3.
# f(5M) = 2*5M/3 = 10M/3.
# g(4M) = floor(2025! * 10M/(3M)) / 2025! = floor(2025! * 10/3) / 2025! = 10/3.
# (2025!/3 is integer, so 10*2025!/3 is integer. floor = 10*2025!/3.)
# g(4M) = 10/3.

print("g(4M) = 10/3")
print()

# For the remaining c values, n is NOT divisible by 3 (since c mod 3 != 0 for all of them).
# c=1848374: c mod 3 = 1848374 mod 3 = ?
for c_name, c_val in [("1848374", 1848374), ("10162574", 10162574), ("265710644", 265710644), ("44636594", 44636594)]:
    print(f"c = {c_val}:")
    print(f"  c mod 3 = {c_val % 3}")
    print(f"  c+1 = {c_val + 1}")
    # Factor c+1
    temp = c_val + 1
    factors = {}
    for p in range(2, 1000):
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    print(f"  c+1 factorization: {factors}")

    # n ≡ 0 mod p (p!=3) iff p | (c+1)
    # n NOT divisible by 3
    # So we need ratios that DON'T require 3-divisibility of n.

    # Best ratios without 3-divisibility:
    # (k1=3,k2=2): need 11|n. Does 11|(c+1)?
    c1 = c_val + 1
    print(f"  11 | (c+1): {c1 % 11 == 0}")
    print(f"  7 | (c+1): {c1 % 7 == 0}")
    print(f"  17 | (c+1): {c1 % 17 == 0}")
    print(f"  10 | (c+1): {c1 % 10 == 0} (need also 2|n and 5|n)")
    print(f"  23 | (c+1): {c1 % 23 == 0}")
    print(f"  13 | (c+1): {c1 % 13 == 0}")
    print(f"  29 | (c+1): {c1 % 29 == 0}")
    print(f"  19 | (c+1): {c1 % 19 == 0}")
    print(f"  31 | (c+1): {c1 % 31 == 0}")
    print(f"  37 | (c+1): {c1 % 37 == 0}")
    print(f"  41 | (c+1): {c1 % 41 == 0}")
    print(f"  43 | (c+1): {c1 % 43 == 0}")
    print(f"  47 | (c+1): {c1 % 47 == 0}")
    print(f"  53 | (c+1): {c1 % 53 == 0}")
    print(f"  59 | (c+1): {c1 % 59 == 0}")
    print()
