from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# For n >= 25 (verified up to 500), the optimal triple always has d1|d3 and d2|d3.
# For our problem, n = M+c where M is huge, so this is fine.

# Now let me also verify my ratio computation more carefully.
# The (k1, k2) approach: m = d3, d1 = m/k1, d2 = m/k2, d3 = m.
# d1+d2+d3 = m(1/k1 + 1/k2 + 1) = n.
# m = n / (1/k1 + 1/k2 + 1) = n * k1*k2 / (k1*k2 + k1 + k2).

# But we can also have triples (d1, d2, d3) where d1|d3, d2|d3 but d1, d2 are not
# necessarily m/k for integer k. Well, d1 | d3 means d3/d1 is an integer, so k1 = d3/d1 IS an integer.
# Similarly k2 = d3/d2. So the (k1,k2) parametrization IS complete for the d3=m case.

# However, there's a subtlety: the (k1,k2) approach requires m to be a SPECIFIC value
# (m = n*k1*k2/S). But for the d3=m approach, m can be any value where m has proper divisors
# d1, d2 with d1+d2 = n-m. So m doesn't have to follow the exact formula.

# Wait, if d1 = m/k1 and d2 = m/k2, then d1+d2 = m/k1+m/k2 = m(k1+k2)/(k1*k2).
# And d1+d2 = n-m. So n-m = m(k1+k2)/(k1*k2), hence n = m + m(k1+k2)/(k1*k2) = m * (k1*k2+k1+k2)/(k1*k2).
# m = n * k1*k2 / (k1*k2+k1+k2). This is exactly the formula.
# So yes, the (k1,k2) approach covers ALL triples with d1|d3, d2|d3. Good.

# But there's ANOTHER subtlety: d3 doesn't have to be m. d3 is the LARGEST of three divisors of m.
# m = lcm(d1,d2,d3). If d1|d3 and d2|d3, then lcm = d3 = m.
# But if d3 < m, that means m = lcm > d3. Then m has MORE divisors.
# For example: m = 12, divisors include 1,2,3,4,6,12.
# Triple (2,4,6): sum = 12, lcm(2,4,6) = 12. Here d3=6 < m=12.
# But 2|12, 4|12, 6|12. So all three divide m=12.
# From the (k1,k2) perspective with d3=m: m=12, d1=m/k1, d2=m/k2.
# d1+d2+d3 = m/k1+m/k2+m = 12. But d3=m=12 means d1+d2=0. Contradiction.
# Actually, the triple (2,4,6) has sum 12 but m=lcm=12. The THREE divisors of m=12 are 2,4,6.
# In the (k1,k2) approach with d3=m: (m/k1, m/k2, m) where k1=6, k2=3: (2, 4, 12) sums to 18, not 12.
# So n=12 is NOT captured by (k1,k2) with d3=m=12.
# Instead, (2,4,6) gives m=12 but NOT through d3=m. d3=6, m=12.

# So I was wrong: the (k1,k2) approach does NOT cover all cases.
# The general case: pick ANY m, find three distinct divisors of m summing to n, minimize m.

# For large n, the best strategy:
# Method A (d3=m): pick (k1,k2), m = n*k1*k2/(k1*k2+k1+k2).
# Method B (d3 < m): pick three divisors d1 < d2 < d3 of some m > d3, summing to n.
#   m must be divisible by all three. m = lcm(d1,d2,d3).
#   Example: d1=a, d2=2a, d3=3a. Sum = 6a = n. m = lcm(a,2a,3a) = 6a = n.
#   So m = n, ratio = 1. Bad.
#   Example: d1=a, d2=2a, d3=4a. Sum = 7a = n. m = lcm(a,2a,4a) = 4a = 4n/7.
#   Same as (k1,k2)=(4,2) with d3=m=4n/7. But wait: d3=4a = 4n/7 and m = lcm(a,2a,4a) = 4a = d3.
#   So this IS d3=m. OK.
#   Example: d1=2, d2=4, d3=6, n=12. m = lcm(2,4,6) = 12. d3=6 < m=12.
#   Here m = 12 = n. Ratio m/n = 1.
#   But f(12) = 12 from brute force? Let me check.

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

print(f"f(12) = {f_norwegian(12)}")
# f(12) = (12, (2,4,6)). So f(12) = 12 and the ratio is 1. But earlier table showed f(12) = 12.
# But the (k1,k2) approach with (3,2): need 11|n. 11 doesn't divide 12.
# (4,2): need 7|n. 7 doesn't divide 12. (6,3): need 9|n. 9 doesn't divide 12.
# So for n=12, the d3=m approach gives no valid triple, and the best is the lcm=12 approach.
# But 12/12 = 1, which is the worst possible ratio. However n=12 is small.

# For our problem, n is huge and has specific prime structure. The d3=m approach should work.
# Let me verify: for n = M+c where c values give specific prime structures,
# is the (k1,k2) approach valid?

# Actually, I realize there's a more general approach for d3 < m.
# Pick divisors d1, d2, d3 of m where d1+d2+d3 = n and m = lcm(d1,d2,d3).
# For d3 < m: m > d3, and m is a multiple of d3. Let m = j*d3 for j >= 2.
# Then d1, d2 must also divide m = j*d3.
# d1 + d2 = n - d3.
# m = lcm(d1, d2, d3). Since d3 | m, we need m to also be divisible by d1 and d2.
# We're minimizing m = lcm(d1, d2, d3).

# For large n, Method B gives m = n at worst (trivially: d1=1, d2=n-2, d3=n-... hmm no).
# Actually with (1, 2, n-3): lcm(1,2,n-3). If n-3 is even: lcm = n-3. d3=m=n-3 (Method A).
# If n-3 is odd: lcm = 2(n-3). Method B. m = 2(n-3) ≈ 2n. Bad.

# So for large n with nice prime structure, Method A (d3=m) gives much better results.
# And for our specific problem, Method A applies.

# Let me now also consider: maybe there are even BETTER triples using more complex patterns.
# For instance, (d1, d2, d3) where d1 | d2 (instead of both dividing d3).
# Then lcm = lcm(d2, d3). If d2 | d3: lcm = d3 (back to Method A).
# If d2 doesn't divide d3: lcm = lcm(d2, d3) = d2*d3/gcd(d2,d3).
# For this to be small, we need large gcd(d2,d3) relative to d2*d3.
# Let g = gcd(d2,d3). d2 = g*a, d3 = g*b, gcd(a,b)=1. lcm = g*a*b.
# d1 | d2: d1 | g*a. Let d1 = g*a/k for integer k > a (since d1 < d2 = g*a).
# Sum: g*a/k + g*a + g*b = n. g(a/k + a + b) = n.
# m = g*a*b. ratio m/n = a*b/(a/k + a + b) = a*b*k / (a + ka + kb) = abk/(a(1+k)+kb).

# This is a more general parametrization. The (k1,k2) approach with d3=m is a special case
# where a=1 (d2 = g, d3 = g*b, d1 = g/k... hmm this doesn't simplify nicely).

# I think for the purposes of this problem, the (k1,k2) approach captures the relevant cases.
# Let me now also consider: are there patterns with 3 divisors where gcd structure helps?

# For instance, for n divisible by p (large prime), the (k1,k2) = (p-1, 2)/2 approach gives:
# Wait, for (k1,k2): S = k1*k2+k1+k2. Need S | n*k1 and S | n*k2.
# For k2=2: S = 2k1+k1+2 = 3k1+2. Need (3k1+2) | 2n and (3k1+2) | k1*n.
# The 2nd: (3k1+2) | k1*n. Since gcd(3k1+2, k1) = gcd(2, k1):
#   If k1 even: gcd = 2, need (3k1+2)/2 | n/2 * ... hmm complex.
#   If k1 odd: gcd = 1, need (3k1+2) | n.
# For k1 = (p-1)/2 where p is odd: k1 is integer if p is odd. And 3k1+2 = 3(p-1)/2+2 = (3p+1)/2.
# Hmm, this doesn't simplify to p.

# Let me just directly check: for n divisible by 31, what's the best (k1,k2)?
# (k1,k2) where 3k1+2 divides n (for k2=2):
# 3k1+2 = 31 -> k1 = 29/3. Not integer.
# 3k1+2 = 62 -> k1 = 20. S = 62. Need 62 | 2n -> 31 | n. And 62 | 20n -> 31 | n.
#   ratio = 40/62 = 20/31. Let me check: k1=20, k2=2: S = 40+20+2 = 62. ratio = 40/62 = 20/31.

# For k2=2, general: the requirement is often that (3k1+2)/gcd(3k1+2, 2k1) | n.
# Let me just tabulate which (k1,2) pairs need which primes.

print("\nTabulating (k1, 2) requirements:")
for k1 in range(2, 300):
    S = 3*k1 + 2
    # Need S | 2n and S | k1*n
    g1 = gcd(S, 2)
    req1 = S // g1  # from S | 2n
    g2 = gcd(S, k1)
    req2 = S // g2  # from S | k1*n
    req = lcm(req1, req2)
    ratio = Fraction(2*k1, S)
    if req <= 200:  # only show manageable requirements
        print(f"  k1={k1}: S={S}, ratio={ratio}={float(ratio):.6f}, need {req}|n")

# Also check (k1, 3):
print("\nTabulating (k1, 3) requirements:")
for k1 in range(4, 200):
    k2 = 3
    S = k1*k2 + k1 + k2
    ratio = Fraction(k1*k2, S)
    req1_d = S // gcd(S, k1)
    req2_d = S // gcd(S, k2)
    req = lcm(req1_d, req2_d)
    if req <= 200:
        print(f"  k1={k1}: S={S}, ratio={ratio}={float(ratio):.6f}, need {req}|n")
