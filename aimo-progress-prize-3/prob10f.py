from fractions import Fraction

# g(c) = (1/2025!) * floor(2025! * f(M+c) / M)
# where M = 3^(2025!), K = 2025!.
#
# f(M+c)/M is some real number. We multiply by K! = 2025!, floor, divide by K!.
# This gives the floor of f(M+c)/M to precision 1/K!.
# Since K! = 2025! is a specific integer, g(c) is a rational number with denominator dividing K!.
#
# Actually: g(c) = floor(K! * f(M+c)/M) / K!.
# This is the largest multiple of 1/K! that is <= f(M+c)/M.
#
# For c=0: f(M)/M = 2/3. g(0) = floor(K! * 2/3) / K! = 2/3 (since 3|K!).
# For c=4M: f(5M)/M = 10/3. g(4M) = 10/3.
#
# For non-multiple c: we need f(M+c)/M exactly, and then floor to nearest 1/K!.
#
# The key insight: for M = 3^K with very large K, and fixed c:
# f(M+c) is determined by the number theory of M+c.
# The strategy with d1, d2, d3 summing to M+c:
# Use d1 = c (if c > 0 and c | m), d2 = 3^(K-1), d3 = rest? No...
#
# Actually, let me think about this differently.
# For n = M + c = 3^K + c:
# Strategy: use divisors that divide m, summing to n.
# If we use (1, d, n-1-d): m = lcm(d, n-1-d), want to maximize gcd(d, n-1-d) | (n-1).
# n-1 = 3^K + c - 1.
#
# For large K (K = 2025!), 3^K - 1 is divisible by all primes up to 2025.
# n-1 = (3^K - 1) + c.
# gcd(d, n-1-d) divides n-1 = 3^K + c - 1.
# The factorization of n-1 depends heavily on c.
#
# For c = 0: n-1 = 3^K - 1. Factor structure is well known.
# For c = 1: n-1 = 3^K. Largest divisor of 3^K less than 3^K/2 is 3^{K-1}.
#   d = 3^{K-1}, n-1-d = 3^K - 3^{K-1} = 2*3^{K-1}.
#   lcm(3^{K-1}, 2*3^{K-1}) = 2*3^{K-1}.
#   m = 2*3^{K-1} = 2M/3.
#   f(M+1) = 2M/3. f(M+1)/M = 2/3.
# Hmm, but the data shows f(2188)/2187 = 1456/2187 ≈ 0.6658 which is close to 2/3 but not exact.

# Wait, for K=7: n=2188, n-1=2187=3^7. Largest divisor of 3^7 less than 3^7/2 = 1093.5 is 3^6=729.
# d=729, n-1-d = 2187-729 = 1458 = 2*729 = 2*3^6.
# lcm(729, 1458) = 1458. m=1458.
# But f(2188) = 1456, which is LESS than 1458!
# So there's a BETTER strategy than (1, d, n-1-d).

# Let me find what divisors achieve f(2188) = 1456.
m = 1456
n = 2188
divs = sorted([d for d in range(1, m+1) if m % d == 0])
print(f"Divisors of {m}: {divs}")
for i in range(len(divs)):
    for j in range(i+1, len(divs)):
        for k in range(j+1, len(divs)):
            if divs[i] + divs[j] + divs[k] == n:
                print(f"  Triple: ({divs[i]}, {divs[j]}, {divs[k]})")

# 1456 = 2^5 * 7 * 6.5? No. 1456/2 = 728, 728/2 = 364, 364/2 = 182, 182/2 = 91, 91/7 = 13.
# 1456 = 2^4 * 91 = 16 * 91. 91 = 7*13. So 1456 = 2^4 * 7 * 13.
print(f"\n1456 = 2^4 * 7 * 13")

# For k=8: f(6562) = 3860. Let me check.
m = 3860
n = 6562
divs = sorted([d for d in range(1, m+1) if m % d == 0])
for i in range(len(divs)):
    for j in range(i+1, len(divs)):
        for k in range(j+1, len(divs)):
            if divs[i] + divs[j] + divs[k] == n:
                print(f"\nf({n})={m}: Triple: ({divs[i]}, {divs[j]}, {divs[k]})")
                break

# OK this problem is harder than I thought. Let me reconsider.
# For non-multiples of M, the answer isn't simply 2/3.
# Let me re-examine what g(c) actually computes.

# g(c) = (1/K!) * floor(K! * f(M+c) / M)
# Since K! and M are both huge, this is really about the "fractional expansion" of f(M+c)/M.

# Hmm, maybe I should think about this from the formula perspective.
# f(n) = min over all valid (a,b,c) triples of lcm(a,b,c) or similar.

# Actually, we proved f(n) = 2n/3 for n = c*3^K when c is a positive integer.
# More precisely, f(c*3^K) = 2c*3^{K-1} for K large enough.
# The triple is (c*3^{K-2}, 2c*3^{K-2}, 2c*3^{K-1}).

# For c = M + small: n = M + c where c is a fixed integer.
# We need f(3^K + c).
# For the specific c values in the problem:
# c = 0: g(0) = 2/3
# c = 4M: n = 5M, g(4M) = 10/3
# c = 1848374: fixed integer
# c = 10162574: fixed integer
# c = 265710644: fixed integer
# c = 44636594: fixed integer

# For fixed integer c and M = 3^K with K -> infinity:
# f(M + c) should be approximately 2(M+c)/3 = 2M/3 + 2c/3.
# But this isn't exact. The correction depends on the arithmetic of c.

# Hmm wait, can we use the triple (3^{K-2}, 2*3^{K-2}, m) where the three sum to 3^K + c?
# 3^{K-2} + 2*3^{K-2} + m_part = 3^K + c => m_part = 3^K + c - 3*3^{K-2} = 3^K - 3^{K-1} + c = 2*3^{K-1} + c.
# We need all three to divide m.
# m >= lcm(3^{K-2}, 2*3^{K-2}, 2*3^{K-1} + c).
# For c = 0: lcm(3^{K-2}, 2*3^{K-2}, 2*3^{K-1}) = 2*3^{K-1}. Good.
# For c != 0: lcm depends on gcd(2*3^{K-1}+c, 3^{K-2}).
# 2*3^{K-1}+c mod 3^{K-2} = 2*3^{K-1} mod 3^{K-2} + c mod 3^{K-2} = 0 + c mod 3^{K-2} = c.
# So gcd(2*3^{K-1}+c, 3^{K-2}) = gcd(c, 3^{K-2}) = 3^{v_3(c)} (if v_3(c) < K-2).

# This means the approach from f(M) = 2M/3 doesn't simply extend to f(M+c) for non-zero c.

# Let me take a completely different approach and think about what g(c) means.
# g(c) = (1/K!) * floor(K! * f(M+c) / M)
#
# Note that M = 3^K where K = 2025!.
# K! = 2025! * floor(K! / 2025!)... wait, K = 2025!, so K! = (2025!)!.
# Oh wait, the problem says g(c) = (1/2025!) * floor(2025! * f(M+c) / M).
# So the "K!" in g is 2025!, NOT (2025!)! = K!.
# Let me re-read: M = 3^(2025!). g(c) = (1/2025!) * floor(2025! * f(M+c)/M).
# So L = 2025!, M = 3^L, g(c) = (1/L) * floor(L * f(M+c)/M).

# f(M+c)/M is some ratio. We multiply by L, floor, divide by L.
# This gives the floor of f(M+c)/M to precision 1/L.

# For c = 0: f(M)/M = 2/3. L*2/3 = 2L/3. Since 3|L, this is exact. g(0) = 2/3.

# For the other fixed c values: f(M+c)/M is very close to some rational number with small denominator.
# The floor to precision 1/L should exactly capture this rational number.

# The question: what IS f(3^L + c) for L = 2025! and fixed moderate c?

# Strategy: pick three distinct divisors of m summing to n = 3^L + c.
# Good triples: (a, b, n-a-b) where all three divide m = lcm(a, b, n-a-b).
# The optimal strategy likely uses divisors related to the factorization of n or n-1.

# Let me think about f(n) for n = 3^L + c where L is a multiple of all integers up to 2025.
# Actually K = 2025! = L. L is divisible by all primes up to 2025.

# Here's an approach: try (1, d, n-1-d) with d | (n-1).
# n-1 = 3^L + c - 1.
# We need large divisors of 3^L + c - 1.
#
# 3^L + c - 1 = (3^L - 1) + c.
# 3^L - 1 is divisible by all primes p where ord_p(3) | L.
# Since L = 2025!, ord_p(3) | (p-1) | L for all primes p <= 2025 (except p=3).
# So 3^L - 1 is divisible by all primes <= 2025 (except 3).
# Then 3^L + c - 1 = (3^L - 1) + c.
# For prime p <= 2025 with p != 3: (3^L - 1) = 0 mod p, so 3^L + c - 1 = c mod p.
# So p | (n-1) iff p | c (for p <= 2025, p != 3).
# Also 3 | (3^L + c - 1) iff 3 | (c-1) (since 3^L = 0 mod 3).

# Very interesting! The small prime factors of n-1 = 3^L + c - 1 are determined by c.

# For the strategy with (1, g, n-1-g) where g is largest divisor of n-1 less than (n-1)/2:
# f(n) = n - 1 - g.
# g is the largest divisor of n-1 less than (n-1)/2.
# n-1 ≈ 3^L (huge). g ≈ (n-1)/p_min where p_min is the smallest prime factor of n-1.
# f ≈ n-1 - (n-1)/p_min = (n-1)(1 - 1/p_min) = (n-1)(p_min-1)/p_min.
# f/M ≈ (p_min - 1)/p_min.

# The smallest prime factor of n-1 = 3^L + c - 1:
# - If c is even: c-1 is odd. 3^L is odd. n-1 = odd + even - 1 = even. So 2 | n-1. p_min = 2.
#   Wait: 3^L is odd. c even means n-1 = 3^L + c - 1 = odd + even - 1 = even. Yes, 2 | n-1.
# - If c is odd: n-1 = odd + odd - 1 = odd. 2 does not divide n-1.
#   Next: 3 | n-1 iff 3 | (c-1).
#   5 | n-1 iff 5 | c (since 3^L = 1 mod 5 for 4|L, so n-1 = 1+c-1 = c mod 5).
#   7 | n-1 iff 7 | c.
#   etc.

# So p_min(n-1) is the smallest prime p such that p | c (if c even) or p | (c-1) if 3|(c-1), etc.
# Actually: p_min(n-1) depends on the residues of c modulo small primes.

# For c = 0: n-1 = 3^L - 1. 2 | (3^L-1) (yes, 3^L is odd). p_min = 2.
#   f ≈ (n-1)/2 ≈ M/2. But we showed f = 2M/3. Contradiction!
#   So the (1, g, n-1-g) strategy is NOT optimal for c=0!

# Hmm, that means my earlier analysis was wrong for powers of 3.
# For n = 3^K: using (3^{K-2}, 2*3^{K-2}, 2*3^{K-1}) gives m = 2*3^{K-1} = 2M/3.
# Using (1, g, n-1-g): g = (n-1)/4 = (3^K-1)/4, m = 3(3^K-1)/4 ≈ 3M/4.
# 2M/3 < 3M/4. So the non-trivial triple is better!

# I need to also consider triples not starting with 1.
# This makes the problem much harder.

# General approach: f(n) = min m such that some 3 distinct divisors of m sum to n.
# Equivalent: f(n) = min_{a < b < c, a+b+c=n} lcm(a,b,c) where we also need a|m, b|m, c|m.
# Wait, m just needs a,b,c as divisors, so m must be divisible by lcm(a,b,c).
# And the SMALLEST such m is lcm(a,b,c) itself.
# So f(n) = min_{a < b < c, a+b+c=n} lcm(a,b,c).

# Great, so f(n) = minimum of lcm(a,b,c) over all triples of distinct positive integers summing to n.

# For n = 3^K: triple (3^{K-2}, 2*3^{K-2}, 2*3^{K-1}), lcm = 2*3^{K-1} = 2M/3.
# Check: 3^{K-2} + 2*3^{K-2} + 2*3^{K-1} = 3^{K-2}(1+2+6) = 9*3^{K-2} = 3^K. YES.
# lcm(3^{K-2}, 2*3^{K-2}, 2*3^{K-1}) = lcm(3^{K-2}, 2*3^{K-1}) = 2*3^{K-1}. YES.

# For n = 3^K + c, we want to find similar good triples.
# The idea: (a, 2a, n-3a) where a and 2a divide lcm(a, 2a, n-3a) = lcm(2a, n-3a).
# n - 3a = 3^K + c - 3a.
# If a = 3^{K-2}: n-3a = 3^K + c - 3^{K-1} = 2*3^{K-1} + c.
# lcm(2*3^{K-2}, 2*3^{K-1}+c).
# 2*3^{K-1}+c = 3*(2*3^{K-2}) + c, so gcd(2*3^{K-2}, 2*3^{K-1}+c) = gcd(2*3^{K-2}, c).
# lcm = 2*3^{K-2} * (2*3^{K-1}+c) / gcd(2*3^{K-2}, c).
# = (2*3^{K-2}/gcd(2*3^{K-2}, c)) * (2*3^{K-1}+c).
# For c = 0: = 2*3^{K-2} * 2*3^{K-1} / (2*3^{K-2}) = 2*3^{K-1}. Good.
# For small c: gcd(2*3^{K-2}, c) = gcd(c, 2*3^{K-2}).
# Let g = gcd(c, 2*3^{K-2}).
# lcm = (2*3^{K-2}/g) * (2*3^{K-1}+c).
# For c small: 2*3^{K-1}+c ≈ 2M/3. And 2*3^{K-2}/g depends on g.
# If c = 0: g = 2*3^{K-2}, lcm = 1 * 2*3^{K-1} = 2M/3.
# If gcd(c, 6) = 6 (e.g., c = 6): g = gcd(6, 2*3^{K-2}) = 6.
#   lcm = (2*3^{K-2}/6) * (2*3^{K-1}+6) = (3^{K-2}/3) * (2*3^{K-1}+6) = 3^{K-3} * (2*3^{K-1}+6).
#   = 3^{K-3} * 2 * (3^{K-1}+3) = 2*3^{K-3}*(3^{K-1}+3) = 2*3^{K-3}*3*(3^{K-2}+1) = 2*3^{K-2}*(3^{K-2}+1).
#   ≈ 2*3^{K-2} * 3^{K-2} = 2*3^{2K-4}. WAY too large!

# So the triple (a, 2a, n-3a) with a = 3^{K-2} only works well when c = 0.

# For small c, we need a different triple. Let me think...
# We want lcm(a, b, c) small with a+b+c = n = 3^K + c.
# The triple should have large gcd to keep lcm small.

# If all three are multiples of some large d: a=dx, b=dy, c=dz, x+y+z = n/d.
# lcm(a,b,c) = d * lcm(x,y,z). We need n/d to be an integer, so d | n = 3^K + c.
# For c = 0: d | 3^K, so d = 3^j. Then x+y+z = 3^{K-j}. lcm = 3^j * lcm(x,y,z).
# Minimize 3^j * lcm(x,y,z) with x+y+z = 3^{K-j}.
# Recursively: f(3^K) = min_j 3^j * f(3^{K-j}).
# With f(3^K) = 2*3^{K-1}: f(3^K)/3^K = 2/3 for all K >= 2.
# Check: f(3) = ? f(3) doesn't exist (no 3 distinct positive integers with 3 divisors summing to 3).
# f(9) = 6. 6/9 = 2/3. Triple (1,2,6): lcm = 6. YES.

# For c != 0: d | (3^K + c). The common divisors of 3^K + c are divisors of c if gcd(3, c) = 1,
# or divisors of gcd(3^K, c) if 3 | c, etc.

# This is getting very complex. Let me try a different approach.
# Since g(c) involves floor to precision 1/L where L = 2025!, and the specific c values are given,
# maybe there's a pattern.

# Key observation: f(n) = min lcm(a,b,c) over a+b+c=n, a<b<c.
# For n = M + c where M = 3^L:
# If c is a multiple of M, say c = jM: n = (j+1)M, f = 2(j+1)M/3, g = 2(j+1)/3.
# For c = 4M: g(4M) = 2*5/3 = 10/3.
# For fixed small c: need to analyze f(M+c)/M.

# Actually, for the triple (1, (n-1)/2, (n-1)/2) with n odd:
# Both equal (n-1)/2, not distinct. So no.
# (1, (n-1)/2 - 1, (n+1)/2) for odd n: sum = 1 + (n-3)/2 + (n+1)/2 = 1 + n - 1 = n.
# lcm(1, (n-3)/2, (n+1)/2). For large n, this is ≈ n^2/4 (if coprime). Way too big.

# Better: (a, a+1, n-2a-1) for some a. This gives small lcm when n-2a-1 is a multiple of a or a+1.
# Or: (a, ka, n-(k+1)a) where n-(k+1)a is related.
# E.g., (a, 2a, n-3a): lcm = lcm(2a, n-3a). If (n-3a) = 2ja for some j:
# lcm = 2a * j = (n-3a). Then n = 3a + 2ja = a(3+2j). So a | n.
# And f(n) = n - 3a (since lcm(2a, 2ja) = 2ja = n-3a).
# This gives f(n) = n*(1 - 3/(3+2j)) = 2jn/(3+2j).
# Minimize over j >= 1 (need a < 2a < n-3a, so n-3a > 2a means n > 5a):
# j=1: f = 2n/5. a = n/5.
# j=2: f = 4n/7. a = n/7.
# j=3: f = 6n/9 = 2n/3. a = n/9.
# So with (a, 2a, 6a) where a = n/9: f = 6a = 2n/3 when 9|n.
# This matches f(3^K) = 2*3^{K-1} = 2*3^K/3.

# For n NOT divisible by 9 (or 5 or 7): we can't use this triple directly.
# But for n = 3^K + c with c small: n mod 9 = c mod 9 (since 3^K = 0 mod 9 for K >= 2).
# If c = 0 mod 9: n = 0 mod 9, f(n) = 2n/3 achievable.
# If c != 0 mod 9: need different approach.

# For n = M + c with specific c:
# c = 1848374: c mod 9 = ?
print(f"1848374 mod 9 = {1848374 % 9}")
print(f"10162574 mod 9 = {10162574 % 9}")
print(f"265710644 mod 9 = {265710644 % 9}")
print(f"44636594 mod 9 = {44636594 % 9}")
print(f"0 mod 9 = 0")
print(f"4M mod 9 = 0 (since M = 3^K)")

# Also check divisibility by 5, 7
for c_val in [0, 1848374, 10162574, 265710644, 44636594]:
    print(f"\nc = {c_val}:")
    for p in [2, 3, 5, 7, 9, 11, 13, 15, 21, 35, 45, 63, 105]:
        print(f"  c mod {p} = {c_val % p}")
