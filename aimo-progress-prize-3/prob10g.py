from fractions import Fraction
import math

# f(n) = min over {a<b<c, a+b+c=n} of lcm(a,b,c).
#
# Key: for n = M + c where M = 3^L with L = 2025!:
# Using the triple (a, 2a, n-3a) requires 9 | n. n = M + c, 9 | n iff 9 | c.
# For c = 0: 9 | 0, f = 2n/3.
# For c = 4M: n = 5M. 9 | 5M since 9 | M. f = 10M/3 = 2n/3.
#
# For fixed non-zero c not divisible by 9: need other triples.
# Approach: (a, ka, n-(k+1)a) for various k.
# Need (k+1)a | n, or at least the triple gives small lcm.
# Actually, just need a | n for the factored form to work.
#
# Let me try: (a, 2a, n-3a) where a = floor(n/9) or nearby.
# If n = 9q + r (r = c mod 9):
# (a, 2a, n-3a): need a < 2a < n-3a, so n > 5a.
# lcm(a, 2a, n-3a) = lcm(2a, n-3a).
# Let d = gcd(2a, n-3a). lcm = 2a(n-3a)/d.
# n - 3a = n - 3a. If a = (n-r)/9 = q + (9q+r - r)/(9) = q... wait.
# n = 9q + r. If a = q: 3a = 3q, n-3a = 6q+r.
# lcm(2q, 6q+r) = 2q(6q+r)/gcd(2q, 6q+r).
# gcd(2q, 6q+r) = gcd(2q, r) (since 6q = 3*2q).
# lcm = 2q(6q+r)/gcd(2q, r).
#
# For r = 0: lcm = 2q*6q/2q = 6q = 2n/3. Good.
# For r != 0: lcm = 2q(6q+r)/gcd(2q,r).
# If gcd(2q,r) = r (i.e., r | 2q): lcm = 2q(6q+r)/r = 2q*6q/r + 2q = 12q^2/r + 2q.
# This can be very large (quadratic in q) for small r.
#
# Better approach: let me think about what triples give lcm ≈ 2n/3.
# We want lcm(a,b,c) ≈ 2n/3 with a+b+c = n.
# A natural choice: a ≈ n/9, b ≈ 2n/9, c ≈ 2n/3.
# a:b:c ≈ 1:2:6.
# lcm(a, 2a, 6a) = 6a. And a+2a+6a = 9a = n means a = n/9.
# If 9 does not divide n, we adjust.
#
# For n = M + c where M = 3^L:
# n/9 = 3^{L-2} + c/9. Not integer if 9 does not divide c.
# But we can write:
# a = 3^{L-2}, b = 2*3^{L-2}, c_val = n - 3*3^{L-2} = 3^L + c - 3^{L-1} = 2*3^{L-1} + c.
# Triple: (3^{L-2}, 2*3^{L-2}, 2*3^{L-1}+c).
# Sum = 3^{L-2} + 2*3^{L-2} + 2*3^{L-1} + c = 3*3^{L-2} + 2*3^{L-1} + c = 3^{L-1} + 2*3^{L-1} + c = 3^L + c = n. Good.
# lcm(3^{L-2}, 2*3^{L-2}, 2*3^{L-1}+c).
# = lcm(2*3^{L-2}, 2*3^{L-1}+c).
# gcd(2*3^{L-2}, 2*3^{L-1}+c):
#   2*3^{L-1}+c = 3*(2*3^{L-2}) + c. So gcd(2*3^{L-2}, c) = gcd(2*3^{L-2}, c).
#   Since L-2 is huge, 3^{L-2} has only factor 3. So gcd = 2^{min(1,v_2(c))} * 3^{min(L-2, v_3(c))}.
#   For c not divisible by 3: gcd = gcd(2, c) * 1.
#   For c even and 3 does not divide c: gcd = 2.
#
# lcm = 2*3^{L-2} * (2*3^{L-1}+c) / gcd.
# For c even, gcd(c, 2*3^{L-2}) = 2 * 3^{v_3(c)}.
# Since the c values all have c mod 3 = 2 (not divisible by 3) and are even:
# gcd = 2. lcm = 2*3^{L-2} * (2*3^{L-1}+c) / 2 = 3^{L-2} * (2*3^{L-1}+c).
# = 2*3^{2L-3} + c*3^{L-2}.
# This is approximately 2*3^{2L-3} which is MUCH larger than M = 3^L.
# So this triple is terrible for small nonzero c!

# The issue is that 2*3^{L-1}+c is coprime to 3, so the lcm explodes.
# We need a fundamentally different approach for non-multiples of 9.

# Let me think about the (1, d, n-1-d) approach again.
# f(n) = min(n-1-g) where g = max divisor of n-1 less than (n-1)/2.
# For n = M+c: n-1 = 3^L + c - 1.
# 3^L + c - 1 = (3^L - 1) + c.
# For c = 1848374 (even): n - 1 = 3^L + 1848373. This is even (since 3^L is odd and 1848373 is odd... wait 3^L + c - 1 = 3^L + 1848373. 3^L is odd. 1848373 is odd. Odd + odd = even. So n-1 is even.
#
# Actually wait: n = 3^L + c. c = 1848374 (even). n = odd + even = odd. n-1 = even. Good.
# Smallest prime factor of n-1: 2. So g = (n-1)/4 if 4 | n-1.
# (n-1) = 3^L + 1848373. 3^L mod 4: L = 2025! is even, so 3^L = 1 mod 4.
# 1848373 mod 4: 1848373/4 = 462093.25, so 1848373 mod 4 = 1. n-1 mod 4 = 2. NOT divisible by 4.
# So n-1 = 2 * ((3^L + 1848373)/2) where (3^L + 1848373)/2 is odd.
# g = largest divisor of n-1 less than (n-1)/2.
# n-1 = 2 * m where m is odd. Divisors of n-1: divisors of m, and 2*divisors of m.
# Largest divisor less than (n-1)/2 = m: largest divisor of m less than m, which is m/p
# where p is the smallest prime factor of m.
# m = (3^L + 1848373)/2.
# m mod 3: (3^L + 1848373)/2 mod 3 = (0 + 1848373 mod 3)/... wait, 1848374 mod 3 = 2, so 1848373 mod 3 = 1. (1 + 1)/2 mod 3... this doesn't work simply.
# m = (3^L + 1848373)/2. 3^L = 0 mod 3. 1848373 mod 3 = 1. So numerator = 1 mod 3. m mod 3: m = (0 + 1)/2 = 1/2 mod 3. In mod 3: 2^{-1} = 2. So m = 2 mod 3. NOT divisible by 3.
# m mod 5: 3^L mod 5 = 1 (since 4|L). 1848373 mod 5 = 1848374 mod 5 - 1 = 4 - 1 = 3.
# m = (1 + 3)/2 = 2 mod 5. NOT 0 mod 5.
# m mod 7: 3^L mod 7 = 1 (since 6|L). 1848373 mod 7 = 1848374 mod 7 - 1 = 3 - 1 = 2.
# m = (1+2)/2 mod 7. 3/2 mod 7 = 3*4 = 12 = 5 mod 7. NOT 0.
# m mod 11: 3^L mod 11 = 1 (since 5|L and ord_11(3)=5). 1848373 mod 11 = (1848374-1) mod 11 = (0-1) mod 11 = 10. m = (1+10)/2 mod 11 = 11/2 mod 11 = 0. YES! 11 | m.
#
# Hmm but wait: c = 1848374. c mod 11 = 0 from our earlier computation.
# n-1 = 3^L + c - 1. (n-1) mod 11: 3^L mod 11 = 1 (5|L). c-1 mod 11 = -1 mod 11 = 10.
# (n-1) mod 11 = 11 = 0 mod 11. So 11 | (n-1). And m = (n-1)/2, m mod 11 = 0.
# Smallest prime factor of m: need to check 3, 5, 7, 11...
# m mod 3 = 2 (not 0). m mod 5 = 2 (not 0). m mod 7 = 5 (not 0). m mod 11 = 0. YES.
# So smallest prime factor of m is 11 (assuming nothing between 7 and 11 works).
# g = m/11 (if this is the largest divisor of n-1 less than (n-1)/2 = m).
# Wait: g must be a divisor of n-1, not of m. g < (n-1)/2 = m.
# Divisors of n-1 = 2m: they are d and 2d for d | m.
# Divisors of n-1 less than m: all divisors of m less than m, plus 2d for d | m with 2d < m.
# The largest divisor of n-1 less than m: either m/p (largest proper divisor of m, where p = smallest prime factor of m), or 2*(m/q) for some small prime q of m.
# m/11 vs 2*(m/11): 2*(m/11) = 2m/11. Since m is huge, m/11 > 2m/11? No! m/11 vs 2m/11: m/11 < 2m/11.
# So 2m/11 > m/11.
# But is 2m/11 a divisor of n-1 = 2m? Yes if m/11 is integer (which it is since 11|m).
# 2m/11 < m iff 2/11 < 1, which is true. So 2m/11 is a divisor of n-1 less than m.
# Is there a larger one? We need the largest d | 2m with d < m.
# d can be m/p for prime p | m, or 2m/p for prime p | m (with 2m/p < m iff p > 2, always true).
# So candidates: m/p for primes p | m, and 2m/p for primes p | m with p > 2.
# m/p < 2m/p for p > 2. So 2m/p > m/p.
# Largest: max over prime p | m of 2m/p = 2m/p_min where p_min is smallest prime factor of m.
# And also m/p for large p.
# 2m/p_min vs m/2: 2m/p_min > m/2 iff p_min < 4. So if p_min = 3: 2m/3 > m/2. YES.
# But we showed p_min of m is 11 (for c = 1848374).
# 2m/11 vs m/11: 2m/11 > m/11. And 2m/11 < m since 11 > 2.
# Actually I realize I should also consider whether m itself has factor 2.
# m = (n-1)/2. n-1 is even. m = (n-1)/2. Is m even? Only if 4 | (n-1).
# We showed (n-1) mod 4 = 2, so m is odd. So m doesn't have factor 2.
# Divisors of n-1 = 2m with m odd: d | n-1 means d = d_1 * (1 or 2) where d_1 | m.
# Divisors less than m: {d_1 : d_1 | m, d_1 < m} union {2*d_1 : d_1 | m, 2*d_1 < m}.
# The largest is max(m/p_min, 2*m/p_min) = 2m/p_min (since p_min >= 3 > 2).
# Wait: 2*m/p_min = 2m/11. Is 2m/11 < m? Yes (11 > 2).
# But also m/p_min = m/11. And 2m/11 > m/11.
# But there might be larger divisors: 2m/q for q | m with q small.
# Since p_min = 11, there's no q < 11 dividing m. So 2m/11 is the largest.
#
# Hmm but could there be composite divisors of m between m/11 and m?
# Like m/p for p | m where p < 11? We said p_min = 11, so no.
# What about 2*(m/11) vs m/11? 2*(m/11) = 2m/11.
# And what about divisors like 2m/13 (if 13 | m)? 2m/13 < 2m/11. So 2m/11 is largest.

# So g = 2m/11 = 2(n-1)/(2*11) = (n-1)/11.
# f(n) = n-1-g = n - 1 - (n-1)/11 = (n-1)(1 - 1/11) = 10(n-1)/11.
# f(M+c)/M = 10(M+c-1)/(11M) = 10/11 + 10(c-1)/(11M).
#
# g(c) = (1/L) * floor(L * f(M+c)/M) = (1/L) * floor(L * [10/11 + 10(c-1)/(11M)]).
# = (1/L) * floor(10L/11 + 10L(c-1)/(11M)).
# Since M = 3^L and L = 2025!, 10L(c-1)/(11M) is astronomically small (essentially 0).
# So g(c) = (1/L) * floor(10L/11).
# 10L/11: is L divisible by 11? L = 2025!, so yes. 10L/11 is an integer.
# g(c) = 10/11.

# But wait, I assumed the (1, g, n-1-g) strategy is optimal. Let me verify.
# For n = M+c with M huge: f(n) = n-1-g ≈ 10n/11.
# But 10/11 > 2/3. So this is WORSE than 2/3!
# But 2/3 requires 9 | n, which fails for c not divisible by 9.
# What if there's a triple giving f/M < 10/11?

# Let me try other triples for n not divisible by 9.
# (a, 3a, n-4a): lcm(3a, n-4a). Sum = 4a + (n-4a) = n. But we need 3 terms: (a, 3a, n-4a).
# a + 3a + (n-4a) = n. YES.
# Need a < 3a < n-4a, so n-4a > 3a means n > 7a.
# lcm(a, 3a, n-4a) = lcm(3a, n-4a). gcd(3a, n-4a).
# If a | n: n-4a = n - 4a, gcd(3a, n-4a) = gcd(3a, n) = gcd(3a, n).
# For n = 0 mod a: gcd(3a, n) = a * gcd(3, n/a).
# Hmm this is getting complicated.

# Better: (a, b, n-a-b) with a:b:c in ratio r:s:t where r+s+t divides n.
# For ratio 1:2:6 (sum 9): needs 9|n.
# For ratio 1:3:5 (sum 9): needs 9|n. lcm(a, 3a, 5a) = 15a. n=9a, f=15a=5n/3>n. Worse.
# For ratio 1:2:4 (sum 7): needs 7|n. lcm = 4a. n=7a, f=4a=4n/7.
# For ratio 1:2:8 (sum 11): needs 11|n. lcm = 8a. n=11a, f=8a=8n/11.
# For ratio 1:3:8 (sum 12): needs 12|n. lcm = 24a. Way worse.
# For ratio 1:2:2 -> not distinct.
# For ratio 1:4:6 (sum 11): needs 11|n. lcm(a,4a,6a) = 12a. f=12n/11 > n. Worse.
# For ratio 2:3:4 (sum 9): needs 9|n. lcm(2a,3a,4a) = 12a. n=9a, f=12a=4n/3. Worse.
# For ratio 1:2:10 (sum 13): needs 13|n. lcm = 10a. n=13a, f=10n/13.
# For ratio 1:4:16 (sum 21): lcm = 16a. f = 16n/21.
# For ratio 1:2:12 (sum 15): needs 15|n. lcm = 12a. f = 12n/15 = 4n/5.
# For ratio 2:3:6 (sum 11): needs 11|n. lcm = 6a. n=11a, f=6n/11.
# For ratio 1:5:6 (sum 12): lcm = 30a. n=12a, f=30a. Worse.
# For ratio 3:4:6 (sum 13): lcm = 12a. n=13a, f=12n/13.
# For ratio 1:6:8 (sum 15): lcm = 24a. f = 24n/15. Worse.
# For ratio 2:4:6 -> gcd 2. So (1,2,3) * 2a: lcm = 6a. n = 6*2a = 12a. Hmm wrong.
#   Actually a=2t, b=4t, c=6t. n=12t. lcm=12t. f=12t=n. Terrible.
# For ratio 2:3:10 (sum 15): lcm = 30a. Worse.
# For ratio 3:4:8 (sum 15): lcm = 24a. f=24n/15. Worse.
# For ratio 1:2:6: f = 6a = 2n/3 (best so far, needs 9|n).
# For ratio 1:2:4: f = 4a = 4n/7 ≈ 0.571n (needs 7|n).
# For ratio 1:2:8: f = 8a = 8n/11 ≈ 0.727n (needs 11|n).
# For ratio 1:2:10: f = 10a = 10n/13 ≈ 0.769n (needs 13|n).
# For ratio 2:3:6: f = 6a = 6n/11 ≈ 0.545n (needs 11|n).
# Wait, (2a, 3a, 6a): 2a+3a+6a = 11a = n. lcm(2a, 3a, 6a) = 6a. f = 6a = 6n/11.
# 6/11 ≈ 0.545, which is less than 2/3 ≈ 0.667!
# But this needs 11 | n.

# Actually, for (2a, 3a, 6a): n = 11a. lcm = 6a = 6n/11.
# Is 6n/11 < 2n/3? 6/11 = 0.545 < 0.667. YES! This is better!
# But we need 2a < 3a < 6a (which is true) and all distinct (yes).

# So f(n) <= 6n/11 when 11 | n. Let me verify.
for k in [4,5,6,7]:
    n = 11 * 3**k
    m = 6*n//11
    a = n // 11
    print(f"n={n}, a={a}, triple=({2*a},{3*a},{6*a}), sum={2*a+3*a+6*a}, lcm=6*{a}={6*a}, f_fast=", end="")
    # Verify
    fv = None
    for mm in range(1, m+1):
        divs = []
        for d in range(1, int(mm**0.5)+1):
            if mm % d == 0:
                divs.append(d)
                if d != mm // d:
                    divs.append(mm // d)
        divs.sort()
        nd = len(divs)
        ok = False
        for i in range(nd - 2):
            target = n - divs[i]
            lo, hi = i + 1, nd - 1
            while lo < hi:
                s = divs[lo] + divs[hi]
                if s == target:
                    ok = True
                    break
                elif s < target:
                    lo += 1
                else:
                    hi -= 1
            if ok:
                break
        if ok:
            fv = mm
            break
    print(fv)

# Great! So we can do even better. Let me find the optimal ratio systematically.
# For (a, b, c) with a < b < c = n-a-b:
# lcm(a, b, c). We want to MINIMIZE this.
# If a, b, c are pairwise coprime: lcm = abc. Very large.
# If a | b and a | c: lcm = lcm(b, c).
# Best: maximize common factors.

# The best known results for the "minimum lcm of 3 distinct numbers summing to n":
# Use the triple (a, 2a, n-3a) with lcm = lcm(a, 2a, n-3a).
# Or (2a, 3a, n-5a) with lcm = lcm(2a, 3a, n-5a) = lcm(6a, n-5a).
# Or (a, ka, (n-(k+1)a)) for various k.

# The best approach: (a, b, c) where gcd is large and lcm is small.
# With all three multiples of g: a=gx, b=gy, c=gz, x+y+z = n/g.
# lcm = g*lcm(x,y,z).
# To minimize: lcm(x,y,z) should be small, and g should be large.
# But g | n. For n = M + c with c small: g | c (approximately, since M = 3^L is coprime to most things).

# For n = M + c where M = 3^L and c is not divisible by 9:
# We can't use g = 3^j for large j.
# The best g dividing n = 3^L + c: gcd(3^L + c, c) = gcd(3^L, c) = gcd(c, 3^L) = 3^{v_3(c)}.
# Actually: g | n and g | (n - 3^L) = c. So g | gcd(n, c). And gcd(n, c) = gcd(3^L + c, c) = gcd(3^L, c) = 3^{min(L, v_3(c))}.
# For c not divisible by 3: gcd(n, c) divides gcd(3^L, c) = 1. So g = 1.
# For our c values: c mod 3 = 2 for most of them (1848374, 10162574, 265710644 all have c mod 3 = 2). So g = 1.

# This means the triple must have gcd 1. So lcm(a,b,c) = abc/[gcd stuff].
# The minimum lcm of three distinct integers summing to n with gcd = 1:
# This is a hard optimization problem.

# But wait: we can also have triples where not all divide g.
# Like (2a, 3a, 6a) with sum = 11a: here gcd(2a,3a,6a) = a.
# The ratio 2:3:6 has lcm 6, sum 11. So f/n = 6/11.

# What are all ratios (p:q:r) with p < q < r, gcd(p,q,r)=1, and lcm(p,q,r)/sum(p,q,r) small?

best_ratios = []
for p in range(1, 30):
    for q in range(p+1, 50):
        for r in range(q+1, 100):
            g = math.gcd(p, math.gcd(q, r))
            if g != 1:
                continue
            s = p + q + r
            l = (p * q * r) // math.gcd(p, q) // math.gcd(q // math.gcd(p,q) * p // math.gcd(p,q), r)
            # Actually compute lcm properly
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if ratio < Fraction(2, 3):
                best_ratios.append((ratio, p, q, r, s, l))

best_ratios.sort()
print("\nBest ratios (p,q,r) with lcm/sum < 2/3:")
seen_sums = set()
for ratio, p, q, r, s, l in best_ratios[:30]:
    if s not in seen_sums:
        print(f"  ({p},{q},{r}): sum={s}, lcm={l}, ratio={ratio} = {float(ratio):.6f}")
        seen_sums.add(s)
