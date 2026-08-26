from fractions import Fraction
import math

# Wait! For c = 4M: n = 5M = 5*3^L.
# Sum 25 divides 5*3^L (25 = 5^2, and 5 | 5*3^L, but 25 | 5*3^L only if 5 | 3^L. 3^L mod 5: 3^4=1 mod 5, 4|L, so 3^L=1 mod 5. So 5*3^L = 5 mod 25. 25 does NOT divide 5*3^L.)
# Hmm wait: 5*3^L mod 25 = 5*(3^L mod 5) = 5*1 = 5 != 0. So 25 does NOT divide 5*3^L!
# My pow3_mod_s function was wrong for this case.

# Let me reconsider. For c = 4M: n = M + 4M = 5M. pow3_mod_s(s) computes 3^L mod s.
# Then we check if (3^L + c) % s == 0, i.e., (3^L + 4*3^L) % s == 0, i.e., 5*3^L % s == 0.
# But my function computes 3^L mod s, not 5*3^L mod s!
# The check should be: (pow3_mod_s(s) + c_val) % s == 0.
# For c = 4M: c_val = 4*3^L. So (3^L + 4*3^L) % s = 5*3^L % s.
# But c_val is given as "4M" which is 4*3^L, a huge number. We can't compute c_val % s directly...
# unless we know that c_val = 4*3^L = 4*M.

# Actually wait: the problem says g(4M), where c = 4M. So c = 4 * 3^L.
# n = M + c = 3^L + 4*3^L = 5*3^L.
# s | n = 5*3^L. So s = 5^a * 3^b for non-negative a, b.
# For sum = 25 = 5^2: 25 | 5*3^L? 5*3^L / 25 = 3^L/5. Not integer since gcd(3^L, 5)=1.
# So 25 does NOT divide n. Good, my earlier code was buggy for this case.

# For c = 4M, the sums that work are exactly the divisors of 5*3^L.
# Divisors: 3^i * 5^j where i <= L and j <= 1.
# So s in {1, 3, 5, 9, 15, 27, 45, 81, 135, 243, ...} union {multiples of just 3 and 5 with 5^j, j<=1}.
# Actually 5*3^L = 5 * 3^L. Divisors are 3^i and 5*3^i for 0 <= i <= L.
# So sums can be: 3, 5, 9, 15, 27, 45, 81, 135, ...

# For sum = 9: triple (1,2,6), ratio 2/3. f = 2n/3 = 2*5*3^L/3 = 10*3^{L-1}.
# g(4M) = (1/L) * floor(L * f(5M) / M) = (1/L)*floor(L * 10*3^{L-1}/3^L) = (1/L)*floor(10L/3).
# 10L/3: L = 2025!. 3 | L. 10L/3 integer. g(4M) = 10/3.

# OK so for c = 4M, g(4M) = 10/3 is confirmed.

# Now let me reconsider the fixed c values.
# The key issue with my search was limited range of sums.
# Let me extend to larger sums.

def pow3_mod_s(s):
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

# Precompute best triples for each sum up to 1000
best_for_sum = {}
for s in range(3, 1000):
    best = None
    # Only check ratios (p,q,r) with p+q+r = s
    for p in range(1, s//3 + 1):
        for q in range(p+1, (s-p)//2 + 1):
            r = s - p - q
            if r <= q:
                continue
            l = math.lcm(p, math.lcm(q, r))
            ratio = Fraction(l, s)
            if best is None or ratio < best[0]:
                best = (ratio, (p, q, r), l)
    if best:
        best_for_sum[s] = best

# For fixed c values, search for best ratio
c_values_fixed = [1848374, 10162574, 265710644, 44636594]

for c_val in c_values_fixed:
    best_ratio = Fraction(10**9)
    best_info = None
    for s in range(3, 1000):
        if s not in best_for_sum:
            continue
        ratio, triple, l = best_for_sum[s]
        if ratio >= best_ratio:
            continue
        r3L = pow3_mod_s(s)
        if (r3L + c_val) % s == 0:
            best_ratio = ratio
            best_info = (s, triple, l, ratio)
    print(f"c = {c_val}: BEST sum={best_info[0]}, triple={best_info[1]}, lcm={best_info[2]}, ratio={best_info[3]} = {float(best_info[3]):.8f}")

# Now: g(c) = (1/L) * floor(L * ratio) where ratio = lcm/sum * (n/M) ???
# Wait no. f(n) = lcm(p,q,r) * a where a = n/s.
# f(n)/M = (lcm/s) * (n/M) = ratio * (M+c)/M = ratio * (1 + c/M).
# For large M and fixed c: (M+c)/M ≈ 1 + c/M ≈ 1.
# So f(n)/M ≈ ratio.
# More precisely: f(n) = lcm * a = lcm * (M+c)/s.
# f(n)/M = lcm*(M+c)/(s*M) = (lcm/s)(1 + c/M).
# g(c) = (1/L)*floor(L * lcm*(M+c)/(s*M)).
# = (1/L)*floor(L*lcm/s + L*lcm*c/(s*M)).
# Since M = 3^L >> L*lcm*c/s (because 3^L grows much faster than L), the second term is 0.
# So g(c) = (1/L)*floor(L*lcm/s).
# = lcm/s if L*lcm/s is an integer, i.e., if s | L*lcm.
# L = 2025!. s divides L*lcm iff s/gcd(s,lcm) divides L.
# Since lcm is the lcm of the triple, and the triple has sum s:
# s/gcd(s,lcm). Hmm.

# Actually: g(c) = (1/L)*floor(L * ratio) where ratio = lcm/s.
# L*ratio = L*lcm/s. For this to be an integer, we need s | L*lcm.
# L = 2025!. Any factor <= 2025 divides L.
# For s < 1000: s < 2025, so s | 2025!. Then L*lcm/s = (L/s)*lcm = integer * lcm. So yes, it's integer.
# Therefore g(c) = lcm/s = ratio (exactly).

# But wait: we assumed f(n) = lcm * a is optimal. We showed it's an upper bound.
# Could there be a non-ratio triple that does better?
# For n = 3^L + c with L very large and c fixed, the answer is no:
# any triple (a,b,c) with a+b+c=n and lcm(a,b,c) < ratio*n would contradict
# our search (we checked all ratio triples up to sum 1000).
# But could a non-ratio triple work? Like (a, b, n-a-b) where a, b are NOT in ratio?
# lcm(a, b, n-a-b). For this to be small, a, b, n-a-b need large common factors.
# If gcd(a,b) = g and g | n: a = gx, b = gy, n-a-b = gz where x+y+z = n/g.
# lcm = g*lcm(x,y,z). So f(n) >= g*f_ratio(n/g) where f_ratio is the ratio-triple min.
# Hmm this doesn't directly help.

# Actually: any triple (a,b,c) with a+b+c = n can be written as (gx, gy, gz) where
# g = gcd(a,b,c) and x+y+z = n/g with gcd(x,y,z)=1.
# lcm(a,b,c) = g*lcm(x,y,z). n/g must be an integer.
# So f(n) = min over g|n of g * f_reduced(n/g) where f_reduced(m) = min over coprime (x,y,z)
# with x+y+z=m of lcm(x,y,z).
# f_reduced(m)/m = ratio = the best ratio for sum m with coprime components.
# f(n)/n = min over g|n of (g/n) * f_reduced(n/g) = min over g|n of f_reduced(n/g)/(n/g).
# So f(n)/n = min over d|n of f_reduced(d)/d = min over d|n of best_ratio(d).

# For n = M = 3^L: d | 3^L means d = 3^j.
# d = 9: best_ratio = 2/3. d = 27: best_ratio = 2/3. d = 3: no triple of 3 distinct.
# So f(M)/M = 2/3.

# For n = M + c: d | (M+c). The divisors of M+c that are worth checking:
# we need f_reduced(d)/d to be small. The best possible ratios are:
# 6/11 (d=11), 4/7 (d=7), 10/17 (d=17), 3/5 (d=10), etc.
# So we need divisors of (M+c) that are these specific sums.
# d | (M + c) means d | (3^L + c).
# This is the same as s | n condition from before!

# So the analysis is consistent. f(n)/n = min over d|n of best_ratio(d).
# And g(c) = f(M+c)/M ≈ f(M+c)/(M+c) * (M+c)/M ≈ best_ratio (since (M+c)/M ≈ 1).

# More precisely: g(c) = (1/L)*floor(L * f(M+c)/M).
# f(M+c) = best_ratio * (M+c).
# f(M+c)/M = best_ratio * (1 + c/M).
# L * f(M+c)/M = L * best_ratio + L * best_ratio * c/M.
# L * best_ratio * c / M = L * best_ratio * c / 3^L ≈ 0 (exponentially small).
# So g(c) = (1/L) * floor(L * best_ratio) = best_ratio (since L*best_ratio is integer as argued).

# Summary:
# g(0) = 2/3 (from d=9)
# g(4M) = 10/3 (from f(5M) = 2*5M/3, ratio 2/3 applied to 5M)
# Wait, for c = 4M: n = 5M = 5*3^L. Divisors of 5*3^L include 9 (gives ratio 2/3).
# f(5M) = 2/3 * 5M = 10M/3.
# g(4M) = (1/L)*floor(L * 10M/(3M)) = (1/L)*floor(10L/3) = 10/3.

# For fixed c: g(c) = best_ratio where best_ratio is min over d|(3^L+c), d>=3 of best_ratio(d).
# From our search:
# g(1848374) = 16/25
# g(10162574) = 30/47
# g(265710644) = 64/97
# g(44636594) = 110/167

# But wait - I need to verify these are truly optimal. Could there be a better ratio for a larger sum that I missed?

# Actually, the families of best ratios are:
# (2k, 3k, 6k) -> sum = 11k, ratio = 6/11. Need 11k | n.
# (k, 2k, 4k) -> sum = 7k, ratio = 4/7. Need 7k | n.
# (2k, 5k, 10k) -> sum = 17k, ratio = 10/17. Need 17k | n.
# etc.

# For 6/11: need 11 | n = 3^L + c. 3^L = 1 mod 11 (since 5 | L and ord_11(3)=5).
# So need c = -1 = 10 mod 11. Check:
for c_val in c_values_fixed:
    print(f"c={c_val}: c mod 11 = {c_val % 11}", end="")
    if c_val % 11 == 10:
        print(" -> 6/11 AVAILABLE!")
    else:
        print()

# For 4/7: need 7 | n. 3^L = 1 mod 7. Need c = -1 = 6 mod 7.
for c_val in c_values_fixed:
    print(f"c={c_val}: c mod 7 = {c_val % 7}", end="")
    if c_val % 7 == 6:
        print(" -> 4/7 AVAILABLE!")
    else:
        print()

print()
print("Results:")
print(f"g(0) = 2/3")
print(f"g(4M) = 10/3")

results = {}
for c_val in c_values_fixed:
    best_ratio = Fraction(10**9)
    for s in range(3, 1000):
        if s not in best_for_sum:
            continue
        ratio = best_for_sum[s][0]
        if ratio >= best_ratio:
            continue
        r3L = pow3_mod_s(s)
        if (r3L + c_val) % s == 0:
            best_ratio = ratio
    results[c_val] = best_ratio
    print(f"g({c_val}) = {best_ratio}")

# Compute g(0) + g(4M) + g(1848374) + g(10162574) + g(265710644) + g(44636594)
total = Fraction(2, 3) + Fraction(10, 3) + results[1848374] + results[10162574] + results[265710644] + results[44636594]
print(f"\nTotal = {total} = {float(total):.10f}")
p, q = total.numerator, total.denominator
print(f"p = {p}, q = {q}")
print(f"p + q = {p + q}")
print(f"(p + q) mod 99991 = {(p + q) % 99991}")
