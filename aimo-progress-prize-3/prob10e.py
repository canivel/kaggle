from fractions import Fraction

# Key findings:
# f((1+j)*3^K) = (2/3)(1+j)*3^K = (2/3)*n for n = (1+j)*3^K.
# Actually more precisely: f(c*3^K) = (2/3)*c*3^K when c is a positive integer not divisible by 3?
# Wait: f(3^K) = (2/3)*3^K (c=1), f(2*3^K) = (4/3)*3^K = (2/3)*2*3^K (c=2),
# f(3*3^K) = f(3^{K+1}) = (2/3)*3^{K+1} = 2*3^K (c=3). So f(c*3^K) = (2/3)*c*3^K for c >= 1.
#
# f(n) = 2n/3 when n is a multiple of 3? Let me verify:
# f(6) = 6. Wait, 6 is 2*3, and 2*6/3 = 4, but f(6)=6. So NO!
# f(9) = 6 = 2*9/3. YES.
# f(12) = 12. 2*12/3 = 8. NO, f(12)=12.
#
# Hmm, f(n) = 2n/3 only when n is a POWER of 3 (times some coefficient)?
# Actually from the data: f(n) for n = multiple of 3:
# f(3) = DNF, f(6) = 6, f(9) = 6, f(12) = 12, f(15) = 12, f(18) = 12, f(21) = 12,
# f(24) = 24, f(27) = 18, f(30) = 18, ...
# f(9) = 6 = 2/3 * 9. f(27) = 18 = 2/3 * 27. f(81) = 54 = 2/3 * 81.
# These are all powers of 3!
# f(6) = 6 = 6. 2*6/3 = 4. f(6) != 2*6/3.

# So f(3^K) = 2*3^{K-1} for K >= 2.
# And f(c*3^K) where c is not a power of 3: f/M ratios were 4/3, 2, 8/3, 10/3, 4 for c'=1..5.
# c'=1: f(2*M) = (4/3)*M. So f(2*3^K) = (4/3)*3^K.
# c'=2: f(3*M) = 2*M. f(3*3^K) = f(3^{K+1}) = 2*3^K. And 2*3^K/3^{K+1} = 2/3. Consistent!
# c'=3: f(4*M) = (8/3)*M. f(4*3^K) = (8/3)*3^K.
# c'=4: f(5*M) = (10/3)*M. f(5*3^K) = (10/3)*3^K.
# c'=5: f(6*M) = 4*M. f(6*3^K) = 4*3^K.

# Pattern: f(c*3^K) = (2c/3)*3^K? Check:
# c=1: 2*1/3 * 3^K = (2/3)*3^K. YES.
# c=2: 2*2/3 * 3^K = (4/3)*3^K. YES.
# c=3: 2*3/3 * 3^K = 2*3^K. YES (and = f(3^{K+1}) = 2/3*3^{K+1} = 2*3^K).
# c=4: 8/3*3^K. YES.
# c=5: 10/3*3^K. YES.
# c=6: 4*3^K. And f(6*3^K) = f(2*3^{K+1}) = (4/3)*3^{K+1} = 4*3^K. YES!

# So f(c * 3^K) = (2c/3) * 3^K for large K and positive integer c.
# i.e., f(n) = 2n/3 when n is a multiple of 3^K for large K? Not quite.
# f(c*3^K) = 2c/3 * 3^K = 2c * 3^{K-1}.
# For this to equal 2n/3: 2c*3^{K-1} = 2c*3^K/3. YES.
# So f(n) = 2n/3 when n is divisible by a large power of 3. That's the key!

# The divisors used: (c*3^{K-2}, 2c*3^{K-2}, 2c*3^{K-1}) sum to c*3^{K-2}(1+2+6) = 9c*3^{K-2} = c*3^K.
# And m = 2c*3^{K-1}. These all divide m = 2c*3^{K-1}:
# c*3^{K-2} | 2c*3^{K-1}? Need 3^{K-2} | 2*3^{K-1} and c | 2c. Yes!
# 2c*3^{K-2} | 2c*3^{K-1}? Yes!
# So this works for any c.

# Now for the g function:
# g(c) = (1/K!) * floor(K! * f(M+c) / M) where K = 2025! and M = 3^K.
#
# For c = 0: f(M) = 2M/3. g(0) = (1/K!) * floor(K! * 2/3) = 2/3 (since K! div by 3).
#
# For c = j*M (where j is a positive integer):
# f(M + j*M) = f((1+j)*M) = 2(1+j)M/3.
# g(j*M) = (1/K!) * floor(K! * 2(1+j)/3) = 2(1+j)/3.
# Since K! is divisible by 3, this is exact: g(j*M) = 2(1+j)/3.
#
# For c = 4M: g(4M) = 2*5/3 = 10/3.
#
# For c NOT a multiple of M (like c = 1848374):
# n = M + c = 3^K + c where c is small compared to M.
# f(n) = f(3^K + c). For small c (compared to M), what is f?
# From the data above, f(3^K + c) varies a lot with c and doesn't simply equal 2n/3.
# But g(c) involves floor(K! * f(M+c)/M). Since K! is huge and f/M is approximately 2/3 + small correction,
# the floor function picks up the correction to high precision.

# Wait, let me re-read the problem statement more carefully.
# g(c) = (1/2025!) * floor(2025! * f(M+c) / M).
# Note: 2025! is a specific fixed number. M = 3^(2025!).

# For c = 0: f(M)/M = 2/3 exactly.
# g(0) = (1/2025!) * floor(2025! * 2/3) = 2/3 (since 3 | 2025!).

# For c = 4M: f(5M)/M = 10/3.
# g(4M) = (1/2025!) * floor(2025! * 10/3) = 10/3.

# For c NOT a multiple of M: f(M+c)/M = 2/3 + epsilon where epsilon depends on c and K.
# The question is: does g(c) simplify for the specific values given?

# The given c values: 0, 4M, 1848374, 10162574, 265710644, 44636594.
# For c = 4M: answered above.
# For the others: c is a specific fixed integer.

# For fixed integer c and M = 3^K with K = 2025! very large:
# n = 3^K + c. Since c is fixed and 3^K is enormous, n/3 is not an integer
# (since c is not divisible by 3 in general).

# To find f(3^K + c) for large K and fixed small c:
# Strategy: use divisors 1, d, n-1-d. n-1 = 3^K + c - 1.
# Want largest g | (n-1) with g < (n-1)/2.
# n-1 = 3^K + (c-1).
# Hmm, actually f(n) is the smallest m, not just from the (1, d, n-1-d) strategy.
# Let me think about what happens for general fixed c.

# For LARGE 3^K and fixed c, consider n = 3^K + c.
# If c is even: n is odd. n-1 = 3^K + c - 1 is even.
# If c is odd: n is even. n-1 is odd.

# Let me look at the pattern more carefully for moderate K.
# For K = 7 (M = 2187):
# f(2187 + 1) = 1456, f/M = 1456/2187
# f(2187 + 2) = 1194, f/M = 1194/2187

# These don't seem to converge to a nice fraction as K grows.
# Let me check if f(3^k + c) / 3^k converges for fixed c as k -> infinity.

for c in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    print(f"\nc = {c}:")
    for k in range(4, 9):
        M_k = 3**k
        n = M_k + c
        # Find f(n)
        found_m = None
        for m in range(1, 5*n + 1):
            divs = []
            for d in range(1, int(m**0.5)+1):
                if m % d == 0:
                    divs.append(d)
                    if d != m // d:
                        divs.append(m // d)
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
                found_m = m
                break
        if found_m:
            ratio = Fraction(found_m, M_k)
            print(f"  k={k}: f({n})={found_m}, f/M={ratio} = {float(ratio):.8f}")
