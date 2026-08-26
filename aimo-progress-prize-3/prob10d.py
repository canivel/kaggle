from fractions import Fraction

def f_fast(n, limit=None):
    if limit is None:
        limit = 10*n
    for m in range(1, limit + 1):
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

def find_divisor_triple(m, n):
    divs = []
    for d in range(1, m+1):
        if m % d == 0:
            divs.append(d)
    nd = len(divs)
    for i in range(nd):
        for j in range(i+1, nd):
            for k in range(j+1, nd):
                if divs[i] + divs[j] + divs[k] == n:
                    return (divs[i], divs[j], divs[k])
    return None

# Confirmed: f(3^k) = 2*3^(k-1) for k >= 2.
# The divisors are (3^(k-2), 2*3^(k-2), 2*3^(k-1)) for k >= 3.
# Check: 3^(k-2) + 2*3^(k-2) + 2*3^(k-1) = 3^(k-2)(1+2+6) = 9*3^(k-2) = 3^k. YES!
# And m = 2*3^(k-1). Divisors of 2*3^(k-1): 1,2,3,...,3^(k-1),2*3,...,2*3^(k-1).
# 3^(k-2) divides 2*3^(k-1)? Yes (k-2 <= k-1).
# 2*3^(k-2) divides 2*3^(k-1)? Yes.

# For k=2: f(9) = 6, divisors (1, 2, 6). Sum = 9. m=6=2*3.

# So f(M) = f(3^K) = 2*3^(K-1) = 2M/3 where K = 2025!.

# Now for f(M + c):
# n = 3^K + c. We need smallest m with 3 distinct divisors summing to 3^K + c.

# g(c) = (1/2025!) * floor(2025! * f(M+c) / M)
# = (1/K!) * floor(K! * f(3^K + c) / 3^K)  where K = 2025!.

# Since M = 3^K is astronomically large, f(M+c) should be approximately (2/3)*M + something*c.
# Let's study f(3^K + c) / 3^K more carefully.

# For c = 0: f = 2*3^(K-1) = (2/3)*3^K. f/M = 2/3.
# g(0) = (1/K!) * floor(K! * (2/3)) = (1/K!) * floor(2*K!/3).
# Since K! = 2025! is divisible by 3 (in fact by very high powers of 3):
# 2*K!/3 is an integer. So g(0) = 2/3.

# For general c: f(M+c) = f(3^K + c). Let me compute f(3^k + c) for small k and various c.

print("f(3^k + c) for k=4..7, c in interesting values:")
for k in range(4, 8):
    M_k = 3**k
    print(f"\nk={k}, M=3^{k}={M_k}:")
    for c in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3**k]:
        n = M_k + c
        m = f_fast(n, max(5*n, 100000))
        if m is not None:
            trip = find_divisor_triple(m, n) if m < 10000 else "?"
            ratio = Fraction(m, M_k)
            print(f"  c={c}: f({n}) = {m}, f/M = {ratio} = {float(ratio):.6f}, triple={trip}")
        else:
            print(f"  c={c}: NOT FOUND")

# Also compute for multiples of M
print("\n\nf(3^k + c*3^k) = f((1+c)*3^k):")
for k in range(4, 8):
    M_k = 3**k
    for mult in [0, 1, 2, 3, 4, 5]:
        c = mult * M_k
        n = M_k + c
        m = f_fast(n, max(5*n, 100000))
        if m is not None:
            ratio = Fraction(m, M_k)
            print(f"  k={k}, c={mult}M: f({n}) = {m}, f/M = {ratio} = {float(ratio):.6f}")
