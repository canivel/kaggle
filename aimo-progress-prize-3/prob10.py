# Problem 10:
# n-Norwegian = positive integer with 3 distinct divisors summing to n.
# f(n) = smallest n-Norwegian number.
# M = 3^(2025!).
# g(c) = (1/2025!) * floor(2025! * f(M+c) / M).
# Compute g(0) + g(4M) + g(1848374) + g(10162574) + g(265710644) + g(44636594) = p/q.
# Find (p+q) mod 99991.

# First, let's understand f(n).
# An n-Norwegian number is a positive integer m that has 3 distinct divisors d1 < d2 < d3
# with d1 + d2 + d3 = n.
# f(n) = smallest such m.

# For m to have at least 3 distinct divisors, m >= 4 (divisors 1,2,4).
# The 3 distinct divisors of m must sum to n.
# We want the SMALLEST m.

# The smallest m with 3 divisors summing to n:
# We need d1, d2, d3 | m, d1+d2+d3 = n, all distinct.
# To minimize m, we want to pick divisors cleverly.

# One approach: try d1=1, d2=d, d3=n-1-d for various d.
# Then we need d | m and (n-1-d) | m, so m = lcm(d, n-1-d) at minimum.
# We want to minimize lcm(d, n-1-d) subject to 1 < d < n-1-d (for distinctness).
# i.e., d >= 2 and d < (n-1)/2.

# Also d1 can be other values, not just 1.

# For large n (like M + c where M = 3^(2025!)), we need to think about asymptotics.

# Let's first compute f(n) for small n to understand the pattern.
def f(n):
    """Find smallest n-Norwegian number."""
    # Try all possible m starting from 1
    for m in range(1, 10*n + 1):
        divs = [d for d in range(1, m+1) if m % d == 0]
        # Check if any 3 distinct divisors sum to n
        from itertools import combinations
        for combo in combinations(divs, 3):
            if sum(combo) == n:
                return m
    return None

# This is slow, let's optimize
def f_fast(n):
    """Find smallest n-Norwegian number."""
    for m in range(1, 10*n + 1):
        divs = []
        for d in range(1, int(m**0.5)+1):
            if m % d == 0:
                divs.append(d)
                if d != m // d:
                    divs.append(m // d)
        divs.sort()
        # Check if any 3 distinct divisors sum to n
        # Use two-pointer for each fixed first element
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

print("f(n) for small n:")
for n in range(3, 100):
    val = f_fast(n)
    if val is not None:
        print(f"f({n}) = {val}", end="  ")
        if n % 10 == 0:
            print()
print()
