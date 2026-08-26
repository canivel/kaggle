"""
Problem 6: Ken's digit sum game - efficient computation.

Key insight from analysis:
- From m >= 4, we can always reach ceil(m/2) by choosing base floor(m/2)+1
  (2-digit representation with leading digit 1).
- This gives us approximately log2(m) moves by repeated halving.
- But can we sometimes do better by reaching a value > ceil(m/2)?

For m, the maximum achievable digit sum is:
  max over all b in [2,m] of digit_sum(m, b)

For 2-digit representations (b > sqrt(m)): digit_sum = m//b + m%b
  Best with q=1: b = floor(m/2)+1, giving ceil(m/2). For even m: m/2. For odd m: (m+1)/2.

But what about multi-digit representations?
For 3+ digits: digit_sum <= (b-1) * num_digits. With b^2 <= m (3+ digits):
  digit_sum <= (b-1)(floor(log_b(m))+1)
  Optimal around b = e, so b=3: digit_sum ≈ 2*log_3(m) = 2*ln(m)/ln(3).
  For m=100: ≈ 2*4.19 = 8.38, actual max ≈ 9 or so.
  But ceil(100/2) = 50. So 2-digit representation is MUCH better.

So the halving strategy dominates for all m >= some threshold.
Let me verify this by computing f(m) and checking if f(m) ≈ log2(m) + const.
"""

import sys
from functools import lru_cache

def digit_sum(m, b):
    s = 0
    while m > 0:
        s += m % b
        m //= b
    return s

def max_achievable(m):
    """Maximum digit sum achievable from m over all valid bases."""
    if m <= 1:
        return 0
    best = 0
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds > best:
            best = ds
        # Optimization: for b > m//2, digit_sum = 1 + (m-b) which decreases as b increases
        # So once we pass b = m//2 + 1, it only gets worse
        if b > m // 2 + 1 and b > 3:
            break
    return best

# Compute f(m) using DP
LIMIT = 10000
f = [0] * (LIMIT + 1)
best_next = [0] * (LIMIT + 1)

for m in range(2, LIMIT + 1):
    best_f = 0
    best_nxt = 0
    # Try all bases
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds >= m or ds < 1:
            continue
        val = 1 + f[ds]
        if val > best_f:
            best_f = val
            best_nxt = ds
    f[m] = best_f
    best_next[m] = best_nxt

# Check f values and look for pattern
print("f(m) for m = 1..30:")
for m in range(1, 31):
    print(f"  f({m}) = {f[m]}")

print()
# Check if f(m) = floor(log2(m)) + something for m = 2^k - 1
import math
for k in range(1, 14):
    m = 2**k - 1
    if m <= LIMIT:
        print(f"  f(2^{k}-1) = f({m}) = {f[m]}, log2({m}) = {math.log2(m):.2f}, floor(log2) = {int(math.log2(m))}")

# Trace optimal path from some values
print("\nOptimal paths:")
for start in [7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191]:
    if start > LIMIT:
        break
    path = [start]
    m = start
    while m > 1:
        best_ds = 0
        best_b = 0
        for b in range(2, m + 1):
            ds = digit_sum(m, b)
            if ds >= m or ds < 1:
                continue
            if 1 + f[ds] == f[m]:
                if ds > best_ds:
                    best_ds = ds
                    best_b = b
        path.append(best_ds)
        m = best_ds
    print(f"  {start}: {' -> '.join(map(str, path))} ({len(path)-1} moves)")

# Now let's think about what happens for VERY large numbers.
# From m, best strategy is to go to ceil(m/2) (roughly).
# f(m) ≈ number of times we can halve before reaching a small number.
#
# But wait - there might be numbers where f is higher than just halving would give.
# Let's check: for m = 2^k - 1, the binary representation is all 1s.
# digit_sum(2^k-1, 2) = k.
# From m via base 2: go to k. Then f(k) more steps.
# Via halving: go to 2^(k-1) - 1 + ... ceil(m/2) = 2^(k-1).
# Hmm, halving 2^k - 1 gives ceil((2^k-1)/2) = 2^(k-1).
# f(2^(k-1)) = ?

# Let's compare strategies for m = 2^k - 1:
# Strategy 1 (base 2): go to k. Total = 1 + f(k).
# Strategy 2 (halving): go to 2^(k-1). Total = 1 + f(2^(k-1)).
# Since 2^(k-1) >> k, halving is better (more moves from a larger number).

# Let's see the actual values:
print("\nComparison for 2^k - 1:")
for k in range(1, 14):
    m = 2**k - 1
    if m > LIMIT:
        break
    half = (m + 1) // 2  # ceil(m/2) = 2^(k-1) for odd m=2^k-1
    print(f"  m=2^{k}-1={m}: f(m)={f[m]}, via halving to {half}: 1+f({half})={1+f[half]}, via base2 to {k}: 1+f({k})={1+f[k]}")
