"""
Problem 6: Ken's digit sum game.
Starting with n, repeatedly choose base b (2 <= b <= m) and replace m with sum of base-b digits.
Find M = max moves over all 1 <= n <= 10^(10^5), then M mod 10^5.

Key insight: The digit sum in base b of m is at most (b-1) * (1 + log_b(m)).
To maximize moves, we want each step to reduce m as slowly as possible.

The sum of digits of m in base b is: m mod b + floor(m/b) mod b + floor(m/b^2) mod b + ...
This equals m - (b-1) * sum_{i>=1} floor(m/b^i).
So digit_sum(m, b) = m - (b-1) * (m - digit_sum(m,b)) / ... no that's circular.

Actually: digit_sum(m, b) ≡ m (mod b-1).
And digit_sum(m, b) <= m, with equality only if m < b (single digit).

If m >= b, then digit_sum(m, b) <= m/b * (b-1) + (b-1) = (b-1)(m/b + 1) < m for m >= b.

To maximize moves, at each step we want the result to be as large as possible.
For a given m, to maximize the digit sum in some base b, we should choose b = 2.
Wait, not necessarily. Let me think...

For m, choosing base b=m gives digit sum = 1 (since m = 1*m + 0).
Wait, m in base m is "10", so digit sum = 1. That's terrible.

Choosing b=m-1: m = 1*(m-1) + 1, so digit sum = 1+1 = 2.
Choosing b=2: digit sum can be up to log_2(m) which is much larger.

So to maximize the number of moves, we want each step to produce the largest possible
output. The largest possible output from m is achieved by choosing base 2,
giving at most floor(log2(m)) + 1 (if m = 2^k - 1, all 1s in binary).

But actually we should think about this more carefully for the full chain.

Let's think about the maximum number of moves from m, call it f(m).
f(1) = 0.
f(m) = 1 + max_{2<=b<=m} f(digit_sum(m, b)) for m >= 2.

For the chain to be as long as possible:
- Start with a huge n (up to 10^(10^5))
- Each move, pick base to get the largest possible next value
- The next value from base b is the digit sum of m in base b

The maximum digit sum of m in base b is achieved when all digits are b-1.
If m has k digits in base b, digit sum <= (b-1)*k.
k = floor(log_b(m)) + 1.

For large m, the digit sum in base 2 is at most floor(log2(m)) + 1.
For a number with d decimal digits, this is about 3.32*d.

Step 1: n ~ 10^(10^5), so d ~ 10^5. In base 2, digit sum <= ~3.32 * 10^5 ~ 332193.
Actually, more precisely: log2(10^(10^5)) = 10^5 * log2(10) ≈ 10^5 * 3.32193 ≈ 332193.
So after one move (base 2), we get at most ~332193.

But wait - we could choose base b > 2 and get a larger digit sum!
If m = 10^(10^5) and we choose base b where b is small, say b=2,
digit sum of 10^(10^5) in base 2 has about 332193 bits, each 0 or 1,
so max digit sum in base 2 = 332194 (number of bits).

If we choose b = 3: digit sum <= 2 * (1 + log_3(10^(10^5))) ≈ 2 * 10^5/log10(3) * ...
Hmm let me compute: log_3(10^(10^5)) = 10^5 / log10(3) ≈ 10^5 / 0.4771 ≈ 209590.
Digit sum <= 2 * 209591 ≈ 419182. That's LARGER than base 2!

So base 3 can give larger digit sums than base 2 for large numbers.

Generally, for base b: max digit sum ≈ (b-1) * log_b(m) = (b-1) * ln(m)/ln(b).
Maximized when d/db [(b-1)/ln(b)] = 0.
[ln(b) - (b-1)/b] / ln(b)^2 = 0
ln(b) = (b-1)/b = 1 - 1/b
For b=3: ln(3) ≈ 1.0986, 1-1/3 ≈ 0.6667. ln(b) > 1-1/b, so...
Actually d/db [(b-1)/ln(b)] = [ln(b) - (b-1)/b] / (ln(b))^2
At b=2: [0.693 - 0.5] / 0.693^2 = 0.193/0.480 > 0. Increasing.
At b=3: [1.099 - 0.667] / 1.099^2 = 0.432/1.207 > 0. Still increasing.
At b=10: [2.303 - 0.9] / 2.303^2 = 1.403/5.303 > 0. Still increasing!
Actually (b-1)/ln(b) → ∞ as b → ∞, so there's no maximum!

Wait, but we're constrained by b <= m. And the digit sum must actually be achievable.

For base b, the digit sum of m is exactly:
sum of digits of m in base b.

The maximum digit sum for a number < b^k is (b-1)*k.
So for m with k digits in base b (k = floor(log_b(m)) + 1),
the max digit sum is (b-1)*k.

But we don't get to choose m's representation - it's fixed!
The digit sum depends on what m actually is.

For our problem: n can be ANY number from 1 to 10^(10^5).
So we get to choose n to maximize the number of moves.

For the first move: we want to choose n and base b to maximize the digit sum.
We can choose n to have all digits equal to b-1 in base b.
n = (b-1) * (b^(k-1) + b^(k-2) + ... + 1) = b^k - 1.

So n = b^k - 1 gives digit sum (b-1)*k.

We need n <= 10^(10^5), so b^k - 1 <= 10^(10^5),
meaning k <= 10^5 * log10(b) / 1 ... wait:
k * log10(b) <= 10^5, so k <= 10^5 / log10(b).

digit sum = (b-1) * k = (b-1) * floor(10^5 / log10(b)).

Let me find the optimal b.
"""

import math

# For the first step: maximize (b-1) * floor(10^5 / log10(b))
# But actually we can be more precise: n <= 10^(10^5) means n has at most 10^5+1 decimal digits
# n = b^k - 1 <= 10^(10^5) means k*log10(b) <= 10^5 (approximately)
# k <= floor(10^5 * ln(10) / ln(b)) if we want b^k <= 10^(10^5) + 1

L = 10**5  # the exponent

best_ds = 0
best_b = 0
best_k = 0

for b in range(2, 1000000):
    # k = floor(L * ln(10) / ln(b))
    k = int(L * math.log(10) / math.log(b))
    # But we need b^k <= 10^L, equivalently k * log_b(10^L) ...
    # Actually k * ln(b) <= L * ln(10)
    # k <= L * ln(10) / ln(b)
    if k < 1:
        break
    ds = (b - 1) * k
    if ds > best_ds:
        best_ds = ds
        best_b = b
        best_k = k
    # Also try k+1 if b^(k+1) <= 10^L
    # k+1 might overflow, so skip for safety

print(f"Optimal first step: base b={best_b}, k={best_k}, digit_sum = {best_ds}")
print(f"  (b-1)*k = {(best_b-1)*best_k}")

# Hmm wait, the optimal b might be very large. Let me think again.
# (b-1) * L * ln(10) / ln(b) as a function of b.
# d/db [(b-1)/ln(b)] = [ln(b) - (b-1)/b] / ln(b)^2
# This is always positive for b >= 2, so the function is always increasing.
# So the optimal b is as large as possible!
#
# But b <= m = n, and we need n <= 10^L.
# If b = n, then n in base n is "10", digit sum = 1. Useless.
# If b = n-1, then n in base n-1: n = 1*(n-1) + 1, digit sum = 2. Also bad.
# So we need b << n for a useful digit sum.
#
# The formula (b-1) * floor(L*ln10/ln(b)) works when n = b^k - 1.
# As b → ∞, L*ln10/ln(b) → 0, so eventually k = 1 and ds = b-1.
# For k=1: b-1 <= 10^L, so max ds = 10^L - 1. But then we just go from n=10^L to n-1...
# Wait no. k=1 means n has 1 digit in base b, which means n < b. Digit sum of n in base b is just n.
# So that's trivial and doesn't help - we go from n to n (no reduction).
# Actually n < b means digit sum = n, no change. We need n >= b for a reduction.
# So for k=2: n = (b-1)*b + (b-1) = b^2-1. Digit sum = 2(b-1).
# Need b^2 - 1 <= 10^L, so b <= 10^(L/2).
# ds = 2*(10^(L/2) - 1) ≈ 2*10^(L/2).
#
# For general k: n = b^k - 1, b = 10^(L/k), ds = (10^(L/k)-1)*k.
# As k→1: ds ≈ 10^L. As k→∞: ds ≈ k*10^(L/k) → ∞? No, 10^(L/k) → 1 so ds → k.
# The max of k * 10^(L/k): take log: ln(k) + L/k * ln(10).
# d/dk = 1/k - L*ln(10)/k^2 = 0 → k = L*ln(10) ≈ 230259.
# At k = L*ln10: ds ≈ L*ln10 * 10^(1/ln10) = L*ln10 * e ≈ L*6.26...
# Hmm wait, 10^(L/k) = 10^(1/ln10) = e^(ln10/ln10) = e.
# So ds ≈ k * e = L * ln(10) * e ≈ 10^5 * 2.30259 * 2.71828 ≈ 10^5 * 6.2596 ≈ 625960.
# And b = 10^(L/k) = 10^(1/ln10) = e^1 ≈ 2.718. Since b must be integer, try b=3.
#
# For b=3: k = floor(L*ln10/ln3) = floor(100000 * 2.30259/1.09861) = floor(209590.5) = 209590
# ds = 2 * 209590 = 419180.
#
# For b=2: k = floor(100000 * ln10/ln2) = floor(332192.8) = 332192
# ds = 1 * 332192 = 332192.
#
# b=3 gives 419180 > 332192 for b=2. As predicted.
#
# Let me try more values around the optimum.

print("\nDigit sum (b-1)*k for various b:")
for b in [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 50, 100]:
    k = int(L * math.log(10) / math.log(b))
    ds = (b-1) * k
    print(f"  b={b}: k={k}, ds={ds}")

# Now the full problem: we need to compute the CHAIN of moves.
# After step 1: we go from n ≈ 10^(10^5) to some value ≈ 6*10^5.
# After step 2: we go from ≈ 6*10^5 to some smaller value.
# And so on until we reach 1.
#
# Let me compute this recursively for reasonable-sized numbers.

# f(m) = max number of moves starting from m
# We compute f greedily: at each step, choose the base that maximizes the next value.

def max_digit_sum(m):
    """Find the base b that maximizes digit_sum(m, b), return (best_digit_sum, best_b)."""
    best = (0, 0)
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds > best[0]:
            best = (ds, b)
        if ds == 1:  # can't do better if we already tried enough
            pass
        # optimization: for b > sqrt(m), digit sum = q + r where m = q*b + r
        # and q < b, r < b, so ds = q + r = m // b + m % b
        # this is maximized at the boundary cases
    return best

def digit_sum(m, b):
    s = 0
    while m > 0:
        s += m % b
        m //= b
    return s

# For the chain, we need to be smart. Let me first compute f(m) for small m.
# Then figure out the pattern.

# Actually, let's think about it differently.
# The maximum chain length from a value V is:
# f(V) = 1 + max_{b} f(digit_sum(V, b))
#
# For very large V (like 10^100000), we can't enumerate all bases.
# But for step 1, we CHOOSE n, so we can pick n = b^k - 1 for optimal b,k.
# Then digit_sum(n, b) = (b-1)*k which is about 6.26 * 10^5.
#
# After that, the values are small enough to compute.

# Let me compute f(m) for m up to about 10^6 using dynamic programming.
# Actually that's too big. Let me think about the recursion.
#
# After 2 steps, the value is much smaller. Let me estimate:
# Step 1: ~10^(10^5) → ~6*10^5
# Step 2: ~6*10^5: in base 3, about 2 * log_3(6*10^5) ≈ 2 * 12.1 ≈ 24
# In base b optimizing: (b-1) * log_b(6*10^5).
# ln(6*10^5) ≈ 13.3, optimal at b≈e: ds ≈ e * 13.3 ≈ 36.
# Trying b=3: k ≈ 13.3/1.1 ≈ 12, ds ≈ 24. b=4: k≈9.6, ds≈29. b=7: k≈6.8, ds≈41.
# Actually let me just compute for specific values.

print("\n--- Computing max digit sums for moderate values ---")
for m in [419180, 332192, 625960, 500000, 200000, 100000, 50000, 10000, 1000, 100]:
    best_ds = 0
    best_b = 0
    # Only need to check bases up to m
    # For large m, the optimal base is around e^(ln(m)/W(ln(m))) or something.
    # But brute force for bases 2..min(m, some_limit) + spot checks
    for b in range(2, min(m+1, 100000)):
        ds = digit_sum(m, b)
        if ds > best_ds:
            best_ds = ds
            best_b = b
    print(f"  m={m}: best digit_sum={best_ds} at base {best_b}")
