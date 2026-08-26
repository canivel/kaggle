"""
Problem 6 deeper analysis.

Key observation: for m, choosing base b = floor(m/2) + 1 (when m >= 4):
m = 1 * b + (m - b) = 1 * b + (m - floor(m/2) - 1)
If m is even: b = m/2 + 1, remainder = m - m/2 - 1 = m/2 - 1. digit_sum = 1 + m/2 - 1 = m/2.
If m is odd: b = (m-1)/2 + 1 = (m+1)/2, remainder = m - (m+1)/2 = (m-1)/2. digit_sum = 1 + (m-1)/2 = (m+1)/2.

So from m we can always reach approximately m/2. This gives ~log2(m) moves just from halving.

But we saw digit_sum=m/2 for even m. Can we do better?

Wait, the digit_sum(m, b) for b = m/k + 1 (approximately) when m = (k-1)*b + r:
if we choose b so that m has exactly 2 digits in base b, i.e., b > sqrt(m),
then m = q*b + r where q = m // b, r = m % b, and digit_sum = q + r.
q + r = m // b + m % b.

For 2-digit representations: m = q*b + r, ds = q + r.
Since m = q*b + r, we have r = m - q*b, so ds = q + m - q*b = m - q*(b-1).
To maximize: minimize q*(b-1). Since q = m // b, and b > sqrt(m) ensures q < b.
If q = 1: b > m/2, so b ranges from ceil(m/2)+1 to m-1 (for 2-digit).
  Wait, q=1 when b > m/2 (assuming m >= 4). Then ds = 1 + (m - b).
  Max when b is minimized: b = floor(m/2) + 1. ds = 1 + m - floor(m/2) - 1 = ceil(m/2).

If q = 2: m/3 < b <= m/2. ds = 2 + m - 2b. Max at b = floor(m/3)+1.
  ds = 2 + m - 2*(floor(m/3)+1) = m - 2*floor(m/3).
  For m = 3k: ds = 3k - 2k = k = m/3. Worse than m/2.
  For m = 3k+1: ds = 3k+1 - 2k = k+1. ≈ m/3. Worse.
  For m = 3k+2: ds = 3k+2 - 2k = k+2. ≈ m/3. Worse.

So 2-digit representation with q=1 is optimal among 2-digit options: gives ceil(m/2).

What about 3+ digits?
q = 1, next digit up to b-1, last digit up to b-1: max ds = 1 + (b-1) + (b-1) = 2b-1.
This requires m >= b^2. With b^2 <= m, ds <= 2*sqrt(m) - 1.
For m = 500000: 2*sqrt(500000) - 1 ≈ 1413. Much less than 250000.

So the dominant strategy is to use the 2-digit representation with q=1, getting ceil(m/2).

Wait, but for VERY large m (like in the first step), we have m up to 10^(10^5).
Using base b = m/2 + 1 gives digit sum ≈ m/2, which is ≈ 5*10^(10^5 - 1).
That's WAY more than the ~10^5 we computed with the (b-1)*k formula!

Hmm wait. n = 10^(10^5). Base b = n/2 + 1 ≈ 5*10^(99999).
digit_sum = 1 + (n - b) = n - n/2 = n/2 ≈ 5*10^(99999).
Then from n/2, we do the same thing: get n/4, then n/8, etc.
After t halvings: n/2^t. We stop when n/2^t ≈ 1, so t ≈ log2(n) ≈ 10^5 * log2(10) ≈ 332193.

But wait, can we go EVEN slower? From m, if we choose base b = 2, and m = 2^k - 1,
digit_sum = k = log2(m+1). That's MUCH less than m/2. So base 2 gives a HUGE reduction.

Choosing base b = m/2 + 1 gives m/2, which reduces very slowly (by half each time).
This seems optimal for maximizing the number of moves!

From n = 10^(10^5), repeatedly halving:
Number of halvings until we reach 1: about log2(10^(10^5)) = 10^5 * log2(10) ≈ 332193.

But wait, we can be even more clever. From m, ceil(m/2) ≈ m/2.
So each step reduces by factor 2. That gives ~log2(n) steps from n.

But what if instead of halving, we could reduce by subtracting 1?
That would give m-1 steps which is enormous. Can we achieve that?

From m, can we get m-1? digit_sum(m, b) = m-1?
We need digit_sum(m, b) = m - 1 for some b.
m in base b has digit sum = m - (b-1) * S where S = sum of floor(m/b^i) for i >= 1.
So we need (b-1)*S = 1. Since b >= 2, b-1 >= 1, so S = 1 and b-1 = 1, meaning b=2.
S = sum_{i>=1} floor(m/2^i) = m - s_2(m) where s_2(m) is the binary digit sum.
Wait, no. S = floor(m/2) + floor(m/4) + ...
And digit_sum(m, 2) = m - S*(2-1) ... no.
digit_sum(m, 2) = number of 1-bits in m.
For digit_sum = m-1: need m-1 ones in binary. But m in binary has at most floor(log2(m))+1 bits.
So digit_sum(m,2) <= floor(log2(m)) + 1. For m >= 4, this is much less than m-1.

So we can't go from m to m-1 in general. The halving strategy seems close to optimal.

Actually, the key question is: what is f(m) = max number of moves from m?
With the halving strategy: f(m) >= 1 + f(ceil(m/2)).
f(1) = 0, f(2) = 1 (base 2: digit_sum = 1, done in 1 step).
f(3) = 1 + f(2) = 2? Let's check: from 3, base 2 gives 1+1=2, so go to 2 then 1. 2 moves.
Or base 3: digit_sum = 1. 1 move. So f(3) = 2 (choose base 2, go to 2, then 1).
Wait: from 3, choosing base 2 gives digit_sum(3,2) = 1+1 = 2. Then from 2, choose base 2, digit_sum = 1. That's 2 moves. Yes, f(3)=2.
f(4) = 1 + f(2) = 2. From 4: base 3 → 1+1=2. Then 1 more move. 2 total.
  Or: base 2 → 1+0+0 = 1. Just 1 move. So from 4, digit_sum(4,3) = 2 (going to 2), giving f=2.
  Or base 2: go to 1, just 1 move. So actually max is 2 via base 3.

Hmm wait: digit_sum(4, 3) = digit_sum of 11_3 = 1+1 = 2. Then from 2, one more step.
digit_sum(4, 2) = digit_sum of 100_2 = 1. That reaches 1 directly, just 1 move.
So f(4) = max(1 + f(2), 1 + f(1)) = max(2, 1) = 2. Correct, f(4)=2.

f(5): base 3 → 12_3 → 1+2=3, f(3)=2, so 1+2=3.
  base 4 → 11_4 → 2, f(2)=1, so 1+1=2.
  base 2 → 101_2 → 2, f(2)=1, so 1+1=2.
  base 5 → 10_5 → 1, so 1.
  Best: base 3, f(5) = 3.

Let me just compute f(m) for small m by brute force.
"""

def digit_sum(m, b):
    s = 0
    while m > 0:
        s += m % b
        m //= b
    return s

# Compute f(m) for m = 1 to some limit
LIMIT = 100000

f = [0] * (LIMIT + 1)
best_base = [0] * (LIMIT + 1)
# f[1] = 0
for m in range(2, LIMIT + 1):
    best_f = 0
    best_b = 2
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds >= m:
            continue  # no reduction
        val = 1 + f[ds]
        if val > best_f:
            best_f = val
            best_b = b
    f[m] = best_f
    best_base[m] = best_b

# Show some results
print("f(m) for small m:")
for m in range(1, 51):
    print(f"  f({m}) = {f[m]} (best base: {best_base[m]})")

print(f"\nf(100) = {f[100]}")
print(f"f(1000) = {f[1000]}")
print(f"f(10000) = {f[10000]}")
print(f"f(100000) = {f[100000]}")

# Find the pattern
print("\nLargest f values:")
max_f_at = {}
for m in range(1, LIMIT+1):
    if f[m] not in max_f_at or m < max_f_at[f[m]]:
        max_f_at[f[m]] = m

for fv in sorted(max_f_at.keys()):
    print(f"  First m with f(m)={fv}: {max_f_at[fv]}")
