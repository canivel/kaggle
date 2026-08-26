"""
Problem 6 final computation.

f(m) = ceil(log2(m)) for m >= 2.
We want M = max f(n) for 1 <= n <= 10^(10^5).

Since f is increasing (more or less), M = f(10^(10^5)) = ceil(log2(10^(10^5))).

But wait: n = 10^(10^5) means n can be AT MOST 10^(10^5), not 10^(10^5) - 1.
Actually the problem says 1 <= n <= 10^(10^5). So the max is at n = 10^(10^5).

ceil(log2(10^(10^5))) = ceil(10^5 * log2(10))

log2(10) = ln(10)/ln(2) = 3.321928094887362...

10^5 * log2(10) = 332192.8094887362...

So ceil(...) = 332193.

Wait, but we need to be more precise. Is 10^5 * log2(10) an integer?
log2(10) = log2(2*5) = 1 + log2(5). log2(5) is irrational (since 5 is not a power of 2).
So 10^5 * log2(10) is irrational, hence not an integer.

Since 10^5 * log2(10) = 332192.80948..., ceil = 332193.

But actually, we should double-check: is there a larger n' <= 10^(10^5) with f(n') > f(10^(10^5))?
No, because f is non-decreasing: for n1 < n2, f(n1) <= f(n2) since f(n) = ceil(log2(n)).
Wait, that's not quite right since ceil(log2) is not strictly increasing.
f(n) = f(n+1) when they're in the same range. But f(10^(10^5)) is the max.

So M = 332193.
M mod 10^5 = 32193.

Actually wait. Let me reconsider. I need to double check: is n = 10^(10^5) achievable?
The problem says 1 <= n <= 10^(10^5). So n can equal 10^(10^5). Good.

But I should also check: can we do BETTER than ceil(log2(n)) by choosing a different n?
For example, n = 10^(10^5) - 1 might also give f(n) = 332193 since it's one less.
In fact, 2^332193 > 10^(10^5) > 2^332192.
So for any n with 2^332192 < n <= 2^332193, f(n) = 332193.
And 10^(10^5) is in this range.
The maximum over [1, 10^(10^5)] is 332193.

Hmm wait. Could f(n) = ceil(log2(n)) actually be wrong for very large n?
Let me re-examine the proof.

From m, the maximum reachable value is ceil(m/2) (achieved by base ceil(m/2) + 1 for m >= 4).
Actually: for even m, digit_sum(m, m/2+1) = 1 + (m - m/2 - 1) = m/2.
For odd m, digit_sum(m, (m+1)/2) = 1 + (m - (m+1)/2) = 1 + (m-1)/2 = (m+1)/2.

So max reachable = ceil(m/2). But wait, could multi-digit representations sometimes give more?

For m=6: base 3 gives 20_3, ds = 2. base 4 gives 12_4, ds = 3 = ceil(6/2). Same.
For m=10: base 6 gives 14_6, ds = 5 = ceil(10/2). Or base 3: 101_3, ds = 2. Halving wins.
For m=100: base 51 gives 1,49 in base 51 (digits 1 and 49), ds = 50 = 100/2.
  In base 99: 1,1 → ds = 2. In base 10: ds = 1.
  What about base 9? 100 = 1*81 + 2*9 + 1 = 121_9, ds = 4.
  So halving to 50 is best.

But wait, I should also check: can we ever get MORE than ceil(m/2)?
digit_sum(m, b) for any b with m = q*b + r (2-digit): ds = q + r = q + m - q*b = m - q*(b-1).
For q=1: ds = m - (b-1) = m - b + 1. Max at b=2: ds = m-1. WAIT!

For b=2, 2-digit representation means m < 4. For m=2: ds = 10_2 → 1. For m=3: ds = 11_2 → 2.
For m=3, b=2: ds=2 = ceil(3/2) = 2. ✓

For general b=2: digit_sum(m, 2) is the number of 1-bits, not m-1.
I made an error above. Let me reconsider.

If m has only 2 digits in base b, i.e., b <= m < b^2:
  m = q*b + r, 0 <= r < b, 1 <= q < b
  ds = q + r

For q=1 (i.e., b <= m < 2b): ds = 1 + (m-b) = m - b + 1.
  Maximized when b is as small as possible: b = ceil(m/2) + 1... no.
  Wait, we need m < 2b (for q=1), so b > m/2, i.e., b >= floor(m/2)+1.
  Then ds = m - b + 1. Maximized at b = floor(m/2)+1:
    For even m: ds = m - m/2 - 1 + 1 = m/2.
    For odd m: ds = m - (m-1)/2 - 1 + 1 = (m+1)/2.
  So ds = ceil(m/2). ✓

But with more digits:
b = 2, m >= 4: m has 3+ digits. ds = popcount(m).
  For m = 2^k - 1: ds = k. For large m, this is much less than m/2.

For b = 3, m >= 9: ds <= 2 * floor(log_3(m)) + 2. Much less than m/2 for large m.

So indeed the maximum over all bases is ceil(m/2), achieved with a 2-digit representation.

Proof that f(m) = ceil(log2(m)):
Upper bound: f(m) <= 1 + f(ceil(m/2)) since the best we can do is ceil(m/2).
  f(m) <= 1 + 1 + f(ceil(ceil(m/2)/2)) <= ...
  After k steps: f(m) <= k + f(ceil(m/2^k)). When ceil(m/2^k) = 1, we stop.
  ceil(m/2^k) = 1 iff m <= 2^k, i.e., k >= ceil(log2(m)).
  So f(m) <= ceil(log2(m)).

Lower bound: We can achieve ceil(log2(m)) by the halving strategy.
  From m, go to ceil(m/2), then ceil(ceil(m/2)/2), etc.
  This reaches 1 in exactly ceil(log2(m)) steps.
  (Because: starting from m, after k halvings we have ceil(m/2^k) which is 1 iff k >= ceil(log2(m)).)

So f(m) = ceil(log2(m)) for all m >= 2. f(1) = 0.

For n = 10^(10^5): f(n) = ceil(log2(10^(10^5))) = ceil(10^5 * log2(10)).

Let me compute this precisely.
"""

import math
from decimal import Decimal, getcontext

# We need ceil(100000 * log2(10))
# log2(10) = log(10)/log(2)

# Use high precision
getcontext().prec = 50

# log2(10) = 3.32192809488736234787031942948939017758...
# 100000 * log2(10) = 332192.809488736234787031942948939017758...
# ceil = 332193

# But wait, we should check: can n = 10^(10^5) - 1 give more moves?
# f(10^(10^5) - 1) = ceil(log2(10^(10^5) - 1))
# Since 10^(10^5) - 1 < 10^(10^5), and 10^(10^5) is not a power of 2,
# we have ceil(log2(10^(10^5) - 1)) = ceil(log2(10^(10^5))) = 332193.
# (Because 2^332192 < 10^(10^5) - 1 < 10^(10^5) < 2^332193.)

# Actually, is 10^(10^5) < 2^332193? Let's check.
# 2^332193 vs 10^100000
# 332193 * log10(2) vs 100000
# 332193 * 0.30103 = ?
val = 332193 * Decimal('0.3010299957316877293527446341505872366942805253810380628055806')
print(f"332193 * log10(2) = {val}")
print(f"This should be > 100000 for 2^332193 > 10^100000")

# Also check 332192 * log10(2)
val2 = 332192 * Decimal('0.3010299957316877293527446341505872366942805253810380628055806')
print(f"332192 * log10(2) = {val2}")
print(f"This should be < 100000 for 2^332192 < 10^100000")

M = 332193
print(f"\nM = {M}")
print(f"M mod 10^5 = {M % 100000}")

# Actually wait, I need to be much more careful. The problem says n up to 10^(10^5).
# 10^5 = 100000. So n <= 10^100000.
#
# But actually, n up to 10^(10^5) where 10^5 = 100000.
# So the max value is 10^(100000).
#
# f(10^100000) = ceil(log2(10^100000)) = ceil(100000 * log2(10)).
#
# We need this to be computed exactly.
# 100000 * log2(10) = 100000 * log(10)/log(2)
#
# Is 2^332192 < 10^100000 < 2^332193?
# 2^332192 = 10^(332192 * log10(2)) = 10^(332192 * 0.301029995...)
# 332192 * 0.301029995... = 99999.69886...
# So 2^332192 = 10^99999.699 < 10^100000. ✓
#
# 2^332193 = 2 * 2^332192 = 10^(99999.699 + log10(2)) = 10^(99999.699 + 0.301) = 10^100000.000...
# More precisely: 332193 * 0.30103... = 99999.699... + 0.30103... = 100000.000...
# Let's compute exactly.

# 332193 * log10(2):
# We need VERY precise log10(2).
# log10(2) = 0.30102999566398119521373889472449302676818596588058...
getcontext().prec = 100
log10_2 = Decimal('0.30102999566398119521373889472449302676818596588058635042586592614252245205728026862065462830968066440')
v1 = 332193 * log10_2
v2 = 332192 * log10_2
print(f"\n332193 * log10(2) = {v1}")
print(f"332192 * log10(2) = {v2}")
print(f"\nSo 2^332193 = 10^{v1}")
print(f"Since {v1} > 100000, we have 2^332193 > 10^100000.")
print(f"And 2^332192 = 10^{v2} < 10^100000.")
print(f"Therefore ceil(log2(10^100000)) = 332193.")

print(f"\nFinal answer: M = 332193, M mod 10^5 = {332193 % 100000}")
