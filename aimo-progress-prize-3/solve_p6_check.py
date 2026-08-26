"""
Double-check: is there ever a case where we can go from m to something > ceil(m/2)?

For 2-digit representation in base b (b <= m < b^2):
  m = q*b + r, ds = q + r = q + m - q*b = m - q*(b-1)
  For q=1, b = ceil(m/2)+1 if even, (m+1)/2 if odd: ds = ceil(m/2)

But what if m = (b-1)*b + (b-1) = b^2 - 1 (all digits b-1 in base b)?
  ds = (b-1) + (b-1) = 2*(b-1). This is a 2-digit number in base b.
  We need 2*(b-1) > ceil(m/2) = ceil((b^2-1)/2).
  For even b: ceil((b^2-1)/2) = b^2/2. 2(b-1) > b^2/2? 4b-4 > b^2? b^2-4b+4 < 0? (b-2)^2 < 0? Never.
  For b=2: 2(1) = 2, ceil(3/2) = 2. Equal.
  For b=3: 2(2) = 4, ceil(8/2) = 4. Equal.
  For b=4: 2(3) = 6, ceil(15/2) = 8. 6 < 8. Worse.

So 2-digit all-(b-1) gives ds = 2(b-1), while halving gives ds = ceil(m/2) ≈ m/2 = (b^2-1)/2.
For b >= 4, halving is better.

What about 3-digit: m = (b-1)*b^2 + (b-1)*b + (b-1) = b^3 - 1.
  ds = 3*(b-1). Halving gives ceil((b^3-1)/2) ≈ b^3/2.
  3(b-1) > b^3/2? Only for very small b. b=2: 3 vs 4. No.

So for m >= 4, halving (2-digit with q=1) is always optimal.

For m=2: ds(2, b) for b=2: "10" → 1. That's the only option. f(2) = 1. ✓
For m=3: b=2 → "11" → 2, b=3 → "10" → 1. Best is 2. f(3) = 1 + f(2) = 2. ✓

OK, I'm convinced. f(m) = ceil(log2(m)) is correct.

But wait - I want to REALLY make sure about the definition of the problem.
"the largest possible number of moves Ken could make"
This is over all choices of n AND all sequences of base choices.
We showed the optimal strategy from any m is halving, giving ceil(log2(m)) moves.
The optimal n is the largest possible, n = 10^(10^5).

But hold on: is n = 10^(10^5) the right interpretation? "1 <= n <= 10^(10^5)"
10^5 = 100000. 10^(10^5) = 10^100000. Yes.

So M = ceil(log2(10^100000)) = 332193.
M mod 10^5 = 32193.

But wait, I should double-check: is the maximum at n = 10^100000 or could it be at
some n slightly less (like 2^332193 - 1 which has ceil(log2(2^332193-1)) = 332193 as well)?
No, the max f value is the same: 332193 for all n in [2^332192+1, 2^332193].
Since 10^100000 is in this range (as we verified), f(10^100000) = 332193.

Actually, f(2^332193) = ceil(log2(2^332193)) = 332193 as well.
And 2^332193 > 10^100000, so we can't use it.
The largest n we can use is 10^100000, and f(10^100000) = 332193.

So M = 332193, answer = 32193.
"""

# Actually wait. I want to very carefully verify that my formula f(m) = ceil(log2(m))
# is correct by rechecking the optimality of halving for ALL bases, not just 2-digit ones.

def digit_sum(m, b):
    s = 0
    while m > 0:
        s += m % b
        m //= b
    return s

# For several values of m, find the absolute maximum digit sum over ALL bases
test_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
for m in test_values:
    best_ds = 0
    best_b = 0
    half_target = (m + 1) // 2  # ceil(m/2)
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds > best_ds:
            best_ds = ds
            best_b = b
    print(f"m={m}: max digit_sum = {best_ds} (base {best_b}), ceil(m/2) = {half_target}, match = {best_ds == half_target}")

# Also verify the formula for some larger numbers using just the halving check
print("\n--- Checking halving optimality for larger m ---")
import random
random.seed(42)
for _ in range(10):
    m = random.randint(100000, 1000000)
    # Check: is there any base giving ds > ceil(m/2)?
    half_target = (m + 1) // 2
    found_better = False
    # Only need to check bases where we might get > half_target
    # For 2-digit: ds = q + r = m - q*(b-1). For q=1: ds = m-b+1. Max at b = m//2+1.
    # For q>=2: ds = m - q*(b-1) <= m - 2*(b-1). For this to exceed half_target = m/2:
    #   m - 2*(b-1) > m/2 → m/2 > 2*(b-1) → b < m/4 + 1.
    #   But for q=2: b <= m/2 and b > m/3. So b in (m/3, m/4+1) is empty for m >= 12. Good.
    # For multi-digit: digit_sum is much smaller. So no need to check exhaustively.
    # But let's verify for a few bases anyway.
    for b in range(2, min(m + 1, 200)):
        ds = digit_sum(m, b)
        if ds > half_target:
            print(f"  FOUND BETTER: m={m}, b={b}, ds={ds} > {half_target}")
            found_better = True
            break
    if not found_better:
        # Also check the optimal 2-digit base
        b = m // 2 + 1
        ds = digit_sum(m, b)
        assert ds == half_target, f"m={m}: expected {half_target}, got {ds} at b={b}"
    print(f"  m={m}: halving is optimal (ceil(m/2) = {half_target})")
