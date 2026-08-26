"""
Reconsidering: A 500x500 square divided into k rectangles with integer sides,
all with distinct perimeters. Find max k.

Each rectangle has dimensions a_i x b_i with a_i, b_i positive integers.
Both a_i, b_i <= 500 (since they fit in the square).
Perimeter = 2*(a_i + b_i), all distinct => all (a_i + b_i) distinct.
Sum of areas = 250000.

For semi-perimeter s = a+b, with 1 <= a <= b <= 500:
  - s ranges from 2 to 1000
  - a ranges from max(1, s-500) to s//2
  - min area at a=max(1,s-500):
      if s<=501: min_area = s-1
      if s>=502: min_area = (s-500)*500

Alternative approach: Think of it differently. Each rectangle has dimensions
(a, s-a) with distinct s values. We want to pack as many distinct s values
as possible while keeping total area = 250000.

The key insight I might be missing: we have freedom in choosing a for each s.
For s <= 501, we can use area as small as s-1 (with a=1) or as large as s^2/4.
For s >= 502, minimum area is (s-500)*500.

The s values from 502 to 1000 are very expensive (min area 1000 to 250000).
So we should mainly use s from 2 to 501.

With s = 2 to 501 (500 rectangles), min area = 125250, max area huge.
We need total area = 250000, which is between min and max. So we can do 500.

Can we get 501? We need one more distinct s value.
Best option: s from 502 to 1000. Cheapest is s=502 with min area 1000.
New min total = 125250 + 1000 = 126250 <= 250000.
New max total >= 250000 (definitely, since sum of max areas is huge).
So k=501 works!

Can we get 502? Add s=503 (cost 1500). Total min = 127750 <= 250000. Yes!

Continue adding cheap s values from 502 upward...
"""

AREA = 250000

# Min areas
def min_area(s):
    if s <= 501:
        return s - 1
    else:
        return (s - 500) * 500

# Sort all s by min_area and greedily pick cheapest
all_s = list(range(2, 1001))
all_s_sorted = sorted(all_s, key=min_area)

# Print the cost structure
print("Cost structure (min_area):")
print("  s=2: cost", min_area(2))
print("  s=100: cost", min_area(100))
print("  s=500: cost", min_area(500))
print("  s=501: cost", min_area(501))
print("  s=502: cost", min_area(502))
print("  s=503: cost", min_area(503))
print("  s=510: cost", min_area(510))
print("  s=520: cost", min_area(520))
print("  s=521: cost", min_area(521))
print("  s=550: cost", min_area(550))
print("  s=600: cost", min_area(600))

# Greedy accumulation
total = 0
count = 0
for s in all_s_sorted:
    c = min_area(s)
    if total + c <= AREA:
        total += c
        count += 1
    else:
        print(f"\nStopped at s={s}, cost={c}")
        print(f"Current total = {total}, adding would give {total+c}")
        break

print(f"\nGreedy: k = {count}, total min area = {total}, deficit = {AREA - total}")

# The ordering: first s=2..501 (costs 1,2,...,500), then s=502 (1000), s=503 (1500)...
# s=2..501: 500 values, cumulative cost = 125250
# s=502: 501st value, cumulative = 126250
# s=503: 502nd, cumulative = 127750
# s=504: 503rd, cumulative = 129750
# ...
# s=500+d (d >= 2): cost = 500d
# Cumulative after adding s=502..500+D:
# 125250 + 500*(2+3+...+D) = 125250 + 500*(D*(D+1)/2 - 1)

# Find max D such that 125250 + 500*(D*(D+1)/2 - 1) <= 250000
# 500*(D*(D+1)/2 - 1) <= 124750
# D*(D+1)/2 - 1 <= 249.5
# D*(D+1) <= 501
# D=21: 21*22=462 ✓, D=22: 22*23=506 ✗
# So D=21, meaning s=502..521 (20 values since d ranges from 2 to 21)

# Total with D=21: 125250 + 500*(462/2 - 1) = 125250 + 500*230 = 125250 + 115000 = 240250
# k = 500 + 20 = 520

# Now, the REAL question: is there a smarter selection?
# What if instead of the standard s=2..501 + s=502..521, we skip some s values
# and use the freed area differently?

# Key observation: We CANNOT gain by swapping because:
# - All s=2..501 have cost < 500 each
# - All s >= 502 have cost >= 1000 each
# - Removing any selected s saves at most 10500 (s=521, cost=10500)
# - But the cheapest available s=522 costs 11000, so no room for 2 new ones.

# Wait, but what about removing a s >= 502 value and replacing with
# multiple s values from... but all cheaper ones are already selected!

# Hmm, unless we consider a COMPLETELY different approach to the problem.
# Let me reconsider: maybe the problem is asking about rectangles in a more
# general sense, where a rectangle doesn't need both sides <= 500.
# In some formulations, "divided into rectangles" just means the pieces
# are rectangles. In a square, yes, each piece must fit, so both dims <= 500.

# Actually, wait. Let me reconsider the problem statement:
# "A 500x500 square is divided into k rectangles, each having integer side lengths."
# This IS a partition of the square, so each rectangle fits inside.

# But hmm, I just realized: when we place a 1x(s-1) rectangle, it doesn't
# need s-1 <= 500 in all cases! A 1x500 rectangle fits fine. But 1x501 doesn't.
# So for s <= 501, the minimum area choice 1x(s-1) fits (max dim = s-1 <= 500).
# For s = 501, min rectangle is 1x500, fits in 500x500. OK.
# For s = 502, we need both dims <= 500, so a >= 2 and b = 500. Rect: 2x500.

# Alright, my analysis seems correct. Let me also consider:
# What if we allow non-rectangular cuts? No, the problem says "divided into rectangles".

# Let me verify once more with a complete brute-force for smaller squares,
# then extrapolate.

def solve_for_n(n):
    """Max k for n x n square."""
    area = n * n

    # All possible s values: 2 to 2n
    # Min area for s: max(1, s-n) * min(s-1, n)
    # But actually: for s <= n+1, min_area = s-1 (using 1 x (s-1), fits since s-1 <= n)
    # For s >= n+2: min_area = (s-n) * n (using (s-n) x n, fits since s-n >= 2, n <= n)

    def min_area_n(s):
        if s <= n + 1:
            return s - 1
        else:
            return (s - n) * n

    all_s = list(range(2, 2*n + 1))
    all_s.sort(key=min_area_n)

    total = 0
    selected = []
    for s in all_s:
        c = min_area_n(s)
        if total + c <= area:
            total += c
            selected.append(s)

    # Verify achievability: need total_max >= area
    def max_area_n(s):
        a = s // 2
        a = max(a, max(1, s - n))
        a = min(a, min(s // 2, n))
        return a * (s - a)

    total_max = sum(max_area_n(s) for s in selected)
    achievable = total_max >= area

    return len(selected), total, area - total, achievable

# Test with small squares
for n in [2, 3, 4, 5, 10, 20, 50, 100, 500]:
    k, total_min, deficit, achievable = solve_for_n(n)
    print(f"n={n}: k={k}, total_min={total_min}, deficit={deficit}, achievable={achievable}")

print(f"\n=== For n=500 ===")
K, _, _, _ = solve_for_n(500)
print(f"K = {K}")
print(f"K mod 10^5 = {K % 100000}")
