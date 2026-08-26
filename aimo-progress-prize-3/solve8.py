"""
Let me reconsider the problem from scratch.

"A 500x500 square is divided into k rectangles, each having integer side lengths."

CRITICAL INSIGHT: The rectangles must tile the 500x500 square, but the problem
says "integer side lengths" -- it does NOT say the sides are at most 500!

Wait, yes it does implicitly: a rectangle in a tiling of a 500x500 square
necessarily has both dimensions <= 500.

BUT ACTUALLY... I realize I might be wrong about needing BOTH sides <= 500.
Consider a rectangle that is 1 x 500 placed horizontally. Now consider if we
could have a rectangle that is 600 x 1... No, that's 600 units in one direction,
but the square is only 500. It doesn't fit.

So both sides must be <= 500. My analysis with k=520 is correct.

Hmm, but let me double-check by considering the problem differently.
What if the intended interpretation doesn't require a physical tiling, but rather:
"The square has area 250000. Split this area into k rectangles with integer sides
and distinct perimeters. Maximize k."

Under this interpretation:
- No constraint that dimensions <= 500
- Just need areas to sum to 250000
- Each rectangle has integer sides a_i x b_i, perimeter 2(a_i + b_i) all distinct

Then: min area for semi-perimeter s is s-1 (using 1x(s-1)).
Sum of s-1 for s=2..k+1 = k(k+1)/2.
Need k(k+1)/2 <= 250000.
k=706: 706*707/2 = 249571 <= 250000. Yes!
k=707: 707*708/2 = 250278 > 250000. No!

Under this interpretation, K = 706, K mod 10^5 = 706.

Actually, I think this IS the intended interpretation for a competition problem.
The problem says "divided into k rectangles" meaning the square is partitioned
into rectangles that tile it. But in competition math, the standard meaning IS
a tiling where rectangles fit inside the square.

HOWEVER: let me reconsider whether the 706 answer is reachable even with
the tiling constraint. The issue was: a 1x706 rectangle doesn't fit.
But we don't have to use 1x(s-1)! We could use a different aspect ratio.

For s=707: a+b=707. We need a,b >= 1 and both a,b <= 500.
So a >= 207 (since b=707-a <= 500 requires a >= 207).
Min area: 207*500 = 103500. That's HUGE.

For s=600: a >= 100, b <= 500. Min area: 100*500 = 50000.

These large-s values have prohibitively large areas. My earlier analysis was correct.

BUT WAIT: what if we use ONLY s values where 1x(s-1) fits, i.e., s <= 501?
Then we have 500 distinct s values, with areas summing to 125250 < 250000.
We need more area, so we increase some rectangles' dimensions.

The total area is 250000, and with 500 1-tall rectangles, max total area with
widths up to 500 is 1+2+...+500 = 125250. Not enough!

So we MUST have some rectangles with height > 1. These rectangles have larger
area, which helps reach 250000 but uses up area budget.

Hmm, I keep going in circles. Let me try yet another angle.

What if the problem is NOT about geometric tiling but about a NUMBER THEORY
problem? "Divided into k rectangles" might be a pure combinatorial statement:
express 250000 as a sum of k products a_i * b_i where a_i + b_i are all distinct
positive integers.

Under this purely arithmetic interpretation:
- No constraint on a_i, b_i beyond being positive integers
- Distinct a_i + b_i values
- Sum of a_i * b_i = 250000

This gives the k=706 answer (or close to it).

For AIMO, this arithmetic interpretation is more likely since the "mod 10^5"
hint suggests a specific numerical answer, and 706 mod 10^5 = 706.

Actually wait, 706 is also small. Let me reconsider if maybe k is much larger.

Hmm, if perimeters just need to be distinct (not semi-perimeters), then
perimeter 2(a+b) can be any even number >= 4.
Two rectangles with the same a+b have the same perimeter.
Two rectangles with different a+b have different perimeters.
So distinct perimeters <=> distinct a+b. Same as before.

What if we allow a_i = b_i = 0? No, rectangle sides are positive.

I think the answer is 706. Let me verify this cleanly.
"""

N = 500
AREA = N * N  # 250000

# Pure arithmetic interpretation: pick distinct s values from {2, 3, ...}
# For each, pick a,b >= 1 with a+b=s, contributing area a*b.
# Min area for semi-perim s: 1*(s-1) = s-1.
# Max area for semi-perim s: (s//2)*((s+1)//2) = floor(s^2/4).

# Maximize k with sum of areas = 250000.
# Need: sum(min_area) <= 250000 <= sum(max_area)
# Use s=2,3,...,k+1. Min total = k(k+1)/2.

# Find max k with k(k+1)/2 <= 250000
k = 0
while (k + 1) * (k + 2) // 2 <= AREA:
    k += 1
print(f"Max k (arithmetic): {k}")
print(f"k(k+1)/2 = {k*(k+1)//2}")
print(f"Deficit = {AREA - k*(k+1)//2}")

# For k=706: total min = 249571, deficit = 429
# We showed earlier this can be absorbed.

# Under the geometric tiling interpretation with both dims <= 500:
# Max k = 520.

# The question is: which interpretation does the problem intend?
# "A 500x500 square is divided into k rectangles"
# This clearly means a tiling/partition. So geometric.
# And each rectangle has both dimensions <= 500.

# UNLESS... there's a cleverer tiling I'm not seeing.
# Let me reconsider: can we tile with s > 501 if we use non-minimum area?

# For example, s = 510: a+b = 510, with a >= 10 (so b <= 500).
# If we use a = 10, b = 500: area = 5000. This is NOT minimum area.
# Wait, IS this minimum? For s=510:
# a ranges from max(1, 510-500) = 10 to min(s//2, 500) = 255.
# Area = a*(510-a). At a=10: 10*500 = 5000. At a=255: 255*255 = 65025.
# Min area = 5000 (at boundary).
# That's exactly what I had before: (s-500)*500 = 10*500 = 5000.

# The issue is that for s > 501, the min area is (s-500)*500, which grows fast.
# With 500 cheap s-values (2..501) costing 125250, and budget 124750 left,
# we can only fit ~20 expensive s-values.

# But what if we DON'T use all of s=2..501?
# For instance, skip s=501 (saving area 500) and add s=522 (costing 11000).
# Net cost change: +10500, losing one rect and gaining one. k stays same.

# Or: skip s=501 AND s=500 (saving 999) and add s=502 was already added.
# We already have s=502..521. The next is s=522 (cost 11000).
# By skipping s=501 (save 500), s=500 (save 499), we free 999.
# Can we add s=522 (cost 11000)? Still need 11000 - 9750 - 999 = 251. Not enough.
# Skip more: s=499 (498), total freed = 1497. Still need 11000-9750-1497 = -247.
# Wait: 9750 + 1497 = 11247 >= 11000. Yes!
# But we lost 3 rects and gained 1 = net -2. Worse!

# There's no way to improve beyond 520 with the tiling constraint.

# Actually, wait. I just realized something. Let me reconsider whether we need
# to constrain dimensions to <= 500. What if the problem means we can use
# rectangles where the side lengths are integers, but they don't need to physically
# fit inside a 500x500 square? That doesn't make sense geometrically.

# Let me consider: maybe the answer IS 706 and the tiling always works because
# we CAN always tile. Maybe I was wrong about the dimension constraint.

# Actually, re-reading: "A 500x500 square is divided into k rectangles."
# In a tiling/partition, each rectangle is a subset of the square.
# Its dimensions are bounded by 500 in each direction.
# This is a hard geometric constraint.

# So the answer should be 520 with the geometric interpretation.

# But let me compute what competition answer databases say...
# The "mod 10^5" hint suggests the answer might be around 520.

# Let me also double check: 520 mod 10^5 = 520.

# Actually, maybe I should reconsider the problem once more.
# What if by "integer side lengths" they mean the SQUARE has integer side lengths
# (which it does: 500), and the rectangles can have non-integer sides?
# Then distinct perimeters among real-valued rectangles... but that would make k
# potentially infinite. So no, they must mean the rectangles have integer sides.

# Let me also consider: what if rectangles can overlap? No, "divided into" means partition.

# I'll go with K = 520, answer = 520.

# Wait, actually I want to reconsider once more. Competition problems usually have
# elegant answers. 520 = 500 + 20 = 500 + D where D*(D+1) <= 501.
# D = 21 gives 462, D=22 gives 506. So D_max = 21, extra = 21-1 = 20.
# k = 500 + 20 = 520.

# Hmm, actually... is the answer perhaps different if I think about it as:
# s can range from 2 to 1000.
# For s <= 501: rectangle 1 x (s-1), min area = s-1
# For 502 <= s <= 1000: rectangle (s-500) x 500, min area = (s-500)*500

# What if instead of using ALL s from 2 to 501, I skip some small ones
# whose area contribution is negligible, and use that to add more large s values?
# Small s values contribute tiny areas. Skip s=2 (saves area 1). That's nothing.
# The cheapest large s (=502) costs 1000. We'd need to skip 1000 small values.
# But there are only 500 small values! And skipping all saves only 125250.
# With 125250 + 124750 = 250000, and adding zero small + max large...
# That gives sum(min_area for s=502..D) = 250000.
# 500 * (D-500)*(D-499)/2 ... this is complex.

# Let me just compute: if we use ONLY large s values (s >= 502), what's max k?
# s=502: cost 1000, s=503: cost 1500, ..., s=500+d: cost d*500.
# Sum for d=2..D: 500*(D*(D+1)/2 - 1)
# Need <= 250000: D*(D+1)/2 - 1 <= 500. D*(D+1) <= 1002. D <= 31.
# So 30 large rectangles. Much worse than 520.

# What about mixing? Use some small and some large.
# The greedy (sorted by cost) gives the optimal answer: 520.

# Let me also verify with an integer linear programming approach for small cases.

print("Final answer: K = 520")
print("K mod 10^5 =", 520 % 100000)
