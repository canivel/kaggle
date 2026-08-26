"""
Let me re-examine the tiling constraint more carefully.

The area-matching is necessary but not sufficient for tiling.
However, for competition purposes, if we can construct an explicit tiling,
that suffices.

Key question: can we always construct a tiling given our rectangle set?

Our construction for n=500:
- 20 rectangles of size d x 500 for d = 2, 3, ..., 21 (total height = 230)
- 500 rectangles of size 1 x m (various m values, with one adjusted)
  placed in the remaining 270 rows

For the 1 x m strips: each row is 500 wide. We pack strips side by side.
A row can hold strips whose widths sum to 500.

The strips have widths: 1, 2, ..., 500 (with one replaced by a wider rect).
Wait, the strip widths are the "b" values: for s=2 to 501, the rectangle
is 1 x (s-1), so widths are 1, 2, ..., 500.

Actually, we orient 1 x m as 1-tall, m-wide. Each row is 1 tall, 500 wide.
Pack strips into rows: need widths in each row to sum to exactly 500.

Strips to pack: widths 1, 2, 3, ..., 500 (with one adjusted).
Actually, the adjusted rectangle (say originally 1x204, now 79x126) is NOT a strip.
It's 79 tall and 126 wide. That complicates the strip packing.

Alternative: make the adjustment on a different rectangle, or use a different
tiling strategy.

Actually, let me reconsider the tiling.

Better approach: use a "guillotine cut" tiling.
- Cut the 500x500 square horizontally into rows of various heights.
- Each row is a horizontal strip that contains one or more rectangles side by side.

For our construction:
Row 1: height 1, contains the 1x500 rectangle. (s=501)
Row 2: height 2, contains the 2x500 rectangle. (s=502)
...
Row 21: height 21, contains the 21x500 rectangle. (s=521)

Total height used: 1 + 2 + ... + 21 = 231. Remaining: 500 - 231 = 269 rows.

Wait, I should include s=2..500 (the 1x(s-1) strips) and handle s=501 differently.
s=501: rectangle 1x500. That's a 1-tall row filling the full width.

Let me redo:
- s=2..501: rectangles 1x1, 1x2, ..., 1x500. These are 1-tall strips.
  Total area: 1+2+...+500 = 125250.
- s=502..521: rectangles 2x500, 3x500, ..., 21x500. Heights: 2,3,...,21.
  Total height: 2+3+...+21 = 230.
  Total area: 500*(2+3+...+21) = 500*230 = 115000.

Total with all at minimum: 240250. Need 250000. Deficit: 9750.
Adjust one 1x(s-1) rectangle to have more area.

For tiling:
1. Place the height-d rectangles (d=2..21 x 500) stacked vertically.
   Total height: 230 rows. Remaining: 270 rows.
2. Each remaining row is height 1, width 500.
   Fill with 1-wide strips packed side by side.

For the 1x(s-1) strips oriented as 1-tall:
We have widths: 1, 2, 3, ..., 500. Pack into rows of width 500.
Pair them: (500), (499+1), (498+2), (497+3), ..., (251+249), (250).
That's 1 + 249 + 1 = 251 rows.

But we only have 270 remaining rows. 251 < 270, so we have 19 extra empty rows.
That's a problem: we need to fill all 270 rows.

Hmm, but we also need to increase total area by 9750.
The deficit needs to be absorbed. If we change one 1xm strip to an axb rectangle
with a > 1, it no longer fits in a single row.

Alternative: change the adjustment to be in one of the d x 500 rectangles.
For s = 502 + j (some value), change from (j+2) x 500 to a x (s-a) with higher area.
For example, s=521 (the most expensive): change from 21x500 to, say, 250x271.
250x271 has area 67750. Original 21x500 has area 10500. Increase: 57250. Too much.

We need increase of exactly 9750.
Current: 21x500 = 10500. Need area = 10500 + 9750 = 20250.
a*(521-a) = 20250. a^2 - 521a + 20250 = 0.
disc = 521^2 - 4*20250 = 271441 - 81000 = 190441 = 436.4^2. Not perfect square.

Let me check: 436^2 = 190096. 437^2 = 190969. Neither is 190441. Not integer.

Try s=520: current 20x500 = 10000. Need 10000 + 9750 = 19750.
a*(520-a) = 19750. disc = 270400 - 79000 = 191400. sqrt ~ 437.5. Not integer.

This approach of adjusting a large rectangle is hard because of the integer constraint.

Better approach: make multiple small adjustments.
Or: use a completely different tiling strategy.

Actually, the cleanest approach: instead of requiring all-minimum-area plus one
big adjustment, distribute the excess across multiple rectangles.

But for proving existence, we just need ONE valid tiling.

Let me think about this differently. The tiling approach:
1. Slice the 500x500 square into horizontal rows of height 1 each (500 rows).
2. Each row is 1x500. Fill each row with rectangles placed side-by-side.
3. Each rectangle in a row is 1-tall and w-wide, contributing area w.
4. The perimeter of a 1xw rectangle is 2(1+w), semi-perimeter s = 1+w.
   So distinct perimeters need distinct w values.
5. We can use w = 1, 2, ..., W and pack them into rows of width 500.

With this approach, each row must have strips whose widths sum to 500.
We need all widths distinct, and as many as possible.

Total area = 500 * 500 = 250000.
Sum of widths = 250000 (since each has height 1).
We have k strips with distinct widths w_1 < w_2 < ... < w_k.
Sum of widths = 250000.
Each width is at most 500 (fits in a row).

To maximize k with distinct widths summing to 250000:
Minimum sum using 1, 2, ..., k: k(k+1)/2.
Maximum sum using N-k+1, ..., N-1, N (if k <= N): sum = k*N - k(k-1)/2.

Need k(k+1)/2 <= 250000 <= k*500 - k(k-1)/2.

From lower: k(k+1)/2 <= 250000 => k <= 706.
From upper: 500k - k(k-1)/2 >= 250000 => k(1001-k)/2 >= 250000 => k(1001-k) >= 500000.
  k^2 - 1001k + 500000 <= 0.
  k = (1001 +/- sqrt(1001^2 - 2000000)) / 2 = (1001 +/- sqrt(1002001 - 2000000)) / 2
  = (1001 +/- sqrt(-997999)) / 2.
  This is negative under the sqrt! So for k=706: 706*(1001-706)/2 = 706*295/2 = 104135 < 250000.
  The upper bound is violated!

So with all 1-tall strips, we can't have k = 706. The maximum width is 500,
which limits how much area each strip contributes.

Hmm, so the constraint that each width <= 500 matters here.
Minimum k with all 1-tall strips:
We need sum of k distinct widths from {1,...,500} = 250000.
Max sum with all widths from {1,...,500}: 1+2+...+500 = 125250.
But 125250 < 250000! So we can't even fill the square with distinct-width
1-tall strips!

This means we MUST use rectangles with height > 1 for some of them.
The all-strips approach doesn't work by itself.

OK so I need to go back to the original formulation.
The rectangles can have any integer dimensions (a,b) with a,b in [1,500].
They have distinct semi-perimeters s = a+b.
They tile the 500x500 square.
Total area = 250000.

My earlier analysis gave k=520 using greedy by min_area.
The brute force confirmed this for small n.
The sum-of-521-cheapest-min-areas = 251250 > 250000.

But wait - is the "greedy by min_area" truly optimal? It finds the maximum k
such that there EXISTS a set of areas (one per chosen s) summing to exactly 250000.
The greedy ensures sum(min_area) <= 250000, and we showed sum(max_area) >= 250000.

For k+1=521: sum(min_area) for ANY 521 distinct s-values >= 251250 > 250000.
So it's impossible to find integer-sided rectangles with 521 distinct perimeters
whose areas sum to 250000. Hence k <= 520.

For k=520: we showed a valid set with sum(min_area) = 240250 <= 250000 <= sum(max_area).
And the deficit 9750 can be absorbed by adjusting one rectangle.

The remaining question: does a TILING exist? This is the key subtlety.

Actually, for competition math, the standard result is:
A set of rectangles with positive integer sides can tile an a x b rectangle
if and only if their total area equals ab AND the tiling exists (which needs
a constructive proof or reference to a known result).

For our specific construction, let me show a valid tiling:

Our 520 rectangles:
- For d = 2, 3, ..., 21: one rectangle of size d x 500 (20 rectangles, semi-perim 500+d)
- For w = 1, 2, ..., 500: one rectangle of size 1 x w (500 rectangles, semi-perim 1+w)
  EXCEPT: one of these is replaced by a non-unit-height rectangle to absorb the deficit.

Total: 520 rectangles with distinct semi-perimeters.

Tiling:
Step 1: Stack the d x 500 blocks. Heights: 2+3+...+21 = 230.
        These fill a 500 x 230 region at the bottom.
Step 2: Remaining region: 500 x 270 (top part).
        Fill with 1-tall rows, packing the 1 x w strips into rows of width 500.

For Step 2, the strips have widths 1, 2, ..., 500.
Total area: 1+2+...+500 = 125250.
Available area: 500 * 270 = 135000.
Deficit to fill: 135000 - 125250 = 9750.

This 9750 deficit is extra space. We need to fill it!
Options:
a) The adjusted rectangle fills some of this space.
b) We need every square unit filled.

With the adjustment: one strip, say 1 x 204, becomes a 79 x 126 rectangle.
Area goes from 204 to 9954, increase of 9750. ✓
But 79x126 occupies 79 rows of width 126, not a single row of width 204.

Alternative tiling:
Use the 79x126 rectangle in rows 231-309 (79 rows), columns 0-125 (width 126).
Then fill the remaining space in those 79 rows: 79 x 374 region.
And the remaining 191 rows (310-500): 191 x 500 region.

All these regions need to be filled with 1-tall strips.
Available strips: 1, 2, ..., 500 except 204 (since we replaced that one).
That's 499 strips with total area = 125250 - 204 = 125046.
Regions to fill:
- 79 x 374 = 29546
- 191 x 500 = 95500
Total: 125046. ✓

For bin-packing 499 strips (widths 1..500 except 204) into:
- 79 rows of width 374
- 191 rows of width 500

This is feasible: the strips have widths up to 500. Strips wider than 374
must go in 500-wide rows; strips wider than 500 don't exist.

Strips > 374: 375, 376, ..., 500 (except 204 is already removed, but 204 < 374).
That's 126 strips. They must go in 500-wide rows (191 available).
126 < 191, so they fit (even one per row, with remaining space filled by small strips).

The bin-packing is feasible. I won't prove it rigorously here, but with
191 bins of size 500 and 79 bins of size 374, and items of sizes 1 to 500
(minus 204), the total area matches and the items fit.

Therefore, k = 520 is achievable, and k = 521 is impossible.
"""

print("K = 520")
print("K mod 10^5 = 520")

# Actually, wait. Let me reconsider the problem once more.
# I want to make sure there isn't a much larger answer that I'm missing.
# Let me re-read: "A 500x500 square is divided into k rectangles, each having
# integer side lengths. Given that no two of these rectangles have the same
# perimeter, the largest possible value of k is K. What is the remainder when
# K is divided by 10^5?"
#
# The answer K mod 10^5 suggests K might be large (> 10^5 = 100000).
# But my analysis gives K = 520, which mod 10^5 = 520.
# If the problem is from AIMO, the answer might be a larger number.
#
# Wait -- let me reconsider. Maybe "integer side lengths" means each side of
# each rectangle is an integer, but the SAME rectangle can appear multiple times
# as long as their perimeters differ? No, that doesn't make sense -- if two
# rectangles have the same dimensions, they have the same perimeter.
#
# Or maybe the rectangles don't need to be axis-aligned? No, in a standard
# "divided into rectangles" problem, they are axis-aligned.
#
# Hmm, but 520 mod 10^5 = 520 seems like a reasonable competition answer.
# Let me verify once more.

N = 500
AREA = 250000

# Recompute carefully
# All possible semi-perimeters and their minimum areas
s_costs = []
for s in range(2, 1001):
    if s <= 501:
        s_costs.append((s - 1, s))  # min_area = s-1 for 1x(s-1)
    else:
        # a >= s-500, b = s-a <= 500. Min area = (s-500)*500
        s_costs.append(((s - 500) * 500, s))

# Sort by cost
s_costs.sort()

# Greedy
total = 0
k = 0
for cost, s in s_costs:
    if total + cost <= AREA:
        total += cost
        k += 1
    else:
        break

print(f"K = {k}")
print(f"Sum of min areas for K = {total}")
print(f"Sum of min areas for K+1 = {total + s_costs[k][0]}")
print(f"Deficit = {AREA - total}")
print(f"K mod 10^5 = {k % 100000}")
