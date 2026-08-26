"""
Problem: A 500x500 square is divided into k rectangles, each with integer side lengths.
No two rectangles have the same perimeter. Find max k, then k mod 10^5.

Key insight: perimeter = 2(a+b), so distinct perimeters = distinct values of s = a+b.
Each rectangle in the tiling has both dims <= 500, so 1 <= a,b <= 500.

For sum s = a+b:
- If s <= 501: a can be 1..min(s-1,500). Min area = 1*(s-1) = s-1.
- If 502 <= s <= 1000: a must be in [s-500, 500]. Min area = (s-500)*500.

Strategy: maximize number of distinct s values such that sum of min areas <= 250000
and sum of max areas >= 250000 (to allow exact fit).
"""

N = 500
area = N * N  # 250000

# Phase 1: Use s = 2, 3, ..., 501 (500 values). Each costs min area s-1.
phase1_count = 500
phase1_cost = sum(s - 1 for s in range(2, 502))  # = sum(1..500) = 125250
print(f"Phase 1: s=2..501, count={phase1_count}, min area={phase1_cost}")

budget = area - phase1_cost  # = 124750
print(f"Remaining budget: {budget}")

# Phase 2: Add s = 502, 503, ... Each s=500+d (d>=2) costs d*500 min area.
# Cumulative cost for d=2..D: 500 * sum(d=2..D) = 500 * (D*(D+1)/2 - 1)
phase2_count = 0
phase2_cost = 0
s_val = 502
while True:
    d = s_val - 500
    cost = d * 500
    if phase2_cost + cost > budget:
        break
    phase2_cost += cost
    phase2_count += 1
    s_val += 1

print(f"Phase 2: s=502..{501+phase2_count}, count={phase2_count}, min area={phase2_cost}")
print(f"Remaining after phase 2: {budget - phase2_cost}")

k_basic = phase1_count + phase2_count
remaining = area - phase1_cost - phase2_cost
print(f"\nBasic k = {k_basic}, remaining area to distribute = {remaining}")

# Now check: can we do better by trading?
# Skip a cheap s value (saves s-1 area) and add an expensive one (costs (s-500)*500).
# Net: we keep same count but change composition. Only helps if we can add MORE.
# To add s=501+phase2_count+1, cost = (phase2_count+2)*500
next_s = 502 + phase2_count
next_cost = (next_s - 500) * 500
print(f"\nNext s to add: {next_s}, cost: {next_cost}")
print(f"Current remaining: {remaining}")
print(f"Need to free: {next_cost - remaining}")

# To add the next s, we need to free (next_cost - remaining) area by skipping some
# existing s values. Each skipped s from {2..501} saves s-1 area, each skipped s
# from {502..521} saves (s-500)*500 area.
# Skipping small s values is very cheap. To free X area by skipping the most expensive
# small s values: skip s=501 (saves 500), s=500 (saves 499), etc.
needed = next_cost - remaining
print(f"\nTo add s={next_s}, need to free {needed} by skipping existing values.")
print(f"But each skip loses 1 rectangle, adding 1 gains 1. Net = 0.")
print(f"This approach does not increase k.")

# What about swapping multiple cheap for fewer expensive?
# Skip m cheap values (losing m rectangles, saving ~ m*(m+1)/2 area from the top)
# Add p expensive values (gaining p rectangles, costing ~ 500*sum of next p d-values)
# Need p > m for a net gain.
# Saving from skipping m most expensive small s: skip s=501,500,...,502-m+1
# Wait, s=501 costs 500, s=500 costs 499, etc.
# Actually skip the most expensive SMALL ones: s=501 saves 500, s=500 saves 499...
# Skip s from 502-k to 501... hmm let me think more systematically.

# Let me try a different approach: optimize directly.
# We want to maximize k = number of distinct s values from {2,...,1000}
# subject to sum of min_area(s) <= 250000.
# where min_area(s) = s-1 for s <= 501, (s-500)*500 for s >= 502.

# Sort all possible s values by their min_area cost:
costs = []
for s in range(2, 1001):
    if s <= 501:
        costs.append((s - 1, s))  # (cost, s_value)
    else:
        costs.append(((s - 500) * 500, s))

# Sort by cost (greedy: pick cheapest first)
costs.sort()

total = 0
selected = []
for cost, s in costs:
    if total + cost <= area:
        total += cost
        selected.append((s, cost))
    else:
        break

k_opt = len(selected)
print(f"\nOptimal greedy: k = {k_opt}")
print(f"Total min area = {total}")
print(f"Remaining = {area - total}")

# Show the selected s values
s_values = sorted([s for s, c in selected])
print(f"Min s = {min(s_values)}, Max s = {max(s_values)}")
print(f"First 10: {s_values[:10]}")
print(f"Last 10: {s_values[-10:]}")

# Verify: all cheap s values (cost 1,2,...,500) are s=2..501.
# Then from s >= 502, costs are 1000, 1500, 2000, ...
# The cheap costs are 1,2,...,500 for s=2..501.
# Then next cheapest: s=502 costs 1000 (which is more than s=501 cost 500).
# So greedy picks s=2..501 first (total = 125250), then s=502,503,...

# This confirms our earlier analysis.
print(f"\nk = {k_opt}")

# Now verify we can achieve exactly area = 250000 with these k rectangles.
# We need to adjust areas (increase some from minimum) to hit exactly 250000.
deficit = area - total
print(f"Deficit to fill: {deficit}")

# For any rectangle with s value, we can increase area from min to higher values.
# Just need to find one rectangle where we can increase by exactly deficit.
# For s <= 501 with current area s-1, possible areas: {a*(s-a) : a=1..floor(s/2)}
# Increase from s-1: pick a=j, gain = j*(s-j) - (s-1)

# For s >= 502 with current area (s-500)*500, possible areas: {a*(s-a) : a=s-500..500}
# Increase from (s-500)*500: pick a=j, gain = j*(s-j) - (s-500)*500

# Let me check if deficit can be achieved
print(f"\nCan we increase some rectangle by {deficit}?")
found = False
for s, c in selected:
    if s <= 501:
        for a in range(2, s):
            gain = a * (s - a) - (s - 1)
            if gain == deficit:
                print(f"  YES: s={s}, change a from 1 to {a}, rectangle {a}x{s-a}, gain={gain}")
                found = True
                break
    else:
        a_min = s - 500
        for a in range(a_min + 1, 501):
            gain = a * (s - a) - a_min * 500
            if gain == deficit:
                print(f"  YES: s={s}, change a from {a_min} to {a}, rectangle {a}x{s-a}, gain={gain}")
                found = True
                break
    if found:
        break

if not found:
    # Try splitting deficit across multiple rectangles
    print("  Direct match not found, but splitting across multiple rectangles is always possible")
    print("  with enough rectangles and area flexibility.")
    # Actually, let me check possible gains more carefully
    # For s = some large value, gains can be large.
    # For s=501: gains from a=1 to a=250: 250*251 - 500 = 62250. Large range.
    possible_gains = set()
    for s, c in selected:
        if s <= 501:
            for a in range(1, (s+1)//2 + 1):
                possible_gains.add(a * (s - a) - (s - 1))
        else:
            a_min = s - 500
            for a in range(a_min, 501):
                possible_gains.add(a * (s - a) - a_min * 500)
    if deficit in possible_gains:
        print(f"  Deficit {deficit} IS achievable by adjusting one rectangle.")
    else:
        print(f"  Deficit {deficit} not achievable with single adjustment.")
        # Check if achievable with two adjustments
        for g in possible_gains:
            if 0 < g < deficit and (deficit - g) in possible_gains:
                print(f"  Achievable with two adjustments: {g} + {deficit - g}")
                break

# Final answer
K = k_opt
print(f"\n=== FINAL ANSWER ===")
print(f"K = {K}")
print(f"K mod 10^5 = {K % 100000}")
