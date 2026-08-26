"""
Problem: A 500x500 square is divided into k rectangles with integer side lengths.
No two rectangles have the same perimeter. Find max k, then k mod 10^5.

Re-examination: In a tiling, each rectangle must fit in the 500x500 square.
Both dimensions of each rectangle are at most 500.

Perimeter = 2(a+b) where 1 <= a <= b <= 500. So a+b ranges from 2 to 1000.
Distinct perimeters <=> distinct a+b values.

For each s = a+b, the rectangle has dimensions a x (s-a) where:
  max(1, s-500) <= a <= s//2 (since a <= b = s-a means a <= s/2)

The area = a*(s-a).
  Min area: a*(s-a) is minimized at the boundary of the a range.
    - For s <= 501: a=1, area = s-1
    - For s >= 502: a = s-500, area = (s-500)*500

Total area must equal 250000.
"""

N = 500
AREA = N * N  # 250000

def min_area(s):
    """Minimum area for a rectangle with perimeter 2s and both sides <= 500."""
    if s <= 501:
        return s - 1  # 1 x (s-1)
    else:
        return (s - 500) * 500  # (s-500) x 500

def max_area(s):
    """Maximum area for a rectangle with perimeter 2s and both sides <= 500."""
    # a*(s-a) maximized at a = s//2
    a = s // 2
    # But also need a >= max(1, s-500)
    a = max(a, max(1, s - 500))
    # And a <= min(s//2, 500)
    a = min(a, min(s // 2, 500))
    return a * (s - a)

# Greedy: sort all possible s values by min_area, pick cheapest first
all_s = list(range(2, 1001))
all_s.sort(key=min_area)

total_min = 0
selected = []
for s in all_s:
    c = min_area(s)
    if total_min + c <= AREA:
        total_min += c
        selected.append(s)

k = len(selected)
deficit = AREA - total_min
print(f"Greedy k = {k}")
print(f"Total min area = {total_min}")
print(f"Deficit = {deficit}")
print(f"Selected s range: {min(selected)} to {max(selected)}")

# Verify greedy is optimal. Since we sorted by min_area and picked all that fit,
# this is optimal for maximizing count under a budget constraint (classic knapsack
# with unit weights, sorted by cost).

# Actually wait - this is NOT a standard knapsack. The constraint is:
# sum of areas = AREA (exact), not sum <= AREA.
# We need sum of min_area <= AREA and sum of max_area >= AREA for the selected set.
# The greedy gives sum of min_area <= AREA. But we also need the flexibility to
# increase areas to hit exactly AREA.

# Check: can we increase areas by the deficit?
total_max = sum(max_area(s) for s in selected)
print(f"Total max area = {total_max}")
print(f"Total max >= AREA? {total_max >= AREA}")

# Can we also check k+1? If we add the next cheapest s value, is the
# constraint sum(min_area) > AREA truly binding?
remaining_s = [s for s in all_s if s not in set(selected)]
if remaining_s:
    next_s = min(remaining_s, key=min_area)
    print(f"\nNext s to add: {next_s}, cost: {min_area(next_s)}")
    print(f"Total + next = {total_min + min_area(next_s)} > {AREA}? {total_min + min_area(next_s) > AREA}")

    # Could we swap? Remove the most expensive selected s and add this one?
    # That changes count by 0 (remove 1, add 1). No gain.

    # Could we remove 1 selected s that costs more than next_s, add two cheaper?
    # We can only add s values not already selected. The next cheapest unselected
    # is next_s. There might not be another cheap one.

    # Actually, what if we can INCREASE some rectangle's area to compensate for
    # a reduced budget? If we add s=next_s, our min total becomes total_min + min_area(next_s).
    # If that's > AREA, we're stuck because we can't reduce any rectangle below its min area.

    # BUT: what if we REMOVE a rectangle with a large min_area and replace it with
    # two rectangles with smaller min_areas? Then we gain 1 rectangle.
    # This is the swap: remove s_big, add s_new1 and s_new2.
    # Net gain: +1 rectangle.
    # Need: total_min - min_area(s_big) + min_area(s_new1) + min_area(s_new2) <= AREA.
    # And: s_new1, s_new2 not in selected set after removal.

    # The most expensive s in our selected set:
    selected_sorted = sorted(selected, key=min_area, reverse=True)
    most_expensive = selected_sorted[0]
    print(f"\nMost expensive selected: s={most_expensive}, cost={min_area(most_expensive)}")

    # After removing s_big, we free min_area(s_big).
    # The two cheapest available s values:
    available_after_remove = sorted([s for s in all_s if s not in set(selected) or s == most_expensive], key=min_area)
    if len(available_after_remove) >= 2:
        # But wait, we need to account for the removed s being re-available
        # Actually: remove most_expensive from selected. Now available pool includes most_expensive.
        # But we want two NEW ones (not most_expensive, since we removed it to make room).
        # Hmm, we want two that aren't in selected (except most_expensive is now available).

        new_selected = set(selected) - {most_expensive}
        available = sorted([s for s in all_s if s not in new_selected], key=min_area)

        if len(available) >= 2:
            s1, s2 = available[0], available[1]
            new_total = total_min - min_area(most_expensive) + min_area(s1) + min_area(s2)
            print(f"Swap: remove s={most_expensive} (save {min_area(most_expensive)})")
            print(f"  Add s={s1} (cost {min_area(s1)}) and s={s2} (cost {min_area(s2)})")
            print(f"  New total min area: {new_total}")
            print(f"  Fits? {new_total <= AREA}")

            if new_total <= AREA:
                print(f"  YES! k increases to {k + 1}!")
                # Keep doing this until we can't anymore

# Let me do this optimization systematically
print("\n=== Systematic swap optimization ===")

# Start with greedy solution
selected_set = set(selected)
k_current = k
total_current = total_min

improvements = 0
while True:
    # Try to gain 1 rectangle by swapping:
    # Remove the most expensive rectangle, add 2 cheapest available
    sel_by_cost = sorted(selected_set, key=min_area, reverse=True)
    best_improvement = None

    for s_remove in sel_by_cost[:50]:  # try removing top 50 most expensive
        cost_saved = min_area(s_remove)
        new_set = selected_set - {s_remove}
        available = sorted([s for s in range(2, 1001) if s not in new_set], key=min_area)

        if len(available) < 2:
            continue

        s1, s2 = available[0], available[1]
        cost_added = min_area(s1) + min_area(s2)
        new_total = total_current - cost_saved + cost_added

        if new_total <= AREA:
            best_improvement = (s_remove, s1, s2, new_total)
            break

    if best_improvement is None:
        break

    s_remove, s1, s2, new_total = best_improvement
    selected_set.remove(s_remove)
    selected_set.add(s1)
    selected_set.add(s2)
    total_current = new_total
    k_current = len(selected_set)
    improvements += 1

    if improvements % 100 == 0:
        print(f"  After {improvements} swaps: k={k_current}, total min area={total_current}")

print(f"\nAfter all swaps: k={k_current}, total min area={total_current}")
print(f"Deficit = {AREA - total_current}")
print(f"Improvements made: {improvements}")

# Verify
assert total_current == sum(min_area(s) for s in selected_set)
assert total_current <= AREA
total_max_check = sum(max_area(s) for s in selected_set)
print(f"Total max area = {total_max_check}, >= AREA? {total_max_check >= AREA}")

# Final answer
K = k_current
print(f"\n=== FINAL ANSWER ===")
print(f"K = {K}")
print(f"K mod 10^5 = {K % 100000}")
