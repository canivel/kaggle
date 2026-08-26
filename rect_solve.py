"""
DEFINITIVE solution for: 500x500 square divided into k rectangles with
integer sides, all perimeters distinct. Find K (max k), then K mod 10^5.
"""

TARGET = 250000
MAX_SIDE = 500

# Build min_area table: for half-perimeter s, what's the minimum possible area
# given both sides must be in [1, 500]?
min_areas = {}
for s in range(2, 1001):
    a_min = max(1, s - MAX_SIDE)
    a_max = s // 2
    if a_min > a_max:
        continue
    # Area = a * (s - a), minimized at a = a_min (furthest from s/2)
    min_areas[s] = a_min * (s - a_min)

# Sort by min_area to greedily pick cheapest half-perimeters
sorted_by_cost = sorted(min_areas.items(), key=lambda x: (x[1], x[0]))

# Greedy: accumulate cheapest until we can't add more
cumulative = 0
selected = []
rejected_first = None
for s, ma in sorted_by_cost:
    if cumulative + ma <= TARGET:
        cumulative += ma
        selected.append(s)
    elif rejected_first is None:
        rejected_first = (s, ma)

k = len(selected)
deficit = TARGET - cumulative

print(f"Greedy selection: k = {k}")
print(f"Cumulative min area: {cumulative}")
print(f"Deficit to absorb: {deficit}")
print(f"First rejected: s={rejected_first[0]}, min_area={rejected_first[1]}")
print(f"Would exceed by: {cumulative + rejected_first[1] - TARGET}")
print()

# Verify the selected set
selected_set = set(selected)
print(f"Selected half-perimeters: {min(selected)} to {max(selected)}")
print(f"Contiguous? {selected == list(range(min(selected), max(selected)+1))}")
print()

# Can we absorb the deficit? Find integer-dimension adjustments summing to exactly deficit.
# For half-perimeter s currently using a=a_min, area = a_min*(s-a_min).
# Changing to a=a_min+d: new_area = (a_min+d)*(s-a_min-d), increase = new_area - old_area.
# We need increases from one or more rectangles summing to deficit.

print(f"Finding adjustments to absorb deficit = {deficit}:")
remaining = deficit

# Try single adjustment for each s
found_single = False
for s in selected:
    a_min_s = max(1, s - MAX_SIDE)
    base = a_min_s * (s - a_min_s)
    for a in range(a_min_s + 1, s // 2 + 1):
        inc = a * (s - a) - base
        if inc == remaining:
            print(f"  Single: s={s}, a: {a_min_s} -> {a}, area: {base} -> {a*(s-a)}, +{inc}")
            found_single = True
            break
    if found_single:
        break

if not found_single:
    # Try two adjustments
    print("  No single adjustment works. Trying two adjustments...")
    found_pair = False
    # Build dictionary of possible increases for each s
    all_increases = {}  # inc_value -> (s, new_a)
    for s in selected:
        a_min_s = max(1, s - MAX_SIDE)
        base = a_min_s * (s - a_min_s)
        for a in range(a_min_s + 1, s // 2 + 1):
            inc = a * (s - a) - base
            if inc <= remaining and inc not in all_increases:
                all_increases[inc] = (s, a)
            if inc > remaining:
                break

    for inc1 in sorted(all_increases.keys(), reverse=True):
        inc2 = remaining - inc1
        if inc2 > 0 and inc2 in all_increases:
            s1, a1 = all_increases[inc1]
            s2, a2 = all_increases[inc2]
            if s1 != s2:
                a_min_1 = max(1, s1 - MAX_SIDE)
                a_min_2 = max(1, s2 - MAX_SIDE)
                base1 = a_min_1 * (s1 - a_min_1)
                base2 = a_min_2 * (s2 - a_min_2)
                print(f"  Adj 1: s={s1}, a: {a_min_1} -> {a1}, area: {base1} -> {a1*(s1-a1)}, +{inc1}")
                print(f"  Adj 2: s={s2}, a: {a_min_2} -> {a2}, area: {base2} -> {a2*(s2-a2)}, +{inc2}")
                print(f"  Total increase: {inc1 + inc2} = {remaining}? {inc1+inc2 == remaining}")
                found_pair = True
                break

    if not found_pair:
        print("  ERROR: Could not find adjustments (should not happen)")

print()

# Now verify k+1 is impossible
# The minimum total area for ANY set of k+1 distinct half-perimeters is the sum
# of the (k+1) smallest min_areas.
k_plus_1_min = sum(ma for _, ma in sorted_by_cost[:k+1])
print(f"Min total area for k+1={k+1} rectangles: {k_plus_1_min}")
print(f"Target: {TARGET}")
print(f"k+1 is {'POSSIBLE' if k_plus_1_min <= TARGET else 'IMPOSSIBLE'}")
print()

# FINAL ANSWER
K = k
print("=" * 60)
print(f"K = {K}")
print(f"K mod 10^5 = {K % 100000}")
print("=" * 60)
