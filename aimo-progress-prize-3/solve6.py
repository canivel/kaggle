"""
Final verification: is there any way to beat k=520?

The greedy algorithm picks all s-values sorted by min_area, in increasing order.
For n=500:
  s=2: cost 1
  s=3: cost 2
  ...
  s=501: cost 500
  s=502: cost 1000
  s=503: cost 1500
  ...

The 500 cheapest are s=2..501 (costs 1..500), totaling 125250.
The next 20 cheapest are s=502..521 (costs 1000..10500), totaling 115000.
Grand total: 240250. Budget = 250000. Remaining: 9750.

The next s=522 costs 11000 > 9750. So we can't add it.

Could we use a DIFFERENT set of 521 values that costs <= 250000?
To beat greedy, we'd need to find 521 values with sum of min_areas <= 250000.
The greedy already minimizes the sum for any given k, since it picks cheapest first.
Any other set of 521 values would have sum >= 240250 + 11000 - 10500 = 240750
(by swapping the most expensive selected with the cheapest unselected).
Still < 250000? Let me check.

Actually, to get k=521, we need the cheapest 521 s-values.
The 521st cheapest is s=522 with cost 11000.
Sum of cheapest 521 = 240250 + 11000 = 251250 > 250000.

So k=521 is impossible: any 521 distinct s-values have min area sum >= 251250 > 250000.

Wait, that's only true if s=522 is truly the 521st cheapest. Let me verify the ordering.
"""

N = 500
AREA = N * N

def min_area(s):
    if s <= N + 1:  # s <= 501
        return s - 1
    else:
        return (s - N) * N

# List all s values and their costs
all_s = [(min_area(s), s) for s in range(2, 2*N + 1)]
all_s.sort()

# The 520th and 521st cheapest
print("Top of sorted s-values:")
for i, (cost, s) in enumerate(all_s[:525]):
    if i >= 518:
        print(f"  #{i+1}: s={s}, cost={cost}")

print(f"\nSum of cheapest 520: {sum(c for c, s in all_s[:520])}")
print(f"Sum of cheapest 521: {sum(c for c, s in all_s[:521])}")
print(f"Budget: {AREA}")

# Verify ordering around the boundary
print("\nAround boundary:")
for i in range(498, 525):
    cost, s = all_s[i]
    print(f"  #{i+1}: s={s}, cost={cost}")

# Confirm: cheapest 520 sum = 240250, cheapest 521 sum = 251250
sum_520 = sum(c for c, s in all_s[:520])
sum_521 = sum(c for c, s in all_s[:521])
print(f"\nSum of 520 cheapest = {sum_520}")
print(f"Sum of 521 cheapest = {sum_521}")
print(f"521 cheapest > {AREA}? {sum_521 > AREA}")

# So max k = 520 is confirmed.
K = 520
print(f"\nK = {K}")
print(f"K mod 10^5 = {K % 100000}")

# Let me also verify with the formula for general n
def solve(n):
    area = n * n
    # Phase 1: s=2..n+1, cost = n*(n+1)/2
    p1 = n * (n + 1) // 2
    budget = area - p1  # = n*(n-1)/2
    # Phase 2: s = n+2, n+3, ... costs n*2, n*3, ...
    # Cumulative for d extra values: n * sum(j=2..d+1) = n*(d*(d+3)/2 - ... )
    # Actually d values: s = n+2, ..., n+d+1, costs n*2, n*3, ..., n*(d+1)
    # Cumulative: n * sum(j=2..d+1) = n * ((d+1)*(d+2)/2 - 1)
    d = 0
    cum = 0
    while True:
        d += 1
        cum += n * (d + 1)
        if cum > budget:
            d -= 1
            cum -= n * (d + 1 + 1)  # undo
            break
    # Recalculate properly
    d = 0
    cum = 0
    while True:
        next_cost = n * (d + 2)  # cost for s = n + d + 2
        if cum + next_cost > budget:
            break
        cum += next_cost
        d += 1
    return n + d

for n in [2, 3, 4, 5, 10, 20, 50, 100, 200, 500]:
    print(f"n={n}: k={solve(n)}")

# Final: n=500
print(f"\nFinal: K = {solve(500)}, K mod 10^5 = {solve(500) % 100000}")
