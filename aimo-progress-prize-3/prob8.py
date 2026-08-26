from sympy import totient, cyclotomic_poly
from sympy import Symbol
from collections import defaultdict
from itertools import combinations

x = Symbol('x')

even_n_list = []
for n in range(2, 200):
    if n % 2 == 0 and totient(n) <= 8:
        even_n_list.append(n)

print("Even n with phi(n) <= 8:", even_n_list)

# Group by 2-adic valuation
groups = defaultdict(list)
for n in even_n_list:
    v2 = 0
    tmp = n
    while tmp % 2 == 0:
        v2 += 1
        tmp //= 2
    groups[v2].append(n)

print("\nGroups by v_2:")
for v2 in sorted(groups.keys()):
    ns = groups[v2]
    degs = [totient(n) for n in ns]
    print(f"  v_2 = {v2}: n = {ns}, degrees = {degs}")

count_subsets = 0
# Empty product (A = 1)
count_subsets += 1

for v2 in sorted(groups.keys()):
    ns = groups[v2]
    degs = [totient(n) for n in ns]
    cnt = 0
    for r in range(1, len(ns)+1):
        for combo in combinations(range(len(ns)), r):
            total_deg = sum(degs[i] for i in combo)
            if total_deg <= 8:
                cnt += 1
                if total_deg <= 8:
                    names = [ns[i] for i in combo]
                    # print(f"    Subset: Phi_{names}, deg={total_deg}")
    print(f"Group v_2={v2}: {cnt} non-empty subsets with deg <= 8")
    count_subsets += cnt

print(f"\nTotal valid A(x) up to sign: {count_subsets}")
print(f"Total shifty functions (with +/- sign): {2 * count_subsets}")
