#!/usr/bin/env python3
"""
Problem 4 verification.

f(m) + f(n) = f(m + n + mn) = f((m+1)(n+1) - 1) for all m, n >= 1.
g(x) = f(x-1), then g(ab) = g(a) + g(b) for all a, b >= 2.
g is completely additive: g(n) = sum v_p(n) * g(p) over primes p dividing n.
g(p) >= 1 for all primes p.
f(n) = g(n+1) <= 1000 for all n <= 1000, i.e. g(k) <= 1000 for 2 <= k <= 1001.
f(2024) = g(2025) = g(3^4 * 5^2) = 4*g(3) + 2*g(5).

We set g(p) = 1 for all primes p != 3, 5 (this is the least restrictive for feasibility).

Constraints on (alpha, beta) = (g(3), g(5)):
For each k in [2, 1001], if k = 3^a * 5^b * (other primes):
  a * alpha + b * beta + (number of other prime factors with mult) <= 1000

We want: count of distinct values of 4*alpha + 2*beta.
"""

from sympy import factorint

# Enumerate ALL constraints
constraints = []
for k in range(2, 1002):
    f = factorint(k)
    v3 = f.get(3, 0)
    v5 = f.get(5, 0)
    other = sum(v for p, v in f.items() if p not in (3, 5))
    rhs = 1000 - other
    constraints.append((v3, v5, rhs))

# Find feasible (alpha, beta) with alpha >= 1, beta >= 1
values = set()
for alpha in range(1, 1001):
    max_beta = 10**9
    feasible = True
    for v3, v5, rhs in constraints:
        rem = rhs - v3 * alpha
        if rem < 0:
            # Even if v5 = 0, this constraint is v3*alpha <= rhs which fails
            if v5 == 0 and v3 > 0:
                feasible = False
                break
            elif v5 > 0:
                feasible = False
                break
            # if v3 = v5 = 0, rhs = 1000 - omega(k) >= 0 always, rem = rhs >= 0
        elif v5 > 0:
            max_beta = min(max_beta, rem // v5)

    if not feasible:
        break
    if max_beta < 1:
        continue

    for beta in range(1, max_beta + 1):
        values.add(4 * alpha + 2 * beta)

print(f"Number of distinct f(2024) values: {len(values)}")
print(f"Min: {min(values)}, Max: {max(values)}")

# Verify: max alpha
for v3, v5, rhs in constraints:
    if v3 > 0 and v5 == 0:
        max_a = rhs // v3
        if max_a < 200:
            print(f"  Constraint {v3}*alpha <= {rhs}: alpha <= {max_a}")

# Max beta
for v3, v5, rhs in constraints:
    if v5 > 0 and v3 == 0:
        max_b = rhs // v5
        if max_b < 300:
            print(f"  Constraint {v5}*beta <= {rhs}: beta <= {max_b}")

# Verify some specific function values
# alpha=166, beta=1: 4*166+2*1 = 666
# Check: 6*166 = 996 <= 1000? Yes (from k=729=3^6, rhs=1000-6=994... wait)
# k=729=3^6, other primes=0, so omega(729)=6, rhs=1000-0=1000... no.
# Wait: other = sum of v_p for p != 3,5. For k=729=3^6, other=0.
# rhs = 1000 - 0 = 1000. Constraint: 6*alpha + 0*beta <= 1000.
# alpha <= 166.

# alpha=166: check all constraints
alpha = 166
max_beta_166 = 10**9
for v3, v5, rhs in constraints:
    rem = rhs - v3 * alpha
    if rem < 0 and (v3 > 0 or v5 > 0):
        print(f"  INFEASIBLE at alpha=166: {v3}*166 + {v5}*beta <= {rhs}")
        break
    if v5 > 0:
        max_beta_166 = min(max_beta_166, rem // v5)
print(f"alpha=166: max_beta = {max_beta_166}")
# 4*166 + 2*beta, beta in [1, max_beta_166]
# Values: 664+2 to 664+2*max_beta_166

# alpha=1: max beta
alpha = 1
max_beta_1 = 10**9
for v3, v5, rhs in constraints:
    rem = rhs - v3 * 1
    if v5 > 0:
        max_beta_1 = min(max_beta_1, rem // v5)
print(f"alpha=1: max_beta = {max_beta_1}")
# 4*1 + 2*beta = 4 + 2*beta, beta in [1, max_beta_1]
# Values: 6 to 4+2*max_beta_1

# Check that all even values from 6 to max are achieved
sorted_v = sorted(values)
for i in range(len(sorted_v) - 1):
    if sorted_v[i+1] - sorted_v[i] != 2:
        print(f"GAP: {sorted_v[i]} -> {sorted_v[i+1]}")

print(f"\nAll even from {min(values)} to {max(values)}: {len(values)} values")
print(f"Expected if contiguous: {(max(values) - min(values))//2 + 1}")
print(f"Match: {len(values) == (max(values) - min(values))//2 + 1}")

print(f"\nFINAL ANSWER: {len(values)}")
