#!/usr/bin/env python3
"""
Verify Problem 4 solution more carefully.

f(m) + f(n) = f(m + n + mn) for all m, n >= 1
m + n + mn = (m+1)(n+1) - 1

Let g(x) = f(x-1), so g is defined for x >= 2 (since f is defined for x >= 1).
f(m) = g(m+1), f(n) = g(n+1), f(m+n+mn) = g(m+n+mn+1) = g((m+1)(n+1)).

So: g(a) + g(b) = g(ab) for all a, b >= 2.

This means g is completely additive on the multiplicative monoid {2, 3, 4, ...}.

For any n >= 2, n = p1^{e1} * ... * pk^{ek},
g(n) = e1*g(p1) + ... + ek*g(pk).

Proof: g(p^2) = g(p*p) = g(p) + g(p) = 2g(p). By induction g(p^e) = e*g(p).
g(p*q) = g(p) + g(q) for distinct primes p, q.
g(p^a * q^b) = g(p^a) + g(q^b) = a*g(p) + b*g(q). (Using g(xy) = g(x)+g(y) repeatedly.)

Wait, but g(ab) = g(a) + g(b) only for a, b >= 2. So g(4) = g(2*2) = g(2)+g(2) = 2g(2). OK.
g(6) = g(2*3) = g(2)+g(3). OK.
g(12) = g(2*6) = g(2)+g(6) = g(2)+g(2)+g(3) = 2g(2)+g(3).
Also g(12) = g(3*4) = g(3)+g(4) = g(3)+2g(2). Consistent.

What about g(8) = g(2*4) = g(2)+g(4) = g(2)+2g(2) = 3g(2). OK.

So g(n) = sum_p v_p(n) * g(p) for all n >= 2. Confirmed.

Now, g: {2,3,...} -> Z>=1 (since f: Z>=1 -> Z>=1 means g(x) = f(x-1) >= 1 for x >= 2).

For a prime p: g(p) >= 1, so g(p) is a positive integer.
For composite n with Omega(n) >= 2: g(n) = sum v_p(n)*g(p) >= sum v_p(n) = Omega(n) >= 2 >= 1. OK.

So the only constraints on the g(p) values are:
1) g(p) >= 1 for all primes p
2) f(n) <= 1000 for all n <= 1000, i.e., g(n+1) <= 1000 for 1 <= n <= 1000,
   i.e., g(k) <= 1000 for 2 <= k <= 1001.

We want f(2024) = g(2025).
2025 = 3^4 * 5^2.
g(2025) = 4*g(3) + 2*g(5).

The constraints g(k) <= 1000 for 2 <= k <= 1001 translate to:
For each k in [2, 1001]: sum_{p|k} v_p(k) * g(p) <= 1000.

The g(p) for primes p not equal to 3 or 5 don't affect g(2025), so we want
to find the feasible range of (g(3), g(5)) subject to:
- g(p) >= 1 for all primes p
- For each k in [2,1001]: sum_{p|k} v_p(k)*g(p) <= 1000

For a given (g(3), g(5)), we set g(p) = 1 for all other primes to minimize
the LHS of each constraint. If constraints are satisfied with g(p)=1 for other primes,
then (g(3), g(5)) is feasible.

So the effective constraints are:
For each k in [2, 1001]:
v_3(k)*g(3) + v_5(k)*g(5) + sum_{p|k, p!=3,5} v_p(k) <= 1000

Let alpha = g(3), beta = g(5), both >= 1.
For each k: v_3(k)*alpha + v_5(k)*beta <= 1000 - sum_{p|k,p!=3,5} v_p(k)

The binding constraints are from k with large v_3 and/or v_5 and small other contributions.

Now let me carefully enumerate feasible (alpha, beta) and count distinct 4*alpha + 2*beta.
"""

from sympy import factorint

constraints_35 = []
for k in range(2, 1002):
    f = factorint(k)
    v3 = f.get(3, 0)
    v5 = f.get(5, 0)
    other = sum(v for p, v in f.items() if p != 3 and p != 5)
    rhs = 1000 - other
    if v3 > 0 or v5 > 0:
        constraints_35.append((v3, v5, rhs))

# Remove dominated constraints
# A constraint (a1, b1, r1) is dominated by (a2, b2, r2) if
# a2 >= a1, b2 >= b1, r2 <= r1 (strictly more restrictive)
# Keep only non-dominated constraints
def remove_dominated(constraints):
    result = []
    n = len(constraints)
    for i in range(n):
        a1, b1, r1 = constraints[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            a2, b2, r2 = constraints[j]
            # j dominates i if for all (alpha, beta) >= 1:
            # a2*alpha + b2*beta <= r2 implies a1*alpha + b1*beta <= r1
            # This happens if a2 >= a1 and b2 >= b1 and r2 <= r1
            if a2 >= a1 and b2 >= b1 and r2 <= r1 and (a2 > a1 or b2 > b1 or r2 < r1):
                dominated = True
                break
        if not dominated:
            result.append((a1, b1, r1))
    return result

non_dom = remove_dominated(constraints_35)
print(f"Non-dominated constraints: {len(non_dom)}")
for c in sorted(non_dom):
    print(f"  {c[0]}*alpha + {c[1]}*beta <= {c[2]}")

# Enumerate feasible (alpha, beta) with alpha >= 1, beta >= 1
values = set()
for alpha in range(1, 1001):
    feasible = True
    max_beta = 1000  # upper bound
    for v3, v5, rhs in non_dom:
        remaining = rhs - v3 * alpha
        if remaining < 0:
            # Even beta=0 doesn't work, but beta >= 1
            # Actually if v5 > 0: need v5*beta <= remaining < 0, impossible
            # If v5 == 0: remaining < 0 means v3*alpha > rhs, constraint violated
            feasible = False
            break
        if v5 > 0:
            max_b = remaining // v5
            max_beta = min(max_beta, max_b)
        # if v5 == 0: no constraint on beta from this

    if not feasible or max_beta < 1:
        # Can't have any valid beta for this alpha
        if alpha > 1 and not feasible:
            break  # alpha too large
        continue

    for beta in range(1, max_beta + 1):
        val = 4 * alpha + 2 * beta
        values.add(val)

print(f"\nNumber of distinct values of f(2024) = 4*g(3) + 2*g(5): {len(values)}")
print(f"Min: {min(values)}, Max: {max(values)}")

# Check if all even values in range are present
sorted_vals = sorted(values)
expected = set(range(min(values), max(values) + 1, 2))
missing = expected - values
extra = values - expected
print(f"Missing even values: {sorted(missing)[:20]}")
print(f"Number missing: {len(missing)}")
