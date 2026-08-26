"""
Problem 4: f: Z_>=1 -> Z_>=1 such that f(m) + f(n) = f(m + n + mn) for all m,n >= 1.
f(n) <= 1000 for all n <= 1000.
How many different values can f(2024) take?

Key observation: m + n + mn = (m+1)(n+1) - 1.
So f(m) + f(n) = f((m+1)(n+1) - 1).

Let g(k) = f(k-1) for k >= 2 (i.e., f(n) = g(n+1)).
Then g(m+1) + g(n+1) = g((m+1)(n+1)).

Let h = g restricted to integers >= 2.
h(a) + h(b) = h(ab) for all a, b >= 2.

This means h: (Z_>=2, *) -> (Z_>=1, +) is additive wrt multiplication.

So h(ab) = h(a) + h(b). This extends: h(a^k) = k*h(a).

For any n >= 2, write n = p1^e1 * p2^e2 * ... * pk^ek.
Then h(n) = e1*h(p1) + e2*h(p2) + ... + ek*h(pk).

So h is completely determined by its values on primes.
h(p) can be any positive integer for each prime p, with the constraint
that f(n) = h(n+1) >= 1 and f(n) <= 1000 for n <= 1000.

Wait, h maps to positive integers? Let's check.
f: Z_>=1 -> Z_>=1, so f(n) >= 1 for all n.
h(n+1) = f(n) >= 1 for n >= 1, so h(k) >= 1 for k >= 2.
For prime p: h(p) >= 1.
For composite n = ab (a,b >= 2): h(n) = h(a) + h(b) >= 2.

The constraint f(n) <= 1000 for n <= 1000 means h(n+1) <= 1000 for n <= 1000,
i.e., h(k) <= 1000 for 2 <= k <= 1001.

For a prime p <= 1001: h(p) <= 1000 and h(p) >= 1.
For prime p > 1001: h(p) can be anything >= 1.

We want the number of possible values of f(2024) = h(2025).
2025 = 45^2 = (9*5)^2 = 3^4 * 5^2.
So h(2025) = 4*h(3) + 2*h(5).

Constraints:
h(3) >= 1, h(5) >= 1.
h(k) <= 1000 for 2 <= k <= 1001.
h(k) >= 1 for all k >= 2 (and h(k) >= 2 for composite k).

The binding constraints on h(3) and h(5) come from numbers k <= 1001
that are of the form 3^a * 5^b.

h(3^a * 5^b) = a*h(3) + b*h(5) <= 1000.
Also need >= 1 (or >= 2 if composite, but since h(p) >= 1 and a+b >= 1,
this is automatic for a+b >= 1).

Numbers of form 3^a * 5^b with 3^a * 5^b <= 1001:
"""

# Find all 3^a * 5^b <= 1001
constraints = []
a = 0
while 3**a <= 1001:
    b = 0
    while 3**a * 5**b <= 1001:
        if a + b > 0:  # skip 1
            val = 3**a * 5**b
            constraints.append((a, b, val))
        b += 1
    a += 1

print("Numbers 3^a * 5^b <= 1001:")
for a, b, v in sorted(constraints, key=lambda x: x[2]):
    print(f"  3^{a} * 5^{b} = {v}, constraint: {a}*h3 + {b}*h5 <= 1000")

# But we also need to check OTHER numbers <= 1001 that involve primes 3 and 5.
# E.g., h(6) = h(2*3) = h(2) + h(3). The constraint h(6) <= 1000 gives
# h(2) + h(3) <= 1000. But h(2) can be chosen freely (>= 1), so this
# doesn't constrain the relationship between h(3) and h(5) alone.
# Since we want to count values of 4*h(3) + 2*h(5), we need constraints
# that involve only h(3) and h(5).

# The binding constraints are from numbers k <= 1001 of the form 3^a * 5^b
# (no other prime factors), because for numbers with other prime factors,
# h(k) = a*h(3) + b*h(5) + (other terms) and the other terms are free (>= 1),
# making the constraint less restrictive.

# Wait, but numbers with other prime factors also constrain.
# E.g., h(750) = h(2 * 3 * 5^3) = h(2) + h(3) + 3*h(5) <= 1000.
# Since h(2) >= 1: h(3) + 3*h(5) <= 999.
# But h(3) + 3*h(5) is also constrained by h(3 * 5^3) = h(375) = h(3) + 3*h(5) <= 1000.
# Since 375 <= 1001, that gives h(3) + 3*h(5) <= 1000, which is weaker than 999.
# Hmm, so the constraint from 750 is actually tighter (999 vs 1000).

# More generally, for k = 2^c * 3^a * 5^b * ... <= 1001,
# h(k) = c*h(2) + a*h(3) + b*h(5) + ... <= 1000.
# Since all other terms >= 1 each:
# a*h(3) + b*h(5) <= 1000 - c*h(2) - (sum of other terms).
# To get the tightest constraint on a*h(3) + b*h(5), we minimize the other terms.
# Minimum other terms: each prime's h value is 1.
# So a*h(3) + b*h(5) <= 1000 - c - (number of other prime factors counted with multiplicity).

# Actually, for k = p1^e1 * p2^e2 * ... = 3^a * 5^b * (other primes),
# h(k) = a*h(3) + b*h(5) + sum_{p != 3,5} e_p * h(p).
# Each h(p) >= 1 for prime p, so sum_{p != 3,5} e_p * h(p) >= sum_{p != 3,5} e_p = Omega(k) - a - b
# where Omega(k) is the number of prime factors with multiplicity.
# Hence a*h(3) + b*h(5) <= 1000 - (Omega(k) - a - b)... no wait.
# h(k) = a*h(3) + b*h(5) + (other terms) <= 1000.
# other terms >= Omega_other = total Omega minus a minus b.
# So a*h(3) + b*h(5) <= 1000 - Omega_other.

# For the constraint to be tightest, we want Omega_other as small as possible
# given that k <= 1001 and k = 3^a * 5^b * (product of other primes^powers).
# For fixed (a,b), we want k <= 1001 with smallest possible other contribution.
# The smallest other contribution is 0 (no other primes), giving k = 3^a * 5^b.
# If 3^a * 5^b > 1001, we can't use this. Then we'd need Omega_other >= 1.
# But then k >= 2 * 3^a * 5^b > 2*1001, which is > 1001 in most cases.
# So for most (a,b), the binding constraint comes from k = 3^a * 5^b.

# Let me find the actual binding constraint for h(3) and h(5).
# The constraint region is:
# For all (a,b) with 3^a * 5^b <= 1001 and a+b > 0:
#   a*h(3) + b*h(5) <= 1000

# Also h(3) >= 1, h(5) >= 1.

# But also: for numbers k <= 1001 that involve primes 3 and 5 along with
# other primes, we get: a*h3 + b*h5 <= 1000 - (min contribution of other primes).
# With other primes contributing at least 1 each (multiplied by multiplicity),
# the constraint is: a*h3 + b*h5 <= 1000 - Omega_other.

# Since for pure 3^a * 5^b, Omega_other = 0, these give the weakest constraints
# (a*h3 + b*h5 <= 1000). Including other primes gives TIGHTER constraints.

# But we want to maximize the RANGE of 4*h3 + 2*h5.
# So we need to check: are there tighter constraints from non-pure powers?

# For example: k = 2 * 3^6 = 1458 > 1001, so no constraint from 3^6 * 2.
# k = 3^6 = 729 <= 1001: gives 6*h3 <= 1000, h3 <= 166.
# k = 2 * 3^6 = 1458 > 1001: no additional constraint.
# k = 3^6 * 5 = 3645 > 1001: no constraint.

# What about k = 2 * 3^5 * 5 = 2 * 243 * 5 = 2430 > 1001. No.
# k = 3^5 * 5 = 1215 > 1001. No.
# k = 2 * 3^4 * 5 = 2 * 81 * 5 = 810 <= 1001!
# h(810) = h(2) + 4*h(3) + h(5) <= 1000.
# Since h(2) >= 1: 4*h3 + h5 <= 999.
# But from k = 3^4 * 5 = 405: 4*h3 + h5 <= 1000.
# So 810 gives a tighter constraint: 4*h3 + h5 <= 999.

# Similarly, k = 4 * 3^4 * 5 = 1620 > 1001. No.
# k = 2 * 3^3 * 5^2 = 2 * 27 * 25 = 1350 > 1001. No.
# k = 3^3 * 5^2 = 675 <= 1001: 3*h3 + 2*h5 <= 1000.
# k = 2 * 3^3 * 5^2 = 1350 > 1001. No.
# k = 2 * 3^2 * 5^2 = 450 <= 1001: h(2) + 2*h3 + 2*h5 <= 1000, so 2*h3 + 2*h5 <= 999.
# But from k = 3^2 * 5^2 = 225: 2*h3 + 2*h5 <= 1000.
# So 450 gives 2*h3 + 2*h5 <= 999, tighter by 1.

# Hmm, but these differences of 1 might matter. Let me be systematic.

# For each pair (a,b) with a,b >= 0 and a+b > 0:
# Find the smallest k <= 1001 of the form 3^a * 5^b * m where m >= 1.
# Then Omega(m) is the number of prime factors of m (with multiplicity, excluding 3 and 5).
# Constraint: a*h3 + b*h5 <= 1000 - Omega(m).
# To get tightest constraint, we want k <= 1001 with largest Omega(m) for given (a,b).
# Wait no, we want to find ALL constraints, not just the tightest.
# The binding constraint for given (a,b) is: a*h3 + b*h5 <= 1000 - max_omega
# where max_omega = max Omega_other over all k <= 1001 of the form 3^a * 5^b * m.

# For k = 3^a * 5^b * m <= 1001, m is coprime to 15.
# Omega_other = Omega(m).
# m <= 1001 / (3^a * 5^b).
# To maximize Omega(m), take m = 2^j where 2^j <= 1001/(3^a * 5^b).
# j = floor(log2(1001/(3^a * 5^b))).

import math

all_constraints = []

for a in range(20):
    for b in range(20):
        base = 3**a * 5**b
        if base > 1001:
            break
        if a + b == 0:
            continue
        # Max Omega_other: largest j with 2^j * base <= 1001
        max_m = 1001 // base
        if max_m == 0:
            continue
        # Omega_other of m for m <= max_m, m coprime to 15
        # To maximize Omega_other: m = 2^j, j = floor(log2(max_m))
        if max_m >= 2:
            j = int(math.log2(max_m))
            omega_other = j
        elif max_m >= 1:
            omega_other = 0
        else:
            continue

        bound = 1000 - omega_other
        all_constraints.append((a, b, bound, base, omega_other))
        if a <= 6 and b <= 4:
            print(f"(a={a}, b={b}): 3^{a}*5^{b}={base}, max_m={max_m}, omega_other={omega_other}, {a}*h3+{b}*h5 <= {bound}")

print("\n--- Most restrictive constraints on h3 and h5 ---")
# For a given direction (a,b), the tightest constraint is a*h3 + b*h5 <= bound.
# We need the full set of constraints.

# Actually, let me reconsider. We want to count values of 4*h3 + 2*h5
# subject to:
# h3 >= 1, h5 >= 1
# For each constraint: a*h3 + b*h5 <= C_{a,b}

# Let me collect the tightest constraint for each (a,b):
tightest = {}
for a, b, bound, base, omega in all_constraints:
    key = (a, b)
    if key not in tightest or bound < tightest[key]:
        tightest[key] = bound

print("\nTightest constraints:")
for (a,b), bound in sorted(tightest.items()):
    if a <= 8 and b <= 5:
        print(f"  {a}*h3 + {b}*h5 <= {bound}")

# Now enumerate valid (h3, h5) pairs and find all possible values of 4*h3 + 2*h5
values = set()
for h3 in range(1, 1001):
    for h5 in range(1, 1001):
        valid = True
        for (a, b), bound in tightest.items():
            if a*h3 + b*h5 > bound:
                valid = False
                break
        if valid:
            values.add(4*h3 + 2*h5)

print(f"\nNumber of distinct values of f(2024) = 4*h3 + 2*h5: {len(values)}")
if values:
    print(f"Range: {min(values)} to {max(values)}")
