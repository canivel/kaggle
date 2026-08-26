#!/usr/bin/env python3
"""
Problem 4: f: Z>=1 -> Z>=1 such that f(m) + f(n) = f(m + n + mn) for all m,n >= 1.
Constraint: f(n) <= 1000 for all n <= 1000.
How many different values can f(2024) take?

Key observation: m + n + mn = (m+1)(n+1) - 1
So if we let g(k) = f(k-1) for k >= 2, then:
f(m) + f(n) = f((m+1)(n+1) - 1)
g(m+1) + g(n+1) = g((m+1)(n+1))

Let h(x) = g(x) for x >= 2 (where x = m+1, m >= 1, so x >= 2).
Then h(a) + h(b) = h(ab) for all a, b >= 2.

This is a multiplicative-to-additive homomorphism on integers >= 2.

For primes p, q: h(p) + h(q) = h(pq)
More generally, h(p^a * q^b * ...) = a*h(p) + b*h(q) + ...

So h is determined by its values on primes, and h(n) = sum of h(p) * v_p(n)
where v_p(n) is the p-adic valuation of n.

We need h to map to Z>=1 (since f maps to Z>=1, so g maps to Z>=1, so h maps to Z>=1).

Wait, let me re-check. f: Z>=1 -> Z>=1.
g(k) = f(k-1) for k >= 2. So g: Z>=2 -> Z>=1.
h = g on {2, 3, 4, ...}. h: Z>=2 -> Z>=1.
h(a) + h(b) = h(ab) for a, b >= 2.

For any n >= 2, write n = p1^{a1} * ... * pk^{ak}.
h(n) = a1*h(p1) + ... + ak*h(pk).

We need h(n) >= 1 for all n >= 2.
For primes p: h(p) >= 1 (since h(p) is in Z>=1).

For composite n = p1^a1 * ... * pk^ak with sum ai >= 2:
h(n) = sum ai * h(pi) >= sum ai >= 2 >= 1. OK.

So the only constraint is h(p) >= 1 for each prime p, i.e., h(p) is a positive integer.

Now f(n) = g(n+1) = h(n+1).

Constraint: f(n) <= 1000 for all n <= 1000.
So h(n+1) <= 1000 for all 1 <= n <= 1000, i.e., h(k) <= 1000 for all 2 <= k <= 1001.

We want to count the number of possible values of f(2024) = h(2025).
2025 = 45^2 = (9*5)^2 = 3^4 * 5^2.
So h(2025) = 4*h(3) + 2*h(5).

Now we need:
- h(p) >= 1 for all primes p
- h(k) <= 1000 for 2 <= k <= 1001

h(k) = sum over primes p | k of v_p(k) * h(p).

The constraint h(k) <= 1000 for 2 <= k <= 1001 constrains the values of h(p) for
primes p <= 1001.

The most restrictive constraints come from prime powers (which give large coefficients
for a single h(p)):
- For h(2): largest power of 2 <= 1001 is 2^9 = 512. h(512) = 9*h(2) <= 1000 => h(2) <= 111.
- For h(3): largest power of 3 <= 1001 is 3^6 = 729. h(729) = 6*h(3) <= 1000 => h(3) <= 166.
- For h(5): largest power of 5 <= 1001 is 5^4 = 625. h(625) = 4*h(5) <= 1000 => h(5) <= 250.
- For h(7): 7^3 = 343 <= 1001. h(343) = 3*h(7) <= 1000 => h(7) <= 333.

But we also need to check products. For instance:
k = 768 = 2^8 * 3 = 256*3. h(768) = 8*h(2) + h(3) <= 1000.
k = 512 = 2^9. h(512) = 9*h(2) <= 1000.
k = 960 = 2^6 * 3 * 5. h(960) = 6*h(2) + h(3) + h(5) <= 1000.

Actually, for each k in [2, 1001], we get a constraint:
sum_{p | k} v_p(k) * h(p) <= 1000.

We want to count the number of distinct values of 4*h(3) + 2*h(5)
subject to: h(p) >= 1 for all primes p, and
sum_{p|k} v_p(k) * h(p) <= 1000 for all 2 <= k <= 1001.

Since h(2025) = 4*h(3) + 2*h(5), and these only involve h(3) and h(5),
we need to find the range of 4*h(3) + 2*h(5) subject to constraints.

The constraints involving h(3) alone (from powers of 3):
- 3^6 = 729: 6*h(3) <= 1000, so h(3) <= 166

The constraints involving h(5) alone:
- 5^4 = 625: 4*h(5) <= 1000, so h(5) <= 250

Mixed constraints involving h(3) and h(5) (and possibly other primes which are >= 1):
We need k in [2, 1001] whose prime factorization involves both 3 and 5.
k = 3^a * 5^b * (other primes)^...
The constraint becomes: a*h(3) + b*h(5) + (contributions from other primes) <= 1000.
Since other prime contributions are >= their valuations (each h(p) >= 1),
the effective constraint on h(3), h(5) is:
a*h(3) + b*h(5) <= 1000 - (sum of valuations of other primes in k).

But actually, the other primes' h values are free (as long as >= 1), so the constraint
for a specific k is weakest when other h(p) = 1. But wait - the h(p) values for other
primes are shared across all constraints. We want to MAXIMIZE the range of 4h(3)+2h(5),
so we should consider: for given h(3) and h(5), do there EXIST values h(2) >= 1, h(7) >= 1,
h(11) >= 1, ... such that all constraints are satisfied?

The most restrictive constraints on h(3) and h(5) come from k values that have
large powers of 3 and 5 with minimal contribution from other primes.

Let me systematically find all constraints.
"""

from sympy import factorint
from collections import defaultdict

# Find all constraints: for each k in [2, 1001], factorize k
# and get constraint sum v_p(k) * h(p) <= 1000

constraints = []
for k in range(2, 1002):
    f = factorint(k)
    constraints.append(f)

# We want to know: what are the possible values of 4*h(3) + 2*h(5)?
# where h(p) >= 1 for all primes p, and for each k in [2,1001],
# sum_{p|k} v_p(k)*h(p) <= 1000.

# The primes up to 1001:
primes_up_to_1001 = set()
for f in constraints:
    primes_up_to_1001.update(f.keys())
primes_up_to_1001 = sorted(primes_up_to_1001)
print(f"Number of primes up to 1001: {len(primes_up_to_1001)}")

# For any fixed h(3) and h(5), we need: for each k in [2,1001],
# the constraint sum v_p(k)*h(p) <= 1000 must be satisfiable with h(p) >= 1 for other primes.
#
# For a given k, the constraint is:
# v_3(k)*h(3) + v_5(k)*h(5) + sum_{p != 3,5, p|k} v_p(k)*h(p) <= 1000
# Since we want this to be satisfiable and h(p) >= 1 for other primes:
# sum_{p != 3,5, p|k} v_p(k)*h(p) >= sum_{p != 3,5, p|k} v_p(k)
# = Omega(k) - v_3(k) - v_5(k) where Omega(k) is total number of prime factors with multiplicity
#
# So the necessary condition is:
# v_3(k)*h(3) + v_5(k)*h(5) + sum_{p != 3,5, p|k} v_p(k) <= 1000
# i.e., v_3(k)*h(3) + v_5(k)*h(5) <= 1000 - sum_{p != 3,5, p|k} v_p(k)
#
# But also, other constraints might force h(p) > 1 for some other prime p.
# However, since we only care about existence, and h(p) for p != 3, 5 only appears
# in constraints involving k that are divisible by p, we can set h(p) = 1 for p != 3, 5
# when checking if h(3), h(5) are feasible. This minimizes the "other prime" contributions.
#
# BUT WAIT: we also need h(p) >= 1 for ALL primes, including those > 1001. For those,
# there are no constraints from [2,1001], so h(p) = 1 works fine. And for primes p <= 1001,
# setting h(p) = 1 is the minimum, so it's the least restrictive choice.
#
# Actually, can we always set h(p) = 1 for p != 3, 5? Let me check: the constraints involving
# p (for p != 3, 5) are constraints on k divisible by p. The constraint for k only involves
# primes dividing k. So setting h(p) = 1 for all p != 3, 5 satisfies: for each k,
# sum v_p(k)*h(p) = v_3(k)*h(3) + v_5(k)*h(5) + sum_{other p|k} v_p(k)*1
# This needs to be <= 1000.
#
# So the constraints on (h(3), h(5)) when h(p)=1 for p != 3, 5 are:
# For each k in [2, 1001]:
# v_3(k)*h(3) + v_5(k)*h(5) + (Omega(k) - v_3(k) - v_5(k)) <= 1000
# i.e., v_3(k)*(h(3)-1) + v_5(k)*(h(5)-1) + Omega(k) <= 1000
# i.e., v_3(k)*(h(3)-1) + v_5(k)*(h(5)-1) <= 1000 - Omega(k)

# Let a = h(3) - 1 >= 0, b = h(5) - 1 >= 0
# For each k: v_3(k)*a + v_5(k)*b <= 1000 - Omega(k)
# where Omega(k) = sum of all v_p(k)

# The binding constraints are those where v_3(k) and/or v_5(k) are large relative to 1000-Omega(k).

# Let's enumerate the binding constraints:
binding = []
for k in range(2, 1002):
    f = factorint(k)
    v3 = f.get(3, 0)
    v5 = f.get(5, 0)
    omega = sum(f.values())
    rhs = 1000 - omega
    if v3 > 0 or v5 > 0:
        binding.append((v3, v5, rhs, k))

# Sort by most restrictive for h(3): highest v3/rhs ratio
# Find the constraint that most restricts h(3) alone (v5=0)
print("\nConstraints involving only h(3) (v5=0):")
only3 = [(v3, rhs, k) for v3, v5, rhs, k in binding if v5 == 0 and v3 > 0]
only3.sort(key=lambda x: x[1]/x[0])
for v3, rhs, k in only3[:5]:
    print(f"  k={k}: {v3}*a <= {rhs}, so a <= {rhs//v3} (h(3) <= {rhs//v3 + 1})")

print("\nConstraints involving only h(5) (v3=0):")
only5 = [(v5, rhs, k) for v3, v5, rhs, k in binding if v3 == 0 and v5 > 0]
only5.sort(key=lambda x: x[1]/x[0])
for v5, rhs, k in only5[:5]:
    print(f"  k={k}: {v5}*b <= {rhs}, so b <= {rhs//v5} (h(5) <= {rhs//v5 + 1})")

print("\nConstraints involving both h(3) and h(5):")
both = [(v3, v5, rhs, k) for v3, v5, rhs, k in binding if v3 > 0 and v5 > 0]
both.sort(key=lambda x: x[2])
for v3, v5, rhs, k in both[:10]:
    print(f"  k={k}: {v3}*a + {v5}*b <= {rhs}")

# Now let's find all feasible (a, b) pairs and count distinct values of 4*(a+1) + 2*(b+1) = 4a + 2b + 6
# i.e., count distinct values of 4a + 2b + 6 = 2*(2a + b) + 6

# First find max a (when b=0): from only3 constraints
max_a = min(rhs // v3 for v3, rhs, k in only3)
print(f"\nMax a (h(3)-1) when b=0: {max_a}")

# Max b when a=0:
max_b = min(rhs // v5 for v5, rhs, k in only5)
print(f"Max b (h(5)-1) when a=0: {max_b}")

# Now collect ALL constraints (including mixed)
all_constraints = []
for v3, v5, rhs, k in binding:
    all_constraints.append((v3, v5, rhs))
# Also add constraints from k with v3=v5=0: these give Omega(k) <= 1000, which is always true.

# For each pair (a, b) with a >= 0, b >= 0, check all constraints
# Then compute 4*(a+1) + 2*(b+1) = 4a + 2b + 6

# This could be slow if max_a and max_b are large. Let's check.
print(f"\nmax_a = {max_a}, max_b = {max_b}")

# Let's compute the target = 4a + 2b + 6
# We want to count distinct target values.
# target = 4a + 2b + 6
# Since 4a + 2b = 2(2a + b), target is always even. target >= 6.

# Let's enumerate. Instead of checking all (a,b) pairs, let's be smarter.
# For each value of a from 0 to max_a, find max b such that all constraints are satisfied.

values = set()
for a in range(max_a + 1):
    # Find max b given a
    max_b_given_a = 10**9
    for v3, v5, rhs_val in all_constraints:
        remaining = rhs_val - v3 * a
        if remaining < 0:
            max_b_given_a = -1
            break
        if v5 > 0:
            max_b_given_a = min(max_b_given_a, remaining // v5)

    if max_b_given_a < 0:
        break

    for b in range(max_b_given_a + 1):
        target = 4 * (a + 1) + 2 * (b + 1)
        values.add(target)

print(f"\nNumber of distinct values of f(2024) = 4*h(3) + 2*h(5): {len(values)}")
print(f"Min value: {min(values)}")
print(f"Max value: {max(values)}")

# Let's also verify: are there gaps?
sorted_vals = sorted(values)
gaps = []
for i in range(1, len(sorted_vals)):
    if sorted_vals[i] - sorted_vals[i-1] > 1:
        gaps.append((sorted_vals[i-1], sorted_vals[i]))
print(f"Gaps: {gaps[:20]}")

# Since target = 4a + 2b + 6, the possible values are 6, 8, 10, 12, ...
# (always even, step 2). But 4a + 2b can give: for a=0: 2b (even), for a=1: 4+2b (even), etc.
# Actually 4a + 2b is always even, so target is always even.
# But 4a + 2b can be: 0, 2, 4, 6, 8, ... For a=0: 0,2,4,6,... For a=1: 4,6,8,...
# So 4a+2b ranges over all even numbers from 0 to max(4a+2b).
# Wait, can we get 4a+2b = 2? a=0, b=1: yes. 4a+2b=0? a=b=0: yes.
# So target = 2k+6 where k=2a+b >= 0. But not all k are achievable because of constraints.

# Actually, let me reconsider. For a given feasibility region of (a,b),
# what values of 4a + 2b are achievable?
# 4a + 2b = 2(2a + b). Let s = 2a + b.
# For s = 0: (0,0). For s=1: (0,1). For s=2: (0,2) or (1,0). Etc.
# All integer s >= 0 can be achieved if a and b ranges allow.

# But the constraint region is a polygon. Let me think about it differently.
# Values of target = 4a+2b+6: since target is always even, step is at least 2.
# Can we get all even values from 6 to some max?

print(f"\nAll values are even: {all(v % 2 == 0 for v in values)}")
even_vals = sorted(v // 2 for v in values)
print(f"Range of target/2: {even_vals[0]} to {even_vals[-1]}")
gaps_half = []
for i in range(1, len(even_vals)):
    if even_vals[i] - even_vals[i-1] > 1:
        gaps_half.append((even_vals[i-1], even_vals[i]))
print(f"Gaps in target/2: {gaps_half[:20]}")
print(f"Number of distinct values: {len(values)}")
