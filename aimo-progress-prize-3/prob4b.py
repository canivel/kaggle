"""
Problem 4 revisited.

f(m) + f(n) = f((m+1)(n+1) - 1) for all m,n >= 1.
Let g(k) = f(k-1), so g: Z_>=2 -> Z_>=1.
g(a) + g(b) = g(ab) for all a,b >= 2.

Wait -- actually the functional equation is for all m,n >= 1.
With m=n=1: 2f(1) = f(3). So f(3) = 2f(1).
With m=1, n arbitrary: f(1) + f(n) = f(2n+1).
With m=2, n=1: f(2) + f(1) = f(5).
With m=n=2: 2f(2) = f(8).
With m=2, n=3: f(2)+f(3) = f(11).

Let's be more careful with the substitution g(k) = f(k-1).
g: {2,3,4,...} -> Z_>=1.
g(m+1) + g(n+1) = g((m+1)(n+1)) for m,n >= 1.
Setting a = m+1, b = n+1 where a,b >= 2:
g(a) + g(b) = g(ab) for all a,b >= 2.

So g is completely multiplicative -> additive: g(ab) = g(a) + g(b) for a,b >= 2.

For primes p: g(p) can be any positive integer.
For prime power: g(p^k) = k * g(p).
For general n: g(n) = sum_p v_p(n) * g(p).

Constraints: f(n) <= 1000 for n <= 1000.
f(n) = g(n+1), so g(k) <= 1000 for 2 <= k <= 1001.

For each k in {2,...,1001}, we need:
sum_p v_p(k) * g(p) <= 1000.

We want to count possible values of f(2024) = g(2025).
2025 = 3^4 * 5^2.
g(2025) = 4*g(3) + 2*g(5).

The constraints involving g(3) and g(5) come from ALL k in {2,...,1001}
that are divisible by 3 or 5 (or both).

For any k in {2,...,1001}: sum_p v_p(k) * g(p) <= 1000.
If k = 3^a * 5^b * m where gcd(m, 15) = 1:
a*g(3) + b*g(5) + sum_{p | m} v_p(m)*g(p) <= 1000.

The "other" terms sum_{p | m} v_p(m)*g(p) are >= Omega(m) (where Omega counts
prime factors with multiplicity), since each g(p) >= 1.

So: a*g(3) + b*g(5) <= 1000 - Omega(m).

For the tightest constraint on a*g(3) + b*g(5), we want to find, for each (a,b),
the maximum Omega(m) such that 3^a * 5^b * m <= 1001 with gcd(m,15)=1.

But actually the "other" terms can be larger than Omega(m) since g(p) >= 1.
Wait, but we're looking for which (g(3), g(5)) are feasible. A pair is feasible
if there EXIST valid values for all other g(p) >= 1.

For a pair (g3, g5) to be feasible:
For every k in {2,...,1001}, we need sum_p v_p(k) * g(p) <= 1000.
The other g(p) (p != 3,5) can be chosen to be 1 (minimizing the constraint burden).

So (g3, g5) is feasible iff for every k in {2,...,1001}:
v_3(k)*g3 + v_5(k)*g5 + sum_{p|k, p!=3,5} v_p(k) * 1 <= 1000.
i.e., v_3(k)*g3 + v_5(k)*g5 + Omega_other(k) <= 1000.
i.e., v_3(k)*g3 + v_5(k)*g5 <= 1000 - Omega_other(k).

where Omega_other(k) = sum of v_p(k) for primes p != 3,5.

Wait, but the g(p) for p != 3,5 are SHARED across all k.
If we set g(p) = 1 for all p != 3,5, then for k = 2^j * 3^a * 5^b * ...:
g(k) = j*1 + a*g3 + b*g5 + (other primes contribute 1 each).

Actually, setting ALL g(p) = 1 for p != 3,5 might not work because
that might make g(k) > 1000 for some k even with small g3, g5.

For example, k = 2^9 = 512 <= 1001: g(512) = 9*g(2) = 9*1 = 9 <= 1000. Fine.
k = 2^{10} = 1024 > 1001. Not constrained.

Actually with g(p) = 1 for all p != 3,5, the constraint g(k) <= 1000 for k <= 1001 becomes:
Omega_other(k) + v_3(k)*g3 + v_5(k)*g5 <= 1000.

This is what I had. Now, for the tightest constraint for a given (a,b) = (v_3, v_5),
I need the maximum Omega_other over all k <= 1001 with v_3(k)=a, v_5(k)=b.

But actually, I also need to check k where v_3(k) >= a and v_5(k) >= b... no,
the constraint is specifically for each k. So for each k, we get one constraint
based on v_3(k), v_5(k), and Omega_other(k).

But wait, CAN we always set g(p) = 1 for p != 3, 5? Yes, because we're looking
for which (g3, g5) can possibly occur. If we can find ANY assignment of other g(p)
that works, (g3, g5) is feasible. Setting other g(p) = 1 minimizes g(k) for each k,
giving the most room for g3, g5.

Hmm wait, that's not quite right either. Setting g(p) = 1 for ALL other primes
simultaneously might overconstrain in some unexpected way. But actually it doesn't:
each g(k) = sum v_p(k)*g(p) is minimized when all g(p) are minimized (set to 1).
So setting g(p)=1 for p != 3,5 gives the least restrictive constraints on g3, g5.

Therefore (g3, g5) is feasible iff for ALL k in {2,...,1001}:
v_3(k)*g3 + v_5(k)*g5 <= 1000 - Omega_other(k)

where Omega_other(k) = sum_{p|k, p!=3,5} v_p(k).

Let me enumerate all binding constraints.
"""

from sympy import factorint

# For each k in {2,...,1001}, compute v_3(k), v_5(k), Omega_other(k)
constraints = {}  # (a, b) -> minimum bound (i.e., 1000 - Omega_other)

for k in range(2, 1002):
    factors = factorint(k)
    a = factors.get(3, 0)
    b = factors.get(5, 0)
    omega_other = sum(e for p, e in factors.items() if p != 3 and p != 5)
    bound = 1000 - omega_other

    # Only constraints involving g3 or g5 matter
    if a == 0 and b == 0:
        continue  # This constrains only other primes

    key = (a, b)
    if key not in constraints or bound < constraints[key][0]:
        constraints[key] = (bound, k, omega_other)

print("Binding constraints (tightest for each (a,b)):")
for (a, b), (bound, k, omega) in sorted(constraints.items()):
    print(f"  v3={a}, v5={b}: {a}*g3 + {b}*g5 <= {bound}  (from k={k}, Omega_other={omega})")

# Now enumerate feasible (g3, g5) and collect values of 4*g3 + 2*g5
# g3 >= 1, g5 >= 1

# First find upper bounds
max_g3 = 1000  # from v3=1,v5=0 constraint
max_g5 = 1000
for (a, b), (bound, k, omega) in constraints.items():
    if b == 0 and a > 0:
        max_g3 = min(max_g3, bound // a)
    if a == 0 and b > 0:
        max_g5 = min(max_g5, bound // b)

print(f"\nMax g3 = {max_g3}, Max g5 = {max_g5}")

values = set()
constraint_list = [(a, b, bound) for (a, b), (bound, k, omega) in constraints.items()]

for g3 in range(1, max_g3 + 1):
    for g5 in range(1, max_g5 + 1):
        feasible = True
        for a, b, bound in constraint_list:
            if a*g3 + b*g5 > bound:
                feasible = False
                break
        if feasible:
            values.add(4*g3 + 2*g5)

print(f"\nNumber of distinct values of f(2024) = 4*g3 + 2*g5: {len(values)}")
if values:
    sv = sorted(values)
    print(f"Range: {sv[0]} to {sv[-1]}")
    # Check if consecutive
    expected = set(range(sv[0], sv[-1]+1, 2))  # even spacing?
    print(f"All even? {all(v % 2 == 0 for v in values)}")

    # Find gaps
    gaps = []
    for i in range(len(sv)-1):
        if sv[i+1] - sv[i] > 2:
            gaps.append((sv[i], sv[i+1]))
    if gaps:
        print(f"Gaps: {gaps[:10]}")
    else:
        print("No gaps (with step 2)")
