"""
Problem 4: Extra verification.

Need to verify that setting g(p)=1 for all p != 3,5 doesn't cause any constraint
violation for k in {2,...,1001}.

For each k with no factors of 3 or 5:
g(k) = sum_{p|k, p!=3,5} v_p(k) * g(p) = sum_{p|k} v_p(k) * 1 = Omega(k).
Need Omega(k) <= 1000.
Max Omega(k) for k <= 1001: k=512=2^9, Omega=9. 9 <= 1000 ✓.
k=768 = 2^8*3, Omega_other = 8. g(768)=8+g(3). Need 8+g(3) <= 1000, so g(3) <= 992.
k=960 = 2^6*3*5, Omega_other = 6. g(960)=6+g(3)+g(5). Need 6+g(3)+g(5) <= 1000.

So the constraints are exactly as computed before.

Now, the question is: could choosing g(p) > 1 for some p != 3,5 HELP?
No! Increasing any g(p) can only increase g(k) for k that are multiples of p,
making constraints harder to satisfy, not easier.

So g(p)=1 for p!=3,5 is the OPTIMAL choice to maximize feasibility of (g3, g5).

The answer is 580.
"""

# Final check: enumerate all constraints and verify the count
from sympy import factorint

constraints = []
for k in range(2, 1002):
    fac = factorint(k)
    a = fac.get(3, 0)
    b = fac.get(5, 0)
    omega_other = sum(e for p, e in fac.items() if p not in (3, 5))

    if a == 0 and b == 0:
        # Only constrains other primes, not g3 or g5
        continue

    # Constraint: a*g3 + b*g5 <= 1000 - omega_other
    bound = 1000 - omega_other
    constraints.append((a, b, bound))

# Remove dominated constraints
# (a1,b1,B1) dominates (a2,b2,B2) if a1<=a2, b1<=b2, B1<=B2 and at least one strict
# But we just use all of them since they're linear

# Find all feasible (g3, g5)
# g3 in [1, 166], g5 in [1, 250]

values = set()
for g3 in range(1, 993):  # max from g3 <= 992
    feasible = True
    max_g5 = 1000  # upper bound

    for a, b, bound in constraints:
        remaining = bound - a * g3
        if remaining < 0:
            feasible = False
            break
        if b > 0:
            max_g5_local = remaining // b
            max_g5 = min(max_g5, max_g5_local)

    if not feasible or max_g5 < 1:
        continue

    for g5 in range(1, max_g5 + 1):
        values.add(4*g3 + 2*g5)

print(f"Number of distinct values of f(2024): {len(values)}")
sv = sorted(values)
print(f"Min: {sv[0]}, Max: {sv[-1]}")

# Check for gaps
expected_count = (sv[-1] - sv[0]) // 2 + 1
print(f"Expected if no gaps (step 2): {expected_count}")
print(f"Match: {len(values) == expected_count}")

# Also verify boundary cases
print(f"\ng3=1,g5=1: f(2024)={4+2}=6")
print(f"g3=166,g5=250: f(2024)={4*166+2*250}={664+500}")

# Final answer
print(f"\nAnswer: {len(values)}")
