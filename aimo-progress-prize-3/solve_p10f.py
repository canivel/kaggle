from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# Let me verify the (k1,k2) approach more carefully.
# Claim: f(n) = min over triples (d1,d2,d3) with d1<d2<d3, d1+d2+d3=n, of lcm(d1,d2,d3).
# And the minimum is always achieved when d1|d3, d2|d3 (so lcm = d3 = m).

# Let me check this claim with a brute-force for small cases.

def f_norwegian(n):
    best = float('inf')
    best_triple = None
    for d1 in range(1, n//3 + 1):
        for d2 in range(d1+1, (n-d1)//2 + 1):
            d3 = n - d1 - d2
            if d3 <= d2:
                continue
            l = lcm(lcm(d1, d2), d3)
            if l < best:
                best = l
                best_triple = (d1, d2, d3)
    return best, best_triple

# Verify that optimal always has d1|d3, d2|d3
print("Checking whether optimal triple always has d1|d3 and d2|d3:")
for n in range(6, 200):
    fn, (d1, d2, d3) = f_norwegian(n)
    if d3 % d1 != 0 or d3 % d2 != 0:
        # Check if lcm = d3
        l = lcm(lcm(d1, d2), d3)
        if l != d3:
            print(f"  n={n}: ({d1},{d2},{d3}), lcm={l}, d3={d3}. lcm != d3!")
        else:
            print(f"  n={n}: ({d1},{d2},{d3}), but lcm=d3={d3} still (different reason)")
print("Check complete")
print()

# Hmm, but actually the divisors d1, d2, d3 must all divide m (the n-Norwegian number).
# m = lcm(d1, d2, d3). If lcm > d3, then m > d3 and the triple (d1, d2, d3) all divide m.
# But f(n) = m = lcm, not d3. So we want to minimize lcm.
# In many cases lcm = d3 (when d1|d3 and d2|d3), but not always.
# When lcm > d3, it could still be smaller than other triples' lcm.

# Wait, I already handle this: f_norwegian computes lcm correctly.
# The (k1,k2) approach only covers cases where lcm = d3. Are there cases where lcm != d3 but lcm is still optimal?

# Let me check for cases where the optimal has lcm != d3:
print("Cases where optimal lcm != d3:")
count = 0
for n in range(6, 500):
    fn, (d1, d2, d3) = f_norwegian(n)
    l = lcm(lcm(d1, d2), d3)
    if l != d3:
        count += 1
        if count <= 20:
            print(f"  n={n}: ({d1},{d2},{d3}), lcm={l}")
print(f"Total cases with lcm != d3 for n in [6,500): {count}")
print()

# Now verify: for those cases, is the lcm-based approach still captured by (k1,k2)?
# Actually, the (k1,k2) approach is: d3 = m (with d1=m/k1, d2=m/k2, d3=m).
# This means lcm = m = d3. So the (k1,k2) approach misses cases where lcm > d3.
# BUT: we're minimizing lcm. If there exists a triple where lcm = d3 < lcm of any triple where lcm > d3,
# then (k1,k2) approach is fine. The question is: can a triple with lcm > d3 have lcm smaller than
# any triple with lcm = d3?

# From the check above, if there are cases with lcm != d3 in the optimal, that would mean yes.
# Let me check explicitly.

# Actually, let me reconsider. Even in the lcm != d3 case:
# d3 < lcm = f(n). But is there another triple (d1', d2', d3') with d1'|d3', d2'|d3', d3'=lcm,
# and d1'+d2'+d3' = n? Not necessarily.
# The lcm approach may genuinely give a smaller m than any d3=m approach.

# Let me check: for each n where lcm != d3 in optimal, compare with best d3=m approach.
print("Comparing lcm!=d3 optimal vs best d3=m approach:")
for n in range(6, 300):
    fn, (d1, d2, d3) = f_norwegian(n)
    actual_lcm = lcm(lcm(d1, d2), d3)

    # Best d3=m approach: d3 = m, d1|m, d2|m
    best_m = float('inf')
    for dd1 in range(1, n//3 + 1):
        for dd2 in range(dd1+1, (n-dd1)//2 + 1):
            dd3 = n - dd1 - dd2
            if dd3 <= dd2:
                continue
            if dd3 % dd1 == 0 and dd3 % dd2 == 0:
                if dd3 < best_m:
                    best_m = dd3

    if actual_lcm < best_m:
        print(f"  n={n}: lcm approach ({d1},{d2},{d3}) gives {actual_lcm} < d3=m best {best_m}")

print("Comparison complete")
