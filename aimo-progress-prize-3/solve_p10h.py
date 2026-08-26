from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# c = 0: n = M = 3^(2025!).
# n is divisible by 3^K for arbitrarily large K. No other prime divides n.
# For (k1,k2) with k2=2: need S = 3k1+2 to divide n (up to factors of k1 and 2).
# Actually need: S | n*k1 and S | n*k2.
# S = 3k1+2.
# S | n*k1: (3k1+2) | 3^K * k1. Since gcd(3k1+2, 3) might be > 1.
# 3 | (3k1+2) iff 3|2 which is false. So gcd(3k1+2, 3) = 1.
# Therefore gcd(3k1+2, 3^K) = 1. So (3k1+2) | k1.
# But 3k1+2 > k1 for k1 >= 1. So (3k1+2) never divides k1. Impossible!
# So NO (k1, 2) pair works for c=0. That's because n = 3^K has no prime factors other than 3.

# For (k1,k2) with k2=3: need S = 4k1+3 to appropriately divide n.
# S | n*k1: (4k1+3) | 3^K * k1. gcd(4k1+3, 3) = gcd(k1, 3)... let me compute.
# 4k1+3 mod 3 = k1 mod 3. So if 3|k1, then 3 | (4k1+3), and (4k1+3)/3 | 3^(K-1)*k1/...
# This gets messy. Let me just enumerate.

# For k2=3, k1 must have 4k1+3 divisible only by powers of 3 (and k1, k2).
# Actually the condition is: all prime factors of S must be either 3 or divide k1*k2=3k1.
# S = 4k1+3. If k1 = 3^a * b for some b, then 3k1 = 3^(a+1)*b.
# We need every prime factor of 4k1+3 to be 3 or a factor of k1.
#
# For k1 = 3: S = 15 = 3*5. Need 5 | something. 5 does not divide k1=3 or k2=3.
#   From the condition: S/gcd(S,k1) | n and S/gcd(S,k2) | n.
#   S/gcd(S,k1) = 15/gcd(15,3) = 15/3 = 5. Need 5 | n. But n = 3^K. 5 does not divide n. Fail.
#
# k1=6: S = 27 = 3^3. gcd(S,k1) = gcd(27,6)=3. S/gcd(S,k1) = 9. Need 9|n. Yes!
#   gcd(S,k2) = gcd(27,3) = 3. S/gcd(S,k2) = 9. Need 9|n. Yes!
#   So (6,3) works if 9|n. n = 3^K with K >= 2: yes. Ratio = 18/27 = 2/3.

# Can we find something better (ratio < 2/3) for n = 3^K?
# Need (k1,k2) where ALL prime factors of the "requirement" are just 3.
# From the (k1,3) table: (6,3) needs 9|n. Ratio 2/3.
# Others need non-3 primes. What about (k1,k2) with both k1,k2 being powers of 3 or multiples?

# General: for (k1,k2), S = k1*k2+k1+k2 = (k1+1)(k2+1)-1.
# Need all prime factors of S/gcd(S,k1) and S/gcd(S,k2) to be 3.
# i.e., S/gcd(S,k1) is a power of 3, and S/gcd(S,k2) is a power of 3.

# Let me search more broadly.
print("Searching for (k1,k2) valid for n = 3^K:")
results = []
for k1 in range(3, 1000):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        # Need S | (n*k1) and S | (n*k2)
        # n = 3^K (huge). So n*k1 = 3^K * k1. For S | (3^K * k1):
        # Every prime factor of S must be 3 or divide k1.
        # Similarly for k2.
        ok1 = True
        temp = S
        # Remove factors that are 3 or divide k1
        while temp % 3 == 0:
            temp //= 3
        t = temp
        for p in range(2, t+1):
            if t == 1:
                break
            while t % p == 0:
                if k1 % p != 0:
                    ok1 = False
                    break
                t //= p
            if not ok1:
                break
        if t > 1:
            ok1 = False

        if not ok1:
            continue

        ok2 = True
        temp = S
        while temp % 3 == 0:
            temp //= 3
        t = temp
        for p in range(2, t+1):
            if t == 1:
                break
            while t % p == 0:
                if k2 % p != 0:
                    ok2 = False
                    break
                t //= p
            if not ok2:
                break
        if t > 1:
            ok2 = False

        if ok1 and ok2:
            results.append((ratio, k1, k2, S))

results.sort(key=lambda x: x[0])
print(f"Found {len(results)} valid (k1,k2) pairs")
for r, k1, k2, S in results[:20]:
    print(f"  (k1={k1}, k2={k2}): S={S}, ratio={r}={float(r):.6f}")

# Now for c = 4M: n = 5M.
# n = 5*3^K. Only primes dividing n: 3 and 5.
# Need all prime factors of S not covered by k1 or k2 to be 3 or 5.
# Wait, the exact condition: S | (n*k1) where n = 5*3^K.
# So every prime p | S: either p=3, or p=5, or p | k1.
# Similarly: p | S: either p=3, or p=5, or p | k2.
# The binding condition: for each prime p | S with p != 3 and p != 5: p | k1 AND p | k2.
# (Because we need S | n*k1 AND S | n*k2, and n only has 3,5 as prime factors.)

print("\nSearching for (k1,k2) valid for n = 5*3^K:")
results_4M = []
for k1 in range(3, 1000):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        # For each prime p | S, p not in {3,5}: need p | k1 and p | k2.
        # But also: for p = 5: if 5 | S, then 5 must divide n*k1 = 5*3^K*k1. Always true.
        # And 5 | n*k2 = 5*3^K*k2. Always true.
        # For p = 3: 3 | n always. So 3 is fine.
        # So we need: for p | S, p not in {3,5}: p | k1 AND p | k2.
        # But gcd(k1,k2) divides both, so p | gcd(k1,k2).

        ok = True
        temp = S
        while temp % 3 == 0:
            temp //= 3
        while temp % 5 == 0:
            temp //= 5
        # Now temp should divide gcd(k1, k2)^? Actually we need each prime factor of temp to divide BOTH k1 and k2.
        t = temp
        for p in range(2, t+1):
            if t == 1:
                break
            e = 0
            while t % p == 0:
                e += 1
                t //= p
            if e > 0:
                # Need p^e | k1 sufficient powers in S condition... actually it's more nuanced.
                # Let v = v_p(S). Need v_p(k1) + v_p(n) >= v and v_p(k2) + v_p(n) >= v.
                # v_p(n) = 0 for p != 3,5. So need v_p(k1) >= v and v_p(k2) >= v.
                # v is actually v_p(S), not e of temp. Let me recompute.
                pass

        # Let me redo this properly.
        ok = True
        temp2 = S
        p = 2
        while p * p <= temp2:
            if temp2 % p == 0:
                vS = 0
                while temp2 % p == 0:
                    vS += 1
                    temp2 //= p
                if p not in (3, 5):
                    # Need p^vS | k1 and p^vS | k2. No wait:
                    # Need v_p(k1) >= vS - v_p(n) = vS (since v_p(n)=0 for p!=3,5).
                    # Actually: S | n*k1 means v_p(S) <= v_p(n) + v_p(k1) for each p.
                    # For p != 3,5: v_p(n) = 0. So need v_p(k1) >= v_p(S).
                    # Similarly need v_p(k2) >= v_p(S).
                    vk1 = 0
                    tt = k1
                    while tt % p == 0:
                        vk1 += 1
                        tt //= p
                    vk2 = 0
                    tt = k2
                    while tt % p == 0:
                        vk2 += 1
                        tt //= p
                    if vk1 < vS or vk2 < vS:
                        ok = False
                        break
            p += 1
        if temp2 > 1 and ok:
            p = temp2
            vS = 1
            if p not in (3, 5):
                vk1 = 0
                tt = k1
                while tt % p == 0:
                    vk1 += 1
                    tt //= p
                vk2 = 0
                tt = k2
                while tt % p == 0:
                    vk2 += 1
                    tt //= p
                if vk1 < vS or vk2 < vS:
                    ok = False

        if ok:
            results_4M.append((ratio, k1, k2, S))

results_4M.sort(key=lambda x: x[0])
print(f"Found {len(results_4M)} valid (k1,k2) pairs")
for r, k1, k2, S in results_4M[:20]:
    print(f"  (k1={k1}, k2={k2}): S={S}, ratio={r}={float(r):.6f}")
