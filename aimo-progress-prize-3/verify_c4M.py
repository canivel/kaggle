from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# For c=4M: n = 5M = 5*3^K, primes: 3 (power K) and 5 (power 1).
# The condition for (k1,k2) to work:
# For each prime p and each i in {1,2}: v_p(S) <= v_p(n) + v_p(k_i)
# where S = k1*k2 + k1 + k2.

# For p=3: v_3(n) = K (huge), always fine.
# For p=5: v_5(n) = 1. Need v_5(S) <= 1 + v_5(k1) AND v_5(S) <= 1 + v_5(k2).
# For p != 3,5: v_p(n) = 0. Need v_p(S) <= v_p(k1) AND v_p(S) <= v_p(k2).

print("Searching (k1,k2) for n = 5*3^K:")
best = Fraction(1, 1)
best_k = None
for k1 in range(3, 2000):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        if ratio >= best:
            continue

        # Check conditions
        ok = True
        temp = S
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                vS = 0
                while temp % p == 0:
                    vS += 1
                    temp //= p
                if p == 3:
                    pass  # always fine
                elif p == 5:
                    vk1 = 0
                    t = k1
                    while t % 5 == 0:
                        vk1 += 1
                        t //= 5
                    vk2 = 0
                    t = k2
                    while t % 5 == 0:
                        vk2 += 1
                        t //= 5
                    if vS > 1 + vk1 or vS > 1 + vk2:
                        ok = False
                        break
                else:
                    vk1 = 0
                    t = k1
                    while t % p == 0:
                        vk1 += 1
                        t //= p
                    vk2 = 0
                    t = k2
                    while t % p == 0:
                        vk2 += 1
                        t //= p
                    if vS > vk1 or vS > vk2:
                        ok = False
                        break
            p += 1
        if ok and temp > 1:
            p = temp
            vS = 1
            if p == 3:
                pass
            elif p == 5:
                vk1 = 0
                t = k1
                while t % 5 == 0:
                    vk1 += 1
                    t //= 5
                vk2 = 0
                t = k2
                while t % 5 == 0:
                    vk2 += 1
                    t //= 5
                if vS > 1 + vk1 or vS > 1 + vk2:
                    ok = False
            else:
                vk1 = 0
                t = k1
                while t % p == 0:
                    vk1 += 1
                    t //= p
                vk2 = 0
                t = k2
                while t % p == 0:
                    vk2 += 1
                    t //= p
                if vS > vk1 or vS > vk2:
                    ok = False

        if ok:
            if ratio < best:
                best = ratio
                best_k = (k1, k2)
                print(f"  New best: (k1={k1}, k2={k2}): ratio={ratio}={float(ratio):.6f}, S={k1*k2+k1+k2}")

print(f"\nBest: ratio={best}={float(best):.6f}, k={best_k}")
print(f"g(4M) = 5 * {best} = {5*best}")
