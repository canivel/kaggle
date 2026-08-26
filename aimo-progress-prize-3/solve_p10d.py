from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# Verify (k1=16, k2=2) pattern:
# S = 16*2 + 16 + 2 = 50
# ratio = 32/50 = 16/25
# m = n*32/50 = 16n/25
# Need S | (n*k1): 50 | 16n -> 25 | 8n -> 25 | n (since gcd(8,25)=1). So need 25 | n.
# Need S | (n*k2): 50 | 2n -> 25 | n. Same condition.
# Also need k1 | m: 16 | 16n/25. Need 25 | n. Then m = 16n/25, and 16 | 16n/25 iff 25 | n.
# Which is the same.
# Also need k2 | m: 2 | 16n/25. Need 25 | n, then m = 16n/25. If 25|n and n is odd,
# m = 16*(n/25) which is even. So 2 | m. Good.
#
# Triple: (m/16, m/2, m) = (n/25, 8n/25, 16n/25). Sum = n/25 + 8n/25 + 16n/25 = 25n/25 = n. Good.
# All three are distinct: n/25 < 8n/25 < 16n/25. Good.
# All divide m = 16n/25: m/(n/25) = 16, m/(8n/25) = 2, m/m = 1. Good.

# Condition: 25 | n. For n = M + c:
# For p = 5: n ≡ 1+c mod 5. 25 | n requires 25 | (1+c)... wait no.
# v_5(n) = v_5(c+1) for p=5 (since M ≡ 1 mod 5^k for any k with phi(5^k) | 2025!).
# phi(5^k) = 4*5^(k-1). For k such that 4*5^(k-1) | 2025!:
# 2025! contains factor 5 to power floor(2025/5)+floor(2025/25)+... ≈ 504.
# And 4 | 2025!. So phi(5^k) = 4*5^(k-1) divides 2025! for k-1 <= 504, i.e., k <= 505.
# So M ≡ 1 mod 5^505 (at least). So v_5(n) = v_5(M+c) = v_5(c+1) (since M ≡ 1 mod 5^k for large k).

# c = 1848374: c+1 = 1848375 = 3^2 * 5^3 * 31 * 53. v_5(c+1) = 3. So 25 | n (and 125 | n).
# So (16,2) works: f(n)/n = 16/25, g = 16/25.

# c = 265710644: c+1 = 265710645 = 3^3 * 5 * 97 * 103 * 197. v_5(c+1) = 1. So 5 | n but 25 does NOT.
# 25 does NOT divide n! So (16,2) does NOT work here.

# Wait, I had a bug in my check. Let me re-examine.
# The requirement for (16,2) is 25 | n. But v_5(n) = v_5(c+1).
# For c = 265710644: v_5(c+1) = 1. So 5 | n but 25 does not divide n. (16,2) fails!

# I need to fix the check. The issue is that I was checking if all prime factors of req
# divide n, but I wasn't checking POWERS.

def best_ratio_for_c(c_val, label=""):
    c1 = c_val + 1
    c_mod3 = c_val % 3

    # Find prime factorization of c+1 (for primes p != 3)
    n_prime_vals = {}  # prime -> exact valuation in c+1 (= valuation in n for p != 3)
    temp = c1
    for p in range(2, 10000):
        if temp == 1:
            break
        e = 0
        while temp % p == 0:
            e += 1
            temp //= p
        if e > 0:
            if p != 3:
                n_prime_vals[p] = e
    if temp > 1 and temp != 3:
        n_prime_vals[temp] = 1
    elif temp == 3:
        pass  # 3 divides c+1 but not n

    # For p = 3: v_3(n) = v_3(c) if 3|c, else 0.
    if c_mod3 == 0 and c_val > 0:
        v3 = 0
        t = c_val
        while t % 3 == 0:
            v3 += 1
            t //= 3
        n_prime_vals[3] = v3

    print(f"{label}: n prime valuations = {n_prime_vals}")

    # Now search for best (k1,k2)
    best_ratio = Fraction(1)
    best_k = None
    for k1 in range(2, 500):
        for k2 in range(2, k1):
            S = k1*k2 + k1 + k2
            ratio = Fraction(k1*k2, S)
            if ratio >= best_ratio:
                continue
            # Check divisibility: need S | (n*k1) and S | (n*k2)
            # Equivalently: for each prime p^a in factorization of S:
            #   p^a must divide n*k1 (so p^(a - v_p(k1)) must divide n)
            #   p^a must divide n*k2 (so p^(a - v_p(k2)) must divide n)
            # The binding constraint on n: for each prime p dividing S,
            #   n must be divisible by p^max(a - v_p(k1), a - v_p(k2), 0) where a = v_p(S).

            # Factor S
            s_temp = S
            s_factors = {}
            for p in range(2, S+1):
                if s_temp == 1:
                    break
                e = 0
                while s_temp % p == 0:
                    e += 1
                    s_temp //= p
                if e > 0:
                    s_factors[p] = e

            # For each prime in S, check n has enough power
            ok = True
            for p, a in s_factors.items():
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
                need = max(a - vk1, a - vk2, 0)
                have = n_prime_vals.get(p, 0)
                if have < need:
                    ok = False
                    break
            if ok:
                best_ratio = ratio
                best_k = (k1, k2)

    print(f"  Best ratio: {best_ratio} = {float(best_ratio):.10f}, k = {best_k}")
    return best_ratio

# c = 0
print("=== c = 0 ===")
r0 = best_ratio_for_c(0, "c=0")
print(f"g(0) = {r0}")
print()

# c = 4M: need special handling
print("=== c = 4M ===")
# n = 5M = 5*3^K. v_5(n) = 1, v_3(n) = K (huge).
# For p != 3,5: n ≡ 5 mod p. So p|n iff p|5, i.e., only p=5.
n_prime_vals_4M = {3: 1000, 5: 1}  # 3 with huge power, 5 with power 1
best = Fraction(1)
best_k = None
for k1 in range(2, 500):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        if ratio >= best:
            continue
        s_temp = S
        s_factors = {}
        for p in range(2, S+1):
            if s_temp == 1:
                break
            e = 0
            while s_temp % p == 0:
                e += 1
                s_temp //= p
            if e > 0:
                s_factors[p] = e
        ok = True
        for p, a in s_factors.items():
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
            need = max(a - vk1, a - vk2, 0)
            have = n_prime_vals_4M.get(p, 0)
            if have < need:
                ok = False
                break
        if ok:
            best = ratio
            best_k = (k1, k2)
print(f"c=4M: best ratio = {best}, k = {best_k}")
print(f"g(4M) = {best * 5}")
print()

# Other c values
g_values = {}
g_values[0] = r0

for c_name, c_val in [("1848374", 1848374), ("10162574", 10162574),
                        ("265710644", 265710644), ("44636594", 44636594)]:
    print(f"=== c = {c_name} ===")
    r = best_ratio_for_c(c_val, c_name)
    g_values[c_val] = r
    print(f"g({c_name}) = {r}")
    print()

g_values['4M'] = best * 5

print("\n=== SUMMARY ===")
total = Fraction(0)
for key in [0, '4M', 1848374, 10162574, 265710644, 44636594]:
    v = g_values[key]
    print(f"g({key}) = {v} = {float(v):.10f}")
    total += v

print(f"\nSum = {total} = {float(total):.10f}")
print(f"p/q = {total} where p = {total.numerator}, q = {total.denominator}")
print(f"gcd check: gcd(p,q) = {gcd(total.numerator, total.denominator)}")
print(f"p + q = {total.numerator + total.denominator}")
print(f"(p + q) mod 99991 = {(total.numerator + total.denominator) % 99991}")
