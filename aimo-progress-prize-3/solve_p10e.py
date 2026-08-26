from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

def best_ratio_search(n_prime_vals, label="", max_k1=500):
    """Find the best f(n)/n ratio for n with given prime valuations.
    n_prime_vals: dict mapping prime -> valuation in n.
    """
    best_ratio = Fraction(1)
    best_k = None
    for k1 in range(2, max_k1):
        for k2 in range(2, k1):
            S = k1*k2 + k1 + k2
            ratio = Fraction(k1*k2, S)
            if ratio >= best_ratio:
                continue
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

    print(f"  {label}: Best ratio: {best_ratio} = {float(best_ratio):.10f}, k = {best_k}")
    return best_ratio

# c = 0: n = M = 3^(2025!).
# v_3(n) is huge (= 2025!). For all other primes p, v_p(n) = 0.
print("=== c = 0 ===")
r0 = best_ratio_search({3: 10000}, "c=0")  # 3 with effectively infinite power
print(f"g(0) = {r0}")
print()

# c = 4M: n = 5M = 5*3^(2025!).
# v_3(n) = 2025! (huge), v_5(n) = 1. All other primes: 0.
print("=== c = 4M ===")
r4M = best_ratio_search({3: 10000, 5: 1}, "c=4M")
print(f"g(4M) = f(5M)/(5M) * 5 = {r4M} * 5 = {r4M * 5}")
g_4M = r4M * 5
print()

# c = 1848374: c+1 = 3^2 * 5^3 * 31 * 53
# v_3(n) = 0 (c mod 3 = 2, so 3 does not divide n)
# v_5(n) = 3, v_31(n) = 1, v_53(n) = 1. Others: 0.
print("=== c = 1848374 ===")
r1 = best_ratio_search({5: 3, 31: 1, 53: 1}, "c=1848374")
print(f"g(1848374) = {r1}")
print()

# c = 10162574: c+1 = 3^2 * 5^2 * 31^2 * 47
# v_3(n) = 0, v_5(n) = 2, v_31(n) = 2, v_47(n) = 1. Others: 0.
print("=== c = 10162574 ===")
r2 = best_ratio_search({5: 2, 31: 2, 47: 1}, "c=10162574")
print(f"g(10162574) = {r2}")
print()

# c = 265710644: c+1 = 3^3 * 5 * 97 * 103 * 197
# v_3(n) = 0, v_5(n) = 1, v_97(n) = 1, v_103(n) = 1, v_197(n) = 1.
print("=== c = 265710644 ===")
r3 = best_ratio_search({5: 1, 97: 1, 103: 1, 197: 1}, "c=265710644")
print(f"g(265710644) = {r3}")
print()

# c = 44636594: c+1 = 3 * 5 * 103 * 167 * 173
# v_3(n) = 0, v_5(n) = 1, v_103(n) = 1, v_167(n) = 1, v_173(n) = 1.
print("=== c = 44636594 ===")
r4 = best_ratio_search({5: 1, 103: 1, 167: 1, 173: 1}, "c=44636594")
print(f"g(44636594) = {r4}")
print()

# Now compute g(c) properly.
# g(c) = floor(2025! * f(M+c) / M) / 2025!
# For c != 4M: f(M+c) = ratio * (M+c) = ratio * M + ratio * c.
# f(M+c)/M = ratio * (1 + c/M) = ratio + ratio*c/M.
# 2025! * f(M+c)/M = 2025! * ratio + 2025! * ratio * c / M.
# The second term: 2025! * ratio * c / M. Since M = 3^(2025!), this is astronomically small.
# BUT: is f(M+c) exactly ratio * (M+c)?
# Yes! If the divisibility conditions are satisfied (which they are for our specific n).
# Because m = ratio * n is an integer (the conditions ensure this).
# So f(M+c)/M = ratio * (M+c)/M = ratio + ratio*c/M.
# 2025! * ratio is rational. Is it an integer?
# ratio = p_r/q_r in lowest terms. 2025! * p_r/q_r. Need q_r | 2025!.
# For the ratios we found, q_r is a product of small primes (< 500), so q_r | 2025!.
# Therefore 2025! * ratio is an integer.
# And 2025! * ratio * c / M is positive but extremely tiny (< 1).
# So floor(2025! * ratio + tiny_positive) = 2025! * ratio.
# Hence g(c) = ratio exactly.

# For c = 4M: f(5M) = ratio_4M * 5M. f(5M)/M = 5 * ratio_4M.
# g(4M) = floor(2025! * 5 * ratio_4M) / 2025! = 5 * ratio_4M (since 2025!*5*ratio is integer).

print("\n=== SUMMARY ===")
g_vals = {
    0: r0,
    '4M': g_4M,
    1848374: r1,
    10162574: r2,
    265710644: r3,
    44636594: r4,
}

total = Fraction(0)
for key in [0, '4M', 1848374, 10162574, 265710644, 44636594]:
    v = g_vals[key]
    print(f"g({key}) = {v} = {float(v):.10f}")
    total += v

print(f"\nSum = {total}")
print(f"Sum as decimal = {float(total):.10f}")
p = total.numerator
q = total.denominator
print(f"p/q = {p}/{q}")
print(f"gcd(p,q) = {gcd(p,q)}")
print(f"p + q = {p + q}")
print(f"(p + q) mod 99991 = {(p + q) % 99991}")
