from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

# For c where 3 does not divide n (which is ALL four remaining c values since c mod 3 = 2):
# n is NOT divisible by 3.
# n is divisible by p (p!=3) iff p | (c+1).

# Summary of c+1 factorizations:
# c=1848374: c+1 = 3^2 * 5^3 * 31 * 53. Primes dividing c+1 (hence n): 5, 31, 53 (and 3 but 3 doesn't divide n)
#   Wait: p|(c+1) means p|n for p != 3. So 5|n, 31|n, 53|n. But 3 does NOT divide n.
# c=10162574: c+1 = 3^2 * 5^2 * 31^2 * 47. So 5|n, 31|n, 47|n.
# c=265710644: c+1 = 3^3 * 5 * 97 * 103 * 197. So 5|n, 97|n, 103|n, 197|n.
# c=44636594: c+1 = 3 * 5 * 103 * 167 * 173. So 5|n, 103|n, 167|n, 173|n.

# For none of these is n divisible by 2! (Since c+1 is always odd, c is even, M is odd, n = M+c is odd.)
# Wait: M = 3^(2025!) is odd. c is even (c mod 3 = 2, but c could be even or odd).
# Let me check: c = 1848374 (even). n = M + 1848374. M is odd + even = odd. So 2 does not divide n.
# c = 10162574 (even). n is odd. 2 does not divide n.
# c = 265710644 (even). n is odd.
# c = 44636594 (even). n is odd.
# So 2 never divides n for these c values.

# Available primes dividing n (for p != 3): only those dividing c+1.
# And 3 does NOT divide n.

# Now I need to find the best ratio for each c.
# Recall: for (k1,k2), ratio = k1*k2/(k1*k2+k1+k2), need certain divisibility of n.
# The requirement is complex. Let me think about it from the prime perspective.

# If p is a prime dividing n, what's the best ratio achievable using p?
# (k1=p-1, k2=2): ratio = 2(p-1)/(2(p-1)+p-1+2) = 2(p-1)/(3p-1).
#   Wait let me recompute. S = k1*k2+k1+k2 = 2(p-1) + (p-1) + 2 = 3p - 1.
#   ratio = 2(p-1)/(3p-1). Need: S/(gcd(S, k1*k2)) | n and lcm(k1,k2) | m.
#   Hmm, this doesn't directly require p|n. Let me think again.

# Actually, the requirement is more nuanced. Let me re-derive it properly.
# m = n * k1*k2 / S. We need:
# 1) m is a positive integer: S | (n * k1*k2)
# 2) k1 | m: k1 | (n*k1*k2/S) i.e., S | (n*k2) (or more precisely S/gcd(S,k2) | n... no)
#    k1 | m means k1 | (n*k1*k2/S). Since k1 already divides n*k1*k2 trivially (k1 is a factor),
#    we need k1 | (n*k1*k2/S), i.e., S | (n*k2). Actually:
#    m/k1 = n*k2/S. We need this to be a positive integer: S | (n*k2).
# 3) k2 | m: m/k2 = n*k1/S. Need S | (n*k1).
# 4) The three values m/k1, m/k2, m must be distinct positive integers: guaranteed by k1 > k2 > 1.

# So conditions: S | (n*k1) AND S | (n*k2).
# These imply S | (n*gcd(k1,k2)) and also S | (n*k1*k2) (already implied).
# S | (n*k1) means S/gcd(S,k1) | n. Similarly S/gcd(S,k2) | n.
# The binding constraint is: lcm(S/gcd(S,k1), S/gcd(S,k2)) divides n.

# This is getting complex. Let me just directly check for each (k1,k2) whether it works for given n.

# For c = 1848374: n is divisible by 5, 25, 125, 31, 53, and products thereof (but NOT 2 or 3).
# For c = 10162574: n divisible by 5, 25, 31, 961(=31^2), 47.
# For c = 265710644: n divisible by 5, 97, 103, 197.
# For c = 44636594: n divisible by 5, 103, 167, 173.

# Let me check small (k1,k2) pairs systematically for each c.

def check_ratio(k1, k2, n_divs):
    """Check if (k1,k2) pattern works given the set of primes dividing n.
    n_divs: dict mapping prime -> max power dividing n.
    Returns True if the divisibility conditions are satisfied.
    """
    S = k1*k2 + k1 + k2
    # Need S | (n*k1) and S | (n*k2)
    # i.e., S/gcd(S,k1) | n and S/gcd(S,k2) | n.
    req1 = S // gcd(S, k1)
    req2 = S // gcd(S, k2)
    # n must be divisible by lcm(req1, req2).
    req = lcm(req1, req2)
    # Check if req divides n. Factor req and check against n_divs.
    temp = req
    for p, e in sorted(n_divs.items()):
        while temp % p == 0:
            temp //= p
    # If temp == 1, all prime factors of req are in n_divs. But we also need sufficient powers.
    # Actually, n_divs should have the exact structure. For our problem:
    # n is divisible by ALL primes in n_divs (at least to the power given).
    # For primes not in n_divs, n is NOT divisible by them.
    # So we need every prime factor of req to be in n_divs.
    if temp == 1:
        return True
    # Check if remaining factors are covered
    return False

def best_ratio_for_c(c_val, label=""):
    """Find the best f(n)/n ratio for n = M + c where M = 3^(2025!)."""
    c1 = c_val + 1
    c_mod3 = c_val % 3

    # Find primes dividing n = M+c:
    # For p != 3: p | n iff p | (c+1)
    # For p = 3: 3 | n iff 3 | c

    # Get prime factorization of c+1 (for p != 3)
    n_primes = {}  # prime -> power in c+1 (sufficient condition for dividing n)
    temp = c1
    for p in range(2, 10000):
        if temp == 1:
            break
        while temp % p == 0:
            if p != 3:  # 3 dividing c+1 doesn't mean 3 divides n
                n_primes[p] = n_primes.get(p, 0) + 1
            temp //= p
    if temp > 1 and temp != 3:
        n_primes[temp] = 1

    # For powers of 3: if 3 | c, then 3 divides n. Actually since M = 3^K with K huge,
    # the 3-adic valuation of n = 3^K + c is v_3(c) if 3|c, or 0 if 3 doesn't divide c.
    if c_mod3 == 0 and c_val > 0:
        v3 = 0
        t = c_val
        while t % 3 == 0:
            v3 += 1
            t //= 3
        n_primes[3] = v3  # Not huge, just v_3(c)
    # If c = 0: n = M, n is divisible by 3^K (huge). Handled separately.

    # For the (k1,k2) approach, need to find divisibility:
    # n is divisible by p^a for each p^a in n_primes.
    # But n is also divisible by higher powers if the prime divides M-component differently.
    # For p != 3: n ≡ c+1 mod p. So v_p(n) = v_p(c+1). n_primes already has this.
    # For p = 3: v_3(n) = v_3(c) if c > 0 and 3|c; 0 if 3 doesn't divide c; K if c=0.
    # (For c = 4M: different analysis.)

    print(f"{label}: n_primes = {n_primes}")

    # Now search for best (k1,k2)
    best_ratio = Fraction(1)  # worst case
    best_k = None
    for k1 in range(2, 500):
        for k2 in range(2, k1):
            S = k1*k2 + k1 + k2
            ratio = Fraction(k1*k2, S)
            if ratio >= best_ratio:
                continue
            # Check divisibility
            req1 = S // gcd(S, k1)
            req2 = S // gcd(S, k2)
            req = lcm(req1, req2)
            # Factor req and check all prime factors are covered by n_primes
            temp = req
            for p in sorted(n_primes.keys()):
                pa = p ** n_primes[p]
                while temp % p == 0:
                    temp //= p
            if temp == 1:
                best_ratio = ratio
                best_k = (k1, k2)

    print(f"  Best ratio: {best_ratio} = {float(best_ratio):.10f}, k = {best_k}")
    return best_ratio

# c = 0
print("=== c = 0 ===")
# n = M, divisible by 3^K only. Among primes: only 3.
# Best: (6,3) giving 2/3. Let me verify by scanning.
n_primes_c0 = {3: 100}  # effectively infinite power of 3
best = Fraction(1)
best_k = None
for k1 in range(2, 200):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        if ratio >= best:
            continue
        req1 = S // gcd(S, k1)
        req2 = S // gcd(S, k2)
        req = lcm(req1, req2)
        # All prime factors of req must be 3
        temp = req
        while temp % 3 == 0:
            temp //= 3
        if temp == 1:
            best = ratio
            best_k = (k1, k2)
print(f"c=0: best ratio = {best}, k = {best_k}")
print(f"g(0) = {best}")
print()

# c = 4M: n = 5M
print("=== c = 4M ===")
n_primes_4M = {3: 100, 5: 1}  # 3^K and 5^1
best = Fraction(1)
best_k = None
for k1 in range(2, 200):
    for k2 in range(2, k1):
        S = k1*k2 + k1 + k2
        ratio = Fraction(k1*k2, S)
        if ratio >= best:
            continue
        req1 = S // gcd(S, k1)
        req2 = S // gcd(S, k2)
        req = lcm(req1, req2)
        temp = req
        while temp % 3 == 0:
            temp //= 3
        while temp % 5 == 0:
            temp //= 5
        if temp == 1:
            best = ratio
            best_k = (k1, k2)
print(f"c=4M: best ratio = {best}, k = {best_k}")
print(f"f(5M)/5M = {best}, f(5M)/M = {best * 5}")
print(f"g(4M) = {best * 5}")
print()

# Remaining c values
for c_name, c_val in [("1848374", 1848374), ("10162574", 10162574),
                        ("265710644", 265710644), ("44636594", 44636594)]:
    print(f"=== c = {c_name} ===")
    r = best_ratio_for_c(c_val, c_name)
    print(f"g({c_name}) = {r}")
    print()
