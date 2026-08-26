"""Final computation for Problem 5."""

def legendre(n, p):
    """v_p(n!) using Legendre's formula"""
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def v_catalan(m, p):
    """v_p(C_m) where C_m is the m-th Catalan number = C(2m,m)/(m+1)"""
    return legendre(2*m, p) - legendre(m, p) - legendre(m+1, p)

# N(20) = prod_{j=0}^{19} C_{2^j}^{2^{19-j}}
# v_p(N(20)) = sum_{j=0}^{19} 2^{19-j} * v_p(C_{2^j})

n = 20
v2_total = 0
v5_total = 0
print("j | m=2^j | v_2(C_m) | v_5(C_m) | 2^(19-j) | v2_contrib | v5_contrib")
print("-" * 80)
for j in range(n):
    m = 2**j
    v2 = v_catalan(m, 2)
    v5 = v_catalan(m, 5)
    exp = 2**(n-1-j)
    v2_c = exp * v2
    v5_c = exp * v5
    v2_total += v2_c
    v5_total += v5_c
    print(f"{j:2d} | {m:>7d} | {v2:>8d} | {v5:>8d} | {exp:>8d} | {v2_c:>10d} | {v5_c:>10d}")

print()
print(f"v_2(N(20)) = {v2_total}")
print(f"v_5(N(20)) = {v5_total}")
k = min(v2_total, v5_total)
print(f"k = v_10(N(20)) = min({v2_total}, {v5_total}) = {k}")
print(f"k mod 10^5 = {k % 100000}")
