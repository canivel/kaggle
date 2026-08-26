def f_fast(n, limit=None):
    if limit is None:
        limit = 10*n
    for m in range(1, limit + 1):
        divs = []
        for d in range(1, int(m**0.5)+1):
            if m % d == 0:
                divs.append(d)
                if d != m // d:
                    divs.append(m // d)
        divs.sort()
        nd = len(divs)
        found = False
        for i in range(nd - 2):
            target = n - divs[i]
            lo, hi = i + 1, nd - 1
            while lo < hi:
                s = divs[lo] + divs[hi]
                if s == target:
                    found = True
                    break
                elif s < target:
                    lo += 1
                else:
                    hi -= 1
            if found:
                break
        if found:
            return m
    return None

print("f(3^k) for small k:")
for k in range(1, 10):
    n = 3**k
    m = f_fast(n, 5*n)
    frac34 = 3*(n-1)//4
    if m is not None:
        print(f"  f(3^{k}) = f({n}) = {m}, f/n = {m/n:.6f}, 3(n-1)/4 = {frac34}")
    else:
        print(f"  f(3^{k}) = f({n}) = NOT FOUND, 3(n-1)/4 = {frac34}")

# Also check: what divisors achieve the minimum for each?
print()
for k in range(1, 8):
    n = 3**k
    m = f_fast(n, 5*n)
    if m is None:
        continue
    divs = []
    for d in range(1, m+1):
        if m % d == 0:
            divs.append(d)
    # Find all triples
    nd = len(divs)
    for i in range(nd):
        for j in range(i+1, nd):
            for kk in range(j+1, nd):
                if divs[i] + divs[j] + divs[kk] == n:
                    print(f"  3^{k}={n}: m={m}, divisors ({divs[i]}, {divs[j]}, {divs[kk]})")
