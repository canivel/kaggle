import math
from decimal import Decimal, getcontext

def max_digit_sum(m):
    best = 0
    for b in range(2, m+1):
        s = 0
        val = m
        while val > 0:
            s += val % b
            val //= b
        if s > best:
            best = s
    return best

# Verify max digit sum = ceil(m/2) for small values
all_match = True
for m in range(2, 200):
    mds = max_digit_sum(m)
    expected = (m + 1) // 2  # ceil(m/2)
    if mds != expected:
        print(f"m={m}: max_digit_sum={mds}, ceil(m/2)={expected}")
        all_match = False

if all_match:
    print("Verification complete. All match for m in [2, 199].")

# Compute M = f(10^(10^5)) = floor(log2(10^(10^5) - 1)) + 1
# = floor(10^5 * log2(10)) + 1  (since 10^(10^5) is not a power of 2)

val = 100000 * math.log2(10)
print(f"10^5 * log2(10) = {val}")
print(f"floor = {math.floor(val)}")

getcontext().prec = 50
log2_10 = Decimal(10).ln() / Decimal(2).ln()
print(f"log2(10) high precision = {log2_10}")
result = 100000 * log2_10
print(f"10^5 * log2(10) = {result}")
print(f"floor = {int(result)}")

M = int(result) + 1
print(f"M = {M}")
print(f"M mod 10^5 = {M % 100000}")
