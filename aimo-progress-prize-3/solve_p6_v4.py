"""
Problem 6: Let's check if there's any m where f(m) > floor(log2(m)) + 1.
Also let's find the exact formula and check larger starting values.

The pattern seems to be f(m) = floor(log2(m)) + 1 for m >= 2.
Let's verify and then compute for n up to 10^(10^5).
"""

import math

def digit_sum(m, b):
    s = 0
    while m > 0:
        s += m % b
        m //= b
    return s

# Compute f(m) exactly for m up to LIMIT
LIMIT = 50000
f = [0] * (LIMIT + 1)

for m in range(2, LIMIT + 1):
    best_f = 0
    for b in range(2, m + 1):
        ds = digit_sum(m, b)
        if ds >= m or ds < 1:
            continue
        val = 1 + f[ds]
        if val > best_f:
            best_f = val
        # Early termination: once we've tried up to m//2 + 2,
        # larger bases give smaller digit sums
        if b > m // 2 + 2:
            break
    f[m] = best_f

# Check if f(m) = floor(log2(m)) + 1 for all m >= 2
violations = []
for m in range(2, LIMIT + 1):
    expected = m.bit_length()  # = floor(log2(m)) + 1
    if f[m] != expected:
        violations.append((m, f[m], expected))

if violations:
    print(f"Found {len(violations)} violations!")
    for m, actual, expected in violations[:20]:
        print(f"  m={m}: f(m)={actual}, expected={expected}")
else:
    print(f"f(m) = floor(log2(m)) + 1 for all m in [2, {LIMIT}]")

# Wait, let me double check: f(3) = 2, floor(log2(3)) + 1 = 1 + 1 = 2. OK.
# f(2) = 1, floor(log2(2)) + 1 = 1 + 1 = 2. WAIT, that's wrong!
# f(2) = 1 but bit_length(2) = 2.
# Let me recheck.

print(f"\nf(2) = {f[2]}, bit_length(2) = {(2).bit_length()}")
print(f"f(3) = {f[3]}, bit_length(3) = {(3).bit_length()}")
print(f"f(4) = {f[4]}, bit_length(4) = {(4).bit_length()}")

# Hmm f(2) = 1 but bit_length = 2. So it's not bit_length.
# f(m) for m=2..16: 1,2,2,3,3,3,3,3,4,4,4,4,4,4,4
# That's: f(2)=1, f(3..4)=2, f(5..8)=3, f(9..16)=4
# So f(m) = floor(log2(m-1)) + 1 for m >= 2?
# f(2) = floor(log2(1))+1 = 0+1 = 1. ✓
# f(3) = floor(log2(2))+1 = 1+1 = 2. ✓
# f(5) = floor(log2(4))+1 = 2+1 = 3. ✓
# f(9) = floor(log2(8))+1 = 3+1 = 4. ✓
# f(17) = floor(log2(16))+1 = 4+1 = 5. ✓
#
# Or equivalently: f(m) = floor(log2(m)) when m is a power of 2, and floor(log2(m))+1 otherwise?
# f(2) = 1 = floor(log2(2)) = 1. ✓
# f(4) = 2 = floor(log2(4)) = 2. ✓
# f(8) = 3 = floor(log2(8)) = 3. ✓
# f(3) = 2, floor(log2(3))+1 = 1+1 = 2. ✓
# f(5) = 3, floor(log2(5))+1 = 2+1 = 3. ✓
#
# Actually f(m) = ceil(log2(m)) for m >= 2?
# ceil(log2(2)) = 1. f(2)=1. ✓
# ceil(log2(3)) = 2. f(3)=2. ✓
# ceil(log2(4)) = 2. f(4)=2. ✓
# ceil(log2(5)) = 3. f(5)=3. ✓
# ceil(log2(8)) = 3. f(8)=3. ✓
# ceil(log2(9)) = 4. f(9)=4. ✓
# YES!

# Let's verify: f(m) = ceil(log2(m)) for m >= 2.
print("\nVerifying f(m) = ceil(log2(m)):")
violations2 = []
for m in range(2, LIMIT + 1):
    expected = math.ceil(math.log2(m))
    if f[m] != expected:
        violations2.append((m, f[m], expected))

if violations2:
    print(f"Found {len(violations2)} violations!")
    for m, actual, expected in violations2[:20]:
        print(f"  m={m}: f(m)={actual}, expected ceil(log2({m}))={expected}")
else:
    print(f"f(m) = ceil(log2(m)) for all m in [2, {LIMIT}]")

# Hmm, but log2 might have floating point issues. Let me use exact computation.
# ceil(log2(m)) = bit_length(m-1) for m >= 2. Let's verify.
# Actually ceil(log2(m)) for m >= 2:
# If m is a power of 2, m = 2^k: ceil(log2(m)) = k = bit_length(m) - 1
# If m is not a power of 2: ceil(log2(m)) = bit_length(m) - 1... no.
# bit_length(m) = floor(log2(m)) + 1.
# ceil(log2(m)) = floor(log2(m)) if m is a power of 2, else floor(log2(m)) + 1.
# Hmm that doesn't simplify nicely. Let me just use (m-1).bit_length().
# For m=2: (1).bit_length() = 1. ceil(log2(2)) = 1. ✓
# For m=3: (2).bit_length() = 2. ceil(log2(3)) = 2. ✓
# For m=4: (3).bit_length() = 2. ceil(log2(4)) = 2. ✓
# For m=5: (4).bit_length() = 3. ceil(log2(5)) = 3. ✓
# For m=8: (7).bit_length() = 3. ceil(log2(8)) = 3. ✓
# For m=9: (8).bit_length() = 4. ceil(log2(9)) = 4. ✓
# Great!

print("\nVerifying f(m) = (m-1).bit_length():")
violations3 = []
for m in range(2, LIMIT + 1):
    expected = (m - 1).bit_length()
    if f[m] != expected:
        violations3.append((m, f[m], expected))

if violations3:
    print(f"Found {len(violations3)} violations!")
    for m, actual, expected in violations3[:20]:
        print(f"  m={m}: f(m)={actual}, expected (m-1).bit_length()={expected}")
else:
    print(f"f(m) = (m-1).bit_length() for all m in [2, {LIMIT}]")
    print(f"This equals ceil(log2(m)) for m >= 2.")
