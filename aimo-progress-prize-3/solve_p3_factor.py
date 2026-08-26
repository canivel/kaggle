#!/usr/bin/env python3
from sympy import symbols, factor, solve, sqrt, Rational

a, b, c = symbols('a b c', positive=True)

expr = a**2 * b - b**3 - b**2*c + b*c**2 + c**3
print("Expression:", expr)
print("Factored:", factor(expr))

# a^2*b = b^3 + b^2*c - b*c^2 - c^3
# a^2*b = b^2(b+c) - c^2(b+c) = (b+c)(b^2 - c^2) = (b+c)(b-c)(b+c) = (b+c)^2*(b-c)
# Wait let me check:
check = (b+c)**2 * (b-c) - (b**3 + b**2*c - b*c**2 - c**3)
print("Check (b+c)^2*(b-c):", factor(check))

# So a^2*b = (b+c)^2*(b-c)
# a^2 = (b+c)^2*(b-c)/b
# For a^2 to be a positive integer (with b > c > 0), we need b | (b+c)^2*(b-c)

# Actually let's verify:
from sympy import expand
print("(b+c)^2*(b-c) =", expand((b+c)**2*(b-c)))
print("b^3 + b^2*c - b*c^2 - c^3 =", expand(b**3 + b**2*c - b*c**2 - c**3))

# Yes! They match. So the condition is:
# a^2 = (b+c)^2*(b-c)/b

# For integer solutions: a, b, c positive integers, c < b, a^2*b = (b+c)^2*(b-c)
# Also need: acute triangle (all angles < 90 deg), and t_D in (0,1) [D on segment BC]

# a^2 = (b+c)^2*(b-c)/b
# For a to be integer, b must divide (b+c)^2*(b-c)

# Let g = gcd(b, c). Write b = g*B, c = g*C with gcd(B,C) = 1.
# a^2 = (g*B + g*C)^2 * (g*B - g*C) / (g*B) = g^2*(B+C)^2 * g*(B-C) / (g*B)
# = g^2 * (B+C)^2 * (B-C) / B

# Since gcd(B,C)=1 and gcd(B, B+C) = gcd(B, C) = 1, B does not divide (B+C)^2*(B-C)
# unless B divides (B-C). But gcd(B, B-C) = gcd(B, C) = 1.
# So B must divide 1, meaning B = 1.

# So b = g, c = g*C with gcd(1, C) = 1 (always true), but b = g > c = g*C requires C < 1...
# Wait that's wrong. b = g*B = g*1 = g. c = g*C. Need c < b: g*C < g => C < 1. But C >= 1.
# Unless C = 0 but c must be positive. Contradiction!

# Let me re-check. b > c > 0, so B > C > 0 with B, C positive integers, gcd(B,C) = 1.
# b = g*B, c = g*C.

# a^2 = g^2*(B+C)^2*(B-C)/B

# For a^2 to be a perfect square, g^2*(B+C)^2*(B-C)/B must be a perfect square.
# g^2*(B+C)^2 is already a perfect square. So we need (B-C)/B to be a perfect square of a rational.
# I.e., (B-C)/B = (B-C)/B, and we need B | (B-C) in some sense for the whole thing to be integer.

# Since gcd(B,C) = 1, gcd(B, B-C) = gcd(B, C) = 1. So gcd(B, B-C) = 1.
# Thus for (B-C)/B to yield integer a^2, we need... hmm.

# Actually a^2 = g^2 * (B+C)^2 * (B-C) / B
# For this to be a positive integer: B | g^2 * (B+C)^2 * (B-C)
# Since gcd(B, B+C) = gcd(B,C) = 1 and gcd(B, B-C) = gcd(B,C) = 1:
# B | g^2

# So write g^2 = B * k^2 * m where... let's say B | g^2.
# Let g = B^alpha * h where gcd(h, B) = 1.
# g^2 = B^{2*alpha} * h^2. B | g^2 requires 2*alpha >= 1, so alpha >= 1.
# Actually just need B | g^2. Since B is a positive integer with gcd(B,C)=1,
# let's parametrize differently.

# Let B = d^2 * e where e is square-free.
# g^2/B = g^2 / (d^2*e). For this to contribute to a perfect square with (B+C)^2*(B-C):
# a^2 = g^2*(B+C)^2*(B-C)/(d^2*e)
# = (g*(B+C)/d)^2 * (B-C)/e
# Need e | (B-C) and (B-C)/e to be a perfect square.

# This is getting complicated. Let me just do a direct numerical search with the condition
# a^2*b = (b+c)^2*(b-c), checking for integer a, and the acute/validity conditions.

print("\n\nNumerical search for a^2*b = (b+c)^2*(b-c):")
import math

results = []
for b_val in range(2, 500):
    for c_val in range(1, b_val):
        # a^2 = (b+c)^2*(b-c)/b
        num = (b_val + c_val)**2 * (b_val - c_val)
        if num % b_val != 0:
            continue
        a_sq = num // b_val
        a_val = int(math.isqrt(a_sq))
        if a_val * a_val != a_sq:
            continue
        if a_val <= 0:
            continue
        # Check triangle inequality
        if a_val + b_val <= c_val or a_val + c_val <= b_val or b_val + c_val <= a_val:
            continue
        # Check acute
        if a_val**2 + b_val**2 <= c_val**2:
            continue
        if a_val**2 + c_val**2 <= b_val**2:
            continue
        if b_val**2 + c_val**2 <= a_val**2:
            continue
        # Check D on BC: t_D = (a^2+c^2-b^2)/a^2 in (0,1)
        t_num = a_val**2 + c_val**2 - b_val**2
        if t_num <= 0 or t_num >= a_val**2:
            continue
        # Also need denominator of radical axis nonzero (a^2 != b^2 + b*c)
        if a_val**2 == b_val**2 + b_val*c_val:
            continue
        peri = a_val + b_val + c_val
        results.append((peri, a_val, b_val, c_val))

results.sort()
print(f"Found {len(results)} solutions")
for peri, a_val, b_val, c_val in results[:20]:
    abc = a_val * b_val * c_val
    print(f"  a={a_val}, b={b_val}, c={c_val}, perimeter={peri}, abc={abc}, abc%100000={abc%100000}")

# Check: is (7,8,6) the unique primitive solution?
# Primitive means gcd(a,b,c) = 1
print("\nPrimitive solutions (gcd(a,b,c)=1):")
for peri, a_val, b_val, c_val in results[:50]:
    if math.gcd(math.gcd(a_val, b_val), c_val) == 1:
        abc = a_val * b_val * c_val
        print(f"  a={a_val}, b={b_val}, c={c_val}, perimeter={peri}, abc={abc}, abc%100000={abc%100000}")
