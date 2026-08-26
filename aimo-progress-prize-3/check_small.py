#!/usr/bin/env python3
"""Check ALL triangles with perimeter <= 20 for the condition."""
import math

for b in range(2, 20):
    for c in range(1, b):
        # a^2 = (b+c)^2*(b-c)/b
        num = (b+c)**2 * (b-c)
        if num % b != 0:
            continue
        a_sq = num // b
        a = int(math.isqrt(a_sq))
        if a*a != a_sq or a <= 0:
            continue
        if a+b <= c or a+c <= b or b+c <= a:
            continue
        peri = a + b + c
        acute = a*a+b*b > c*c and a*a+c*c > b*b and b*b+c*c > a*a
        t_num = a*a + c*c - b*b
        d_on_bc = 0 < t_num < a*a

        print(f"a={a}, b={b}, c={c}, peri={peri}, acute={acute}, D_on_BC={d_on_bc}")

# Also check: are there solutions where the condition involves a different branch?
# For instance, maybe D is not between B and C for some triangles?
# t_D = (a^2+c^2-b^2)/a^2. For acute triangle: a^2+c^2 > b^2 (angle B < 90),
# so t_D > 0. And t_D < 1 iff c^2 < b^2 iff c < b. So D is always on BC.
print("\nAll valid (acute, D on BC) solutions with perimeter <= 25:")
for b in range(2, 25):
    for c in range(1, b):
        num = (b+c)**2 * (b-c)
        if num % b != 0:
            continue
        a_sq = num // b
        a = int(math.isqrt(a_sq))
        if a*a != a_sq or a <= 0:
            continue
        if a+b <= c or a+c <= b or b+c <= a:
            continue
        if not (a*a+b*b > c*c and a*a+c*c > b*b and b*b+c*c > a*a):
            continue
        t_num = a*a + c*c - b*b
        if not (0 < t_num < a*a):
            continue
        peri = a + b + c
        if peri <= 25:
            print(f"  a={a}, b={b}, c={c}, peri={peri}")
