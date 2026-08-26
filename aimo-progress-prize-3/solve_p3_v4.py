#!/usr/bin/env python3
"""
More careful analysis. The condition is:
- Radical axis of circles BXD and CED is line AD.
This means Y (second intersection) lies on AD.

But wait - the radical axis IS the line through the two intersection points D and Y.
So Y being on line AD is equivalent to the radical axis being line AD, which is
equivalent to saying the line DY is line DA.

Let me verify: for (7,8,6), the radical axis passes through D in the direction of AD.
That's exactly the condition.

Actually I realize ALL my solutions have radical_axis_is_AD = True.
This means the condition simplifies to: the radical axis direction at D is the same as DA.

Let me derive the algebraic condition.

Actually wait, the radical axis always passes through D (since both circles pass through D).
The radical axis also passes through Y. If Y is on line AD, then the radical axis
passes through both D and a point on line AD, meaning the radical axis IS line AD
(since it passes through D and is in direction DA).

So the condition "Y on line AD" is equivalent to "radical axis = line AD",
which is equivalent to: direction of radical axis at D equals direction of DA.

Radical axis: (p1-p2)*x + (q1-q2)*y + (r1-r2) = 0
Direction of this line: perpendicular to (p1-p2, q1-q2), i.e., direction (-(q1-q2), p1-p2).

Line AD has direction D - A = D = (Dx, Dy).

These must be parallel:
Dx * (p1-p2) + Dy * (q1-q2) = 0  ... (*)

AND the line passes through D:
(p1-p2)*Dx + (q1-q2)*Dy + (r1-r2) = 0 ... (**)

But (**) is equivalent to (*) (with r1-r2 = 0) only if r1 = r2.

Actually, since D is on both circles:
Dx^2 + Dy^2 + p1*Dx + q1*Dy + r1 = 0
Dx^2 + Dy^2 + p2*Dx + q2*Dy + r2 = 0
Subtracting: (p1-p2)*Dx + (q1-q2)*Dy + (r1-r2) = 0

So (**) is automatically satisfied! The radical axis always passes through D.

Then (*) says the radical axis direction is perpendicular to (p1-p2, q1-q2),
i.e., direction (-(q1-q2), p1-p2). For this to be parallel to (Dx, Dy):
(-(q1-q2)) * Dy = (p1-p2) * Dx  ... wait that's wrong.

Parallel means: -(q1-q2)/Dx = (p1-p2)/Dy
i.e., -(q1-q2)*Dy = (p1-p2)*Dx
i.e., (p1-p2)*Dx + (q1-q2)*Dy = 0

But we already showed this equals -(r1-r2) from (**).
So the condition (*) is: r1 = r2.

So the condition Y on line AD is equivalent to r1 = r2 where the circles are
x^2+y^2+p1*x+q1*y+r1=0 and x^2+y^2+p2*x+q2*y+r2=0.

This is a clean algebraic condition! Let me now derive r1 and r2 in terms of a, b, c.
"""

import math
import numpy as np
from fractions import Fraction

def compute_exact(a, b, c):
    """Compute r1 - r2 exactly using rational arithmetic where possible."""
    # A = (0,0), B = (c, 0)
    # cos_A = (b^2+c^2-a^2)/(2bc)
    # sin_A^2 = 1 - cos_A^2

    # C = (b*cos_A, b*sin_A)

    # t_D = (a^2+c^2-b^2)/a^2
    # D = B + t_D*(C-B)
    # E = (c/b)*C

    # X: intersection of DE with x-axis
    # X = (Xx, 0) where Xx = Dx - Dy*(Ex-Dx)/(Ey-Dy)

    # Circle BXD: x^2+y^2+p1*x+q1*y+r1=0
    # Points B=(c,0), X=(Xx,0), D=(Dx,Dy)
    # B: c^2 + p1*c + r1 = 0
    # X: Xx^2 + p1*Xx + r1 = 0
    # => p1*(c-Xx) = Xx^2 - c^2 = (Xx-c)(Xx+c) => p1 = -(Xx+c) [if Xx != c]
    # r1 = -c^2 - p1*c = -c^2 + (Xx+c)*c = -c^2 + c*Xx + c^2 = c*Xx
    # So r1 = c*Xx

    # Circle CED: x^2+y^2+p2*x+q2*y+r2=0
    # Points C=(Cx,Cy), E=(Ex,Ey), D=(Dx,Dy)
    # We need r2.

    # Actually, r = -x1*x2 - y1*y2 when the circle passes through origin? No.
    # The general formula: for circle through P1, P2, P3:
    # Using the matrix formula:
    # | x^2+y^2   x   y   1 |
    # | x1^2+y1^2 x1  y1  1 | = 0
    # | x2^2+y2^2 x2  y2  1 |
    # | x3^2+y3^2 x3  y3  1 |

    # We found r1 = c*Xx. Now we need r2.

    # From the 3 points C, E, D and the system:
    # C: Cx^2+Cy^2 + p2*Cx + q2*Cy + r2 = 0
    # E: Ex^2+Ey^2 + p2*Ex + q2*Ey + r2 = 0
    # D: Dx^2+Dy^2 + p2*Dx + q2*Dy + r2 = 0

    # Since Dx^2+Dy^2 = c^2 (AD = c), the D equation gives:
    # c^2 + p2*Dx + q2*Dy + r2 = 0 ... (D)

    # Since Ex^2+Ey^2 = c^2 (AE = c), the E equation gives:
    # c^2 + p2*Ex + q2*Ey + r2 = 0 ... (E)

    # (D)-(E): p2*(Dx-Ex) + q2*(Dy-Ey) = 0
    # So (Dx-Ex)*p2 + (Dy-Ey)*q2 = 0 ... (*)

    # Since Cx^2+Cy^2 = b^2 (AC = b), the C equation gives:
    # b^2 + p2*Cx + q2*Cy + r2 = 0 ... (C)

    # From (D): r2 = -c^2 - p2*Dx - q2*Dy
    # Substituting into (C): b^2 + p2*Cx + q2*Cy - c^2 - p2*Dx - q2*Dy = 0
    # (b^2-c^2) + p2*(Cx-Dx) + q2*(Cy-Dy) = 0 ... (**)

    # From (*): p2 = -q2*(Dy-Ey)/(Dx-Ex) [if Dx != Ex]
    # Substituting into (**):
    # (b^2-c^2) + (-q2*(Dy-Ey)/(Dx-Ex))*(Cx-Dx) + q2*(Cy-Dy) = 0
    # (b^2-c^2) + q2*[-(Dy-Ey)*(Cx-Dx)/(Dx-Ex) + (Cy-Dy)] = 0
    # q2 = -(b^2-c^2) / [-(Dy-Ey)*(Cx-Dx)/(Dx-Ex) + (Cy-Dy)]
    # q2 = -(b^2-c^2)*(Dx-Ex) / [-(Dy-Ey)*(Cx-Dx) + (Cy-Dy)*(Dx-Ex)]

    # And r2 = -c^2 - p2*Dx - q2*Dy

    # The condition is r1 = r2, i.e., c*Xx = r2.

    # Let me compute everything symbolically with exact fractions.

    # Use Fraction for exactness
    a2 = Fraction(a*a)
    b2 = Fraction(b*b)
    c2 = Fraction(c*c)

    cos_A = (b2 + c2 - a2) / (2*b*c)
    # sin_A^2 = 1 - cos_A^2
    sin_A_sq = 1 - cos_A * cos_A

    Cx = b * cos_A
    Cy_sq = b2 * sin_A_sq  # Cy^2

    t_D = (a2 + c2 - b2) / a2
    # D = B + t_D*(C-B)
    Dx = c + t_D * (Cx - c)
    # Dy = t_D * Cy (keep symbolic using Dy^2 = t_D^2 * Cy^2)
    Dy_sq = t_D * t_D * Cy_sq

    # E = (c/b)*C
    Ex = Fraction(c, b) * Cx
    Ey_over_Cy = Fraction(c, b)  # Ey = (c/b)*Cy

    # X on x-axis: X = D + s*(E-D) where s = -Dy/(Ey-Dy)
    # Ey - Dy = (c/b)*Cy - t_D*Cy = Cy*(c/b - t_D)
    # s = -t_D*Cy / (Cy*(c/b - t_D)) = -t_D / (c/b - t_D) [Cy cancels]
    cb = Fraction(c, b)
    s_X = -t_D / (cb - t_D)
    Xx = Dx + s_X * (Ex - Dx)

    # r1 = c * Xx
    r1 = c * Xx

    # Now compute r2
    # From the system above, using the trick that AD=AE=c:
    # r2 = -c^2 - p2*Dx - q2*Dy

    # First, Dx - Ex
    DmE_x = Dx - Ex
    # Dy - Ey = t_D*Cy - (c/b)*Cy = Cy*(t_D - c/b)
    DmE_y_over_Cy = t_D - cb  # (Dy - Ey)/Cy

    # (*): DmE_x * p2 + DmE_y_over_Cy * Cy * q2 = 0
    # p2 = -DmE_y_over_Cy * Cy * q2 / DmE_x   ... (*)

    # Cx - Dx
    CmD_x = Cx - Dx
    # Cy - Dy = Cy - t_D*Cy = Cy*(1 - t_D)
    CmD_y_over_Cy = 1 - t_D  # (Cy - Dy)/Cy

    # (**): (b^2 - c^2) + p2*CmD_x + q2*Cy*CmD_y_over_Cy = 0

    # Substituting (*) into (**):
    # (b^2-c^2) + (-DmE_y_over_Cy*Cy*q2/DmE_x)*CmD_x + q2*Cy*CmD_y_over_Cy = 0
    # (b^2-c^2) + q2*Cy*[-DmE_y_over_Cy*CmD_x/DmE_x + CmD_y_over_Cy] = 0
    # q2 = -(b^2-c^2) / (Cy * [-DmE_y_over_Cy*CmD_x/DmE_x + CmD_y_over_Cy])

    bracket = -DmE_y_over_Cy * CmD_x / DmE_x + CmD_y_over_Cy
    # q2*Cy = -(b^2-c^2) / bracket
    q2_times_Cy = -(b2 - c2) / bracket

    # p2 = -DmE_y_over_Cy * q2_times_Cy / DmE_x
    p2 = -DmE_y_over_Cy * q2_times_Cy / DmE_x

    # r2 = -c^2 - p2*Dx - q2*Dy = -c^2 - p2*Dx - q2_times_Cy * (Dy/Cy)
    # Dy/Cy = t_D
    r2 = -c2 - p2 * Dx - q2_times_Cy * t_D

    diff = r1 - r2
    return diff

# Test with (7, 8, 6)
for a, b, c in [(7,8,6), (5,6,4), (5,7,4), (6,7,5), (8,9,7), (3,4,2), (4,5,3), (28,27,15), (10,11,9)]:
    try:
        if a+b<=c or a+c<=b or b+c<=a:
            continue
        if c >= b:
            continue
        if a*a+b*b<=c*c or a*a+c*c<=b*b or b*b+c*c<=a*a:
            continue
        t = (a*a+c*c-b*b)
        if t <= 0 or t >= a*a:
            continue
        diff = compute_exact(a, b, c)
        print(f"a={a}, b={b}, c={c}: r1-r2 = {diff} = {float(diff):.10f}")
    except Exception as e:
        print(f"a={a}, b={b}, c={c}: error {e}")

print("\n\nSearching for r1=r2 condition algebraically...")

# The condition r1 = r2 gives us an algebraic equation in a, b, c.
# Let me try to simplify it.

# Let me derive the condition symbolically.
# Set up: a, b, c integers, c < b, acute triangle.
# cos_A = (b^2+c^2-a^2)/(2bc)
# t_D = (a^2+c^2-b^2)/a^2
# E_ratio = c/b (E is c/b of the way along AC)

# Cx = b*cos_A = (b^2+c^2-a^2)/(2c)
# Dx = c + t_D*(Cx - c) = c + ((a^2+c^2-b^2)/a^2) * ((b^2+c^2-a^2)/(2c) - c)
#     = c + ((a^2+c^2-b^2)/a^2) * ((b^2+c^2-a^2-2c^2)/(2c))
#     = c + ((a^2+c^2-b^2)/a^2) * ((b^2-c^2-a^2)/(2c))
#     = c - ((a^2+c^2-b^2)*(a^2-b^2+c^2))/(2c*a^2)
#     = c - ((a^2+c^2-b^2)^2)/(2c*a^2)

# Hmm that's the same thing squared. Let me use u = a^2+c^2-b^2 for shorthand.
# t_D = u/a^2
# Cx - c = (b^2+c^2-a^2)/(2c) - c = (b^2+c^2-a^2-2c^2)/(2c) = (b^2-a^2-c^2)/(2c) = -u/(2c)
# So Dx = c + (u/a^2)*(-u/(2c)) = c - u^2/(2c*a^2)
# Dx = (2c^2*a^2 - u^2) / (2c*a^2)

# Ex = (c/b)*Cx = (c/b)*(b^2+c^2-a^2)/(2c) = (b^2+c^2-a^2)/(2b)
# Ey/Cy = c/b, Dy/Cy = t_D = u/a^2

# DmE_x = Dx - Ex = (2c^2*a^2 - u^2)/(2c*a^2) - (b^2+c^2-a^2)/(2b)
# Let me get a common denominator 2bca^2:
# = b*(2c^2*a^2 - u^2)/(2bca^2) - ca^2*(b^2+c^2-a^2)/(2bca^2)
# = [b(2c^2*a^2 - u^2) - ca^2(b^2+c^2-a^2)] / (2bca^2)

# Note: b^2+c^2-a^2 = -u + 2b^2... no. u = a^2+c^2-b^2, so b^2+c^2-a^2 = 2c^2 - u... no.
# u = a^2+c^2-b^2
# b^2+c^2-a^2 = -(a^2-c^2-b^2) = -(u - 2c^2)... hmm.
# Let me just say w = b^2+c^2-a^2 = 2b^2 - (a^2+b^2-c^2)...
# Actually simply: u + w = 2c^2 where w = b^2+c^2-a^2... no.
# u = a^2+c^2-b^2, w = b^2+c^2-a^2. u+w = 2c^2. And u-w = 2(a^2-b^2).

# This is getting messy. Let me just compute numerically for many triangles
# and see which satisfy r1=r2.

count = 0
found_all = []
for a in range(2, 300):
    for c in range(1, a):
        for b in range(c+1, a+c):
            if a + b <= c or a + c <= b or b + c <= a:
                continue
            if a*a+b*b <= c*c or a*a+c*c <= b*b or b*b+c*c <= a*a:
                continue
            t_num = a*a + c*c - b*b
            if t_num <= 0 or t_num >= a*a:
                continue
            try:
                diff = compute_exact(a, b, c)
                if diff == 0:
                    count += 1
                    found_all.append((a+b+c, a, b, c))
                    if count <= 30:
                        print(f"a={a}, b={b}, c={c}, peri={a+b+c}")
            except:
                pass

print(f"\nTotal found: {count}")
found_all.sort()
print("\nSmallest perimeter solutions:")
for p, a, b, c in found_all[:10]:
    print(f"  a={a}, b={b}, c={c}, perimeter={p}, abc={a*b*c}, abc%100000={(a*b*c)%100000}")
