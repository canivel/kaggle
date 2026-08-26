#!/usr/bin/env python3
"""
Problem 3: Acute triangle ABC with integer sides, AB < AC.
Points D on BC, E on AC with AD = AE = AB.
Line DE intersects line AB at X.
Circles BXD and CED intersect again at Y != D.
Y lies on line AD.
Find unique triangle with minimal perimeter, then abc mod 10^5.

Let me redo the geometry more carefully with exact/numerical computation.
"""

import math
import numpy as np

def check_triangle(a, b, c):
    """
    a = BC, b = CA, c = AB
    Need: c < b (AB < AC), acute, integer sides, triangle inequality.
    Returns (True, info) if the Y-on-AD condition holds.
    """
    # Triangle inequality
    if a + b <= c or a + c <= b or b + c <= a:
        return False
    # Acute
    if a*a + b*b <= c*c or a*a + c*c <= b*b or b*b + c*c <= a*a:
        return False
    # AB < AC
    if c >= b:
        return False

    # Coordinates: A = (0,0), B = (c, 0)
    cos_A = (b*b + c*c - a*a) / (2.0*b*c)
    sin_A_sq = 1 - cos_A*cos_A
    if sin_A_sq <= 1e-15:
        return False
    sin_A = math.sqrt(sin_A_sq)

    A = np.array([0.0, 0.0])
    B = np.array([float(c), 0.0])
    C = np.array([b*cos_A, b*sin_A])

    # D on segment BC with AD = AB = c
    # D = B + t*(C - B), |D|^2 = c^2
    # t = (a^2 + c^2 - b^2) / a^2
    t_D = (a*a + c*c - b*b) / float(a*a)
    if t_D <= 0 or t_D >= 1:
        return False

    D = B + t_D * (C - B)

    # Verify AD = c
    ad = np.linalg.norm(D - A)
    if abs(ad - c) > 1e-6:
        return False

    # E on segment AC with AE = AB = c
    # E = (c/b) * C
    E = (float(c) / b) * C

    # Verify AE = c
    ae = np.linalg.norm(E - A)
    if abs(ae - c) > 1e-6:
        return False

    # Line DE intersects line AB at X
    # Line AB is y = 0 (x-axis)
    # Parametric line through D and E: P = D + s*(E - D)
    # P_y = 0: D_y + s*(E_y - D_y) = 0
    denom = E[1] - D[1]
    if abs(denom) < 1e-12:
        return False
    s_X = -D[1] / denom
    X = D + s_X * (E - D)

    # X should be on line AB (y=0)
    assert abs(X[1]) < 1e-9

    # Now find circles BXD and CED, and their second intersection Y != D.
    # Then check if Y is on line AD.

    # Use the radical axis approach.
    # Both circles pass through D. Their radical axis is the line through D and Y.
    # The radical axis of two circles: for circle1 (x^2+y^2+p1*x+q1*y+r1=0)
    # and circle2 (x^2+y^2+p2*x+q2*y+r2=0), radical axis is:
    # (p1-p2)*x + (q1-q2)*y + (r1-r2) = 0

    # Circle 1: through B, X, D
    # Circle 2: through C, E, D

    # General circle: x^2 + y^2 + px + qy + r = 0
    # For 3 points, solve linear system for p, q, r.

    def circle_through_3_points(P1, P2, P3):
        """Find p, q, r such that x^2+y^2+px+qy+r=0 passes through P1, P2, P3."""
        # P_i: x_i^2 + y_i^2 + p*x_i + q*y_i + r = 0
        x1, y1 = P1
        x2, y2 = P2
        x3, y3 = P3
        A_mat = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])
        b_vec = np.array([
            -(x1*x1 + y1*y1),
            -(x2*x2 + y2*y2),
            -(x3*x3 + y3*y3)
        ])
        try:
            sol = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            return None
        return sol  # [p, q, r]

    circ1 = circle_through_3_points(B, X, D)
    circ2 = circle_through_3_points(C, E, D)

    if circ1 is None or circ2 is None:
        return False

    p1, q1, r1 = circ1
    p2, q2, r2 = circ2

    # Radical axis: (p1-p2)*x + (q1-q2)*y + (r1-r2) = 0
    # This line passes through both intersection points D and Y.
    # Line DY direction: normal to radical axis is (p1-p2, q1-q2)
    # The radical axis line has direction perpendicular to (p1-p2, q1-q2),
    # i.e., direction (-(q1-q2), p1-p2).

    dp = p1 - p2
    dq = q1 - q2
    dr = r1 - r2

    # Radical axis: dp*x + dq*y + dr = 0
    # This passes through D (verify):
    rad_D = dp*D[0] + dq*D[1] + dr
    if abs(rad_D) > 1e-6:
        # Numerical issue
        return False

    # Direction of radical axis: (-dq, dp)
    # Line AD: direction is D - A = D
    # Y is on both the radical axis and line AD.
    # Line AD: parametric: P = t * D for t in R.
    # On radical axis: dp*(t*D[0]) + dq*(t*D[1]) + dr = 0
    # t*(dp*D[0] + dq*D[1]) + dr = 0
    denom2 = dp*D[0] + dq*D[1]
    if abs(denom2) < 1e-12:
        # Radical axis is parallel to line AD through D, meaning they coincide (since D is on both)
        # This means all points on line AD satisfy the radical axis, which means
        # the radical axis IS line AD. This would mean Y can be anywhere on AD that's on both circles.
        # Let me handle this case separately.

        # Actually, if the radical axis is line AD, then Y is the other intersection of
        # line AD with either circle.
        # Line AD: P = t*D. On circle 1: |tD|^2 + p1*(tD[0]) + q1*(tD[1]) + r1 = 0
        # t^2*(D[0]^2+D[1]^2) + t*(p1*D[0]+q1*D[1]) + r1 = 0
        # t^2*c^2 + t*(p1*D[0]+q1*D[1]) + r1 = 0
        # One root is t=1 (point D), other is t = r1/c^2 (by Vieta's)
        t_Y = r1 / (c*c)
        Y = t_Y * D

        # Verify Y is on circle 2 as well
        check2 = Y[0]**2 + Y[1]**2 + p2*Y[0] + q2*Y[1] + r2
        if abs(check2) > 1e-4:
            return False
        return True

    t_Y = -dr / denom2
    if abs(t_Y - 1) < 1e-9:
        # Y = D, degenerate
        return False

    Y = t_Y * D

    # Verify Y is on both circles
    check1 = Y[0]**2 + Y[1]**2 + p1*Y[0] + q1*Y[1] + r1
    check2 = Y[0]**2 + Y[1]**2 + p2*Y[0] + q2*Y[1] + r2
    if abs(check1) > 1e-4 or abs(check2) > 1e-4:
        return False

    return True


# Search
found = []
for perimeter in range(4, 2000):
    for c in range(1, perimeter // 3 + 1):  # c is smallest or near smallest
        for b in range(c + 1, perimeter - c):  # b > c
            a = perimeter - b - c
            if a <= 0:
                continue
            if a + c <= b or a + b <= c or b + c <= a:
                continue
            result = check_triangle(a, b, c)
            if result:
                print(f"FOUND: a={a}, b={b}, c={c}, perimeter={perimeter}")
                found.append((perimeter, a, b, c))

    if found:
        break

if found:
    _, a, b, c = found[0]
    print(f"\nMinimal perimeter: a={a}, b={b}, c={c}")
    print(f"abc = {a*b*c}")
    print(f"abc mod 10^5 = {(a*b*c) % 100000}")
else:
    print("No solution found!")
