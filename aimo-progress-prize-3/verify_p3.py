#!/usr/bin/env python3
"""Verify that (7,8,6) and (17,27,24) both satisfy all conditions."""
import numpy as np
import math

def full_verify(a, b, c):
    print(f"\n=== Verifying a={a}, b={b}, c={c} ===")
    print(f"  perimeter = {a+b+c}")

    # Basic checks
    assert a + b > c and a + c > b and b + c > a, "Triangle inequality fails"
    assert c < b, "Need AB < AC"
    assert a*a + b*b > c*c and a*a + c*c > b*b and b*b + c*c > a*a, "Not acute"
    print("  Triangle inequality: OK, Acute: OK, AB < AC: OK")

    # Algebraic condition: a^2*b = (b+c)^2*(b-c)
    lhs = a*a*b
    rhs = (b+c)**2 * (b-c)
    print(f"  a^2*b = {lhs}, (b+c)^2*(b-c) = {rhs}, match = {lhs == rhs}")

    # Coordinate geometry verification
    cos_A = (b*b + c*c - a*a) / (2.0*b*c)
    sin_A = math.sqrt(1 - cos_A**2)

    A = np.array([0.0, 0.0])
    B = np.array([float(c), 0.0])
    C = np.array([b*cos_A, b*sin_A])

    # D on BC with AD = c
    t_D = (a*a + c*c - b*b) / float(a*a)
    D = B + t_D * (C - B)
    print(f"  AD = {np.linalg.norm(D):.6f} (should be {c})")
    print(f"  t_D = {t_D:.6f} (in (0,1))")

    # E on AC with AE = c
    E = (float(c)/b) * C
    print(f"  AE = {np.linalg.norm(E):.6f} (should be {c})")

    # X: intersection of DE with line AB
    s = -D[1] / (E[1] - D[1])
    X = D + s * (E - D)
    print(f"  X = ({X[0]:.6f}, {X[1]:.10f})")

    # Circles
    def circle_3pts(P1, P2, P3):
        M = np.array([[P1[0], P1[1], 1],
                       [P2[0], P2[1], 1],
                       [P3[0], P3[1], 1]])
        rhs = np.array([-(P1[0]**2+P1[1]**2), -(P2[0]**2+P2[1]**2), -(P3[0]**2+P3[1]**2)])
        return np.linalg.solve(M, rhs)

    c1 = circle_3pts(B, X, D)
    c2 = circle_3pts(C, E, D)

    # Y: second intersection
    # Line AD: Y = t*D
    # On circle 1: t^2*|D|^2 + t*(p1*Dx+q1*Dy) + r1 = 0
    p1, q1, r1 = c1

    A1 = D[0]**2 + D[1]**2
    B1 = p1*D[0] + q1*D[1]
    # t=1 is D. Other root: t = r1/A1 (by Vieta's)
    t_Y = r1 / A1
    Y = t_Y * D

    # Verify Y on both circles
    chk1 = Y[0]**2 + Y[1]**2 + c1[0]*Y[0] + c1[1]*Y[1] + c1[2]
    chk2 = Y[0]**2 + Y[1]**2 + c2[0]*Y[0] + c2[1]*Y[1] + c2[2]
    print(f"  Y = ({Y[0]:.6f}, {Y[1]:.6f})")
    print(f"  Y on circle BXD? residual = {chk1:.2e}")
    print(f"  Y on circle CED? residual = {chk2:.2e}")
    print(f"  Y != D? t_Y = {t_Y:.6f} != 1")
    print(f"  Y on line AD? YES (by construction)")

    # Check that Y is actually a real point and the circles actually intersect
    disc = B1**2 - 4*A1*r1
    print(f"  Discriminant for circle 1 intersection with AD: {disc:.6f} >= 0? {disc >= -1e-9}")

full_verify(7, 8, 6)
full_verify(17, 27, 24)
full_verify(28, 27, 15)
