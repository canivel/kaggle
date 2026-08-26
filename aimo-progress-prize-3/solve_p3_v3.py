#!/usr/bin/env python3
"""
Detailed verification for Problem 3.
"""

import math
import numpy as np

def analyze_triangle(a, b, c, verbose=True):
    """
    a = BC, b = CA, c = AB
    """
    if verbose:
        print(f"\n=== Triangle a={a}, b={b}, c={c} ===")
        print(f"  Perimeter: {a+b+c}")

    # Checks
    if a + b <= c or a + c <= b or b + c <= a:
        if verbose: print("  Not a valid triangle")
        return None
    if a*a + b*b <= c*c or a*a + c*c <= b*b or b*b + c*c <= a*a:
        if verbose: print("  Not acute")
        return None
    if c >= b:
        if verbose: print("  Need c < b (AB < AC)")
        return None

    cos_A = (b*b + c*c - a*a) / (2.0*b*c)
    sin_A = math.sqrt(1 - cos_A*cos_A)

    A = np.array([0.0, 0.0])
    B = np.array([float(c), 0.0])
    C = np.array([b*cos_A, b*sin_A])

    if verbose:
        print(f"  A = {A}")
        print(f"  B = {B}")
        print(f"  C = {C}")
        print(f"  AB = {np.linalg.norm(B-A):.6f} (should be {c})")
        print(f"  AC = {np.linalg.norm(C-A):.6f} (should be {b})")
        print(f"  BC = {np.linalg.norm(C-B):.6f} (should be {a})")

    # D on BC with AD = c
    t_D = (a*a + c*c - b*b) / float(a*a)
    D = B + t_D * (C - B)

    if verbose:
        print(f"  t_D = {t_D:.6f} (should be in (0,1))")
        print(f"  D = {D}")
        print(f"  AD = {np.linalg.norm(D-A):.6f} (should be {c})")

    # E on AC with AE = c
    E = (float(c) / b) * C

    if verbose:
        print(f"  E = {E}")
        print(f"  AE = {np.linalg.norm(E-A):.6f} (should be {c})")

    # X: intersection of line DE with line AB (x-axis)
    denom = E[1] - D[1]
    if abs(denom) < 1e-12:
        if verbose: print("  DE parallel to AB")
        return None
    s_X = -D[1] / denom
    X = D + s_X * (E - D)

    if verbose:
        print(f"  X = {X}")
        print(f"  X on AB? y = {X[1]:.10f}")

    # Check if B, X, D are collinear (degenerate circle)
    area_BXD = abs((B[0]-D[0])*(X[1]-D[1]) - (X[0]-D[0])*(B[1]-D[1])) / 2
    if verbose:
        print(f"  Area BXD = {area_BXD:.10f}")
    if area_BXD < 1e-8:
        if verbose: print("  B, X, D collinear - degenerate circle!")
        return None

    # Check if C, E, D are collinear
    area_CED = abs((C[0]-D[0])*(E[1]-D[1]) - (E[0]-D[0])*(C[1]-D[1])) / 2
    if verbose:
        print(f"  Area CED = {area_CED:.10f}")
    if area_CED < 1e-8:
        if verbose: print("  C, E, D collinear - degenerate circle!")
        return None

    def circle_through_3(P1, P2, P3):
        x1, y1 = P1
        x2, y2 = P2
        x3, y3 = P3
        M = np.array([
            [x1, y1, 1],
            [x2, y2, 1],
            [x3, y3, 1]
        ])
        rhs = np.array([
            -(x1**2 + y1**2),
            -(x2**2 + y2**2),
            -(x3**2 + y3**2)
        ])
        return np.linalg.solve(M, rhs)

    circ1 = circle_through_3(B, X, D)  # Circle BXD
    circ2 = circle_through_3(C, E, D)  # Circle CED

    p1, q1, r1 = circ1
    p2, q2, r2 = circ2

    if verbose:
        print(f"  Circle BXD: p={p1:.6f}, q={q1:.6f}, r={r1:.6f}")
        print(f"  Circle CED: p={p2:.6f}, q={q2:.6f}, r={r2:.6f}")

    # Radical axis: (p1-p2)*x + (q1-q2)*y + (r1-r2) = 0
    dp = p1 - p2
    dq = q1 - q2
    dr = r1 - r2

    if verbose:
        print(f"  Radical axis: {dp:.6f}*x + {dq:.6f}*y + {dr:.6f} = 0")
        # Check D is on radical axis
        rad_D = dp*D[0] + dq*D[1] + dr
        print(f"  D on radical axis? {rad_D:.10f}")

    # Line AD: direction D, passes through origin
    # Parametric: P = t*D
    # Y on radical axis and on line AD:
    # dp*(t*D[0]) + dq*(t*D[1]) + dr = 0
    # t*(dp*D[0] + dq*D[1]) = -dr

    coeff = dp*D[0] + dq*D[1]
    if verbose:
        print(f"  dp*Dx + dq*Dy = {coeff:.10f}")

    if abs(coeff) < 1e-9:
        if verbose:
            print("  Radical axis contains line AD (or parallel)")
            # Check dr ≈ 0
            print(f"  dr = {dr:.10f}")
        if abs(dr) > 1e-6:
            if verbose: print("  Parallel but not coincident")
            return None

        # Radical axis is line AD. Find Y as the other intersection of line AD with circle 1.
        # t^2*(Dx^2+Dy^2) + t*(p1*Dx+q1*Dy) + r1 = 0
        # t^2*c^2 + t*(p1*Dx+q1*Dy) + r1 = 0
        c_sq = c * c
        B_coeff = p1*D[0] + q1*D[1]
        # By Vieta's: t1*t2 = r1/c^2, t1+t2 = -B_coeff/c^2
        # t1 = 1 (D), so t2 = r1/c^2
        t_Y = r1 / c_sq
        Y = t_Y * D

        # Verify
        chk1 = Y[0]**2 + Y[1]**2 + p1*Y[0] + q1*Y[1] + r1
        chk2 = Y[0]**2 + Y[1]**2 + p2*Y[0] + q2*Y[1] + r2
        if verbose:
            print(f"  t_Y = {t_Y:.6f}")
            print(f"  Y = {Y}")
            print(f"  Y on circle 1? {chk1:.10f}")
            print(f"  Y on circle 2? {chk2:.10f}")
            print(f"  Y != D? t_Y != 1? {abs(t_Y - 1) > 1e-6}")

        if abs(chk1) > 1e-4 or abs(chk2) > 1e-4:
            return None
        if abs(t_Y - 1) < 1e-6:
            return None  # Y = D

        return {'Y_on_AD': True, 'radical_axis_is_AD': True, 't_Y': t_Y}

    t_Y = -dr / coeff
    Y = t_Y * D

    if verbose:
        print(f"  t_Y = {t_Y:.6f}")
        print(f"  Y = {Y}")

    if abs(t_Y - 1) < 1e-6:
        if verbose: print("  Y = D (degenerate)")
        return None

    # Verify Y on both circles
    chk1 = Y[0]**2 + Y[1]**2 + p1*Y[0] + q1*Y[1] + r1
    chk2 = Y[0]**2 + Y[1]**2 + p2*Y[0] + q2*Y[1] + r2

    if verbose:
        print(f"  Y on circle BXD? residual = {chk1:.10f}")
        print(f"  Y on circle CED? residual = {chk2:.10f}")

    if abs(chk1) > 1e-4 or abs(chk2) > 1e-4:
        if verbose: print("  Y not on both circles!")
        return None

    if verbose:
        print(f"  SUCCESS: Y is on line AD and on both circles")

    return {'Y_on_AD': True, 'radical_axis_is_AD': False, 't_Y': t_Y}


# Analyze the found solution
result = analyze_triangle(7, 8, 6)

# Also search more broadly to confirm uniqueness and check other small cases
print("\n\n=== Systematic search ===")
found = []
for perimeter in range(4, 500):
    for c in range(1, perimeter // 3 + 1):
        for b in range(c + 1, perimeter - c):
            a = perimeter - b - c
            if a <= 0:
                continue
            if a + c <= b or a + b <= c or b + c <= a:
                continue
            if a*a + b*b <= c*c or a*a + c*c <= b*b or b*b + c*c <= a*a:
                continue
            r = analyze_triangle(a, b, c, verbose=False)
            if r is not None:
                print(f"  a={a}, b={b}, c={c}, perimeter={a+b+c}, radical_axis_is_AD={r['radical_axis_is_AD']}, t_Y={r['t_Y']:.6f}")
                found.append((a+b+c, a, b, c))

if found:
    found.sort()
    print(f"\nAll solutions found (sorted by perimeter):")
    for p, a, b, c in found[:20]:
        print(f"  a={a}, b={b}, c={c}, perimeter={p}, abc={a*b*c}, abc%100000={(a*b*c)%100000}")
