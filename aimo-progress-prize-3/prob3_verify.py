import math
import numpy as np

def circumcircle(P1, P2, P3):
    """Return center and radius of circle through 3 points."""
    ax, ay = P1
    bx, by = P2
    cx, cy = P3
    D = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-12:
        return None, None
    ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / D
    uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / D
    r = math.sqrt((ax-ux)**2 + (ay-uy)**2)
    return (ux, uy), r

def circle_circle_intersect(c1, r1, c2, r2):
    """Find intersection points of two circles."""
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    d = math.sqrt(dx*dx + dy*dy)
    if d > r1 + r2 + 1e-10 or d < abs(r1-r2) - 1e-10:
        return []
    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h2 = r1*r1 - a*a
    if h2 < 0:
        h2 = 0
    h = math.sqrt(h2)
    mx = c1[0] + a*dx/d
    my = c1[1] + a*dy/d
    p1 = (mx + h*dy/d, my - h*dx/d)
    p2 = (mx - h*dy/d, my + h*dx/d)
    return [p1, p2]

def verify_triangle(a, b, c):
    """Full verification for triangle with BC=a, CA=b, AB=c."""
    print(f"\nVerifying triangle: a=BC={a}, b=CA={b}, c=AB={c}")

    # Coordinates: A=(0,0), B=(c,0)
    cosA = (b*b + c*c - a*a) / (2*b*c)
    sinA = math.sqrt(1 - cosA*cosA)

    A = (0.0, 0.0)
    B = (float(c), 0.0)
    C = (b * cosA, b * sinA)

    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = ({C[0]:.6f}, {C[1]:.6f})")

    # Verify sides
    print(f"AB = {math.dist(A,B):.6f} (should be {c})")
    print(f"BC = {math.dist(B,C):.6f} (should be {a})")
    print(f"CA = {math.dist(C,A):.6f} (should be {b})")

    # E on AC with AE = c
    E = (c * cosA, c * sinA)
    print(f"E = ({E[0]:.6f}, {E[1]:.6f})")
    print(f"AE = {math.dist(A,E):.6f} (should be {c})")

    # D on BC with AD = c
    t = (a*a + c*c - b*b) / (a*a)
    D = (B[0] + t*(C[0]-B[0]), B[1] + t*(C[1]-B[1]))
    print(f"D = ({D[0]:.6f}, {D[1]:.6f})")
    print(f"AD = {math.dist(A,D):.6f} (should be {c})")
    print(f"D on BC: BD={math.dist(B,D):.6f}, DC={math.dist(D,C):.6f}, BC={math.dist(B,C):.6f}")
    print(f"BD+DC = {math.dist(B,D)+math.dist(D,C):.6f}")

    # X = line DE intersects line AB (x-axis)
    if abs(E[1] - D[1]) < 1e-12:
        print("DE parallel to AB!")
        return
    s = -D[1] / (E[1] - D[1])
    X = (D[0] + s*(E[0]-D[0]), 0.0)
    print(f"X = ({X[0]:.6f}, {X[1]:.6f})")

    # Verify X on line DE
    # Check X is on line AB
    print(f"X on AB (y=0): y={X[1]:.10f}")

    # Circle BXD
    center1, r1 = circumcircle(B, X, D)
    print(f"Circle BXD: center=({center1[0]:.6f}, {center1[1]:.6f}), r={r1:.6f}")

    # Circle CED
    center2, r2 = circumcircle(C, E, D)
    print(f"Circle CED: center=({center2[0]:.6f}, {center2[1]:.6f}), r={r2:.6f}")

    # Find intersections
    pts = circle_circle_intersect(center1, r1, center2, r2)
    print(f"Intersection points: {len(pts)}")

    for i, p in enumerate(pts):
        print(f"  Point {i}: ({p[0]:.6f}, {p[1]:.6f})")
        dist_to_D = math.dist(p, D)
        print(f"    Distance to D: {dist_to_D:.6f}")

        # Check if on line AD
        # Line AD: (x,y) = t*(Dx, Dy)
        if abs(D[0]) > 1e-10:
            t_x = p[0] / D[0]
        else:
            t_x = None
        if abs(D[1]) > 1e-10:
            t_y = p[1] / D[1]
        else:
            t_y = None

        # Cross product with D to check collinearity with A and D
        cross = p[0]*D[1] - p[1]*D[0]
        print(f"    Cross product (should be 0 if on line AD): {cross:.10f}")

        if dist_to_D > 0.01:
            print(f"    This is Y (not D)")
            print(f"    Y on line AD: {'YES' if abs(cross) < 1e-6 else 'NO'}")

verify_triangle(7, 8, 6)
