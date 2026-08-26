import math

def solve():
    results = []

    for perimeter in range(4, 1000):
        for a in range(1, perimeter):
            for c in range(1, min(a, perimeter - a)):
                b = perimeter - a - c
                if b <= c:  # need c < b (AB < AC means c < b)
                    continue
                if b >= a + c or a >= b + c or c >= a + b:
                    continue
                # Check acute: all angles < 90
                if a*a + b*b <= c*c:
                    continue
                if a*a + c*c <= b*b:
                    continue
                if b*b + c*c <= a*a:
                    continue

                cos_A = (b*b + c*c - a*a) / (2.0*b*c)
                sin_A_sq = 1 - cos_A*cos_A
                if sin_A_sq <= 0:
                    continue
                sin_A = math.sqrt(sin_A_sq)

                Bx, By = float(c), 0.0
                Cx = b * cos_A
                Cy = b * sin_A

                # D on BC with AD = c
                t_D = (a*a + c*c - b*b) / float(a*a)
                if t_D <= 0 or t_D >= 1:
                    continue

                Dx = c + t_D * (Cx - c)
                Dy = t_D * Cy

                # Verify AD = c
                ad_sq = Dx*Dx + Dy*Dy
                if abs(ad_sq - c*c) > 1e-6:
                    continue

                # E on AC with AE = c
                Ex = c * cos_A
                Ey = c * sin_A

                # Line DE intersects AB (x-axis) at X
                if abs(Ey - Dy) < 1e-12:
                    continue
                s_X = -Dy / (Ey - Dy)
                Xx = Dx + s_X * (Ex - Dx)

                # Circle 1 through B, X, D: x^2+y^2+p1*x+q1*y+r1=0
                if abs(Xx - Bx) < 1e-12:
                    continue
                p1 = Xx + Bx
                r1 = -Bx*Bx - p1*Bx
                if abs(Dy) < 1e-12:
                    continue
                q1 = -(Dx*Dx + Dy*Dy + p1*Dx + r1) / Dy

                # Circle 2 through C, E, D
                dCE_x = Cx - Ex
                dCE_y = Cy - Ey
                dCE_sq = (Cx*Cx - Ex*Ex) + (Cy*Cy - Ey*Ey)

                dCD_x = Cx - Dx
                dCD_y = Cy - Dy
                dCD_sq = (Cx*Cx - Dx*Dx) + (Cy*Cy - Dy*Dy)

                det2 = dCE_x * dCD_y - dCD_x * dCE_y
                if abs(det2) < 1e-12:
                    continue

                p2 = (-dCE_sq * dCD_y + dCD_sq * dCE_y) / det2
                q2 = (-dCD_sq * dCE_x + dCE_sq * dCD_x) / det2
                r2 = -(Cx*Cx + Cy*Cy + p2*Cx + q2*Cy)

                # Y = (lam*Dx, lam*Dy) on both circles
                # Circle 1: lam^2*c^2 + lam*(p1*Dx+q1*Dy) + r1 = 0
                A1 = c*c
                B1 = p1*Dx + q1*Dy
                C1 = r1

                disc1 = B1*B1 - 4*A1*C1
                if disc1 < -1e-9:
                    continue
                if disc1 < 0:
                    disc1 = 0

                lam1_a = (-B1 + math.sqrt(disc1)) / (2*A1)
                lam1_b = (-B1 - math.sqrt(disc1)) / (2*A1)

                # One should be 1 (point D)
                if abs(lam1_a - 1) < 1e-6:
                    lam_Y1 = lam1_b
                elif abs(lam1_b - 1) < 1e-6:
                    lam_Y1 = lam1_a
                else:
                    continue

                # Circle 2: lam^2*c^2 + lam*(p2*Dx+q2*Dy) + r2 = 0
                A2 = c*c
                B2 = p2*Dx + q2*Dy
                C2 = r2

                disc2 = B2*B2 - 4*A2*C2
                if disc2 < -1e-9:
                    continue
                if disc2 < 0:
                    disc2 = 0

                lam2_a = (-B2 + math.sqrt(disc2)) / (2*A2)
                lam2_b = (-B2 - math.sqrt(disc2)) / (2*A2)

                if abs(lam2_a - 1) < 1e-6:
                    lam_Y2 = lam2_b
                elif abs(lam2_b - 1) < 1e-6:
                    lam_Y2 = lam2_a
                else:
                    continue

                if abs(lam_Y1 - lam_Y2) < 1e-6:
                    P = a + b + c
                    results.append((P, a, b, c, lam_Y1))
                    print(f"Found: a={a}, b={b}, c={c}, perimeter={P}, lam_Y={lam_Y1:.6f}")

        if results:
            break

    return results

results = solve()
if results:
    res = results[0]
    a, b, c_side = res[1], res[2], res[3]
    print(f"\nMinimal perimeter triangle: a={a}, b={b}, c={c_side}")
    print(f"abc = {a*b*c_side}")
    print(f"abc mod 10^5 = {(a*b*c_side) % 100000}")
else:
    print("No solution found in search range")
