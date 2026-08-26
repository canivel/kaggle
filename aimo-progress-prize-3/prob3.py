import math

def check_triangle(a, b, c):
    """Check if triangle with sides a=BC, b=CA, c=AB satisfies the condition.
    AB < AC means c < b.
    Triangle is acute."""

    if c >= b:
        return False

    # Check triangle inequality
    if a + b <= c or a + c <= b or b + c <= a:
        return False

    # Check acute
    if a*a >= b*b + c*c:
        return False
    if b*b >= a*a + c*c:
        return False
    if c*c >= a*a + b*b:
        return False

    # Coordinates: A=(0,0), B=(c,0)
    cosA = (b*b + c*c - a*a) / (2*b*c)
    if abs(cosA) >= 1:
        return False
    sinA = math.sqrt(1 - cosA*cosA)

    Cx = b * cosA
    Cy = b * sinA

    # E on AC with AE = c
    Ex = c * cosA
    Ey = c * sinA

    # D on BC with AD = c
    t = (a*a + c*c - b*b) / (a*a)
    if t <= 0 or t >= 1:
        return False

    Bx, By = c, 0.0
    Dx = Bx + t * (Cx - Bx)
    Dy = By + t * (Cy - By)

    # Verify AD = c
    ad2 = Dx*Dx + Dy*Dy
    if abs(ad2 - c*c) > 1e-6:
        return False

    # Line DE: X is intersection with y=0 (line AB)
    if abs(Ey - Dy) < 1e-12:
        return False

    s = -Dy / (Ey - Dy)
    Xx = Dx + s * (Ex - Dx)

    # Condition: Xx = b (derived from radical axis argument)
    # pow(A, circle BXD) = c * Xx
    # pow(A, circle CED) = c * b
    # Need c * Xx = c * b => Xx = b

    return abs(Xx - b) < 1e-6


results = []
for perimeter in range(3, 3000):
    for a in range(1, perimeter):
        for c in range(1, min(a, perimeter - a)):
            b = perimeter - a - c
            if b <= c:
                continue
            if b >= a + c or a >= b + c or c >= a + b:
                continue
            # c < b guaranteed
            # Check acute
            if a*a >= b*b + c*c:
                continue
            if b*b >= a*a + c*c:
                continue
            if c*c >= a*a + b*b:
                continue

            if check_triangle(a, b, c):
                results.append((a, b, c, a+b+c))
                print(f"Found: a={a}, b={b}, c={c}, perimeter={a+b+c}, abc={a*b*c}")

    if results and perimeter > results[0][3] + 10:
        break

if results:
    results.sort(key=lambda x: x[3])
    a, b, c, p = results[0]
    print(f"\nMinimal perimeter: a={a}, b={b}, c={c}, perimeter={p}")
    print(f"abc = {a*b*c}")
    print(f"abc mod 10^5 = {(a*b*c) % 100000}")
else:
    print("No triangle found!")
