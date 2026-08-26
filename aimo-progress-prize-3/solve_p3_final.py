#!/usr/bin/env python3
"""
Final verification for Problem 3.
Let me derive the algebraic condition r1 = r2 and find the simplest solution.

Using the Fraction computation, let me explicitly compute r1 - r2 as a polynomial in a, b, c
by using sympy.
"""
from sympy import symbols, simplify, Rational, sqrt, factor, solve, cancel

a, b, c = symbols('a b c', positive=True)

# Setup
cos_A = (b**2 + c**2 - a**2) / (2*b*c)

# t_D = (a^2+c^2-b^2)/a^2
u = a**2 + c**2 - b**2  # = 2*c^2 - w where w = b^2+c^2-a^2
t_D = u / a**2

# Cx = (b^2+c^2-a^2)/(2c)
Cx = (b**2 + c**2 - a**2) / (2*c)

# Dx = c + t_D*(Cx - c) = c + (u/a^2)*((b^2+c^2-a^2)/(2c) - c)
# = c + (u/a^2)*((b^2+c^2-a^2-2c^2)/(2c))
# = c + (u/a^2)*(b^2-c^2-a^2)/(2c)
# = c - u*(a^2+c^2-b^2)/(2c*a^2)  ... wait b^2-c^2-a^2 = -(a^2+c^2-b^2) = -u
# = c - u^2/(2*c*a^2)
Dx = c - u**2 / (2*c*a**2)

# Ex = (c/b) * Cx = (b^2+c^2-a^2)/(2*b)
Ex = (b**2 + c**2 - a**2) / (2*b)

# Dy/Cy = t_D = u/a^2
# Ey/Cy = c/b

# s_X = -Dy/(Ey-Dy) = -t_D/(c/b - t_D) = -(u/a^2)/((c/b) - (u/a^2))
# = -(u/a^2) / ((ca^2 - bu)/(ba^2))
# = -(u/a^2) * (ba^2)/(ca^2 - bu)
# = -bu/(ca^2 - bu)
s_X = -b*u / (c*a**2 - b*u)

# Xx = Dx + s_X*(Ex - Dx)
Xx = Dx + s_X * (Ex - Dx)
Xx_simplified = simplify(Xx)
print("Xx =", Xx_simplified)

# r1 = c*Xx
r1 = c * Xx_simplified
r1_simplified = simplify(r1)
print("r1 =", r1_simplified)

# Now r2 from circle CED
# Using the formulas from before:
# DmE_y/Cy = t_D - c/b = u/a^2 - c/b = (bu - ca^2)/(ba^2)
DmE_y_over_Cy = t_D - c/b
# CmD_y/Cy = 1 - t_D = 1 - u/a^2 = (a^2-u)/a^2 = (b^2-c^2)/a^2  [since u=a^2+c^2-b^2]
CmD_y_over_Cy = 1 - t_D

DmE_x = Dx - Ex
CmD_x = Cx - Dx

bracket = -DmE_y_over_Cy * CmD_x / DmE_x + CmD_y_over_Cy
q2_Cy = -(b**2 - c**2) / bracket
p2 = -DmE_y_over_Cy * q2_Cy / DmE_x
r2 = -c**2 - p2 * Dx - q2_Cy * t_D

r2_simplified = simplify(r2)
print("r2 =", r2_simplified)

diff = simplify(r1_simplified - r2_simplified)
print("\nr1 - r2 =", diff)
diff_factored = factor(diff)
print("r1 - r2 (factored) =", diff_factored)

# Set diff = 0 and solve for the relationship between a, b, c
# Since the triangle has integer sides with c < b, and is acute
print("\nSetting r1 - r2 = 0:")
eq = diff_factored
print("Equation:", eq, "= 0")

# Try to extract the numerator
from sympy import numer, denom, fraction
num, den = fraction(eq)
print("Numerator:", factor(num))
print("Denominator:", factor(den))
