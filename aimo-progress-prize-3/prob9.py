# Problem 9: alpha = phi^3 = 2 + sqrt(5). p=2, q=5.
# Find floor(p^(q^p)) mod 99991 = floor(2^(5^2)) mod 99991 = floor(2^25) mod 99991
# Wait, the problem says floor(p^(q^p)). p=2, q=5, so q^p = 5^2 = 25.
# p^(q^p) = 2^25 = 33554432.
# floor(2^25) = 33554432 (already integer).
# 33554432 mod 99991.

val = pow(2, 25, 99991)
print(f"2^25 = {2**25}")
print(f"2^25 mod 99991 = {val}")
