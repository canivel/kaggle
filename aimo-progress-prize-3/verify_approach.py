from math import gcd
from fractions import Fraction

def lcm(a, b):
    return a * b // gcd(a, b)

def f_norwegian(n):
    """Brute force: find smallest m with 3 distinct divisors summing to n."""
    best = float('inf')
    best_triple = None
    for d1 in range(1, n//3 + 1):
        for d2 in range(d1+1, (n-d1)//2 + 1):
            d3 = n - d1 - d2
            if d3 <= d2:
                continue
            l = lcm(lcm(d1, d2), d3)
            if l < best:
                best = l
                best_triple = (d1, d2, d3)
    return best, best_triple

# Verify for c=1848374 pattern: n divisible by 5^3, 31, 53 but not 2 or 3.
# Test with n = 5^3 * 31 * 53 = 125 * 31 * 53 = 205375
n = 5**3 * 31 * 53
print(f"n = {n}")
fn, tr = f_norwegian(n)
predicted = Fraction(16, 25) * n
print(f"f({n}) = {fn}, predicted = {predicted} = {int(predicted)}")
print(f"f/n = {fn/n:.8f}, predicted ratio = {float(Fraction(16,25)):.8f}")
print(f"triple = {tr}")
print()

# That's too big for brute force. Let me try smaller n with just 5^2 | n.
n = 25 * 31  # = 775
fn, tr = f_norwegian(n)
print(f"n = {n}")
print(f"f({n}) = {fn}, f/n = {fn/n:.8f}")
print(f"triple = {tr}")
# (k1=16, k2=2): need 25|n. Yes. m = 16*775/25 = 496.
print(f"Predicted by (16,2): m = {16*775//25}, ratio = 16/25 = {16/25}")
print()

# n = 25 * 47 = 1175
n = 25 * 47
fn, tr = f_norwegian(n)
print(f"n = {n}")
print(f"f({n}) = {fn}, f/n = {fn/n:.8f}")
print(f"triple = {tr}")
# (k1=16, k2=2): need 25|n. Yes. m = 16*1175/25 = 752
print(f"Predicted by (16,2): m = {16*1175//25}")
# But also (k1=15, k2=2): need 47|n. Yes! m = 30*1175/47 = 750. ratio = 30/47 = 0.6383
print(f"Predicted by (15,2): m = {30*1175//47}, ratio = 30/47 = {30/47:.8f}")
print()

# n = 5 * 97 = 485
n = 5 * 97
fn, tr = f_norwegian(n)
print(f"n = {n}")
print(f"f({n}) = {fn}, f/n = {fn/n:.8f}")
print(f"triple = {tr}")
# (k1=64, k2=2): need 97|n. Yes. m = 128*485/194 = 128*485/194. S = 194.
S = 64*2+64+2
print(f"S = {S}")
m_pred = 64*2*485 // S
print(f"Predicted by (64,2): m = {64*2*485}/{S} = {64*2*485/S}")
# Also (k1=16,k2=2): need 25|n? 485/25 = 19.4. No, 25 does not divide 485.
# (k1=3,k2=2): need 11|n. 485/11=44.09. No.
# (k1=4,k2=2): need 7|n. 485/7=69.28. No.
# (k1=48,k2=2): need 97|n. S=3*48+2=146. 485*96/146=485*96/146. Hmm.
# Let me check what k1 works with k2=2 and 97|n:
# S = 3k1+2. Need primes of S (other than those dividing k1 or n) to be absent.
# n has primes 5 and 97. For S | n*k1: every prime of S must be 5, 97, or divide k1.
# For S | n*k2=n*2: every prime of S must be 2, 5, 97.
# But n is odd, so 2 does not divide n. S | 2n: for p | S, p != 2: v_p(S) <= v_p(n).
# And p=2: v_2(S) <= 1.
# So S must have only prime factors from {2, 5, 97} with v_2 <= 1, v_5 <= 1, v_97 <= 1.
# S = 3k1+2. S values: 2*5=10 (k1=8/3, no), 2*97=194 (k1=64), 5*97=485 (k1=161), etc.
# S=194: k1=64. ratio = 128/194 = 64/97. Check: f/n = 64/97 = 0.6598.
# S=485: k1=161. ratio = 322/485. Hmm: 322/485 = 0.6639. Worse than 64/97.
# S=970: k1=322.67... not integer.
# So best for k2=2 is k1=64, ratio = 64/97.
print()

# n = 5 * 103 = 515
n = 5 * 103
fn, tr = f_norwegian(n)
print(f"n = {n}")
print(f"f({n}) = {fn}, f/n = {fn/n:.8f}")
print(f"triple = {tr}")
# (k1=68, k2=2): S = 3*68+2=206=2*103. Need 103|n, yes. And S | n*k2=2*515: 206|1030? 1030/206=5. Yes.
# ratio = 136/206 = 68/103.
# But (k1=55, k2=2): S=167. Need 167|n? 515/167=3.08. No.
# So with just {5, 103} dividing n, best is 68/103. Wait but my earlier code found 110/167 for c=44636594.
# That's because c=44636594 has c+1 with factor 167! So 167 | n.
# Let me verify: for n divisible by 5, 103, 167, 173:
# (k1=55, k2=2): S = 167. Need 167|n. Yes! ratio = 110/167.
# (k1=68, k2=2): S = 206 = 2*103. Need 103|n and 2|n*k2=2n (always). ratio = 68/103.
# 110/167 = 0.6587, 68/103 = 0.6602. So 110/167 < 68/103. Better.
# What about (k1=57, k2=2): S=173. Need 173|n. Yes! ratio = 114/173 = 0.6590.
# 110/167 = 0.6587 < 114/173 = 0.6590. So 110/167 is better.
# What about (k1=3, k2=2) with 11? 11 does not divide n. Etc.
# So 110/167 is confirmed best for that c.

print()
# Now let me verify f(515) against prediction
# For n=515 = 5*103: best is (68,2) giving 68/103.
# m = n * 136/206 = 515 * 136/206 = 515 * 68/103 = 5*68 = 340.
# Triple: (m/68, m/2, m) = (5, 170, 340). Sum = 5+170+340 = 515. Check.
# Verify: does 5 divide 340? Yes. Does 170 divide 340? Yes (340/170=2). Does 340 divide 340? Yes.
# lcm(5, 170, 340) = lcm(lcm(5,170), 340) = lcm(170, 340) = 340.
print("Verify n=515: triple (5, 170, 340), sum =", 5+170+340, "lcm =", lcm(lcm(5,170), 340))
print()

# Also verify n = 5*97*103 = 49955 (too large for brute force, but let me check prediction)
# For n divisible by 5, 97, 103 (no 2 or 3):
# (k1=64, k2=2): need 97|n. ratio = 64/97 = 0.6598.
# (k1=68, k2=2): need 103|n. ratio = 68/103 = 0.6602.
# So best is 64/97 (since 97|n).
# m = 64/97 * n. Triple: (n/97, n*32/97, n*64/97). Let's verify n=97*5=485:
n = 485
m = 64*485//97
d1 = m // 64
d2 = m // 2
d3 = m
print(f"n={n}: m={m}, triple=({d1}, {d2}, {d3}), sum={d1+d2+d3}, lcm={lcm(lcm(d1,d2),d3)}")
print(f"f({n}) brute force = {f_norwegian(n)}")
