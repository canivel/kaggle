"""
Problem 5: Swiss tournament with 2^20 runners.

Key insight: In a Swiss tournament with 2^n players:
- After round k, players are grouped by their win/loss record
  (k-bit binary string of wins/losses)
- Each group has players that won/lost the same set of rounds
- Within each group, ANY matching is possible, and faster player always wins

The final ordering is determined by the set of matchings chosen in each round.

For the recursive structure:
- After round 1, we have two groups: winners (W) and losers (L).
  The partition of 2^n players into W and L determines who won round 1.
  Within W: the top 2^(n-1) speeds in each chosen pair.

Actually, let me think about it differently.

Consider the problem recursively. In round 1, we partition 2^n players into
2^(n-1) pairs. In each pair, the faster player wins and gets 2^(n-1) points.

After round 1: winners form a group of 2^(n-1) players with score 2^(n-1),
losers form a group of 2^(n-1) players with score 0.
These two groups then independently run (n-1)-round Swiss tournaments
(since within each group, players have the same score and will be matched together).

So the structure is: choose a matching in round 1, determine winners/losers,
then recursively run tournaments in each subgroup.

The final ordering interleaves the two subgroup orderings based on scores.
Actually, since winners have scores >= 2^(n-1) and losers have scores < 2^(n-1),
ALL winners rank above ALL losers. So the final ordering is:
[winners' ordering] ++ [losers' ordering].

So N(n) depends on:
1. How many ways to partition 2^n players into winners/losers in round 1
2. N(n-1) for the winners' sub-tournament
3. N(n-1) for the losers' sub-tournament

But these aren't independent! The set of players in the winners' group depends
on the matching chosen.

In round 1, we choose a perfect matching of 2^n players. In each pair, the
faster wins. The set of winners W and losers L is determined by the matching.

Different matchings can produce different W/L sets. We need to count the number
of distinct (W ordering, L ordering) pairs, which equals the number of distinct
final orderings.

The number of possible W sets: Given 2^n players with distinct speeds, how many
subsets W of size 2^(n-1) are achievable as the winner set of some perfect matching?

For W to be a valid winner set: for each player in W, there must be a player in L
that is slower. Equivalently, we need a perfect matching between W and L such that
each W player is faster than their L partner.

This is possible iff for every k, the k-th fastest player in W is faster than
the k-th fastest player in L (by Hall's theorem / dominance condition).

Actually, by Hall's condition (or a simpler argument): W can be the winner set
iff for each w in W, there exists an l in L slower than w. Since we need a
perfect matching where each w beats their l partner, by Hall's theorem this is
possible iff for every subset S of W, |{l in L : l < max(S)}| >= |S|...
Actually, the condition simplifies:

Sort all players by speed. For W to be achievable, we need:
For each i from 1 to 2^(n-1), the i-th fastest player in W must be faster
than the i-th fastest player in L. (This is necessary and sufficient by
Hall's marriage theorem applied to the bipartite graph W-L where w->l iff w>l.)

Equivalently: if we sort all 2^n players and look at which are in W,
the condition is that W dominates L in the stochastic ordering sense.

Let players be labeled 1 (slowest) to 2^n (fastest).
W must contain 2^(n-1) players, and when we sort W and L:
w_1 < w_2 < ... < w_{2^{n-1}} (W sorted)
l_1 < l_2 < ... < l_{2^{n-1}} (L sorted)
Condition: w_i > l_i for all i.

This is equivalent to: for each position j among {1,...,2^n} sorted by speed,
the number of W-players among {1,...,j} is at most floor(j/2)... no wait.

Actually, the condition w_i > l_i for all i is equivalent to:
In the sorted sequence of all 2^n players, if we mark W and L,
the number of W-players in {1,...,j} <= number of L-players in {1,...,j}
for all j. Wait no:

w_i > l_i means the i-th smallest winner is larger than the i-th smallest loser.
Equivalently, the i-th smallest of all 2^n players is NOT in the situation where
too many winners are too small.

Formally: among the first j players (sorted by speed), the number of winners
is at most j/2 (rounded appropriately). This is equivalent to saying that in
the sorted sequence, every initial segment has at least as many L's as W's... no.

Let me think again. Consider players sorted 1,2,...,2^n.
Mark each as W or L. Let w_i, l_i be as above.
w_i > l_i for all i iff: among {1,...,k}, #{W} <= #{L} for all k.
(This is the ballot problem / Catalan number connection.)

Wait, that's not quite right either. Let me verify for small cases.

n=2: 4 players {1,2,3,4}. W has 2 players.
Possible W: {2,3}: w1=2>l1=1, w2=3<l2=4. Need w2>l2: 3>4 false. ✗
{2,4}: w1=2>l1=1, w2=4>l2=3. ✓
{3,4}: w1=3>l1=1, w2=4>l2=2. ✓
{1,4}: w1=1>l1=2? No. ✗
{1,3}: w1=1>l1=2? No. ✗
{1,2}: w1=1>l1=3? No. ✗

So valid W sets: {2,4} and {3,4}. That's 2.

Sequence encoding:
{2,4}: L=1,W=2,L=3,W=4. Prefix counts of W: 0,1,1,2. Counts of L: 1,1,2,2.
#{W in {1,...,k}} <= #{L in {1,...,k}}? 0<=1,1<=1,1<=2,2<=2. All yes. ✓

{3,4}: L=1,L=2,W=3,W=4. W counts: 0,0,1,2. L counts: 1,2,2,2.
0<=1,0<=2,1<=2,2<=2. ✓

{2,3}: L=1,W=2,W=3,L=4. W counts: 0,1,2,2. L counts: 1,1,1,2.
At k=3: 2 <= 1? No! ✗

Great, so the condition is: #{W in {1,...,k}} <= #{L in {1,...,k}} for all k=1,...,2^n.
This is equivalent to #{W in {1,...,k}} <= k/2 for all k.
This is the Ballot-problem sequence!

The number of such sequences is the Catalan number C_{2^{n-1}} = C(2^n, 2^{n-1})/(2^{n-1}+1).
Wait, not exactly Catalan. Let me think...

The number of sequences of 2^{n-1} W's and 2^{n-1} L's such that every prefix has
at least as many L's as W's is the Catalan number C_{2^{n-1}} = C(2^n, 2^{n-1})/(2^{n-1}+1).

Hmm wait, is this correct? For n=2: C_2 = 2. And we found 2 valid W sets. ✓

For n=1: 2 players. C_1 = 1. W = {2}, L = {1}. 1 valid W set. ✓

So the number of valid W sets for a given round = C_{2^{n-1}} (Catalan number).

Now, N(n) = sum over valid W-sets of N_W(n-1) * N_L(n-1),
where N_W(n-1) and N_L(n-1) are the number of orderings achievable in the
sub-tournaments for winners and losers respectively.

But N_W(n-1) depends on WHICH players are in W, not just that it's a valid set.
"""

# Let me compute N for small n to find the pattern.
from itertools import combinations

def count_orderings(n, players=None):
    """Count distinct final orderings for 2^n tournament on given players."""
    if players is None:
        players = tuple(range(1, 2**n + 1))

    if n == 0:
        return 1  # One player, one ordering

    m = 2**n
    half = m // 2

    orderings = set()

    # Try all possible winner sets
    player_list = sorted(players)

    for W in combinations(player_list, half):
        W_set = set(W)
        L = tuple(p for p in player_list if p not in W_set)
        W = tuple(sorted(W))
        L = tuple(sorted(L))

        # Check validity: w_i > l_i for all i
        valid = all(W[i] > L[i] for i in range(half))
        if not valid:
            continue

        # Recursively count orderings for W and L sub-tournaments
        # The W sub-tournament preserves relative speeds
        # The L sub-tournament preserves relative speeds
        # The final ordering is W-ordering concatenated with L-ordering

        # For the recursive call, what matters is only the relative order
        # (since faster always wins). So the sub-tournament result depends
        # only on the SIZE, not the specific players!

        # Wait, is this true? In the sub-tournament, the matching determines
        # winner/loser sets. The set of possible W sets in the sub-tournament
        # depends on the relative speeds, which are the same regardless of
        # the specific players (since speeds are distinct and we only care
        # about relative order).

        # Yes! So N(n-1) for any 2^{n-1} players is the same, regardless of
        # which specific players they are.

        # Therefore: N(n) = (number of valid W sets) * N(n-1)^2
        # Since each valid W set leads to N(n-1) * N(n-1) orderings
        # (independent choices in W and L sub-tournaments)

        # BUT WAIT: different W sets might lead to the SAME final ordering!
        # Two different W sets that produce the same W-ordering and L-ordering...
        # Actually no: different W sets mean different players in W and L,
        # so the orderings (which list players by rank) will differ.

        # Hmm, actually could two different W-sets give the same ranking?
        # W-set determines who ranks 1st through 2^{n-1} and who ranks
        # 2^{n-1}+1 through 2^n. Different W-sets means different top-half sets,
        # so different rankings. And within each half, the number of orderings
        # is N(n-1) (same for all W-sets since relative order is what matters).

        # So N(n) = (number of valid W-sets) * N(n-1)^2.
        break  # We just need the count of valid W-sets

    # Actually let me just compute the count of valid W-sets
    count_valid_W = 0
    for W in combinations(player_list, half):
        W_sorted = sorted(W)
        L_sorted = sorted(p for p in player_list if p not in set(W))
        if all(W_sorted[i] > L_sorted[i] for i in range(half)):
            count_valid_W += 1

    return count_valid_W  # * N(n-1)^2, but we compute the full N(n) separately

# Verify the formula N(n) = C_{2^{n-1}} * N(n-1)^2
from math import comb

def catalan(n):
    return comb(2*n, n) // (n + 1)

print("Catalan numbers C_k:")
for k in range(1, 10):
    print(f"  C_{k} = {catalan(k)}")

print("\nValid W-set counts:")
for n in range(1, 5):
    m = 2**n
    half = m // 2
    players = list(range(1, m+1))
    count = 0
    for W in combinations(players, half):
        W_sorted = sorted(W)
        L_sorted = sorted(p for p in players if p not in set(W))
        if all(W_sorted[i] > L_sorted[i] for i in range(half)):
            count += 1
    print(f"  n={n}: valid W-sets = {count}, C_{half} = {catalan(half)}")

# So N(n) = Product_{i=0}^{n-1} C_{2^i}^{2^{n-1-i}}
# Because: N(n) = C_{2^{n-1}} * N(n-1)^2
# N(1) = C_1 = 1
# N(2) = C_2 * N(1)^2 = 2 * 1 = 2
# N(3) = C_4 * N(2)^2 = 14 * 4 = 56
# N(4) = C_8 * N(3)^2 = 1430 * 56^2 = 1430 * 3136 = 4,484,480

print("\nN values:")
N = 1
for n in range(1, 6):
    half = 2**(n-1)
    c = catalan(half)
    N = c * N * N
    print(f"  N({n}) = {N}")

# For n=20:
# N(20) = product_{k=0}^{19} C_{2^k}^{2^{19-k}}
# log10(N(20)) = sum_{k=0}^{19} 2^{19-k} * log10(C_{2^k})

# Actually, the exponents in the recursion:
# N(n) = C_{2^{n-1}} * N(n-1)^2
# So log N(n) = log C_{2^{n-1}} + 2 * log N(n-1)
# Unrolling: log N(n) = sum_{k=1}^{n} 2^{n-k} * log C_{2^{k-1}}
#                      = sum_{j=0}^{n-1} 2^{n-1-j} * log C_{2^j}

print("\nExponents of Catalan numbers in N(n):")
for n in range(1, 6):
    print(f"  N({n}) = ", end="")
    terms = []
    for j in range(n):
        exp = 2**(n-1-j)
        terms.append(f"C_{2**j}^{exp}")
    print(" * ".join(terms))

# We need v_{10}(N(20)) = largest power of 10 dividing N(20).
# v_{10}(N) = min(v_2(N), v_5(N)).
# v_p(N(20)) = sum_{j=0}^{19} 2^{19-j} * v_p(C_{2^j})

# Recall C_m = C(2m,m)/(m+1) = (2m)! / (m! * (m+1)!)

# v_p(C_m) = v_p((2m)!) - v_p(m!) - v_p((m+1)!)

# For m = 2^j:
# v_p(C_{2^j}) = v_p((2^{j+1})!) - v_p((2^j)!) - v_p((2^j+1)!)

# Legendre's formula: v_p(n!) = sum_{i=1}^{inf} floor(n/p^i)

def legendre(n, p):
    """Compute v_p(n!)"""
    result = 0
    pk = p
    while pk <= n:
        result += n // pk
        pk *= p
    return result

def v_catalan(m, p):
    """Compute v_p(C_m) where C_m is the m-th Catalan number."""
    # C_m = (2m)! / (m! * (m+1)!)
    return legendre(2*m, p) - legendre(m, p) - legendre(m+1, p)

print("\nv_2 and v_5 of Catalan numbers:")
for j in range(21):
    m = 2**j
    v2 = v_catalan(m, 2)
    v5 = v_catalan(m, 5)
    print(f"  C_{m}: v_2={v2}, v_5={v5}")

# Compute v_2(N(20)) and v_5(N(20))
v2_total = 0
v5_total = 0
for j in range(20):
    exp = 2**(19-j)
    m = 2**j
    v2 = v_catalan(m, 2)
    v5 = v_catalan(m, 5)
    v2_total += exp * v2
    v5_total += exp * v5

print(f"\nv_2(N(20)) = {v2_total}")
print(f"v_5(N(20)) = {v5_total}")
print(f"v_10(N(20)) = k = {min(v2_total, v5_total)}")
print(f"k mod 10^5 = {min(v2_total, v5_total) % 100000}")
