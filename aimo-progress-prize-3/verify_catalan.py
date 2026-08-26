"""
Verify that the number of valid winner sets W for 2m players
where W has m elements and there exists a matching between W and complement
with each w < l (w faster) equals the m-th Catalan number.
"""
from itertools import combinations
from math import comb

def has_valid_matching(W, L):
    """Check if there's a matching where each w_i < l_i (sorted)."""
    W_sorted = sorted(W)
    L_sorted = sorted(L)
    # By a theorem, valid matching exists iff w_i <= l_i for all i (when both sorted ascending)
    # Actually: w_i < l_i since all players are distinct
    return all(w < l for w, l in zip(W_sorted, L_sorted))

for m in range(1, 8):
    n = 2 * m
    players = list(range(1, n + 1))
    count = 0
    for W in combinations(players, m):
        L = [p for p in players if p not in W]
        if has_valid_matching(list(W), L):
            count += 1
    catalan = comb(2*m, m) // (m + 1)
    print(f"m={m}: valid W sets = {count}, Catalan C_{m} = {catalan}, match = {count == catalan}")
