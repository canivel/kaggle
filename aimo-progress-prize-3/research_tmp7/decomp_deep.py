import math

def p_majority(p, N):
    threshold = (N + 1) // 2
    total = 0
    for k in range(threshold, N+1):
        binom = math.comb(N, k) * p**k * (1-p)**(N-k)
        total += binom
    return total

# Decomposition sensitivity analysis
# Key question: is the +0.15 per-attempt accuracy boost realistic?
print("=== DECOMPOSITION SENSITIVITY ANALYSIS ===")
print()
print("P(majority correct) at different (p, N) combinations:")
print(f"{'p':>6} {'N=6':>8} {'N=8':>8} {'N=10':>8} {'N=12':>8}")
for p in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    print(f"{p:6.2f} {p_majority(p, 6):8.4f} {p_majority(p, 8):8.4f} {p_majority(p, 10):8.4f} {p_majority(p, 12):8.4f}")

print()
print("Key insight: at p=0.25, even N=12 gives only 5.4% majority correct.")
print("Any boost that pushes p above 0.40 is transformative.")
print()

# But is a +0.15 boost on hard problems realistic from decomposition alone?
# For an untrained model (GPT-OSS-120B not trained for decomposition):
# Literature says Intern-S1-MO got huge gains BUT with RL-trained model
# GPT-OSS-120B: the decomposition prompt adds information (sub-problems) but
# the model may not decompose well.
#
# More realistic scenarios:
for boost in [0.05, 0.10, 0.15, 0.20]:
    p_new = 0.25 + boost
    for n in [6, 8, 10, 12]:
        gain = 5 * (p_majority(p_new, n) - p_majority(0.25, 12))
        print(f"boost={boost:.2f}, N={n:2d}: p_maj={p_majority(p_new, n):.4f}, "
              f"gain on 5 hard probs: {gain:+.3f}")
    print()

print()
print("=== THE DECOMPOSITION TRAP ===")
print()
print("The math LOOKS great: even a small boost in p is huge when p is low.")
print("BUT the decomposition boost estimate of +0.15 is SPECULATIVE.")
print()
print("Reality check from literature:")
print("1. Intern-S1-MO: +11/35 over vanilla => +31% per problem (but RL-trained)")
print("2. GPT-OSS-120B is NOT trained for hierarchical decomposition")
print("3. Decomposition adds a failure mode: wrong decomposition")
print("4. Sub-problems in math are often NOT independent (coupling)")
print("5. The combination step is itself a hard reasoning task")
print()
print("Realistic per-attempt boost on untrained model: +0.05 to +0.10")
print("Not +0.15 to +0.20 as assumed above.")
print()

# With realistic boost of +0.05 to +0.10
for boost in [0.05, 0.08, 0.10]:
    # N reduced due to 2x time cost per attempt
    for n in [6, 8]:
        p_new = 0.25 + boost
        gain = 5 * (p_majority(p_new, n) - p_majority(0.25, 12))
        print(f"Realistic: boost={boost:.2f}, N={n}: net gain on hard = {gain:+.3f}")

print()
print("=== ARCHITECTURAL CONCERN ===")
print()
print("Current _process_attempt is STREAMING with parallel workers:")
print("  - 12 attempts run in parallel via ThreadPoolExecutor")
print("  - Each streams tokens and executes code in a loop")
print("  - Early stopping when 4+ agree")
print()
print("Decomposition requires SEQUENTIAL turns within each attempt:")
print("  Turn 1: Ask for sub-problems (1 API call)")
print("  Parse sub-problems")
print("  Turn 2: Solve each sub-problem (1 API call with tool use)")
print("  Turn 3: Combine (1 API call)")
print()
print("This triples the number of serial API calls per attempt.")
print("With streaming + Jupyter sandboxing, this could take 3x as long.")
print("At 300s per problem, 12 attempts currently take ~300s (parallel).")
print("With decomposition: each attempt takes ~90-150s (3 serial turns).")
print("Even with 16 workers, 12 decomposed attempts take ~600-900s.")
print("This DOES NOT FIT in the 300s per-problem budget.")
print()
print("CONCLUSION: Decomposition requires cutting N to 4-6 for hard problems,")
print("or only applying it as a FALLBACK after initial attempts fail.")
