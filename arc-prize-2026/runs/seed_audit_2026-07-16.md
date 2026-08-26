# Seed audit — unseeded-randomness sweep (discussion #726552 ADAPT)

Date: 2026-07-16. Trigger: Kaggle discussion 726552 (byte-identical submissions scored
0.20 vs 0.03; author's cause was an *unseeded* random ACTION6 fallback policy).

Sweep: `grep -rniE "import random|np\.random|random\.(choice|randint|shuffle|sample|seed)"`
over `duck_eval/` + `notebooks/duckwar/` (checkpoints/pycache excluded).

Findings (3 hits, all seeded):
1. `duck_eval/taaf_bundle/src/ARC3-Inference/inference/tools/significance.py:308` —
   `rng = random.Random(seed)` (explicit seed param). SEEDED.
2. `duck_eval/taaf_bundle/src/tufa-arc-agi-framework/src/taaf/diagnostics.py:1429` —
   `np.random.default_rng(abs(hash(observed)) & 0xFFFFFFFF)` (content-derived seed;
   diagnostics only, not action selection). SEEDED/deterministic-per-input.
3. `duck_eval/taaf_bundle/src/tufa-arc-agi-framework/src/taaf/solver_examples.py` —
   Random baseline agent uses `random.Random(self.seed + i)` per game. SEEDED. Not part
   of the duck/war submission path anyway.

Fallback-path check: warpack's failure fallback is **vanilla duck** (LLM policy), not a
random policy (`warpack_patch.py:7,81,135`). No unseeded random action selection exists
anywhere in the submission path.

Residual nondeterminism: vLLM sampling (temperature > 0) — by design, present identically
in all arms, and is the draw-generating mechanism the ledgers are built to average over.

**Verdict: PASS — #726552's failure mode is absent. Our LB draws are seeded-agent draws
in the relevant sense (no policy-level RNG noise beyond LLM sampling).**
