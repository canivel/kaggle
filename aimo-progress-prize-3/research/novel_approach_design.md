# Novel Approach Design: Breaking the 0.69 Per-Attempt Ceiling

## Current Paradigm (everyone does this)
```
Problem → 8 parallel attempts → entropy vote → answer
```
Per-attempt accuracy: p=0.69. Expected: 39.7/50. Ceiling: ~44 with luck.

## Why p=0.69?
The model fails when:
1. It sets up the wrong equations (conceptual error)
2. It picks a wrong approach (geometry when algebra is better)
3. Its code has bugs it can't fix within the turn budget
4. The problem is too hard (requires insight the model lacks)
5. It runs out of context/time before reaching an answer
6. It reaches the right answer but doesn't put it in \boxed{}

## Novel Paradigm: Adaptive Multi-Phase Solving

### Phase 1: Quick Triage (4 attempts, short budget)
- Run 4 fast attempts with 2-minute budget each
- If ≥3 agree → submit immediately (saves time for harder problems)
- If 4 different answers → this is a HARD problem → go to Phase 2

### Phase 2: Deep Solving (4 careful attempts, full budget)
- For hard problems only
- Each attempt gets the FULL remaining time budget
- Inject the Phase 1 results as context: "Previous attempts got answers X, Y, Z, W. 
  These disagree, so at least 3 are wrong. Be extra careful and verify thoroughly."
- This is NOT diverse prompts (which hurts). This is INFORMING the model about the 
  disagreement so it can be more careful.

### Phase 3: Verification Tiebreak
- If Phase 2 still disagrees, take top 2 candidates
- Run a verification call for each: "Check if X is correct"
- Submit the one that verifies

## Why This Might Work
- Easy problems (65% of them): solved in Phase 1, saving time for hard ones
- Hard problems: get 2x more time AND benefit from knowing previous attempts disagreed
- The "previous attempts disagreed" injection is NOT a prompt change — it's problem-specific 
  context that helps the model avoid the SAME wrong approach

## Implementation in Harmony Protocol
- Phase 1: 4 attempts with `deadline = start + 120` (2 min each)
- Check agreement. If unanimous → return
- Phase 2: 4 more attempts with `deadline = original_deadline`
- The user message becomes: `{problem} {preference_prompt}. Note: initial quick analysis 
  produced conflicting answers [{A}, {B}, {C}, {D}]. Be extra thorough and verify with code.`
- Phase 3: verification calls on top 2 candidates

## Risk Assessment
- The Phase 1 quick solve might be too short for some "easy" problems → loss
- The disagreement injection might confuse the model → unknown
- More complex code → more bugs → risk of regression

## Alternative: PRM Reranking
If we can fit a 7B reward model alongside GPT-OSS-120B:
- Generate 8 solutions normally
- Score each solution trace with PRM
- Pick highest-scored solution
- This changes p from 0.69 to potentially 0.80+ (documented in literature)
