# Lessons Learned - v29 Scoring Analysis

## The Mistake
We forked FORGE v16 (a REGRESSION from v10) instead of v10/v18/v28 (the 0.39 baseline).
Then we added 23 "improvements" on top of a broken base, making it worse.

## What Hurt Us (-0.19 RHAE total)
1. **Missing action dedup** (-0.10 to -0.15): v16 removed it, exploded BFS branching
2. **5-phase BFS** (-0.03 to -0.05): counter A*, ACMD, permutation, IDDFS wasted time
3. **Broken CLTI** (-0.02 to -0.03): injected all-black frames into CNN buffer
4. **CNN persistence with corrupt data** (-0.01 to -0.02)
5. **Higher exploration params** (-0.01)

## What to Do Differently
- Start from the PROVEN base (v10/v28 which scores 0.39)
- Add improvements ONE AT A TIME with validation
- Test each change locally before pushing
- Don't add complexity unless it's proven to help
- When in doubt, SIMPLIFY

## Score Trajectory
0.00 → 0.02 → 0.10 → 0.20 → ??? (v32 = CHRONOS v28 base, targeting 0.39)

## Next Steps
1. Submit v32 (CHRONOS v28 = 0.39 baseline)
2. Validate improvements individually
3. Add only proven improvements for 0.40+
4. DQN value model for 0.50 (Sergei's approach)
