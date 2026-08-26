"""
AIMO3 BREAKTHROUGH STRATEGY: FINAL ANALYSIS & IMPLEMENTATION
=============================================================

SIMULATION RESULTS SUMMARY (calibrated to match observed 39/50):

Strategy                              Mean   Delta vs Baseline
--------------------------------------------------------------
baseline (8 att, 1/entropy)          38.62   +0.00
quality_weighted (8 att)             39.80   +1.18
exec_verified (8 att)                40.28   +1.66  <-- BEST SIMPLE CHANGE
ev + adaptive code-only cascade      40.57   +1.95  <-- BEST OVERALL
oracle @ 8 attempts                  45.22   +6.60  (theoretical max)

KEY FINDINGS:

1. EXECUTION-VERIFIED VOTING (+1.66 problems, ZERO extra compute)
   - Code-verified (python ran, no errors): 10x weight
   - Code with errors: 0.1x weight
   - No code execution: 0.2x weight
   - Works because: code-using correct attempts are 78% accurate vs 49% for no-code
   - The differential is the signal: wrong answers with code have MORE errors

2. ADAPTIVE CODE-ONLY CASCADE (+0.29 more, uses idle compute)
   - After base 8 attempts, check code-verified consensus
   - If no 3+ code-verified consensus: run 16 more, ONLY count code-verified
   - If still no 5+ code-verified consensus: run 24 more, only code-verified
   - Extra attempts that don't have clean code are DISCARDED

3. STRATEGIES THAT DON'T WORK:
   - Program synthesis: -0.14 (synthesis attempts bring their own errors)
   - Temperature diversity: -0.50 (reduces clustering of CORRECT answers too)
   - Adversarial filtering: +0.13 (too risky, 5% false positive kills it)
   - Cross-problem learning: -0.10 (no signal to exploit)
   - Plain cascading (all extra): +0.35 (dilutes signal with wrong answers)
   - Answer space analysis: +0.39 (only helps 30% of problems)

4. WHY CASCADE ALONE HURTS:
   Adding more attempts adds more wrong answers. With 70% wrong-answer clustering
   and confident-wrong entropy (~1.8), extra wrong attempts cluster and gain weight.
   The FIX: only count code-verified extra attempts. This filters out ~65% of wrong
   extra attempts (wrong+code+error or wrong+no-code) while keeping ~69% of correct
   extra attempts (correct+code+no-error).

5. THE REAL CEILING:
   Oracle@8 = 45.22 (correct answer appears in at least one attempt)
   Oracle@32 = 47.95
   Gap from best (40.57) to oracle@8 (45.22) = 4.65 problems
   These 4.65 problems are lost to: correlated wrong answers with low entropy
   that outvote the correct answer. No voting scheme can fix this.
   Fixing it requires: model improvement or ensemble with different model.
"""

# ============================================================
# THE IMPLEMENTATION: Drop-in replacement for _select_answer
# ============================================================

# Replace this single method in AIMO3Solver to get +1.66 problems

SELECT_ANSWER_CODE = '''
    def _select_answer(self, detailed_results: list) -> int:
        """
        Execution-Verified Voting (EV voting).

        Key insight: Code execution is not just a tool -- it's a SIGNAL.
        When the model uses Python and the code runs without errors,
        it's strong evidence the answer is correct.

        Simulation: +1.66 problems over baseline 1/entropy voting.
        """
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result['Answer']
            entropy = result['Entropy']
            python_calls = result.get('Python Calls', 0)
            python_errors = result.get('Python Errors', 0)

            if answer is None:
                continue

            # Base weight: inverse entropy (proven signal)
            weight = 1.0 / max(entropy, 1e-9)

            # EXECUTION VERIFICATION
            has_code = python_calls > 0
            has_errors = python_errors > 0

            if has_code and not has_errors:
                weight *= 10.0    # Code verified: 10x boost
            elif has_code and has_errors:
                weight *= 0.1     # Code with errors: 0.1x penalty
            elif not has_code:
                weight *= 0.2     # No code: 0.2x penalty

            answer_weights[answer] += weight
            answer_votes[answer] += 1

        scored_answers = []

        for answer, total_weight in answer_weights.items():
            scored_answers.append({
                'answer': answer,
                'votes': answer_votes[answer],
                'score': total_weight
            })

        scored_answers.sort(key=lambda x: x['score'], reverse=True)

        vote_data = []

        for item in scored_answers:
            vote_data.append((
                item['answer'],
                item['votes'],
                item['score']
            ))

        vote_dataframe = pd.DataFrame(
            vote_data,
            columns=['Answer', 'Votes', 'Score']
        )

        vote_dataframe = vote_dataframe.round({'Score': 3})
        display(vote_dataframe)

        if not scored_answers:
            print('\\nFinal Answer: 0\\n')
            return 0

        final_answer = scored_answers[0]['answer']
        print(f'\\nFinal Answer: {final_answer}\\n')

        return final_answer
'''

print(SELECT_ANSWER_CODE)

print("=" * 80)
print("IMPLEMENTATION NOTES")
print("=" * 80)
print("""
1. MINIMUM VIABLE CHANGE (recommended for next submission):
   - Replace only _select_answer with EV voting above
   - Expected improvement: +1.66 problems (39 -> 40.7)
   - Risk: NONE. Only changes voting weights. Same attempts, same compute.

2. FULL IMPLEMENTATION (if time permits):
   - Replace _select_answer with EV voting
   - Modify solve_problem to add adaptive cascade for split votes
   - Expected improvement: +1.95 problems (39 -> 40.6)
   - Risk: LOW. Uses more compute but never changes prompts.

3. WHAT NOT TO DO:
   - Do NOT change prompts (proven harmful per v22 regression)
   - Do NOT change temperature (diversity hurts in simulation)
   - Do NOT add verification/retry logic (adds complexity, marginal benefit)
   - Do NOT add program synthesis pass (hurts in simulation)

4. THE HARD TRUTH:
   - Best achievable with voting optimization: ~41/50
   - Top private LB: 46/50
   - Gap of 5 problems requires either:
     a) A better model (not available)
     b) A fundamentally different approach to the hard problems
     c) The 46/50 team has a secret sauce we don't know about

   The 46/50 team likely uses: huikang model + VOI + context=131K +
   some form of verification cascade. Their edge is probably from
   the model + context length, not from voting innovation.
""")
