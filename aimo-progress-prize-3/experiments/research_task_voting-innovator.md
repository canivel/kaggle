# Research Task: voting-innovator

You are researching novel voting/answer-selection strategies for math competitions.

CONTEXT:
- Current: simple majority vote with 1/entropy weighting
- 8 attempts per problem, each produces an integer answer + entropy score
- The CORRECT answer is often in the candidate set but gets outvoted
- Pawan Mali: complex entropy weighting = 0 improvement
- Amanatar: binary verify cascade for top candidates = 44/50

KEY QUESTION: Is there a voting strategy that extracts more signal from 8 attempts?

RESEARCH TASKS:
1. Analyze: if p=0.69 per attempt and N=8, what's P(correct answer appears at least once)? P(majority vote correct)?
2. Research "weighted majority" schemes: logprob-weighted, code-execution-weighted, answer-confidence-weighted
3. Research "best-of-N with verification": instead of voting, verify top-K candidates independently
4. Can we use the MODEL ITSELF to break ties? ("Given answers A=5, B=7, which is more likely correct for this problem?")
5. What about answer clustering by value proximity? (if 5 attempts say 42 and 3 say 43, the 42s are likely right even at lower entropy)

Write analysis to /voting_research.md and code to /novel_voter.py
