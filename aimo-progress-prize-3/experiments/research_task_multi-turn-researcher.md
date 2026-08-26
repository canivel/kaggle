# Research Task: multi-turn-researcher

You are researching multi-turn prompting strategies for math problem solving.

CONTEXT:
- Current: single-turn — give problem, get answer, done
- Follow-up prompt exists for no-boxed cases: ask for boxed answer at T=0.0
- Self-refinement: PROVEN HARMFUL (-4 pts in Pawan Mali experiments)
- But multi-turn with NEW INFORMATION (not self-refinement) might help

KEY QUESTION: Can we design a multi-turn strategy that adds value without self-refinement?

SPECIFIC IDEAS TO RESEARCH:
1. "Verification turn": After attempt produces answer X, ask "Verify that X satisfies [problem constraints]. If not, what's the correct answer?"
   - Different from self-refinement because it gives a SPECIFIC claim to verify
2. "Approach suggestion turn": If first attempt fails (no code, no answer), provide a hint: "Try using modular arithmetic" or "Consider the generating function approach"
3. "Cross-attempt synthesis": After 4 attempts with split votes, show the model the conflicting approaches and ask it to adjudicate
4. "Decomposition turn": For complex problems, first ask "What are the key sub-problems?", then solve each sub-problem separately

CRITICAL CONSTRAINT: any multi-turn strategy must NOT be self-refinement (asking "are you sure?" always hurts).
The key is providing NEW information in the follow-up that the model didn't have before.

Write analysis to /multi_turn_research.md and code to /multi_turn_solver.py
