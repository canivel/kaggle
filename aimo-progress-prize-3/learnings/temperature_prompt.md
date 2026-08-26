# Learning: Temperature 0.5 + Structured Prompt = 44/50

## Temperature
- 43/50 base: temperature=1.0
- Both 44/50 notebooks: temperature=0.5
- Cross-validated independently by two different teams

## System Prompt Structure
The 44/50 prompt follows UNDERSTAND→EXPLORE→PLAN→EXECUTE→VERIFY:
1. UNDERSTAND: Read, rephrase, identify given/wanted/constraints
2. EXPLORE: Multiple strategies, relevant theorems, don't commit early
3. PLAN: Select approach, outline steps
4. EXECUTE: Methodical, show all steps
5. VERIFY: Substitute back, edge cases, alternative methods

Additional sections:
- Mathematical Reasoning Principles (symmetry, patterns, extremes, work backwards)
- Verification Requirements (cross-check, constraints, simple cases)
- Output Format (\\boxed{42} example)
- "Quality of reasoning is as important as the final answer"

## Tool Prompt
- Lists 5 use cases (calculations, verification, conjectures, visualization, brute force)
- "Code should support your mathematical reasoning, not replace it"
- "Explain what you're computing and why before running code"

## Preference Prompt
- Categorized by library (sympy, numpy, math)
- Best practices: "derive symbolically, verify numerically"

## Impact
+1 point (43→44) from these text-only changes. No architecture changes needed.
