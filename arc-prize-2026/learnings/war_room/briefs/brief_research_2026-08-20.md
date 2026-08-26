TASK: ARC-AGI-3 research sweep for 2026-08-20 (papers/results published or trending in the last ~7 days).

Fields to sweep: LLM agents on interactive/game benchmarks; ARC-AGI-3 specifically; test-time training/learning
for agents; agentic harness design (memory, context, replay); exploration strategies for sparse-reward games.

Our context (do not re-derive; these are SETTLED and must not be re-proposed):
- REJECTED already: "let the model notice it's repeating" (RedundancyBench 24.88%); JEPA on ARC-AGI-3 (3 strikes,
  always ERRORs on Kaggle); animation/probe-B rate arms (don't survive a real agent); capping MCTS budget for
  long BFS; prompt A/B iteration (noise); ARChitect-style grid transduction (ARC-AGI-3 has no demo pairs).
- OPEN QUESTION worth evidence: our agent FORGETS. 31,744-token context in use, though the model's
  max_position_embeddings is 262,144 — the ceiling is OURS, self-imposed. The harness exposes a `transitions`
  history the agent never queries. A prior mechanism (mech-C) achieved 96.3% delivery of history into context
  with NO behaviour change — so: is forgetting REFUTED, or is this DELIVERY-WITHOUT-USE?
- Constraint: ZERO cloud spend. Anything adopted must run as a free Kaggle kernel build/rerun.

DELIVERABLE: 5-10 items max, each: PAPER/RESULT (title + arXiv id or URL + date) | ADOPT / ADAPT / IGNORE |
one-sentence reason tied to OUR constraints. Flag hard any paper that speaks to the DELIVERY-WITHOUT-USE
question (i.e. why in-context history fails to change agent behaviour) — that is the highest-value hit today.
Be honest if the week was thin. Do not pad. Do not cite papers you did not verify exist.
Write your findings ONLY as your final message.
