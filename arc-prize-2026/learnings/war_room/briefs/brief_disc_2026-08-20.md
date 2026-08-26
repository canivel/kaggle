TASK: ARC-AGI-3 Kaggle discussion sweep for 2026-08-20.

Competition: arc-prize-2026-arc-agi-3 (Kaggle). Check the discussion feed for posts NEW since 2026-08-19.
Use WebSearch and/or fetch https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion sorted recent.

Our context (do not re-derive):
- We run a frozen fork of the Tufa Labs duck harness. LB best 1.33 (banked 07-18). Gold ~1.62, prize ~1.90.
- New #1 "cstl" entered 08-12 at 2.52, untraced.
- Our frozen-fork nightly draws are a stationary-ish band mean ~0.93 s~0.18; 08-20 drew 0.41 (z=-3.44), a new minimum.
  Working hypothesis: a MIXTURE — normal draws plus occasional PARTIAL-RUN deaths (vLLM/server dies mid-run,
  remaining games score 0, submission still completes).
- 08-18: the Kaggle rerun BATCH pool migrated its input layout; we built a pathsafe fork against that.

DELIVERABLE: for each NEW post since 08-19, one line: TITLE | ADOPT / ADAPT / IGNORE | one-sentence reason.
Prioritise anything about: (a) scoring variance / partial runs / harness deaths, (b) the input-layout migration,
(c) cstl or any top-10 method disclosure, (d) engine/model changes in public duck forks.
If NOTHING is new since 08-19, say so explicitly and do not invent posts. Do not speculate; cite post titles/authors.
Write your findings ONLY as your final message.
