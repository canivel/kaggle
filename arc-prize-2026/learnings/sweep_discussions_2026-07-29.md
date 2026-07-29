# Discussions sweep — 2026-07-29 (ARC-AGI-3)

**Method:** chrome-devtools MCP (killed stale profile lock via PowerShell PID-tree + lockfile removal, then navigated). Discussion list sorted by Recent Comments. Read only threads whose last activity is NEWER than the 2026-07-28 sweep (learnings/daily_brief_2026-07-28.md §5). WebFetch failed (JS-rendered), so used page snapshots + evaluate_script(innerText) for comment bodies.

**Threads checked (activity since 07-28):**

1. **#730225 — "Active Neuro-Symbolic Search Engine via MDL for Interactive ARC-AGI-3"** (Hayford Kofi Quaye, posted 1d ago, **NEW thread**). Self-promotional writeup with self-assigned "5/5" criterion scores and an MDL/Kolmogorov-complexity framing over the 7-action space. "Empirical" table claims all 25 public envs (ar25…wa30) = SUCCESS, but every row is exactly **3 execution steps in 0.02–0.11s** — i.e. it fires RESET + a couple actions and declares success; no level actually solved, no LB score. Community downvoted to **-5**, 0 comments.
   **Verdict: IGNORE** — no real solve, no throughput/scoring signal; a "SUCCESS" table that is 3 trivial actions per game is exactly the low-ΣN trap we already know.

2. **#729985 — "Three clarifications on final scoring mechanics"** (thread covered 07-28; **2 NEW comments** 17h/14h ago). Author accepted host answers. **Hendrik Nowak (14h ago)** resolved the 9h-vs-12h ambiguity: the "<12 hours" line lives on arcprize.org/policy and applies to the **Verified Leaderboard only**, NOT the Kaggle prize competition. Host (Greg) already confirmed **v3 = 9 hours** for scored runs (also reconfirms: private scored at original run time, no rerun; each run plays BOTH datasets, 50% public tasks shown on LB).
   **Verdict: ADOPT (confirmation)** — locks our planning assumption: 9h wall-clock is the real budget, no headroom to 12h. Reinforces the A17/v6 throughput crisis framing (9h is all we get).

3. **#717133 — Tufa Labs' Milestone-1 writeup** (covered 07-28; 1 NEW comment). Mustang Liu, 1d ago: "Great job! thx~". No content.
   **Verdict: IGNORE** — courtesy comment, nothing new.

4. **#713634 (pinned) — "Clarification on deadline for milestone prizes"** (covered 07-28; 1 NEW comment). KostasMouratidis, 1d ago: participant re-stating that open-sourcing must land on the deadline day (e.g. Sept 30) or the day after. No host reply, no new fact.
   **Verdict: IGNORE** — deadline chatter, already known (Sept 30 milestone 2, public LB decides both milestones).

**Nothing new on:** 8–9h window model serving mechanics, vLLM on RTX PRO 6000, large-model throughput in kernels, action-loop stalls, or scoring >1.33 public forks. No new high-interest activity.

**Net one-liner:** Only genuinely new item since 07-28 is a downvoted self-promo notebook (#730225, IGNORE — its "25/25 SUCCESS" is 3 trivial actions/game, the low-ΣN mirage) plus a host-thread confirmation that the scored-run wall-clock is firmly 9h (the "<12h" is Verified-LB-only). No throughput/serving/vLLM/high-score-fork intel. Zero impact on the A17 72B build priority.
