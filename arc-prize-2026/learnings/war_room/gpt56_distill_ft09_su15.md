# GPT-5.6-sol probe distillation: ft09 (what transfers) vs su15 (what doesn't)

Sources (all local, read-only):
- GPT: `runs/gpt56_probe/experiment_full/transcripts/{ft09-0d8bbf25,su15-4c352900}_p0.txt`, `benchmark.json`, `run_config.json`
- Qwen (same tool-agent harness, same system prompt): `runs/kernel_pulls/war_eval_v{1,2,3}/transcripts/ft09-0d8bbf25_p0.txt`, `su15-1944f8ab_p0.txt`
- Ground truth: `learnings/war_room/su15_mechanics.md` (engine-verified 18-click L2 policy), `transcript_forensics.md`, `duck_eval/ledger/ledger_core.py`

Headline numbers (benchmark.json): ft09 GPT = 5/6 levels, 100 actions, per-level 8/17, 7/19, 14/15, 16/21, 45/65, 10/26, score 71.4, 16.1k generated tokens. su15 GPT = 1/9 levels, **54 actions total** (21 on L1, 33 on L2), cancelled at the 60-min cap, score 1.63. Qwen ft09: v1 = L1 at action 135 (of ~136), v2 = reached L3 region late, v3 = 0 levels. Qwen su15: v1/v2 = 1 level, v3 = 0.

---

## PART A — ft09: the concept GPT-5.6 grasps immediately

### The concept

ft09 is a flip-tiles game where each 3×3 panel of large clickable tiles carries a **miniature 3×3 legend at its own center**: each mini cell is white exactly where the corresponding large tile must equal the legend's center color, gray where it must be a different palette color. The example panels are **self-labeling** — every unframed panel already satisfies its own legend, so the mapping can be read off the board with zero environment actions. Clicking a large tile cycles its color through the level palette.

GPT-5.6's first tool call dumps segmentation; its second reads the mini and outer cells of all four panels; its first world model (before any click) is already the right concept class: *"the miniature likely encodes a rule derived from each panel's surrounding colors... infer that rule from the three examples, then click the required answer cell(s)."* It wastes exactly 4 actions (clicking mini cells; `board_changed: False` falsifies "minis are editable" in one shot) and clears L1 in 8.

### How it encodes + reuses the rule across levels

1. **Explicit rule slot in the carried world model.** The harness re-injects the previous turn's world-model text; GPT dedicates a line to it under the offered `Cross-level notes:` / "Cross-level rule" prefix and keeps it *abstract*: "white = same color as the clue center, gray = the other color", not "click tiles (44,20),(46,20)...". Each level it restates the rule as a prior, then checks it against the new layout before acting.
2. **Monotone revision under contradiction.** L3: "white means same as center" (generalizes over red- and orange-centered clues). L4: "gray means *not* the center color, disambiguated by overlaps; clicks cycle blue→red→orange". L5: nested editable panels, "dark gray = clipped/absent positions". L6: fixed magenta markers = decorative. Each revision is the minimal edit that explains all evidence; refuted variants are never retried.
3. **Free (offline) verification via overlap consistency.** Whenever puzzles share tiles (L2–L5), GPT checks that the merged constraint system is consistent *before* spending a single click — "their shared-row requirements agree, confirming the mapping". This is verification against information already on the board, at zero action cost.
4. **Predict-then-audit after acting.** After every batch it checks `board_changed` and, when a level fails to complete, recomputes desired-vs-current color for every tile ("isolate mismatches rather than making further speculative clicks") — L5's 45-action recovery from three wrong sub-hypotheses is entirely driven by these audits.

### Why Qwen burns 135 actions on L1 (war_eval_v1)

Qwen sees the same panels and same segmentation but frames the puzzle with an ARC-1 prior: *"find the transformation between the three source grids and apply it to the target."* It then enumerates: copy-nearest, majority vote, mirror, XOR mask, UL⊕UR, UL∧UR, OR, binary-sum mod 256, all-blue, all-red, 4-corner symmetric, 2×2 blocks, all eight 1-red patterns, top row, left column... ~30 hypotheses, **each falsified only by spending 1–8 environment actions and watching `level_completed` stay false**. It triggers GAME_OVER twice. It clicks known-dead targets repeatedly (source grids ×3, gray corners ×3, yellow HUD bar). Critically, it *never once* tests a candidate rule against the example panels' own miniatures — every one of its ~30 guesses could have been refuted offline in a single python call, because the examples are self-labeling. It finally wins L1 by luck (left-column pattern) and — because it learned a *pattern*, not a *rule* — starts L2 from scratch ("Try clicking to understand the mechanics").

The delta is not perception (both read the same cells), not action-format, not harness memory (same world-model carry is available to both). It is: **hypothesis space selection (legend/correspondence vs transformation) + where verification happens (on-board data vs environment actions).**

### Transferable deltas, ranked

**1. PROMPT — "verify offline against on-board examples before spending actions" (top delta).**
Game-agnostic prompt lines, roughly: *"Boards often contain a legend/key: small decorated or miniature regions that encode the goal state of the interactive region (examples that already satisfy the rule, a swatch of the target object, an ordered color strip). Before testing any goal hypothesis with environment actions, check it in python against every such region already on the board — a correct rule must explain all examples. Reject offline-falsified hypotheses for free. After acting, compute the exact frame you expect; if the actual frame differs, the hypothesis is refuted — do NOT retry the same action family without a new rule that explains the mismatch."*
This is precisely the behavior gap and it applies to at least 4 of 5 probe games (ft09 clue minis, su15 key swatch + upgrade strip, sb26 top-row order, lp85 target row). Honest expected transfer: **medium**. It attacks the single biggest per-draw action sink (30 env-tested guesses → in the best case ~4–8), but Qwen must still *execute* a correspondence search when prompted, and its L1 transcript shows it can extract sub-grids but defaults to inter-grid transformations; prompt A/B history (feedback_prompt_is_noise) says expect noise unless the instruction changes what code gets written, so gate on Δlevels_completed. Worst case it cuts wasted actions without unlocking the concept — that alone matters on budgeted games (Qwen's two ft09 GAME_OVERs were exhaustion-driven).

**2. LEDGER-HEURISTIC — predicted-frame check wired into the FACT/RESULT contract.**
`ledger_core.py` already persists HYPOTHESIS/FACT records outside the message window, auto-refutes superseded same-family variants, and fires a one-shot 4-family escalation after 3 refutations. Two additions, both game-agnostic:
   - **PREDICT gate:** extend the prompt contract so a GOAL: line carrying an action plan must include the predicted post-action observation ("PREDICT: cells X,Y turn purple / level completes"); the harness diff engine scores it and writes the RESULT: refuted/confirmed record *mechanically* instead of trusting the model's self-report. Qwen's ft09 run had a perfectly reliable refutation signal (`level_completed=False` after each full pattern) that it processed correctly but slowly; the mechanical record removes re-tries (it re-tested source-grid clicks and corner clicks 3× each) and survives GAME_OVER restarts — exactly the wipe that cost sb26 its 120 refuted permutations.
   - **No-effect FACTs auto-injected by the harness:** whenever `board_changed=False` or only the HUD row changed, inject "FACT: clicking region R has no gameplay effect" from the diff engine (no model cooperation required). Cheap, verified, and it removes the dead-target re-probing (ACTION7 re-tested 8–20×/run per forensics).
Expected transfer: **modest but reliable** — this is budget preservation and restart-amnesia repair, not concept discovery. It converts Qwen's ~30-guess grind into a ~30-guess grind *without repeats and without budget suicide*; on its own it probably doesn't clear ft09 L2+, because the rule-induction step is still missing.

**3. SCRIPTED-ASSIST — a generic example-consistency utility (borderline; rank low).**
A harness-side helper that, given segmentation, detects repeated panel structure (k congruent regions, one visually framed/distinct) and reports per-panel (inner-region, outer-region) cell correspondences. This would hand Qwen the ft09 legend on a plate and is expressible without game ids, but it is one bad generalization away from game-specific overfitting (su15's "panels" are nothing like ft09's), and it competes with the same engineering budget as the ledger. Only worth building if delta #1 shows signal but Qwen fails the extraction step specifically.

**4. NOT-DISTILLABLE.**
One-shot correspondence induction from 3 examples; keeping a 20+-constraint system with provenance in working memory (L5 recovery); choosing the *minimal* rule edit under contradiction. These are raw model-tier capabilities — no prompt line produced them in 13 Qwen forensics runs, and the L5 transcript segment (45 actions, 5 sub-hypothesis revisions, zero repeated mistakes) has no scaffolding analogue. Do not budget for them.

---

## PART B — su15: failure taxonomy

### What actually happened in the GPT run (corrects the premise)

The run is **not** a 60-minute search that exhausted a frontier model. Timeline from the transcript:

- **13:13–13:14 (21 actions):** L1 solved cleanly — waypoint-path model, click successive path cells.
- **13:14–13:16 (33 actions):** L2. Hypotheses in order, each discriminating, each falsified in ≤3 actions, no same-family retry loop: (1) yellow 4×4 is the piece → wrong, it's the header target swatch (1 action); (2) click the blue destination → HUD-only change (1 action); (3) click a seed → **productive**: two seeds merged into a 2×2 magenta, and GPT immediately induced the upgrade chain S→M→purple→yellow *from the header strip* plus a proximity constraint; (4) distant magentas don't auto-merge (2 actions); (5) clicking a piece drags it ~1 cell toward its partner → adopted as the movement model. It then executed 1-px-hop drags (MOUSE(53,48),(52,47),(51,46)...) and merged magenta pairs into purple. **GAME_OVER at action 54** — the L2 step budget (32) ran out mid-drag; the final 14-action batch was still mid-plan.
- **13:16–14:12 (0 actions, 284 harness turns):** the deadlock. The system prompt commands "if an action result reports game_over ... stop acting immediately"; the environment auto-reset the board but kept `state=GAME_OVER` until the next action; `valid_actions` still listed MOUSE/ACTION7. GPT concluded "no action is permitted in a terminal run; a new session is required" and spent ~56 minutes re-verifying the flag in 184 inspection-only python calls. It even computed the correct restart move each turn and guarded it behind `if terminal: print('No action executed')` — the plan was sitting in an unreachable `elif`. 87% of the run's visible output text (51k of 59k chars) was produced in this loop; the 103k output tokens are deadlock liturgy + hidden reasoning, not search. From 13:47 the local proxy's upstream quota died (251× HTTP 429 insufficient_quota), so the final ~25 minutes could not have recovered regardless. The 60-min cap then cancelled the run.
- **Tool limits:** the 30s python cap never fired and no tool result was ever truncated by the 1024-token cap. Neither bit.

### Hypothesis space: GPT vs Qwen vs the scripted policy

GPT searched *systematically* (5 mechanically distinct hypotheses in 33 actions, zero repeats) and got the mechanics half right: pairs of same-tier pieces merge and upgrade along the header chain, movement happens by clicking, there's a range limit. Qwen (war_eval v1, 90+ actions on L2; forensics seed1, ~117) also found seed→magenta merges and even bracketed the range limit ("distance 14 too far"), but oscillated between two position-based goal families (dots-to-marked-positions, match-the-indicator) for the entire budget, proved neither, and re-tested ACTION7 constantly. Same game, different failure shapes: GPT = correct-ish physics + budget death + deadlock; Qwen = correct-ish physics + wrong goal family forever.

What the engine-verified 18-click policy (`su15_mechanics.md`) knows that **both** models missed:

1. **Click = vacuum-to-point, radius 8.** The click location is a *destination*: everything within 8px is pulled TO it. So a fruit moves up to ~7px per click, and a midpoint click merges a pair in ONE action. Both models induced "click nudges a piece 1 cell" from adjacent clicks and paid ~7× the necessary step cost — this alone makes the 32-step budget unbeatable (1-px hops need 40+ clicks for the L2 merge tree vs the verified 18).
2. **Exactly-two same-tier merge; 3+ catch = one fruit = irreversible mass loss**, signaled only by the goal/key sprites turning gray (color 9→2). Invisible unless you decode a 1px-scale color change nobody is looking at.
3. **ACTION7 = undo** (restores positions/levels, un-grays, doesn't refund steps). GPT never pressed it on L2; Qwen pressed it a dozen times *as a "submit" button* and refuted "no effect".
4. **Mixed-tier overlap = fault: escalating −2/−4/−6 step penalty + silent rollback** — reads as "nothing happened, timer dropped", actively teaching the wrong lesson while draining the budget.
5. **Goal = exact count: one L3 fruit whose center lands inside the 9×9 ring** (`goal=[3,1]`). Neither model ever reached the stage where this binds.

Items 2/4/5 are the killers: they are near-zero-observability at 64×64 with 1px fruits, and the environment punishes the probes that would reveal them by draining the same 32-step budget you need to win. su15 L2's information cost exceeds its action budget unless you already know the physics — that is an information-theoretic wall, not a model-tier wall, and GPT-5.6 clearing the other four probe games while stalling here is consistent with it.

### Could ANY scaffolding bridge it?

- **Game-over-continuation (mandatory, and it's a harness bug, not a game fix):** replace/augment the "stop acting immediately" rule with *"GAME_OVER is not terminal for the run: the board auto-resets; on the next turn, re-ground and keep playing; your ledger/world model persists."* Game-agnostic, zero risk, and it converts GPT's 56 wasted minutes into ~5 fresh L2 attempts. Qwen already behaves this way naturally (it played through two ft09 game-overs), so this delta mostly protects rule-following models — but our own duck prompt contains similar stop language, and the ledger's whole design (persists across GAME_OVER restarts, `observe_game_over` FACT) presumes continuation. Ship it as prompt hygiene regardless of su15.
- **Range-probe prompt line** (*"when a click moves an object, measure the action's reach: test a far click before paying per-step costs on short hops"*): game-agnostic, directly attacks miss #1, and both models' transcripts show they had the evidence to exploit it. This is the only scaffold with a plausible path to su15 L2 for a frontier-tier model: vacuum reach + merge chain + persistence across resets ≈ the verified policy minus the mass-loss trap. For Qwen tier it likely just produces faster budget death, because the goal-family lock (forensics: "never generates a third family") remains.
- **Ledger:** preserves refuted families across the resets that continuation enables — necessary companion, not sufficient.
- **Not bridgeable by any shippable scaffold:** the 3-way-merge/gray-goal/undo semantics. Discovering them requires either decoding 1px sprite recolors (perception ceiling for both tiers) or sacrificial probe runs the budget doesn't grant. Our knowledge of them comes from reading the engine source — anything that injects them is game-id branching by definition.

### Verdict

su15 is an **accept-the-loss game for Qwen-tier models**: both its failure layers (goal-family lock + hidden fault semantics) sit below our scaffolding waterline, and the expected Δlevels from any legal intervention is ~0 since every tier already banks L1. For frontier-tier models it is *not* proven resistant — the GPT run spent 4 minutes playing and 56 minutes in a fixable game-over deadlock atop a dead upstream quota; with continuation + a reach-probe line, L2 is plausibly grindable. Ship the game-over-continuation fix and the reach-probe line as general hygiene (they're free and protect all games), but spend zero su15-specific effort and expect zero su15 gain in the gate statistic.

---

## Final summary

**Part A top 2 deltas:** (1) prompt: *verify hypotheses offline against on-board legends/examples before spending actions, and predict the exact next frame before repeating any action family* — the single behavior separating GPT's 8-action ft09 L1 from Qwen's 135-action lucky grind; (2) ledger: *mechanical PREDICT→RESULT wiring plus harness-injected no-effect FACTs* — converts refutation from a self-reported, window-evicted event into a persistent verified record, eliminating retries and restart amnesia. Expected transfer: medium and modest-but-reliable respectively; both game-agnostic; gate on Δlevels_completed per prereg.

**Part B verdict:** su15's L2 defeats both tiers for different reasons — Qwen locks into position-goal families it never escapes; GPT induces half the physics quickly but pays 7× step cost under a 1-px-drag misconception, exhausts the 32-step budget, and then loses 56 of 60 minutes to a game_over-compliance deadlock (plus a dead upstream quota) rather than to the puzzle. The mechanics that the verified 18-click policy relies on (vacuum-to-point reach, exactly-two merge with irreversible 3-way mass loss, undo, escalating silent fault penalties, exact-count ring goal) are near-unobservable within the level's own budget, making su15 an information-theoretic wall: accept the loss for Qwen-tier, fix the game-over-continuation prompt bug for everyone, and treat frontier-tier su15 as unproven rather than impossible.
