# GPT-5.6-sol vs Qwen3.6-27B on the grinders (sb26, lp85) — what actually transfers

Date: 2026-07-16. Read-only forensics for war-v3 scaffolding.

Sources:
- GPT-5.6: `runs/gpt56_probe/experiment_full/` (transcripts `sb26-7fbdac44_p0.txt`, `lp85-305b61c3_p0.txt`, `benchmark.json`). Same duck harness, same system prompt, 100-action cap, 1h wall.
- Qwen3.6-27B: `runs/kernel_pulls/war_eval_{v1,v2,v3}/transcripts/` (same game files). **Ledger-OFF controls** (0 hits for `HYPOTHESIS LEDGER` / `GOAL:` in all six transcripts), ~2.2h wall each.
- Priors: `sb26_mechanics.md`, `lp85_mechanics.md`, `transcript_forensics.md` (Qwen failure classes CONCEPT / MEMORY / PERCEPTION).

Headline (benchmark.json vs war_eval results):

| game | GPT-5.6 | actions used | Qwen (best of v1-v3) | actions used |
|---|---|---|---|---|
| sb26 | **5/8 levels** | 100 (cap; L6 in progress) | 1/8 | 143-257+, 2 GAME_OVERs |
| lp85 | **4/8 levels** | 100 (cap; L5 = systematic sweep) | 1/8 | 68-133, GAME_OVER loop |

GPT per-level actions beat the human baseline on most levels (sb26 `[15,30,15,15,17,8]` vs base `[18,16,15,15,31,24]`; lp85 `[7,10,19,19]` vs base `[33,22,31,23]`). Qwen burned 240+ actions failing sb26 L2 alone.

Critical negative result up front: the shared system prompt already says, verbatim, "IMPORTANT: … it is usually safer to write an explicit search algorithm such as BFS … search in the inferred state space." **GPT wrote BFS 6 times in lp85 alone (`from collections import deque`); Qwen wrote it 0 times across all six grinder transcripts.** Prompt exhortation to plan/search is empirically inert on Qwen. Anything we ship must be code or ledger machinery, not advice.

---

## sb26 pivot table (L2 = the CALL/connector level that walls Qwen)

| # | pivot | GPT-5.6 (evidence) | Qwen (evidence) |
|---|---|---|---|
| S1 | Attributing what a click does | Step 2 (action 5), after ONE batched 4-click probe, reads object-level diff: "Bottom clicks only move a white selection outline; they do not fill the center slots … the earlier direct-entry model was wrong." Two-click select+place confirmed by action 9. | v1 spends 111 actions on L1. v2 step 28: "my swap logic isn't working correctly … I'm not tracking the positions correctly after each swap" — tracks slot contents in prose, mis-predicts swap outcomes repeatedly, has to re-derive slot coords from ascii mid-level. |
| S2 | First failed submit → hypothesis surgery | Step 9: "SPACE rejected the arrangement, so simple row-major order is wrong … routed/snake reading order." Step 11 (2nd fail): "Both row-major orientations fail. **Treating the green T as a blocked cell yields a seven-cell U-shaped adjacency graph with two endpoints**" → enumerates continuous paths, solves on 4th submit, 30 actions total for L2. | v2 steps 13-43: ~10 arrangements over ~240 actions, ALL from the same family, all verbal color-set heuristics: "warm/cool split", "primary/secondary", "hue angle order", "match the border colors", "maybe click the arrow". Step 21: "Let me think about this differently" — then proposes another arrangement variant. Never converts the puzzle into an enumerable path/graph space. |
| S3 | Reading the connector | Step 7: one small ascii crop (rows 16-42 × cols 16-48) → "The green structure is not a fixed sequence tile; it is a **connector** … linking to the lower panel," and the slot count is re-derived (3 upper + 4 lower = 7). | v2 step 11 SEES it: "The green arrow seems to indicate that the bottom green box feeds into the top red box." Perception was NOT the gap in this seed — Qwen holds the connector observation for 2 hours and only ever derives "flow" vibes, never execution-order semantics. Divergence is concept conversion, not vision. |
| S4 | Level-to-level abstraction reuse | L3 first try in one 15-action batch: "Read the red panel left-to-right, **recursively traversing a child panel when its connector is encountered**." L4: "movable connector token"; L5: "the same blue child panel referenced twice". One abstraction, upgraded per level. | Never reaches L3 in any seed. After GAME_OVER reset (v2 step 29), restarts the same arrangement family from scratch. |
| S5 | Interface vs gameplay evidence | Step 3: "ACTION7 was not executed — the API rejected its name despite advertising it as valid … This is an **interface/action-alias issue, not gameplay evidence**." Never mutates the world model on API errors. | v2 step 40: "What is ACTION7? I haven't tried that" (2h in, having already gotten `Unknown action` for it at step 22) — re-probes a known-broken action and briefly treats the failure as a game clue. |

## lp85 pivot table (L2 = crossings/step-budget level)

| # | pivot | GPT-5.6 (evidence) | Qwen (evidence) |
|---|---|---|---|
| L1 | Budget read on first contact | Step 2: "the top HUD lost one unit, **confirming an eight-click budget** … seven counterclockwise rotations … exactly matches the remaining budget." Plans inside the budget from turn 2. | Budget never identified in any seed. v3 step 12 post-mortem: "The game ended because I ran out of steps **or some other condition**." Dies to the 60-click budget, auto-resets, dies again the same way (GAME_OVERs at actions 68 and 131-133). v1 invents a false model: "clicking the progress bar submits". |
| L2 | Canonical state extraction | Samples ordered tile centers into strings: "cycle pre `SbpYpbSgwgpwbgppSSgYbwwbg` / post `pSbp…`" → exact +1 rotation verified. All later reasoning runs on this canonical list, not on re-reads. | Compares ad-hoc color lists per turn; re-derives layout repeatedly; whole-loop strings with no saliency ("I don't see an obvious pattern", 3× in v3 steps 11-13). |
| L3 | Impossibility → mechanism revision | Step 4: "A single vertical rotation cannot fill both sockets because the yellow tiles are **16 positions apart while sockets are 3 apart**. The crossing loops must therefore be used independently." → models crossings as shared/aliased cells (`T=['T0','T1','V23',…]`), BFS over 2-token state → "shortest 5 ['L+','V+','V+','V+','L-']" → L2 cleared, score 2. This is exactly the "crossings transfer tiles" key fact from lp85_mechanics.md, derived from 2 probes. | v3 step 9 SEES the ingredient: "the loops share tiles, so the rotations are not independent" — and treats it as a nuisance, not the mechanism. Strategy stays "rotate each loop a full cycle and check level_completed": 21+8+9 clicks that provably return to start. Never identifies goal tiles or brackets; clicks the yellow markers once, gets no-op, drops them. |
| L4 | Goal-token saliency | Tracks the unique Y/O tokens among shuffling filler: "Ordinary colors are filler." Reduces state to 2 token positions + fixed socket overlays. | Tracks 20-26-cell full color strings; the two color-11 goal tiles are never distinguished from the six shuffling filler colors (matches lp85_mechanics diff-noise prediction). |
| L5 | Plan fails → fit model to history | L4 step 13: BFS plan fails once → "compare the eight actual transitions against candidate vertical module orders … identify the exact permutation" → `itertools.permutations` scoring over recorded before/after frames → correct order → L4 cleared. | On plan-mismatch, Qwen re-clicks and re-observes from scratch; history is only ever consulted to re-find the initial state, never to fit a model. |
| L6 | Graceful degradation | L5, goal relation ambiguous → "the reachable puzzle state is only (top mod 5, snake mod 16), totaling **80 configurations** … enumerate systematically, stopping immediately on completion." Bounded sweep with early-stop. | Unbounded re-rotation of the same 1-D subspace until the budget kills it. |

---

## Classification and ranked candidates

Legend: (a) prompt line, (b) ledger heuristic (extends `duck_eval/ledger/ledger_core.py`), (c) scripted assist, (d) not distillable. Ranked by expected transfer to Qwen, applying the simplicity-wins prior hard. All items are game-agnostic; no game-id branching.

### Ranked (a)/(b)/(c) list

1. **(c) Auto probe-diff summarizer** — after every `action(...)`, the harness appends a fixed ~120-token block to the tool result: changed-cell count + bbox, color-transition `Counter`, moved-object correspondences (hash-matched), and an `unchanged_regions`/`HUD-only` flag. This is precisely the diff discipline GPT hand-rolled every single turn and the thing Qwen demonstrably cannot sustain (pivots S1, L2: "not tracking the positions"). Pure information add, no behavior forced, so regression risk is low; Qwen already consumes structured tool output well. Cap the block hard so it can't blow the 1024-token tool budget.
2. **(c)+(b) Budget sentinel** — scripted detector: an edge-adjacent bar/counter object that shrinks or increments monotonically with actions ⇒ write `FACT: budget-like bar at <bbox>, ~N units, K consumed` into the ledger (persists across GAME_OVER, which is exactly when Qwen needs it) and inject one line each turn: `BUDGET: ~M actions remaining`. On <25% remaining, one-shot trigger: "stop exploratory clicking; act only on your best confirmed plan." Every Qwen grinder death was a budget death it never saw coming (pivot L1); GPT read the budget on turn 2. The system prompt already warns about HUD bars, so the delta is *persistence + arithmetic*, not awareness.
3. **(b)+(c) Submission-fingerprint refutation** — at every reward-check action (submit-like action or any action taken with all slots filled), hash the canonical interactive-region object layout (sorted `(color, bbox)` of small movable objects from segmentation). Store the hash as a refuted-arrangement FACT on failure; if a new submission's hash matches a stored one, prepend "IDENTICAL to refuted arrangement #k — do not submit; change family." Qwen re-submitted literally identical arrangements after window eviction (v2 steps 13→20 both try red=first-3/green=last-4). Piggybacks on the existing `Ledger`/escalation machinery; the fingerprint feeds `refuted_count` so the existing N=3 family escalation fires on *fact* rather than on prose extraction.
4. **(a) Interface-error hygiene line** — one sentence: "If an action result contains `error` or `executed: False`, it is a harness/interface problem: retry a different encoding once, and do NOT update the game world model or re-probe it later as a game mechanic." Trivial cost, directly patches pivot S5. Qwen can execute this.
5. **(b) Escalation-template addition (reachability arithmetic)** — extend `ESCALATION_PROMPT_TEMPLATE` with: "Before proposing another variant, compute with Python whether the current goal is arithmetically reachable (object spacings vs target spacings, parity, counts). If unreachable, the *mechanism* model is wrong, not the arrangement." This is the pivot-L3 move (16-apart vs 3-apart) in template form. One sentence in an existing one-shot prompt; worst case it is ignored, as prompts often are — but it fires only at escalation time, when Qwen is already being interrupted, which is the one context where injected instructions have shown traction.
6. **(c) Saliency helper** — preloaded `salient(frame)` returning: objects whose color appears ≤2× (candidate tokens/targets), objects unchanged across the last K frames while neighbors changed (candidate sockets/overlays), and never-changing edge strips (HUD). Attacks pivot L4. Medium confidence: helper output is only as good as Qwen's willingness to look at it; ship together with #1 as one combined "auto-annotation" block, not as a separate API Qwen must remember to call.
7. **(c) Permutation-cycle extractor + BFS planner library** — `infer_permutation(before, after)` → cycle decomposition; `bfs_tokens(cycles_by_button, token_positions, targets)` → click sequence. Highest ceiling (this is the entire lp85 L2-L4 win) and lowest certainty: Qwen must still map buttons→permutations and choose to call the API, and the 30s tool timeout bounds state size. Given that Qwen ignored the BFS instruction ~50 turns per run, treat as A/B-only, never a default. If shipped, make it a single mega-call (`auto_plan(history)`) that does button-effect inference from recorded transitions itself — anything requiring Qwen to assemble inputs will not be used correctly.

Do NOT ship: more prose strategy advice in the system prompt (BFS instruction proven inert; the prompt is already at the edge of the 31,744-token context budget), per-turn "think harder" nags, or anything keyed on a game id (illegal).

### NOT distillable (do not chase with prompts)

- **One-shot recursive abstraction** (sb26 L3-L6: "recursive target expansion", "same child referenced twice" read off a single segmentation dump, each level solved in one batch). Raw capability.
- **Inventing the aliased-cell representation** for crossing tracks (lp85 L2 `T=[…,'V23',…]`). A helper can compute cycle decompositions, but *choosing* to model crossings as shared cells given "loops share tiles" is the reasoning step Qwen missed while holding the same observation.
- **Model-fitting over recorded transitions** (lp85 L4: scoring 24 module-order×direction candidates against history frames). Requires composing hypothesis space + scorer on the fly.
- **Terse load-bearing world models** (GPT: ~500 content chars/turn, 0 reasoning chars, every sentence a commitment; Qwen: kilobytes of THINKING with weak conclusions under the identical world-model contract). Style transfer via prompt does not survive contact.
- **In-head enumeration convergence** (sb26 L2 solved in 4 submits with no enumeration code — the ordering families were enumerated mentally and pruned by two bits of feedback).

---

## Top 3 transferable deltas for war-v3

1. **Auto probe-diff summary appended to every action result** (candidate #1) — the substrate of every GPT-5.6 win is per-action diff attribution, which Qwen provably cannot maintain by hand; injecting it costs tokens, not behavior.
2. **Budget sentinel: scripted shrinking-bar detector + persistent ledger FACT + low-budget one-shot trigger** (candidate #2) — every Qwen grinder run ended as an unseen budget death; GPT planned inside the budget from turn 2, and this is detectable game-agnostically.
3. **Submission-fingerprint refutation in the ledger** (candidate #3) — turns Qwen's verbatim re-submissions into hard blocks and gives the existing N=3 family escalation a reliable, prose-free trigger signal.

Each is information-only or fires at most once per condition, keeping regression risk within the simplicity-wins prior; the BFS helper library (#7) is the only high-ceiling item and should go behind its own A/B, not into the default rail.
