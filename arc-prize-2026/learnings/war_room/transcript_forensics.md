# Transcript forensics: sb26 / su15 / lp85 level-2 grinds

Source: `runs/phase1_ab/seed1/transcripts/*_p0.txt` (THINKING blocks extracted), cross-checked vs `runs/null10/seed10*/benchmark.json` action histories. No `war_room/*_mechanics.md` files exist; ground-truth mechanics inferred. None of the 3 games clears level 2 in any of 13 runs, so no stated hypothesis was ever confirmed.

## sb26 (92% of tokens on L2; seed1: L1 in 16 acts, L2 = 158 acts, GAME_OVER at 140, restart, wall)

**Stated goal:** from step ~15 on, locked to one family: "fill the 7 slots (3 red-container + 4 green-box, skipping the charcoal marker) with the 7 top-row colors in left-to-right order." Fills all slots by action 36; level doesn't complete; spends the remaining ~120 actions cycling ~30 "what if the goal is…" reformulations — every one an *arrangement/ordering* variant (reverse order, warm/cool split, match container border color, move red into the hole). Never proposes a different mechanic class.

**Correct mechanic stated?** No. The structural hint — the light-green stem/arrow making slot-3 + arrow + green box ONE connected object (visible in segmentation from frame 1) — is first noticed at action ~172, 6 min before wall, and never used. gap_forensics already flags sb26 as animation-blind; the 6-frame placement animations are summarized but never mined.

**Loops:** re-clicks the same slot coords 16–32× per null seed; SPACE re-tested 8–20×/run despite establishing in L1 that SPACE only decrements the timer; hits the move-limit GAME_OVER at 140 having never budgeted moves; after ACTION7 restart, re-runs the same fill-in-order plan — the 120 refuted permutations are gone from context.

**Class: CONCEPT** (wrong goal model, single hypothesis family) + MEMORY secondary (re-tests refuted actions, restart amnesia) + PERCEPTION contributor (arrow connectivity, animations).

## su15 (86%; seed1: L1 in 25 acts, L2 = ~117 acts to wall)

**Stated goal:** oscillates between exactly two for 2 hours: (a) "move all magentas into the blob" (direct L1 transfer), (b) "match the 6-color top bar." Mechanics are learned *correctly and quantitatively*: sky-blue dot click → 2×2 magenta; magenta click → moves up-left on fixed diagonal d=r−c; blob click → shifts contained square down-right (Δd=−2). It then **proves both goals geometrically impossible with its own arithmetic** ("d=19 → col=1−19=−18, off board") — and keeps them anyway. Never generates a third family. Ends by converting all 6 dots as a shrug; no completion.

**Correct mechanic stated?** Action physics: yes, fully. Goal: never.

**Loops:** near-verbatim repeated paragraphs ("The blob moves the magenta square inside it down-right by (+1,+1)…" ≥3× across steps 55–58) — direct signature of the 14-message history window recycling. Null seeds show the *least* motor repetition (uniq/acts up to 119/169): su15 explores broadly; the failure is purely in the head.

**Class: CONCEPT** (right physics, no goal model; disproves own hypotheses but cannot replace them) + MEMORY secondary (verbatim hypothesis recycling).

## lp85 (65%; seed1: only 41 actions in 2h10m, 188k tokens — most of any game)

**Stated goal:** L1 was won *by accident* ("Level 1 completed after 5 clicks on the red shape" — cause never understood), so the L1→L2 transfer is a **wrong lesson** ("click red repeatedly"), worse than amnesia. Null seeds confirm: uniq/acts 9/22…12/108, x=20 mashed 64/99 times (seed105) — control-clicking without reading the resulting shift. In seed1 the failure mode is the opposite pole: analysis paralysis — 1.5–2.5 min/turn, and the final 75 minutes are 8+ consecutive analysis-only turns at action 41, re-deriving the same 3-row layout from ascii each turn, with sentences duplicated *inside single THINKING blocks*.

**Correct mechanic stated?** Closest of the three: at 15:56 it states the plausibly-correct model — 3 shiftable rows with R/N shift controls, target sequence Y,g,S,S,p at rows 44–45, goal = shift rows so intersection blocks match target — then executes zero actions on it before wall.

**Class: PERCEPTION** primary (cannot stably parse the multi-row scene; re-parses from scratch every turn) + MEMORY strong secondary (re-derivation loop, false L1 causal story). Concept was nearly solved.

## L1→L2 transition & amnesia

The transition itself is not the leak — L1 strategies transfer as (sometimes harmful) priors. The killer amnesia is *within* level 2: the 14-message window evicts refuted hypotheses, which then re-enter verbatim (su15, lp85), and evicts action-effect facts (sb26 SPACE=timer), which get re-probed. sb26's mid-level GAME_OVER restart is a second, total wipe.

## Cross-cutting fix implications

1. Persistent per-game **hypothesis ledger** (proposed / refuted / evidence) injected each turn — directly attacks all three MEMORY loops.
2. **Goal-family enumeration prompt** when a fully-executed hypothesis fails N times (sb26/su15 never left family #1).
3. Surface **connectivity/containment deltas** (sb26 arrow) and force a plan-execution step after any stated goal (lp85 paralysis).
