# RHAE Math: What It Takes to Score > 0.1

## The Formula
- Per-level: S(l) = min(1, human/agent)^2
- Per-game: weighted average, weight = level_index (1,2,3...)
- Overall: average across ALL games (including 0s)

## Current Reality (25 public games)
- 10 games complete L1
- Best: LP85 (27 actions vs 33 human = perfect L1)
- But LP85 game RHAE = 0.028 (L1 only contributes weight=1 out of 36 total)
- Mean RHAE = 0.0017

## What We Need for 0.1
With 25 games, mean RHAE = 0.1 means total RHAE sum = 2.5

Options:
A) Solve L1 perfectly in ALL 25 games: ~25 * 0.028 = 0.7 -> mean = 0.028 (not enough)
B) Solve L1-L3 in 10 games efficiently: harder but possible
C) Solve ALL levels in 5 games: ~5 * 0.5 = 2.5 -> mean = 0.1 (THIS IS THE TARGET)

## Key Insight
Must solve DEEP (many levels) not WIDE (many games at L1).
Completing 5 games fully is better than L1 in 25 games.

## What We Need
1. CNN agent running on Kaggle (not lightweight heuristic)
2. Per-game time budget of 30+ minutes (not 3 min)
3. Focus on easy games and go deep
4. Sequential game processing (not 110 parallel)
