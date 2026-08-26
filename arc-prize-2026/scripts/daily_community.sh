#!/bin/bash
# ARC daily community brief - headless Claude session, 06:00 daily.
# Research-only: sweeps discussions/kernels/datasets/LB, ranks top-10 finds,
# updates the top-3 pattern doc. Consumed by daily_iterate at 08:23.
# macOS port of daily_community.cmd (launchd label: com.arc.dailycommunity).
. "$(dirname "$0")/_arc_env.sh"
mkdir -p "$REPO/learnings/community"
"$CLAUDE" -p --dangerously-skip-permissions --max-turns 150 \
    < "$REPO/scripts/daily_community_prompt.md" >> "$REPO/runs/daily_community.log" 2>&1
