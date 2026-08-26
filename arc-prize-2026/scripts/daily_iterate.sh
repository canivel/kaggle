#!/bin/bash
# ARC daily iteration loop - headless Claude session each morning.
# Reads overnight results, executes the plan's next step, refills the queue.
# macOS port of daily_iterate.cmd (launchd label: com.arc.dailyiterate).
. "$(dirname "$0")/_arc_env.sh"
"$CLAUDE" -p --dangerously-skip-permissions --max-turns 250 \
    < "$REPO/scripts/daily_iterate_prompt.md" >> "$REPO/runs/daily_iterate.log" 2>&1
