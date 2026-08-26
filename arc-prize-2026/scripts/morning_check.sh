#!/bin/bash
# ARC morning check - headless Claude session, 06:00 daily.
# macOS port of morning_check.cmd (launchd label: com.arc.morningcheck).
. "$(dirname "$0")/_arc_env.sh"
"$CLAUDE" -p --dangerously-skip-permissions --max-turns 40 \
    < "$REPO/scripts/morning_check_prompt.md" >> "$REPO/runs/morning_check.log" 2>&1
