@echo off
REM ARC daily iteration loop - headless Claude session each morning.
REM Reads overnight results, executes the plan's next step, refills the queue.
cd /d F:\kaggle\arc-prize-2026
type scripts\daily_iterate_prompt.md | claude -p --dangerously-skip-permissions --max-turns 250 >> runs\daily_iterate.log 2>&1
