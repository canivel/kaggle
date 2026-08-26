@echo off
cd /d F:\kaggle\arc-prize-2026
type scripts\morning_check_prompt.md | claude -p --dangerously-skip-permissions --max-turns 40 >> runs\morning_check.log 2>&1
