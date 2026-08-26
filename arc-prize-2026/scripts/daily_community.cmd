@echo off
REM ARC daily community brief - headless Claude session, 06:00 daily.
REM Research-only: sweeps discussions/kernels/datasets/LB, ranks top-10 finds,
REM updates the top-3 pattern doc. Consumed by ARCDailyIterate at 08:23.
cd /d F:\kaggle\arc-prize-2026
if not exist learnings\community mkdir learnings\community
type scripts\daily_community_prompt.md | claude -p --dangerously-skip-permissions --max-turns 150 >> runs\daily_community.log 2>&1
