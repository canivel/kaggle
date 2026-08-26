@echo off
REM Windows Task Scheduler entry point for ARC daily submit daemon.
REM Resolves PATH for kaggle CLI (installed via pipx/uv), then runs the
REM daemon in the repo root.

set REPO=F:\kaggle\arc-prize-2026
cd /d %REPO%

REM Ensure uv and kaggle CLI are reachable for the SYSTEM session.
set PATH=%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%LOCALAPPDATA%\pipx\venvs\kaggle\Scripts;%USERPROFILE%\.local\bin;%PATH%

uv run python scripts\daily_submit.py >> runs\daily_submit_stdout.log 2>&1
