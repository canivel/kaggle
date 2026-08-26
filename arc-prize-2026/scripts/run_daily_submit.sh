#!/bin/bash
# launchd entry point for the ARC daily submit daemon (com.arc.dailysubmit).
# Fires 18:37 and 20:07 LOCAL time. The daemon is idempotent per UTC day, so a
# double fire -- or an overlap with the still-live Windows box -- submits at
# most once. macOS port of run_daily_submit.cmd.
. "$(dirname "$0")/_arc_env.sh"

# Use the prebuilt interpreter, NOT `uv run`: uv re-locks on every invocation,
# which needs the network and dies on any unresolvable path dep (the ../kaos
# dev-group entry is empty until that repo is cloned). A nightly rail must not
# depend on dependency resolution succeeding at fire time. daily_submit.py and
# preflight.py are stdlib-only, so the venv interpreter is sufficient.
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$("$UV" python find 2>/dev/null || echo python3)"

"$PY" "$REPO/scripts/daily_submit.py" \
    >> "$REPO/runs/daily_submit_stdout.log" 2>&1
