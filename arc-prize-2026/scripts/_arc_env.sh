#!/bin/bash
# Shared environment for the ARC launchd jobs (macOS port of the .cmd wrappers).
# launchd hands jobs a minimal PATH, so every binary is resolved ABSOLUTELY here.
# Sourced by daily_iterate.sh / daily_community.sh / morning_check.sh /
# run_daily_submit.sh. Keep this the single place that knows about paths.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
cd "$REPO" || exit 1

mkdir -p "$REPO/runs"

# --- uv -------------------------------------------------------------------
for c in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -x "$c" ] && UV="$c" && break
done

# --- kaggle CLI (2.0.x: pushes + `kernels output`) ------------------------
# daily_submit.py finds it via shutil.which, so it must be on PATH.
# 2.2.x lives in its own venv at ~/.venvs/kaggle22/bin/kaggle (competitions
# topics, kernels logs) -- deliberately NOT on PATH, so it can't shadow 2.0.x.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export KAGGLE22="$HOME/.venvs/kaggle22/bin/kaggle"

# --- secrets from .env (gitignored; never commit) -------------------------
# Auth is the KGAT-format KAGGLE_API_TOKEN. The classic KAGGLE_USERNAME +
# KAGGLE_KEY pair does NOT work with this token (verified: 401), so do not
# "restore" a ~/.kaggle/kaggle.json in its place.
if [ -f "$REPO/.env" ]; then
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        key=${line%%=*}
        val=${line#*=}
        val=${val%$'\r'}
        val=${val%\"}; val=${val#\"}
        val=${val%\'}; val=${val#\'}
        export "$key=$val"
    done < <(cat "$REPO/.env"; echo)   # trailing echo: .env may lack a final newline
fi

# --- claude CLI -----------------------------------------------------------
# Prefer the standalone install. The VS Code extension's bundled binary is a
# version-pinned fallback ONLY -- it moves on every extension update, so a job
# that depends on it will silently die. Install the standalone CLI.
CLAUDE=""
for c in "$HOME/.local/bin/claude" /opt/homebrew/bin/claude /usr/local/bin/claude; do
    [ -x "$c" ] && CLAUDE="$c" && break
done
if [ -z "$CLAUDE" ]; then
    CLAUDE="$(ls -1d "$HOME"/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1)"
fi

# --- bench token (gitignored; recreate by hand, never commit) -------------
[ -f "$REPO/scripts/bench_token.sh" ] && . "$REPO/scripts/bench_token.sh"

export UV CLAUDE
