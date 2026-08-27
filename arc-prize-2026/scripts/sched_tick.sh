#!/bin/bash
# TICK SCHEDULER -- the ARC nightly rail on macOS.
#
# WHY THIS EXISTS (verified on this machine 2026-08-27, controlled probes):
#   * launchd StartCalendarInterval  -> DOES NOT FIRE. A bare probe job due at
#     11:24 was still at runs=0 twenty minutes later, on AC, machine awake,
#     UserEventAgent-Aqua running, nothing disabled. Real jobs fired ~3h late
#     or not at all.
#   * launchd StartInterval          -> FIRES ON TIME, every time.
#   * crontab                        -> writes hang (>120s timeout, never lands).
#   * launchctl kickstart            -> always works.
# So: schedule with StartInterval (the mechanism that works) and do the
# calendar logic here, in the shell.
#
# HOW IT WORKS
#   com.arc.tick runs this every TICK_SECONDS. For each target below, if the
#   local time is at/after the target and today's stamp for that target is
#   absent, it stamps and runs the job exactly once.
#
#   This is STRICTLY MORE ROBUST than a calendar trigger for a laptop: if the
#   machine is asleep, off, or busy at the exact minute, the job still runs at
#   the next tick instead of being silently skipped. Every downstream job is
#   already idempotent (daemon per UTC day; brief/iterate per day-file), so a
#   late or repeated tick cannot double-submit.
#
# CATCH-UP WINDOW
#   A target is only honoured within CATCHUP_MIN of its time, so a machine that
#   was off all day does not fire a stale 06:00 job at 23:00. The submit daemon
#   is exempt: its windows are anchored to the 20:00 EDT UTC-day boundary and a
#   late submission is better than none, but it self-skips if the UTC day is
#   already covered.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMPS="$REPO/runs/.sched_stamps"
LOG="$REPO/runs/sched_tick.log"
mkdir -p "$STAMPS" "$REPO/runs"

TODAY="$(date +%Y-%m-%d)"
NOW_MIN=$(( 10#$(date +%H) * 60 + 10#$(date +%M) ))
CATCHUP_MIN="${ARC_SCHED_CATCHUP_MIN:-180}"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

# name | HH:MM | script | catch-up minutes (0 = no limit)
TARGETS=(
    "dailycommunity|06:00|daily_community.sh|180"
    "morningcheck|06:00|morning_check.sh|180"
    "dailyiterate|08:23|daily_iterate.sh|240"
    "dailysubmit_1|18:37|run_daily_submit.sh|0"
    "dailysubmit_2|20:07|run_daily_submit.sh|0"
)

for entry in "${TARGETS[@]}"; do
    IFS='|' read -r name hhmm script catchup <<< "$entry"
    target_min=$(( 10#${hhmm%%:*} * 60 + 10#${hhmm##*:} ))
    stamp="$STAMPS/${TODAY}_${name}"

    # not due yet
    [ "$NOW_MIN" -lt "$target_min" ] && continue
    # already ran today
    [ -f "$stamp" ] && continue
    # too late to be meaningful (0 = always catch up)
    if [ "$catchup" -ne 0 ] && [ $(( NOW_MIN - target_min )) -gt "$catchup" ]; then
        if [ ! -f "${stamp}.skipped" ]; then
            : > "${stamp}.skipped"
            log "SKIP $name (due $hhmm, now $(date +%H:%M), past ${catchup}m catch-up)"
        fi
        continue
    fi

    # claim the slot BEFORE running, so a long job cannot be double-started by
    # the next tick.
    : > "$stamp"
    log "RUN  $name (due $hhmm, started $(date +%H:%M))"
    if [ -x "$REPO/scripts/$script" ]; then
        "$REPO/scripts/$script"
        log "DONE $name rc=$? at $(date +%H:%M)"
    else
        log "FAIL $name -- $REPO/scripts/$script not executable"
    fi
done

# keep the stamp dir from growing without bound
find "$STAMPS" -type f -mtime +14 -delete 2>/dev/null || true
