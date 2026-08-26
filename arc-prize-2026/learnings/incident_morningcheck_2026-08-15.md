# INCIDENT — `ARCMorningCheck` did not run, 2026-08-15

## Impact (bounded, and it did not bite today)

The 06:00 morning check produces the `### <date>` heading in `ITERATION_LOG.md`.
`scripts/daily_submit.py:208` gates the 18:37 submission on exactly that string:

```python
if f"### {today}" not in log_text:
    log({"skip": "audit-trail-gate-blocked", ...})
```

No heading ⇒ **the nightly submission does not fire**. This is the same failure that
blocked the 2026-07-20 20:07 fire. Today the heading was written manually at 10:15 EDT,
so tonight's 18:37 fire is **not at risk**. `runs/morning_check.log` still ends at 08-13.

## Confirmed root cause

**The machine was powered off across the trigger.**

- `LastBootUpTime` = **2026-08-15 10:07:46** — the host was down at 06:00, so the
  `CalendarTrigger` (StartBoundary `2026-08-05T06:00:00`, daily) could not fire.
- `WakeToRun` is **false**, and in any case wake-to-run does not recover a full shutdown.
- `StartWhenAvailable` is **true**, so both ARC tasks attempted catch-up at boot:
  `ARCMorningCheck` and `ARCDailyIterate` share `LastRunTime` **10:12:13**.

## Unexplained residue — and we cannot diagnose it

The catch-up attempt **did not run the check**. `LastTaskResult` = **2147946720**
(`0x800710E0`), a refusal, not a script error. `ARCDailyIterate` caught up at the same
instant and *did* start (`0x00041301`, "task is currently running" — that is this session).

Why one caught up and the other was refused is **not recoverable from the host**:

```
> wevtutil gl Microsoft-Windows-TaskScheduler/Operational
enabled: false
```

**The Task Scheduler operational log is disabled.** There is no event-level record of the
refusal and there will be none for the next one either. This is a plain instance of the
campaign's own standing lesson (`feedback_audit_the_instrument`): the scheduler is an
instrument we depend on daily and it has been running with its recorder switched off.

## Actions

1. **OWED — needs elevation (could not be done from this session):** enable the log so the
   next refusal is diagnosable. One line in an **admin** shell:
   ```
   wevtutil sl Microsoft-Windows-TaskScheduler/Operational /e:true
   ```
   Attempted here; `Access is denied` (exit 5). Until this is run, any repeat is a
   dead end again.
2. **NOT DONE, deliberately.** `DisallowStartIfOnBatteries` / `StopIfGoingOnBatteries` are
   both `true` on this task and are the usual suspects for `0x800710E0` — but
   `Win32_Battery` returns **no instances** (desktop, no UPS visible), so battery refusal is
   **not supported by evidence**. Changing them now would be tuning config on a guess with
   the log still off. Left alone; revisit once the log exists.
3. **PANEL AGENDA (Sunday 08-16) — design question, not a bug.** The audit gate blocks
   *every* queue entry, including the **trusted-fork frozen filler**, which is the eternal
   fallback and requires no review by construction. On a day where the host boots after
   18:37, we lose a filler draw — and the streak — for want of a heading. Ask: should the
   gate exempt `preflight_mode: trusted-fork`, or is the coupling the point? Do **not**
   unilaterally weaken a safety gate to protect a streak.

## Standing note

Two producers write the stub: this task at 06:00 and STEP 0 of the daily loop
(`ARCDailyIterate`, 08:23). Today the second one covered for the first, which is the design
working. The residual exposure is only the case where **both** are prevented — i.e. the host
is down all morning — and that case is environmental, not fixable in software.
