# Daily Brief — 2026-07-15

## 1. Yesterday's results

### 1a. NO new LB score — war draw #2 never fired (daemon quota-day collision)
- war-v1 σ-draw #2 was queue head, but ARCDailySubmit at 22:37Z Jul 14 skipped with
  `already-submitted-today`: draw #1 had gone out at **00:13Z Jul 14** (= 20:13 EDT Jul 13),
  same UTC day. The 18:37 EDT daemon time and the 20:00 EDT window refresh straddle the UTC
  date line, so any submission made right after a window refresh (00:xxZ) makes the *next*
  evening's daemon run a guaranteed skip. **One LB window (UTC Jul 15's early hours) went unused.**
- Consequence: war-v1 ledger still n=1 ({0.91}). Today's UTC quota is FREE right now —
  draw #2 can be submitted immediately (protocol: submit as soon as validated; it is
  byte-identical kernel v1, validated). That also restores the 00:xxZ cadence and un-breaks
  the daemon for tomorrow.
- Proposed fix (code, free): daemon should compare against the *UTC submission-window day*
  it is about to submit into, not "any submission with today's UTC date" — or simply move
  the scheduled task to 20:07 EDT (00:07Z) where it historically worked for 3 months.

### 1b. `arc3-duck-war-eval` v1 COMPLETE — first heavy warpack run on Kaggle hardware
- Build rail, 25 games × 1 pass, budgets byte-comparable to null10 (war 146 actions/run,
  7923 s wallclock vs null 140 / 7931 s). Artifacts: `runs/kernel_pulls/war_eval_v1/`.
- **Screen vs null10** (validated scorer 0e+00; `runs/war_eval_v1/screen_report.md`):
  - **PRIMARY (prereg §2) paired Δlc = +0.272/game, 12W/5L/8T, exact sign-flip p = 0.0074.**
    lc totals 22 vs null 15.2. First positive primary-statistic screen of the campaign.
  - **Secondary Δlog1p(RHAE) = −0.036, p = 0.61 — flat.** RHAE run-mean 1.579 vs null 1.636.
  - Read together: warpack (recovery/retry_guard/shortcircuit, ledger OFF) clears ~45% more
    levels but converts none of it into RHAE — the extra clears are action-expensive, exactly
    the pooled-single-run tax pattern that killed sched-v1's recovered L1s. Consistent with
    LB draw #1 = 0.91 (no lift in official currency).
  - Wins concentrate where null is stuck (sc25 +1.8, m0r0 +1.0, ar25 +0.9, s5i5 +0.9, ls20 +0.7,
    tu93 +0.7, vc33 +0.6, ka59 +0.6); losses small (cn04 −0.5, ft09 −0.4, re86 −0.3).
- **Banking canary: ZERO replay events** in the whole run (banner confirms
  `banking=True … bank_strict=True`, but no `banking: replayed` lines). With 1 pass/game and
  0 wins there may be nothing to replay — vacuous, not divergent — but it means banking has
  **never once been observed executing its core mechanism** outside local smoke tests. The
  prereg §7 side-channel canary (replay_attempted/succeeded counts) is still unbuilt.
- Handoff condition check: no lc regression (opposite: lift), no observed banking divergence
  → per yesterday's handoff, war-v1 draw #3 queues behind draw #2.

### 1c. Leaderboard
- KOJIMA holds #1 at 1.86; Tecnod8.AI 1.61; trio at 1.56. **12 teams ≥ 1.44** (wall
  thickening: Lonnie/Tshithihi/Figuring-out 1.44, MLRush/Biubiu/Arunodhayan 1.46,
  hiranorm 1.48, Dinesh 1.50, paul 1.54). Our best 1.02; gap to wall 0.42 unchanged.

## 2. Today's open questions (for panel)

- **Q1 — Submit draw #2 NOW (midday) vs tonight?** Author position: now. UTC quota is free,
  the build is byte-identical validated, score returns ~8h earlier, and it un-jams the
  daemon cadence. No design change — it is the same pre-registered ledger draw.
- **Q2 — Does Δlc(+, p=0.007)/RHAE(flat) count as a positive screen worth more seeds, or a
  Goodhart trap?** Prereg §2 made Δlc primary purely on power grounds; this is the first case
  where the two statistics dissociate. Options: (a) push war-eval seed 2 today (free,
  1 push) → 3 seeds by Jul 17 for a real gate look; (b) declare lc-without-RHAE a known
  mechanism (recovery buys stuck-game L1s at full action cost) and prioritize converting
  clears to *clean* clears (retry_guard tuning / earlier banking) before more seeds.
  Author: (a) — seeds are free and the gate look needs them regardless.
- **Q3 — R2 (ledger-ON) window 1 timing.** Prereg §4 requires war-v1 ledger n≥3. If draw #2
  goes today and #3 tomorrow, first ledger-ON window is Jul 17, alternate-nightly after.
  Any reason today's screen changes the design lock (banking identical in both arms)? Author: no.
- **Q4 — Banking canary build.** Prereg §7 scheduled the replay-count side-channel for the
  next war kernel version. war-v2 (ledger flags ON) is that next version — fold the canary
  into it now so both A/B arms carry it? (Canary is observability-only; keeps arms identical
  in mechanism.) Author: yes, build today, smoke, ready for Jul 17.
- **Q5 — Daemon fix.** 1a's proposal — window-day logic or 20:07 EDT schedule. Free, no
  model risk. Author: do today.

## 3. Constraints unchanged
Zero cloud spend (reserve panel-gated); ≤2 kernel pushes/day (0 used today); fork-never-build;
gates need ≥3 seeds (screens are 1-seed, non-binding); queue never empty by 18:00; submit as
soon as validated (19:00 EDT hard stop); phase-1 line CLOSED (retry look unspent).
