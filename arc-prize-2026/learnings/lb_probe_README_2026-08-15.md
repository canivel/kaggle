# LB probe — full-board archiver + differ (2026-08-15)

Implements **probe #1** of `learnings/top6_evidence_audit_2026-08-15.md` §5: *archive the FULL
leaderboard daily with `SubmissionCount` and diff it; read Tufa Labs and Jack Cole as the control arm.*

No spend, no GPU, no Kaggle push, no submission. Read-only against Kaggle; writes only to this repo.

| Artifact | Path |
|---|---|
| Collector | `scripts/lb_archive.py` |
| Differ | `scripts/lb_diff.py` |
| Daily archive | `runs/lb_daily/lb_full_<date>.csv` (2331 rows) |
| Heartbeat | `runs/lb_daily/heartbeat/lb_archive_<date>.json` |
| Coverage index | `runs/lb_daily/archive_index.json` |
| Wired into | `scripts/morning_check_prompt.md` steps 6–8 |

---

## §1 — Scope of claim (read this before quoting anything downstream)

**This instrument measures `Score`, `SubmissionCount`, `Rank`, `TeamName` and `LastSubmissionDate`.
Those are the only things it may claim.** It does not observe method, model, engine, prompt or
compute, and no amount of movement in it licenses a statement about any of those.

If method is surfaced anywhere downstream it carries an evidence class, unblurred:
**DISCLOSED** (they said so, linkable) / **INFERRED** (derived, not a statement by them) /
**UNKNOWN** (no public trace — the expected answer, and a complete one).
Today's tally across the top eight is **0 DISCLOSED / 1 INFERRED / 7 UNKNOWN**.

---

## §2 — Which CLI form, and why (both tested 2026-08-15)

| Form | Columns returned | Rows | Verdict |
|---|---|---|---|
| `kaggle competitions leaderboard <c> --show -v` | `teamId, teamName, submissionDate, score` | 20/page, paginated | **Insufficient.** No `SubmissionCount`, no `Rank`, no team members. Its `--page-token` windows are **non-contiguous** (defect already logged in `runs/lb_ground_truth.md` — 425 rows with a hole straight through our own tie block). |
| `kaggle competitions leaderboard <c> -d -p <dir>` | `Rank, TeamId, TeamName, LastSubmissionDate, Score, SubmissionCount, TeamMemberUserNames` | **all 2331** | **Use this.** The only form that carries `SubmissionCount`. |

`SubmissionCount` **is obtainable.** It was in the download all along and had never been used — it
is the single most informative unused column we hold, because it is what makes best-of-N inflation
measurable per team instead of only board-wide.

Implementation notes worth keeping:

- The zip member name embeds the **server-side pull time in UTC**:
  `arc-prize-2026-arc-agi-3-publicleaderboard-2026-08-15T14:50:39.csv`. The collector parses it into
  `pull_utc` in the heartbeat. This is better provenance than local mtime and is the timestamp any
  timing claim should quote.
- That member name contains **colons**, which are illegal in Windows filenames, so `zf.extract()`
  raises `OSError: [Errno 22]`. The collector reads the member and writes a sanitised name instead.
- `kaggle==2.0.0` has **no `--force`** on this subcommand; the collector downloads into a fresh
  temp dir each run so there is never a stale zip to skip.
- The downloaded CSV has a **UTF-8 BOM**; the archive written to the repo is BOM-free. Always read
  archives with `encoding="utf-8-sig"`.

---

## §3 — Field semantics, VERIFIED (not trusted from the field name)

The whole reason this probe exists is that a field name misled the campaign. Every field relied on
below was checked against a source of truth.

| Field | Verified meaning | How verified |
|---|---|---|
| `Score` | The team's **BEST (maximum)** public score over all their submissions. Not their latest. | First-party, §3.1 below. |
| `LastSubmissionDate` | The team's **MOST RECENT** submission timestamp, **whatever it scored**. **It does not date the scoring submission and cannot.** | First-party, §3.1 below. |
| `SubmissionCount` | Count of the team's submissions. Monotone non-decreasing; observed to increment by exactly +1 across a 23-minute window on a team that submitted once in it (`Ilakk manoharan` 53 → 54). | Two live pulls 23 min apart, §3.2. |
| `Rank` | Dense rank over the full board, **ties share a rank**. 2331 rows, max rank 2328. | Row/rank count mismatch on today's board. |
| `TeamId` | **Stable identity. This is the join key.** | Team 15564282 renamed *Stepwise* → *Sankalp* inside the same 23-minute window with `TeamId` unchanged. `TeamName` is not a key. |
| `TeamMemberUserNames` | Comma-separated Kaggle usernames. Used for watchlist matching (`jcole75`), which survives a team rename. | Matched all 11 watchlist entries on today's board. |

### §3.1 — The `LastSubmissionDate` ≠ `Score` defect, demonstrated on our own team

Our leaderboard row today:

```
119,15503635,Canivel,"2026-08-15 00:07:11",1.33,111,canivel
```

Our own submission history (`kaggle competitions submissions -v`, first-party):

```
2026-08-15 00:07:11  COMPLETE  public=0.89   AUTO-REFILL frozen-fork filler
2026-08-14 00:07:11  COMPLETE  public=0.70
2026-08-13 00:07:10  COMPLETE  public=0.78
...
2026-07-18 00:07:11  COMPLETE  public=1.33   <- the submission that produced our displayed Score
```

**Our `LastSubmissionDate` is 2026-08-15 and our `Score` was banked 2026-07-18 — 28 days apart.**
The submission named by `LastSubmissionDate` scored **0.89**, not 1.33.

So: *"team X's last submission is after time T, therefore X's score was achieved after T"* is
**unsound**, and the leaderboard exposes no field that repairs it. Any argument of that shape —
including the Qwen3.8-release timing argument — must be re-derived or dropped. This is an exact,
first-party counter-example, not an analogy.

*(Caveat on the verification route: `competitions submissions` is page-capped at 50 rows — we hold
111 — and only ever returns **your own** team. It cannot be used to date anyone else's scoring run.)*

### §3.2 — Two pulls, 23 minutes apart: the differ's first real output

`runs/lb_daily/intraday/lb_full_2026-08-15T1427Z.csv` vs `lb_full_2026-08-15.csv` (14:50:39Z).
2331 rows both sides; exactly **two** lines differed:

```
1428,15564282,Stepwise,"2026-04-08 04:01:10",0.20,3,sankalp
1428,15564282,Sankalp ,"2026-04-08 04:01:10",0.20,3,sankalp     <- rename, TeamId stable

2147,15988786,"Ilakk manoharan","2026-08-14 14:45:21",0.03,53,...
2147,15988786,"Ilakk manoharan","2026-08-15 14:12:46",0.03,54,...  <- +1 sub, Δscore 0.00
```

That second row is the instrument working: **Δsubs +1, Δscore +0.00, Δscore/Δsub = 0.0000** — a draw
bought, no gain. It is the exact confound the differ exists to separate, observed in the wild inside
23 minutes.

---

## §4 — Why `Δscore / Δsubmission` is the decisive quantity

The public score is a **maximum over a team's submissions**. A team that adds draws to an entirely
unchanged agent drifts upward for free. Board-wide this is large and already measured:
`r(log1p(subs), score) = 0.561` across all 2331 teams; median submissions are **44.5** for teams
≥1.60 versus **3** for the rest (audit §3).

`Δscore/Δsub` decomposes a day's movement:

| Pattern | Reading |
|---|---|
| Large Δscore, 1–3 new subs (high Δ/draw) | **STEP** — a capability change. Flagged `STEP` at ≥0.05/draw. |
| Small Δscore, many new subs (low Δ/draw) | **DRIFT** — climbing the order statistic, not the agent. |
| Δsubs > 0, Δscore ≈ 0 | `DREW-NO-GAIN` — submitted and did not improve. Informative. |
| Δsubs = 0, Δscore > 0 | `MOVED-WITHOUT-NEW-SUBS` — **inspect**: rescore or archive artifact, not a result. |
| Δscore < 0 | `SCORE-FELL(anomalous)` — a max cannot fall. Rescore, withdrawal, or a broken archive. |

The differ never reports a bare Δscore without the Δsubs beside it.

---

## §5 — The watchlist, and the rationale for each entry

### Control arm (§1 of every report, first-class, never left to the reader)

| Team | Handle | Today (08-15 14:50Z) | Why they are the control |
|---|---|---|---|
| **Jack Cole** | `jcole75` | **1.59, #22, 95 subs** | MindsAI. **Originator of test-time training for ARC.** ARC Prize 2025 **3rd place**. |
| **Tufa Labs** | `jeroencottaar` +5 | **1.62, #15, 107 subs** | His 2025 teammates. **Authors of the duck harness we fork.** |

These two are the control because they have the **means** (they wrote the methods), the **motive**
(they are 1.0+ off the leader), and the **cadence** (95 and 107 submissions — they submit
constantly). If a commodity engine swap were what moved the board, they are among the most likely
people alive to execute it, and among the fastest.

**They are also a live counter-example to the pedigree inference.** The campaign is being invited to
conclude "the top regime is TTT/fine-tuning" from **one** entrant's *prior-competition* authorship.
The two other most TTT-credentialed teams on this same board sit at **#22 and #15**. Credential does
not map to position here.

### Top eight (audit §1)

cstl (2.70/#1/25), Daniel Franzen (2.58/#2/41), Nikita Sorokin (2.10/#3/6), Yusaku Muroya
(1.98/#4/71), AbeLincoln1865 (1.90/#5/7), YUTO KOJIMA (1.86/#6/69), MLRush (1.75/#7/49),
Andy liu (1.69/#8/7). Tracked because they are the population the pivot argument is about.
Note two of them are **outside the 08-14/15 event entirely** — KOJIMA has been flat since ≥07-24,
Andy liu's last submission is 08-03 — and the differ will keep showing that with data rather than
memory.

### Us

Canivel, **1.33, #119, 111 subs**, last submission 2026-08-15 00:07:11 (which scored 0.89).

### The 1.55–1.65 band (aggregate)

**22 teams, median score 1.605, median 51 submissions** as of today.

This band is the duck-harness lineage's plateau — the dense cluster sitting on a **shared public
artifact we also run**. cstl sat inside it at 1.59 from 08-04 to 08-09 before stepping to 2.52.
*We are currently at 1.33, i.e. 0.22 **below** the band floor* — the band is the family's home and
our target, not our current address; say it that way and do not round it off.

**Why the band is the discriminator:** if the mechanism behind the 08-14/15 event were a drop-in
engine swap on a common artifact, the band lifts **broadly** within ~72 h — many teams, small-to-
medium Δscore, no single team's private breakthrough required. If instead only the same five teams
hold their gains and the band stays flat, the mechanism is **team-specific** and no model swap buys
it for us. Both readings are about **scores**. Neither names a method.

---

## §6 — The 72-hour decision rule

Evaluate on the 08-18 report (three full-board diffs: 08-15→16, 16→17, 17→18).

| Observation over 72 h | Reading | Consequence |
|---|---|---|
| **Neither Jack Cole nor Tufa Labs gains** (Δscore ≤ 0.005) **and** the 1.55–1.65 band is flat (Δmedian ≤ 0.02, Δcount ≤ ±2) | The **commodity-engine / shared-regime story is WEAK**. Two teams with means, motive and cadence did not gain, and the artifact family they anchor did not lift. | **Do not pivot the campaign on the shared-regime story.** An engine swap is not established as the mechanism and must not be logged as one. |
| Cole **and/or** Tufa gain materially (Δscore > 0.05) on few new draws, **and** the band lifts broadly | The story **survives this window**. It is **not confirmed** — a score move is a score move. | Engine/artifact work becomes a live candidate lane, logged **INFERRED**, gated before spend. |
| The band lifts broadly but the controls do not | Something moved the family without moving its authors. | Inspect band composition — entries vs in-place gains — before any reading. |
| Controls gain but the band is flat | Team-specific work by strong teams, not a commodity effect. | No swap will buy it. Prioritise capability, not engine. |
| Any team posts a large Δscore on ≤3 new submissions | **Highest-information event on the board.** Capability-dense, cheap to detect, and this is exactly the profile of Sorokin (2.10 on 6 subs) and AbeLincoln1865 (1.90 on 7). | Flag it. It still says **nothing** about method. |

**Everything in that table is a statement about scores and submission counts.** A verdict of "weak"
means *the score data does not support the story*, not *the story is false*; six of eight remain
UNKNOWN and the correct response to UNKNOWN is to stop asserting, not to assert the negation.

---

## §7 — Backfill: what history the differ actually has on day one

`runs/lb_daily/archive_index.json` (regenerate: `python scripts/lb_archive.py --index`).

| Date | Coverage | Rows | `SubmissionCount`? | Visibility floor | Supports full-board diff? |
|---|---|---|---|---|---|
| 2026-08-06 | top20 | 16 | **no** | 1.54 | **no** |
| 2026-08-07 | top20 | 20 | **no** | 1.49 | **no** |
| 2026-08-08 | top20 | 20 | **no** | 1.50 | **no** |
| 2026-08-09 | top20 | 20 | **no** | 1.50 | **no** |
| 2026-08-10 | top20 | 20 | **no** | 1.50 | **no** |
| 2026-08-11 | top20 | 20 | **no** | 1.50 | **no** |
| 2026-08-12 | top20 | 20 | **no** | 1.54 | **no** |
| 2026-08-13 | top20 | 20 | **no** | 1.56 | **no** |
| 2026-08-14 | top20 | 20 | **no** | 1.56 | **no** |
| 2026-08-15 | top20 | 20 | **no** | 1.60 | **no** |
| **2026-08-15** | **full** | **2331** | **yes** | 0.00 | **yes — the first one** |

**Stated plainly: every archive before 2026-08-15 is top-20-only and cannot support a full-board
diff.** They carry no `SubmissionCount` at all, so no Δsubmissions and no Δscore/Δsub is
recoverable for any prior day, for any team, ever. That is not a bug in the differ; the data was
never collected. This is exactly why the audit could only bound the 08-14/15 event to a **25.8-hour**
window instead of ~6 hours.

What the backfill *does* give us: the differ ingests the top-20 schema and will produce a **DEGRADED**
diff on request (`--allow-partial`), with the visibility floor printed and these consequences stated
in the banner every time:

- Δsubmissions unavailable ⇒ Δscore/Δsub uncomputable;
- `ENTRY` may mean *crossed that day's floor*, not *new team*;
- `EXIT` may mean *fell below the floor*, not *left the competition*.

Without `--allow-partial` the differ **refuses** and exits 4. It will not silently produce a number
that looks like a full-board diff and is not one.

The first true full-board diff is possible **2026-08-16**. Until then `python scripts/lb_diff.py`
with no arguments exits **3** and says so.

---

## §8 — Heartbeat: silence from an automation is not success

**Standing incident:** `ARCMorningCheck` was refused on two consecutive days by
`MultipleInstancesPolicy=IgnoreNew` and nobody noticed, because *a refused scheduled task looks
identical to a healthy idle one*. A collector that quietly does not run produces "no change" — which
reads exactly like a flat board.

Therefore every run writes `runs/lb_daily/heartbeat/lb_archive_<date>.json`, **on success and on
failure**, containing: `status`, `pull_utc`, `pull_local`, `rows`, `columns`, `sha256` of the
archive, the exact `source_command`, the full watchlist snapshot, the band aggregate, and — on
failure — `error`. A later step asserts on it:

```
python scripts/lb_archive.py --check     # prints HEARTBEAT OK ...; exit 0
                                          # exit 1 if absent, status!=OK, archive missing,
                                          #        or sha256 no longer matches the pull
```

This is wired as **step 7** of `scripts/morning_check_prompt.md`, immediately after the collector,
and the prompt instructs the agent to say so loudly in `ITERATION_LOG.md` if it fails. Verified on
both branches today: a malformed CLI invocation produced `status=FAILED` with the error captured,
and the successful run produced `HEARTBEAT OK 2026-08-15 rows=2331 pull_utc=2026-08-15T14:50:39Z`.
The `sha256` check also catches an archive edited *after* the pull, which is a different failure
from "did not run" and should not be confused with it.

---

## §9 — Usage

```bash
python scripts/lb_archive.py                 # pull + archive + heartbeat + index   (daily)
python scripts/lb_archive.py --check         # assert today's heartbeat; exit 1 if not
python scripts/lb_archive.py --index         # rebuild coverage index, no network
python scripts/lb_archive.py --dry-run       # show intended actions, no network

python scripts/lb_diff.py                    # latest two FULL archives
python scripts/lb_diff.py 2026-08-15 2026-08-16
python scripts/lb_diff.py --allow-partial 2026-08-14 2026-08-15   # labelled DEGRADED
python scripts/lb_diff.py --md learnings/lb_diff_2026-08-16.md    # also write markdown
```

Differ exit codes: `0` ok · `3` fewer than two full archives · `4` mixed coverage without
`--allow-partial`. Collector exit codes: `0` ok · `2` pull failed (heartbeat still written with
`status=FAILED`).

Report sections, in order: **1 control arm** · 2 watchlist · 3 the 1.55–1.65 band · 4 largest score
moves · 5 draws bought vs gain · 6 entries/exits/renames · scope-of-claim footer.

## §10 — Known limits

1. **Daily granularity only.** A one-per-day pull cannot resolve intra-day ordering. If ordering
   ever matters again, add a second pull; the archiver is safe to run more than once a day
   (intraday snapshots belong in `runs/lb_daily/intraday/`, which the index deliberately ignores).
2. **`Score` remains a max.** The differ measures the max's movement; it never observes a per-draw
   distribution. The best-of-N correction in audit §3 is a separate, **INFERRED** lens with its own
   σ assumption, and the two must not be merged into one number.
3. **No field dates a scoring run.** §3.1. This will not be fixed by more collection.
4. **A watchlist team that disbands and re-forms under a new `TeamId`** is caught by name/member
   fallback, but a team that renames *and* changes members would be missed. The report prints
   `ABSENT from the new archive` rather than silently dropping the row.
5. **Membership in the 1.55–1.65 band is a score bucket, not a claim that those teams run the duck
   harness.** The overlap is historical and partial. Do not upgrade it.
