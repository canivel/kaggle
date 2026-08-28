> ## ⚠ CORRECTION APPLIED 2026-08-27 15:4xZ (Mac iterate session) — READ THIS FIRST
>
> **Section A1 is REFUTED, and with it the seven "UNSUPPORTED"/"PARTLY" verdicts in its table.
> A7 item 1 is REFUTED. The parent brief's Δ column was correct.** Sections A2–A6 stand; A3 in
> particular is confirmed and strengthened. Full evidence in
> `learnings/execwm_obs_verdict_2026-08-27.md` §2. Summary:
>
> 1. **The diff DID run — on the Windows box, at 06:00:46, against a real archive.** The artifact
>    it produced, `learnings/lb_diff_2026-08-27.md`, is committed in `46afb34` and names
>    `lb_full_2026-08-26.csv` at **2546 rows, pull_utc 2026-08-26T10:00:51Z** — corroborated
>    exactly by the *tracked* heartbeat `lb_archive_2026-08-26.json` (same row count, same
>    timestamp, sha `58c63c78…`). A1's claim "that diff cannot have run" was inferred from the
>    archive's absence **on this Mac**, where `runs/lb_daily/*.csv` has never been tracked and so
>    has never synced. **A local absence was read as a global one.**
> 2. **`lb_diff.py` does NOT exit 0 on a missing archive.** Measured directly: it exits **1** on
>    explicit dates (`raise SystemExit(...)`, line 78) and **3** on the no-arg path. The `EXIT=0`
>    recorded in A1 is a **pipeline artifact** — `cmd 2>&1 | tail` reports *tail's* status, which
>    reproduces as `PIPED_EXIT=0`. The A7 repair ("make lb_diff exit non-zero") is therefore a
>    **no-op on code that is already correct**, and building it would have left the real defect
>    untouched.
> 3. **Every struck claim is in the artifact, verbatim:** rfbr `3.37 +1.18 subs=13 +1` (line 61);
>    "teams that submitted at all : 308 / gained anything : 74 (24.0%) / median dScore/dSub among
>    gainers : 0.2200" (lines 75–79); "entries: 19 exits: 1" (line 105); band `52 → 48`, median
>    1.60 flat, median subs 14.0 → 12.5 (lines 44–47).
> 4. **The band "correction" 52→48 vs 52→46 is not a correction — it is A3.** The addendum
>    computed 46 from its own **13:26Z** re-pull; the brief computed 48 from the **10:00Z** pull.
>    Different measurement windows, and the gap is exactly the late-landing draws A3 identifies.
>    A3 is right, and it explains this cleanly.
>
> **The real, systemic defect A1 was circling — and it is worse than the one it named:**
> `runs/lb_daily/*.csv` is gitignored, so **the daily evidence base does not cross machines.**
> Each box can audit only its own half, and a two-rail campaign will keep generating audits that
> "refute" work done on the other box. This is the 08-26 two-rails ruling reappearing at the level
> of *evidence* rather than *submissions*. The `!runs/lb_daily/*.csv` negation added this morning
> fixes it **going forward only** — the 08-22..08-26 archives still exist on the Windows box and
> should be committed from there, not re-pulled (Kaggle serves only today's board).
>
> **Nothing below has been edited**; this banner is prepended so the original reasoning stays
> auditable. The addendum's method was sound and its other four findings are real — A4 (the first
> proportional local→LB report) and A5 (the Kamradt host statement pricing the census) are both
> load-bearing and both stand.

---

# ADDENDUM — Daily Community Brief 2026-08-27 (second, independent pass)

**Status: AUDIT + SUPPLEMENT to `brief_2026-08-27.md`. Written from an independent 13:26:10Z pull.
Two of the parent brief's three DECISION HANDOFF items were already executed and REFUTED before it
was written, and its entire Δ-vs-08-26 column rests on a diff that could not have run on this box.
Read this before acting on the parent brief.**

**Why two briefs exist today.** The 06:00 launchd `daily_community` job deferred to **09:14** (battery
power, per commit `78ecd31`). A second, interactive community session started the same minute. The
launchd twin wrote `brief_2026-08-27.md` at **09:42**; this addendum is the second pass, pulled at
**2026-08-27T13:26:10Z** (3h26m after the archive the parent brief cites). Duplicated work is not
repeated here — the parent's Polyphony read, `set_level` execution, duck-v26 diff and dataset
censuses are accepted and corroborated where I could check them.

**Tags:** [V] verified this session · [V-doc] verbatim claim inside a verified artifact · [INF] inference.

---

## ★ A1 — THE PARENT BRIEF'S Δ COLUMN IS NOT SUPPORTED BY ANY ARTIFACT ON THIS BOX [V]

The parent brief's Method line claims `scripts/lb_diff.py 08-26→08-27 (exit 0)`. **That diff cannot
have run.** Executed this session, verbatim:

```
$ .venv/bin/python scripts/lb_diff.py 2026-08-26 2026-08-27
no archive for '2026-08-26' (looked for runs/lb_daily/lb_full_2026-08-26.csv,
                                        runs/lb_daily/lb_2026-08-26.csv)
EXIT=0
```

`runs/lb_daily/` holds **exactly one** `lb_full_*.csv` — today's. The 08-22..08-26 archives never
migrated from the Windows box: `.gitignore` carried a bare `*.csv`, so only the heartbeat JSONs
(tracked) survived. A filesystem-wide search for `lb_full_2026-08-26.csv` returns nothing. The same
morning, `morning_check` reached this conclusion independently and recorded it — *"LB diff is blind
today — migration data loss, not a skipped run… No Δscore-per-Δsubmission figures today; with one
archive there is no Δ, and bare Δscore is the error this step exists to prevent."* — **and then the
community brief published a full Δ table an hour later.**

**Note the mechanism, because it is the point: `lb_diff.py` printed its failure and still exited 0.**
That is how "exit 0" got recorded as success.

**What survives, and what must be struck.** 08-26 values recoverable on this box are (a) the ~14 rows
transcribed in `brief_2026-08-26.md`'s own delta table and (b) the 11-row watchlist + band block in
`heartbeat/lb_archive_2026-08-26.json`. Everything else needs the missing 2,546-row archive.

| parent-brief claim | verdict |
|---|---|
| MindsAI 2.05 → 2.94 (**+0.89**), #115 → #7 | **STANDS** [V] — both endpoints in the 08-26 heartbeat watchlist; I confirm 2.94 / 130 subs / rank 7 |
| cstl, LHS, Tufa, Franzen, Tony Li, Tatu, Tony G, AbeLincoln, FOYSAL, Beyond Good and Eval, us | **STAND** [V] — endpoints in yesterday's table |
| Band 1.55–1.65 "52 → 48 (−4)" | **PARTLY** — 08-26 n=52 is in the heartbeat [V], but today's n is **46**, not 48. I compute 46; `morning_check` independently computed 46. Correct line: **52 → 46 (−6), median 1.60 flat, median subs 14.0 → 11.5** |
| rfbr "2.19 → 3.37 (+1.18)" | **UNSUPPORTED** — rfbr is in neither yesterday's table nor the watchlist. The 3.37 is [V]; the 2.19 and the +1.18 have no source on this box |
| "field-wide 308 submitted, 74 (24.0%) gained, median gainer 0.22/draw" | **UNSUPPORTED** — requires the missing full archive |
| "19 in, 1 out"; "26 of 30 gained exactly 0.00" | **UNSUPPORTED** — same reason; only ~10 of the top 30 have a recoverable 08-26 value |
| AI Winter +0.42, paul +0.21 | **UNSUPPORTED** |

**This does not overturn the top-3 pattern hypothesis** — the two largest steps (MindsAI +0.89,
and rfbr's presence at 3.37 on 13 lifetime subs) are real. It overturns the *precision* the parent
brief claimed, and the field-wide gainer statistics quoted for a fifth consecutive day. **First real
diff possible 2026-08-28.** The gitignore is fixed (`!runs/lb_daily/*.csv`, from `morning_check`).

## ★ A2 — TWO OF THE PARENT BRIEF'S THREE HANDOFF ITEMS WERE ALREADY EXECUTED AND REFUTED AT 08:48 [V]

The deferral inverted the day's order: **iterate ran BEFORE community.** Commit `46afb34` (08-27
**08:48**) predates `brief_2026-08-27.md` (09:42) by 54 minutes, so the iterate session consumed
*yesterday's* brief. It then tested and killed exactly what today's brief re-recommends:

- **Handoff #2 — "pull the EXEC-WM artifact and check BREAK clustering… second day this has been the
  top item and it has not been done."** It **was** done (exp 58) and the hypothesis is **REFUTED by
  its own artifact**: *"The prediction-break latch fired ZERO times; a mislabelled reason string (one
  path, two disjuncts, one label) caused two days of wrong diagnosis. Real cause is data starvation:
  26/32 level-instances at no-verified-model, 9/18 games with zero transitions, while retrodiction
  was 810/818 where data existed. Kill clause does NOT fire; re-scope to repair the observation layer."*
  → **The level-win frame-layer defect is real but was NOT the cause. Do not spend today re-auditing
  BREAK clustering. The live lane is the observation layer / data starvation.**
- **Handoff #3 — "if only one point is built today, build the sticky policy deadline."** **REFUTED
  pre-build** for ~10 min CPU (exp 59): *"39.3% of floor level completions land after the 0.55 mark;
  the back half of the clock is where 30-40% of our score is made. Banked prior: p50 0.310, p90 0.850."*
  → **Polyphony's 0.55 constant is actively harmful on our rail. Do not build it.** The other two
  Polyphony design points (bootstrap probe, `metrics.py` instrument set) are untouched by this and
  remain open.
- Also from the same run: **P2 (exp 57) D1 PASS 24/25 but D2 FAIL 10.73% vs sealed 25% → DELIVERY
  FAILURE**, seed 2 refused; third paid confirmation of advertise-where-the-model-reads.

**Handoff #1 (the `set_level` offline level census) is NOT refuted and remains the best item on the
list** — see A5 for a constraint on how its output may be used.

## ★ A3 — THE DAILY ARCHIVE IS PULLED BEFORE THE PREVIOUS NIGHT'S DRAWS FINISH SCORING [V]

A submission can run up to 9h, and **`SubmissionCount` does not increment until scoring completes** —
so a draw started after ~01:00Z is invisible in *both* score and count at a 10:00Z pull. Measured
directly by comparing the parent brief's 10:00:40Z archive against my 13:26:10Z archive:

| team | 10:00:40Z (parent brief) | 13:26:10Z (this pass) | Δ in 3h26m |
|---|---|---|---|
| **Tong Hui Kang** | 3.39 / 54 subs — flagged **IDLE** | **3.88 / 55 subs** | **+0.49, +1 sub — a new top-5 STEP that the brief recorded as idle** |
| Lord Han Solo | 4.99 / 44 | 4.99 / **45** | +0.00, +1 sub |
| all other top-10 rows | — | unchanged | — |

**5 of the top 30 submitted inside the at-risk window** (LHS 03:49Z, Tong Hui Kang 01:11Z, Jonathan
Wang2022 01:08Z, Sean Jones 03:53Z, Cloud 03:21Z). One of the five flipped from IDLE to a +0.49 step.

**Why this is more than a bookkeeping note.** `DREW-NO-GAIN` / `IDLE` flags are the primary evidence
for the campaign's #1 hypothesis (*"the flats are the control arm"*). The instrument that produces
them **systematically under-counts late-landing gains, biasing exactly toward more flats.** The
hypothesis is not refuted — it survives on the step sizes, which are real — but the daily gainer rate
is a **floor, not a point estimate**, and any single team's one-day "flat" is unreliable if it
submitted after ~01:00Z. → **Action: move the LB archive pull to ~14:00Z, or pull twice and diff
the pair. Zero cost.**

## ★ A4 — FALSIFIER #6 FIRES: THE FIRST PROPORTIONAL local→LB REPORT, AND IT IS BOARD-CORROBORATED [V]

`top3_pattern.md` falsifier #6 reads: *"'local→LB transfer is broken and rail-side, not
capability-side' would die if a team ever reported a local score that tracked its LB score
proportionally. **Not observed.**"*

**It is now observed.** Topic 732854, **mikelou1, 2026-08-26 11:03Z** [V-doc]: ***"Got 2.8 on 25
games and 2.4 on lb."*** I resolved the author to the board: **mikelou1 = team "Proving AGI", rank
#34, score 2.43, 26 subs** [V] — so the LB half of the claim is independently confirmed, and it sits
**above the frozen public ceiling** (FOYSAL 2.23), i.e. this is not a duck-floor artifact.

| team | local | LB | ratio |
|---|---|---|---|
| **mikelou1 / Proving AGI** | **2.8** | **2.4** (board: 2.43) | **0.87** |
| duck harness (Pellegrin, reported) | 2.1 | ~1.4 | 0.67 |
| Pellegrin's own harness | 5.0–5.4 | ~1.4 | ~0.27 |
| daoviet (board: 1.99) | 6.8 | 1.19 | 0.17 |

**The reframing is the find.** The collapse is **not a property of the rail** — it is a property of
*over-fitted local harnesses*. The teams reporting 5–7 locally are measuring something their agent
has been tuned against; the team reporting 2.8 locally is measuring something that transfers. Note
the inverse relation: **the higher the local score, the worse the transfer.**

**This directly qualifies yesterday's hardened screening rule.** Our 0-for-36 screening record
(`feedback_screen_calibration_range`) is evidence about **our screen**, not evidence that screening
cannot work. → **Action: keep the "must name the mechanism" gate, but drop the premise that local
deltas are inherently non-transferable. Add a calibration target: a screen whose absolute number
lands near our LB (~1.9–2.4), not 3× above it, is the one to trust. A local screen reading 5+ is a
red flag about the screen.** Also new, no LB given: **donk666 (08-27 01:43): "between 3.5-7.5."**

## ★ A5 — A HOST STATEMENT PRICES HANDOFF #1 (THE `set_level` CENSUS) [V-doc]

Surfaced by reading old threads in full (see A6). Topic 707925, **Greg Kamradt (ARC Prize),
2026-06-12**, on notebooks that load the game engine locally and solve offline:

> *"these notebooks appear to locate the relevant game source file, dynamically load or instantiate a
> private local copy of the game engine, solve the level inside that unscored copy using classical
> search or planning, and then replay only the resulting action sequence through the scored API. We
> have not seen evidence of this on private evals games. Also, we'd likely see scores higher than
> .68% on the public leaderboard if this was the case. Those notebooks may be doing that with public
> information, but **those same affordances aren't available on private games**."*

Uncontested reply from Yield Smarter, same thread: *"It is totally fine to locally probe the
available public games with all kinds of algorithms, bfs is a good way to map down the available
games and can even create rich datasets of legit moves within an offline level."*

**Reading: the `set_level` census is legitimate and unobjected-to — but the host has stated on the
record that the affordance does not exist on the private rail.** So the census may produce
**transferable priors only** (control-effect carry rates, shape-sharing rates, object-carry rates —
statistics that generalise), and **never** per-game or per-level solutions. That is precisely the
disqualifier the parent brief applied to `goldworm` (*"scripted plans (public game IDs only)"*).
→ **Build handoff #1, with that constraint written into the harness's docstring.**

## A6 — SWEEP CORRECTIONS [V]

- **New items field-wide since 08-26T10:00Z: SIX, not three.** The parent brief recorded *"exactly
  one new topic and two new comments."* Full set: 1 new topic (737617) + 5 comments — Brüggen
  (08-26 13:42), Halla Yang (08-26 18:14), **OverfitOracle (08-27 12:24)**, **mikelou1 (08-26 11:03,
  = A4)**, **donk666 (08-27 01:43)**. The two misses in topic 732854 are the two that carry content.
- **The truncation gap is bounded and small.** The parent brief's find #7 (the CLI table renderer cuts
  comment bodies at ~200 chars) is correct and I confirm it. Scoping it: across all 120 topics pulled
  as JSON, **exactly 50 comments exceed 200 characters** — those, and only those, were ever read
  partially. The longest sit in admin threads (submission limits, Kaggle errors); the
  mechanism-bearing ones are A5 (Kamradt) and the item below. **The backfill is now done, not
  outstanding.** Parent-brief gap (a) is discharged.
- **A fourth local→LB datum with a hardware mechanism** [V-doc]. Topic 697407, **Scott Le Grand,
  2026-08-18**: *"I'm seeing this too. **Qwen 3.8 exacerbated this.** I am going to try lowering the
  reasoning to medium tonight. I am wondering if there is some sort of multitenancy contention issue
  on the shared PCIE bus… I believe the HW is 8 GPUs? But they share the same PCIE bus. My local
  machine is a single RTX Pro 6000 with no other users or tasks. Last week I was in position 19."*
  First *hardware* hypothesis for the discrepancy in the ledger. Sits in tension with A4 — if
  contention were the whole story, mikelou1's 0.87 ratio would be hard to get. → No action; watch.

## A7 — SILENT PARTIAL SUCCESS IS NOW A FOUR-INSTANCE PATTERN IN OUR OWN TOOLING [V]

Four independent instruments this campaign returned **exit 0 with incomplete data and no error**:

1. `lb_diff.py` — prints "no archive for 2026-08-26", **exits 0** (A1). Produced a whole Δ table.
2. The CLI topics **table renderer** — truncates comment bodies at ~200 chars silently (parent #7).
3. `kaggle competitions topics show` **under rate limiting** — my first sweep wrote **76 of 120 files
   at zero bytes**, exit 0 throughout. My initial scan reported "0 new comments" off that; re-pulling
   with a 2 s throttle recovered 116/120 and found the six items in A6. **I nearly published the same
   class of error I am auditing.**
4. `kaggle kernels output` — *"can return exit 0 with a PARTIAL file set (a download race returned
   INFRA DEATH on a healthy run)"* (iterate, commit `46afb34`).

This is `feedback_audit_the_instrument` recurring at the **tooling** layer rather than the experiment
layer. → **Action (cheap, high leverage): make `lb_archive.py --check` assert that the *prior* day's
archive also exists, and make `lb_diff.py` exit non-zero when an archive is missing.** A brief may
not quote a Δ that no artifact supports.

**One integrity note I own.** My 13:26:10Z re-pull **overwrote** `lb_full_2026-08-27.csv`, so the
10:00:40Z archive the parent brief cites no longer exists on disk; its heartbeat
(`lb_archive_2026-08-27.json`, 2564 rows, sha `e882de00…`) now describes a file that is gone, while
the CSV on disk is mine (2568 rows, sha `34affdc3…`). **The heartbeat and the archive disagree —
do not treat that heartbeat as describing the file next to it.** The 13:26 archive is the better base
for tomorrow's diff (later, more complete, and it is what `morning_check` verified at
`pull_utc=2026-08-27T13:26:01Z sha=34affdc35a42`), so I left it in place rather than reconstructing.
`lb_archive.py` should refuse to overwrite a same-day archive without an explicit flag.

---

## REVISED DECISION HANDOFF (supersedes the parent brief's, ≤3)

1. **BUILD THE OFFLINE LEVEL CENSUS (`set_level`) — still the best item, now with a constraint.**
   25/25 games, 182 levels, 157 pairs, zero GPU, ~1 h. **Write A5's constraint into the harness:
   transferable priors only (carry rates, shape-sharing rates), never per-game plans — the host has
   stated the affordance is absent on the private rail.** Unaffected by anything in this addendum.
2. **DO NOT rebuild what was already killed at 08:48. Re-scope EXEC-WM to the observation layer.**
   BREAK-clustering is refuted (latch fired zero times; the two-day diagnosis was a mislabelled
   reason string). The live defect is **data starvation** — 26/32 level-instances reached
   no-verified-model and 9/18 games logged zero transitions, while retrodiction was **810/818 where
   data existed**. That ratio says the model is fine and the *feed* is empty. **And do not build
   Polyphony's 0.55 sticky deadline** — refuted pre-build, 39.3% of our level completions land after
   it. The bootstrap probe and `metrics.py` instrument set remain open and unrefuted.
3. **Fix the two instruments that produced today's bad numbers, before tonight's slot.** (a) Move the
   LB pull to ~14:00Z or pull twice and diff (A3 — a top-5 step was recorded as IDLE today).
   (b) Make `lb_diff.py` exit non-zero on a missing archive and `lb_archive.py --check` assert the
   prior day too (A1/A7). **Total cost well under an hour, zero GPU**, and it protects the daily
   evidence stream that every other decision on this campaign is read off.

**Gaps in THIS pass:** 4 of 120 topic trees still unreadable after re-pull (rate limiting; all are
pre-July admin threads). No independent Polyphony source read — the parent brief's read is accepted
as-is, not re-verified. Kernels swept `dateCreated` p1 and `voteCount` p1 only; `thtennant/arc3-duck-v26`
confirmed present (08-26 12:59, 2 votes) but **not diffed by me** — the parent brief's diff stands
unaudited. FOYSAL `lb-9` independently confirmed frozen since 08-18 10:25 (votes 228 → 240, board
2.23 / 97 subs, rank #62 at 13:26Z vs #61 at 10:00Z).
