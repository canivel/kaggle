# DAILY BRIEF — 2026-08-30 (Sunday)

STEP 1b/1c (discussions + research sweep) were delivered in full by the collect session as
`learnings/community/brief_2026-08-30.md` (2,624-row board archive, 193 topics paged to
exhaustion, 675/675 comments, 4 new kernels read cell-by-cell). **That document is the 1b/1c
record and is not restated here.** Below is what THIS session verified, corrected, or newly
measured, plus the Sunday panel.

---

## 1a. RESULT DEEP-DIVE — the 1.62 (TV28 fork draw 1)

**The number:** public **1.62**, 2026-08-30 00:19:33 UTC. Banked max **2.05** unchanged.
Board **#153 / 2,624**.

**What was actually shipped — now verified at RUNTIME, not just in source.** The pulled
kernel log carries the graft banner verbatim:

    TAAF_GRAFTS FEATURES={"carryover":true,"clickmap":true,"clockwatch":true,
    "efficiency":true,"goalkeep":true,"hudmask":true,"lawbook":true,"retry_guard":true,
    "searchmap":true,"shortcircuit":true,"undo":true,"untried":true,"winframe":true}
    API_VERSION=1

**All 13 grafts were LIVE and `install()` did not fall back** (no fallback or graft-failure
line anywhere in the log). This upgrades the community brief's find #3 from [V on source] to
**[V on the run]**, and closes its own stated gap (*"the 1.62's runtime graft banner has not
been pulled back"*). The submission description's *"Grafts armed:
clickmap/goalkeep/hudmask/searchmap"* understates the artifact by nine grafts: **the 1.62 is a
THIRTEEN-graft datapoint and must be entered as one.**

**Pre-registered expectation: there was none — and that is the honest reading.** The arm
shipped self-declared uncertified. Under `feedback_seed_vs_own_config` a single draw of a
fresh config carries no verdict, and 1.62 sits inside the field-floor's own draw band. The
correct read is *"one draw of an unvalidated 13-graft config"* — **not** "the grafts hurt".

**Offline commit artifact** (`runs/kernel_pulls/tv28_fork_v1/`; 4 game-runs, 2 h 12 m, 340
actions, 783 k tokens): mean score 2.69, median 0.24, 3 levels cleared total — `m0r0` 1/6
(score 0.475), `sk48` 0/8, `sk48-dup` 0/8, `tn36` 2/7 (score 10.29). Run healthy; the
fingerprint writer scanned this log and found **no failure signal**.

**Does the delta vs the ledger imply anything? No — see §2.** 1.62 is not comparable to the
0.9316-mean frozen-fork ledger at all: that ledger measures a *different artifact*.

---

## 2. CORRECTION — `runs/ledger.json` is NOT stale

The community brief's decision item 3 states *"`runs/ledger.json` is stale — `latest_date` is
2026-08-20 and ten draws have landed since."* **Re-derived today from the live Kaggle API
(`scripts/ledger.py --no-write`), the file is exactly right:**

    frozen-fork ledger  n=37  mean=0.9316  s=0.1771
      latest 2026-08-20  0.41   trailing-4 0.8425   bar 1.089

Byte-for-byte identical to the file on disk. The ledger's **membership rule is explicit** in
`scripts/ledger.py`: a submission counts iff its description carries `frozen-fork filler` or
`frozen-fork sigma draw`. It is the *artifact-noise distribution of one byte-identical
artifact*, deliberately NOT our capability record. **None of the ten recent draws is a
frozen-fork filler** — they are field-floor fillers and capability arms — so `latest_date
2026-08-20` is correct. **This was a category error: a null-pool statistic read as a running
score log.** The promotion bar **1.089 stands unchanged.**

---

## 3. ★★★ THE DAY'S MAIN FINDING — WE AUDITED THE FREE INSTRUMENT AND ITS GAME IS THE WORST OF THE 25

The community brief's #1 find was `bench`: a paired, within-run, **zero-draw** A/B already
mounted in the bundle our kernel serves from. We built it — and then, per
`feedback_audit_the_instrument`, priced its game **before the data landed**.

**`BENCH_GAME = "m0r0"`, the author's choice, has essentially no dynamic range.** Two
independent instruments, two different models, same verdict:

| instrument | n | lc mean | lc sd | min | max | p(lc>0) |
|---|---|---|---|---|---|---|
| all-config archive (every m0r0 run we retain) | 76 | 0.197 | 0.398 | 0 | **1** | 0.20 |
| within-config, `tufa_example_run` 20 clone replicates (Qwen3.6) | 20 | 0.050 | **0.218** | 0 | 1 | **0.05** |

`m0r0` has **never cleared 2 levels in 76 retained runs**, and **1 of 20 clone replicates**
cleared a single level. It ranks **#18 of 25** by lc sd on the all-config archive and **LAST**
on the within-config clone set. **A 2-v-2 on it returns `0,0,0,0` by construction.**

This is `feedback_verify_treatment_can_fire` in a new form: not *"can the treatment fire?"*
but **"can the MEASUREMENT resolve anything?"** — and it is the second time in two days that
an inherited component was adopted without pricing it (yesterday: the unpinned bundle).

**Action taken:** slot 1 (`canivel/arc3-tv28-bench`, m0r0) is **re-designated a replication of
the author's exact rig plus an infra shakedown of the bench path**, expected NO-VERDICT. Slot 2
(`canivel/arc3-tv28-bench-sb26`) is the same rig with **one token changed**.

**Why `sb26`** (within-config clone set, n=20): lc mean 1.100, **sd 0.300, min 1, max 2,
p(lc>0) = 1.00** — the only candidate clearing a level in *every* replicate, so the outcome is
graded rather than zero-inflated, with the lowest noise of the viable set. `ft09` has more
range (sd 0.975, 0–3) but its noise manufactures false 2–0 separations, exactly the failure
mode a KILL instrument must not have. Independently, `sb26` carries **50.4% of the certified
field floor's entire mean_score**, so it is also the game that actually moves our number.
**Caveat up front:** the clone set is Qwen3.6, we run Qwen3.8, so `p(lc>0)=1.00` is not
guaranteed on our stack. What is robust is the negative: **m0r0 is bottom-ranked on both
instruments and both models.**

**Cost of the whole exercise: 0 submissions, 0 draws.**

---

## 4. SUNDAY PANEL — ROUND 29: 0 ACCEPT / 5 MAJOR-REVISION / 1 FATAL, and the FATAL was right

R29 ran clean — no infra death, all five reviewers returned 6.8–9.2 k chars (scores 5,6,6,6).
**Twelfth consecutive round with zero accepts**, consistent with the standing finding that
MAJOR-REVISION is the panel's absorbing state. Its advisory value today was real:

**[FATAL, `rl-planning`] The 2v2 bench cannot license a kill, and our own variance data proves
it.** Minimum permutation p at n=2/arm is 1/C(4,2) = **1/6 ≈ 0.167** one-sided — the maximally
separated outcome happens by chance one time in six. *"There is no symmetric 'do not let it
kill anything at n=2' guard, and killing is exactly what the handoff invites."*

**Accepted in full.** The base prereg guarded the PROMOTION direction and left KILL unguarded
— the same asymmetry error, committed in the opposite direction.
**`learnings/war_room/bench_prereg_amendment2_2026-08-30.md` sealed before any data landed:**

- **Neither session may license a stop/continue decision.** Both runs are descriptive only.
- **Numeric gate now pre-registered:** statistic = `levels_completed` (B−A); **no read until
  n ≥ 6/arm** (3 zero-draw sessions); KILL requires one-sided Wilcoxon **α = 0.05** *and*
  median gap ≥ 1.0 levels; anything else is NO-VERDICT. Pool **resets** if the bundle is
  republished mid-accumulation.
- **Per-arm validity gate:** no `TAAF_GRAFTS` banner on an A replica ⇒ session **VOID, not
  null** (else a silent `install()` fallback compares placebo to placebo and reads as "the
  stack does nothing"). Symmetric check that B's suppression actually changed the prompt.
- **Order/budget confound logged:** 4 replicas × 7920 s ≈ the whole session, so arm is
  confounded with session position; per-replica termination cause required at readout, tag
  order alternated across sessions. **Treatment definition stated:** arm B is *"no graft info
  AND the token refund" —* a null means the information is worth about its own token cost, not
  that it is worthless.
- **Extrapolation rule sealed:** a result on `sb26` licenses a claim about **`sb26` only**.

Three further MAJORs carried forward in A2.5: **zero-inflation** (byte-identical 1.82/0.00 plus
our 0.41 suggest a mixture — if so, config mean conflates capability with infra reliability,
which bears directly on `project_arc_final_selection_rule`; classifying near-zero draws by
cause from logs we already hold is free and is tomorrow's top candidate); **no exploration
policy, only measurement policy** (*"a plan that can only lose more slowly is a plan to lose"*
— the gap to #10 widened 0.75 → 0.93 while we banked a draw); and **inconsistent single-draw
inference** (our +2.01σ called variance, Nader's +1.58 called a capability step — both
downgraded to UNKNOWN until a variance model exists).

---

## 5. BUILD RECORD (both daily slots spent; neither is a submission)

| | slot 1 | slot 2 |
|---|---|---|
| kernel | `canivel/arc3-tv28-bench` | `canivel/arc3-tv28-bench-sb26` |
| game | m0r0 (author's choice) | **sb26** (dynamic-range-corrected) |
| notebook sha256 | `7f822e8c…dda71f33` | `abb8c970…cbd95826` |
| vs author's rig | **all 17 cells byte-identical** | cell 14 only (1 token) |
| preflight | trusted-fork **0 FAIL** / 1 WARN (unpushed) | structural **ALLOW 0/0**; D4 matches declared diff `[14]` |
| status | RUNNING | RUNNING |

Pull-back verified on slot 1: `enable_gpu`, `dataset_sources`, `model_sources`,
`competition_sources`, `docker_image`, `machine_shape` all **exact**. Both replica blocks are
guarded on `if not TRUE_SUBMISSION:` (build-time assert), so both are **provably inert in a
real rerun**; neither kernel is queued or submitted.

**Minor instrument note, root-caused and benign:** the Kaggle CLI push/pull round-trip
mojibakes 7 em-dashes (U+2014 → `â€"`), all inside comments. `preflight.py`'s `_tf_norm`
already treats this as equal (*"differ ONLY by non-ASCII pull round-trip mangling"*), and the
same artifact class ran COMPLETE at 1.62. No action.

**New rail constraint found:** KAOS's `claude_code` provider has **no permission-mode setting**,
so spawned agents can read files but **every shell call is blocked** ("required approval").
Analysis is delegable; build/shell work is not. Recorded against `feedback_kaos_improvements`.

---

## 6. WEEKLY — FAILURE FINGERPRINTS (writer run, THEN reader; store FRESH)

`store FRESH: 57 retained logs all scanned (written 2026-08-30T12:29:41Z, newest incident
2026-08-18, newest log 2026-08-30)` — **23 incidents, 11 recurring families.**

| family | n | first | last |
|---|---|---|---|
| class:ERROR:none | 7 | 2026-05-26 | 2026-06-28 |
| provenance:scratch-built | 5 | 2026-05-26 | 2026-06-28 |
| slug:canivel/arc3-final | 4 | 2026-05-26 | 2026-06-10 |
| class:COMPLETE:0.00 | 3 | 2026-03-29 | 2026-06-10 |
| slug:canivel/arc3-forge35 | 3 | 2026-04-24 | 2026-06-22 |
| slug:canivel/arc3-pilot-eval | 3 | 2026-07-07 | 2026-07-08 |
| t1:07d0f5248c48401d | 3 | 2026-07-07 | 2026-07-08 |
| class:COMPLETE:null-band | 2 | 2026-06-01 | 2026-06-08 |
| slug:canivel/arc3-a17-72b-canary | 2 | 2026-07-25 | 2026-07-25 |
| t1:fb1e96c3815797ad | 2 | 2026-07-25 | 2026-07-25 |

**No new incident since 2026-08-18**, and today's tv28 log was scanned and carried no failure
signal. The first read this session fired the `STALE` banner (I pulled the tv28 log *after* the
writer ran); the writer was re-run and the table above is from the fresh store.
`provenance:scratch-built` (n=5) remains the most expensive family and is exactly what
`feedback_arc_kernel_structural_drift` blocks — both of today's builds are forks.

---

## 7. TONIGHT'S HEAD — unchanged, and deliberately so

Queue head stays **`canivel/arc3-q38-field-eval` v1, the certified field floor** (queue lives at
repo root `submission_queue.json` — *not* `runs/` — 1 pending; `ARCDailySubmit` `Ready`, next
fire 18:37). **The TV28 fork is NOT promoted on the strength of 1.62**: n=1, its author's own
kernel states the stack has never been shown to move the score, and his board row (1.93, 39
subs, #207) sits 0.12 *below* ours on a fifth of the draws. No arm cleared a promotion gate
today, and neither bench kernel is eligible to — both are sealed KILL-only *and* now
decision-barred at n=2 (§4), and neither is a submission.

---

## 8. OPEN QUESTIONS

1. **Does `sb26` keep p(lc>0)=1.00 on Qwen3.8?** The clone set is Qwen3.6. If sb26 lands
   0,0,0,0, the bench family is unusable at small n on ANY game and should be retired rather
   than re-aimed a third time.
2. **Classify every near-zero draw by cause** (panel A2.5). Free, uses artifacts already held,
   and decides whether `project_arc_final_selection_rule`'s mean-based selection is operating
   on a bimodal distribution. **Tomorrow's top candidate.**
3. **The unpinned bundle remains unmitigated.** Kaggle kernel `dataset_sources` has no version
   pin, so `thtennant/taaf-kaggle-source-share-fork` can change under all three of our kernels.
   Today's census recorded (`bench.py` 11,430 B, `composite.py` 24,798 B). A vendored snapshot
   is the only real fix; not built.
4. **Exploration policy.** The panel's sharpest strategic point and the one with no owner:
   every action today was defensive while the gap to #10 widened. Named candidates for the next
   Sunday agenda — the never-built stagnation supervisor (`feedback_arc_supervision_gap`), and
   forking a *scoring* lineage rather than a demonstrably non-scoring one.
5. **Should `bench`'s placebo trick be re-pointed at our OWN floor?** Suppressing prompt text
   while holding the code path fixed is a general technique; its most valuable target may not be
   tennant's 13 grafts but our certified field floor.
