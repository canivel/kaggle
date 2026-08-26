# R24 review — prog-synthesis (Program Synthesis / Neurosymbolic AI)

Copy check: proposal body received intact, ends with `## END OF PROPOSAL ##`. Both addenda read.
Claims below were verified by direct read of the repo (paths given); where I could not verify, I say so.

## Summary

The lane decision is right and I would ratify it: (a) state-externalisation with Tycho as artifact
schema rather than as a competing lane, (b) as a component arm, (c) on its own clock, no pushes spent
this week on anything but two free offline tests — that ordering follows from the evidence and from
A22's death, and the §6.1 arm-defining invariant plus the K4 `namespace_reuse_rate` floor are two of
the best-designed falsifiers this campaign has produced. But the single item I was assigned to grade,
**S1/L0, cannot be executed as written on the artifacts it names**: the 25 sims on disk have no
abstention channel (so "coverage" is the constant 1.0), 22 of 25 have no hidden state (so "state
threading" has nothing to thread), and the on-trajectory numbers L0 would produce **already exist** in
`runs/ewm_dryrun/report.md` with a stream-to-stream spread of ~0.40 median and up to 0.85 — which is
larger than any plausible carrier threshold. As specified, the accepted-match + coverage protocol is
the same class of error as the 91.7%: a point-estimate fidelity number with no variance, no
pre-registered threshold, and one degenerate channel.

## Objections

### [FATAL] L0's protocol cannot be run on the artifacts it names, and its gate cannot pass by construction

I read all 25 active sims' interface. Two verified facts kill the protocol as written:

**(i) Coverage is a constant.** Every sim is `simulate(state, action_id, x, y) -> (next_state,
reward_class, done)`, and `exec_wm/validate_sim.py` raises unless the returned grid is exactly
`(64, 64)` — there is no channel by which a sim can decline to predict. `grep -l
"UNKNOWN\|abstain" exec_wm/sims/*.py` returns **0 files**. So "report accepted transition match AND
coverage per game" degenerates to accepted-match alone, and §6.4's headline answer to the round-18
unfalsifiability charge — *"Tycho's abstention + coverage channel is adopted precisely so a model that
knows what it doesn't know scores low coverage rather than high-confidence-wrong"* — is **false for
the artifacts L0 will actually run on**. Coverage only becomes a variable after the `State` /
`init_state` / `transition` / `render` / `outcome` + `UNKNOWN=-1` migration, which is **L1** — the item
gated *behind* L0. That is a circular gate. A metric with no variance carries no information; this is
precisely the failure mode the campaign says it learned from, restated in Tycho's vocabulary.

**(ii) Threading has nothing to thread, and the executable reading makes the gate unpassable.** Only
`g50t_sim.py`, `re86_sim.py`, `tr87_sim.py` hold module-level mutable counters; the other 22 are pure
functions of the visible grid. And those three have no `init_state`, so nothing resets them at "level
frame 0" — their L0 numbers will be an artifact of call ordering, not fidelity. The only reading of
"replay from level frame 0 with state threading" that is executable on this interface is *closed-loop
rollout* (feed the sim's own prediction forward), which is **strictly harsher** than the open-loop,
teacher-forced shadow measurement already reported in `runs/ewm_dryrun/report.md`. Under either
reading the protocol can only move the numbers **down**. A gate whose pass branch is unreachable is
not a falsifier; and its fail branch — *"close exec-wm permanently, retain C1/C2/C3 as schema only"* —
would retire the artifact half of the very lane the panel is being asked to adopt, on a foregone
conclusion. **Fix:** re-scope S1 as a two-part item — (S1a) re-report the *existing* open-loop
on-trajectory numbers under a pre-registered per-game criterion, and (S1b′) a **2–3 sim abstention +
hidden-state pilot** (I would pick `sp80` and `lp85`: frame-Markov clean, worst step-0 abort mass, so
the pilot cleanly separates "threading helps" from "sim is wrong"). The pilot is the item that
actually tests the hypothesis, it is free and workstation-only, and it is a fraction of L1.

### [MAJOR] The carrier gate is not identifiable — the metric's stream-to-stream spread already exceeds any threshold it could use

`runs/ewm_dryrun/report.md` reports on-trajectory shadow accuracy for a **fixed** sim across three
recorded runs of the **same** game. Per-game (v1 / v2 / v3): sp80 **0.026 / 0.879 / 0.067** (range
0.853), su15 0.309 / 0.149 / 0.808 (0.659), tn36 0.530 / 1.000 / 0.984 (0.470), ft09 0.985 / 0.556 /
1.000 (0.444 — and **0.070** on the `gpt56_full` stream), vc33 0.239 / 0.667 / 0.368, lf52 0.301 /
0.496 / 0.752, lp85 0.113 / 0.458 / 0.087. **Median across-stream range ≈ 0.40.** The proposal states
no per-game accepted-match threshold, no coverage threshold, no aggregation rule across streams, and
no uncertainty statement — only "carrier set must **expand beyond ~4 games**", with a tilde. At a
0.40 median spread, carrier/non-carrier flips for the majority of games depending on which of four
streams you happen to replay. Contrast K4, which is pre-registered at a number (0.15). **Fix, before
the run:** name the streams, name the per-game accepted-match *and* coverage thresholds, name the
aggregation rule (min across streams is the honest one), and require the per-game across-stream range
to be reported alongside every point estimate. Without that, S1 reproduces the 91.7% error exactly —
a single fidelity point estimate treated as a property of the artifact when it is mostly a property of
the trajectory.

### [MAJOR] The proposal's own diagnosis contradicts a prior repo finding that predicts L0 will fail for a reason threading cannot fix

§2/§3 and the parent Tycho file collapse the exec-wm failure into one cause: *"stateless sims →
latent-state aliasing."* `learnings/panel/r16_circulation.md` ~L1251 says otherwise, and says it in the
campaign's own words: step-0 aborts split into **(i)** aliased games (s5i5, sb26, vc33, tr87) where
phase augmentation/resync helps, and **(ii)** "lf52, lp85, sp80, su15 are frame-Markov **CLEAN** yet
still have step_acc < 0.6 — those sims are just **wrong** (sim bugs / engine-version drift), and **NO**
amount of state augmentation or resync will save them; they need sim fixes." Those four carry the
worst abort mass (lp85: 126 aborts / 138 plans in v3). So the majority of the carrier-set deficit is
in the class threading provably cannot repair, and a null L0 is **uninterpretable** — it will not
distinguish "externalisation-with-threading doesn't work" from "these particular programs are buggy."
**Fix:** L0 must report per-game against the r16 aliased/clean partition and evaluate carrier
expansion **separately on the two classes**. Related and unhandled: **engine-version drift is an
uncontrolled confound** — accepted-match measures sim-vs-*whichever-engine-recorded-the-stream*
agreement, not synthesis quality, unless the engine version behind each stream is pinned and
disclosed. That alone could account for the sp80 0.026→0.879 swing.

### [MAJOR] The "91.7% held-out state_exact" lesson is mis-stated in a way that misdirects the fix

`exec_wm/scale_summary.md`, line 3, verbatim: *"Validation harness: `exec_wm/validate_sim.py` over all
200 **held-out** tuples per game (**split=all**)."* `split=all` means **no split was taken** —
`validate_sim.py` only partitions when `--split train|test` is passed. Those 200 tuples are the same
tuples the authoring model was handed (`collect_observations.py` docstring: *"produce small focused
datasets (~200-500 tuples per game) for opus-4-8 to study and write a Python simulator from"*), and
the reported figure is for the *"chosen v1/v2 winner from per-game evolutions"* — model **selection**
on the same set. So 91.7% was an **in-sample fit-plus-selection** number labelled "held-out", not an
IID-generalisation failure. Separately, `validate_sim.py`'s split is a *prefix* split
(`tuples[:split_at]`), not the "IID random-split" the Tycho file blames. This matters because the
proposal builds its central methodological lesson on the wrong diagnosis: the sharper lesson is **"we
never had a holdout, and we selected on the eval set"**, which is a discipline failure, not a
metric-choice failure — and discipline failures are not fixed by swapping in a new metric. Correct
§6.4 and the Tycho file. To be fair and explicit: L0 run on the `runs/ewm_dryrun` streams **is**
genuinely out-of-sample relative to those 200 tuples, and that is a real improvement — but it must be
*stated* as the protocol's holdout claim, not left implicit.

### [MAJOR] The generalisation rail is applied to lane (c) and silently withheld from L1, which is the same object

§3(c) kills banking partly because *"banking is by construction a public-set optimisation… contributes
nothing on unseen private games, colliding with the standing generalisation rail."* That argument
applies with equal force to L1 — a **per-game hand-migrated sim set for the 25 public games** — and the
parent file concedes it (`tycho_portability_2026-08-08.md` §6.9: *"a per-game hand-migrated sim set is
by construction overfit to the public 25 and contributes nothing on unseen games"*). The proposal
never carries that concession into §4. If L0 passes and S5 fires, the campaign spends its **largest
single item** (~10 h / ~7 M tokens by the June benchmark) on an asset with zero private-LB value, under
a standing priority that ARC solutions must generalise. The generalising asset is the schema + the
verifier + **the ability to synthesise a sim in-kernel** — and that last one is C8/L5, which the
proposal correctly refuses to authorise on the 52× LM-call gap. So the lane's generalising half is
exactly the half that is out of reach at 27B. State this as the lane's central strategic risk in §4
and price S5 against it, or apply the rail consistently and drop L1 in favour of schema-only.

### [MAJOR] S1 is scoped over 24 sims; only 12 have on-trajectory replay streams

Verified on disk: **25** active `<game>_sim.py`, **25** observation files (200 tuples each) — so the
off-trajectory half of L0 is runnable today, and ADDENDUM A6's "unverified" flag can be discharged for
that half. But `scripts/ewm_replay_dryrun.py` L11/L56 hard-codes *"the 12 saturated exec_wm sims"* and
`runs/ewm_dryrun/report.md` covers exactly those 12 (the `gpt56_full` stream covers only 5). A
"carrier set over 24 games" **cannot be computed on-trajectory today** without extending the replay
harness to the remaining 13 — unscoped work not in the "M effort, 0 pushes" estimate. Either scope and
price that extension, or restate the gate over the 12 games that have streams and say so in the seal.
As a bonus, the denominator itself is inconsistent across documents: 24 (proposal, scale_summary) vs 25
(disk) vs 12 (streams). Fix the denominator before it becomes a gate denominator.

### [MINOR] L0's most likely result is already visible, so it should be framed as confirmation, not as a fresh falsifier

Reading `runs/ewm_dryrun/report.md` across v1/v2/v3, the games clearing ~0.8 in most streams are
**ft09, tn36, tr87, tu93** (plus ls20 in 2 of 3) — i.e. ~4, essentially the known carrier set
`{tn36, tu93, ls20, ft09-L1}` from `learnings/stuck_review_v2_2026-07-23.md`. A pre-registration should
state its own prior and say what result would surprise it. As written, S1 is sold as "the gate that
decides whether anything else is worth a push" while the answer is largely on disk. Say so — it costs
nothing and it protects the negative result from being over-read later.

### [MINOR] ADDENDUM2 B3's second-hand finding, if it holds, contradicts §4's choice of Tycho as *the* artifact schema

B3 reports (second-hand, flagged unverified) that 2608.06370's ablation shows a **filesystem-based
store degrading 32%**. Tycho's artifact is exactly a filesystem store — `world_model.py` + `notes/` on
the workspace — while P1's persistent namespace is in-process. The addendum draws the right conclusion
for S2 but does not notice that the same result undercuts the proposal's "Tycho as the artifact schema"
framing for S5. Do the direct read before S5 is scoped; if it holds, the schema should be adopted as a
*type discipline* (State / transition / render / outcome / UNKNOWN) held in the live namespace, not as
a workspace-file layout. Also log, per the addenda: §6.5's arithmetic is one draw stale (1.0823 →
1.0773, immaterial as it is illustrative); §3(c)'s "field is near-empty" premise must be **withdrawn**
per B2 in those words, with (c)'s down-ranking resting on the generalisation rail instead; and B5's
"typed knowledge graph hurt retrieval −11.2pp (p=0.0007)" should be written into P3 as a design
constraint (keep the memory flat).

## Questions for the authors

1. Under the existing `simulate(state, action_id, x, y)` interface, what exactly is "state threading"?
   If it means closed-loop rollout, do you accept that L0 will score **below** the open-loop numbers
   already in `runs/ewm_dryrun/report.md`, and that the "expand beyond ~4" gate is therefore unreachable?
2. What is the per-game numeric criterion for "carrier"? Accepted-match ≥ what, at coverage ≥ what,
   aggregated how across the four recorded streams?
3. Given that no sim can emit `UNKNOWN`, what non-constant quantity does "coverage" denote at L0 — and
   if the answer is "none until L1", why is L1 gated behind L0 rather than the reverse?
4. Which engine version produced each of `war_eval_v1/v2/v3` and `gpt56_full`, and were the sims
   authored against that version? Without this, accepted-match confounds synthesis quality with drift.
5. Do you accept the r16 two-cause split (aliased vs frame-Markov-clean-but-wrong), and will L0 report
   carrier expansion separately on the two classes?
6. Will you restate the 91.7% figure as in-sample-with-selection rather than "held-out", in both §6.4
   and `tycho_portability_2026-08-08.md` §6.6?
7. If L1 is conceded to be a public-25-only asset with no private-LB value, what is the argument for
   spending the campaign's largest item on it, given the standing generalisation rail that was used to
   down-rank lane (c)?
8. Is the 12-game replay-harness extension inside the "M effort, 0 pushes" estimate, or is the S1 gate
   in fact scoped over 12 games?

## What I cannot judge

Kaggle push/quota economics, wall-clock feasibility of the 9 h / 25-game envelope, and the
`RLIMIT_CPU` per-game re-accounting risk in P1 (systems reviewer). The stationarity/watch-rule
statistics, the n=26 promotion arithmetic, and whether the ledger's variance structure supports any
draw-level inference (methodology reviewer). Whether a 27B actor will in fact trigger the namespace —
K4 is the right instrument and I endorse it, but I cannot predict its outcome, and ADDENDUM2 B3's
"zero open-weight models" caveat means no external result can. The RL/planning merits of the advisory
one-action-at-a-time hook (L2) versus whole-plan beam search. Anything about Tycho's or Prime Agent's
internals beyond what the parent files quote — I did not read those repos. The panel-continuity
consequences of the R24 reviewer-model change (ADDENDUM A5) are for the methodology reviewer; I note
only that I took no round-over-round comparison as input.

## Verdict: MAJOR-REVISION

To be explicit about what I am **not** asking to relitigate, so this verdict terminates rather than
recycles: **the lane selection ((a) with Tycho as schema, (b) as component, (c) on its own clock) is
correct and I ratify it; §1.1's A22 death record is sound and sealable; §6.1's byte-identical
trimmed-message invariant is the right structural guarantee and should be sealed verbatim; K4 at 0.15
is a genuine non-score falsifier and should be sealed verbatim; S1b (bank re-fire, pruning disabled) is
a sharp root-cause falsifier with its assets verified present (`duck_eval/warpack/bank_fire_validation.py`,
`runs/war_eval_v1/bank_fire_validation.json`, `warpack_patch.py::prune_trace` L164) and should be
authorised now, unchanged; P1/S2 should be authorised now, unchanged; and the §5.3 rulings should be
taken as proposed.** The revision I require is bounded to **S1 alone**, and to six lines:

1. Split S1 into S1a (re-report existing open-loop on-trajectory numbers) and a **2–3 sim abstention +
   hidden-state pilot** that actually tests the hypothesis; drop the claim that L0 measures coverage
   until abstention exists.
2. Pre-register per-game numeric thresholds for accepted-match and coverage, plus the aggregation rule
   across streams, plus mandatory reporting of the per-game across-stream range.
3. Report carrier expansion separately on the r16 aliased vs clean-but-wrong partition; pin and
   disclose the engine version behind each stream.
4. Scope the gate honestly to the 12 games that have replay streams, or price the harness extension.
5. Correct "91.7% held-out" to in-sample-with-selection in §6.4 and the parent file.
6. Carry the generalisation rail into §4 as the lane's central strategic risk on L1/S5, and withdraw
   §3(c)'s "field is quiet" premise per ADDENDUM2 B2 in those words.

None of these requires a new panel. All six are edits to §5/§6 and one table row, executable before the
free run starts. If the authors make them, S1 fires this week as scheduled and I would sign it off
without another round.

## Score: 6/10
