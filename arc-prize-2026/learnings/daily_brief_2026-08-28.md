# ARC-AGI-3 — DAILY BRIEF 2026-08-28 (iterate session)

**Slots: 1 of 2 spent (seed arm pushed 08:35).** GPU $0. Submissions by this
session so far: 0. Registry: exp 61, 62 ADMITTED. Panel: none — Friday, and the
07-27 restructure puts full panels on Sundays only.

---

## 1a. RESULT DEEP-DIVE — last night's draw

**1.26** (`submission.parquet`, 2026-08-28 00:11:01, COMPLETE), the deliberate
Arm-0 field-floor head set by the 08-27 session.

**Pre-registered expectation vs outcome.** The 08-27 head rationale did not
predict a score; it predicted *information*: "this draw tightens the estimate we
will actually select on." That is exactly what it bought and nothing more. The
certified field floor is now **n=7, mean 1.4686, sd 0.2897**
(1.59/1.58/1.63/1.16/1.92/1.14/1.26). The config mean **fell again** — 1.5760 (n=5)
-> 1.5033 (n=6) -> 1.4686 (n=7). Three consecutive downward revisions of the config
we intend to select on.

**Against the null.** The corrected null ledger is **n=37, mean 0.9316, sd 0.1771,
bar 1.089** (see §3 — the bar was wrong this morning). 1.26 clears the promotion
bar, but a single floor draw was never a promotion candidate; the floor IS the
comparator.

**What it says about EXEC-WM.** Its 1.05 first draw is now **z = -1.44** against
its own comparator. The 08-26 "board scare" has fully dissolved: the floor itself
has since drawn 1.14 and 1.26, i.e. **twice within 0.21 of the arm the board was
supposed to have indicted.** The exec-WM prereg's decisive-kill clause must not
fire on board evidence, and did not.

**Honest read of the trend.** The floor mean sliding 1.576 -> 1.469 over three
draws is the more important fact than any single number, because
`project_arc_final_selection_rule` selects the final two submissions by CONFIG
MEAN. Our best *config* is measurably weaker than our best *draw* (1.92, a max of
seven) and the gap is widening as n grows. **Public max 1.92 is not what we will
be scored on.**

## 1b/1c. DISCUSSIONS + RESEARCH SWEEP

Performed at 06:14 by the community rail and delivered in
`learnings/community/brief_2026-08-28.md` (193 topics paged to exhaustion, six
new tracked kernels diffed line-by-line, external radar). Not repeated here.
My evaluation of its three decision handoffs:

1. **Thuitanium seed lever + TEETH — ADOPTED, built and pushed today.** See §2.
2. **"Burn no slot on making decisions cheaper" — ACCEPTED as a standing FORBID.**
   Their R35 (no fat tail; a cap at 8192 saves 0.98%, at 12288 saves nothing) and
   our own 08-27 trim refutation (4.9x worse per action) close the family on two
   independent instruments. Recorded so no future session re-proposes it.
3. **RESET replication — ALREADY DONE this morning** (commit `cdb234e`,
   `scripts/reset_probe.py`): **192/192 = 100.0%** perfect undo across all 25
   official games, byte-exact board comparison. Tennant's fidelity number is now
   OUR measurement, not a [V-doc] borrowed one. The one-wayness figure is
   deliberately NOT claimed as replicated: ours is "random walks that did not
   happen to return" (10/200 = 5.0%), a strictly weaker metric than his
   search-based "states with no way back" (90.1%). Same direction, different
   instrument, and the commit says so.

## 2. TODAY'S BUILD — THE SEED ARM (slot 1, `canivel/arc3-q38-seed-eval` v1)

Certified field floor, byte-identical except cell 5, plus ONE injected variable:
`LOCAL_ANALYZER_SEED = 20260828`.

**This is an instrument, not a score arm.** Our floor sd is 0.2897 and that sd is
why nothing we build is distinguishable at n=1. Two explanations are live and we
cannot currently tell them apart: **sampler** variance (seed pinning removes it)
vs **scheduler** variance (Spen, 08-27: game workloads get unequal time; seed
pinning does nothing to it). A seeded replicate set can split them.

**★ Treatment-can-fire PROVEN BEFORE the build — a campaign first.** Five times
this campaign an affordance has been shipped and then found unused or uninjected
(P1 notes 1.3%, P2 `attempt()` 10.73%, `animation()` advertised-not-injected,
RESET never advertised, schema-only affordances). This time the chain was read
first, in the pinned bundle source:

```
LOCAL_ANALYZER_SEED -> tool_agent.py:159 _get_env_int(..., -1)
                    -> tool_agent.py:1536 build_chat_payload(seed=...)
                    -> openai_compat.py: if provider=="vllm" and seed>=0: payload["seed"]=seed
```

Provider is `vllm`; the **-1 default means no seed reaches the wire today**; the
key is **ABSENT** from the pinned bundle (anchor count 1, seed count 0), so this
is an injection. The lever is real and currently off.

**TEETH adopted (community brief item 4).** The kernel refuses at setup — before
the benchmark starts — if the anchor moved, if a seed already exists, if
injections != 1, or if any of six untested analyzer variables drifted. All six
refusals are exercised as **negative controls**, because a guard that never
refused may be one that cannot.

**Gates at seal time:** q38-seed **55/0**; gate `--self-test` **13/0** (the
refusal controls still fire after I widened N3 to pin string literals);
q38-field **52/0**, no regression from the sibling registration; `seed_graft`
**18/0** incl. 6 negative controls; N4 determinism byte-identical; preflight
structural 6 ok / 0 warn. Pull-back verified: code byte-identical on Kaggle,
all 2 datasets + 1 model source retained.

**Sealed read** (`learnings/war_room/seed_arm_prereg_2026-08-28.md`): primary
statistic is **draw SD vs the floor's 0.2897 (n=7)**, one-sided F, alpha 0.05.
**n=4 committed BEFORE the first draw; a 2-draw read is FORBIDDEN.** Declared
MDE is honest and unflattering: n=4-vs-7 detects only a **halving of sd**, so a
null means "no large collapse", never "seed is inert". Pre-registered limitation:
vLLM per-request seed does not guarantee determinism under continuous batching
with 25 concurrent games, so even a true sampler effect may be attenuated — that
is stated now so it cannot be discovered later as an excuse.

## 3. ★ INSTRUMENT REPAIR — THE PROMOTION BAR WAS COMPUTED ON A SLIDING WINDOW

The community brief listed `scripts/ledger.py` as **broken on the Mac**
(`ModuleNotFoundError: scipy`). **That is refuted**: scipy 1.18.1 is in the repo
venv and the import is unguarded, so no silent-degradation path exists. The
report was a caller running system `python3`.

**But auditing it found a real and worse defect.** Run correctly, the ledger
returned **n=35, mean 0.936, sd 0.181, bar 1.0975** against a record of **n=37,
0.9316/0.1771, bar 1.089**. Cause: `kaggle competitions submissions` returns only
the **50 most recent** submissions, while the null pool is defined by a
description tag across *all* history. As non-ledger submissions accumulate, the
oldest ledger members fall off the end. **The null pool that sets the sealed
promotion bar for every arm was a sliding window that silently shrinks** — and it
shrinks in the direction that *raises* the bar, i.e. that rejects arms.

Fixed: page at the CLI maximum, plus a **fail-closed guard** that refuses to
report any number obtained at the page limit (a larger limit still fails silently
on the day it is reached). Post-fix the ledger reproduces **n=37, 0.9316/0.1771,
bar 1.089** exactly. Also pinned the decode to utf-8; the `cp1252` hardcode was a
Windows-console assumption that does not travel (it was *not* the cause — both
decodes parse this feed identically, checked).

**The standing campaign note that "the bar drifts — always re-read
runs/ledger.json" was at least partly this artifact rather than real movement.**
`feedback_audit_the_instrument`, and the fourth instrument defect in three days.

## 4. OPEN QUESTIONS

1. **Does seed pinning collapse our draw variance?** The whole point of the arm.
   Decides the required n for every future arm; needs 4 draws, so ~4 nights.
2. **Is the floor config mean still falling?** Three consecutive downward
   revisions (1.576 -> 1.503 -> 1.469). If it settles near 1.45 our selection-rule
   estimate is materially below our banked 1.92 and we should say so plainly.
3. **EXEC-WM v2, pre-authorised and unbuilt.** Yesterday's replay verdict (exp 61,
   admitted today) ranks **Gate B** (641 transitions, 26.8%, multi-object
   dynamics) above **Gate A** (+12 rules, 2 games made plannable, cheaper). Gate A
   is the honest first build. CPU-only, slot-free.
4. **Unclaimed free measurement:** sub-classify the exec-WM `residual` bucket
   (animation vs second object vs enemy). Free, and it decides Gate B's design.
5. **Two nightly rails.** The 08-27 addendum's ruling request. The Mac now owns
   the window (`com.arc.tick`); Windows tasks were disabled 08-27 per memory.
   Treat as resolved unless a double-fire appears in `submission_log.jsonl`.
