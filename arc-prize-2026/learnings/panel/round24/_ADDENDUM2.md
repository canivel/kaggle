# R24 PACKET ADDENDUM 2 — research sweep, circulated 2026-08-09

Not part of the sha256-hashed proposal body. Landed after `_ADDENDUM.md` was sealed.
Full record: `learnings/sweeps/research_2026-08-09.md`.

---

## B1. The 08-07→08-09 arXiv window is genuinely empty

The API frontier is 2026-08-06 and arXiv does not announce Saturday or Sunday; the newest
announcement (Fri 08-07) is the 08-06 submission batch, which the 08-08 sweep already covered.
Next real batch: Monday 08-10. **A quiet window is the honest result — no new-paper pressure
on today's decision.**

## B2. A correction the panel must act on: the proposal's §3(c) premise is false

The 08-08 sweep asserted that the banking/replay field "produced exactly one hit… nothing new."
That is **wrong**. Re-enumerating the 08-05/08-06 batches surfaced a **~20-paper
skill-library / self-evolution cluster that is logged nowhere in `learnings/`.** R24 §3(c)
leans on the "field is quiet" premise and **must withdraw it.**

**But the correction does not promote lane (c).** The cluster is *cross-task skill libraries*,
not *within-game trace replay* — a different object. And its headline result cuts against
naive banking: **2608.05810 (VaG) finds skill accumulation is non-monotonic and irreversibly
harmful past a pool-size threshold.** Lane (c)'s down-ranking therefore survives, but it now
survives on the **generalisation-rail objection**, which this sweep did not touch — not on a
false claim that nobody is working in the area. Reviewers should require the proposal to say
so in those words.

## B3. The decision-relevant new item: 2608.06370, "The Bitter Lesson of Tool Calling"

Programmatic tool calling (Python stubs) beats JSON tool calling in **11 of 14 models**, and —
the load-bearing part — **improves +5.5% under context flood, where JSON degrades −2.3%.**
That is the **opposite sign** to everything A22 measured about our context pressure, and the
duck's existing single-`python`-tool shape is already the favoured form. **ADOPT (design):**
it independently supports the substrate half of lane (a).

**Caveat that must be stated aloud at panel: the study contains zero open-weight models.** The
§3(a) weak-model hole is **not** closed by it. This is exactly why `namespace_reuse_rate ≥ 0.15`
(K4) remains the right instrument — it measures whether *our* 27B actually uses the substrate,
which no frontier-model result can answer for us.

**[SECOND-HAND — needs a direct read before it is relied on]** the same ablation is reported to
show a **filesystem-based store degrading 32%**. If that holds on direct read, it favours the
**in-process persistent namespace (P1) over Tycho-style workspace files** — i.e. it supports S2
as currently designed. Flagged as unverified; do not let it into a sealed gate at this round.

## B4. 2608.05906 (MERIT) amends lane (b)'s blocker

Training-free dual-polarity memory — verified corrections **plus observed unsuccessful
directions** — demonstrated on **Qwen2.5-7B**, i.e. in weak-model range. Lane (b)'s blocker
changes from *"infeasible without training"* to *"feasible, thin evidence"*: +3.45pp, on
text-to-SQL, and it needs an oracle we do not have. **Stays a component arm** — the amendment
is to the reason, not the rank.

## B5. 2608.06196 puts a number on the de-rating rule, and kills one P3 design option

Self-authored query benchmarks inflate by **up to 44pp** — a quantitative floor under the
§5.3.2 provenance de-rating ruling the panel is being asked to adopt as standing policy.
Same paper: a **typed knowledge graph over skills *hurt* retrieval by −11.2pp (p=0.0007)**.
Actionable: **do not build a relational/typed graph for P3**; keep the memory flat.

## B6. Quiet elsewhere, and one backfill gap

Nothing new in **ARC-AGI-3** (newest remains 2608.04066), **test-time learning**, or
**compaction theory** — the third consecutive quiet compaction sweep, which makes §5.2's death
record **sealable as written**. Backfill gap: **2607.20709 (OO Agents, 07-22)** is untracked and
may be a fourth convergent team, but it publishes **no ARC-AGI-3 numbers** — MONITOR only, and
it must not be counted toward the "three-team convergence" argument in §2.1.

## B7. One free addition to already-planned work

Fold a **code-well-formedness pre-mortem** into the transcript-forensics pass the proposal
already schedules in §5.4 (literal-`\n` and subprocess syntax errors in the duck's existing
`python` tool calls). This is the documented failure mode that sank programmatic tool calling
for 3 of 14 models in 2608.06370. **$0, zero pushes**, and it directly de-risks P1's most
likely null mode.

## B8. Declared coverage gaps

The arcprize.org leaderboard fetch **failed** (client-rendered page), so the official-LB
counterweight figure (Opus 5 = 30.2%) is **≥2 days stale** — it is still the right counterweight
to quote, but the panel should not treat it as refreshed today. One search returned 152 hits of
which 40 were read.

## B9. Sweep's own verdict

**Lane ranking (a) > (c) > (b) UNCHANGED. S1 → S1b → S2 sequence UNCHANGED.**

## END OF ADDENDUM 2 ##
