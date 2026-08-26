# ARC-AGI-3 daily brief — 2026-08-18 (Tuesday)

## 1a. RESULT DEEP-DIVE

### Overnight draw: **1.01** — interior, uninformative, sixth straight fallback-only day
`canivel/arc3-duck-repro` v3 (frozen-fork auto-refill filler), COMPLETE. Ledger re-derived from the
API: **n=35, mean 0.9403, s 0.1541**, z(1.01) = **+0.46** (interior), trailing-4 0.89 → 0.9675,
promotion bar **1.0778** (*drifts — re-read `runs/ledger.json` at prereg time, never cache it*).
Record stays **resolved-STATIONARY**; public max unchanged at **1.33 since 2026-07-18**.
**Pre-registered expectation met:** a frozen fork should draw from its own distribution, and it did.
No information about any mechanism. **Never read against ×1.10 — this is a single draw.**

### ★ Today's two pushes both died before the arm ran — and the cause is not ours
**`arc3-graft-floor-eval` v1 (slot 1, 10:47Z) and v2 (slot 2, 12:28Z): both INFRA DEATH at t≈6-7 s**,
identical, in the **STOCK** wheels-install cell (papermill `In [2]`, code ordinal 1, **not** one of
our modified cells `[2,6,12]`). `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels`
was **absent at runtime**. Per sealed prereg §4 this is **INFRA DEATH — never a NULL, never a HARM**;
the mechanism is untested in either direction and the arm's own code never executed.

**The morning's "transient platform instability" read DID NOT SURVIVE:** a transient does not
reproduce byte-identically 1h41m later on a clean API.

Hypotheses killed with evidence — transient (reproduced) · fresh-slug defect (`arc3-q38-low-eval`,
also fresh, mounted fine and died later at `In [4]`) · input-size cap (the *working* `arc3-duck-repro`
attaches the same 35.9 GB engine) · bad metadata (remote metadata **identical** to two kernels that
mount correctly; the sole delta is the bundle dataset ref).

**★ Leading explanation — platform-side input-mount regression — corroborated INDEPENDENTLY:**
`matthewblakeward/notebook1d22107bd4`, different author, last run 09:23Z, is ERROR with
`FileNotFoundError: Attached dataset kehhill/gemma3-llm-cli was not mounted`. Public ARC kernels at
07:15Z and 08:33Z are COMPLETE. **Best estimate: the regression begins between 08:33Z and 09:12Z —
before both of our pushes.** *Honest limits: theirs is a dataset mount and ours a competition mount
(same subsystem, same root cause NOT proven), external n=1.*

**Escalation was pre-committed before the outcome was known and is binding: no third re-push.**
→ `learnings/war_room/graft_floor_infra_death_2_2026-08-18.md`, KAOS exp_id **16**, bench-admitted.

### Engine-eval v3 (the 08-17 misfire): ruled **DESCRIPTIVE / NON-INFERENTIAL**
It does **not** become PRIMARY-B's n=2 — never pre-registered, and the disposition was necessarily
decided with the data already read. Nothing rides on it (Q38 is REFUTE-2× decisive). What it *did*
deliver, free, because it is a same-arm **SAME-SEED** replicate:

- **★ the sealed σ̂ = 0.141740 lc/game (df 6) got its FIRST-EVER check and PASSED.** Replicates scored
  **21 vs 17 levels**; gap **4** against an implied difference-SD of 3.5435·√2 = **5.011** ⇒ **0.80σ,
  CONSISTENT.** The live arm's bands (**HARM ≤12 · NULL 13–26 · SIGNAL ≥27**) are **not** too tight.
  **No threshold moves.** *Converse, honestly: ~4 levels of pure nondeterminism is real, so a reading
  that lands adjacent to a boundary is not decisive — which the prereg already accounts for.*
- an independent replicate **reproduces REFUTE-2×** (17/25, dlc −0.0933/game, z = −0.57);
- **the first clean decode-rate on this rail** after three lanes failed to get one: **42.6 tok/s @
  concurrency 1, 312.8 @ concurrency 8** — **ABSOLUTE-ONLY**, no predecessor-engine point exists, so
  **no cross-engine speed claim may be built on it**;
- effort pin certified bound.

→ `learnings/war_room/q38_engine_v3_disposition_2026-08-18.md`, KAOS exp_id **15**, bench-admitted.

## 1b. DISCUSSIONS SWEEP
One new topic since yesterday's cutoff (nothing at all from today).

- **735662 "qwen3.8-27B model vs dataset"** (Rahul Ray, 08-17 13:03, no substantive replies) — asks
  whether a Kaggle **Model** input and a **Dataset** input of the same weights differ.
  **IGNORE as a plan change; ADOPT-as-corroboration.** We resolved this on 2026-07-26: the Model-mount
  API **silently drops** `model_sources`, so weights ride as **datasets**. And our own sealed
  **REFUTE-2×** says the engine is not the lever anyway. Recorded only as evidence the field is still
  fiddling with engine plumbing.
- Re-verified: **the forum still discloses NOTHING about banking / transfer / grafts.**

## 1c. RESEARCH SWEEP
- **Prime Intellect "95.5%, beats the 95.4% human baseline"** — **IGNORE, unchanged: NON-COMPARABLE.**
  Public 183-level suite, not the private 8h-GPU no-internet rail.
- **MemoPilot / "From Player to Master" (arXiv 2606.08656)** — RL over memory updates. **PARK:** needs
  a training rail we do not have under `feedback_arc_zero_budget`. Thematically our exact problem.
- **Evo-Memory (2511.20857), AgentOdyssey, EvoAgentBench (2607.05202)** — self-evolving-memory
  benchmarks. **IGNORE for now:** benchmarks, not liftable mechanisms.
- Still **no published work** on prune-then-replay action-count banking in a scored environment.

## 1d. ★★★ THE DAY'S REAL RESULT — the 734843 memory defect is CONFIRMED on our rail, at zero slot cost
Yesterday's sweep flagged forum topic **734843** (Jason Feng): the duck harness may capture persistent
memory updates **only from visible output**, while the model puts them in **hidden reasoning**. The
handoff said this "needs no slot at all and should probably just be done." It was done. **It replicates.**

**Source-level proof** (`ARC3-Inference/inference/agent/tool_agent.py`, stock bundle):

```python
if content:                                             # lines ~1896 / ~1930
    self._update_summarized_knowledge_from_assistant(content)
```

`content` is the assistant's **visible** message. Reasoning is pulled separately by
`_extract_reasoning_text` (`message["reasoning"] / ["reasoning_content"]`) and is **never** passed to
the knowledge writer. Guarded by `if content:` ⇒ **a turn with no visible prose updates nothing.**
And `_update_summarized_knowledge_from_step_summary` is not a fallback — it **blanket-wipes**
`world_model/goal_model/action_model/recent_findings/open_questions/current_plan` on **any**
`level_transition`, `run_complete` **or** `game_over`.

**Behavioural measurement** (ours, on retained `solver_analysis` transcripts from engine-eval v3;
3 games chosen to span the outcome range — the best game and two zero-level games — 149 turns):

| game | turns | ASSISTANT segs | THINKING segs | ASSISTANT chars | THINKING chars | visible share |
|---|---|---|---|---|---|---|
| ft09 (best, 4/6) | 35 | 2 | 42 | 326 | 205,292 | **0.16 %** |
| cd82 (0/6) | 56 | 3 | 65 | 2,555 | 284,739 | **0.89 %** |
| m0r0 (0/6) | 58 | 33 | 69 | 15,313 | 262,220 | 5.52 % |
| **TOTAL** | **149** | **38** | **176** | **18,194** | **752,251** | **2.36 %** |

**97.64 % of all generated content goes to the hidden channel.** Turns with any visible assistant text:
**38/149**, and on two of three games it is **~5 %**. *(m0r0 is a genuine outlier at 57 %; report the
spread, not just the mean. Caveat: multiple assistant messages can occur within one `analysis_step`,
so these are segment counts, not strictly per-turn rates.)*

**Independently corroborated by the graft authors themselves** — `taaf_grafts/goalkeep.py` docstring,
measured on 4 games / 481 turns: *"a 27B under a tool-calling grammar emits prose in only **2–9 % of
turns** — it goes straight to the tool call"*, with carry of a non-empty working model at
**33/481 = 6.9 %**. Our 0.16 % / 0.89 % sit inside that range; our m0r0 does not.
**The two failure modes form a ratchet that only turns one way.**

### ★ Why this reorganises the board — the two fixes are COMPLEMENTARY, not redundant
- **`goalkeep` patches the WIPE** (`_update_summarized_knowledge_from_step_summary` +
  `_summarized_knowledge_lines`). It **does NOT touch the capture path.**
- **734843's fix patches the CAPTURE** (make the model emit its updates in visible output).

⇒ Our **graft-floor** arm carries goalkeep and attacks **half** the ratchet. The staged **obirdy**
candidate carries a prompt-only **visible-updates contract** and attacks the **other half**.

⇒ **Correction to yesterday's staging note:** it recorded that if obirdy outperforms, "the interesting
variable is the CONTRACT, not the engine." That now has a **measured mechanism** behind it rather than
being a guess — and the two arms are worth running as a **pair**, not as alternatives.

## OPEN QUESTIONS (for tomorrow / Sunday panel)
1. **Is the Kaggle input-mount regression over?** Free pre-check before any push: `kernels list
   --sort-by dateRun` + `kernels status` on the newest few public ARC kernels. **Do not re-push into
   an unmounted rail a third time.**
2. **H5 is weakened, not eliminated** — if graft-floor v3 dies again *while public kernels succeed*,
   the `thtennant` fork dataset becomes the suspect and the arm needs a different delivery route.
3. Should the visible-updates contract be added to the graft-floor arm directly (one build, both
   halves of the ratchet) or kept as a separate second arm to preserve one-variable attribution?
   **Recommendation: keep them separate** — we have never obtained a clean read of *either*.
4. Does `goalkeep`'s "hand back objective evidence" path partly compensate for the capture defect by
   re-deriving state from harness result keys? Checkable offline in `goalkeep.py`, no slot.

## HOUSEKEEPING
- **Slots: 08-18 = 2 of 2 SPENT.** Nothing further may be pushed today.
- **Queue: NOT EMPTY** — frozen-fork filler armed (`trusted-fork`); daemon clean.
- **`graft_push.sh` is date-guarded to 2026-08-18** ⇒ fail-closed for tomorrow **by design**.
- **Two instrument defects fixed today**: gate 1b had been **broken open since inception**; the arm
  now carries a fail-loud **MOUNTCHECK**. Rebuilt sha `79aa21fdbecbccf7`, smoke 36/36, scorer
  selftest 22/22, differing cells still `[2,6,12]`.
- KAOS exp_ids **14, 15, 16** logged and **all admitted** to the public registry.
