# TASK — GRAFT BUNDLE AUDIT (build-blocking verification for the 08-18 slot-1 arm)

You are working in the ARC-AGI-3 campaign repo `F:\kaggle\arc-prize-2026`. This is a
**read-only, verification-only** task. **DO NOT push any Kaggle kernel. DO NOT submit
anything. DO NOT spend money.** Both of 2026-08-17's kernel push slots are already spent.
Your entire output is a written report.

## CONTEXT

Yesterday we traced the field's 2.5+ leaderboard recipe (full writeup:
`learnings/war_room/conversion_trace_2026-08-17.md` — READ IT FIRST, sections 3, 8 and 9).
Summary: a CC0, default-OFF "score-mechanics graft stack" (`taaf_grafts`) is vendored inside
the Kaggle dataset `thtennant/taaf-kaggle-source-share-fork`. It plugs into the same
Tufa duck/TAAF harness our frozen fork already runs, via one `install(bm, flags={...})`
call, and attacks the action denominator of the per-level score formula
`min(115, (baseline/actions)**2 * 100)`.

The coordinator has AUTHORIZED one build for 2026-08-18 slot 1: our frozen fork +
the graft stack with `banking`/`transfer`/`shortcircuit` ON, incumbent Qwen3.6 engine,
ONE variable, SCORE-primary preregistered read.

**Your job is the verification that decides whether that build is a one-variable
experiment or a confounded mess.** A local copy of both bundles may already exist at:
`C:\Users\dcani\AppData\Local\Temp\claude\f--kaggle\62c35e7c-0d05-4da2-99b0-f9b400a45a97\scratchpad\conv_trace\`
(`taaf_banking/` = thtennant fork; also `taaf_jakob/`, `taaf_chew/`, `taaf_poby/`, `tennant_arc3-duck-v12/v18/v19/`,
`kevin250304_arc3-duck-v9b-recovery-banking/`). If a bundle you need is missing, download it
yourself with `uvx --from kaggle==2.0.0 kaggle datasets download -d <ref> -p <scratchdir> --unzip`
(read-only operations are fine). Our frozen fork's own stock bundle ref is
`jeroencottaar/taaf-kaggle-source-share`.

## THE FIVE QUESTIONS — answer each with evidence, not impression

### Q1 (DECISIVE) — Is the fork's harness byte-identical to the stock bundle?
Recursively diff `thtennant/taaf-kaggle-source-share-fork` against
`jeroencottaar/taaf-kaggle-source-share`. Report:
- files ONLY in the fork (expected: the `src/taaf-grafts/` tree)
- files ONLY in stock
- files present in BOTH but with differing sha256 — **this is the critical list.** For every
  such file, show the actual diff (or a precise characterization if large).
Use sha256 over every file in both trees; give exact counts. If the harness proper
(`ARC3-Inference`, `tufa-arc-agi-framework`) differs at all from stock, the "one variable"
claim is FALSE and you must say so plainly and quantify the delta.

### Q2 (DECISIVE) — Which ENGINE does the fork's bundle serve?
Compare `setup_commands.json` between the two bundles byte-for-byte, and any
vllm/serve/model-path config they reference. Our incumbent engine is Qwen3.6
(`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`). There is a parallel Qwen3.8 channel in the
field (`jakobbrggen/*`). **If the fork's setup expects Qwen3.8 weights or a different
snapshot slug, the arm silently changes TWO variables (engine + grafts) and the build must be
redesigned.** State exactly which model path / snapshot each bundle's setup commands resolve,
and whether the fork's setup works unmodified against our attached Q36 snapshot.

### Q3 — The `install()` contract.
Read `src/taaf-grafts/taaf_grafts/__init__.py` (and `composite.py`, `solver_base.py`).
Report verbatim: the exact `install()` signature; the complete list of valid flag names and
their default values; what happens on an UNKNOWN flag name (silently ignored, or raises?);
the documented failure/degradation behaviour; and whether any flag IMPLIES another
(the trace claims `transfer` implies `banking` — verify). Confirm every flag defaults OFF.
Quote the code, do not paraphrase.

### Q4 — The bundle-discovery collision (a real hazard I found in our own notebook).
Our frozen fork's cell 6 does:
`for marker in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"): return marker.parent`
— it takes the FIRST match and `rglob` order is not guaranteed. Determine:
(a) does the fork bundle ALSO contain a `taaf-kaggle-bundle.json` marker? (b) therefore, if we
attach BOTH the stock bundle and the fork, is `BUNDLE_DIR` ambiguous?
(c) Recommend the correct resolution and justify it — the two candidates are
**(i) REPLACE the stock dataset_sources entry with the fork** (clean single marker, but only
valid if Q1 says the harness is identical) versus **(ii) attach both and pin BUNDLE_DIR
explicitly**. Note that cell 8 also iterates `sorted((bundle_dir/"src").iterdir(), reverse=True)`
to build sys.path — check whether `taaf-grafts` is importable as `taaf_grafts` under that
logic (note the hyphen/underscore mismatch: directory `taaf-grafts`, package `taaf_grafts`)
and say exactly which sys.path entry makes the import work.

### Q5 — What do the public kernels actually set, and does `transfer` depend on an unverified premise?
From the local copies of `thtennant/arc3-duck-v12/v18/v19` and
`kevin250304/arc3-duck-v9b-recovery-banking`, extract the VERBATIM `install(...)` call from each
(with its flags) and the kernel's dataset attachments. Then read `transfer_solver.py` +
`family_store.py` and state precisely what premise `transfer` relies on
(the trace says: "the 110 competition runs are 25 public games cloned round-robin, sharing one
process") and **whether anything in the bundle or our own past run logs VERIFIES that premise**,
or whether it is an unverified assertion. Also report whether `banking` alone is safe if
`transfer`'s premise is false, and flag any behaviour that could REDUCE our score
(e.g. replays consuming scored actions, divergence handling, interaction with the 8h wall-clock
budget) — we need the downside named, not just the upside.

## OUTPUT

Write your report to `F:\kaggle\arc-prize-2026\duck_eval\graft\bundle_audit_2026-08-17.md`.

Requirements:
- Tag EVERY load-bearing claim with provenance: **[V]** verified by direct read/sha this
  session · **[V-doc]** verbatim claim inside a verified artifact (their words, not
  reproduced by you) · **[INF]** inference · **[UNK]** unknown. This discipline is mandatory
  in this campaign; an untagged claim is treated as unusable.
- Lead with a **BUILD VERDICT** section: `GO-AS-DESIGNED` / `GO-WITH-AMENDMENT` (state the
  exact amendment) / `NO-GO` (state the blocker). Then the per-question evidence.
- Include a short **EXACT CELL-12 PAYLOAD** section: the precise Python we should paste into
  cell 12, and the precise `dataset_sources` list for `kernel-metadata.json`, given your findings.
- If you cannot verify something, write **[UNK]** and say what would verify it. Do NOT guess,
  and do NOT smooth over a discrepancy — this campaign has been burned repeatedly by
  instruments and summaries that were internally consistent but wrong. A negative or
  blocking finding is a SUCCESSFUL outcome for this task.
