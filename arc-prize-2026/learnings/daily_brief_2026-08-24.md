# DAILY BRIEF — 2026-08-24 (Monday)

**Protocol:** STEP 1a (result deep-dive) + 1b (discussions) + 1c (research) merged per the canonical
daily protocol. 1b/1c were executed this morning by the community lane and are **not repeated here** —
the full sweep with per-item adopt/adapt/ignore is `learnings/community/brief_2026-08-24.md`
(13 discussion threads re-pulled, kernels sweep, AVO arXiv 2603.24517 + NVIDIA blog + VISTA read,
LB full download 2506 teams). This brief carries the interpretation, the day's gate, and open questions.
**No panel today** — full panel is Sundays only (process restructure 2026-07-27).

---

## 1a. THE RESULT, INTERPRETED (not the number)

**Draw: 1.16 COMPLETE** — `canivel/arc3-q38-field-eval` v1, 08-24T00:07:11Z, "ARM 0 NIGHTLY FLOOR".

**What it is: the certified field floor's first LOW draw, and it kills the tightening story.**
The config's four draws are now **1.59 / 1.58 / 1.63 / 1.16 → mean 1.4900, sd 0.2211**.

- **The pre-registered expectation was NOT met, and the miss is informative.** On 08-23 the three-draw
  spread of 0.05 supported a *suggestive* variance-tightening reading (χ² = 0.0446 on 2 df,
  P(sd ≤ obs | s = 0.1771) = 0.022, single comparison). That session explicitly declined to call it
  settled. **One draw took the realised sd from 0.027 to 0.221 — straight back onto the ledger's own
  s = 0.1771.** The caution was correct and the tightening is dead. This is the
  "price extremes as min-of-n" lesson landing on our own instrument: a 3-draw spread is a
  small-n order statistic, not a variance estimate.
- **No investigation is licensed.** Against the config mean 1.600 the draw is z ≈ −2.48 on ledger s;
  **as the min of 4 draws that is ≈2.6%** — unusual-ish, not anomalous. The run was healthy (COMPLETE,
  not an infra death). It still clears the frozen-fork promotion bar (1.16 > 1.089). No watch rule fires.
- **The delta that matters is not the board delta.** Public score is a MAX over submissions, so our
  board Score is **unchanged at 1.63**. But the **final-selection rule reads the CONFIG MEAN**
  (`project_arc_final_selection_rule.md`), and that fell **1.600 → 1.490**. The number the deadline
  decision actually consumes got *worse* on a night the board showed us flat.
- **Per-game / mechanism evidence: none new.** No new eval artifact landed; this was a redraw of an
  already-certified config, so it carries no mechanism information by construction. That is the point
  of Arm 0 — it is a floor, not an experiment.
- **Strategic read, unchanged and reinforced: redrawing this floor is not a path to gold.** The lane
  asymptotes near ~2.0 against a gold line now at **2.57** and a prize line at **2.88**. We are
  **#273→#274 of 2506** on an unchanged score — pure field drift, now *below* the top-250 line.
  A better floor does not close this; only a different ceiling does.

## 1b/1c. SWEEPS — see `learnings/community/brief_2026-08-24.md`

Carried forward here, only the items that bind on today's work:

- **[V] Tufa Labs +1.54 on ONE draw → 4.58 #1** — largest single-draw step of the campaign, by the
  authors of the harness we fork, with zero public artifacts. Method **UNKNOWN and undateable**
  (`LastSubmissionDate` is latest, `Score` is best). Top-3 pattern sharpened and confirmed:
  1 sub/day, multi-day flats, +0.6..+1.8 steps when a code change lands.
- **[V] AVO discloses NO quantitative supervisor triggers** — qualitative only. → S1 must derive its own
  thresholds from our transcripts. **Actioned today** (see §3).
- **[V] Both tracked bundles republished** (jakob 08-22, tennant 08-23 "(banking)" + `clockwatch.py`).
  → **Actioned today and DISCHARGED for our vehicle** (see §2).
- **[V] `clockwatch` graft is the public implementation of our own decision-budget finding**, but the
  distributor's team sits at **1.46**, below our 1.63 — implementation source, not evidence of effect.
  Coordinator ruling: fold into C3's design space, **do NOT co-build**.

## 2. TODAY'S GATE AND WHAT IS ALREADY DISCHARGED

Slot 1 is **C3 = yield-match**: raise `LOCAL_ANALYZER_YIELD_SECONDS` 60 → ~150 on the certified
field-floor vehicle. It is gated on the discard-vs-truncation question (ordered 08-22, still owed).

Discharged by this session, zero GPU, before any build:

- **BUNDLE RE-AUDIT (coordinator item 4) — DISCHARGED for this vehicle.** The floor mounts
  `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`; **all 75 files in that dataset are dated
  2026-08-07** (single distinct date, `datasets files --page-size 500`). The "jakob republished 08-22"
  flag refers to a *different* jakob slug (`taaf-kaggle-source`). **The C3 vehicle is not exposed to the
  bundle-generation confound that double-caveated the 08-21 Arm 3 read.**
- **THE YIELD IS ARMED — C3 modifies a live setting, not an addition.** `tool_agent.py:151` defaults the
  env var to `0.0` (= disabled), which had raised the possibility that C3 had no premise at all. It does:
  the floor's own bundle sets **`'LOCAL_ANALYZER_YIELD_SECONDS': '60'`** in `setup_commands.json`, so the
  tool_agent default never applies. (Same value in `duck_eval/taaf_bundle` and `bundle_20260815`.)
- **BUILD ANCHOR DE-RISKED OFFLINE**, exactly as C1's was on 08-23: the literal
  `'LOCAL_ANALYZER_YIELD_SECONDS': '60'` occurs **exactly once** in the bundle's `setup_commands.json`,
  so a C1-style `assert count == 1` will not kill the build. The only other timeout-shaped env value is
  `LOCAL_ANALYZER_TOOL_TIMEOUT: '30'`, which a targeted replace cannot touch.
- **RAIL GREEN.** `local_gate --self-test` **PASS 13/13**. No open kernel builds.
- **QUEUE CORRECT.** `pending` is **EMPTY** and pathsafe sits in `parked_insurance` — the 08-24 morning
  root-cause fix held, so Arm 0's auto-refill arms the certified floor by itself tonight.
  Re-verify at 18:00.

**Still open at time of writing: the gate itself** — does a yield-deadline turn DISCARD its deliberation
or PRESERVE it? DISCARD ⇒ seal and push. PRESERVED/UNRESOLVED ⇒ do not push, release the slot.

## 3. INSTRUMENT: THE KAOS SPAWN RAIL WAS BROKEN AND IS NOW FIXED

The KAOS-native mandate could not execute as written. Four defects stack, each of which alone reads as
"KAOS can't do this"; all four are now root-caused and patched (backups `*.bak-20260824`):

1. `fable-panel`/`opus5-panel` are `provider: agent_sdk` = **text-only, max_turns=1** — they cannot hold
   a tool, so no forensic or build task can run on them. Added `opus5-code` (`provider: claude_code`).
2. **The 60 s idle watchdog was hardcoded and unreachable from `kaos.yaml`** (the recorded ★ blocker).
   Now plumbed: `ModelConfig.idle_timeout` → yaml loader → `create_provider` → `ClaudeCodeProvider`.
   Default unchanged at 60.0, so nothing else moves.
3. **★ The agent is sandboxed to its CWD, and `uv --directory` sets it.** This is the same refusal that
   killed panel round R27 ("outside the allowed working directory `F:/kaggle/kaos`") — it was never a
   panel bug. Fix: `uv run --project <kaos>` (cwd stays in the project) plus absolute
   `--config-file` / `--db`.
4. **★ On Windows the agent's shell tool died on any non-ASCII byte** — `_shell_exec` ran
   `subprocess.run(text=True)` with no `encoding`, so the reader thread threw
   `UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d` on our own war-room notes (★ → σ ±).
   Fixed with `encoding="utf-8", errors="replace"`.
5. **The tool-call protocol only accepted one dialect.** The model emits
   `<parameter name="…">…</parameter>` closed by `</invoke>`; KAOS's regex demanded a JSON body closed by
   `</tool_call>`, so **the call silently never ran** and the agent returned its own raw XML as the
   answer. Parser now accepts either closer and parses `<parameter>` bodies; both dialects unit-tested,
   legacy JSON form still matches.

**This is a `feedback_guard_never_fired`-class finding about our own tooling:** R27's panel death was
attributed to the panel and is actually defect 3, and defect 5 makes an agent *look* like it answered
when it never ran a single tool. Verified working: agent conversation state now grows past 16 k chars
where dead runs sat at ~1.3 k.

## 4. OPEN QUESTIONS

1. **The gate:** discard vs preserved. Everything about slot 1 hangs on it.
2. **Would 150 s move the wall or clear it?** If yield-deadline turns are discarded, the design number
   must come from the distribution of elapsed time at the yield check, not from the median generation
   length alone. A longer yield also buys FEWER turns/game, and the marginal turn is worth ~21% of the
   average — the trade needs arithmetic before it needs a slot.
3. **`runs/ledger.json` is scoped to a retired config.** It tracks the frozen-fork null (n=37, latest
   08-20, mean 0.9316) and should not ingest field-floor draws — but the live selection statistic is now
   the field-floor series (n=4, mean 1.4900, sd 0.2211) and no instrument maintains it.
   `ledger.py` should key by config and derive the promotion bar from the ACTIVE config's series.
   **Carried, not actioned today** (slot-1 work has priority).
4. **Replication precision becomes load-bearing the moment we hold two certified configs within ~0.1.**
   At n=4 the sem is ~0.11. Budget the replicates rather than rediscovering this at selection time.
5. **P1 is parked pending a re-scope**, and its certification is unevaluable until kernel-log capture is
   fixed (0-byte logs on two independent pulls). Not today's work; do not let it age out silently.
