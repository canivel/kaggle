# DAILY BRIEF — 2026-08-25

**Slots spent:** 0 kernel pushes of 2. **GPU spend:** $0.
**Tonight's queue head:** `canivel/arc3-q38-field-eval` (certified field floor, `trusted-fork`) — correct and armed.
**Headline:** the day's slot-1 arm was **killed by its own gate before it cost anything**, and the
build that replaced it caught a bundle-drift defect that would have burned a slot as INFRA DEATH.

---

## 1a. RESULT DEEP-DIVE — the 1.92 draw

The morning-check session already logged the mechanics; this is the interpretation, not the number.

`submission.parquet 2026-08-25 00:07:10 — AUTO-REFILL FIELD-FLOOR filler — COMPLETE — 1.92`.

- **Pre-registered expectation: met, and unremarkably so.** The field-floor config is now **n=5,
  mean 1.5760, sd 0.2713**. A 1.92 as the **max of five draws** from that distribution is a ~31%
  event. It is a **max, not a level**, and it licenses no change of plan. Every comparator today
  was re-based to 1.5760 before use (coordinator rule 0).
- **It arrived unattended**, which is the real result: it confirms the 08-24 queue root-cause fix
  (empty `pending` + parked pathsafe ⇒ Arm 0 auto-arms the certified floor with no hand-edit).
- **Board effect:** Score 1.92, **#273 → #146 of 2526**. Gold 2.65, prize 2.88 — the selection
  statistic moved 1.4900 → 1.5760, roughly an eighth of the way to a line that itself rose.
- **Instrument note carried forward, unchanged:** `runs/ledger.json` still reads `latest_date
  2026-08-20` and tracks the **retired frozen-fork null** (n=37, mean 0.9316, sd 0.1771). It is
  **not** the live statistic. Re-derive at selection time; never quote it from memory.
- **Independent corroboration arrived today** (see §1b): another competitor reports the *same
  submission* scoring **2.11 then 0.89**. Our own spread is narrower than theirs. This is the
  strongest external support yet for `project_arc_final_selection_rule.md`.

---

## 1b + 1c. SWEEP

Full detail in **`learnings/sweep_2026-08-25.md`**. Condensed:

- **Discussions (2 new posts).** `737230` — a competitor reports an identical submission scoring
  **2.11 then 0.89** (range 1.22, wider than our whole n=5 series). **ADOPT as corroboration** of
  the config-mean selection rule; nothing to build. A commenter independently wonders whether teams
  are "improving their **memory management**" — outside convergence on our P1/P2 lane. `737227` —
  a rerun stuck "Running" >11 h; **IGNORE**, platform UI, our lane is COMPLETE and healthy.
  **No post published a bundle, an ablation, or a reproducible method.**
- **Research — the priority item came back NEGATIVE, and that is the useful part.** AVO's
  supervisor specification **does not exist publicly**. Checked the NVIDIA blog and arXiv
  **2603.24517** (abstract + full HTML): **no trigger conditions, no thresholds, no window sizes,
  no redirect content**, and the paper explicitly concedes *"this experiment does not isolate its
  individual contribution"* — i.e. **no ablation**.
- **And the counterexample that actually reweights the program: VISTA reaches 100% (all 183 levels,
  7,542 actions, Opus 5) with NO supervisor at all** — no stagnation detector, no replanning
  trigger. What it has instead is trajectory memory (`inspect` / `read_pixels`) and a
  continuation-state written at the context limit.

**⇒ S1 is DEMOTED.** Its charter was *"adopting a specified mechanism beat inventing one."* There is
no specification to adopt, and "the supervisor is the largest single gap" now has a clean
counterexample. **The memory lane (P1/P2) is the better-supported lane** — it is what both 100%
systems share, and it is where our own measured defect lives (97.64% of generated content goes to
the hidden channel and never enters the carried world model).

---

## 2. C3 — REFUTED ON MECHANISM, BEFORE IT COST A SLOT

Full read: **`learnings/war_room/c3_gate_read_2026-08-25.md`**. KAOS **exp_id 47**, verdict
admitted to the public registry (`tb1:4401f1f2…`).

C3 was sealed on 08-22 on this claim: *"it only stops **discarding completed generations**."*
The ordered gate was to prove discard-vs-truncation **in the shipped harness before building**.

**Verdict: WORK-PRESERVING — neither discard nor truncation.** [V, file:line]
- The yield predicate is checked at the **top of the tool-step loop before any request is issued**
  (`tool_agent.py:2168-2170`), and otherwise only **after** a response is fully received (`:2288`)
  or a tool dispatch returns (`:2353`). It never runs mid-generation ⇒ **no truncation**.
- On yield, `preserve_history` stays **True** (False only on exceptions, or a yield landing
  mid-batch of a multi-tool-call response), and the `finally` block at `:2400-2403` **commits the
  completed work to `self._history_messages`**, which retains the most recent 30 assistant turns
  (`:173`, `:2037-2054`) ⇒ **no discard**. The next invocation resumes from it.
- Rail defaults confirm the arm was live and unbounded otherwise: `kaggle.py:114`
  `YIELD_SECONDS=60`, and `TOOL_STEPS=0` ⇒ **unlimited** tool steps.

**The sealed conditional was "if discard is confirmed". It is not. So C3 takes no slot.**

**Artifacts** (`runs/cad_field_0825.json`, all 25 runs reconciled): `tail_tokens_no_action` =
**282,209 / 2,103,403 = 13.4%**; median **242.9 s per acting turn** against a 60 s yield, with
**25/25** games above it ⇒ the yield fires **~4× per acting turn**. That is the mechanism working,
not waste. Direct yield counts are **NOT RETAINED** (the analyzer transcript is not in the pull;
0 occurrences of the status string) — stated rather than estimated.

**Residual, priced honestly:** all that 60→150 s buys is fewer control-returns (≈4.05 → ≈1.62),
an **input-side** saving already served by `--enable-prefix-caching`. And it carries a
counter-effect the seal never priced: every game ends `gave_up` on the 7,920 s clock
(**7,925 s/game, 25/25**), and a 150 s window abandons up to **~2.5× more in-flight work** at that
terminal boundary. Small, **ambiguous in sign**, far inside the comparator (mean 29.0, pooled seed
sd 2.80). **NOT WORTH A SLOT.**

---

## 3. P2 PROMOTED TO SLOT 1 — BUILT AND SMOKED, DELIBERATELY NOT PUSHED

With C3 dead and today's sweep favouring the memory lane, **P2 takes slot 1**. Its gate (P0.1,
RESET returns to level start) was already discharged on the real simulator on 08-22.

**★ A bundle-drift defect was caught before the push, not after.** The prereg's `tool_agent.py`
anchors were verified against the vendored **`bundle_20260815`**, but the vehicle (certification
item 4) mounts **`anim-20260807`**. Those two `tool_agent.py` **differ by 233 diff lines** —
different behaviour-flag architecture, and a different `_PYTHON_TOOL_DESCRIPTION`, which is itself
a declared P2 anchor. Building to the prereg verbatim would have tripped the runtime `count == 1`
assert and died as a **LOUD INFRA DEATH**, costing the slot and ~8 h. Re-verified in the **vehicle**:
`_PYTHON_TOOL_DESCRIPTION = (` **=1**, `def _dispatch_tool` **=1**, `current_level` **=10** (needs a
tighter anchor, line 1402).

**★ A design simplification that removes most of the remaining drift risk.** The prereg assumed
`attempt()` needed a host-side handler wired through `tool_agent.py`. Reading the shipped sandbox
protocol shows it does not: `action(...)` is already a complete round-trip primitive, and `RESET`
is always legal (`action_names.py:14`; `taaf/game.py:184` — *"Legal action ids, with RESET (0)
always present"*). So **`attempt()` composes entirely from `action()` inside the child process** —
no new message type, no host handler, no `tool_agent.py` change for the episode machinery. It lands
only in `python_tool_sandbox.py`, which is **byte-identical across both bundles** (md5
`465f3e4fb9b1`) ⇒ **zero drift risk on the episode leg**.

**Delivered today:** `duck_eval/p2/p2_patch.py` + `duck_eval/p2/p2_smoke.py` — **18/18 PASS**,
executed inside the **real sandbox subprocess** against a scripted environment:
sealed params H=4 / K=5 / cap=40 · stops the instant the level clears · **RESET-after-WIN refused**
(engine full-resets after WIN) · post-RESET frame matches level start · cap enforced · RESET-in-
sequence refused · stock `action()` unregressed · **drift negative control fires**.

**Deliberately NOT pushed.** Remaining before it earns the slot: (a) the stuck-trigger leg (H=4
bookkeeping + `retry_mode: on, episodes_available: K` on the tool result), (b) `attempt()` announced
in `_PYTHON_TOOL_DESCRIPTION`, (c) the `[p2] reset semantics OK` boot check, (d) notebook cell at
position 6 + preflight + local_gate + pull-back verify. Pushing a half-built patch is the exact
failure class this campaign keeps paying for; the slot is worth more than the day.

---

## 4. HOUSEKEEPING

- Kernels `arc3-q38-field-eval` and `arc3-p1-notes-eval`: both **COMPLETE**, both already pulled/read.
- Queue: 1 pending, head = certified field floor, `trusted-fork`. Daemon fires 18:37. **Never empty.**
- Lane locks: `p2-reset-retry` registered to this session with the re-anchor warning inline.
- `kaos bench rejections` at session start: **empty** (nothing previously rejected to avoid re-proposing).

---

## 5. OPEN QUESTIONS

1. **Does the P2 stuck trigger fire at all?** D1 requires ≥15/25 games entering `retry_mode`.
   `hard_noop_guard` **has never fired in 5,255 actions** — a guard that cannot fire is this
   campaign's signature defect (`feedback_guard_never_fired.md`). The H=4 counter must be tested
   against **retained `benchmark.json` histories** before the push, not after.
2. **D2 is the real risk, not D1.** ≥25% of retry-mode turns must actually **call** `attempt()`.
   The affordance is advertised in the tool description — and `feedback_advertise_where_model_reads.md`
   records that a schema-only affordance drew **1.3% use against a 30% bar**. Measure USE, not delivery.
3. **Should S1 be retired rather than demoted?** With no AVO spec and a 100% counterexample that has
   no supervisor, its remaining case is our own telemetry (45.2% immediate repeats; 88% of wallclock
   after last clear). That is a real defect but a different arm from the one that was chartered.

---

## 6. PROCESS DEFECT — MOSTLY MINE, NOT KAOS'S

Both KAOS agents spawned today returned nothing usable. My first diagnosis was that `kaos run`
agents are structurally sandboxed away from this repo. **I tested that claim before letting it
stand, and it is WRONG.** The correction matters because it changes tomorrow's actions.

**Probed directly, under the invocation the 08-24 memory prescribes:**

| probe | invocation | result |
|---|---|---|
| `fs-probe-0825` | `uv run --project /f/kaggle/kaos kaos run … --config-file <abs> --db <abs>`, launched **from the project dir** | **reads the repo fine** — returned `238` lines and `H_STUCK_TURNS = 4`, both correct |
| `web-probe-0825` | same | **`TOOL BLOCKED`** — `WebSearch` genuinely unavailable |

So the two halves separate cleanly:

- **Filesystem access is NOT a KAOS defect — it was my invocation error.** I spawned with
  `cd /f/kaggle/kaos && uv run kaos run …`. That chdir is precisely the failure
  `feedback_kaos_improvements.md` root-cause **#3** already documents (*"THE AGENT IS SANDBOXED TO
  ITS CWD, AND `uv --directory` SETS IT"*), together with its fix (`uv run --project`, stay in the
  project dir, pass `--config-file` and `--db` absolute). **The 08-24 session's fix was sufficient.
  I simply did not apply it** — and the agents' own "isolated virtual filesystem" wording led me to
  read a self-inflicted scoping error as an architectural one.
- **The web-tool gap is real** and survives the correct invocation. **STEP 1b/1c sweeps cannot be
  delegated to KAOS** and must be run directly until `WebSearch`/`WebFetch` are granted.

**The cost was still concrete:** the C3 gate was ordered **08-22**, assigned to a KAOS agent on
**08-24** (`c3_yield_verification_2026-08-24.md` — **the file does not exist**), and re-assigned
today under the same bad invocation. It sat undischarged for three days while C3 stood as slot-1;
it took ~20 minutes to answer directly, and ~30 seconds to prove the delegation *could* have worked.

**The lesson survives the correction, sharpened.** *A verification assigned to an instrument that
cannot reach the evidence is indistinguishable from one never ordered* — and today the instrument
**could** reach the evidence; nobody checked. **Rule adopted: before delegating a check, run a
ten-second probe proving the agent can reach the evidence, and read the existing memory on how to
invoke it before inventing a diagnosis.** Related: `feedback_audit_the_instrument.md`,
`feedback_guard_never_fired.md`.

What in KAOS demonstrably works and was used today: `kaos experiment log` + `bench
harvest/validate/push` (exp 47 logged and **admitted**), and — as of this probe — tool-using
forensic spawns, provided they are invoked correctly.
