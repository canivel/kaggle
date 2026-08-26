# `scripts/local_gate.py` — the LOCAL pre-push gate

**Lane:** `local-rail`. **One command, run before every `kaggle kernels push`.**

```bash
uv run python scripts/local_gate.py --arm <name> [--notebook <path>]
```

Exit code `0` = PASS, `1` = at least one FAIL. `--json out.json` writes a machine
report; `--quiet` prints JSON only.

---

## 1. Why this exists

Kernel *builds* are free (30 GPU-h/week). What is scarce is **calendar days** and
**submission slots** — and what has actually been burning them is not GPU time
but **broken instruments**. Three times in two days a scorer was validated
against fixtures its own author wrote, and the fixture keys did not match the
keys the real taaf harness emits:

| # | date | reader | fixture key | REAL key | consequence |
|---|------|--------|-------------|----------|-------------|
| D1 | 08-21 | benchmark reader | `games` | `game_runs` | `n_games = 0` → **false INFRA DEATH on a healthy arm** |
| D2 | 08-21 | actions reader | `total_actions` (scalar) | `actions_per_level` (list) | `actions = 0` → the sealed wallclock-KILL fired on "ratio 0.00" |
| D3 | 08-22 | score reader | `score` | `final_score` | `mean_score = 0` |

*Internal consistency is not correctness.* Every one of the three was green on
its own selftest at the moment it was wrong. So the gate's first and largest
check group does one thing: **run every reader against REAL artifacts already on
disk and cross-check what it extracts against an independent direct computation
from the raw JSON.**

The gate itself is negative-controlled (`--self-test`): it deliberately breaks a
notebook and a reader — including verbatim reconstructions of D1/D2/D3 — and
asserts that it FAILs on each. A gate that has never been proven able to refuse
is a gate that may not be able to.

---

## 2. What it covers

### Group R — real-artifact reader certification *(priority 1)*

| check | what it does |
|-------|--------------|
| `R0.corpus` | indexes every real `benchmark.json` under `runs/` and `duck_eval/` **by path, never copied**; splits contract-clean from degenerate (aborted/legacy) artifacts |
| `R1.<reader>` | every directly-callable reader × every contract-clean real artifact, cross-checked against the oracle. Any mismatch, omission or **silent zero on a non-zero artifact** FAILs |
| `R1b.<reader>` | degenerate real artifacts (null `final_score`, never-finished runs) must be **refused**, not reported as confident numbers |
| `R2.<arm>` | each scorer's **full `score(run_dir)` pipeline** against its own arm's REAL pulled kernel output, metrics cross-checked against the oracle. A scorer that returns no metrics for its own arm FAILs |
| `R3.<arm>` | **reject tests at pipeline level**: 8 malformed `benchmark.json` shapes (including the D1/D2/D3 key shapes) dropped beside the arm's own real log must be REFUSED, never scored |
| `R3r.<reader>` | the same shapes against the bare reader function, advisory: records silent zeros and tolerated key fallbacks |
| `R4.historical` | the three historical defects, reconstructed, run against a real artifact — each must be **caught** by the R1 cross-check machinery |

**The oracle** (`oracle()` in the gate) shares no code with any scorer, accesses
every key strictly, and raises on anything missing or non-numeric. It is the
second opinion each of D1/D2/D3 would have failed against on day one.

Only the arm named by `--arm` (and its declared siblings) can FAIL R2/R3.
Another lane's broken scorer is reported as `OTHER LANE -- advisory` so that
lane X is never blocked from pushing by lane Y's instrument.

### Group N — notebook static gate

`N1` every code cell `ast.parse`s · `N1b` cell count matches the prereg ·
`N2` zero forbidden tokens for this arm · `N3` **flag literals match the arm you
asked for** (this is what catches "the staging dir holds a different arm's
bytes") · `N4` **builder determinism** — the builder is re-run twice with its
`OUT_*` paths redirected into scratch and the outputs must be byte-identical ·
`N5`–`N8` **compose `scripts/preflight.py` at the function level** so it runs
fully offline: `structural_checks` (K-series), `duck_diff_checks` (diff vs the
local base notebook against the prereg's declared cell indices), `host_gates`
(H1–H4), plus `kernel-metadata.json` sanity.

`preflight.py` is *wrapped, never reimplemented*. Its CLI is pull-based because
it is the **at-push** gate; the pre-push gate calls the same pure functions
against the local staged bytes. `N6a` first removes the arm's declared inserted
cells so preflight's cell-shape check applies, and `N6b` verifies the declared
non-code (markdown) diffs really differ.

### Group H — harness smoke without the 27B

A stdlib `http.server` speaks the OpenAI-shaped `/chat/completions` that the
harness's raw `requests.post` expects — **no litellm, no openai SDK** (the
harness itself only ever uses `requests`). The game is the **real competition
simulator** (`taaf.game_examples.ExampleGame`) driven by the **real
`HarnessSolver` + `ToolAgent`** out of the vendored 08-15 bundle, with the real
`arc_agi` / `arcengine` / `taaf` wheels. CPU only, ~20 s for all five scenarios.

| check | what it proves |
|-------|----------------|
| `H1` | the agent completes a scripted game end-to-end (2/2 levels, 6 actions) via native `tool_calls` |
| `H1b` | the outbound payload carries the `python` tool and a full system prompt |
| `H2` | the **text tool-call parser** works: a run with *zero* native `tool_calls`, only `<tool_call><function=python>` markup, still completes |
| `H3` | the **visible-capture contract**: a `World model:` line on the visible channel is carried into later requests' world-model block; the identical line delivered on `reasoning_content` is not. This is the exp-17 mechanism edge-2 exists to exploit, measured on the real agent bytes |
| `H4` | behaviour flags propagate end-to-end and change the resolved configuration — *a flag that cannot fire is an arm that cannot be measured* |
| `H5` | **the loop closes**: the artifact the real `taaf.Benchmark` emits is read correctly by the arm's reader. Ground truth generated by the library, not by us |
| `H6` | the known-noop guard is armed and transparent to a healthy run |

### Group A — arm matrix

`A1` the arm's own certification **accepts** its own real pulled artifact.
`A2` it **refuses** every other arm's real artifact — cross-arm negative
controls run against real data, not fixtures. Arms that share vehicle bytes
(a replication seed) are declared as siblings and excluded, because a control
that cannot help but fire is noise.

### Group P — P0 permanent instruments *(added 2026-08-22, weekend-prep lane)*

The compile of ARM P0's flagged follow-up (`perturn_program_2026-08-22.md`
§5.3 item 4). It is a **regression harness for two instruments other lanes'
verdicts now lean on**, and it runs on synthetic grids in-process plus the
sealed P0 evidence files — no GPU, no network, no writes.

`P0` the shipped `NoopGuard` and `duck_eval/p0/board_signature_fix` both load.
`P1`/`P2` the guard's **two independent defeat paths** under a ticking HUD,
reproduced separately: the record path (a full-grid `board_changed` hides
every no-op) and the match path (the full-grid key never recurs even when
recording is forced). Both must block **0/12**, reproducing the field's
0-in-1,630. `P3` the interior re-key **fires** (11/12). `P4` and **zero** false
blocks on a genuinely changing interior. `P5` the animation exemption survives
the re-key (the ft09/sb26 regression stays fixed). `P6` `HudMask` finds the
border strip, ignores mid-board activity, and degrades to byte-identical
shipped signatures before convergence. `P7`/`P8` the sealed P0.2/P0.3 and P0.1
evidence files are present and internally consistent — P8 is the gate that
P2's whole premise (RESET returns to the level start) still holds. `P9` the
**cadence instrument** (`duck_eval/cadence/cadence_instrument.py`) re-derives
the 08-22 BP35 diagnostic table from the real artifacts on disk.

Negative controls live in `--self-test`: **S12** reverts the P0.4 interior
re-key and requires `P3` to FAIL; **S13** poisons the cadence instrument's
expectation and requires `validate()` to report failures. An instrument that
cannot refuse is `feedback_guard_never_fired` all over again.

**Fixture note, learned the hard way:** the synthetic HUD strip must tick
**monotonically**. A wrapping strip lets the shipped full-grid key recur by
accident, and `P2` then measures the fixture instead of the guard.

### Group X — wrapped suites and do-no-harm

`X1` every scorer's own `--selftest`, as a subprocess. `X2` the arm's existing
deep suites (`private_smoke.py`, `graft_smoke.py`, `graft_bundle_check.py`).
`X3` **standing** suites that are not arm-specific because they guard shared
instruments — currently `duck_eval/p0/test_noop_guard_repro.py` (8/8); it
SKIPs gracefully when `pytest` is absent from the interpreter.
`X4` **do-no-harm**: `notebooks/`, `submission_queue.json` and
`runs/lane_locks.json` are hashed before and after; any byte that moved is a
FAIL against the gate itself.

---

## 3. What it CANNOT cover

This gate never seals a verdict. **Kaggle remains the only certification rail**
(env-mismatch confirmed 5×). A PASS licenses a *build*, nothing more.

Explicitly **not** covered, and still requiring a Kaggle build:

1. **Anything GPU-dependent.** vLLM start-up, the FP8 checkpoint loading, KV-cache
   sizing, tensor-parallel layout, CUDA/driver compatibility, OOM behaviour. The
   local box is an RTX 3080 **10 GB**; Qwen3.8-27B-FP8 is ~28.8 GB of weights
   alone. Nothing about the served model is testable here.
2. **Real-model reasoning quality.** The fake server returns *scripted* tool
   calls. H1–H6 prove the machinery carries a decision; they say nothing about
   whether the model would make a good one. Levels completed, score and the
   sealed bands are Kaggle-only quantities.
3. **Wallclock and throughput.** Tokens/s, the 12-hour window, per-level
   quadratic cost, concurrency of 28, the edge-1 wallclock-KILL arithmetic —
   all wallclock-bound and all measured only on the real rail.
4. **Platform mount layout.** `/kaggle/input/...` paths, which dataset version
   Kaggle attaches (it attaches **latest**, silently), model-source attachment
   (which Kaggle can silently drop — pull-back-verify at push), the wheelhouse,
   internet-off behaviour, the docker image's own package set.
5. **Build status and remote byte identity.** `preflight`'s T1/T4 legs (pull the
   kernel, read the latest build status) need the network and a pushed kernel.
   Run `scripts/preflight.py` at push time as usual — the two gates are
   complementary, not alternatives.
6. **Whether a hypothesis is true.** The gate audits instruments, not claims.

---

## 4. What the 5090 (~2026-08-28) will add

32 GB, so still short of the full 28-game production shape, but it moves three
things from "Kaggle-only" into local reach:

* **A real small/quantised model in the loop.** Swap the scripted stub for an
  actual served model over the same `vllm_client` interface. H1–H6 stay the
  same checks; the responses stop being canned, so prompt-format regressions,
  tool-call-parser drift against a real chat template, and reasoning/visible
  channel split ratios (the exp-17 97.6 % figure) become locally measurable.
* **Reduced-concurrency, short-context end-to-end runs.** 1–4 concurrent games at
  a 8–16 k window on a handful of games: enough to catch a harness-level
  regression (agent stalls, the noop guard mis-firing, transitions never
  queried) hours before a Kaggle slot, though **never** enough to produce a
  score comparable to the 25-game/32 k rail.
* **Cheap flag-direction screening.** Pairwise ON/OFF runs of a behaviour flag on
  a few games — a *direction* signal that can retire an obviously harmful flag
  before it costs a submission day. Bands and verdicts still come from Kaggle.

What the 5090 will **not** add: the FP8 27B at production window/concurrency,
the real KV pool arithmetic, wallclock parity, or the Kaggle mount layout. The
27 B/96 GiB shape is not reachable on 32 GB, so certification stays where it is.
Matched-GPU screening on RunPod (~$50 authorised) remains the only way to see
the production shape off-Kaggle.

---

## 5. Modes

| command | time | use |
|---|---|---|
| `--arm X --fast` | ~15 s | quick sanity while iterating (skips harness smoke, builder determinism, heavy suites) |
| `--arm X` | ~40 s | **the pre-push command.** Everything |
| `--all-arms` | ~2 min | coverage sweep across every registered arm |
| `--self-test` | ~17 s | negative controls on the gate itself. Run after touching the gate |
| `--corpus` | instant | print the real-artifact fixture index |

## 6. Adding an arm

Add an `Arm(...)` to `ARMS` in `scripts/local_gate.py`:

```python
"my-arm": Arm(
    name="my-arm",
    scorer_module="my_score", scorer_path=DUCK / "lane" / "my_score.py",
    scorer_arm_flag=None,                       # or the --arm value
    artifact=RUNS / "kernel_pulls" / "my_v1",   # real pull dir, once one exists
    notebook=NOTEBOOKS / "my-eval" / "arc3-my-eval.ipynb",
    kernel="canivel/arc3-my-eval",
    builder_path=DUCK / "lane" / "build_my_eval.py", builder_args=(),
    base_notebook=..., expect_diff_cells=(...), expect_inserted_cells=(...),
    expect_n_cells=N,
    forbidden_tokens=("litellm", ...),
    required_literals=(("MY_FLAG", "True"),),
    sibling_arms=(),                            # same-vehicle arms, if any
    extra_suites=(("my_smoke", (str(DUCK / "lane" / "my_smoke.py"),)),),
),
```

The arm is then covered by R2/R3 (once it has a pulled artifact), the full N
group, and the A-matrix in both directions. Builders must keep their
`build()` behind an `if __name__ == "__main__":` guard and write only through
module-level `OUT_*` `Path` constants — that is what lets `N4` redirect them
into scratch instead of touching live staging.

## 7. Known live findings (as of 2026-08-22)

* **`duck_eval/q38/q38field_score.py` INFRA-DEATHs every real artifact it will
  ever see.** `certify_runtime` treats `INFRA_SIGNATURES` as a hard certification
  gate and scans the whole log with a naive substring test. Two of those
  signatures — `Timed out waiting for vLLM server` and
  `Missing attached dataset path` — are **error-message literals in the
  notebook's own setup source, which Kaggle echoes into every kernel log.** They
  are present in all six real pull directories. The scorer's own selftest passes
  because its fixtures are hand-written logs that contain no echoed source.
  It is the only scorer that gates on these strings (`graft_score` uses them only
  as a diagnostic on the no-benchmark path), and the artifact it refuses,
  `q38_field_v1`, is the one whose **lc 28 / score 6.173 / actions 1639** is
  hard-coded as the comparator in `private_score.py` and `q38graft_score.py`.
  Owner: the q38-field lane. Fix direction: scope the scan to runtime output, or
  require a signature to appear outside any echoed source block.
* **`duck_eval/private/private_smoke.py` S2 asserts the staged variant is pure
  base.** It has been failing since the moment edge-1 was staged — the
  frozen-certifier landmine the prereg defused in the scorer but not in the
  smoke. Owner: the private lane.
