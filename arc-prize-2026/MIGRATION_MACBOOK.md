# Migration: Windows box → MacBook Pro (40-GPU-core, 64GB unified)

Written 2026-08-26. The repo travels via git (`git@github.com:canivel/kaggle.git`); this file is the checklist for everything git does NOT carry.

## What the Mac buys us

- **Full local Qwen3.8-27B serving** (weights ~28.8GB FP8-equivalent; 64GB unified fits weights + KV + OS comfortably at 8-bit, tightly at bf16). The 5090 (32GB, arriving ~08-28) was marginal; the Mac is not.
- Real model-in-the-loop iteration for the exec-WM agent: rule-mining prompts, constrained-JSON induction calls, cadence measurements — at local speed, no GPU-week budget.

## What the Mac does NOT change (campaign law)

- **Kaggle stays the ONLY certification rail.** Env-mismatch is confirmed 5×, and the Mac makes it worse, not better: no CUDA/vLLM on Apple Silicon — local serving is MLX or llama.cpp (GGUF), different kernels, different quantization (no FP8 on Metal → 8-bit or bf16), different sampler details, different tok/s. **Local = screening + delivery testing only. No sealed verdict, no queue-head promotion, no band read ever comes from a Mac run.** Screening rule stands: a screened arm earns a Kaggle slot only by naming the mechanism and why it binds on the private set.
- Local serving stack on Mac: **MLX (mlx-lm) recommended** for M-series; llama.cpp GGUF as fallback. Match `temp/top_p/top_k/ctx` to the field-floor config; accept that quant ≠ FP8 and label every local number `[MAC-SCREEN]`.

## CRITICAL: automation continuity (the nightly rail must not skip a beat)

Windows Scheduled Tasks do NOT migrate. Currently live on this box:
- `ARCDailySubmit` — submission daemon fires 18:37 + 20:07 EDT (`scripts/daily_submit.py` via cmd wrapper)
- `ARCDailyIterate` — 08:23 headless Claude iterate session (`scripts/daily_iterate.cmd`)
- `ARCCommunityBrief` — 06:00 headless Claude research sweep (`scripts/daily_community.cmd`)
- `ARCMorningCheck` — 06:00 morning check (see Task Scheduler for exact def)

**Rule: keep the Windows box's tasks RUNNING until the Mac equivalents have each fired successfully once.** Overlap is safe (daemon is idempotent per UTC day; brief/iterate are idempotent per day-file). Only then disable the Windows tasks.

macOS equivalents: `launchd` plists (`~/Library/LaunchAgents/com.arc.dailysubmit.plist` etc.) or `cron`. Port the three cmd wrappers to shell scripts (they are 3-5 lines each: cd + pipe prompt file into `claude -p --dangerously-skip-permissions`). Time zone: schedules are LOCAL time — daemon windows are anchored to 20:00 EDT UTC-day boundary; recompute if machine TZ differs.

## What git does not carry — copy or recreate by hand

1. **`~/.kaggle/kaggle.json`** — API credentials. Copy. Install both CLI versions: 2.0.x (pushes + `kernels output`) and 2.2.x in a separate venv (`competitions topics`, `kernels logs`).
2. **`scripts/bench_token.cmd`** — gitignored on purpose (KAOS_BENCH_TOKEN=atk_…). Recreate on the Mac (as `scripts/bench_token.sh` exporting the env var). NEVER commit it.
3. **Claude Code memory** — lives at `C:\Users\dcani\.claude\projects\f--kaggle\memory\` and is keyed to the project PATH. On the Mac the path differs → fresh empty memory. **Copy the whole `memory/` directory** into the Mac's equivalent (`~/.claude/projects/<new-path-slug>/memory/`) or the campaign loses its accumulated doctrine index (MEMORY.md + ~30 files). This is the highest-value non-git artifact on the machine.
4. **`runs/` heavy evidence (68GB, gitignored)** — kernel pulls, benchmark artifacts, intermediate_states. The registry rows point at these via `results_path`. Copy selectively (external drive): at minimum `runs/kernel_pulls/*/benchmark.json` + `execwm/` + `solver_note`-bearing files; movies/pkls optional. Or keep the Windows box as the archive host.
5. **KAOS install** — `F:\kaggle\kaos` is a separate repo/checkout. Clone + `uv sync` + `kaos.yaml` (bench block: endpoint dev.attraktor.dev, ws-neaez4yu; token env-only). Re-apply nothing: the agent_sdk nesting fix is committed.
6. **Repo venvs** — recreate with `uv sync` per project; never copied.
7. **Windows-path assumptions in scripts** — `daily_submit.py` and friends use `ROOT`-relative paths (fine), but check: `scripts/*.cmd` (Windows-only, port to .sh), any `F:\kaggle` absolute paths in ITERATION_LOG-referenced tooling, `march-madness-2026/.venv/Scripts/kaggle` (the CLI-2.2.x path — recreate as a venv on Mac), `bench_token.cmd` callers.

## Suggested sequence

1. Push everything from Windows (done 08-26); clone on Mac; `uv sync` everywhere.
2. Copy creds + token + memory dir + selected runs/ evidence.
3. Stand up MLX serving; validate `[MAC-SCREEN]` label wiring in local_gate (it already refuses to certify — keep it that way).
4. Recreate the four schedules as launchd; dry-run each (daemon with `already-submitted-today` expected; brief writes its file).
5. Watch one full day of dual-running; then disable Windows tasks.
6. Keep the Windows box reachable for 2 weeks as evidence archive + fallback rail.

---

## EXECUTION LOG — 2026-08-26 (Mac, first session)

Machine TZ is **America/New_York**, same as the Windows box, so every schedule
time ports **1:1**. No recomputation needed against the 20:00 EDT UTC-day
boundary.

### DONE

- `uv` installed (`~/.local/bin/uv`).
- **Repo venv built on Python 3.12.14** — see "3.12, not 3.14" below. torch
  2.13.0 with **MPS available and a live matmul on the 40-core GPU**.
- `taaf` installed editable from the in-repo bundle + `imageio-ffmpeg`.
- Both kaggle CLIs, side by side, per item 1:
  - **2.0.2** at `~/.local/bin/kaggle` (PATH primary — what `daily_submit.py`
    picks up via `shutil.which`)
  - **2.2.4** at `~/.venvs/kaggle22/bin/kaggle` (deliberately NOT on PATH so it
    cannot shadow 2.0.x); exported as `$KAGGLE22`.
- Four `.cmd` wrappers ported to `.sh` + a shared `scripts/_arc_env.sh` that
  resolves every binary absolutely (launchd hands jobs a minimal PATH) and
  loads `.env`.
- Four launchd plists written and `plutil`-validated in `scripts/launchd/`.
- **Submit daemon dry-run PASSED end to end on the Mac**: authenticated to
  Kaggle, read the competition's submission list, and correctly logged
  `{"skip": "already-submitted-today"}` — exactly the step-4 expected result.
- `local_gate --self-test`: **11 pass / 0 fail** (1 warn, 1 skip — both
  missing-evidence, see below). The doctrine line still prints: *"This gate
  NEVER certifies."* Step 3's requirement holds; no host-dependent path exists
  in the gate, so there is nothing to re-wire for `[MAC-SCREEN]` — it refuses
  to certify structurally, on every host.

### Corrections to the checklist above

- **Item 1 is superseded: there is no `kaggle.json`.** Auth is a single
  KGAT-format token in a gitignored `.env` (`KAGGLE_API_TOKEN=KGAT…`), loaded
  by `_arc_env.sh`. Verified: the token works with **both** CLI versions; the
  classic `KAGGLE_USERNAME` + `KAGGLE_KEY` pair **401s** against it. Do not
  "restore" a `~/.kaggle/kaggle.json` — it will not help.
- **Item 4 is far cheaper than 68GB.** The gate's real-artifact corpus is
  **26 artifacts / 21.2 MB and is ALREADY PRESENT** (git-tracked). Run
  `local_gate.py --corpus` to see it. Only two things are actually missing —
  see the blocker below.
- **Item 5: `../kaos` is an empty directory**, not a partial checkout — it is
  gitignored in the parent repo, so the clone created nothing. Needs a real
  clone. Until then `uv sync` fails on the `dev` group.

### New findings the checklist did not anticipate

- **CUDA wheels do not exist for macOS arm64.** `pyproject.toml` pinned
  `torch`/`torchvision` to the `pytorch-cu124` index unconditionally, so
  `uv sync` could not resolve at all. Fixed with a
  `marker = "sys_platform != 'darwin'"` on both pins: darwin falls through to
  PyPI (arm64 + MPS), and **the Windows box's resolution is byte-identical**.
- **3.12, not 3.14.** `requires-python = ">=3.11"` let uv pick CPython 3.14.7,
  on which the competition harness cannot run — `taaf` pins
  `requires-python == 3.12.12`. Kaggle kernels are 3.12 as well, so 3.14 both
  broke the harness and widened env-mismatch for no gain. Pinned via a new
  tracked `.python-version` (`3.12`). This does not disturb the Windows box's
  existing venv; it only governs the next venv creation, and it moves both
  boxes toward the Kaggle rail.
- **The daemon must NOT run under `uv run`.** `uv run` re-locks on every
  invocation — needing the network, and dying on any unresolvable path dep
  (the empty `../kaos`). A nightly rail cannot depend on dependency
  resolution succeeding at fire time. `run_daily_submit.sh` now calls
  `.venv/bin/python` directly; `daily_submit.py` and `preflight.py` are
  stdlib-only, so that is sufficient.
- **LAPTOP SLEEP is a real risk to the nightly rail, and Windows never had it.**
  `pmset` shows `sleep 1` on **AC** power and no scheduled wake covering 18:37,
  20:07, 06:00 or 08:23. launchd `StartCalendarInterval` jobs do **not** fire
  during sleep — they fire once, late, on wake, which can push a submission
  past the 20:00 EDT UTC-day boundary the campaign anchors to. Mitigation
  (needs sudo): `sudo pmset -c sleep 0` to match the Windows desktop's
  always-on behaviour on AC. Verify with `pmset -g sched` afterwards.
- The VS Code extension ships a `claude` binary at a **version-pinned path**
  (`~/.vscode/extensions/anthropic.claude-code-<ver>/…`), which moves on every
  extension update and would silently kill the three Claude-session jobs.
  `_arc_env.sh` prefers a standalone `~/.local/bin/claude` and only falls back
  to the newest extension binary. **Install the standalone CLI.**

### BLOCKER — the gate cannot license a push on this Mac yet

`local_gate.py --arm q38-field --fast` → **FAIL (27 pass / 2 fail)**. All of it
is one cause — missing arm-stamped evidence, not anything Mac-specific:
- `X1.p2_score` — its A1 control scans
  `runs/kernel_pulls/{q38_field_v1, execwm_v1, p1_notes_v1, budget_t3_v1,
  private_base_v1, q38graft_v1}` and found **zero** (`tried == 0` fails the
  check). Local `runs/kernel_pulls/` holds only `b122_v1`, `q38_v1`, `q38_v2`.
- `P9.cadence_instrument` / `S13` — need `runs/tufa_example_run/benchmark.json`.
- `A0.matrix`, `S10.cross_arm_can_refuse` — need any real per-arm / graft
  artifact on disk.

**So the minimal copy from the Windows box is small and specific** — the six
`runs/kernel_pulls/<arm>_v1/` pull dirs above, plus `runs/tufa_example_run/`.
Copy those and the two dark negative controls light up. Until then **the Mac
can run the daemon but cannot license a build**, and the Windows box remains
the only box that can.

### NEXT

1. `sudo pmset -c sleep 0` (else the nightly rail is unreliable on a laptop).
2. Install the standalone `claude` CLI so the three Claude jobs stop depending
   on a version-pinned extension path.
3. Load the four launchd agents:
   `cp scripts/launchd/*.plist ~/Library/LaunchAgents/` then
   `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.arc.<job>.plist`
   for each. Re-copy after any edit — launchd reads the file at bootstrap.
4. Copy the six `kernel_pulls` dirs + `tufa_example_run` → gate goes green.
5. Copy the Claude Code memory dir (item 3) — still empty; the Mac slug is
   `~/.claude/projects/-Users-danilocanivel-Projects-kaggle/memory/`.
6. Clone `../kaos`; then a plain `uv sync` (no `--no-dev`) works again.
7. Keep the Windows tasks running until each Mac job has fired once.
