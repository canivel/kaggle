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
