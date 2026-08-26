# Discussion Sweep — 2026-07-18

Feed: kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion (sorted recent-comments).
Baseline: prior sweep covered posts through 2026-07-17 (discussions_2026-07-16.md +
daily_brief_2026-07-17.md §1b). Only NEW material since then is reported below.

Fetch note: Kaggle discussion pages are client-rendered; plain WebFetch returns only the
page title. Used chrome-devtools (navigate + a11y snapshot) to read full OP + comments.

## Summary of window
Two items are genuinely new since 07-17:
1. **NEW THREAD** #727119 — Greg Kamradt (HOST): "500 Submissions Analyzed - Common Errors".
2. **NEW HOST REPLY** on #724841 (RLIMIT_AS thread, already logged 07-16): Greg Kamradt
   posted an official answer (16h ago) enumerating Kaggle rerun death-mode limits.

Everything else on the recent-comments front page was already evaluated in prior sweeps
(#724890 RTX6000, #726367 AGI-timeline, #716295 parallelism, #726903 x(-1)/week,
#725002 Milestone#1, #726552 variance, #726340 GPT-5.6, #716711 Akhil/Gemma-4-31B,
#717133 Tufa Duck). No YUTO KOJIMA footprint. No host answer yet on env seeding in #726552.
No new public notebook > 0.79 shared in-window.

---

## 1. #727119 — "500 Submissions Analyzed - Common Errors" (Greg Kamradt, HOST, ~16h ago)
Kaggle team (Walter) analyzed the last 500 FAILED submissions to reduce the opaque-error
friction. Findings, by frequency:
- **~1/3 of failures have NO traceable error** — notebook "just gets stuck": logic bugs,
  infinite loops, wrong endpoint, async deadlocks. (This is the single biggest bucket.)
- **~20%: GPU code submitted with the GPU accelerator NOT enabled** on the notebook.
- Long tail (<5% each): expected dataset not attached; missing dependencies; logic bugs
  (e.g. `object of type float has no len()`); CUDA OOM; **using the three.arcprize.org API
  instead of the required in-notebook harness** (see the random-agent template); writing to
  read-only `/kaggle/input`.
- Prevention advice: start from a pinned working template, submit, confirm output, then
  iterate; feed this error list to a code-reviewing agent.
- Comments: Van-Phuc Huynh (rank 23) asked if hosts saw notebooks pre-open-source → Greg:
  "**Nope — we don't have access to notebooks until teams open source them.**" (Confirms
  the private test set / opacity is real; nobody is reading our code.) hwe owe: team-up spam.

**Verdict: ADAPT (checklist).** The ~1/3 "silent stuck" bucket is exactly our 0.00 infra-death
class (matches #726552 and our build-rail sd 0.572 finding). Two concrete preflight additions:
(a) our async watchdog / probe-diff summarizer threads must have a hard wall-clock deadline so
we never join the "just gets stuck" third; (b) assert GPU accelerator flag is ON for any
war-v4 (Qwen-72B AWQ) submission — 20% of ALL failures are this one mistake, and our model
line runs on the RTX PRO 6000 rail. Feed this exact list into the preflight.py forbidden-pattern
/ config-assert set. Greg's "no notebook access" line also reinforces: the 1.86 leader stays
opaque; no methods leak.

## 2. #724841 — NEW HOST REPLY: official rerun death-mode answers (Greg Kamradt, ~16h ago)
Bill Ma's original OP (RLIMIT_AS false-OOM) was logged 07-16. New: Greg posted the Kaggle
team's answers to Bill's 5 follow-up questions on invisible rerun deaths:
1. **Native crash (segfault):** Kaggle captures container exit code + stderr; **exit code 139
   = SIGSEGV**. Core dumps / signal detail are NOT exposed to users.
2. **Output size:** Docker logs capped at **10 MB/container**; exceeding it does NOT kill the
   notebook — output simply stops being captured (silent log truncation).
3. **Disk quota:** default **20 GB for /kaggle/working**, measured at session end → over-limit
   = kernel terminated "out of disk". Outside that dir there's ~60 GB temp scratch.
4. **Process/thread:** **no RLIMIT_NPROC**; daemon threads (watchdog etc.) won't trip a
   process cap. On 4-core alloc, many threads → contention, not termination; crashes are more
   likely memory (per-thread stack) than a process cap.
5. **RLIMIT_AS:** Kaggle does **not** set it; memory enforced via **cgroups at physical level
   (30 GB for CPU notebooks)**. scipy/numpy mmap won't exhaust VAS. If you hit a memory wall
   it's physical RAM.

**Verdict: ADOPT (infra constants + checklist).** These are authoritative, previously-unknown
Kaggle limits directly governing our silent-death debugging: (i) capture/log exit code and
scan for **139 = SIGSEGV** in any rerun post-mortem; (ii) our verbose probe-diff / ASCII-dump
logging must stay **under 10 MB total** or later diagnostics vanish (relevant to war-v3
summarizer verbosity); (iii) keep /kaggle/working writes under 20 GB (banking/replay trace
persistence — write scratch to the 60 GB temp area, not /working); (iv) our multi-thread swarm
+ watchdog is safe from a process cap but each thread's stack counts against the 30 GB physical
budget — matters for war-v4 72B-AWQ headroom. Add these five constants to preflight.py notes.

---

## Actions
1. preflight.py: (a) assert GPU accelerator ON for any GPU/model submission (20% of all
   failures); (b) add wall-clock deadline assertion on watchdog/summarizer threads (silent-stuck
   bucket = 1/3 of failures); (c) record the five infra constants (SIGSEGV=139, 10 MB log cap,
   20 GB /working, no RLIMIT_NPROC/RLIMIT_AS, 30 GB cgroup physical).
2. war-v3 summarizer: cap cumulative stdout/stderr < 10 MB so post-mortem logs survive.
3. banking/replay: persist traces to the 60 GB temp scratch, not /kaggle/working (20 GB cap).
4. war-v4 (72B AWQ): budget per-thread stack against the 30 GB physical ceiling before push.
5. Standing watch (unchanged): no YUTO footprint; no env-seeding host answer in #726552 yet.
