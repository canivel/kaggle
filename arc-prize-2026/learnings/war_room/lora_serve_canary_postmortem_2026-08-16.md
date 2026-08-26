# LORA-SERVE-CANARY v1 — POST-MORTEM FROM THE LOG, 2026-08-16

**Arm:** `canivel/arc3-lora-serve-canary` version 1 · pushed and run **2026-08-14 13:40:18 UTC** ·
status **ERROR** · **one version ever pushed** (`kernels list --user canivel`).

**This closes the OWED item.** The v1 death was on the record as *"diagnosed without the log"*
(`learnings/war_room/lora_lane_2026-08-13.md` §12) on the false premise that the log could not be
retrieved. The log has now been retrieved in full. Everything below is read from it.

Provenance tags: **[V]** read directly from the retrieved log this session · **[V-src]** read from
local source/git · **[INF]** inference.

---

## 1. RETRIEVAL PROVENANCE

```bash
# Bash tool, CLI 2.2.3 (NOT the 2.0.x we push with — that binary has no `logs` subcommand)
/f/kaggle/march-madness-2026/.venv/Scripts/kaggle --version
#   -> Kaggle CLI 2.2.3
/f/kaggle/march-madness-2026/.venv/Scripts/kaggle kernels logs canivel/arc3-lora-serve-canary \
    > runs/kernel_logs/lora_serve_canary_v1.log.json
#   -> exit 0, empty stderr
```

| property | value |
|---|---|
| retrieval | streams to **stdout**; writes no file, touches **no** part of `/kaggle/working` |
| bytes | **236,029 B** (230 KB) |
| entries | **1,506** JSON records — `{stream_name, time, data}` |
| split | 910 `stdout` / 596 `stderr` |
| time span | first `t = 4.381 s`, last `t = 102.501 s` (seconds since session start) |
| completeness | ends on `[NbConvertApp] Writing 518651 bytes to __results__.html` — the natural end of a Kaggle session, **not truncated** |
| saved to | `F:\kaggle\arc-prize-2026\runs\kernel_logs\lora_serve_canary_v1.log.json` |

**Provenance caveat, stated so nobody double-counts:** the stream contains **335 duplicate records**
(1,506 total / 1,171 unique). Kaggle emits some lines on both the captured-notebook and raw-process
channels — e.g. the adapter banner appears at `t=8.110` and again at `t=8.310`, and the fatal
traceback appears **three times** (`t=99.049`, `99.065`, `99.065`). This is a logging artifact and
changes nothing; the earliest occurrence is used for every timestamp below.

**`kernels files` is still not evidence** — it returns empty for every errored kernel we have
checked. It was not used here.

---

## 2. STAGE-BY-STAGE TIMELINE  [V]

`t` is seconds since session start. Δ is elapsed within the stage.

| t (s) | Δ (s) | stage | log line |
|---:|---:|---|---|
| 4.381 | — | session start | pydevd frozen-modules warning (papermill boot) |
| 5.411 | 1.03 | **cell 2 — identity banner** | `LORA-SERVE-CANARY mode=boot-only brain=driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot (UNCHANGED, the scored brain) wheels=driessmit1/arc3-vllm-h100-wheelhouse-v3 (UNCHANGED, vLLM 0.19.0) delta=--enable-lora + two RANDOM r16 adapters (noop B=0, probe B~1e-3) AND NOTHING ELSE …` |
| 5.411 | 0.00 | mode confirmed | `taaf.kaggle: TRUE_SUBMISSION=False` |
| 7.512 | 2.10 | **dataset mounts resolved — 4/4** | `taaf.kaggle: input paths = {"canivel/arc3-lora-probe-adapters": …, "driessmit1/arc3-vllm-h100-wheelhouse-v3": …, "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot": …, "jeroencottaar/taaf-kaggle-source-share": …}` |
| 7.534 | 0.02 | source roots wired | `taaf.kaggle: wrote /usr/local/lib/python3.12/dist-packages/taaf_kaggle_sources.pth (3 source roots)` |
| 7.534 | 0.00 | **cell-8 setup rewrite executed on the real bundle** | `LORA-CANARY setup-commands rewrite OK (6 anchors replaced; loud-fail)` |
| 7.535 | 0.00 | setup subprocess launched | `taaf.kaggle: setup command: "$PYTHON" - <<'PYSETUP'` … (full heredoc echoed) |
| 8.110 | 0.58 | **adapter #1 verified** | `LORA-CANARY adapter arc3-noop path=/kaggle/input/datasets/canivel/arc3-lora-probe-adapters/lora-noop bytes=41962184 sha=d777d4c7a7ebec85 r=16 rslora=True` |
| 8.661 | 0.55 | **adapter #2 verified** | `LORA-CANARY adapter arc3-probe path=…/lora-probe bytes=41962184 sha=d7d6918d01ae67f6 r=16 rslora=True` |
| 8.661 | 0.00 | paths echoed | `vLLM wheelhouse path: /kaggle/input/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3` · `Qwen model path: /kaggle/input/datasets/driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` |
| 8.674 | 0.01 | **GPU check** | `CUDA GPU check passed for rtx-pro-6000 x1: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']` |
| 8.676 | 0.00 | wheelhouse install begins | `Installing vLLM wheelhouse into /kaggle/working/vllm-site-packages` |
| 8.994 | 0.32 | pip resolves offline | `Looking in links: /kaggle/input/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3` |
| **91.191** | **82.20** | **wheelhouse install COMPLETE** | `Successfully installed … torch-2.10.0 … transformers-4.57.6 triton-3.6.0 vllm-0.19.0 flashinfer-python-0.6.6 …` |
| 91.4 → 99.0 | 7.65 | *silent gap* — `INSTALL_STAMP.write_text` + fs settle after the `--target` install. **No output at all.** [INF] |
| **99.049** | — | ☠ **FATAL** | `NameError: name '_source_path_entries' is not defined` |
| 99.288 | 0.24 | papermill converts it to a cell failure | `CalledProcessError: Command '"$PYTHON" - <<'PYSETUP' … PYSETUP' returned non-zero exit status 1.` |
| 100.4–102.5 | 3.2 | nbconvert writes artifacts, session ends | `[NbConvertApp] Writing 191216 bytes to __notebook__.ipynb` · `Writing 518651 bytes to __results__.html` |

**Total session cost: 102.5 s. Under two minutes of the RTX PRO 6000.**

### 2.1 Stages that were NEVER reached

Everything after `install_vllm_wheelhouse()` inside `start_vllm_server()`:

- vLLM server log file open (`VLLM_SERVER_LOG.parent.mkdir` is `<stdin>` **line 169**; we died at **168**)
- the vLLM subprocess launch and the `--enable-lora --max-lora-rank 16 --lora-modules arc3-noop=… arc3-probe=…` argv
- the **35.9 GB weight load** — never started
- `/v1/models` adapter registration check
- the stock `2 + 2` smoke
- **the noop/probe differential — the entire point of the canary**
- tool-call round trip, MM image round trip, `preserve_thinking` check
- the LoRA **throughput tax** measurement and `lora_canary.json`

**Zero GPU compute was spent on the model. `nvidia-smi` was the only GPU touch in the whole run.**

---

## 3. THE EXACT FATAL TEXT  [V]

Verbatim, first occurrence at `t = 99.049 s`, `stream_name = "stderr"`:

```
Traceback (most recent call last):
  File "<stdin>", line 495, in <module>
  File "<stdin>", line 168, in start_vllm_server
  File "<stdin>", line 256, in _lora_install_guard
NameError: name '_source_path_entries' is not defined
```

**The three frames resolved against the heredoc the log itself echoed** (the setup command is fed on
stdin, so `<stdin>` line *N* is line *N* of the `PYSETUP` body; the body begins at log entry 15 and
lines were counted off it directly) [V]:

| frame | `<stdin>` line | the line that raised |
|---|---:|---|
| `<module>` | 495 | `start_vllm_server()` |
| `start_vllm_server` | 168 | `    _lora_install_guard()` |
| **`_lora_install_guard`** | **256** | **`    for base in _source_path_entries(BUNDLE_DIR):`** |

Line 169 — the very next statement, never executed — is
`    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)`.

The outer failure, at `t = 99.288 s`, raised in **notebook cell 8, line 66** [V]:

```
Exception encountered at "In [4]":
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_65/2726026478.py in <cell line: 0>()
     64 for command in _lora_patch_setup_commands(json.loads((BUNDLE_DIR / "setup_commands.json").read_text())):
     65     print(f"taaf.kaggle: setup command: {command}", flush=True)
---> 66     subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
…
CalledProcessError: Command '"$PYTHON" - <<'PYSETUP'
… PYSETUP' returned non-zero exit status 1.
```

**Note the irony the log makes visible:** cell-8 line 64 uses `BUNDLE_DIR` *legitimately* — it is a
notebook-scope name and it is in scope there. It is one line above the `subprocess.run` that starts a
**different interpreter**. The same identifier was then written inside the heredoc where it does not
exist. Only `_source_path_entries` is named in the error because Python evaluates the callable before
its argument; `BUNDLE_DIR` is equally undefined and was simply never reached.

---

## 4. THE LOCAL SOURCE — WHERE THE DEFECT LIVED, AND WHERE IT IS NOW

### 4.1 The defect, as pushed  [V-src, `git show 7c6503a`]

The v1 that ran is commit `7c6503a` of
`duck_eval/lora/build_lora_serve_canary.py`. **The offending lines, quoted:**

```
duck_eval/lora/build_lora_serve_canary.py:176   def _lora_install_guard() -> None:
duck_eval/lora/build_lora_serve_canary.py:184       guard_src = None
duck_eval/lora/build_lora_serve_canary.py:185       for base in _source_path_entries(BUNDLE_DIR):
duck_eval/lora/build_lora_serve_canary.py:186           candidate = Path(base) / 'inference' / 'tools' / 'vllm_runtime_lora_guard.py'
```

Injected into the setup heredoc by the rewrite at `build_lora_serve_canary.py:437-441` (v1
numbering), which appended `_lora_install_guard()` to `start_vllm_server()`:

```
duck_eval/lora/build_lora_serve_canary.py:437       (
duck_eval/lora/build_lora_serve_canary.py:438           "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()",
duck_eval/lora/build_lora_serve_canary.py:439           "def start_vllm_server() -> None:\n    install_vllm_wheelhouse()\n"
duck_eval/lora/build_lora_serve_canary.py:440           "    _lora_install_guard()",
```

`build:185` → `<stdin>:256`. `build:440` → `<stdin>:168`. **Exact match, both frames.**

### 4.2 The defect is NOT in the current working tree — it is already fixed  [V-src]

The working copy (modified, uncommitted, 62 insertions since `7c6503a`) is the v2 artifact and
**does not contain the defect**:

- `build_lora_serve_canary.py:189-197` — the guard now resolves its own source by rglobbing
  `/kaggle/input` and `/kaggle/input/datasets` for `inference/tools/vllm_runtime_lora_guard.py`.
  **No notebook-scope name is referenced.**
- `build_lora_serve_canary.py:184-188` — a comment records the death at the exact site.
- `build_lora_serve_canary.py:450-452` — the call site is now
  `try: _lora_install_guard() / except Exception as _guard_exc: print('LORA-CANARY guard=SKIPPED …')`.
  A belt-and-braces probe can no longer kill the run.
- `build_lora_serve_canary.py:524-563` — `_assert_names_resolve()`, an AST scope gate, called at
  `build_lora_serve_canary.py:631` immediately after the `compile()` at line 630.

**One factual error remains in the fixed source and must be corrected** [V vs V-src]:

```
duck_eval/lora/build_lora_serve_canary.py:530       perfectly, installed the wheelhouse, and died at runtime ~1 GPU-h in, before the vLLM
```

The log says **99.05 s**, not ~1 GPU-h. The same overstatement is in
`learnings/war_room/lora_lane_2026-08-13.md` §12.4 (*"One GPU-hour, one slot"*). The **slot** was
spent; the **GPU-hour was not**. See fix 5.

### 4.3 `lora_canary_smoke.py` — the gap the log makes concrete  [V-src]

The smoke's only code-validity check is `compile()`:

```
duck_eval/lora/lora_canary_smoke.py:104           try:
duck_eval/lora/lora_canary_smoke.py:105               compile("".join(cell["source"]), f"cell{i}", "exec",
duck_eval/lora/lora_canary_smoke.py:106                       flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
duck_eval/lora/lora_canary_smoke.py:107           except SyntaxError as exc:
duck_eval/lora/lora_canary_smoke.py:108               bad.append(f"cell {i}: {exc}")
duck_eval/lora/lora_canary_smoke.py:109       check("S4 every code cell compiles", not bad, "; ".join(bad)[:200])
```

`compile()` catches syntax, **not scope**. The v1 body compiled cleanly and died at runtime. There is
still **no test anywhere in the repo that exercises `_assert_names_resolve`** — the scope gate lives
only inside the builder, and `readiness_v2_2026-08-15.md` §6.3 already flags this as ephemeral.

---

## 5. SEALED CLASSIFICATION

> ### **CONFIG/AUTHORING DEFECT.**
> **Not decisive. Zero information about the LoRA-serve lane, in either direction.**

The sealed scorer `duck_eval/lora/lora_serve_score.py` maps this to its third state,
**`INFRA DEATH (not decisive)`** — "the run never reached the differential … A retry, never a
verdict" (`lora_serve_score.py:22-25`, `:68`). That mapping is **correct on licensing** and is the
operative verdict for the ledger. But the *root-cause class* is not infra: nothing about the Kaggle
rail, the wheelhouse, the GPU, CUDA, disk, or memory failed. **Every piece of infrastructure this run
touched behaved perfectly.** The kernel was killed by four words of our own injected Python that
referenced a name from a different interpreter's namespace, and it was **deterministic, reproducible
locally, and detectable with no GPU at all**.

**Contrast with Q38 v1 (`q38_engine_swap_prereg_2026-08-15.md` §9), which is the closest sibling:**
there the engine *served*, generated, tool-called and accepted an image, and a bad probe payload
killed it at t=425 s — so that run positively established several things about the engine. **Here the
engine was never launched.** This is a strictly emptier failure: same authoring-defect class, but at
`t=99 s` instead of `t=425 s`, and with the measurement surface entirely untouched.

### 5.1 What this failure does NOT license  [V]

It says **nothing** about:

- whether vLLM 0.19.0 can serve `--enable-lora` against `vrfai/Qwen3.6-27B-FP8` at all;
- whether the two adapters would **register** on `/v1/models`;
- whether a non-zero adapter would be **applied** or **silently ignored** — the noop/probe
  differential, the entire reason this canary exists, never ran;
- the **LoRA throughput tax**, or whether `--enable-lora` keeps actions/window above the 100 bar;
- tool-calling or the MM path under LoRA;
- boot time with `--enable-lora`;
- anything at all about adapter *quality*, since these were random r16 probes by design.

**`§11.5` of `lora_lane_2026-08-13.md` (the sealed reading rule) remains sealed and unused.** No
constant in `lora_serve_score.py` may be touched on the strength of this run — there is no data to
fit to.

### 5.2 What this run DOES positively establish  [V]

Modest, but real, and worth banking because it retires four named risks for v2:

1. **The adapter dataset push is proven end to end.** Both adapters mounted and hashed **exactly**
   what we built: `arc3-noop sha=d777d4c7a7ebec85`, `arc3-probe sha=d7d6918d01ae67f6`, **41,962,184 B
   each**, `r=16`, `rslora=True`. `feedback_kaggle_dataset_code_sync` risk: **retired for this arm.**
2. **4/4 `dataset_sources` attached.** The `feedback_kaggle_model_attach` silent-drop failure mode
   **did not occur** — the adapter dataset, wheelhouse, brain snapshot and source share were all
   mounted (`t=7.512`).
3. **The correct GPU was allocated:** `NVIDIA RTX PRO 6000 Blackwell Server Edition` ×1 (`t=8.674`).
4. **The 6-anchor cell-8 setup rewrite executed correctly against the real bundle at runtime**
   (`LORA-CANARY setup-commands rewrite OK (6 anchors replaced; loud-fail)`, `t=7.534`) — the
   fail-loud rewrite machinery works.
5. **The wheelhouse installs in 82.2 s** on this rail (vLLM 0.19.0 / torch 2.10.0 / flashinfer 0.6.6
   into `--target`). A useful budget constant.
6. **The cost was 102.5 s, not a GPU-hour.** The slot was lost; the compute was not.

---

## 6. FIXES — NUMBERED, EACH TIED TO A LOG LINE

| # | fix | tied to | status |
|---|---|---|---|
| **1** | **Guard resolves its own source.** Replace `_source_path_entries(BUNDLE_DIR)` with an rglob over `/kaggle/input` for `inference/tools/vllm_runtime_lora_guard.py`; reference **no** notebook-scope name inside the heredoc. | `<stdin>:256` → `NameError: name '_source_path_entries' is not defined` | **LANDED** — `build_lora_serve_canary.py:189-197` |
| **2** | **The guard must not be able to fail the run.** Wrap the call site in `try/except` → `LORA-CANARY guard=SKIPPED <exc>`. It was always documented belt-and-braces; the differential is the real evidence. | `<stdin>:168` `_lora_install_guard()` — sitting between a completed install (`t=91.19`) and a server that was never launched (`<stdin>:169`, never executed) | **LANDED** — `build_lora_serve_canary.py:450-452` |
| **3** | **Build-time AST scope gate.** `_assert_names_resolve()` walks the rewritten heredoc, collects every `Store`/`arg`/`alias`/`ExceptHandler`/comprehension binding, and `SystemExit`s on any loaded name that is neither defined nor a builtin. `compile()` catches syntax, not scope. | the `NameError` itself — the v1 body **compiled clean** and still died | **LANDED** — `build_lora_serve_canary.py:524-563`, called at `:631` |
| **4** | **Persist the v1 fixture as a test.** Add a smoke check that feeds the literal v1 statement `for base in _source_path_entries(BUNDLE_DIR):` to `_assert_names_resolve` and asserts it raises, **plus** a negative control that the shipped v2 body passes. Today the only code-validity check in the suite is `compile()`. | `lora_canary_smoke.py:104-109` (`S4 every code cell compiles`) — the exact category that missed this bug | **OPEN** — flagged in `readiness_v2_2026-08-15.md` §6.3; a *weakened or removed* gate still passes all 75 checks silently |
| **5** | **Correct the cost on the record.** The log says the run died at **99.05 s** and the session ended at **102.5 s**. Two places claim ~1 GPU-hour and must be edited to say **~100 s (one slot, negligible GPU)**. | `t=4.381` → `t=102.501`, the whole log | **OPEN** — `build_lora_serve_canary.py:530`; `lora_lane_2026-08-13.md` §12.4 |
| **6** | **Stage heartbeats in the setup heredoc.** Emit `LORA-CANARY stage=<name> t=<s>` after the wheelhouse install, before the guard, before the server launch, and after the server-log open. v1 produced a **7.65 s completely silent window** between `Successfully installed` (`t=91.19`) and the traceback (`t=99.05`); had the traceback been swallowed, that gap is where the diagnosis would have died. | the silence between log entry 917 (`t=91.395`) and 918 (`t=99.049`) | **OPEN** |
| **7** | **Adopt Q38's observe-before-verdict rule here too.** Not implicated in *this* death — the traceback was fully self-describing — but the canary's own asserts (`_lora_serve_asserts`, `build_lora_serve_canary.py:267-279`, and the MM probe) still raise conclusions without dumping `finish_reason` / `content_chars` / `reasoning_chars`. That is precisely what left Q38 v1's vision question unclosable. | preventative; tied to `q38_engine_swap_prereg_2026-08-15.md` §10.2.3 | **OPEN** |
| **8** | **Doctrine: `kernels logs` FIRST, always.** CLI **2.2.3**, `kaggle kernels logs <slug>` — 236 KB in seconds, never touches `/kaggle/working`. Never diagnose by bracketing artifact presence/absence again. | the whole of this document vs `lora_lane_2026-08-13.md` §12.1 | **LANDED** — banked in `duck_eval/README.md` step 3; correction already appended to §12 of the lane doc |

### 6.1 The blind diagnosis, graded against the log

`lora_lane_2026-08-13.md` §12 bracketed the death from artifact presence/absence. Now that the log
exists, it can be graded:

| §12 claim | log verdict |
|---|---|
| `install_vllm_wheelhouse()` ran to completion | **CONFIRMED** — `Successfully installed … vllm-0.19.0`, `t=91.191` |
| died **before** `VLLM_SERVER_LOG` was opened | **CONFIRMED** — raised at `<stdin>:168`; the log open is `<stdin>:169` |
| the cause is `_source_path_entries(BUNDLE_DIR)`, a notebook-scope name in a separate interpreter | **CONFIRMED VERBATIM** — `NameError: name '_source_path_entries' is not defined`, `<stdin>:256` |
| the setup command never reached its final env block | **CONFIRMED** — the `setup_env = {…}` block is `<stdin>:497+`, well past 495 |
| **cost: "One GPU-hour"** | **REFUTED — 102.5 s.** |

**Three of four structural claims were exactly right; the cost claim was wrong by a factor of ~35.**
That is the honest characterisation: the bracketing reasoning was sound but it also produced one
confidently-stated false number that has been sitting in the war room for two days. **Bracketing gets
you the mechanism and invents the magnitude** — which is why fix 8 is doctrine and not a preference.

---

## 7. BOTTOM LINE — for `ITERATION_LOG.md`

**`canivel/arc3-lora-serve-canary` v1 (pushed 08-14 13:40 UTC) is now diagnosed from the actual
kernel log** (CLI 2.2.3 `kernels logs`, 236,029 B / 1,506 entries, saved to
`runs/kernel_logs/lora_serve_canary_v1.log.json`), closing the OWED item. **It died at t = 99.049 s
with `NameError: name '_source_path_entries' is not defined`, raised at `<stdin>:256` inside
`_lora_install_guard()`, called from `start_vllm_server()` at `<stdin>:168` — one statement before
the vLLM server log would have been opened.** The cause is ours and only ours: the setup command runs
in a separate `"$PYTHON" - <<'PYSETUP'` interpreter, and `_source_path_entries` / `BUNDLE_DIR` are
notebook cell-8 names that do not exist there; the body compiled cleanly, which is exactly why our
`compile()`-only build check waved it through. **Classification: `CONFIG/AUTHORING DEFECT` —
NOT DECISIVE** (the sealed scorer maps it to `INFRA DEATH (not decisive)`, correctly, since the
differential was never reached). **It licenses no conclusion whatever about the LoRA-serve lane**:
vLLM never launched, the 35.9 GB brain never loaded, `--enable-lora` was never exercised, and the
noop/probe differential — the entire point of the canary — never ran; §11.5's reading rule stays
sealed and unused. What the run *did* prove is worth banking: the adapter dataset shipped byte-exact
(`arc3-noop d777d4c7a7ebec85`, `arc3-probe d7d6918d01ae67f6`, 41,962,184 B each, r=16), 4/4
`dataset_sources` attached with no silent drop, the RTX PRO 6000 Blackwell was allocated, the 6-anchor
cell-8 rewrite executed correctly against the real bundle, and the wheelhouse installs in 82.2 s.
**One correction to the standing record: the cost was 102.5 seconds, not the "one GPU-hour" claimed in
`lora_lane_2026-08-13.md` §12.4 and in the builder's own docstring — the slot was lost, the GPU-hour
was not.** Fixes 1-3 (guard resolves its own source; guard wrapped so it can never fail the run;
`_assert_names_resolve()` AST scope gate at build time) are already landed in the v2 working tree;
fixes 4-7 remain open, the important one being that **nothing in the repo currently tests the scope
gate itself** — the same shape of hole as the bug it was built to catch.
