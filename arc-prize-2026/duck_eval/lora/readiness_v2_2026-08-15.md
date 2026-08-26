# LoRA SERVE CANARY v2 — PUSH-READINESS AUDIT, 2026-08-15

**Auditor:** LoRA serve-canary lane agent. **Run at:** 2026-08-15 10:23–10:32 EDT.
**Mandate:** prove v2 is ready to push, then STOP. Nothing was pushed. Nothing was submitted.
No cloud spend. The only network calls were read-only pulls (`kernels pull`, `datasets
download`, `kernels list/status`), all free.

**VERDICT: ARTIFACT = GO. PUSH = NO-GO.**

The artifact is provably clean on every gate — 75/75, 35/35, ALLOW/0/0, adapters byte-exact,
the v1 bug class independently proven caught. **The push is blocked on AUTHORIZATION, not on
readiness:** the ledger re-confirm mandated by §11.4 found the 08-14 authorization
**superseded by a stand-down order entered in today's ledger section**. See §7.

---

## 1. What was claimed vs what was MEASURED

| # | Log/brief CLAIMS | I MEASURED | Verdict |
|---|---|---|---|
| 1 | smoke `75/75` | **75 passed / 0 failed** | ✅ exact |
| 2 | scorer selftest `35/35` | **35 passed / 0 failed** | ✅ exact |
| 3 | AST gate `182 loaded names / 0 unresolved` on v2 | **182 loaded names, 0 unresolved** | ✅ exact |
| 4 | AST gate "catches the real v1 body" | **reproduced independently** — re-injected the exact v1 statement `_source_path_entries(BUNDLE_DIR)`; gate raised `SystemExit: … references names it does not define: BUNDLE_DIR, _source_path_entries` | ✅ true, **but not persisted as a test** (§6.3) |
| 5 | preflight `ALLOW, 0 fails, 0 warns`, D4 = `[2,6,8,14]` | **ALLOW, 0 fails, 0 warns, 5 N/A, D4 = [2,6,8,14] MATCHES --expect-diff-cells** | ✅ exact (via local emulation — §3) |
| 6 | adapters `41,962,184 B each`, shas hard-asserted | **local == Kaggle == manifest == notebook pin, 41,962,184 B each** | ✅ exact |
| 7 | `4/4 dataset_sources`, env byte-matched | **4/4 on the live remote; enable_gpu / docker sha / machine_shape / competition_sources all byte-match the frozen duckfork** | ✅ exact |
| 8 | "the push path now has TWO guards (`--confirm-push` + idempotence exit 3)" | **FALSE for this lane.** Those guards exist ONLY in `duck_eval/a17/b122_push_v2.sh`, hardcoded to the b122 kernel and to `date == 2026-08-14`. This lane had **no push script at all**; v1 was pushed ad-hoc and unguarded | ❌ **DISCREPANCY** (§6.1) |
| 9 | "scorer run against v1 returns INFRA DEATH / decisive=False" | **NOT REPRODUCIBLE from repo state** — no v1 output artifacts were persisted (`runs/kernel_pulls/` has no lora entry; no `lora_canary.json` anywhere) | ⚠️ **unverifiable** (§6.4) |
| 10 | "08-15 slot 2 is free; AUTHORIZED" | Slot 2 **is** arithmetically free (§5) — **but the authorization is SUPERSEDED** (§7) | ❌ **DISCREPANCY** |

---

## 2. Build + gates (observed output, verbatim numbers)

```
$ uv run python duck_eval/lora/build_lora_serve_canary.py
setup-command name resolution: OK (182 loaded names, 0 unresolved)
setup-command rewrite validated locally: 6 anchors, 25,999 B, compiles clean
wrote …/notebooks/lora-serve-canary/arc3-lora-serve-canary.ipynb (49,118 B, 17 cells)
adapter shas: {"lora-noop": "d777d4c7a7ebec85", "lora-probe": "d7d6918d01ae67f6"}

$ uv run python duck_eval/lora/lora_canary_smoke.py      -> 75 passed / 0 failed
$ uv run python duck_eval/lora/lora_serve_score.py --selftest -> selftest: 35 passed / 0 failed
```

**Build is idempotent and matches what was already on disk** (so the artifact reviewed
yesterday IS the artifact the current builder produces):

| file | sha256 before rebuild | after rebuild |
|---|---|---|
| `arc3-lora-serve-canary.ipynb` | `0b54a3131e91e9f0dafa2526b17f55601a6ede58f44a9de5920c4200cefcd044` | **identical** |
| `kernel-metadata.json` | `9121d64649ad6f8016a9a85f962e4989c24794a821d57ea0c0471b4580852974` | **identical** |

Smoke section tallies observed: `[B]` 2, `[S]` 12, `[R]` 26, `[D]` 10, `[A]` 9, `[M]` 6,
`[P]` 5, `[V]` 2 = **75**, 0 failed. Negative paths D7–D10 (dataset silently dropped, flat
layout, tampered weights, wrong rank) each observed raising the intended FATAL.

### 2.1 The AST name-resolution gate, re-derived here (not taken on trust)

```
=== v2 body ===   setup-command name resolution: OK (182 loaded names, 0 unresolved)
                  RESULT: v2 PASSES gate (expected)
=== v1 body (real bug re-injected: _source_path_entries(BUNDLE_DIR)) ===
                  RESULT: CAUGHT -> FATAL: the rewritten setup command references
                  names it does not define: BUNDLE_DIR, _source_path_entries
```

### 2.2 The v1→v2 delta is EXACTLY the two documented fixes, and nothing else

Diffed the local v2 notebook against the live remote (v1). Cell types identical.
Cells differing: `[7, 8, 11, 13, 16]`. **ASCII-normalised (i.e. real) diffs: `[8]` only.**
Cells 7/11/13/16 differ solely by the documented em-dash cp1252 pull round-trip.

Within cell 8, of the 6 setup-rewrite anchors, **2 differ** and both are the fix:

- anchor 1 — `_lora_install_guard()` body: `for base in _source_path_entries(BUNDLE_DIR)` →
  self-resolving `for root in (Path('/kaggle/input'), …): root.rglob('inference/tools/vllm_runtime_lora_guard.py')`
- anchor 4 — call site: `_lora_install_guard()` → wrapped in `try/except` printing
  `LORA-CANARY guard=SKIPPED <exc>` and continuing

No other behavioural change. Fix 3 (the AST build gate) is build-side only, as designed.

---

## 3. Preflight — ALLOW / 0 / 0, and an honest note on HOW it was run

`scripts/preflight.py --family duck-harness` is **PULL-based** (check K1 shells out to
`kaggle kernels pull`), so in structural mode it **cannot see an unpushed notebook** —
`--local-notebook` only applies to `trusted-fork` mode. The smoke suite says this itself
(`lora_canary_smoke.py:456`: "preflight … PULLS the kernel (K1), so it cannot …"). Running it
verbatim today would have certified the **remote v1**, not v2.

So I drove preflight's **own check functions** (`structural_checks`, `load_baseline_notebook`,
`duck_diff_checks`, `host_gates`, `summarize`) against the local v2 artifact, with the same
`--baseline` and `--expect-diff-cells 2,6,8,14` the family uses:

```
==== LOCAL-EMULATED PREFLIGHT (family=duck-harness, host-gates=strict) ====
VERDICT: ALLOW   fails=0 warns=0 n/a=5
  [OK] K1: [LOCAL STAGE] candidate read from disk (unpushed; preflight's real K1 pulls)
  [--] K2/K4/K5/K6/K8: inapplicable to family 'duck-harness'          (5 N/A)
  [OK] K3: nbformat 4.4
  [OK] D1: war-eval baseline resolved: notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb
  [OK] D2: metadata.kaggle byte-identical to baseline — both ABSENT
  [OK] D3: cell shape matches baseline (17 cells, 8 code)
  [OK] D4: differing code cells = notebook cell(s) [2, 6, 8, 14]
           (code-cell ordinal(s) [0, 2, 3, 6]) — MATCHES --expect-diff-cells
  [OK] H1: GPU accelerator enabled (enable_gpu=true)
  [OK] H2: no 'arcprize.org (any subdomain)' calls
  [OK] H3: no writes to /kaggle/input
  [OK] H4: 4 dataset_source(s), 0 model_source(s) attached
```

**Differing code cells vs the pre-registered set: `[2, 6, 8, 14]` — exact match, zero extra,
zero missing.** H1–H4 identical in warn and strict mode (all OK, none vacuous — the slug is
in `DUCK_LINEAGE_SLUGS`, so H1/H4 are real checks, not `n/a`).

---

## 4. Adapter + environment integrity (verified against the LIVE Kaggle dataset)

Downloaded `canivel/arc3-lora-probe-adapters` (56.2 MB zip, free) and hashed it.

| | local build | **Kaggle (live)** | manifest | pinned in notebook cell 8 |
|---|---|---|---|---|
| `lora-noop` bytes | 41,962,184 | **41,962,184** | 41,962,184 | — |
| `lora-noop` sha256 | `d777d4c7a7ebec85cb7694eb3b069509540b4b31605f6c468d93f35d02319ead` | **MATCH** | `d777d4c7a7ebec85` | ✅ `d777d4c7a7ebec85` |
| `lora-probe` bytes | 41,962,184 | **41,962,184** | 41,962,184 | — |
| `lora-probe` sha256 | `d7d6918d01ae67f61b01661570c9c641c3e7184407d748d35a0b9f8005fdc9cf` | **MATCH** | `d7d6918d01ae67f6` | ✅ `d7d6918d01ae67f6` |
| `adapter_config.json` | — | **byte-identical**, `r=16` | — | rank asserted at runtime |

Layout on Kaggle is `lora-noop/` + `lora-probe/` subdirs — the layout the runtime resolver
requires and smoke D8 proves is loud-fail if flattened. Manifest also records
`nonzero_lora_B_modules`: noop **0**, probe **64** (smoke A4/A5/A6/A7 verify this locally).

**Metadata, byte-matched against the frozen duckfork family** (`feedback_kaggle_env_match`):

| field | frozen duckfork | canary v2 | match |
|---|---|---|---|
| `enable_gpu` | true | true | ✅ |
| `enable_internet` | false | false | ✅ |
| `enable_tpu` | false | false | ✅ |
| `docker_image` | `…byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c` | same | ✅ |
| `machine_shape` | `NvidiaRtxPro6000` | same | ✅ |
| `competition_sources` | `["arc-prize-2026-arc-agi-3"]` | same | ✅ |
| `kernel_type`/`language`/`is_private`/`keywords`/`model_sources`/`kernel_sources` | — | — | ✅ all |

Only `{id, title, code_file, dataset_sources}` differ — `dataset_sources` = the scored TRIPLE
unchanged + `canivel/arc3-lora-probe-adapters`, 4 total. The **live remote already shows all
4 survived** the v1 push (`feedback_kaggle_model_attach` — Kaggle drops unattachable sources
silently; it did not here).

---

## 5. Slot arithmetic, re-confirmed at 2026-08-15 10:29 EDT

Campaign budget: **2 kernel pushes per LOCAL day.**

| date | pushes | detail |
|---|---|---|
| 08-13 | 2 of 2 | EFFNOTE eval (slot 1); `arc3-b122-boot-canary` v1 (slot 2) |
| 08-14 | **3** (overrun) | b122 v2 (daily-loop session) · b122 v3 (122B agent, byte-identical duplicate) · **LoRA serve canary v1** |
| | | ⇒ ruling: the LoRA canary v1 overrun is **charged forward to 08-15 slot 1** ("LET IT RUN; count it against 08-15 slot 1") |
| **08-15** | **1 of 2 spent** (slot 1 = the charged-forward v1) | **slot 2 arithmetically FREE** |

**Independently confirmed against Kaggle, not just the log:** `kernels list -m --sort-by
dateRun` shows the most recent run of any kernel of ours is
`canivel/arc3-lora-serve-canary` at **2026-08-14 13:40:18 UTC**. **No kernel has been pushed
or run on 2026-08-15.** Its status is `KernelWorkerStatus.ERROR` (the v1 death).

**Idempotence, checked live BEFORE any push:**

```
LOCAL  v2 code-sha256 = ec487333a7e50da741b113747fed1be4d8ab55a5dbea509823095457bbdde2e3  cells=17
REMOTE    code-sha256 = 5ba90ae03a01c37f4adcf12b3154d3505f8a711376a35ad7ee9409b9c0852bc1  cells=17
IDEMPOTENCE: REMOTE != LOCAL -> push is NOT a duplicate
```

So: **slot 2 of 08-15 is free, and a push would not be a duplicate.** That is the arithmetic.
It is NOT the authorization — see §7.

---

## 6. DISCREPANCIES FOUND

### 6.1 ★★ THE TWO PUSH GUARDS DID NOT EXIST ON THIS LANE

The standing brief states the push path "now has TWO guards: `--confirm-push` (wrong time)
and an idempotence check that exits 3 if remote code already equals local (wrong actor). Do
not defeat either."

**Measured: neither guard was on this lane's push path.** A repo-wide search for
`kernels push` finds exactly **one** first-party script, `duck_eval/a17/b122_push_v2.sh`,
which is hardcoded to `KERNEL="canivel/arc3-b122-boot-canary"`, to the b122 notebook dir, to
the b122 builder/smoke/scorer, and to `if [ "$(date +%Y-%m-%d)" != "2026-08-14" ]` — a lane
that is **CLOSED**. Running it today refuses at step 0, and if it did not it would push the
wrong kernel.

**v1 was therefore pushed by an ad-hoc `kaggle kernels push` with no time interlock and no
idempotence check.** The guards were doctrine, not code, on the path that actually mattered.
The 08-14 lesson ("a push script is SHARED MUTABLE STATE") had been *written down* and not
*installed here*.

**Remediated (not run against the remote):** ported both guards verbatim into
`duck_eval/lora/lora_push_v2.sh`, plus a step 0b **ledger re-confirm** that mechanises §11.4,
plus a `--dry-run` mode so every gate can be exercised without a push. Observed:

```
$ bash duck_eval/lora/lora_push_v2.sh              -> REFUSING: pass --confirm-push … (exit 2)
$ bash duck_eval/lora/lora_push_v2.sh --dry-run
   0.  local date 2026-08-15  OK
   0b. ledger re-confirm -> printed today's slot/push lines  [THIS IS WHAT CAUGHT §7]
   1.  builder OK · 75 passed / 0 failed · selftest: 35 passed / 0 failed
   1b. remote differs from local -- push is not a duplicate
   DRY RUN COMPLETE. All pre-push gates passed. NOTHING WAS PUSHED.   (exit 0)
```

### 6.2 ★★★ THE AUTHORIZATION IS SUPERSEDED — see §7 (the headline)

### 6.3 ★ The v1 regression test is real but EPHEMERAL

The claim "regression-tested against the REAL v1 body (catches it)" is **true** — I
reproduced it (§2.1). But **nothing in the repo pins it.** There is no test anywhere that
exercises `_assert_names_resolve`; the 75-check smoke only checks token presence (`R11`:
`"_lora_install_guard()" in body and "vllm_runtime_lora_guard.py" in body and
"sitecustomize.py" in body`) — no scope assertion, no v1 fixture.

Mitigating: the gate runs inside `main()` before the notebook is written, so a *reintroduced
v1 bug* would `SystemExit` the builder and fail smoke `B1 builder exits 0` transitively.
Residual: a *weakened or removed gate* passes all 75 checks silently. Same shape as the bug
it was built to prevent — the missing check is one category over from the one that runs.

### 6.4 ★ The "scorer independently returns INFRA DEATH on v1" claim is not reproducible

No v1 output artifacts exist in the repo: `runs/kernel_pulls/` has no lora entry, and there
is no `lora_canary.json` anywhere on disk. The claim is consistent with the diagnosis and
with the scorer's precedence rules (which I read and which do implement it), but it **cannot
be re-run**. Given the log's own note that `kernels output` could not usefully retrieve the
log (multi-GB `/kaggle/working`), the scoring input was likely a hand-assembled fixture.
Not alarming; recorded so it is not cited as a reproduced result.

### 6.5 Residual, non-blocking: unbounded `rglob` in the guard

`_lora_install_guard()` now walks `Path('/kaggle/input').rglob('inference/tools/
vllm_runtime_lora_guard.py')` — over a mount that includes the ~36 GB model snapshot, the
wheelhouse and the taaf source tree — with **no timeout**, before the vLLM server starts. The
`try/except` wrapper makes it unable to *fail* the run; it does not bound its *wall time*.
Expected cost is seconds, but it is unmeasured on the real mount. Not a blocker.

---

## 7. ★★★★ THE DECISIVE FINDING — THE AUTHORIZATION IS SUPERSEDED

§11.4 (binding, this lane's own rule, written after the 08-14 overrun):

> **re-confirm slot availability from the ledger IMMEDIATELY before pushing, even under a
> live authorization. A conditional that was true when issued is not evidence that it is
> true now.**

I did. The **2026-08-15 section of `ITERATION_LOG.md`** — entered TODAY, *after* the 08-14
authorization quoted in my brief — contains:

> **★★★★ STOP + RESTART RESEARCH (principal order, 08-15). WE ARE ~#119 AND THE TOP HAS
> EXPLODED IN 48h.** … cstl 2.70, **Daniel Franzen 2.58 (NEW)**, Nikita Sorokin 2.10 (NEW),
> Yusaku Muroya 1.98, AbeLincoln1865 1.90 (NEW), YUTO KOJIMA 1.86, MLRush 1.75 (NEW) …
> Two days ago there was ONE team above 1.86; now there are SIX above 1.75.
>
> **ALL CURRENT LANES STOOD DOWN** pending the research restart. 122B parked (closed).
> **LoRA serve canary v2 NOT pushed. No slot spend until the research lands.**

This is precisely the failure mode §11.4 exists to catch, and it caught it: a lane agent
holding a live conditional from yesterday, while the governing constraint moved today. My own
task brief quotes the 08-14 authorization as "STANDING" — **the brief is stale.**

Note also the LB context makes the stand-down substantive, not procedural: **Daniel Franzen
at 2.58** is the co-author of *The LLM ARChitect* (fine-tuning + TTT + augmented inference +
candidate selection). The ledger's own reading is that this is "precisely the lane we
identified and have not executed". A serve canary is a *prerequisite* for that lane, which
arguably strengthens the case for it — **but that argument is the principal's to make, not
mine.** A peer's ledger entry is not my authorization to override, and neither is a stale
brief.

**⇒ PUSH IS NO-GO until the principal explicitly re-authorizes against today's ledger.**

---

## 8. Sealed decisive read — restated verbatim, before any data exists

> **`noop ≡ base` AND `probe ≠ base` ⇒ PASS. `probe ≡ base` ⇒ the adapter is being SILENTLY
> IGNORED** — the failure that would otherwise read as "LoRA didn't help" after a full
> training run.

Confirmed `duck_eval/lora/lora_serve_score.py` implements exactly this, unmodified by me:

| outcome | verdict | decisive? |
|---|---|---|
| `noop ≡ base` **and** `probe ≠ base` **and** all round-trip markers | `SERVE-PASS` | **True** |
| `probe ≡ base` | `SERVE-FAIL` — silently ignored | **True** |
| `noop ≢ base` | `SERVE-FAIL` — zero-delta adapter changed output ⇒ numerically unsound | **True** |
| explicit vLLM refusal signature | `SERVE-FAIL` — a refusal *is* the answer for this stack | **True** |
| **no differential reached** | **`INFRA DEATH`** | **False — a retry, never a verdict** |

Third state present and correct (`PASS, FAIL, INFRA = "SERVE-PASS", "SERVE-FAIL", "INFRA
DEATH"`; every INFRA return carries `"decisive": False`). Precedence is sealed and ordered:
(1) refusal → decisive FAIL; (2) infra signature with no differential → INFRA DEATH;
(3) truth table read **from the structured booleans** `noop_identical_to_base=… /
probe_differs_from_base=…`, **not** from the `differential=PASS` banner —
`if not banner: … "the evidence is INCONSISTENT and is not scored as a pass"`, and a banner
that contradicts the booleans is recorded as `CONTRADICTION` and **never** a PASS;
(4) a PASS additionally requires every serve round-trip marker; (5) throughput is a
**separate axis**, recomputed from raw tok/s, with the kernel's self-reported verdict only
cross-checked (`MISMATCH:` note) — never trusted.

Selftest observed proving the adversarial cases: `A1 forged/contradictory banner with
probe==base MUST NOT pass -> SERVE-FAIL`, `A4 a summary that claims verdict=PASS with no
differential in the log MUST NOT pass -> INFRA DEATH`, `A5 the silent null (adapter ignored,
everything else green) MUST NOT pass -> SERVE-FAIL`. Sealed constants (`ACTION_BAR 100.0`,
`TOK_S_27B 192.0`, `WINDOW_S 7920.0`, `TOKENS_PER_ACTION 3168.0`, `LORA_RANK 16`,
`ADAPTER_DS`, adapter shas/bytes) all assert equal to the builder's — 10 of the 35 selftests
are exactly that cross-check.

**Scope warning carried in every verdict:** this canary runs on the SCORED wheelhouse,
**vLLM 0.19.0**. A result here does not transfer to 0.24.0.

---

## 9. Bottom line

- **Artifact: GO.** Everything the log claimed about the artifact, I measured and it held —
  75/75, 35/35, 182/0, ALLOW/0 fails/0 warns, D4 `[2,6,8,14]`, adapters byte-exact against
  the live dataset, env byte-matched, v1→v2 delta is exactly the two fixes.
- **Push: NO-GO.** Superseded authorization (§7). Also, until today, no guarded push path
  existed on this lane at all (§6.1).
- **A NO-GO that saves a slot is a win.** This lane has spent two slots on self-inflicted
  faults. The third would have been spent against a stood-down order — and the check that
  caught it is the one this lane wrote for itself after the second.
