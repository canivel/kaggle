# Preflight host common-error gates (H1-H4) — hardening report — 2026-08-02

Additive, non-gating hardening of `scripts/preflight.py`, folding the Kaggle
host's "500 Submissions Analyzed — Common Errors" findings into an explicit,
zero-cost preflight checklist. Approved as today's brief §2 (NON-GATING, zero
cost). Source summary: `learnings/sweeps/discussions_2026-08-02.md` (thread
727119, host Greg Kamradt). Host's headline failure modes: silent stall (~⅓),
**GPU accelerator not enabled** (~20%), long tail of dataset-not-attached,
`three.arcprize.org` calls, and writes to read-only `/kaggle/input`.

---

## 1. Gates added

Four gates, applied to the notebook being preflighted (the fork in trusted-fork
mode). All are **ADDITIVE and OPT-IN** — see §3 for the invariant.

| ID | Check | Source of truth | Applies to |
|----|-------|-----------------|-----------|
| **H1** | GPU accelerator enabled (`enable_gpu: true`) | sibling `kernel-metadata.json` | GPU-required families (`duck`) |
| **H2** | No calls to `three.arcprize.org` | notebook code cells | all families |
| **H3** | No writes to read-only `/kaggle/input` | notebook code cells (write-intent regex; reads are allowed) | all families |
| **H4** | Required dataset/model sources attached (non-empty `dataset_sources` OR `model_sources`) | sibling `kernel-metadata.json` | dataset-loading families (`duck`) |

Family is derived from the slug (`_family_of`): a slug containing `duck` →
family `duck`, which is both GPU-required and dataset-required (it runs a local
vLLM server loading weights from attached datasets). Non-duck families report
H1/H4 as `OK … n/a`.

H3 matches **write-intent constructs** only (`%%writefile /kaggle/input`,
`open(...,'w'|'a'|'x')`, `to_csv`/`to_parquet`, `makedirs`/`mkdir`,
`shutil.copy*/move`, `Path(...).write_*`, `.write_text/.write_bytes` onto a
`/kaggle/input` path). A plain read from `/kaggle/input` (the normal case) does
NOT trip the gate.

### Severity model (the key safety property)

- **`--host-gates` (WARN mode, default when the flag is present):** a real,
  visible violation emits **WARN**. Verdict may become WARN but **never BLOCK**.
- **`--strict-host-gates` (DENY mode, implies `--host-gates`):** a real,
  visible violation emits **FAIL** (DENY / BLOCK).
- **Missing metadata is never a violation.** A bare `kaggle kernels pull` writes
  only the `.ipynb` (no `kernel-metadata.json`), and the frozen fork has an
  empty embedded `metadata.kaggle` block. When H1/H4 cannot see the metadata
  they need, they emit **WARN "cannot verify"** — even under `--strict-host-gates`.
- **Default (neither flag): the gates DO NOT RUN AT ALL.** Behaviour is
  byte-identical to the pre-change preflight.

Implementation: `host_gates(kernel, nb, kmeta, strict)` is a pure function
(easily unit-tested with no kaggle round-trip); `load_kernel_metadata(nb_path)`
finds the sibling `kernel-metadata.json`; both `run_preflight` and
`run_trusted_fork` gained a `host_gates_mode` param (`"off"|"warn"|"strict"`,
default `"off"`) and call the gate only when it is not `"off"`. CLI flags
`--host-gates` / `--strict-host-gates` added to `main()`.

---

## 2. Test results

### 2a. New host-gate unit tests — `scripts/test_host_gates.py` (NEW)

**21/21 PASS.** At least one positive and one negative per gate, plus the
severity contract:

- H1: positive (gpu on → OK), negative WARN (gpu off), negative FAIL (gpu off,
  strict), missing-metadata WARN even under strict, non-duck family n/a.
- H2: positive (clean → OK), negative WARN, negative FAIL (strict).
- H3: positive (read-only → OK), negative WARN (`open(...,'w')`), negative FAIL
  (`%%writefile`, strict), negative WARN (`to_parquet`).
- H4: positive (dataset attached → OK), positive (model_source counts → OK),
  negative WARN (none attached), negative FAIL (strict), missing-metadata WARN
  even under strict.
- Contract: `test_default_mode_never_fails` (all-bad nb + all-bad metadata,
  non-strict → 0 FAIL — the invariant that keeps ALLOW from flipping to BLOCK);
  `test_strict_mode_can_fail`; `_family_of` duck/non-duck.

Runner: `uv run python scripts/test_host_gates.py` → `21/21 passed`.

### 2b. Existing test matrices — all still green

- **`scripts/test_fingerprints.py`: 15/15 PASS** (shared import surface intact).
- **Original single-diff-graft matrix** (`preflight_singlediff_ext_2026-07-30.md`,
  the 7/7 matrix), re-run locally 2026-08-02 — all sealed verdicts reproduced:

  | # | case | expected | observed |
  |---|------|----------|----------|
  | a1 | strict, fork vs itself | T3 OK, not BLOCK | **T3 OK, WARN** ✔ |
  | a2 | strict, gate vs fork | T3 FAIL → BLOCK | **BLOCK** ✔ |
  | b  | single-diff, gate vs fork, N=1 pin+sha | T3 OK → WARN (local T4) | **T3 OK, WARN** ✔ |
  | c1 | N=0 | BLOCK | **BLOCK** ✔ |
  | c2 | wrong pin sha | BLOCK | **BLOCK** ✔ |
  | c2b| wrong pin file (raw boris_16) | BLOCK | **BLOCK** ✔ |

  (c3 = "+1 non-graft cell, N=1 → BLOCK" used an ephemeral temp fixture in the
  sealed doc; the N-budget path it exercises is unchanged and covered by c1.)

### 2c. End-to-end CLI smoke (gates enabled)

- `--host-gates` on staged arm-B (has sibling `kernel-metadata.json`): all four
  gates read real metadata → **H1 OK, H2 OK, H3 OK, H4 OK** (GPU on; 3 datasets).
- `--host-gates` on the frozen-fork bare pull (no sibling metadata): **H1 WARN,
  H2 OK, H3 OK, H4 WARN** — missing-metadata correctly WARNs, never FAILs.

  Both enabled runs yield verdict WARN (WARN checks present). This is exactly why
  the flag is opt-in and the production daemon does NOT pass it.

---

## 3. Regression verdicts (production modes — HARD CONSTRAINT 3)

Both run in the exact modes used in production, **without** the new flags (the
daemon builds its preflight CLI from queue fields only and never passes
`--host-gates`). Both **must** return exactly ALLOW.

### (a) Frozen-fork queue entry (trusted-fork, daemon invocation)

```
uv run python scripts/preflight.py --kernel canivel/arc3-duck-repro \
  --mode trusted-fork \
  --upstream jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner --json-only
```
→ **verdict ALLOW, n_fail 0, n_warn 0**, checks `T1 OK, T2 OK, T3 OK, T4 OK`.
**UNCHANGED.** ✔

### (b) Arm-B pinned single-diff (exact invocation from `entry_gate_discharge_2026-08-02.md` §4)

```
uv run python scripts/preflight.py --mode trusted-fork --kernel canivel/arc3-duck-gate \
  --upstream notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --max-diff-cells 1 --pin runs/fork_diff_boristown/cells/boris_16_gatebody.txt \
  --pin-sha 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b --json-only
```
→ **verdict ALLOW, n_fail 0, n_warn 0**, checks `T1 OK, T2 OK, T3 OK, T4 OK`.
**UNCHANGED.** ✔

**Both regressions return exactly ALLOW.** Nothing flips an existing ALLOW; the
boristown A/B may fire tonight through trusted-fork preflight unaffected.

---

## 4. Files changed / created

- **changed:** `scripts/preflight.py` — module-level host-gate config
  (`GPU_REQUIRED_FAMILIES`, `DATASET_REQUIRED_FAMILIES`, `FORBIDDEN_ENDPOINT`,
  `_KAGGLE_INPUT_WRITE_PATTERNS`); helpers `_family_of`, `load_kernel_metadata`,
  `_all_code_sources`, `_hostwarn`; pure `host_gates(...)`; `host_gates_mode`
  param threaded through `run_preflight` + `run_trusted_fork` (default `"off"`,
  strict paths byte-unchanged); CLI flags `--host-gates` / `--strict-host-gates`.
- **created:** `scripts/test_host_gates.py` — 21 unit tests (pos+neg per gate +
  severity contract), runnable standalone or via pytest.
- **created:** this report.

No kernel pushes, no queue changes, no cloud spend. kaggle==2.0.0 CLI for pulls.
```
