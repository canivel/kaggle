# Preflight single-diff-graft extension — BLOCKER 2(b) discharge report — 2026-07-30

Implements option **(b)** of BLOCKER 2 in
`learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md`: extends
`scripts/preflight.py` trusted-fork mode with `--max-diff-cells N --pin <ref>`
so it can **mechanically certify an audited single-cell graft** (arm B =
`canivel/arc3-duck-gate` = frozen fork + one inserted vLLM readiness-gate cell).

The strict `--mode trusted-fork` T3 requires code cells byte-identical to
upstream and so FAILs arm B by design. The extension is **additive and opt-in**:
with the new flags absent, every check behaves byte-for-byte as before (the
`daily_submit` daemon and other lanes are unaffected — verified below).

---

## 1. Semantics chosen

`run_trusted_fork(..., max_diff_cells, pin_path, pin_sha, local_notebook)`.
When `max_diff_cells is None` → **strict mode, unchanged**. When set → T3 is
relaxed to an *audited single-cell graft* certification. T3 passes iff **all** of:

- **(a) additions only** — the code-cell delta vs upstream (aligned by an LCS
  over `?`-normalised cell bodies, same normalisation strict T3 already uses)
  has **no deleted** upstream cell and **no rewritten** upstream cell. An
  upstream cell edited *in place* is tolerated **only** as a *banner-only
  additive mod*: the upstream body must survive **verbatim (`?`-normalised) as a
  contiguous substring** of the fork body (text appended/prepended, audited body
  untouched). This is exactly the cell-2 env-detect banner append the prereg
  names as "the only differing frozen code cell … run cell + solver surface
  byte-identical." Any edit that does not preserve the upstream body verbatim is
  a **rewrite** and disqualifies.
- **(b) inserted-cell budget** — the number of **inserted** cells (brand-new, no
  upstream peer) is `≤ N`. Banner-only additive mods do **not** consume the
  budget (they are not new cells); only true insertions do. For arm B: 1
  inserted (the gate) ≤ 1.
- **(c) pinned byte-span** — **each inserted cell** must contain the pinned
  byte-span as a **contiguous substring**: the exact bytes of the `--pin`
  reference file (decoded UTF-8) appear inside the inserted cell's source.
  Additive banner/telemetry lines before/after the span are allowed; the audited
  body itself is byte-preserved. `--pin-sha`, if given, asserts the reference
  file's own sha256 (guards against a swapped pin file) and is recorded as
  provenance in the T3 message.

Verdict stays **ALLOW / WARN / BLOCK** with the same output format and the same
`checks / n_fail / n_warn / verdict` JSON contract.

### The pin reference (important nuance)

The prereg names the pin as "`boris_16 sha`", but the raw
`runs/fork_diff_boristown/cells/boris_16_code.txt` is **NOT** itself a contiguous
substring of the gate cell: the gate re-emits boris's trailing bare
`wait_vllm_ready()` call **additively** (wrapped in `_gate_t0 = time.time()` …
latency telemetry), so only the **function-definition body** is contiguous. The
correct pin is therefore that contiguous audited byte-span = `boris_16_code.txt`
**minus its trailing `\n\n\nwait_vllm_ready()` bare call**, byte-derived and
shipped as a sibling of the sealed pin:

- `runs/fork_diff_boristown/cells/boris_16_gatebody.txt`
- sha256 `37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b` (582 bytes)

The sealed `boris_16_code.txt`
(sha256 `de370f0bf79f1065e65f907ce3a0e0209c1a89516c2579380c9e423c0b86eb8b`) was
**not modified**. Feeding it as the pin is correctly REJECTED (test c2b) — proof
the contiguous-substring rule is real, not cosmetic.

### Local-file mode

The original code is pull-based only. The prereg's "ported preflight T3" was run
by hand on staged files because arm B's slug is unpushed. This extension adds
`--local-notebook <path>` to substitute the staged `.ipynb` for the fork pull,
and an `--upstream` that resolves to a local `.ipynb` path is read from disk
instead of pulled. For a staged/unpushed fork the **T4 COMPLETE-status leg is
SKIPPED and reported WARN** (an unpushed kernel has no build status), matching
the prereg's note that the local T3 "has no upstream-pull / no COMPLETE-status
leg." T4 must be re-satisfied by a real push + COMPLETE build before fire.

---

## 2. Test matrix (all runtime-executed 2026-07-30)

| # | case | flags | expected | result |
|---|------|-------|----------|--------|
| a1 | strict, frozen fork **vs itself** | none | T3 OK, verdict not BLOCK | **PASS** — `T3 OK code cells identical to upstream (8 cells)` |
| a2 | strict, gate **vs** fork | none | T3 FAIL (original msg) | **PASS** — `T3 FAIL fork DIFFERS from upstream in 4 code cells …` → BLOCK |
| b  | single-diff, gate vs fork | `--max-diff-cells 1 --pin gatebody --pin-sha 37e30…` | T3 OK → ALLOW (WARN on T4 only) | **PASS** — `T3 OK audited single-cell graft OK: 1 inserted … pinned byte-span sha256=37e30…`; exit 0 |
| c1 | single-diff, **N=0** | `--max-diff-cells 0 --pin gatebody` | BLOCK | **PASS** — `T3 FAIL 1 inserted code cell(s) > --max-diff-cells 0` |
| c2 | **wrong pin sha** | `--max-diff-cells 1 --pin gatebody --pin-sha 0000…` | BLOCK | **PASS** — `T3 FAIL --pin file sha256 37e30… != asserted --pin-sha 0000… (pin file swapped?)` |
| c2b| **wrong pin file** (raw boris_16, non-contiguous) | `--max-diff-cells 1 --pin boris_16_code.txt` | BLOCK | **PASS** — `T3 FAIL 1 inserted cell(s) at fork code-idx [6] do NOT contain the pinned byte-span …` |
| c3 | **second added cell**, N=1 | temp fixture (+1 non-graft cell) `--max-diff-cells 1 --pin gatebody` | BLOCK | **PASS** — verdict BLOCK (2 inserted > 1) |

Backward-compat guard (runtime): `run_trusted_fork` signature keeps
`(kernel, upstream, version)` as the first three positional params and all new
params default `None`; a real kernel ref never matches `Path(ref).is_file()`, so
the pull path is untouched. `daily_submit.py` builds its preflight CLI from
queue fields only (`--kernel`, `--json-only`, optional `--mode/--upstream`) and
never passes the new flags → daemon behavior unchanged.

---

## 3. Exact discharge command (fire-condition 2(b))

```
uv run python scripts/preflight.py \
  --mode trusted-fork \
  --kernel canivel/arc3-duck-gate \
  --local-notebook notebooks/duckgate/arc3-duck-gate.ipynb \
  --upstream notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb \
  --max-diff-cells 1 \
  --pin runs/fork_diff_boristown/cells/boris_16_gatebody.txt \
  --pin-sha 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b
```

Observed verdict **WARN** (not BLOCK), exit code **0**:

```
[OK] T1: staged local arc3-duck-gate.ipynb
[OK] T2: staged local tufa-labs-duck-harness-june-30-milestone-winner.ipynb
[OK] T3: audited single-cell graft OK: 1 inserted code cell(s) (<= 1), 0 deleted,
         0 rewritten, 1 banner-only additive edit(s); each inserted cell contains
         pinned byte-span sha256=37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b
[!!] T4: fork staged from --local-notebook (unpushed) — build-status leg SKIPPED;
         push + re-run for the COMPLETE gate
```

This is the **local single-diff-invariant ALLOW** the prereg's fire-condition
2(b) requires. It does **not** on its own discharge the full fire condition 2:
the T4 COMPLETE leg still requires the pushed slug (re-run the same command with
`--kernel canivel/arc3-duck-gate` and **without** `--local-notebook` once the
kernel is pushed and its build is COMPLETE), and entry-gate #1 (BLOCKER 3)
remains separate. No push, queue, or seal was performed here.

---

## 4. Files changed / created

- **changed:** `scripts/preflight.py` — additive `--max-diff-cells / --pin /
  --pin-sha / --local-notebook` flags; `_tf_norm`, `_lcs_diff`, `_pin_bytes_and_sha`
  helpers; `run_trusted_fork` extended (strict path byte-unchanged).
- **created:** `runs/fork_diff_boristown/cells/boris_16_gatebody.txt` — the
  contiguous audited byte-span pin (boris_16 minus its trailing bare call),
  sha256 `37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b`.
  Sealed `boris_16_code.txt` untouched.
- **created:** this report.
