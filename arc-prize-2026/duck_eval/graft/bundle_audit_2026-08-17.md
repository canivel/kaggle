# GRAFT BUNDLE AUDIT — build-blocking verification for the authorized 08-18 slot-1 arm
**Date:** 2026-08-17 · **Scope:** read-only. Zero pushes (08-17 is 2/2 spent), zero submissions, zero spend.
**Provenance tags:** **[V]** verified by direct read/sha256/download this session · **[V-doc]** verbatim claim inside a verified artifact (their words, not independently reproduced) · **[INF]** inference · **[UNK]** unknown.
**Audited artifact:** `thtennant/taaf-kaggle-source-share-fork`, dataset version dated **2026-08-17 00:26:06**, 1,123 downloads, CC0. Local copy `…/scratchpad/conv_trace/taaf_banking/`.
**Comparator:** `jeroencottaar/taaf-kaggle-source-share` (MIT), downloaded fresh this session to `/tmp/graft/stock`. **[V]**

Manifest shas of the audited copy (path+sha256 manifest, sha256 of that manifest):
- full bundle, 89 files: **`df447f61caa181cca68049e28b139e02`** **[V]**
- `src/taaf-grafts/` only, 16 files: **`7705481551494b141d6a33ffec1d7a20`** **[V]**

---

## BUILD VERDICT: **NO-GO AS AUTHORIZED — GO-WITH-AMENDMENT**

The authorized arm was: frozen fork + share-fork dataset + cell-12 `install(bm, flags={efficiency, retry_guard, shortcircuit, banking, transfer})`, incumbent Qwen3.6, SCORE-primary read.

**The blocker is not the bundle. The bundle is clean, the engine is correct, and the swap is genuinely one variable. The blocker is that the two headline flags cannot fire on our harness — and this is provable from our own artifacts, not from theirs.**

> **★ `banking` gates on `run.state == "won"` (`banking_solver.py:180`, verbatim below). Across EVERY eval artifact this campaign has ever produced — 23 pulled kernel runs, 470 game-runs — the number of runs reaching `state == "won"` is ZERO. Best levels-completed on any single game, ever: 4 (of 6–10). `banking` has never had a trigger and would not have one tomorrow. [V]**
>
> **★ `transfer` requires clone siblings. Our eval harness runs `n_passes=1` over 25 games with 25 UNIQUE `game_id`s — zero clones. `transfer_solver.py`'s own docstring: *"a non-clone hidden set turns the entire stack into a measured no-op (gate phase D)"*. [V]/[V-doc]**

⇒ The arm as authorized reduces, by construction, to **`shortcircuit` alone**, and would return a REFUTE that carries **no information about the mechanism**. That is a precise repeat of the A9/warpack failure this campaign already booked once: *"our gate measured LEVELS on an offline bench where banking's conditions never fired."* The 08-17 conversion trace warned about the regime-mismatch class in §8; the reachability arithmetic was not run until now.

**Amendment recommended (details in §6): run the public, already-measured CONSERVATIVE stack — `{efficiency, retry_guard, shortcircuit, goalkeep, hudmask}` = thtennant's v19 setting — with `banking` and `transfer` explicitly OFF and asserted-absent, LEVELS-primary + score secondary.** It is still exactly one variable ("adopt the public graft floor"), every flag in it is reachable on our rail, and it attacks the constraint that the reachability finding exposes as binding.

---

## 1. Q1 — Is the fork's harness byte-identical to stock? **YES, exactly. [V]**

Recursive sha256 over both trees:

| | count |
|---|---:|
| files in stock | 73 |
| files in fork | 89 |
| only in fork | **16** — all under `src/taaf-grafts/` |
| only in stock | **0** |
| in both, **differing** sha256 | **0** |
| in both, identical | **73** |

**The fork is stock plus a 16-file additive layer and nothing else.** Not one stock byte is modified — harness (`ARC3-Inference`, `tufa-arc-agi-framework`), `setup_commands.json`, `teardown_commands.json`, `benchmark_initial.pkl`, `deploy_target.pkl`, `preamble.txt`, `taaf-kaggle-bundle.json` all identical. **[V]**

⇒ **Replacing** our `dataset_sources` entry `jeroencottaar/taaf-kaggle-source-share` with `thtennant/taaf-kaggle-source-share-fork` is a *true* one-variable change: it adds 16 importable files and alters nothing else in the environment. This is the cleanest experiment substrate this campaign has had.

## 2. Q2 — Which engine does the fork serve? **Qwen3.6 — our incumbent, byte-identically. [V]**

`setup_commands.json` is in the identical-73, so this follows from Q1, and is independently confirmed by reading it:

- `MODEL_OWNER='driessmit1'`, `MODEL_SLUG='vrfai-qwen3-6-27b-fp8-hf-snapshot'`, `SERVED_MODEL_NAME='vrfai/Qwen3.6-27B-FP8'` **[V]**
- `WHEELHOUSE='driessmit1/arc3-vllm-h100-wheelhouse-v3'` **[V]**
- `VLLM_MAX_MODEL_LEN=65536`, `ANALYZER_CONTEXT_WINDOW=32768`, `LOCAL_ANALYZER_YIELD_SECONDS='60'`, temp 0.6 / top_p 0.95 / top_k 20, `LOCAL_ANALYZER_ENABLE_THINKING='true'` **[V]**
- No Qwen3.8 reference anywhere in the fork. **[V]**

⇒ **No engine confound.** The Q38 channel (`jakobbrggen/*`) is a separate bundle lineage and is not involved. Our sealed Q38 REFUTE and this lane stay orthogonal, exactly as the trace claimed.

## 3. Q3 — The `install()` contract **[V]**

`install()` is **not** in `__init__.py` (which exports only `BankingHarnessSolver`) — it lives in **`taaf_grafts/composite.py:243`**. The trace's "one `install(bm, flags=...)` call" is correct in substance; the import path is `from taaf_grafts.composite import install`. Correcting it here because the wrong import is a silent-stock-fallback waiting to happen.

```python
def install(bm, flags: dict[str, Any] | None = None, *, expected_version: int | None = None) -> None:
    """Install the graft stack onto ``bm`` per ``flags`` (all default off).
    Blanket-guarded: on any error the original ``bm.solver`` is restored,
    every applied module-global patch is reverted, and a stock-fallback
    note is printed. Never raises."""
```

- **`GRAFTS_API_VERSION = 1`**; passing `expected_version=1` makes a future API bump **fail closed** to stock instead of silently changing meaning. **[V]** — this is our only lever against the un-pinnable dataset version (§5).
- **All flags default OFF.** Docstring: *"`install(bm, {})` is a proven no-op (the standing all-flags-off byte-identity gate — the 1.15-floor guarantee)."* **[V-doc]**
- **`transfer` implies `banking`** — `composite.py:172`: `want_banking = bool(flags.get("banking")) or want_transfer`. **[V]** Confirms the trace.
- **`hudmask` is NESTED under `goalkeep`** — `composite.py:297`: `if active.get("goalkeep") and flags.get("hudmask")`. Arming `hudmask` alone silently does nothing. **[V]** (Not in the trace; matters for flag-set design.)
- **★ UNKNOWN FLAG NAMES ARE SILENTLY IGNORED.** There is no validation — every flag is read by `flags.get(name)`. A typo (`"shortcircut"`, `"goal_keep"`) yields a **silent stock run that looks like a clean arm**. **[V]** This is the exact failure class that killed the Q38 low arm (a gate that could not tell "worked" from "did not bind"), so the prereg **must** assert the banner, not the source.
- **Verification surface that makes that possible:** `_print_banner(active)` emits `TAAF_GRAFTS FEATURES={...} API_VERSION=1`, plus deterministic per-flag lines — `[banking] armed`, `[recovery] armed`, `[goalkeep] armed`, `[hudmask] armed`, `[schema_*] armed` — explicitly *"so the commit-log gate can verify these grafts installed even on runs where they never fire (banking fires only on wins; recovery only on stalls)."* **[V-doc]** Note the authors state banking's win-dependence in their own code.
- **Failure mode:** one blanket `try/except` restores `bm.solver`, unwinds module-global patches in reverse, prints `[taaf_grafts] install failed -> stock: {err}`. Per-flag `ModuleNotFoundError` degrades that one flag only (forward-compat). **Never raises.** **[V]**
- Chain layers are analyzer wrappers with RetryGuard outermost; solver swap uses `from_solver` field-copy to survive `Benchmark.run`'s two deepcopies. **[V]**

## 4. Q5a — What the public kernels actually set (verbatim) **[V]**

| kernel | verbatim `flags` | banking? |
|---|---|---|
| `thtennant/arc3-duck-v19` (08-17) | `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "goalkeep": True, "hudmask": True}` | **no** |
| `thtennant/arc3-duck-v18` | v12 floor + `goalkeep` | **no** |
| `thtennant/arc3-duck-v12` (40 votes) | `efficiency, retry_guard, shortcircuit` | **no** |
| `kevin250304/arc3-duck-v9b-recovery-banking` (07-12) | `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "recovery": True, "banking": True}` | **yes** |

All four use the identical guarded idiom:
```python
try:
    from taaf_grafts.composite import install
    install(bm, flags={...})
except Exception as exc:  # noqa: BLE001 — any graft failure must fall back to stock
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {type(exc).__name__}: {exc}")
```
**★ And their `dataset_sources` settles Q4 empirically:** v18/v19 carry exactly **three** datasets — `['driessmit1/arc3-vllm-h100-wheelhouse-v3', 'thtennant/taaf-kaggle-source-share-fork', 'driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot']` — i.e. the fork **REPLACES** the stock source-share; it is never attached alongside it. **[V]**

*Corrects the trace's §3.3 characterisation in one detail: the distributor's public lineage never enables banking, but `kevin250304` — a different account — did, publicly, on 07-12. "The distributor does not run the exploit publicly" stands; "banking has been public since 07-12" also stands.*

## 5. Q4 — The bundle-discovery collision: **REAL. Resolution = REPLACE. [V]**

Our cell 6 resolves the bundle by marker:
```python
for marker in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"): return marker.parent
```
first match wins, and `rglob` order is not contractual. **The fork contains `taaf-kaggle-bundle.json` too** (identical bytes, per Q1). **[V]**

⇒ Attaching **both** bundles makes `BUNDLE_DIR` **ambiguous**, and — worse — the failure is asymmetric and silent: if it resolves to the *stock* dir, `sys.path` never gets `taaf-grafts`, the cell-12 import raises `ModuleNotFoundError`, the guarded `except` prints a one-line note, and **the run completes as a perfectly normal-looking stock run**. We would then score it as a REFUTE. **[INF, mechanism verified]**

**Resolution: REPLACE the stock ref with the fork ref, in BOTH places** — `kernel-metadata.json` `dataset_sources`, *and* cell 6's `DATASET_SOURCES` literal (index 0 must remain the bundle: cell 6 does `resolved = BUNDLE_DIR if i == 0`, and `setup_commands.json` resolves the wheelhouse/model by exact ref string, so those two refs must stay verbatim). Valid precisely because Q1 proved the harness identical, and independently corroborated by v18/v19's own metadata (§4). **[V]**

**Import path check:** cell 8 does `for repo in sorted((bundle_dir/"src").iterdir(), reverse=True): for candidate in (repo/"src", repo)` — so `src/taaf-grafts/` (no inner `src/`) is added as **itself**, making the inner package dir `taaf_grafts` importable as `taaf_grafts`. The hyphen/underscore split is intentional: the *directory* `taaf-grafts` is the sys.path entry, the *package* `taaf_grafts` is inside it. Reverse-sort also puts `taaf-grafts` ahead of `tufa-arc-agi-framework` and `ARC3-Inference`. **Import works. [V]**

**★ Residual risk that cannot be fully closed: Kaggle attaches the LATEST dataset version and kernel metadata cannot pin one.** The fork was republished **08-17 00:26** (mid-campaign, actively maintained). Tomorrow's run may attach bytes I did not audit. Mitigations, all cheap, all required: (i) pass `expected_version=1`; (ii) assert the FEATURES banner in the pull-back gate; (iii) record the audited manifest sha (§0) in the prereg and re-diff the bundle at push time.

## 6. Q5b — Reachability: the finding that blocks the arm **[V]**

### 6.1 `banking` — trigger never satisfied, 470/470

`banking_solver.py:180`, verbatim:
```python
if run is None or run.state != "won" or run.final_score is not None:
```
and the docstring: *"once a session's WIN is fully recorded, prune the winning trace per level … and replay it on a fresh play of the same card."* **[V-doc]**

Measured against every `benchmark.json` in `runs/kernel_pulls/` (23 artifacts, the whole campaign):

| | value |
|---|---:|
| total game-runs | **470** |
| runs with `state == "won"` | **0** |
| distinct terminal states observed | `gave_up` (466), `cancelled` (4) |
| best `levels_completed` on any single game | **4** (games have 6–10 levels) |
| per-run lc totals over 25 games | 10 … 22 (baseline trio 18 / 19 / 21) |

**Our agent has never won a single card, in any configuration, in the entire recorded history of this campaign.** **[V]**

### 6.2 `transfer` — premise false on our eval rail
Our eval runs are `n_passes=1`, 25 game-runs, **25 unique `game_id`s** → no clone siblings, so the family store never has a consumer. `transfer_solver.py` says a fingerprint miss makes *"every store call a no-op"* and *"a non-clone hidden set turns the entire stack into a measured no-op."* **[V]/[V-doc]**
*Note the asymmetry the trace already flagged: transfer's premise (110 runs = 25 games cloned round-robin) is asserted by the bundle and is plausible for the COMPETITION rerun, but it is **[UNK]** — we have never verified it on a scored run, and we cannot verify it from an eval build. Banking's premise (max-over-plays) is universal engine mechanics and would hold on the private twin; transfer's may not.*

### 6.3 What this means beyond the arm — the strategic read
Banking multiplies the score of cards you **already win**; it is a *denominator* exploit on cleared content. We clear 1–2 levels of 6–10 and win nothing. **So for us the field's recipe is not a shortcut past our problem — its payoff is gated behind exactly the capability we lack.** The honest reframe: the 1.33 → 2.00 gap is not (for us) primarily an exploit gap; the exploit is monetizable only after capability arrives. This also retro-explains our own July result with far more force than N5 did: `war_eval_v1` (our warpack banking arm) scored the highest lc of any run on disk (22) and still banked nothing — **because it had zero wins to bank, not because the lane was wrong.** **[V]/[INF on the reframe]**

### 6.4 Downside risks of the amended flag set, named
- `shortcircuit`: authors claim *"provably monotonic non-decreasing score, default-off, degrades to stock"* **[V-doc]** — plausible but **not independently verified by us [UNK]**; it can only remove actions it has twice confirmed to be no-ops.
- `goalkeep`/`hudmask`: module-global monkey-patches of `ToolAgent` internals; they change the agent's **prompt content** every turn, so they are a genuine behaviour change and could cut either way on levels. Their revert thunks are unwound on install failure. **[V]**
- Wall-clock: `transfer` contractually never extends per-game budget **[V-doc]**; `goalkeep`'s per-turn digest adds prompt tokens, which is a real (small) time/context cost and interacts with the 31,744-token context ceiling we already documented. Watch the window-drift gate.
- Universal: an unknown-flag typo or a bundle-version bump both produce a **silent stock run**. The banner assert is the only defence.

---

## 7. EXACT PAYLOAD FOR THE AMENDED ARM

`kernel-metadata.json` — `dataset_sources` (order preserved from our frozen fork, stock ref **replaced**):
```json
"dataset_sources": [
  "thtennant/taaf-kaggle-source-share-fork",
  "driessmit1/arc3-vllm-h100-wheelhouse-v3",
  "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
]
```
Cell 6 — the same single-token replacement, index 0 preserved:
```python
DATASET_SOURCES = ["thtennant/taaf-kaggle-source-share-fork", "driessmit1/arc3-vllm-h100-wheelhouse-v3", "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]
```
Cell 12:
```python
try:
    from taaf_grafts.composite import install
    install(bm, flags={"efficiency": True, "retry_guard": True, "shortcircuit": True,
                       "goalkeep": True, "hudmask": True}, expected_version=1)
except Exception as exc:  # noqa: BLE001 — any graft failure must fall back to stock
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {type(exc).__name__}: {exc}")
```
Touched cells: **[6, 12] + metadata** (the trace predicted [12] + metadata; cell 6 is required too, so the preflight D4 allowlist must be `[6,12]`).

**Mandatory gate assertions (the Q38-low lesson: assert the REQUESTED config's positive marker, and fail on the wrong arm's marker):**
1. banner present: `TAAF_GRAFTS FEATURES=` … `API_VERSION=1`
2. `[goalkeep] armed` AND `[hudmask] armed` present
3. `efficiency`, `retry_guard`, `shortcircuit` present in the FEATURES dict
4. **`[banking] armed` ABSENT and `banking`/`transfer` absent from FEATURES** — the arm is defined by their exclusion
5. `[taaf_grafts] install failed -> stock` and `cell-12 graft failed` both ABSENT
6. re-diff the attached bundle against manifest sha `df447f61caa181cca68049e28b139e02`; any mismatch → re-audit before reading
Failing 1–5 ⇒ **INFRA DEATH (not decisive)**, never a REFUTE.

## 8. WHAT REMAINS UNKNOWN
- Whether the agent wins any card on the **competition** 110-run rerun (no logs retained) → whether banking is reachable *there*. **[UNK]** This is the one cheap question that would reopen banking, and it is answerable from a rerun's log, not from an eval.
- Whether the 110 scored runs really are 25 games cloned round-robin. **[UNK]**
- `shortcircuit`'s monotonicity claim, independently. **[UNK]**
- Per-team attribution for any 2.0+ team. **[UNK]** — unchanged from the trace.
