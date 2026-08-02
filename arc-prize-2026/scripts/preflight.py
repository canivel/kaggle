"""Pre-flight validator for ARC-AGI-3 Kaggle kernel submissions.

The supervisor that the daily_submit daemon MUST call before any
`kaggle competitions submit`. Catches structural drift between our
candidate kernel and the known-working `canivel/arc3-baseline`.

WHY: Between 2026-05-26 and 2026-06-28 we burned 5 quota slots on ERRORs
(v45, v62, v63, v64, v65). Build always succeeded, rerun always ERRORed.
Root cause (found 2026-06-28): the rerun cell's `agents/__init__.py` and
`.env` content drifted from baseline:
  - Missing imports: Swarm, Playback, Random, load_dotenv
  - Missing env vars: SCHEME, HOST, PORT
  - Wrong ARC_API_KEY value (`arc-agi-3` instead of `test-key-123`)
Any of these crashes the swarm runner that orchestrates 25 concurrent games.

CHECKS (all deterministic — no LLM round-trip needed):

  K1. Kernel pulls cleanly from Kaggle
  K2. Notebook has metadata.kaggle block with dataSources entry
  K3. nbformat version (warn if drift from baseline 4.4)
  K4. The rerun cell contains all required imports for agents/__init__.py:
        - `from .agent import Agent, Playback`
        - `from .swarm import Swarm`
        - `from .templates.random_agent import Random`
        - `from dotenv import load_dotenv`
        - `load_dotenv()`
        - `AVAILABLE_AGENTS` dict containing both `random` and `myagent`
  K5. The rerun cell contains all required .env keys:
        SCHEME, HOST, PORT, ARC_API_KEY, ARC_BASE_URL, OPERATION_MODE, RECORDINGS_DIR
  K6. The rerun cell invokes `main.py --agent myagent`
  K7. dataset_sources from kernel-metadata is mirrored in
      notebook metadata.kaggle.dataSources (Kaggle should auto-sync; verify)
  K8. The %%writefile of /kaggle/working/my_agent.py is present

USAGE:
  uv run python scripts/preflight.py --kernel canivel/arc3-execwm-v2 --version 1
  # exit code 0 = OK, 1 = BLOCK; prints findings as JSON
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Required string fragments in agents/__init__.py written by the rerun cell.
REQUIRED_INIT_FRAGMENTS = [
    "from .agent import Agent, Playback",
    "from .swarm import Swarm",
    "from .templates.random_agent import Random",
    "from dotenv import load_dotenv",
    "load_dotenv()",
    '"random": Random',
    '"myagent": MyAgent',
]

# Required keys in .env written by the rerun cell.
REQUIRED_ENV_KEYS = [
    "SCHEME=",
    "HOST=",
    "PORT=",
    "ARC_API_KEY=",
    "ARC_BASE_URL=",
    "OPERATION_MODE=",
    "RECORDINGS_DIR=",
]

# Required commands in the rerun cell.
REQUIRED_RUN_FRAGMENTS = [
    "KAGGLE_IS_COMPETITION_RERUN",
    "http://gateway:8001/api/games",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
    "%%writefile /kaggle/working/my_agent.py" if False else None,  # checked elsewhere
    "main.py --agent myagent",
]
REQUIRED_RUN_FRAGMENTS = [f for f in REQUIRED_RUN_FRAGMENTS if f]


# ---------------------------------------------------------------------------
# Host common-error gates (H1-H4).
#
# Derived from the Kaggle host's "500 Submissions Analyzed - Common Errors"
# post (learnings/sweeps/discussions_2026-08-02.md). The host found the top
# failure modes are (a) GPU-required code with the accelerator NOT enabled
# (~20%), and a long tail of: dataset/model not attached, calling
# three.arcprize.org instead of the in-notebook endpoint, and writes to the
# read-only /kaggle/input.
#
# These gates are ADDITIVE and OPT-IN:
#   * They run ONLY when --host-gates (or --strict-host-gates) is passed.
#   * With the flags absent, preflight behaviour is BYTE-IDENTICAL to before
#     (the daily_submit daemon, the frozen-fork lane, and the arm-B single-diff
#     invocation never pass these flags, so their ALLOW verdicts are preserved).
#   * When enabled they emit WARN by default (verdict may become WARN but never
#     BLOCK). --strict-host-gates escalates a genuine violation to FAIL/DENY.
#   * When a gate cannot see the metadata it needs (e.g. a bare `kaggle kernels
#     pull` writes only the .ipynb, no kernel-metadata.json, and this fork has
#     no embedded metadata.kaggle block) it emits WARN "cannot verify" even
#     under --strict — a missing source is never treated as a violation.
# ---------------------------------------------------------------------------

# Kernel families that REQUIRE a GPU accelerator (H1) and that load code/weights
# from attached datasets (H4). The duck lineage runs a local vLLM server.
GPU_REQUIRED_FAMILIES = ("duck",)
DATASET_REQUIRED_FAMILIES = ("duck",)

# Endpoint that must NOT be called from inside a competition rerun (H2). The
# graded run must talk to the in-notebook / local gateway, not the public API.
FORBIDDEN_ENDPOINT = "three.arcprize.org"


def fail(checks, code, msg):
    checks.append({"check": code, "status": "FAIL", "message": msg})


def warn(checks, code, msg):
    checks.append({"check": code, "status": "WARN", "message": msg})


def ok(checks, code, msg=""):
    checks.append({"check": code, "status": "OK", "message": msg})


def find_cell_containing(cells, needle):
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if needle in src:
            return i, src
    return -1, ""


def _family_of(kernel: str) -> str:
    """Coarse family tag from the kernel slug (e.g. 'canivel/arc3-duck-repro'
    -> 'duck'). Used to decide which host gates apply to which kernel."""
    slug = (kernel or "").split("/")[-1].lower()
    if "duck" in slug:
        return "duck"
    return slug


def load_kernel_metadata(nb_path: Path) -> dict | None:
    """Return the sibling kernel-metadata.json (dict) for a pulled/staged
    notebook, or None if absent. A bare `kaggle kernels pull` does NOT write
    this file, so None is common and must be handled as "cannot verify"."""
    if nb_path is None:
        return None
    cand = Path(nb_path).parent / "kernel-metadata.json"
    if not cand.is_file():
        return None
    try:
        return json.loads(cand.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None


def _all_code_sources(nb: dict) -> list[str]:
    return ["".join(c.get("source", []))
            for c in nb.get("cells", [])
            if c.get("cell_type") == "code"]


# Patterns that indicate a WRITE into the read-only /kaggle/input tree (H3).
# We deliberately match write-intent constructs, not any mention of the path
# (reads from /kaggle/input are normal and must not trip the gate).
_KAGGLE_INPUT_WRITE_PATTERNS = [
    re.compile(r"""%%writefile\s+/kaggle/input\b"""),
    re.compile(r"""open\s*\(\s*["'][^"']*?/kaggle/input[^"']*?["']\s*,\s*["'][^"']*?[wax]"""),
    re.compile(r"""\bto_csv\s*\(\s*["'][^"']*?/kaggle/input"""),
    re.compile(r"""\bto_parquet\s*\(\s*["'][^"']*?/kaggle/input"""),
    re.compile(r"""\b(?:os\.)?(?:makedirs|mkdir)\s*\(\s*["'][^"']*?/kaggle/input"""),
    re.compile(r"""\bshutil\.(?:copy\w*|move)\s*\([^)]*?["'][^"']*?/kaggle/input[^"']*?["']\s*\)"""),
    re.compile(r"""\bPath\s*\(\s*["'][^"']*?/kaggle/input[^"']*?["']\s*\)\s*\.write_"""),
    re.compile(r"""["'][^"']*?/kaggle/input[^"']*?["']\s*\)?\s*\.write_(?:text|bytes)\s*\("""),
]


def _hostwarn(strict: bool, checks: list, code: str, msg: str):
    """Emit FAIL under --strict-host-gates, else WARN. Used for genuine
    host-gate violations (a real bad state we can see)."""
    if strict:
        fail(checks, code, msg)
    else:
        warn(checks, code, msg)


def host_gates(kernel: str, nb: dict, kmeta: dict | None,
               strict: bool = False) -> list[dict]:
    """Additive host common-error gates H1-H4 (see module docstring). Pure
    function over the already-parsed notebook + optional kernel-metadata.json.
    Returns a list of check dicts (OK / WARN / FAIL). Never raises on missing
    inputs. Only emits FAIL when `strict` is True AND a real violation is
    visible; a missing metadata source is always a WARN regardless of strict."""
    checks: list[dict] = []
    fam = _family_of(kernel)
    code_srcs = _all_code_sources(nb or {})

    # H1: GPU accelerator enabled (for families that require it).
    if fam in GPU_REQUIRED_FAMILIES:
        if kmeta is None:
            warn(checks, "H1", "cannot verify GPU accelerator: no sibling "
                               "kernel-metadata.json (bare pull); H1 skipped")
        elif kmeta.get("enable_gpu") is True:
            ok(checks, "H1", "GPU accelerator enabled (enable_gpu=true)")
        else:
            _hostwarn(strict, checks, "H1",
                      f"family '{fam}' REQUIRES a GPU but enable_gpu="
                      f"{kmeta.get('enable_gpu')!r} in kernel-metadata.json "
                      f"(host: ~20% of failed submissions = GPU not enabled)")
    else:
        ok(checks, "H1", f"family '{fam}' not GPU-required; H1 n/a")

    # H2: no calls to the public three.arcprize.org endpoint.
    hits = [i for i, s in enumerate(code_srcs) if FORBIDDEN_ENDPOINT in s]
    if hits:
        _hostwarn(strict, checks, "H2",
                  f"forbidden endpoint '{FORBIDDEN_ENDPOINT}' referenced in "
                  f"code cell(s) {hits}; competition rerun must use the "
                  f"in-notebook/local gateway, not the public API")
    else:
        ok(checks, "H2", f"no '{FORBIDDEN_ENDPOINT}' calls")

    # H3: no writes into the read-only /kaggle/input tree.
    write_hits = []
    for i, s in enumerate(code_srcs):
        if any(p.search(s) for p in _KAGGLE_INPUT_WRITE_PATTERNS):
            write_hits.append(i)
    if write_hits:
        _hostwarn(strict, checks, "H3",
                  f"write into read-only /kaggle/input detected in code "
                  f"cell(s) {write_hits}; /kaggle/input is read-only "
                  f"(host common error)")
    else:
        ok(checks, "H3", "no writes to /kaggle/input")

    # H4: required dataset sources attached (for families that load code/weights
    # from datasets).
    if fam in DATASET_REQUIRED_FAMILIES:
        if kmeta is None:
            warn(checks, "H4", "cannot verify dataset attachment: no sibling "
                               "kernel-metadata.json (bare pull); H4 skipped")
        else:
            ds = kmeta.get("dataset_sources") or []
            ms = kmeta.get("model_sources") or []
            if ds or ms:
                ok(checks, "H4",
                   f"{len(ds)} dataset_source(s), {len(ms)} model_source(s) attached")
            else:
                _hostwarn(strict, checks, "H4",
                          f"family '{fam}' loads code/weights from datasets but "
                          f"dataset_sources AND model_sources are BOTH empty in "
                          f"kernel-metadata.json (host: 'dataset not attached')")
    else:
        ok(checks, "H4", f"family '{fam}' not dataset-required; H4 n/a")

    return checks


def run_preflight(kernel: str, version: int | None,
                  host_gates_mode: str = "off") -> dict:
    checks: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="preflight-"))
    try:
        # K1: pull kernel
        r = subprocess.run(
            ["kaggle", "kernels", "pull", kernel, "-p", str(tmp)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            fail(checks, "K1", f"kaggle pull failed: {r.stderr.strip()[-300:]}")
            return summarize(kernel, version, checks)
        # find the .ipynb
        ipynbs = list(tmp.glob("*.ipynb"))
        if not ipynbs:
            fail(checks, "K1", f"no .ipynb found in {tmp}")
            return summarize(kernel, version, checks)
        nb = json.loads(ipynbs[0].read_text(encoding="utf-8"))
        ok(checks, "K1", f"pulled {ipynbs[0].name}")

        # K2: metadata.kaggle block with dataSources
        meta = nb.get("metadata", {})
        kg = meta.get("kaggle")
        if not kg:
            fail(checks, "K2", "metadata.kaggle block is MISSING — "
                              "competition gateway will not spin up")
        else:
            ds = kg.get("dataSources") or []
            if not ds:
                fail(checks, "K2", "metadata.kaggle.dataSources is EMPTY")
            else:
                comp_present = any(
                    d.get("sourceType") == "competition" for d in ds
                )
                if not comp_present:
                    fail(checks, "K2",
                         f"no competition entry in dataSources: {ds!r}")
                else:
                    ok(checks, "K2", f"dataSources OK ({len(ds)} entries)")

        # K3: nbformat
        nbf = f"{nb.get('nbformat')}.{nb.get('nbformat_minor')}"
        if nbf != "4.4":
            warn(checks, "K3", f"nbformat {nbf} != baseline 4.4 (often harmless)")
        else:
            ok(checks, "K3", f"nbformat {nbf}")

        # K4: rerun cell __init__.py imports
        cells = nb.get("cells", [])
        idx, run_src = find_cell_containing(cells, "KAGGLE_IS_COMPETITION_RERUN")
        if idx < 0:
            fail(checks, "K4", "no rerun cell (KAGGLE_IS_COMPETITION_RERUN) found")
        else:
            missing = [f for f in REQUIRED_INIT_FRAGMENTS if f not in run_src]
            if missing:
                fail(checks, "K4",
                     f"agents/__init__.py write block is MISSING required imports: {missing}")
            else:
                ok(checks, "K4", "agents/__init__.py imports OK")

            # K5: .env keys
            env_missing = [k for k in REQUIRED_ENV_KEYS if k not in run_src]
            if env_missing:
                fail(checks, "K5",
                     f".env write block is MISSING required keys: {env_missing}")
            else:
                ok(checks, "K5", ".env keys OK")
            # Also check ARC_API_KEY value matches baseline
            m = re.search(r"ARC_API_KEY=([^\n]+)", run_src)
            if m:
                val = m.group(1).strip()
                if val != "test-key-123":
                    warn(checks, "K5b",
                         f"ARC_API_KEY={val!r} differs from baseline 'test-key-123' "
                         f"(may not matter, but baseline is the working reference)")
                else:
                    ok(checks, "K5b", "ARC_API_KEY value matches baseline")

            # K6: main.py invocation
            run_missing = [f for f in REQUIRED_RUN_FRAGMENTS if f not in run_src]
            if run_missing:
                fail(checks, "K6", f"rerun cell missing fragments: {run_missing}")
            else:
                ok(checks, "K6", "rerun cell fragments OK")

        # K7: %%writefile of my_agent.py
        idx2, _ = find_cell_containing(cells, "%%writefile /kaggle/working/my_agent.py")
        if idx2 < 0:
            fail(checks, "K8", "no %%writefile /kaggle/working/my_agent.py cell")
        else:
            ok(checks, "K8", "my_agent.py writefile cell present")

        # H1-H4: additive host common-error gates (opt-in).
        if host_gates_mode != "off":
            kmeta = load_kernel_metadata(ipynbs[0])
            checks.extend(host_gates(kernel, nb, kmeta,
                                     strict=(host_gates_mode == "strict")))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return summarize(kernel, version, checks)


def summarize(kernel, version, checks):
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    return {
        "kernel": kernel,
        "version": version,
        "checks": checks,
        "n_fail": len(fails),
        "n_warn": len(warns),
        "verdict": "BLOCK" if fails else ("WARN" if warns else "ALLOW"),
    }


def _tf_norm(s: str) -> str:
    # Kaggle pull/push round-trips mangle non-ASCII (em dashes arrive as
    # U+FFFD, or as 3-char mojibake when the CLI reads UTF-8 as cp1252).
    # Compare on ASCII skeleton with ?-runs collapsed so both corruptions
    # of one char compare equal.
    s = s.encode("ascii", errors="replace").decode("ascii")
    return re.sub(r"\?+", "?", s)


def _lcs_diff(fork_cells: list[str], up_cells: list[str]) -> dict:
    """Classify the code-cell delta of `fork_cells` vs `up_cells`, using a
    longest-common-subsequence alignment on the (normalised) cell bodies.

    Returns
      {inserted:      [(fork_idx, raw_src)],   # brand-new cells (no upstream peer)
       deleted:       [up_idx],                # upstream cells with no fork peer
       additive_mods: [(fork_idx, up_idx, raw_src)],  # upstream body PRESERVED as a
                                                       # contiguous substring (banner
                                                       # append/prepend only)
       rewrites:      [(fork_idx, up_idx)]}     # upstream body NOT preserved (real edit)

    `raw_src` is the un-normalised fork source so the pin byte-span check runs
    on exact bytes.

    A "clean single-cell graft" (arm B) is: inserted == the graft cell(s),
    deleted == [], rewrites == [], and additive_mods are banner-only appends
    whose upstream body is byte-preserved. Upstream must appear verbatim (module
    additive banners) as a subsequence of the fork; a deletion or a rewrite of
    an existing upstream cell disqualifies the graft. This is a strict superset
    of strict T3's ?-normalisation tolerance.
    """
    fn = [_tf_norm(s) for s in fork_cells]
    un = [_tf_norm(s) for s in up_cells]
    n, m = len(fn), len(un)
    # LCS table over normalised cell bodies.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if fn[i] == un[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    raw_added: list[int] = []   # fork idxs not in the common subsequence
    raw_deleted: list[int] = []  # up idxs not in the common subsequence
    i = j = 0
    while i < n and j < m:
        if fn[i] == un[j]:
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            raw_added.append(i)
            i += 1
        else:
            raw_deleted.append(j)
            j += 1
    while i < n:
        raw_added.append(i)
        i += 1
    while j < m:
        raw_deleted.append(j)
        j += 1

    inserted: list[tuple[int, str]] = []
    additive_mods: list[tuple[int, int, str]] = []
    rewrites: list[tuple[int, int]] = []
    deleted: list[int] = list(raw_deleted)
    # Pair a leftover added with a leftover deleted 1:1 as an in-place edit of an
    # existing cell (banner append/prepend keeps the same slot). Whatever the
    # editor did, classify the pair by whether the upstream body survives verbatim
    # (normalised) as a contiguous substring of the fork body: if so it is an
    # additive banner mod (allowed); otherwise a rewrite (disqualifying).
    for fi in raw_added:
        if deleted:
            uj = deleted.pop(0)
            if un[uj] and un[uj] in fn[fi]:
                additive_mods.append((fi, uj, fork_cells[fi]))
            else:
                rewrites.append((fi, uj))
        else:
            inserted.append((fi, fork_cells[fi]))
    return {"inserted": inserted, "deleted": deleted,
            "additive_mods": additive_mods, "rewrites": rewrites}


def _pin_bytes_and_sha(pin_path: Path) -> tuple[bytes, str]:
    b = pin_path.read_bytes()
    import hashlib
    return b, hashlib.sha256(b).hexdigest()


def run_trusted_fork(kernel: str, upstream: str, version: int | None,
                     max_diff_cells: int | None = None,
                     pin_path: Path | None = None,
                     pin_sha: str | None = None,
                     local_notebook: Path | None = None,
                     host_gates_mode: str = "off") -> dict:
    """Trusted-fork mode: for unmodified forks of proven public kernels
    (e.g. the Milestone-1 winner). Structural baseline checks don't apply —
    instead verify:
      T1. Fork pulls cleanly (or is staged from --local-notebook)
      T2. Upstream pulls cleanly (or is staged from a local path)
      T3. Fork's code cells are IDENTICAL to upstream's (metadata may differ)
      T4. Fork's latest build status is COMPLETE

    SINGLE-DIFF GRAFT EXTENSION (2026-07-30, discharges BLOCKER 2(b) of the
    boristown A/B prereg). When `max_diff_cells` is not None, T3's byte-identity
    requirement is RELAXED to a mechanically-certified *audited single-cell
    graft*. All of the following must hold for T3 to pass:
      (a) the delta vs upstream is ADDITIONS ONLY: no upstream cell is deleted,
          and no upstream cell is rewritten. An upstream cell that is edited
          in place is tolerated ONLY if it is a "banner-only additive mod" —
          the upstream body survives verbatim (?-normalised) as a contiguous
          substring of the fork body, i.e. text was appended/prepended but the
          audited body was not changed. (This is the cell-2 env-detect banner
          append the prereg names as the sole differing frozen code cell; the
          run cell + solver surface stay byte-identical.) Any edit that does
          not preserve the upstream body verbatim is a rewrite and disqualifies.
      (b) the number of INSERTED cells (brand-new, no upstream peer) is
          <= max_diff_cells;
      (c) each INSERTED cell contains the pinned byte-span as a CONTIGUOUS
          substring: the exact bytes of --pin's reference file (decoded UTF-8)
          appear inside the inserted cell's source. Additive lines (banner
          prints, telemetry) before/after the pinned span are allowed; the
          audited body itself is byte-preserved. --pin-sha, if given, asserts
          the reference file's own sha256 (guards against the pin file being
          swapped) and is the value recorded as provenance in the check message.

    NOTE on the pin reference: the prereg names the pin as "boris_16 sha", but
    the raw runs/fork_diff_boristown/cells/boris_16_code.txt is NOT itself a
    contiguous substring of the gate cell — the gate re-emits boris's trailing
    bare `wait_vllm_ready()` call ADDITIVELY (wrapped in latency telemetry), so
    only the function-definition body is contiguous. The pin file must therefore
    be that contiguous audited byte-span (boris_16 minus its trailing bare call,
    byte-derived, shipped as runs/fork_diff_boristown/cells/boris_16_gatebody.txt,
    sha256 37e3018108c7058ce3295dd27d1a3baa3548a311d28f3e99c43be131bd99078b).

    When `max_diff_cells is None` the behaviour is BYTE-IDENTICAL to the
    original strict mode (the daemon and other lanes depend on this).

    LOCAL STAGING (--local-notebook / local upstream): the original code is
    pull-based only. If `local_notebook` is given, the fork side substitutes
    that staged .ipynb for the `kaggle kernels pull` (the arm-B slug is unpushed
    at prereg time). Likewise an `upstream` that resolves to an existing local
    .ipynb path is read from disk instead of pulled. The COMPLETE-status leg
    (T4) is SKIPPED for a staged/unpushed fork and reported as a WARN, since an
    unpushed kernel has no build status — matching the prereg's note that the
    local single-diff T3 "has no upstream-pull / no COMPLETE-status leg."
    """
    checks: list[dict] = []
    tmp = Path(tempfile.mkdtemp(prefix="preflight-tf-"))
    fork_is_local = local_notebook is not None
    try:
        ok_pulls = True
        nbs = {}
        nb_paths: dict[str, Path] = {}  # tag -> on-disk .ipynb (for H1/H4 sibling metadata)
        # (tag, ref, local_path_or_None) — a local path short-circuits the pull.
        up_local = Path(upstream) if upstream and Path(upstream).is_file() else None
        sources = (
            ("T1", kernel, local_notebook),
            ("T2", upstream, up_local),
        )
        for tag, ref, local in sources:
            if local is not None:
                try:
                    nbs[tag] = json.loads(Path(local).read_text(encoding="utf-8"))
                    nb_paths[tag] = Path(local)
                    ok(checks, tag, f"staged local {Path(local).name}")
                except Exception as exc:  # pragma: no cover - defensive
                    fail(checks, tag, f"failed to read local notebook {local}: {exc}")
                    ok_pulls = False
                continue
            d = tmp / tag
            d.mkdir()
            r = subprocess.run(["kaggle", "kernels", "pull", ref, "-p", str(d)],
                               capture_output=True, text=True, timeout=120)
            files = list(d.glob("*.ipynb"))
            if r.returncode != 0 or not files:
                fail(checks, tag, f"pull failed for {ref}")
                ok_pulls = False
                continue
            nbs[tag] = json.loads(files[0].read_text(encoding="utf-8"))
            nb_paths[tag] = files[0]
            ok(checks, tag, f"pulled {ref}")
        if ok_pulls:
            f_raw = ["".join(c.get("source", [])) for c in nbs["T1"]["cells"]
                     if c.get("cell_type") == "code"]
            u_raw = ["".join(c.get("source", [])) for c in nbs["T2"]["cells"]
                     if c.get("cell_type") == "code"]
            f_cells = [_tf_norm(s) for s in f_raw]
            u_cells = [_tf_norm(s) for s in u_raw]

            if max_diff_cells is None:
                # --- STRICT MODE (unchanged, byte-for-byte original semantics) ---
                if f_cells == u_cells:
                    ok(checks, "T3", f"code cells identical to upstream ({len(f_cells)} cells)")
                else:
                    n_diff = sum(1 for a, b in zip(f_cells, u_cells) if a != b) + abs(len(f_cells) - len(u_cells))
                    fail(checks, "T3", f"fork DIFFERS from upstream in {n_diff} code cells — "
                                       f"not a trusted fork; use structural mode or re-fork")
            else:
                # --- SINGLE-DIFF GRAFT MODE (additive extension) ---
                if pin_path is None:
                    fail(checks, "T3", "single-diff mode (--max-diff-cells) requires --pin "
                                       "<reference file of the audited byte-span>")
                elif not pin_path.is_file():
                    fail(checks, "T3", f"--pin file not found: {pin_path}")
                else:
                    pin_b, pin_actual_sha = _pin_bytes_and_sha(pin_path)
                    if pin_sha is not None and pin_sha.lower() != pin_actual_sha.lower():
                        fail(checks, "T3", f"--pin file sha256 {pin_actual_sha} != asserted "
                                           f"--pin-sha {pin_sha.lower()} (pin file swapped?)")
                    else:
                        pin_src = pin_b.decode("utf-8")
                        delta = _lcs_diff(f_raw, u_raw)
                        inserted = delta["inserted"]
                        deleted = delta["deleted"]
                        add_mods = delta["additive_mods"]
                        rewrites = delta["rewrites"]
                        n_ins = len(inserted)
                        # (a) delta must be additions only: no deletions, no rewrites.
                        #     Banner-only additive mods (upstream body byte-preserved as
                        #     a substring) are permitted — this is the cell-2 env-detect
                        #     banner append the prereg names as the sole differing frozen
                        #     code cell (run cell + solver surface byte-identical).
                        if deleted or rewrites:
                            fail(checks, "T3",
                                 f"delta vs upstream is NOT a clean additive graft: "
                                 f"{len(deleted)} deleted, {len(rewrites)} rewritten code cell(s) "
                                 f"(inserted={n_ins}, additive-banner-mods={len(add_mods)}); "
                                 f"single-diff mode certifies additions + banner-only edits only")
                        elif n_ins == 0:
                            fail(checks, "T3",
                                 "no inserted code cells found — fork adds no graft cell; "
                                 "use strict trusted-fork mode (drop --max-diff-cells) for a "
                                 "byte-identical fork")
                        # (b) inserted-cell count must be within budget N.
                        elif n_ins > max_diff_cells:
                            fail(checks, "T3",
                                 f"{n_ins} inserted code cell(s) > --max-diff-cells {max_diff_cells}")
                        else:
                            # (c) every inserted cell must carry the pinned byte-span
                            #     as a contiguous substring (additive banners around it OK).
                            unpinned = [fi for fi, src in inserted if pin_src not in src]
                            if unpinned:
                                fail(checks, "T3",
                                     f"{len(unpinned)} inserted cell(s) at fork code-idx {unpinned} "
                                     f"do NOT contain the pinned byte-span (sha256 "
                                     f"{pin_actual_sha[:12]}...); graft body not byte-matched")
                            else:
                                mod_note = (f", {len(add_mods)} banner-only additive edit(s)"
                                            if add_mods else "")
                                ok(checks, "T3",
                                   f"audited single-cell graft OK: {n_ins} inserted code cell(s) "
                                   f"(<= {max_diff_cells}), 0 deleted, 0 rewritten{mod_note}; each "
                                   f"inserted cell contains pinned byte-span sha256={pin_actual_sha}")

        # T4: build status. Unpushed/staged forks have no build — WARN, not FAIL.
        if fork_is_local:
            warn(checks, "T4", "fork staged from --local-notebook (unpushed) — "
                               "build-status leg SKIPPED; push + re-run for the COMPLETE gate")
        else:
            r = subprocess.run(["kaggle", "kernels", "status", kernel],
                               capture_output=True, text=True, timeout=60)
            m = re.search(r"KernelWorkerStatus\.(\w+)", r.stdout)
            status = m.group(1) if m else "UNKNOWN"
            if status == "COMPLETE":
                ok(checks, "T4", "latest build COMPLETE")
            else:
                fail(checks, "T4", f"latest build status = {status}")

        # H1-H4: additive host common-error gates (opt-in), run on the FORK.
        if host_gates_mode != "off" and "T1" in nbs:
            kmeta = load_kernel_metadata(nb_paths.get("T1"))
            checks.extend(host_gates(kernel, nbs["T1"], kmeta,
                                     strict=(host_gates_mode == "strict")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return summarize(kernel, version, checks)


def recurrence_field(kernel: str, mode: str, rep: dict) -> dict | None:
    """Failure-fingerprint recurrence lookup (Kimi-3 adopt #3, 2026-07-18).

    WARN-ONLY, NEVER BLOCKS. Reads runs/failure_fingerprints.json; if a
    candidate-matchable family of this kernel (same slug, or scratch-built
    provenance) died >=2 times before, returns the prior incident references.
    Returned as a NEW OPTIONAL top-level field — the existing checks /
    n_fail / n_warn / verdict contract that daily_submit.py parses is
    untouched. Any error here returns None (preflight must never break on
    the fingerprint store)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fingerprints import (candidate_families, load_store,
                                  recurrence_check)
        if mode == "trusted-fork":
            provenance = "trusted-fork"
        else:
            structural_fails = {c["check"] for c in rep.get("checks", [])
                                if c.get("status") == "FAIL"}
            provenance = ("scratch-built"
                          if structural_fails & {"K2", "K4", "K5"}
                          else "baseline-derived")
        fams = candidate_families(kernel, provenance)
        rec = recurrence_check(load_store(), fams)
        rec["provenance_assumed"] = provenance
        return rec
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="canivel/slug")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--mode", default="structural", choices=["structural", "trusted-fork"])
    ap.add_argument("--upstream", default=None,
                    help="upstream kernel ref (required for trusted-fork mode); "
                         "may be a local .ipynb path, which is read from disk instead of pulled")
    ap.add_argument("--max-diff-cells", type=int, default=None,
                    help="trusted-fork single-diff graft mode: allow up to N ADDED code cells "
                         "vs upstream (0 deleted, 0 modified); each added cell must contain the "
                         "--pin byte-span. Omit for strict byte-identical T3 (default).")
    ap.add_argument("--pin", default=None,
                    help="path to the audited byte-span reference file (e.g. the boris_16 gate "
                         "body); its exact bytes must appear as a contiguous substring in each "
                         "added cell. Required with --max-diff-cells.")
    ap.add_argument("--pin-sha", default=None,
                    help="optional expected sha256 of the --pin file (asserts the pinned reference "
                         "was not swapped; recorded as provenance in the T3 message)")
    ap.add_argument("--local-notebook", default=None,
                    help="stage the fork from this local .ipynb instead of pulling the kernel "
                         "(the arm-B slug is unpushed at prereg time). Skips the T4 build gate.")
    ap.add_argument("--host-gates", action="store_true",
                    help="run the additive host common-error gates H1-H4 (GPU-on, "
                         "no three.arcprize.org, no /kaggle/input writes, dataset attached) "
                         "in WARN mode. Additive/opt-in: off by default so existing "
                         "verdicts are unchanged. Never turns an ALLOW into a BLOCK.")
    ap.add_argument("--strict-host-gates", action="store_true",
                    help="run the H1-H4 host gates and ESCALATE real violations to FAIL "
                         "(DENY). Implies --host-gates. Missing metadata still WARNs, never "
                         "FAILs. Use once the gates are trusted; NOT for the frozen-fork lane.")
    ap.add_argument("--json-only", action="store_true",
                    help="print only the JSON report (no prose)")
    args = ap.parse_args()
    host_gates_mode = ("strict" if args.strict_host_gates
                       else "warn" if args.host_gates
                       else "off")
    if args.mode == "trusted-fork":
        if not args.upstream:
            print(json.dumps({"verdict": "BLOCK", "reason": "trusted-fork requires --upstream"}))
            sys.exit(1)
        rep = run_trusted_fork(
            args.kernel, args.upstream, args.version,
            max_diff_cells=args.max_diff_cells,
            pin_path=Path(args.pin) if args.pin else None,
            pin_sha=args.pin_sha,
            local_notebook=Path(args.local_notebook) if args.local_notebook else None,
            host_gates_mode=host_gates_mode,
        )
    else:
        rep = run_preflight(args.kernel, args.version,
                            host_gates_mode=host_gates_mode)
    rec = recurrence_field(args.kernel, args.mode, rep)
    if rec is not None:
        rep["recurrence"] = rec  # warn-only; never affects verdict/exit code
    if not args.json_only:
        print(f"Preflight for {args.kernel} v{args.version or 'latest'}: {rep['verdict']}")
        print(f"  fails: {rep['n_fail']}, warns: {rep['n_warn']}")
        for c in rep["checks"]:
            marker = {"OK": "OK", "WARN": "!!", "FAIL": "XX"}[c["status"]]
            print(f"  [{marker}] {c['check']}: {c['message']}")
        if rep.get("recurrence", {}).get("warn"):
            print("  [!!] RECURRENCE (warn-only): candidate matches failure "
                  "families with >=2 prior deaths:")
            for m in rep["recurrence"]["matches"]:
                print(f"       {m['family']}: {m['n_prior_deaths']} deaths "
                      f"({m['first_seen']} .. {m['last_seen']}) -- "
                      + "; ".join(m["refs"]))
    print(json.dumps(rep))
    sys.exit(0 if rep["verdict"] != "BLOCK" else 1)


if __name__ == "__main__":
    main()
