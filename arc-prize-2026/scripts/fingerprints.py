"""Two-tier failure-fingerprint system for the ARC-AGI-3 campaign.

Adopted 2026-07-18 (corrected adopt #3, Kimi-3 review cycle). Capture happens
at the OBSERVATION path (when we pull/see the failure), NOT at submission time.

Store: runs/failure_fingerprints.json (file-based; consumers read files).

Tier 1 (build rail, rich)
    For pulled kernel logs (runs/kernel_pulls/*/*.log and historical build
    logs) the fingerprint is sha256 of (stage-or-last-heartbeat marker,
    normalized_error), where normalization collapses UUIDs / timestamps /
    paths / hex ids / line numbers into placeholders. For silent/stuck logs
    (no error string at all) the last-progress marker IS the fingerprint
    material: last stdout stage line before silence + an elapsed-time bucket.

Tier 2 (scored rail, coarse)
    Competition submissions are HIDDEN executions — no logs exist, ever.
    The fingerprint is sha256 over (kernel_slug, version, status_class
    [COMPLETE/ERROR/PENDING-stuck], score_class [0.00 / null-band / scored],
    docker/machine metadata where retrievable). Family keys group incidents
    into recurrence families that a *candidate* kernel can be matched
    against BEFORE its window is burned:
      - slug:<owner/slug>            same kernel slug died before
      - provenance:<kind>            scratch-built / baseline-derived /
                                     trusted-fork  (the structural-drift
                                     family is provenance:scratch-built)
      - class:<status>:<score>      coarse, REPORT-ONLY (not candidate-
                                     matchable pre-submission; excluded from
                                     preflight WARN counterfactuals)

Consumers:
    scripts/preflight.py           recurrence WARN (warn-only, never block)
    scripts/fingerprint_report.py  count-by-fingerprint table for the brief
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = ROOT / "runs" / "failure_fingerprints.json"

SCHEMA_VERSION = 1

# Family keys of these prefixes can be matched against a candidate kernel
# before submission; class:* families cannot (a candidate has no status yet).
CANDIDATE_MATCHABLE_PREFIXES = ("slug:", "provenance:", "t1:", "t1root:")

# Only anomalous provenances form recurrence families. baseline-derived and
# trusted-fork are the default safe paths — grouping unrelated deaths under
# them would make the WARN fire on nearly every candidate (pure noise).
FAMILY_FORMING_PROVENANCES = {"scratch-built"}

# ---------------------------------------------------------------------------
# Normalization (tier 1)
# ---------------------------------------------------------------------------
# Order matters: UUIDs before generic hex, timestamps before generic numbers,
# paths before hex (paths may contain hex components). Every placeholder is
# chosen so that no normalization regex can re-match it -> idempotence.

_RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_RE_TS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_RE_PATH = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-@%+~]+){2,}[\\/]?")
_RE_HEX = re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b")
_RE_LINE = re.compile(r"\bline \d+\b")
_RE_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
_RE_NUM = re.compile(r"(?<![\w<>])\d+(?:\.\d+)?(?![\w<>])")
_RE_WS = re.compile(r"\s+")


def normalize_error(text: str) -> str:
    """Collapse volatile tokens (uuids/timestamps/paths/hex/line numbers/
    numbers) to placeholders. Idempotent: normalize(normalize(x)) == normalize(x)."""
    s = text.strip()
    s = _RE_PATH.sub("<path>", s)   # paths first: they may embed uuids/hex
    s = _RE_UUID.sub("<uuid>", s)
    s = _RE_TS.sub("<ts>", s)
    s = _RE_ADDR.sub("<hex>", s)
    s = _RE_HEX.sub("<hex>", s)
    s = _RE_LINE.sub("line <n>", s)
    s = _RE_NUM.sub("<n>", s)
    s = _RE_WS.sub(" ", s)
    return s.strip()


def _sha(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tier 1: build-rail fingerprints from kernel logs
# ---------------------------------------------------------------------------

# Lines that look like real runtime errors (not source-code echoes).
_ERR_LINE = re.compile(
    r"^\s*(?:[\w.]+\.)?\w*(?:Error|Exception|Interrupt)\b\s*(?::|$)")
_TRACEBACK = re.compile(r"^Traceback \(most recent call last\)")
# Source-code echoes / noise to exclude.
_CODE_ECHO = re.compile(
    r"^\s*(def |class |assert |raise |#|\"|'|import |from )"
    r"|-> None:|SyntaxWarning|DeprecationWarning|FutureWarning|UserWarning"
    r"|Debugger warning")

# Exception classes that only ever WRAP another failure. On the Kaggle rail
# every notebook death is re-raised twice — once by `subprocess.run(check=True)`
# inside the cell (CalledProcessError) and once by papermill around the cell
# (PapermillExecutionError) — so the LAST error line in the log is almost always
# one of these wrappers and carries no diagnostic content. The wrapper is still
# the primary tier-1 key (it is what the log terminates on, and it is stable),
# but the FIRST non-wrapper error is captured alongside it as the ROOT error so
# that two deaths behind the same wrapper do not silently look identical.
WRAPPER_ERROR_CLASSES = {
    "CalledProcessError", "PapermillExecutionError", "SystemExit",
    "SubprocessError", "ProcessExecutionError",
}


def error_class_of(line: str) -> str:
    """`papermill.exceptions.PapermillExecutionError: boom` -> `PapermillExecutionError`."""
    head = line.split(":", 1)[0].strip()
    return head.rsplit(".", 1)[-1]


ELAPSED_BUCKETS = [
    (600, "<10m"), (3600, "10-60m"), (4 * 3600, "1-4h"),
    (8 * 3600, "4-8h"), (float("inf"), ">8h"),
]


def elapsed_bucket(seconds: float) -> str:
    for limit, name in ELAPSED_BUCKETS:
        if seconds < limit:
            return name
    return ">8h"


def parse_kaggle_log(path: Path) -> list[dict]:
    """Parse a pulled Kaggle kernel log. Handles the JSON stream format
    ([{"stream_name","time","data"}, ...]) and plain text (fallback)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        recs = json.loads(raw)
        if isinstance(recs, list):
            return [r for r in recs if isinstance(r, dict) and "data" in r]
    except (json.JSONDecodeError, ValueError):
        pass
    # plain text fallback: everything is stdout, no times
    return [{"stream_name": "stdout", "time": None, "data": line + "\n"}
            for line in raw.splitlines()]


def _iter_lines(recs: list[dict]):
    for r in recs:
        stream = r.get("stream_name", "stdout")
        t = r.get("time")
        for line in str(r.get("data", "")).splitlines():
            if line.strip():
                yield stream, t, line


def extract_failure_signal(recs: list[dict]) -> dict:
    """Return {"error", "root_error", "stage", "elapsed"}.

    error      = the terminal exception line (last line of the last Traceback
                 block, or the last standalone FooError: line) — raw.
    root_error = the FIRST error line in the log whose class is not a pure
                 wrapper (see WRAPPER_ERROR_CLASSES), when that is a different
                 line from `error`; otherwise None. On the Kaggle rail this is
                 the line that actually says what broke.
    stage      = last meaningful stdout progress line before the error (or
                 before the end of the log for silent cases).
    """
    lines = list(_iter_lines(recs))
    error, error_idx = None, None
    errors: list[str] = []
    in_tb = False
    for i, (stream, _t, line) in enumerate(lines):
        if _TRACEBACK.match(line.strip()):
            in_tb = True
            continue
        if in_tb:
            if _ERR_LINE.match(line) and not _CODE_ECHO.search(line):
                in_tb = False
                error, error_idx = line.strip(), i
                errors.append(error)
            continue
        if stream == "stderr" and _ERR_LINE.match(line) \
                and not _CODE_ECHO.search(line):
            error, error_idx = line.strip(), i
            errors.append(error)
    root_error = None
    for e in errors:
        if error_class_of(e) not in WRAPPER_ERROR_CLASSES:
            root_error = e if e != error else None
            break
    # stage = last non-warning stdout line before the error (or end of log)
    stop = error_idx if error_idx is not None else len(lines)
    stage = None
    for stream, _t, line in lines[:stop]:
        if stream == "stdout" and not _CODE_ECHO.search(line):
            stage = line.strip()
    elapsed = None
    times = [t for _s, t, _l in lines if isinstance(t, (int, float))]
    if times:
        elapsed = max(times)
    return {"error": error, "root_error": root_error, "stage": stage,
            "elapsed": elapsed}


def tier1_root_fingerprint(root_error: str | None) -> dict | None:
    """Secondary key over the ROOT (non-wrapper) error alone — no stage, so it
    survives the pip-output noise that makes the stage marker unstable."""
    if not root_error:
        return None
    material = f"t1root|{normalize_error(root_error)}"
    return {"fingerprint": _sha(material), "material": material}


def tier1_fingerprint(stage: str | None, error: str | None,
                      elapsed: float | None = None) -> dict:
    """Fingerprint a build-rail failure observation.

    With an error string: sha256 over (normalized stage marker, normalized
    error). Silent/stuck (no error string): the last-progress marker IS the
    material — normalized stage + elapsed bucket.
    """
    norm_stage = normalize_error(stage) if stage else "<no-stage>"
    if error:
        norm_err = normalize_error(error)
        material = f"t1|{norm_stage}|{norm_err}"
        mode = "error"
    else:
        bucket = elapsed_bucket(elapsed) if elapsed is not None else "<unknown>"
        norm_err = None
        material = f"t1-silent|{norm_stage}|{bucket}"
        mode = "silent"
    return {
        "tier": 1,
        "mode": mode,
        "fingerprint": _sha(material),
        "material": material,
        "stage": norm_stage,
        "error": norm_err,
    }


def fingerprint_log_file(path: Path) -> dict:
    """Tier-1 fingerprint of a pulled kernel log file (observation path)."""
    recs = parse_kaggle_log(path)
    sig = extract_failure_signal(recs)
    fp = tier1_fingerprint(sig["stage"], sig["error"], sig["elapsed"])
    fp["source"] = str(path)
    fp["has_error"] = sig["error"] is not None
    fp["elapsed"] = sig["elapsed"]
    root = tier1_root_fingerprint(sig["root_error"])
    fp["root_error"] = normalize_error(sig["root_error"]) if sig["root_error"] else None
    fp["root_fingerprint"] = root["fingerprint"] if root else None
    fp["root_material"] = root["material"] if root else None
    return fp


# ---------------------------------------------------------------------------
# Retained-log inventory (shared by the backfill WRITER and the report READER)
# ---------------------------------------------------------------------------
# One place, so the reader can tell whether the writer has seen everything on
# disk. Anything added here is ingested automatically on the next backfill.

LOG_GLOBS = [
    "runs/kernel_pulls/*/*.log",        # `kaggle kernels output` pulls
    "runs/kernel_pulls/*/*.log.json",
    "runs/kernel_logs/*.log",           # `kaggle kernels logs` (CLI 2.2.3) pulls
    "runs/kernel_logs/*.log.json",
    # kernel pulls that landed outside runs/kernel_pulls/ (a22_*, tmp_pullback_*).
    # `arc3-*` is the kernel-slug naming convention, so this cannot pick up the
    # local (non-Kaggle) harness logs that also live under runs/.
    "runs/*/arc3-*.log",
    "notebooks/output/arc3-*.log",
    "notebooks/output_v12/arc3-*.log",
    "notebooks/output_v29/arc3-*.log",
]

# Side-car logs written BY a subprocess of the kernel. The kernel's own log
# already carries the death; ingesting these would double-count one run.
LOG_EXCLUDE_NAMES = {"vllm-openai-server.log"}
# Derived/flattened copies of a JSON-stream log from the SAME run.
LOG_EXCLUDE_SUFFIXES = ("_flat.log",)


def log_run_key(path: Path, root: Path | None = None) -> str:
    """Identity of the RUN a log belongs to (one incident per run).

    runs/kernel_pulls/<run>/anything.log -> the <run> directory
    runs/kernel_logs/<run>.log[.json]    -> the file stem (one file per run)
    """
    root = root or ROOT
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    if len(parts) >= 2 and parts[-2] == "kernel_logs":
        return rel[: -len(".log.json")] if rel.endswith(".log.json") \
            else rel[: -len(".log")]
    return "/".join(parts[:-1])


def _is_json_stream(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(1) == b"["
    except OSError:
        return False


def iter_log_sources(root: Path | None = None) -> list[dict]:
    """Every retained build-rail log the fingerprint system can see.

    Returns one entry per RUN: {"path", "rel", "run_key", "mtime"} where mtime
    is a tz-aware UTC datetime. Deduplicated with a stable preference
    (JSON-stream form first, then larger file) so a run that was saved twice
    in two formats produces exactly one incident.
    """
    import datetime as _dt
    root = root or ROOT
    best: dict[str, dict] = {}
    for pattern in LOG_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.name in LOG_EXCLUDE_NAMES:
                continue
            if path.name.endswith(LOG_EXCLUDE_SUFFIXES):
                continue
            if not path.is_file():
                continue
            key = log_run_key(path, root)
            st = path.stat()
            entry = {
                "path": path,
                "rel": path.relative_to(root).as_posix(),
                "run_key": key,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime,
                                                    tz=_dt.timezone.utc),
                "_rank": (1 if _is_json_stream(path) else 0, st.st_size),
            }
            prev = best.get(key)
            if prev is None or entry["_rank"] > prev["_rank"]:
                best[key] = entry
    out = sorted(best.values(), key=lambda e: e["rel"])
    for e in out:
        e.pop("_rank", None)
    return out


# ---------------------------------------------------------------------------
# Staleness (the 2026-08-16 defect: the READER never noticed the WRITER had
# not run since 2026-07-18, so a store missing 20 days of deaths read as
# "no new incidents — consistent with the post-preflight regime")
# ---------------------------------------------------------------------------

def _parse_iso(s: str | None):
    import datetime as _dt
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def staleness_report(store: dict, sources: list[dict] | None = None,
                     root: Path | None = None) -> dict:
    """Is the store behind the logs on disk?

    STALE if any of:
      - the store is empty / was never written;
      - a retained log exists that the last backfill never scanned;
      - a retained log was modified AFTER the store was last written.
    A clean COMPLETE log newer than the newest incident is NOT stale on its own
    (a successful run correctly produces no incident) — it is stale only if the
    writer has not looked at it.
    """
    if sources is None:
        sources = iter_log_sources(root)
    scanned = {s.get("log") for s in store.get("scan_log", []) or []}
    incidents = store.get("incidents", []) or []
    updated = _parse_iso(store.get("updated"))
    unscanned = [s for s in sources if s["rel"] not in scanned]
    newer = [s for s in sources if updated and s["mtime"] > updated]
    newest_src = max(sources, key=lambda s: s["mtime"]) if sources else None
    newest_inc = max((i.get("date") or "" for i in incidents), default="")
    reasons = []
    if not incidents:
        reasons.append("store holds ZERO incidents")
    if updated is None:
        reasons.append("store has no `updated` timestamp (never written?)")
    if unscanned:
        reasons.append(
            f"{len(unscanned)} retained log(s) NEVER SCANNED, newest: "
            + max(unscanned, key=lambda s: s["mtime"])["rel"])
    if newer:
        reasons.append(
            f"{len(newer)} retained log(s) modified AFTER the store was written "
            f"({store.get('updated')})")
    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "n_sources": len(sources),
        "n_unscanned": len(unscanned),
        "unscanned": [s["rel"] for s in unscanned],
        "store_updated": store.get("updated"),
        "newest_incident_date": newest_inc or None,
        "newest_log": newest_src["rel"] if newest_src else None,
        "newest_log_date": newest_src["mtime"].strftime("%Y-%m-%d")
        if newest_src else None,
    }


STALE_FIX_CMD = "uv run python scripts/fingerprint_backfill.py"


def format_staleness_banner(st: dict) -> str:
    """Loud, unmissable, and it names the command that fixes it."""
    if not st["stale"]:
        return (f"store FRESH: {st['n_sources']} retained logs all scanned "
                f"(store written {st['store_updated']}, newest incident "
                f"{st['newest_incident_date']}, newest log "
                f"{st['newest_log_date']})")
    bar = "!" * 78
    lines = [bar, "!! STALE FAILURE-FINGERPRINT STORE -- THE TABLE BELOW IS NOT CURRENT"]
    for r in st["reasons"]:
        lines.append(f"!!   - {r}")
    lines.append(f"!!   newest incident in store: {st['newest_incident_date'] or 'NONE'}"
                 f"   newest retained log: {st['newest_log_date'] or 'NONE'}"
                 f" ({st['newest_log'] or '-'})")
    for rel in st["unscanned"][:8]:
        lines.append(f"!!   unscanned: {rel}")
    if len(st["unscanned"]) > 8:
        lines.append(f"!!   ... and {len(st['unscanned']) - 8} more")
    lines.append(f"!! FIX FIRST, THEN RE-READ:  {STALE_FIX_CMD}")
    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier 2: scored-rail fingerprints (hidden executions — no logs, ever)
# ---------------------------------------------------------------------------

STATUS_CLASSES = ("COMPLETE", "ERROR", "PENDING-stuck")
SCORE_CLASSES = ("0.00", "null-band", "scored", "none")


def score_class_of(score: float | None) -> str:
    if score is None:
        return "none"
    if score == 0.0:
        return "0.00"
    if score <= 0.05:
        return "null-band"
    return "scored"


def tier2_fingerprint(kernel: str, version: int | str | None,
                      status_class: str, score_class: str,
                      docker: str | None = None,
                      machine: str | None = None,
                      gpu: bool | None = None) -> dict:
    material = json.dumps({
        "kernel": kernel, "version": version, "status": status_class,
        "score": score_class,
        "docker": (docker or "")[:64], "machine": machine, "gpu": gpu,
    }, sort_keys=True)
    return {
        "tier": 2,
        "fingerprint": _sha("t2|" + material),
        "material": material,
    }


def tier2_families(kernel: str, status_class: str, score_class: str,
                   provenance: str | None) -> list[str]:
    fams = [f"slug:{kernel}"]
    if provenance in FAMILY_FORMING_PROVENANCES:
        fams.append(f"provenance:{provenance}")
    fams.append(f"class:{status_class}:{score_class}")
    return fams


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def empty_store() -> dict:
    return {"schema_version": SCHEMA_VERSION, "updated": None, "incidents": []}


def load_store(path: Path | None = None) -> dict:
    p = Path(path) if path else STORE_PATH
    if not p.exists():
        return empty_store()
    try:
        store = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_store()
    store.setdefault("incidents", [])
    return store


def save_store(store: dict, path: Path | None = None) -> None:
    p = Path(path) if path else STORE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Recurrence (consumer logic — shared by preflight + report)
# ---------------------------------------------------------------------------

def candidate_families(kernel: str, provenance: str | None = None) -> list[str]:
    """Family keys a candidate kernel belongs to BEFORE submission."""
    fams = [f"slug:{kernel}"]
    if provenance in FAMILY_FORMING_PROVENANCES:
        fams.append(f"provenance:{provenance}")
    return fams


def family_index(store: dict, before_date: str | None = None,
                 include_low_confidence: bool = True) -> dict[str, list[dict]]:
    """family key -> chronologically ordered incident list."""
    idx: dict[str, list[dict]] = {}
    incidents = sorted(store.get("incidents", []),
                       key=lambda i: (i.get("date") or "", i.get("id") or ""))
    for inc in incidents:
        if before_date and (inc.get("date") or "") >= before_date:
            continue
        if not include_low_confidence and inc.get("confidence") == "low":
            continue
        for fam in inc.get("families", []):
            idx.setdefault(fam, []).append(inc)
    return idx


def _ref(inc: dict) -> str:
    k = inc.get("kernel") or "?"
    v = inc.get("version")
    return f"{inc.get('id')}:{k}" + (f" v{v}" if v is not None else "") \
        + f" ({inc.get('date')})"


def recurrence_check(store: dict, families: list[str],
                     min_deaths: int = 2,
                     before_date: str | None = None) -> dict:
    """WARN-only recurrence lookup: for each candidate family, if the family
    died >= min_deaths times before, return the prior incident references.
    Never blocks — panel-gated escalation comes later."""
    idx = family_index(store, before_date=before_date)
    matches = []
    for fam in families:
        incs = idx.get(fam, [])
        if len(incs) >= min_deaths:
            matches.append({
                "family": fam,
                "n_prior_deaths": len(incs),
                "first_seen": incs[0].get("date"),
                "last_seen": incs[-1].get("date"),
                "incidents": [i.get("id") for i in incs],
                "refs": [_ref(i) for i in incs],
            })
    return {"warn": bool(matches), "min_deaths": min_deaths,
            "families_checked": families, "matches": matches}
