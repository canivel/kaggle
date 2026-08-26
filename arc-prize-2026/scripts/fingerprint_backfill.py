"""Backfill the two-tier failure-fingerprint store from campaign history.

Sources:
  - ITERATION_LOG.md incident narrative (v45 JEPA ERROR, v30/v36/v38 zero-band,
    the June ERROR streak) + docs/competition_plan.md + memory feedback records
    (curated below with explicit source refs and confidence levels)
  - runs/submission_log.jsonl (scored-rail submission records)
  - every retained build-rail log fingerprints.iter_log_sources() can see:
    runs/kernel_pulls/*/*.log[.json] (`kernels output` pulls),
    runs/kernel_logs/*.log[.json] (`kernels logs` pulls, CLI 2.2.3) and
    notebooks/output*/ historical build logs
    (tier-1 scan — fingerprints only actual failures; clean COMPLETE logs
    produce no incident)

Outputs:
  - runs/failure_fingerprints.json      (live store)
  - runs/failure_fingerprints_backfill.md  (validation: family collapse,
    WARN-firing occurrence per family, counterfactual burned-window count)

This is the WRITER of runs/failure_fingerprints.json. scripts/fingerprint_report.py
is the READER and never writes. Run the writer BEFORE the reader — a store that
is never re-written silently reads as "no new incidents" (that is exactly what
happened between 2026-07-18 and 2026-08-16).

Usage:
  uv run python scripts/fingerprint_backfill.py            # rebuild the store
  uv run python scripts/fingerprint_backfill.py --dry-run  # show what WOULD change
  uv run python scripts/fingerprint_backfill.py --help
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import (  # noqa: E402
    CANDIDATE_MATCHABLE_PREFIXES, ROOT, STORE_PATH, empty_store,
    fingerprint_log_file, iter_log_sources, load_store, save_store,
    tier1_fingerprint, tier2_families, tier2_fingerprint,
)

REPORT_PATH = ROOT / "runs" / "failure_fingerprints_backfill.md"

# ---------------------------------------------------------------------------
# Kernel metadata lookup (docker/machine where retrievable)
# ---------------------------------------------------------------------------

_META_DIRS = {
    "canivel/arc3-forge35": "notebooks/forge35",
    "canivel/arc3-forge62": "notebooks/forge62",
    "canivel/arc3-jepa-v2": "notebooks/jepav2",
    "canivel/arc3-execwm": "notebooks/execwm",
    "canivel/arc3-execwm-v2": "notebooks/execwm2",
    "canivel/arc3-execwm-v3": "notebooks/execwm3",
    "canivel/arc3-baseline": "notebooks/baseline",
    "canivel/arc3-final": "notebooks/arc3-final",
    "canivel/arc3-duck-repro": "notebooks/duckfork",
    "canivel/arc3-duck-war": "notebooks/duckwar",
    "canivel/arc3-pilot-eval": "notebooks/piloteval",
}


def kernel_meta(kernel: str) -> dict:
    d = _META_DIRS.get(kernel)
    if not d:
        return {}
    p = ROOT / d / "kernel-metadata.json"
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        "docker": m.get("docker_image"),
        "machine": m.get("machine_shape"),
        "gpu": m.get("enable_gpu"),
    }


# ---------------------------------------------------------------------------
# Curated tier-2 incidents (scored rail — HIDDEN executions, no logs ever)
# ---------------------------------------------------------------------------
# provenance: scratch-built = notebook programmatically constructed (the
#             structural-drift family, per feedback_arc_kernel_structural_drift)
#             baseline-derived = built from arc3-baseline.ipynb / normal edit
#             trusted-fork = unmodified fork of a proven public kernel
# confidence: low = date/slug reconstructed from memory records, not from a
#             timestamped artifact; excluded from the WARN counterfactual.

TIER2_INCIDENTS = [
    dict(id="inc-t2-001", date="2026-03-29", kernel="canivel/arc3-cnn-frame-change-agent",
         version=6, status="COMPLETE", score=0.0, provenance="original",
         source="docs/competition_plan.md:8",
         note="Pilot-era v6 scored 0.00 - GPU OOM + missing FrameData fields."),
    dict(id="inc-t2-002", date="2026-04-24", kernel="canivel/arc3-forge35",
         version=None, status="COMPLETE", score=0.0, provenance="baseline-derived",
         source="memory/feedback_lock_deadlock.md", confidence="low",
         note="forge_v35 TIPS _TIPS_LOCK deadlock - all 110 game threads froze "
              "at startup -> 0.00. Date/slug reconstructed (last forge35 run "
              "2026-04-24); excluded from WARN counterfactual."),
    dict(id="inc-t2-003", date="2026-05-26", kernel="canivel/arc3-final",
         version=26, status="ERROR", score=None, provenance="scratch-built",
         source="ITERATION_LOG.md sec.3 (v45)",
         note="v45 = v39 + 25M JEPA. ERROR on rerun. Structural-drift death #1 "
              "(root cause found 2026-06-28: build_notebook.py drifted "
              "agents/__init__.py + .env from baseline)."),
    dict(id="inc-t2-004", date="2026-06-01", kernel="canivel/arc3-final",
         version=30, status="COMPLETE", score=0.04, provenance="baseline-derived",
         source="ITERATION_LOG.md sec.6 + sec.2b",
         note="v30 = 0.04 - poisoned pretrained ForgeNet weights (pre-set_data "
              "click coords). Zero-band death."),
    dict(id="inc-t2-005", date="2026-06-08", kernel="canivel/arc3-final",
         version=36, status="COMPLETE", score=0.01, provenance="baseline-derived",
         source="ITERATION_LOG.md sec.5c",
         note="v36 = 0.01 - SG + unbalanced pretrain (78.8% positive labels)."),
    dict(id="inc-t2-006", date="2026-06-10", kernel="canivel/arc3-final",
         version=38, status="COMPLETE", score=0.0, provenance="baseline-derived",
         source="memory/feedback_test_before_submit.md + ITERATION_LOG.md sec.5e",
         note="v38 = 0.00 - `defaultdict` used but never imported; ast.parse "
              "passed, 30s runtime test would have caught it."),
    dict(id="inc-t2-007", date="2026-06-20", kernel="canivel/arc3-forge62",
         version=4, status="ERROR", score=None, provenance="scratch-built",
         source="runs/submission_log.jsonl 2026-06-20 + feedback_arc_jepa_dead",
         note="v62 = v35 + JEPA-XXS MCTS. ERROR on rerun, fresh slug. "
              "Structural-drift death #2."),
    dict(id="inc-t2-008", date="2026-06-21", kernel="canivel/arc3-forge35",
         version=1, status="ERROR", score=None, provenance="baseline-derived",
         prior_success=True,
         source="runs/submission_log.jsonl 2026-06-21/22 messages",
         note="v35 sample 12 ERROR - same kernel+version that scored "
              "0.10-0.43 ten times before (error-after-success)."),
    dict(id="inc-t2-009", date="2026-06-22", kernel="canivel/arc3-forge35",
         version=1, status="ERROR", score=None, provenance="baseline-derived",
         prior_success=True,
         source="runs/submission_log.jsonl 2026-06-22 message",
         note="v35 sample 13 ERROR - deliberate rerun-infra diagnostic after "
              "two consecutive ERRORs; third consecutive ERROR overall."),
    dict(id="inc-t2-010", date="2026-06-26", kernel="canivel/arc3-jepa-v2",
         version=1, status="ERROR", score=None, provenance="scratch-built",
         source="runs/submission_log.jsonl 2026-06-26 + feedback_arc_jepa_dead",
         note="v63 = throttled JEPA on fresh slug. ERROR. Structural-drift "
              "death #3 (family WARN would already have been active)."),
    dict(id="inc-t2-011", date="2026-06-27", kernel="canivel/arc3-execwm",
         version=1, status="ERROR", score=None, provenance="scratch-built",
         source="runs/submission_log.jsonl 2026-06-27",
         note="v64 = v35 + ExecWMHook. ERROR. Structural-drift death #4."),
    dict(id="inc-t2-012", date="2026-06-28", kernel="canivel/arc3-execwm-v2",
         version=1, status="ERROR", score=None, provenance="scratch-built",
         source="runs/submission_log.jsonl 2026-06-28",
         note="v65 = v64 inlined, fresh slug. ERROR. Structural-drift death #5 "
              "- root cause found the same day; v66 rebuilt on baseline "
              "structure + preflight.py born."),
]

# ---------------------------------------------------------------------------
# Curated tier-1 incidents (build rail) whose logs were NOT retained locally.
# Fingerprint material reconstructed from the recorded error class; marked
# reconstructed=True so the report can separate them from log-scanned ones.
# ---------------------------------------------------------------------------

TIER1_CURATED = [
    dict(id="inc-t1-001", date="2026-07-07", kernel="canivel/arc3-pilot-eval",
         version=1, stage="pilot eval: scoring pass", error="IndexError: list index out of range",
         source="campaign record (Kimi-3 review cycle 2026-07-18); duck_eval/pilot/scoring.py crash-class",
         confidence="low",
         note="Pilot eval kernel v1 build died with IndexError. Log not "
              "retained; material reconstructed from record."),
    dict(id="inc-t1-002", date="2026-07-07", kernel="canivel/arc3-pilot-eval",
         version=2, stage="pilot eval: scoring pass", error="IndexError: list index out of range",
         source="campaign record (Kimi-3 review cycle 2026-07-18)",
         confidence="low",
         note="Pilot eval kernel v2 - same IndexError family, death #2 "
              "(recurrence WARN condition met here)."),
    dict(id="inc-t1-003", date="2026-07-08", kernel="canivel/arc3-pilot-eval",
         version=3, stage="pilot eval: scoring pass", error="IndexError: list index out of range",
         source="campaign record (Kimi-3 review cycle 2026-07-18)",
         confidence="low",
         note="Pilot eval kernel v3 - death #3, would have carried the WARN."),
    dict(id="inc-t1-004", date="2026-07-07", kernel="canivel/arc3-duck-repro",
         version=1, stage="duck harness: GPU check",
         error="AssertionError: CUDA GPU check failed: expected RTX PRO 6000, got Tesla P100",
         source="runs/submission_log.jsonl 2026-07-08 message ('v1 failed GPU-type assert on P100')",
         confidence="medium",
         note="duck-repro v1 build died on GPU-type assert (machine_shape "
              "missing from kernel-metadata). Fixed in v3."),
]

# Retained build-rail logs to scan (observation path, real artifacts) are
# enumerated by fingerprints.iter_log_sources() — the SAME inventory the report
# checks itself against for staleness. Add new log locations there, not here.

# ---------------------------------------------------------------------------
# Per-log provenance hints for the scanned rail.
#
# A pulled log knows neither its kernel slug nor its RUN date: the filename is
# arbitrary and the mtime is the PULL date, which can be days after the death
# (lora_serve_canary v1 ran 08-14 and was pulled 08-16). Slug/date are supplied
# here from a timestamped artifact, with the artifact named. Anything not
# listed falls back to: slug inferred from an `arc3-*.log` filename, date =
# file mtime.
# ---------------------------------------------------------------------------

LOG_HINTS = {
    "runs/kernel_pulls/q38_v1/q38.log": dict(
        kernel="canivel/arc3-q38-engine-eval", version=1, date="2026-08-15",
        source_note="ITERATION_LOG.md 2026-08-15 (Q38 engine-swap eval, slot 2); "
                    "learnings/war_room/q38_engine_swap_prereg_2026-08-15.md"),
    "runs/kernel_logs/lora_serve_canary_v1.log.json": dict(
        kernel="canivel/arc3-lora-serve-canary", version=1,
        date="2026-08-14", time="13:40:18",
        source_note="learnings/war_room/lora_serve_canary_postmortem_2026-08-16.md "
                    "(pushed+run 2026-08-14 13:40:18Z; log pulled 08-16, so mtime "
                    "would misdate it)"),
    "runs/kernel_pulls/b122_v1/arc3-b122-boot-canary.log": dict(
        kernel="canivel/arc3-b122-boot-canary", version=1, date="2026-08-14",
        source_note="ITERATION_LOG.md 2026-08-14 (b122 FlashInfer SM120 MoE JIT "
                    "infra death #2)"),
    "runs/kernel_pulls/a17_canary_v1/arc3-a17-72b-canary.log": dict(
        kernel="canivel/arc3-a17-72b-canary", version=1, date="2026-07-25",
        source_note="runs/kernel_pulls/a17_canary_v1 (pull mtime 2026-07-25)"),
    "runs/kernel_pulls/a17_canary_v2/arc3-a17-72b-canary.log": dict(
        kernel="canivel/arc3-a17-72b-canary", version=2, date="2026-07-25",
        source_note="runs/kernel_pulls/a17_canary_v2 (pull mtime 2026-07-25)"),
}


def infer_kernel(rel: str) -> str | None:
    """`.../arc3-duck-war-eval.log` -> `canivel/arc3-duck-war-eval`."""
    name = rel.rsplit("/", 1)[-1]
    for suf in (".log.json", ".log"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return f"canivel/{name}" if name.startswith("arc3-") else None


# ---------------------------------------------------------------------------
# Build the store
# ---------------------------------------------------------------------------

def build_incidents() -> tuple[list[dict], list[dict]]:
    incidents: list[dict] = []
    for spec in TIER2_INCIDENTS:
        meta = kernel_meta(spec["kernel"])
        from fingerprints import score_class_of
        sc = score_class_of(spec["score"])
        fp = tier2_fingerprint(spec["kernel"], spec["version"], spec["status"],
                               sc, meta.get("docker"), meta.get("machine"),
                               meta.get("gpu"))
        incidents.append({
            "id": spec["id"], "tier": 2, "date": spec["date"],
            "kernel": spec["kernel"], "version": spec["version"],
            "status_class": spec["status"], "score_class": sc,
            "score": spec["score"],
            "provenance": spec["provenance"],
            "prior_success": spec.get("prior_success", False),
            "fingerprint": fp["fingerprint"], "material": fp["material"],
            "families": tier2_families(spec["kernel"], spec["status"], sc,
                                       spec["provenance"]),
            "confidence": spec.get("confidence", "high"),
            "source": spec["source"], "note": spec["note"],
        })
    for spec in TIER1_CURATED:
        fp = tier1_fingerprint(spec["stage"], spec["error"])
        incidents.append({
            "id": spec["id"], "tier": 1, "date": spec["date"],
            "kernel": spec["kernel"], "version": spec["version"],
            "status_class": "ERROR", "score_class": "none",
            "mode": fp["mode"], "reconstructed": True,
            "fingerprint": fp["fingerprint"], "material": fp["material"],
            "families": [f"t1:{fp['fingerprint']}", f"slug:{spec['kernel']}"],
            "confidence": spec.get("confidence", "high"),
            "source": spec["source"], "note": spec["note"],
        })

    # Scan retained logs — only actual failures become incidents.
    scanned: list[dict] = []
    found: list[dict] = []
    for src in iter_log_sources(ROOT):
        path, rel = src["path"], src["rel"]
        fp = fingerprint_log_file(path)
        hint = LOG_HINTS.get(rel, {})
        scanned.append({"log": rel, "run_key": src["run_key"],
                        "has_error": fp["has_error"],
                        "fingerprint": fp["fingerprint"],
                        "stage": fp["stage"]})
        if not fp["has_error"]:
            continue
        date = hint.get("date") or src["mtime"].strftime("%Y-%m-%d")
        # sort key: hinted wall-clock where known, else the pull mtime — keeps
        # incident ids stable and chronological across reruns.
        sort_ts = f"{date}T{hint.get('time') or src['mtime'].strftime('%H:%M:%S')}"
        kernel = hint.get("kernel") or infer_kernel(rel)
        families = [f"t1:{fp['fingerprint']}"]
        if fp.get("root_fingerprint"):
            families.append(f"t1root:{fp['root_fingerprint']}")
        if kernel:
            families.append(f"slug:{kernel}")
        found.append({
            "_sort": (sort_ts, rel),
            "tier": 1, "date": date,
            "kernel": kernel, "version": hint.get("version"),
            "status_class": "ERROR", "score_class": "none",
            "mode": fp["mode"], "reconstructed": False,
            "fingerprint": fp["fingerprint"],
            "material": fp["material"],
            "root_fingerprint": fp.get("root_fingerprint"),
            "root_error": fp.get("root_error"),
            "elapsed": fp.get("elapsed"),
            "families": families,
            "confidence": "high", "source": rel,
            "date_source": "hint" if hint.get("date") else "log mtime",
            "note": "scanned from retained build log"
                    + (f" -- {hint['source_note']}" if hint.get("source_note") else ""),
        })
    n = len([i for i in incidents if i["tier"] == 1])
    for inc in sorted(found, key=lambda i: i["_sort"]):
        n += 1
        inc.pop("_sort")
        incidents.append({"id": f"inc-t1-{n:03d}", **inc})
    return incidents, scanned


# ---------------------------------------------------------------------------
# Validation: chronological replay of the recurrence WARN
# ---------------------------------------------------------------------------

def replay(incidents: list[dict], min_deaths: int = 2) -> dict:
    """Replay history in date order. For each incident, determine whether a
    recurrence WARN (>= min_deaths prior deaths in a candidate-matchable
    family the incident's kernel belonged to) was active BEFORE it fired.
    Low-confidence incidents never count as evidence (prior deaths) and are
    reported separately when flagged."""
    ordered = sorted(incidents, key=lambda i: (i["date"], i["id"]))
    seen: dict[str, list[dict]] = {}
    rows = []
    for inc in ordered:
        cand_fams = [f for f in inc["families"]
                     if f.startswith(CANDIDATE_MATCHABLE_PREFIXES)]
        active = []
        for fam in cand_fams:
            prior = [p for p in seen.get(fam, [])
                     if p.get("confidence") != "low"]
            if len(prior) >= min_deaths:
                active.append({"family": fam,
                               "n_prior": len(prior),
                               "refs": [p["id"] for p in prior]})
        rows.append({"incident": inc, "warn_active": bool(active),
                     "matches": active})
        for fam in inc["families"]:
            seen.setdefault(fam, []).append(inc)
    return {"rows": rows, "family_deaths": {k: [i["id"] for i in v]
                                            for k, v in seen.items()}}


def family_table(incidents: list[dict]) -> list[dict]:
    fams: dict[str, list[dict]] = {}
    for inc in sorted(incidents, key=lambda i: (i["date"], i["id"])):
        for fam in inc["families"]:
            fams.setdefault(fam, []).append(inc)
    out = []
    for fam, incs in fams.items():
        out.append({
            "family": fam, "n": len(incs),
            "first": incs[0]["date"], "last": incs[-1]["date"],
            "candidate_matchable": fam.startswith(CANDIDATE_MATCHABLE_PREFIXES),
            "warn_fires_at_death": min_fire(incs),
            "incidents": [i["id"] for i in incs],
        })
    out.sort(key=lambda r: (-r["n"], r["family"]))
    return out


def min_fire(incs: list[dict], min_deaths: int = 2) -> int | None:
    """1-based death index after which the WARN condition becomes active
    (counting only non-low-confidence deaths as evidence)."""
    n = 0
    for i, inc in enumerate(incs, start=1):
        if inc.get("confidence") != "low":
            n += 1
        if n >= min_deaths:
            return i
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(incidents, scanned, rep, fams) -> str:
    rows = rep["rows"]
    flagged = [r for r in rows if r["warn_active"]]
    flagged_scored = [r for r in flagged if r["incident"]["tier"] == 2]
    recurring = [f for f in fams if f["n"] >= 2]
    lines = []
    a = lines.append
    a("# Failure-fingerprint backfill — campaign history validation")
    a("")
    a(f"Generated {dt.date.today().isoformat()} by scripts/fingerprint_backfill.py. "
      f"Store: `runs/failure_fingerprints.json` ({len(incidents)} incidents).")
    a("")
    a("Two-tier design (Kimi-3 review cycle, corrected adopt #3): tier 1 = build-rail "
      "logs (rich: stage marker + normalized error; silent deaths keyed by last-progress "
      "marker + elapsed bucket); tier 2 = scored-rail submissions (coarse: slug, version, "
      "status class, score class, docker/machine). **Scored reruns are hidden executions — "
      "no logs exist, ever**; tier 2 never pretends otherwise.")
    a("")
    a("## 1. Incident inventory")
    a("")
    a("| id | date | tier | kernel | ver | status | score | fingerprint | confidence | source |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for inc in sorted(incidents, key=lambda i: (i["date"], i["id"])):
        a(f"| {inc['id']} | {inc['date']} | {inc['tier']} | {inc.get('kernel') or '-'} "
          f"| {inc.get('version') if inc.get('version') is not None else '-'} "
          f"| {inc['status_class']} | {inc['score_class']} | `{inc['fingerprint']}` "
          f"| {inc['confidence']} | {inc['source'].split(' ')[0]} |")
    a("")
    a(f"Tier-1 log scan: {len(scanned)} retained build logs scanned, "
      f"{sum(1 for s in scanned if s['has_error'])} contained failure signals "
      "(all retained pulls are COMPLETE eval builds — the scored-rail deaths above "
      "have no logs by construction, which is exactly why tier 2 exists).")
    a("")
    a("## 2. Q1 — family collapse")
    a("")
    n_fp = len({i['fingerprint'] for i in incidents})
    a(f"{len(incidents)} incidents collapse into **{n_fp} distinct fingerprints** and "
      f"**{len(recurring)} recurring families (n>=2)** "
      f"({sum(1 for f in recurring if f['candidate_matchable'])} candidate-matchable, "
      f"{sum(1 for f in recurring if not f['candidate_matchable'])} report-only class families). "
      "Every family key of every incident:")
    a("")
    a("| family | n | first | last | WARN active after death # | matchable pre-submission | incidents |")
    a("|---|---|---|---|---|---|---|")
    for f in fams:
        fire = f["warn_fires_at_death"] if f["n"] >= 2 and f["warn_fires_at_death"] else "-"
        a(f"| `{f['family']}` | {f['n']} | {f['first']} | {f['last']} | {fire} "
          f"| {'yes' if f['candidate_matchable'] else 'no (report-only)'} "
          f"| {', '.join(f['incidents'])} |")
    a("")
    a("## 3. Q2 — where the recurrence WARN would have fired")
    a("")
    a("Replay rule: before each submission/build, WARN if any candidate-matchable family "
      "(slug:, provenance:, t1:) of the candidate had **>=2 prior deaths** (low-confidence "
      "reconstructed incidents never count as evidence). Chronological result:")
    a("")
    a("| incident | date | window | WARN active before it? | matching family (prior deaths) |")
    a("|---|---|---|---|---|")
    for r in rows:
        inc = r["incident"]
        win = "scored LB window" if inc["tier"] == 2 else "build slot"
        m = "; ".join(f"`{x['family']}` ({x['n_prior']}: {', '.join(x['refs'])})"
                      for x in r["matches"]) or "-"
        a(f"| {inc['id']} ({inc.get('kernel') or '?'} "
          f"v{inc.get('version') if inc.get('version') is not None else '?'}) "
          f"| {inc['date']} | {win} | {'**YES**' if r['warn_active'] else 'no'} | {m} |")
    a("")
    a("Ground-truth checks:")
    a("")
    drift = [r for r in rows if "provenance:scratch-built" in
             [m["family"] for m in r["matches"]]]
    a(f"- **Structural-drift family** (`provenance:scratch-built`): 5 deaths "
      "(v45 05-26, v62 06-20, v63 06-26, v64 06-27, v65 06-28). WARN condition met at "
      f"death #2 (v62) -> **{len(drift)} subsequent deaths (v63, v64, v65) would have "
      "carried the WARN in advance.** The root cause was only found manually on 06-28, "
      "after death #5.")
    zb = [r for r in rows if r["warn_active"] and
          r["incident"]["kernel"] == "canivel/arc3-final"]
    a(f"- **arc3-final slug family**: deaths v45 (ERROR), v30 (0.04), v36 (0.01), v38 (0.00). "
      f"WARN condition met at death #2 (v30, 06-01) -> **{len(zb)} subsequent deaths "
      "(v36 06-08, v38 06-10 missing-import 0.00) would have carried the WARN.**")
    a("- **arc3-forge35 slug ERRORs** (s12 06-21, s13 06-22): with the TIPS-deadlock "
      "attribution held at low confidence, only 1 high-confidence prior death existed "
      "before s13 -> **no WARN**; both windows burned unflagged. (Sensitivity: if the "
      "TIPS 0.00 is accepted as a forge35 death, s13 fires the WARN -> +1 flagged.) "
      "The 06-24 fresh-slug pivot is what the WARN would have recommended at death #2.")
    pilot = [r for r in rows if r["warn_active"] and
             r["incident"]["kernel"] == "canivel/arc3-pilot-eval"]
    a(f"- **Pilot-eval IndexError family** (t1, v1-v3): identical normalized fingerprint "
      "all three times. Evidence held at low confidence (logs not retained), so the "
      f"strict replay flags {len(pilot)} of them; with the incidents taken at face "
      "value the WARN fires at death #2 and v3 (death #3) is flagged-in-advance.")
    a("")
    a("## 4. Q3 — counterfactual: windows burned that would have carried a WARN")
    a("")
    a(f"- **{len(flagged_scored)} scored LB windows** were burned by deaths that a "
      "recurrence WARN would have flagged before submission: "
      + ", ".join(f"{r['incident']['id']} ({r['incident']['kernel']} "
                  f"v{r['incident']['version']}, {r['incident']['date']})"
                  for r in flagged_scored) + ".")
    a(f"- **{len(flagged) - len(flagged_scored)} build slots** likewise "
      "(strict low-confidence rule; see pilot note above).")
    a("- Sensitivity (taking low-confidence attributions at face value): "
      "+1 scored window (forge35 s13) and +1 build slot (pilot-eval v3) -> "
      f"**{len(flagged_scored) + 1} scored windows + "
      f"{len(flagged) - len(flagged_scored) + 1} build slots** upper bound.")
    a("- The WARN is warn-only by design; the counterfactual claim is that these "
      "windows would have been submitted **with the prior incident references in "
      "hand** (e.g. \"this family died v45+v62 already\"), not that they would "
      "necessarily have been withheld. For the drift family that reference trail "
      "pointed at build_notebook.py 8 days and 3 burned windows before the manual "
      "root-cause hunt found it.")
    a("")
    a("## 5. Limitations")
    a("")
    a("- Scored-rail reruns are hidden: tier-2 fingerprints can never include stack "
      "traces; families are slug/provenance/class only.")
    a("- inc-t2-002 (TIPS deadlock) and inc-t1-001..003 (pilot IndexErrors) have no "
      "surviving local artifacts; they are recorded at low confidence and excluded "
      "from evidence counts.")
    a("- `kaggle competitions submissions` returned 403 during backfill (2026-07-18); "
      "scored-rail records come from runs/submission_log.jsonl + ITERATION_LOG.md.")
    a("- Kernel docker/machine metadata is read from the CURRENT kernel-metadata.json "
      "files; historical metadata drift is not reconstructed.")
    a("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fingerprint_backfill.py",
        description=(
            "WRITER of runs/failure_fingerprints.json. Rebuilds the two-tier "
            "failure-fingerprint store from curated campaign history + every "
            "retained kernel log on disk, and regenerates "
            "runs/failure_fingerprints_backfill.md. Run this BEFORE "
            "scripts/fingerprint_report.py (the read-only consumer)."),
        epilog=("The store is rebuilt from scratch every run - it is a pure "
                "function of the curated tables in this file plus the retained "
                "logs, so it is safe to re-run at any time and idempotent."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute everything and print the diff vs the store "
                         "on disk, but WRITE NOTHING")
    ap.add_argument("--store", default=None,
                    help="override the store path (default: runs/failure_fingerprints.json)")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_path = Path(args.store) if args.store else STORE_PATH

    incidents, scanned = build_incidents()
    rep = replay(incidents)
    fams = family_table(incidents)
    store = empty_store()
    store["updated"] = dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    store["incidents"] = incidents
    store["scan_log"] = scanned

    old = load_store(store_path)
    old_ids = {i.get("id") for i in old.get("incidents", [])}
    new_ids = {i.get("id") for i in incidents}
    added = [i for i in incidents if i["id"] not in old_ids]
    removed = sorted(old_ids - new_ids)

    if args.dry_run:
        print(f"DRY RUN - nothing written (store on disk: {store_path}, "
              f"{len(old.get('incidents', []))} incidents, "
              f"updated {old.get('updated')})")
    else:
        save_store(store, store_path)
        REPORT_PATH.write_text(write_report(incidents, scanned, rep, fams),
                               encoding="utf-8")

    flagged = [r for r in rep["rows"] if r["warn_active"]]
    print(f"store: {store_path} ({len(incidents)} incidents)")
    if not args.dry_run:
        print(f"report: {REPORT_PATH}")
    print(f"logs scanned: {len(scanned)} "
          f"({sum(1 for s in scanned if s['has_error'])} carried a failure signal)")
    print(f"recurring families (n>=2): "
          f"{sum(1 for f in fams if f['n'] >= 2)}")
    print(f"deaths flagged-in-advance (strict): {len(flagged)} "
          f"({sum(1 for r in flagged if r['incident']['tier'] == 2)} scored windows)")
    if not args.quiet:
        if added:
            print(f"NEW incidents vs store on disk ({len(added)}):")
            for i in added:
                print(f"  + {i['id']}  {i['date']}  t{i['tier']}  "
                      f"{i.get('kernel') or '-'}  fp={i['fingerprint']}  "
                      f"{i.get('source')}")
        if removed:
            print(f"incident ids no longer present ({len(removed)}): "
                  f"{', '.join(removed)}")
        if not added and not removed:
            print("no change vs the store on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
