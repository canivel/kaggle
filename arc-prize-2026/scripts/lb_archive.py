#!/usr/bin/env python
"""
Full-leaderboard daily archiver for ARC-AGI-3.  (probe #1 of learnings/top6_evidence_audit_2026-08-15.md)

WHAT THIS MEASURES, AND ONLY THIS: scores, submission counts, ranks, team identity.
It does NOT and MUST NOT infer method from movement.  See learnings/lb_probe_README_2026-08-15.md.

Two things this fixes:
  1. We archived only the top-20, so rank/score history below the visibility floor is
     unreconstructible.  This archives all ~2331 rows.
  2. `SubmissionCount` was in the download all along and never used.  It is the only field that
     separates "a real step" from "bought another draw off a max-over-N leaderboard".

HEARTBEAT: every run writes runs/lb_daily/heartbeat/lb_archive_<date>.json, success OR failure.
Standing incident (ARCMorningCheck refused 2 days via MultipleInstancesPolicy=IgnoreNew, unnoticed
because a refused task looks exactly like a healthy idle one): silence from an automation must not
read as success.  `--check` asserts the dated artifact exists and is fresh (exit 1 if not) AND that the
PRIOR day's archive exists (exit 2 if not) -- a diff needs both sides, and a healthy today hid a
missing yesterday on 2026-08-27.

Usage:
    python scripts/lb_archive.py                 # pull + archive + heartbeat
    python scripts/lb_archive.py --check         # assert today's heartbeat exists (exit 1 if not)
    python scripts/lb_archive.py --index         # (re)write the archive coverage index
    python scripts/lb_archive.py --dry-run       # show what would happen, no network
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile

COMP = "arc-prize-2026-arc-agi-3"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB_DIR = os.path.join(REPO, "runs", "lb_daily")
HB_DIR = os.path.join(LB_DIR, "heartbeat")
DEFAULT_KAGGLE = "uvx --from kaggle==2.0.0 kaggle"

# Canonical schema of the `leaderboard -d` download, in source order.
FULL_COLUMNS = [
    "Rank",
    "TeamId",
    "TeamName",
    "LastSubmissionDate",
    "Score",
    "SubmissionCount",
    "TeamMemberUserNames",
]

# ---------------------------------------------------------------------------
# WATCHLIST.  Rationale is in learnings/lb_probe_README_2026-08-15.md.
# Keyed on TeamId where known (stable across renames -- verified 2026-08-15, team 15564282
# renamed "Stepwise" -> "Sankalp" inside 29 minutes with TeamId unchanged), with name/member
# fallbacks so a team that re-forms under a new id is still caught.
# ---------------------------------------------------------------------------
WATCHLIST = [
    # (label, group, team_id or None, name_match, member_match)
    ("Jack Cole (MindsAI)", "CONTROL", "15587108", "jack cole", "jcole75"),
    ("Tufa Labs", "CONTROL", "15486995", "tufa labs", "jeroencottaar"),
    ("cstl", "TOP8", "16364346", "cstl", "tehnar"),
    ("Daniel Franzen", "TOP8", "16384837", "daniel franzen", "dfranzen"),
    ("Nikita Sorokin", "TOP8", "16438894", "nikita sorokin", "nikitasorokin"),
    ("Yusaku Muroya", "TOP8", None, "yusaku muroya", "ymuroya47"),
    ("AbeLincoln1865", "TOP8", None, "abelincoln1865", "abelincoln1865"),
    ("YUTO KOJIMA", "TOP8", None, "yuto kojima", "kojimatech"),
    ("MLRush", "TOP8", None, "mlrush", "mlrush"),
    ("Andy liu", "TOP8", None, "andy liu", "codinggodandyliu"),
    ("Canivel (US)", "US", "15503635", "canivel", "canivel"),
]

# The 1.55-1.65 band: where we and the duck-harness lineage live.  Tracked as an aggregate.
BAND_LO, BAND_HI = 1.55, 1.65


def _now_local():
    return dt.datetime.now()


def today_str():
    return _now_local().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# archive loading / schema normalisation
# ---------------------------------------------------------------------------
def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def load_archive(path):
    """Read either schema and return (rows, meta).

    Two schemas exist in runs/lb_daily/:
      FULL   `lb_full_<date>.csv`  Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,
                                   TeamMemberUserNames        <- from `leaderboard -d`
      TOP20  `lb_<date>.csv`       [rank,]teamId,teamName,submissionDate,score
                                                              <- from `leaderboard --show -v | head`
    TOP20 archives carry NO SubmissionCount and NO team members.  They cannot support a
    full-board diff and the differ refuses to pretend otherwise.
    """
    with io.open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        raw = list(rdr)
        fields = list(rdr.fieldnames or [])

    lower = {c.lower(): c for c in fields}
    is_full = "submissioncount" in lower
    rows = []
    for idx, r in enumerate(raw):
        get = lambda *names: next(
            (r[lower[n]] for n in names if n in lower and r.get(lower[n]) not in (None, "")), None
        )
        rank = _i(get("rank"))
        if rank is None:
            rank = idx + 1  # derived from source order; valid only inside this archive's view
        rows.append(
            {
                "Rank": rank,
                "TeamId": (get("teamid") or "").strip(),
                "TeamName": (get("teamname") or "").strip(),
                "LastSubmissionDate": get("lastsubmissiondate", "submissiondate"),
                "Score": _f(get("score")),
                "SubmissionCount": _i(get("submissioncount")),
                "TeamMemberUserNames": get("teammemberusernames"),
            }
        )
    meta = {
        "path": path,
        "coverage": "full" if is_full else "top20",
        "rows": len(rows),
        "has_submission_count": is_full,
        "has_native_rank": "rank" in lower,
        "source_fields": fields,
        "date": _date_from_name(path),
        "sha256": sha256_file(path),
    }
    # Attach the server-side pull time from the sidecar heartbeat -- but ONLY if the heartbeat's
    # sha256 matches this exact file. Date alone is not enough provenance: two snapshots of the
    # same day (e.g. an intraday re-pull) would otherwise both claim the same pull time, which is
    # precisely the class of field-semantics error this whole instrument exists to stop.
    hb = os.path.join(HB_DIR, "lb_archive_%s.json" % meta["date"]) if meta["date"] else None
    if hb and os.path.exists(hb):
        try:
            with io.open(hb, encoding="utf-8") as fh:
                h = json.load(fh)
            if h.get("sha256") == meta["sha256"]:
                meta["pull_utc"] = h.get("pull_utc")
                meta["pull_local"] = h.get("pull_local")
        except Exception:
            pass
    if "pull_utc" not in meta:
        # no matching heartbeat: fall back to file mtime and SAY SO rather than imply provenance
        meta["pull_utc"] = None
        meta["pull_local"] = "mtime %s" % dt.datetime.fromtimestamp(
            os.path.getmtime(path)
        ).strftime("%Y-%m-%d %H:%M:%S")
    return rows, meta


def _date_from_name(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def team_key(row):
    """Stable identity.  TeamId is the join key -- names change, ids do not."""
    return row["TeamId"] or ("name:" + row["TeamName"].lower())


def match_watch(row, team_id, name_match, member_match):
    if team_id and row["TeamId"] == team_id:
        return True
    if name_match and row["TeamName"].strip().lower() == name_match:
        return True
    if member_match:
        members = (row.get("TeamMemberUserNames") or "").lower()
        if member_match in [m.strip() for m in members.split(",") if m.strip()]:
            return True
    return False


def find_watch(rows, team_id, name_match, member_match):
    for r in rows:
        if match_watch(r, team_id, name_match, member_match):
            return r
    return None


def band_stats(rows, lo=BAND_LO, hi=BAND_HI):
    inb = [r for r in rows if r["Score"] is not None and lo <= r["Score"] <= hi]
    subs = [r["SubmissionCount"] for r in inb if r["SubmissionCount"] is not None]
    return {
        "lo": lo,
        "hi": hi,
        "count": len(inb),
        "median_score": round(statistics.median([r["Score"] for r in inb]), 4) if inb else None,
        "median_subs": statistics.median(subs) if subs else None,
        "team_ids": sorted(r["TeamId"] for r in inb),
    }


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------
def pull_full_leaderboard(comp, kaggle_cmd, workdir):
    """`leaderboard -d` -> zip -> csv.  Returns (csv_path, pull_utc_iso, cmd_str).

    Verified 2026-08-15: this is the ONLY form that yields SubmissionCount + Rank.
    `--show -v` returns 4 columns (teamId,teamName,submissionDate,score), is paginated, and its
    --page-token windows are non-contiguous (defect already logged in runs/lb_ground_truth.md).
    """
    # NOTE: kaggle 2.0.0 has no --force on this subcommand; we download into a fresh temp dir
    # each run so there is never a stale zip to skip over.
    cmd = kaggle_cmd.split() + ["competitions", "leaderboard", comp, "-d", "-p", workdir]
    cmd_str = " ".join(cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("kaggle CLI failed (rc=%s):\n%s\n%s" % (proc.returncode, proc.stdout, proc.stderr))
    zips = glob.glob(os.path.join(workdir, "*.zip"))
    if not zips:
        raise RuntimeError("no zip produced by: %s\n%s\n%s" % (cmd_str, proc.stdout, proc.stderr))
    with zipfile.ZipFile(zips[0]) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("no csv inside %s" % zips[0])
        # The member name contains colons ("...T14:50:25.csv") which are illegal on Windows,
        # so extract by hand into a sanitised filename rather than zf.extract().
        member = names[0]
        safe = re.sub(r"[:]", "_", os.path.basename(member))
        csv_path = os.path.join(workdir, safe)
        with zf.open(member) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    # Kaggle embeds the server-side pull time (UTC) in the member name:
    #   arc-prize-2026-arc-agi-3-publicleaderboard-2026-08-15T14:50:25.csv
    m = re.search(r"publicleaderboard-(\d{4}-\d{2}-\d{2})T(\d{2})[:_](\d{2})[:_](\d{2})", member)
    pull_utc = "%sT%s:%s:%sZ" % m.groups() if m else None
    return csv_path, pull_utc, cmd_str


def write_archive(src_csv, out_path):
    """Copy through a csv round-trip: strip BOM, normalise line endings, keep column order."""
    with io.open(src_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
        cols = list(rdr.fieldnames or FULL_COLUMNS)
        cols = [c.lstrip("﻿") for c in cols]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, r.get("﻿" + c, "")) for c in cols})
    return len(rows), cols


def snapshot_watchlist(rows):
    out = []
    for label, group, tid, nm, mm in WATCHLIST:
        r = find_watch(rows, tid, nm, mm)
        out.append(
            {
                "label": label,
                "group": group,
                "found": r is not None,
                "team_id": r["TeamId"] if r else tid,
                "team_name": r["TeamName"] if r else None,
                "rank": r["Rank"] if r else None,
                "score": r["Score"] if r else None,
                "submissions": r["SubmissionCount"] if r else None,
                "last_submission_date": r["LastSubmissionDate"] if r else None,
            }
        )
    return out


def write_heartbeat(date, payload):
    os.makedirs(HB_DIR, exist_ok=True)
    path = os.path.join(HB_DIR, "lb_archive_%s.json" % date)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def build_index(lb_dir=LB_DIR):
    """Coverage index over every archive we hold.  States plainly which days are top-20-only."""
    entries = []
    for path in sorted(glob.glob(os.path.join(lb_dir, "lb_*.csv"))):
        rows, meta = load_archive(path)
        scores = [r["Score"] for r in rows if r["Score"] is not None]
        entries.append(
            {
                "file": os.path.basename(path),
                "date": meta["date"],
                "coverage": meta["coverage"],
                "rows": meta["rows"],
                "has_submission_count": meta["has_submission_count"],
                "visibility_floor": min(scores) if scores else None,
                "pull_utc": meta.get("pull_utc"),
                "pull_local": meta.get("pull_local"),
                "sha256": meta["sha256"][:16],
                "supports_full_board_diff": meta["coverage"] == "full",
            }
        )
    idx = {
        "generated": _now_local().strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "top20 archives were produced by `leaderboard --show -v | head -N`: no SubmissionCount, "
            "no team members, and a visibility floor equal to the min score shown. They CANNOT "
            "support a full-board diff or any delta-submissions claim. Full archives come from "
            "`leaderboard -d`."
        ),
        "archives": entries,
    }
    out = os.path.join(lb_dir, "archive_index.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2)
        fh.write("\n")
    return out, idx


def cmd_check(args):
    """Assert the dated heartbeat exists and is today's.  Exit 1 otherwise. Silence != success."""
    date = args.date or today_str()
    path = os.path.join(HB_DIR, "lb_archive_%s.json" % date)
    if not os.path.exists(path):
        print("HEARTBEAT MISSING for %s (%s)" % (date, path))
        print("  -> the LB archiver did NOT run today. Do not treat this as 'nothing changed'.")
        return 1
    with io.open(path, encoding="utf-8") as fh:
        hb = json.load(fh)
    if hb.get("status") != "OK":
        print("HEARTBEAT PRESENT BUT status=%s for %s" % (hb.get("status"), date))
        print("  error: %s" % hb.get("error"))
        return 1
    csv_path = os.path.join(REPO, hb["archive_relpath"])
    if not os.path.exists(csv_path):
        print("HEARTBEAT OK but archive missing: %s" % csv_path)
        return 1
    if sha256_file(csv_path) != hb["sha256"]:
        print("HEARTBEAT sha256 MISMATCH -- archive was modified after the pull: %s" % csv_path)
        return 1
    print(
        "HEARTBEAT OK  %s  rows=%s  pull_utc=%s  sha=%s"
        % (date, hb["rows"], hb.get("pull_utc"), hb["sha256"][:12])
    )

    # ---- PRIOR-DAY ASSERTION (added 2026-08-27) --------------------------------------------
    # A healthy today is NOT enough: a diff needs BOTH sides.  On 2026-08-27 the Mac had a
    # perfectly OK today-heartbeat and no 08-26 archive at all (the `*.csv` gitignore rule
    # dropped 08-22..08-26 during the machine move), so `lb_diff` was blind while every check
    # in this function passed.  A separate exit code, deliberately: "tomorrow's diff will be
    # blind" is a DIFFERENT incident from "the archiver did not run", and collapsing them into
    # exit 1 is how a real gap gets read as a routine failure and skipped.
    prev = (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()
    prev_full = os.path.join(LB_DIR, "lb_full_%s.csv" % prev)
    prev_top = os.path.join(LB_DIR, "lb_%s.csv" % prev)
    if not os.path.exists(prev_full):
        degraded = os.path.exists(prev_top)
        print("PRIOR-DAY ARCHIVE MISSING for %s (%s)" % (prev, prev_full))
        if degraded:
            print("  -> only a top-20 archive exists: NO SubmissionCount, so no dScore/dSub and")
            print("     no full-board diff. --allow-partial gives a labelled DEGRADED diff only.")
        else:
            print("  -> there is NO yesterday. A diff against %s is impossible; any dScore" % date)
            print("     column reported for today has no artifact behind it.")
        print("  -> Kaggle serves only TODAY's board: a missing archive cannot be re-pulled.")
        print("     Recover it from the other machine if one exists, or accept the gap.")
        return 2
    print("PRIOR-DAY OK  %s present -- a full diff %s -> %s is possible" % (prev, prev, date))
    return 0


def cmd_pull(args):
    date = args.date or today_str()
    out_path = os.path.join(args.outdir, "lb_full_%s.csv" % date)
    if args.dry_run:
        print("[dry-run] would run: %s competitions leaderboard %s -d" % (args.kaggle_cmd, args.comp))
        print("[dry-run] would write: %s" % out_path)
        print("[dry-run] would write heartbeat: %s" % os.path.join(HB_DIR, "lb_archive_%s.json" % date))
        return 0

    started = dt.datetime.now(dt.timezone.utc)
    workdir = tempfile.mkdtemp(prefix="lb_archive_")
    hb = {
        "artifact": "lb_archive",
        "date": date,
        "competition": args.comp,
        "status": "FAILED",
        "run_started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_started_local": _now_local().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        src, pull_utc, cmd_str = pull_full_leaderboard(args.comp, args.kaggle_cmd, workdir)
        prior_sha = sha256_file(out_path) if os.path.exists(out_path) else None
        n, cols = write_archive(src, out_path)
        rows, _meta = load_archive(out_path)
        scores = [r["Score"] for r in rows if r["Score"] is not None]
        hb.update(
            {
                "status": "OK",
                "source_command": cmd_str,
                "source_filename": os.path.basename(src),
                "pull_utc": pull_utc,
                "pull_local": _now_local().strftime("%Y-%m-%d %H:%M:%S"),
                "archive_relpath": os.path.relpath(out_path, REPO).replace("\\", "/"),
                "rows": n,
                "columns": cols,
                "has_submission_count": "SubmissionCount" in cols,
                "sha256": sha256_file(out_path),
                "prior_sha256_same_day": prior_sha,
                "overwrote_same_day_archive": bool(prior_sha) and prior_sha != sha256_file(out_path),
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "watchlist": snapshot_watchlist(rows),
                "band_1_55_1_65": band_stats(rows),
                "measures": ["Score", "SubmissionCount", "Rank", "TeamName", "LastSubmissionDate"],
                "does_not_measure": [
                    "method", "model, engine or prompt used", "which submission produced Score",
                ],
            }
        )
        print("archived %s rows -> %s" % (n, out_path))
        print("  pull_utc=%s  SubmissionCount=%s" % (pull_utc, hb["has_submission_count"]))
    except Exception as exc:  # heartbeat is written on failure too -- that is the whole point
        hb["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("LB ARCHIVE FAILED: %s" % hb["error"], file=sys.stderr)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    hbp = write_heartbeat(date, hb)
    print("heartbeat -> %s (status=%s)" % (hbp, hb["status"]))
    idx_path, _ = build_index(args.outdir)
    print("index -> %s" % idx_path)
    return 0 if hb["status"] == "OK" else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comp", default=COMP)
    p.add_argument("--outdir", default=LB_DIR)
    p.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today (local)")
    p.add_argument("--kaggle-cmd", default=DEFAULT_KAGGLE)
    p.add_argument("--check", action="store_true",
                   help="assert today's heartbeat AND yesterday's archive; "
                        "exit 1 if today is absent/stale, 2 if the prior day is missing")
    p.add_argument("--index", action="store_true", help="rebuild archive_index.json only, no network")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.check:
        return cmd_check(args)
    if args.index:
        path, idx = build_index(args.outdir)
        print("index -> %s (%d archives)" % (path, len(idx["archives"])))
        for e in idx["archives"]:
            print(
                "  %s  %-6s rows=%-5s subs=%-5s floor=%s"
                % (e["date"], e["coverage"], e["rows"], e["has_submission_count"], e["visibility_floor"])
            )
        return 0
    return cmd_pull(args)


if __name__ == "__main__":
    sys.exit(main())
