#!/usr/bin/env python
"""
Leaderboard differ for ARC-AGI-3.  Companion to scripts/lb_archive.py.

Given two dated archives it reports, per team:
    dScore, dSubmissions, rank change, entries/exits,
    and the decisive derived quantity  dScore per dSubmission.

WHY dScore/dSub IS THE POINT: the public score is a MAXIMUM over a team's submissions.  A team
that adds 20 draws to an unchanged agent will drift upward for free.  dScore/dSub separates a
capability step (large dScore on few new draws) from buying draws off a max-over-N board.

CONTROL ARM (first-class, never derived by the reader):
    Jack Cole / jcole75  -- MindsAI, originator of test-time training for ARC, ARC Prize 2025 3rd.
    Tufa Labs            -- his 2025 teammates; authors of the duck harness we fork.
    They submit constantly and they know exactly how to swap an engine.
    IF THESE TWO DO NOT MOVE, THE COMMODITY-ENGINE / SHARED-REGIME STORY IS WEAK.

DISCIPLINE: this instrument measures scores and submission counts.  Those are the only things it
may claim.  It never infers method from movement.  See learnings/lb_probe_README_2026-08-15.md.

Usage:
    python scripts/lb_diff.py                          # latest two FULL archives
    python scripts/lb_diff.py 2026-08-15 2026-08-16
    python scripts/lb_diff.py --allow-partial 2026-08-14 2026-08-15   # top20 vs full, degraded
    python scripts/lb_diff.py --md learnings/lb_diff_2026-08-16.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys

# The board carries non-cp1252 team names (CJK, emoji).  On Windows the default console
# codec raised UnicodeEncodeError at print(report) AFTER all the work was done -- a silent
# step-8 failure that looked like a script bug rather than a missing diff.  2026-08-17.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import io
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lb_archive import (  # noqa: E402
    BAND_HI,
    BAND_LO,
    LB_DIR,
    WATCHLIST,
    band_stats,
    find_watch,
    load_archive,
    team_key,
)

MOVE_EPS = 0.005  # a score is "moved" if it changed by more than this


# ---------------------------------------------------------------------------
# archive resolution
# ---------------------------------------------------------------------------
def resolve(spec, lb_dir=LB_DIR, prefer_full=True):
    if spec is None:
        return None
    if os.path.exists(spec):
        return spec
    cands = []
    if prefer_full:
        cands.append(os.path.join(lb_dir, "lb_full_%s.csv" % spec))
    cands.append(os.path.join(lb_dir, "lb_%s.csv" % spec))
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit("no archive for %r (looked for %s)" % (spec, ", ".join(cands)))


def latest_two_full(lb_dir=LB_DIR):
    files = sorted(glob.glob(os.path.join(lb_dir, "lb_full_*.csv")))
    if len(files) < 2:
        return files
    return files[-2:]


# ---------------------------------------------------------------------------
# diff core
# ---------------------------------------------------------------------------
def diff(old_rows, new_rows):
    o = {team_key(r): r for r in old_rows}
    n = {team_key(r): r for r in new_rows}
    recs = []
    for k, nr in n.items():
        orr = o.get(k)
        rec = {
            "key": k,
            "team_id": nr["TeamId"],
            "name": nr["TeamName"],
            "name_old": orr["TeamName"] if orr else None,
            "renamed": bool(orr and orr["TeamName"] != nr["TeamName"]),
            "status": "present" if orr else "ENTRY",
            "score_new": nr["Score"],
            "score_old": orr["Score"] if orr else None,
            "subs_new": nr["SubmissionCount"],
            "subs_old": orr["SubmissionCount"] if orr else None,
            "rank_new": nr["Rank"],
            "rank_old": orr["Rank"] if orr else None,
            "last_sub_new": nr["LastSubmissionDate"],
            "members": nr.get("TeamMemberUserNames"),
            "members_old": orr.get("TeamMemberUserNames") if orr else None,
        }
        rec["d_score"] = _sub(rec["score_new"], rec["score_old"])
        rec["d_subs"] = _sub(rec["subs_new"], rec["subs_old"])
        rec["d_rank"] = _sub(rec["rank_old"], rec["rank_new"])  # positive = moved UP the board
        rec["per_draw"] = per_draw(rec["d_score"], rec["d_subs"])
        rec["flags"] = flags(rec)
        recs.append(rec)
    for k, orr in o.items():
        if k in n:
            continue
        recs.append(
            {
                "key": k, "team_id": orr["TeamId"], "name": orr["TeamName"], "name_old": orr["TeamName"],
                "renamed": False, "status": "EXIT", "score_new": None, "score_old": orr["Score"],
                "subs_new": None, "subs_old": orr["SubmissionCount"], "rank_new": None,
                "rank_old": orr["Rank"], "last_sub_new": None, "members": orr.get("TeamMemberUserNames"),
                "members_old": orr.get("TeamMemberUserNames"),
                "d_score": None, "d_subs": None, "d_rank": None, "per_draw": None,
                "flags": ["EXIT"],
            }
        )
    annotate_merges(recs)
    return recs


# ---------------------------------------------------------------------------
# TEAM-MERGE detection (added 2026-08-31)
#
# WHY: SubmissionCount is ADDITIVE on a merge, so a merge looks exactly like a team
# that bought a pile of draws.  On 2026-08-31 this printed "Kyutai -- 18 new subs,
# DREW-NO-GAIN", which is a manufactured story about a well-funded lab brute-forcing
# the board: `rfbr` renamed to `Kyutai` (TeamId 16609552 stable), absorbed a member,
# and Hippolyte Pilchen's own team EXITed at #421 in the very same diff.  Nobody
# bought 18 draws.  A flag that invents competitor behaviour is worse than no flag.
#
# The check is member-set based: a merge ADDS usernames to the surviving team, and the
# absorbed team leaves the board in the same window.  Renames alone are NOT merges.
# ---------------------------------------------------------------------------
MERGE_MIN_DSUBS = 5  # only volunteer the weaker "grew, no matching exit" reading above this


def _member_set(raw):
    if not raw:
        return set()
    return {m.strip() for m in str(raw).replace(";", ",").split(",") if m.strip()}


def annotate_merges(recs):
    """Reclassify additive submission-count jumps that are merges, not bought draws."""
    exit_owner = {}
    for r in recs:
        if r["status"] == "EXIT":
            for member in _member_set(r.get("members")):
                exit_owner[member] = r["name"]

    for r in recs:
        if r["status"] != "present":
            continue
        new_members = _member_set(r.get("members"))
        old_members = _member_set(r.get("members_old"))
        if not new_members or not old_members:
            continue  # membership unknown on one side -- say nothing rather than guess
        added = new_members - old_members
        if not added:
            continue

        absorbed = sorted({exit_owner[m] for m in added if m in exit_owner})
        d_subs = r["d_subs"] or 0
        if not absorbed and d_subs < MERGE_MIN_DSUBS:
            continue

        r["merge"] = {
            "added_members": sorted(added),
            "absorbed_teams": absorbed,
            "d_subs": r["d_subs"],
            "confidence": "confirmed" if absorbed else "probable",
        }
        label = "TEAM-MERGE/%s(+%d member(s)%s; dSubs %s is ADDITIVE, not draws bought)" % (
            r["merge"]["confidence"],
            len(added),
            (" absorbing " + ", ".join(absorbed)) if absorbed else "",
            dfmt(r["d_subs"], "%+d"),
        )
        r["flags"] = [label] + [
            f for f in r["flags"] if not f.startswith("DREW-NO-GAIN")
        ]


def _sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def per_draw(d_score, d_subs):
    """dScore per NEW submission.  None where undefined; that is informative, not a gap."""
    if d_score is None or d_subs is None:
        return None
    if d_subs <= 0:
        return None
    return d_score / d_subs


def flags(rec):
    f = []
    if rec["status"] == "ENTRY":
        f.append("ENTRY")
        return f
    ds, dn = rec["d_score"], rec["d_subs"]
    if ds is not None and ds < -MOVE_EPS:
        # Score is a max over submissions; it should never fall.  A fall means rescore,
        # withdrawal or an archive artifact -- inspect, do not average over it.
        f.append("SCORE-FELL(anomalous)")
    if ds is not None and ds > MOVE_EPS and (dn == 0):
        f.append("MOVED-WITHOUT-NEW-SUBS(inspect: rescore or archive artifact)")
    if ds is not None and dn is not None:
        if ds > MOVE_EPS and dn > 0:
            f.append("STEP" if rec["per_draw"] and rec["per_draw"] >= 0.05 else "DRIFT")
        elif abs(ds) <= MOVE_EPS and dn and dn > 0:
            f.append("DREW-NO-GAIN(%d new subs, 0.00)" % dn)
        elif abs(ds) <= MOVE_EPS and dn == 0:
            f.append("IDLE")
    if dn is None:
        f.append("NO-SUBCOUNT(top20 archive)")
    return f


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def fmt(v, spec="%.2f", dash="--"):
    return dash if v is None else spec % v


def dfmt(v, spec="%+.2f", dash="--"):
    return dash if v is None else spec % v


def watch_rows(recs, group=None):
    out = []
    for label, grp, tid, nm, mm in WATCHLIST:
        if group and grp != group:
            continue
        rec = None
        for r in recs:
            probe = {
                "TeamId": r["team_id"],
                "TeamName": r["name"] or "",
                "TeamMemberUserNames": r.get("members"),
            }
            if find_watch([probe], tid, nm, mm):
                rec = r
                break
        out.append((label, grp, rec))
    return out


def render(old_meta, new_meta, recs, degraded, top_n=15):
    L = []
    A = L.append
    span = "%s -> %s" % (old_meta["date"], new_meta["date"])
    A("=" * 96)
    A("ARC-AGI-3 LEADERBOARD DIFF   %s" % span)
    A("=" * 96)
    A("old : %s" % os.path.relpath(old_meta["path"], os.getcwd()).replace("\\", "/"))
    A("      coverage=%-5s rows=%-5s SubmissionCount=%-5s pull_utc=%s (local %s)"
      % (old_meta["coverage"], old_meta["rows"], old_meta["has_submission_count"],
         old_meta.get("pull_utc"), old_meta.get("pull_local")))
    A("new : %s" % os.path.relpath(new_meta["path"], os.getcwd()).replace("\\", "/"))
    A("      coverage=%-5s rows=%-5s SubmissionCount=%-5s pull_utc=%s (local %s)"
      % (new_meta["coverage"], new_meta["rows"], new_meta["has_submission_count"],
         new_meta.get("pull_utc"), new_meta.get("pull_local")))
    if degraded:
        A("")
        A("!! DEGRADED DIFF -- at least one side is a TOP-20 archive.")
        A("   Consequences, stated so they are not forgotten:")
        A("     * dSubmissions is UNAVAILABLE on that side => dScore/dSub cannot be computed.")
        A("     * 'ENTRY' below may mean 'crossed that day's visibility floor', not 'new team'.")
        A("     * 'EXIT' may mean 'fell below the floor', not 'left the competition'.")
        for side, meta in (("old", old_meta), ("new", new_meta)):
            if not meta["has_submission_count"]:
                A("     * %s side (%s) is top-20 with a VISIBILITY FLOOR of %s: a team below that"
                  % (side, meta["date"], fmt(meta.get("floor"))))
                A("       score is invisible in it, so its absence bounds the score, nothing more.")
    A("")

    # ---- 1. CONTROL ARM ---------------------------------------------------
    A("-" * 96)
    A("1. CONTROL ARM  --  the two teams who wrote the TTT literature and the harness we fork")
    A("-" * 96)
    ctrl = watch_rows(recs, "CONTROL")
    A("%-22s %6s %6s %7s %6s %6s %7s %8s  %s"
      % ("team", "score", "prev", "dScore", "subs", "dSubs", "dRank", "d/draw", "flags"))
    moved = []
    for label, _g, r in ctrl:
        if r is None:
            A("%-22s  NOT FOUND IN THE NEW ARCHIVE -- investigate before reading this line" % label)
            continue
        A("%-22s %6s %6s %7s %6s %6s %7s %8s  %s"
          % (label, fmt(r["score_new"]), fmt(r["score_old"]), dfmt(r["d_score"]),
             fmt(r["subs_new"], "%d"), dfmt(r["d_subs"], "%+d"), dfmt(r["d_rank"], "%+d"),
             fmt(r["per_draw"], "%.4f"), ",".join(r["flags"])))
        moved.append((label, r))
    A("")
    A("   CONTROL-ARM READOUT (this is the line the campaign needs, not something to derive):")
    if not moved:
        A("   INDETERMINATE -- control teams not resolvable in these archives.")
    elif degraded:
        A("   INDETERMINATE -- a top-20 archive cannot carry dSubmissions for the control arm.")
    else:
        big = [(l, r) for l, r in moved if r["d_score"] is not None and r["d_score"] > MOVE_EPS]
        if not big:
            A("   NEITHER CONTROL TEAM MOVED (dScore <= %.3f over this window)." % MOVE_EPS)
            A("   => the commodity-engine / shared-regime story is WEAK on this evidence.")
            A("      Two teams with the means, the motive and the submission cadence to swap an")
            A("      engine did not gain. That is a measurement about SCORES, not about method.")
        else:
            for l, r in big:
                A("   %s MOVED %+.2f on %s new submissions (%s per new draw)."
                  % (l, r["d_score"], r["d_subs"],
                     fmt(r["per_draw"], "%.4f") if r["per_draw"] is not None else "n/a"))
            A("   => the shared-regime story survives this window. It is NOT confirmed: a score")
            A("      move is a score move. Method remains UNKNOWN unless someone discloses it.")
    A("")

    # ---- 2. watchlist -----------------------------------------------------
    A("-" * 96)
    A("2. WATCHLIST  --  top eight + us")
    A("-" * 96)
    A("%-22s %-8s %6s %6s %7s %6s %6s %7s %8s  %s"
      % ("team", "group", "score", "prev", "dScore", "subs", "dSubs", "dRank", "d/draw", "flags"))
    for label, grp, r in watch_rows(recs):
        if grp == "CONTROL":
            continue
        if r is None:
            A("%-22s %-8s  ABSENT from the new archive" % (label, grp))
            continue
        A("%-22s %-8s %6s %6s %7s %6s %6s %7s %8s  %s"
          % (label, grp, fmt(r["score_new"]), fmt(r["score_old"]), dfmt(r["d_score"]),
             fmt(r["subs_new"], "%d"), dfmt(r["d_subs"], "%+d"), dfmt(r["d_rank"], "%+d"),
             fmt(r["per_draw"], "%.4f"), ",".join(r["flags"])))
    A("")

    # ---- 3. band ----------------------------------------------------------
    A("-" * 96)
    A("3. THE %.2f-%.2f BAND  --  where we and the duck-harness lineage live (aggregate)" % (BAND_LO, BAND_HI))
    A("-" * 96)
    bo, bn = old_meta.get("band"), new_meta.get("band")
    if not (bo and bn) or degraded:
        A("   UNAVAILABLE -- a full board on both sides is required to count band membership.")
        if bn:
            A("   new side only: count=%s median_score=%s median_subs=%s"
              % (bn["count"], bn["median_score"], bn["median_subs"]))
    else:
        A("%-16s %8s %8s %8s" % ("", "old", "new", "delta"))
        A("%-16s %8d %8d %+8d" % ("teams in band", bo["count"], bn["count"], bn["count"] - bo["count"]))
        A("%-16s %8s %8s %8s" % ("median score", fmt(bo["median_score"]), fmt(bn["median_score"]),
                                 dfmt(_sub(bn["median_score"], bo["median_score"]))))
        A("%-16s %8s %8s %8s" % ("median subs", fmt(bo["median_subs"], "%.1f"),
                                 fmt(bn["median_subs"], "%.1f"),
                                 dfmt(_sub(bn["median_subs"], bo["median_subs"]), "%+.1f")))
        entered = set(bn["team_ids"]) - set(bo["team_ids"])
        left = set(bo["team_ids"]) - set(bn["team_ids"])
        A("   entered band: %d   left band: %d" % (len(entered), len(left)))
        A("   READ: a broad lift of this band is what a drop-in engine swap would look like in the")
        A("         score data. A flat band with a few teams stepping is what team-specific work")
        A("         looks like. Neither reading names a method.")
    A("")

    # ---- 4. movers --------------------------------------------------------
    A("-" * 96)
    A("4. LARGEST SCORE MOVES (present on both sides), top %d" % top_n)
    A("-" * 96)
    movers = [r for r in recs if r["d_score"] is not None and abs(r["d_score"]) > MOVE_EPS]
    movers.sort(key=lambda r: -abs(r["d_score"]))
    if not movers:
        A("   none -- no team present on both sides moved by more than %.3f." % MOVE_EPS)
    A("%-30s %6s %7s %6s %6s %8s  %s" % ("team", "score", "dScore", "subs", "dSubs", "d/draw", "flags"))
    for r in movers[:top_n]:
        A("%-30s %6s %7s %6s %6s %8s  %s"
          % (r["name"][:30], fmt(r["score_new"]), dfmt(r["d_score"]), fmt(r["subs_new"], "%d"),
             dfmt(r["d_subs"], "%+d"), fmt(r["per_draw"], "%.4f"), ",".join(r["flags"])))
    A("")

    # ---- 5. draws bought without gain ------------------------------------
    A("-" * 96)
    A("5. DRAWS BOUGHT vs GAIN  --  the best-of-N confound, measured")
    A("-" * 96)
    withsubs = [r for r in recs if r["d_subs"] is not None and r["d_subs"] > 0]
    # A merge's submission count is additive, so it is not a draw purchase.  Excluded
    # from every statistic below and named explicitly -- a silent exclusion would read
    # as "covered everything".
    merged = [r for r in withsubs if r.get("merge")]
    withsubs = [r for r in withsubs if not r.get("merge")]
    if not withsubs or degraded:
        A("   UNAVAILABLE -- needs SubmissionCount on both sides (full archives only).")
    else:
        gained = [r for r in withsubs if r["d_score"] is not None and r["d_score"] > MOVE_EPS]
        A("   teams that submitted at all      : %d" % len(withsubs))
        A("   ... of which gained anything     : %d (%.1f%%)"
          % (len(gained), 100.0 * len(gained) / len(withsubs)))
        A("   total new submissions on the board: %d" % sum(r["d_subs"] for r in withsubs))
        if gained:
            pds = sorted(r["per_draw"] for r in gained if r["per_draw"] is not None)
            A("   median dScore/dSub among gainers : %.4f" % statistics.median(pds))
            A("   max    dScore/dSub among gainers : %.4f" % pds[-1])
        A("")
        A("   most draws bought this window:")
        A("   %-30s %6s %6s %7s %8s  %s" % ("team", "score", "dSubs", "dScore", "d/draw", "flags"))
        for r in sorted(withsubs, key=lambda r: -r["d_subs"])[:top_n]:
            A("   %-30s %6s %6s %7s %8s  %s"
              % (r["name"][:30], fmt(r["score_new"]), dfmt(r["d_subs"], "%+d"), dfmt(r["d_score"]),
                 fmt(r["per_draw"], "%.4f"), ",".join(r["flags"])))
        A("")
        A("   READ: a team whose dScore/dSub is tiny across many new draws is climbing the max-over-N")
        A("         order statistic, not its agent. A large dScore on 1-3 draws is a step.")
    if merged:
        A("")
        A("   EXCLUDED as TEAM MERGES (submission counts ADD on a merge; these are not bought draws):")
        for r in sorted(merged, key=lambda r: -(r["d_subs"] or 0)):
            m = r["merge"]
            A("   %-30s dSubs %s  [%s] +%s%s"
              % (r["name"][:30], dfmt(r["d_subs"], "%+d"), m["confidence"],
                 ", ".join(m["added_members"]),
                 (" <- " + ", ".join(m["absorbed_teams"])) if m["absorbed_teams"] else ""))
    A("")

    # ---- 6. entries / exits ----------------------------------------------
    A("-" * 96)
    A("6. ENTRIES / EXITS")
    A("-" * 96)
    ents = sorted([r for r in recs if r["status"] == "ENTRY"], key=lambda r: (r["rank_new"] or 10 ** 9))
    exits = sorted([r for r in recs if r["status"] == "EXIT"], key=lambda r: (r["rank_old"] or 10 ** 9))
    A("   entries: %d   exits: %d" % (len(ents), len(exits)))
    for r in ents[:top_n]:
        A("   + #%-5s %-30s %5s  subs=%s" % (r["rank_new"], r["name"][:30], fmt(r["score_new"]),
                                             fmt(r["subs_new"], "%d")))
    if len(ents) > top_n:
        A("   ... and %d more entries" % (len(ents) - top_n))
    for r in exits[:top_n]:
        A("   - #%-5s %-30s %5s" % (r["rank_old"], r["name"][:30], fmt(r["score_old"])))
    if len(exits) > top_n:
        A("   ... and %d more exits" % (len(exits) - top_n))
    renamed = [r for r in recs if r.get("renamed")]
    if renamed:
        A("   renames (TeamId stable, TeamName changed): %d" % len(renamed))
        for r in renamed[:8]:
            A("     %s -> %s (id %s)" % (r["name_old"], r["name"], r["team_id"]))
    A("")

    A("-" * 96)
    A("SCOPE OF CLAIM: this instrument measures Score, SubmissionCount, Rank, TeamName and")
    A("LastSubmissionDate. It does NOT observe method, model, engine or prompt, and it CANNOT date")
    A("the scoring submission -- LastSubmissionDate is the team's MOST RECENT submission while Score")
    A("is their BEST, and they need not be the same one. Do not infer method from movement.")
    A("Evidence classes for any method statement remain DISCLOSED / INFERRED / UNKNOWN")
    A("(learnings/top6_evidence_audit_2026-08-15.md). Today's tally: 0 DISCLOSED / 1 INFERRED / 7 UNKNOWN.")
    A("-" * 96)
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("old", nargs="?", default=None, help="date YYYY-MM-DD or path")
    p.add_argument("new", nargs="?", default=None, help="date YYYY-MM-DD or path")
    p.add_argument("--lb-dir", default=LB_DIR)
    p.add_argument("--allow-partial", action="store_true",
                   help="permit a diff where one side is a top-20 archive (output is labelled DEGRADED)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--md", default=None, help="also write the report to this path")
    args = p.parse_args(argv)

    if args.old is None and args.new is None:
        files = latest_two_full(args.lb_dir)
        if len(files) < 2:
            print("Only %d full archive(s) in %s -- a diff needs two." % (len(files), args.lb_dir))
            print("Full archives present: %s" % ", ".join(os.path.basename(f) for f in files))
            print("Run scripts/lb_archive.py daily; the first diff is possible tomorrow.")
            if files:
                print("\nTop-20 archives exist for earlier days but carry NO SubmissionCount;")
                print("use --allow-partial with explicit dates for a labelled DEGRADED diff.")
            return 3
        oldp, newp = files
    else:
        if args.new is None:
            raise SystemExit("give two dates, or none for the latest two full archives")
        oldp, newp = resolve(args.old, args.lb_dir), resolve(args.new, args.lb_dir)

    old_rows, old_meta = load_archive(oldp)
    new_rows, new_meta = load_archive(newp)
    degraded = not (old_meta["has_submission_count"] and new_meta["has_submission_count"])
    if degraded and not args.allow_partial:
        print("REFUSING a full-board diff: one side lacks SubmissionCount.")
        print("  old %s coverage=%s   new %s coverage=%s"
              % (os.path.basename(oldp), old_meta["coverage"], os.path.basename(newp), new_meta["coverage"]))
        print("  Top-20 archives cannot support dSubmissions, dScore/dSub, or entry/exit semantics.")
        print("  Re-run with --allow-partial for a labelled DEGRADED diff.")
        return 4

    for rows, meta in ((old_rows, old_meta), (new_rows, new_meta)):
        scores = [r["Score"] for r in rows if r["Score"] is not None]
        meta["floor"] = min(scores) if scores else None
        meta["band"] = band_stats(rows) if meta["has_submission_count"] else None

    recs = diff(old_rows, new_rows)
    report = render(old_meta, new_meta, recs, degraded, top_n=args.top)
    print(report)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with io.open(args.md, "w", encoding="utf-8") as fh:
            fh.write("# LB diff %s -> %s\n\nGenerated %s by `scripts/lb_diff.py`.\n\n```\n%s\n```\n"
                     % (old_meta["date"], new_meta["date"],
                        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), report))
        print("\nwritten -> %s" % args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
