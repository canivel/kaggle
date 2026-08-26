"""Count-by-fingerprint report over runs/failure_fingerprints.json.

Named consumer #2 of the two-tier failure-fingerprint store (Kimi-3 adopt #3):
prints the top recurring failure families (count, first/last seen, incident
refs) for inclusion in the daily brief.

READ-ONLY. The store is WRITTEN by scripts/fingerprint_backfill.py — run that
FIRST, always. Between 2026-07-18 and 2026-08-16 nobody did, so this report
silently described a store that was 20 days and 3 incidents behind the logs
sitting on disk, and the 2026-08-09 weekly recorded "no new incidents in ~4.5
weeks" as a finding. Hence the staleness banner below: this reader now checks
itself against the same log inventory the writer scans.

Usage:
  uv run python scripts/fingerprint_backfill.py          # WRITE first, always
  uv run python scripts/fingerprint_report.py --brief    # compact table for the brief
  uv run python scripts/fingerprint_report.py            # full: families + incidents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import (  # noqa: E402
    CANDIDATE_MATCHABLE_PREFIXES, STORE_PATH, family_index,
    format_staleness_banner, load_store, staleness_report,
)


def family_rows(store: dict) -> list[dict]:
    idx = family_index(store)
    rows = []
    for fam, incs in idx.items():
        rows.append({
            "family": fam,
            "n": len(incs),
            "first": incs[0].get("date"),
            "last": incs[-1].get("date"),
            "tiers": sorted({i.get("tier") for i in incs}),
            "matchable": fam.startswith(CANDIDATE_MATCHABLE_PREFIXES),
            "refs": [f"{i.get('id')}={i.get('kernel') or '?'}"
                     + (f" v{i['version']}" if i.get("version") is not None else "")
                     for i in incs],
        })
    rows.sort(key=lambda r: (-r["n"], r["family"]))
    return rows


def print_table(rows: list[dict], top: int | None = None,
                min_n: int = 1) -> None:
    rows = [r for r in rows if r["n"] >= min_n]
    if top:
        rows = rows[:top]
    if not rows:
        print("(no failure families on record)")
        return
    w = max(len(r["family"]) for r in rows)
    hdr = f"{'family':<{w}}  {'n':>2}  {'first':<10}  {'last':<10}  refs"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        refs = ", ".join(r["refs"][:4]) + (" ..." if len(r["refs"]) > 4 else "")
        print(f"{r['family']:<{w}}  {r['n']:>2}  {r['first'] or '?':<10}  "
              f"{r['last'] or '?':<10}  {refs}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true",
                    help="compact table (recurring families only) for the daily brief")
    ap.add_argument("--store", default=None, help="override store path")
    ap.add_argument("--top", type=int, default=10,
                    help="max families in --brief mode (default 10)")
    ap.add_argument("--no-stale-check", action="store_true",
                    help="skip the staleness check (do not use in the weekly duty)")
    ap.add_argument("--fail-on-stale", action="store_true",
                    help="exit 3 instead of 0 when the store is stale")
    args = ap.parse_args()

    store = load_store(Path(args.store) if args.store else None)
    incs = store.get("incidents", [])
    rows = family_rows(store)

    stale = None
    if not args.no_stale_check:
        stale = staleness_report(store)
        print(format_staleness_banner(stale))
        print()

    rc = 3 if (args.fail_on_stale and stale and stale["stale"]) else 0

    if args.brief:
        print(f"failure fingerprints: {len(incs)} incidents, "
              f"{sum(1 for r in rows if r['n'] >= 2)} recurring families "
              f"(store: {args.store or STORE_PATH})")
        print_table(rows, top=args.top, min_n=2)
        return rc

    print(f"store: {args.store or STORE_PATH}")
    print(f"incidents: {len(incs)} | families: {len(rows)} "
          f"(recurring: {sum(1 for r in rows if r['n'] >= 2)})")
    print()
    print_table(rows)
    print()
    print("incidents (chronological):")
    for i in sorted(incs, key=lambda x: (x.get("date") or "", x.get("id") or "")):
        print(f"  {i.get('id')}  {i.get('date')}  t{i.get('tier')}  "
              f"{i.get('kernel') or '-'}"
              f"{' v' + str(i['version']) if i.get('version') is not None else ''}  "
              f"{i.get('status_class')}/{i.get('score_class')}  "
              f"fp={i.get('fingerprint')}  [{i.get('confidence')}]")
        print(f"      {i.get('note', '')[:110]}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
