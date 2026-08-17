"""Re-audit the graft bundle against the sha sealed at audit time.

WHY THIS EXISTS. Kaggle attaches the LATEST version of a dataset and kernel metadata cannot pin
one. `thtennant/taaf-kaggle-source-share-fork` was republished 2026-08-17 00:26 and is actively
maintained. The 08-17 audit (`duck_eval/graft/bundle_audit_2026-08-17.md`) verified a specific
89-file tree; if those bytes change before the push, the arm is no longer the arm that was
sealed, and the mismatch must surface BEFORE the slot is spent rather than after the result is
read. `install(..., expected_version=1)` catches an API bump; this catches a same-API content
change, which is the quieter and more dangerous case.

Verified at audit time (recursive path+sha256 manifest, then sha256 of that manifest):
  full bundle, 89 files   df447f61caa181cca68049e28b139e02
  src/taaf-grafts/, 16    7705481551494b141d6a33ffec1d7a20
and, decisively, the fork was stock + 16 additive files with 0 stock files modified.

    python duck_eval/graft/graft_bundle_check.py              # download fresh + compare
    python duck_eval/graft/graft_bundle_check.py --local DIR  # compare an existing copy
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

FORK_DS = "thtennant/taaf-kaggle-source-share-fork"
AUDITED_BUNDLE_SHA = "df447f61caa181cca68049e28b139e02"
AUDITED_GRAFTS_SHA = "7705481551494b141d6a33ffec1d7a20"
AUDITED_N_FILES = 89
AUDITED_N_GRAFTS = 16
AUDITED_VERSION_DATE = "2026-08-17 00:26:06"

# Flags whose modules must exist for the sealed arm to be installable at all.
REQUIRED_MODULES = (
    "composite.py", "shortcircuit_solver.py", "goalkeep.py", "hudmask.py", "retry_guard.py",
)


def manifest_sha(root: Path, base: Path) -> tuple[str, int]:
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            items.append(f"{p.relative_to(base).as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:32], len(items)


def find_bundle_root(d: Path) -> Path:
    """The download may unzip into d or into a single nested dir."""
    for marker in d.rglob("taaf-kaggle-bundle.json"):
        return marker.parent
    raise SystemExit(f"BUNDLE CHECK FAIL: no taaf-kaggle-bundle.json under {d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="an already-downloaded copy (skips the download)")
    args = ap.parse_args()

    if args.local:
        root = find_bundle_root(Path(args.local))
    else:
        tmp = Path(tempfile.mkdtemp())
        print(f"downloading {FORK_DS} -> {tmp}", flush=True)
        subprocess.run(
            ["uvx", "--from", "kaggle==2.0.0", "kaggle", "datasets", "download",
             "-d", FORK_DS, "-p", str(tmp), "--unzip"],
            check=True, capture_output=True, text=True)
        root = find_bundle_root(tmp)

    bundle_sha, n_files = manifest_sha(root, root)
    grafts_dir = root / "src" / "taaf-grafts"
    if not grafts_dir.is_dir():
        raise SystemExit("BUNDLE CHECK FAIL: src/taaf-grafts/ is missing — this is the stock "
                         "bundle, not the graft fork")
    grafts_sha, n_grafts = manifest_sha(grafts_dir, root)

    print(f"bundle root   {root}")
    print(f"bundle files  {n_files:3d}  (audited {AUDITED_N_FILES})   sha {bundle_sha}")
    print(f"grafts files  {n_grafts:3d}  (audited {AUDITED_N_GRAFTS})   sha {grafts_sha}")

    missing = [m for m in REQUIRED_MODULES if not (grafts_dir / "taaf_grafts" / m).exists()]
    if missing:
        raise SystemExit(f"BUNDLE CHECK FAIL: required graft modules missing: {missing}")

    # install() must still live where cell 12 imports it from, and still read our flag names.
    composite = (grafts_dir / "taaf_grafts" / "composite.py").read_text(encoding="utf-8")
    if "def install(" not in composite:
        raise SystemExit("BUNDLE CHECK FAIL: install() is gone from composite.py")
    if "GRAFTS_API_VERSION = 1" not in composite:
        raise SystemExit("BUNDLE CHECK FAIL: GRAFTS_API_VERSION is no longer 1 — the sealed arm's "
                         "flag semantics are not guaranteed; re-audit before pushing")

    ok = (bundle_sha == AUDITED_BUNDLE_SHA and grafts_sha == AUDITED_GRAFTS_SHA
          and n_files == AUDITED_N_FILES and n_grafts == AUDITED_N_GRAFTS)
    if ok:
        print(f"BUNDLE CHECK OK — byte-identical to the {AUDITED_VERSION_DATE} audited version.")
        return 0

    print()
    print("*** BUNDLE CHANGED SINCE THE AUDIT ***")
    print(f"  expected bundle sha {AUDITED_BUNDLE_SHA}, got {bundle_sha}")
    print(f"  expected grafts sha {AUDITED_GRAFTS_SHA}, got {grafts_sha}")
    print()
    print("  This does NOT necessarily mean the arm is invalid — the publisher may have added an")
    print("  unrelated module. But the sealed arm was defined against the audited bytes, so:")
    print("    1. re-diff the fork against jeroencottaar/taaf-kaggle-source-share (the audit's")
    print("       decisive check was: stock + N additive files, 0 stock files MODIFIED)")
    print("    2. re-read composite.py's flag table and install() contract")
    print("    3. update the sealed shas in the prereg, the builder and graft_score.py TOGETHER,")
    print("       and say in the prereg that you did and why")
    print("  Do NOT spend the slot until that is done.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
