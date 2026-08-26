# ============================================================================
# Cell 12 additions — Hypothesis Ledger + Goal-Family Escalation graft (R2)
# Paste this block into the duckfork notebook's customization-hook cell,
# AFTER `bm` is unpickled (cell 10) and the bundled sources are importable
# (cell 8). It composes with phase1/scheduler/taaf_grafts hooks: apply it
# after them (patch layers are independent monkeypatches).
#
# Sourcing order for the ledger module:
#   1. an attached Kaggle dataset containing ledger_patch.py + ledger_core.py
#      (located by marker filename, mount-path agnostic)
#   2. a ledger/ directory next to this notebook (local interactive runs)
#
# Failure policy: unless LEDGER_STRICT=1, any patch failure prints a traceback
# and the run continues as VANILLA duck (score = baseline, never 0).
# Rollback: set LEDGER_FLAGS="" (all flags off = proven no-op).
# ============================================================================
import os
import sys
import traceback
from pathlib import Path

LEDGER_MARKER = "ledger_patch.py"
LEDGER_STRICT = os.environ.get("LEDGER_STRICT", "").strip() in {"1", "true"}

# --- flag configuration (A/B arm switch; both flags ship ON for the R2 arm)
os.environ.setdefault("LEDGER_FLAGS", "ledger,escalation")


def _find_ledger_dir() -> Path | None:
    candidates: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(marker.parent for marker in kaggle_input.rglob(LEDGER_MARKER))
    here = Path.cwd()
    for probe in (here / "ledger", here, here.parent / "ledger"):
        if (probe / LEDGER_MARKER).is_file():
            candidates.append(probe)
    return candidates[0] if candidates else None


try:
    ledger_dir = _find_ledger_dir()
    if ledger_dir is None:
        raise FileNotFoundError(
            f"{LEDGER_MARKER} not found under /kaggle/input or ./ledger — "
            "attach the ledger dataset to the notebook."
        )
    if str(ledger_dir) not in sys.path:
        sys.path.insert(0, str(ledger_dir))
    import ledger_patch

    flags = ledger_patch.apply(bm)
    version = getattr(ledger_patch, "VERSION", "v1")
    print(f"ledger {version}: patches applied from {ledger_dir}")
    print(f"ledger: flags={flags}")
except Exception:
    if LEDGER_STRICT:
        raise
    traceback.print_exc()
    print("ledger: patch failed — continuing as vanilla duck (baseline score)")
