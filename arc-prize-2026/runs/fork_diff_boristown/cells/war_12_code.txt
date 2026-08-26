# ============================================================================
# Cell 12 - Customization hook: warpack fork-band adoption grafts
# Paste this whole block into the duckfork notebook's customization-hook cell.
# Runs AFTER `bm` is unpickled (cell 10) and the bundled sources are
# importable (cell 8). With the fast-submit gate, this cell is inert during
# an interactive Save Version (RUN_HEAVY False -> bm is None).
#
# Sourcing order for the warpack module:
#   1. an attached Kaggle dataset containing warpack_patch.py (marker file,
#      mount-path agnostic)
#   2. a warpack/ directory next to this notebook (local interactive runs)
#
# Failure policy: unless WARPACK_STRICT=1, any patch failure prints a
# traceback and the run continues as VANILLA duck (score = baseline, never 0).
# ============================================================================
import os
import sys
import traceback
from pathlib import Path

WARPACK_MARKER = "warpack_patch.py"
WARPACK_STRICT = os.environ.get("WARPACK_STRICT", "").strip() in {"1", "true"}

# --- warpack arm configuration (env-driven; defaults = all grafts ON)
os.environ.setdefault("WARPACK_ENABLE", "1")         # master kill switch
os.environ.setdefault("WARPACK_BANKING", "1")        # max-over-plays win replay
os.environ.setdefault("WARPACK_RECOVERY", "1")       # GAME_OVER-loop / lock-in refresh
os.environ.setdefault("WARPACK_SHORTCIRCUIT", "1")   # stop homogeneous no-op batches
os.environ.setdefault("WARPACK_RETRY_GUARD", "1")    # report-only counters
os.environ.setdefault("WARPACK_BANK_MIN_TIME", "120")


def _find_warpack_dir() -> Path | None:
    candidates: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(marker.parent for marker in kaggle_input.rglob(WARPACK_MARKER))
    here = Path.cwd()
    for probe in (here / "warpack", here, here.parent / "warpack"):
        if (probe / WARPACK_MARKER).is_file():
            candidates.append(probe)
    return candidates[0] if candidates else None


if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)
    print("warpack: fast-submit save (RUN_HEAVY=False) - customization skipped")
else:
    try:
        warpack_dir = _find_warpack_dir()
        if warpack_dir is None:
            raise FileNotFoundError(
                f"{WARPACK_MARKER} not found under /kaggle/input or ./warpack - "
                "attach the warpack dataset to the notebook."
            )
        if str(warpack_dir) not in sys.path:
            sys.path.insert(0, str(warpack_dir))
        import warpack_patch

        cfg = warpack_patch.apply(bm)  # noqa: F821 - bm from cell 10
        version = getattr(warpack_patch, "VERSION", "v1")
        bm.label = f"{bm.label}-warpack-{version}"  # noqa: F821
        print(f"warpack {version}: patches applied from {warpack_dir}")
        print(
            "warpack: banking={0.enable_banking} recovery={0.enable_recovery} "
            "shortcircuit={0.enable_shortcircuit} retry_guard={0.enable_retry_guard} "
            "bank_min_time={0.bank_min_time_s} bank_strict={0.bank_strict_frames} "
            "recovery_repeats={0.recovery_repeat_threshold} "
            "recovery_gameovers={0.recovery_gameover_threshold}".format(cfg)
        )
    except Exception:
        if WARPACK_STRICT:
            raise
        print("warpack: PATCH FAILED - continuing with VANILLA duck harness")
        traceback.print_exc()
