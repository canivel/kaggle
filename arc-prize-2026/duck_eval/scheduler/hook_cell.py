# ============================================================================
# Cell 12 — Customization hook: R1 attempt scheduler (self-contained)
# Paste this whole block into the duckfork notebook's customization-hook cell.
# It runs AFTER `bm` is unpickled (cell 10) and the bundled TAAF/ARC3-Inference
# sources are importable (cell 8).
#
# Sourcing order for the scheduler module:
#   1. an attached Kaggle dataset containing scheduler_patch.py
#      (any dataset — located by marker filename, mount-path agnostic)
#   2. a scheduler/ directory next to this notebook (local interactive runs)
#
# Failure policy: unless SCHED_STRICT=1, any patch failure prints a traceback
# and the run continues as VANILLA duck (score = baseline, never 0).
# ============================================================================
import os
import sys
import traceback
from pathlib import Path

SCHED_MARKER = "scheduler_patch.py"
SCHED_STRICT = os.environ.get("SCHED_STRICT", "").strip() in {"1", "true"}

# --- R1 arm configuration (env-var driven; defaults = pre-registered R1 arm)
os.environ.setdefault("SCHED_ENABLE", "1")        # kill switch: 0 = vanilla
os.environ.setdefault("SCHED_RESTART_AT", "90")   # null10 p90 time-to-first-level
os.environ.setdefault("SCHED_MAX_RESTARTS", "2")  # cumulative cap; park after


def _find_sched_dir() -> Path | None:
    candidates: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(marker.parent for marker in kaggle_input.rglob(SCHED_MARKER))
    here = Path.cwd()
    for probe in (here / "scheduler", here, here.parent / "scheduler"):
        if (probe / SCHED_MARKER).is_file():
            candidates.append(probe)
    return candidates[0] if candidates else None


try:
    sched_dir = _find_sched_dir()
    if sched_dir is None:
        raise FileNotFoundError(
            f"{SCHED_MARKER} not found under /kaggle/input or ./scheduler — "
            "attach the scheduler dataset to the notebook."
        )
    if str(sched_dir) not in sys.path:
        sys.path.insert(0, str(sched_dir))
    import scheduler_patch

    cfg = scheduler_patch.apply(bm)
    version = getattr(scheduler_patch, "VERSION", "v1")
    if cfg.enable:
        bm.label = f"{bm.label}-sched-{version}"
    print(f"sched {version}: applied from {sched_dir}")
    print(
        f"sched: enable={cfg.enable} restart_at={cfg.restart_at} "
        f"max_restarts={cfg.max_restarts} (per-attempt no-level trigger; "
        f"cumulative cap; park after cap; NO context injection)"
    )
except Exception:
    if SCHED_STRICT:
        raise
    print("sched: PATCH FAILED — continuing with VANILLA duck harness")
    traceback.print_exc()
