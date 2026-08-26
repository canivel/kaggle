# ============================================================================
# Cell 12 — Customization hook: Phase-1 exploration substrate (self-contained)
# Paste this whole block into the duckfork notebook's customization-hook cell.
# It runs AFTER `bm` is unpickled (cell 10) and the bundled TAAF/ARC3-Inference
# sources are importable (cell 8).
#
# Sourcing order for the phase1 module:
#   1. an attached Kaggle dataset containing phase1_patch.py + phase1_core.py
#      (any dataset — located by marker filename, mount-path agnostic)
#   2. a phase1/ directory next to this notebook (local interactive runs)
#
# Failure policy: unless PHASE1_STRICT=1, any patch failure prints a traceback
# and the run continues as VANILLA duck (score = baseline, never 0).
# ============================================================================
import os
import sys
import traceback
from pathlib import Path

PHASE1_MARKER = "phase1_patch.py"
PHASE1_STRICT = os.environ.get("PHASE1_STRICT", "").strip() in {"1", "true"}

# --- Phase-1 arm configuration (env-var driven; defaults = pre-registered v2 arm)
os.environ.setdefault("PHASE1_EXPLORE_AFTER_TURNS", "10")  # p100 of organic streaks
os.environ.setdefault("PHASE1_EXPLORE_BUDGET", "6")        # v2: 8 -> 6
os.environ.setdefault("PHASE1_MAX_EXPLORES", "3")          # v2: 6 -> 3
os.environ.setdefault("PHASE1_EXPLORE_MIN_LEVEL_ACTIONS", "90")  # v2 mode detector
os.environ.setdefault("PHASE1_EXPLORE_LEVELUP_COOLDOWN", "20")   # v2 momentum guard
os.environ.setdefault("PHASE1_EVICT_LOW_FRAC", "0.5")      # evict-to-half watermark


def _find_phase1_dir() -> Path | None:
    candidates: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(marker.parent for marker in kaggle_input.rglob(PHASE1_MARKER))
    here = Path.cwd()
    for probe in (here / "phase1", here, here.parent / "phase1"):
        if (probe / PHASE1_MARKER).is_file():
            candidates.append(probe)
    return candidates[0] if candidates else None


try:
    phase1_dir = _find_phase1_dir()
    if phase1_dir is None:
        raise FileNotFoundError(
            f"{PHASE1_MARKER} not found under /kaggle/input or ./phase1 — "
            "attach the phase1 dataset to the notebook."
        )
    if str(phase1_dir) not in sys.path:
        sys.path.insert(0, str(phase1_dir))
    import phase1_patch

    cfg = phase1_patch.apply(bm)
    version = getattr(phase1_patch, "VERSION", "v1")
    bm.label = f"{bm.label}-phase1-{version}"
    print(f"phase1 {version}: patches applied from {phase1_dir}")
    print(
        "phase1: explore_after_turns={0.explore_after_turns} budget={0.explore_probe_budget} "
        "max_explores={0.max_explores_per_game} "
        "min_level_actions={1} levelup_cooldown={2} "
        "animation={0.enable_animation} "
        "repl_archive={0.enable_repl_archive} evict_hysteresis={0.enable_evict_hysteresis} "
        "evict_low_frac={0.evict_low_frac}".format(
            cfg,
            getattr(cfg, "explore_min_level_actions", "n/a"),
            getattr(cfg, "explore_levelup_cooldown", "n/a"),
        )
    )

    # Sanity: vLLM prefix caching must be on in the solver's setup command.
    try:
        setup_cmds = list(getattr(bm.solver, "kaggle_setup_commands", []) or [])
        if setup_cmds and not any("--enable-prefix-caching" in c for c in setup_cmds):
            print("phase1: WARNING — vLLM setup command lacks --enable-prefix-caching")
        else:
            print("phase1: vLLM --enable-prefix-caching confirmed in setup command")
    except Exception:
        print("phase1: could not verify vLLM prefix-caching flag (non-fatal)")
except Exception:
    if PHASE1_STRICT:
        raise
    print("phase1: PATCH FAILED — continuing with VANILLA duck harness")
    traceback.print_exc()
