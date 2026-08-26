"""exec-WM patch: how the arm's code enters the certified field-floor vehicle.

The pattern is the campaign's proven patch-cell pattern (P1's 8-anchor shadow
patch is the template): the vehicle bundle is COPIED, a NEW module is written
(zero anchor risk), and exactly ONE anchored replacement lands in
inference/framework/solver.py so every analyzer the solver constructs is
wrapped by exec-WM. Every anchor asserts count == 1 at apply time -- upstream
drift dies LOUDLY (ExecWMFatalDrift), never a silent stock run.

Graceful degradation is layered:
  * apply-time: anchor drift  -> ExecWMFatalDrift (INFRA DEATH, loud)
  * import-time: wrap failure -> stock analyzer + "[execwm] wrap-failed" line
  * run-time: controller error -> per-level fallback to the stock agent
  * env kill-switch ARC3_EXECWM=0 -> byte-floor behaviour
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXEC_WM_SOURCE = HERE / "exec_wm.py"

# The vehicle (certification item: the arm rides the certified field floor's
# anim-20260807 bundle, never bundle_20260815 -- their tool_agent.py differ).
REPO = HERE.parents[1]
VEHICLE_BUNDLE = (REPO / "runs" / "harness_diff_0813" / "ds"
                  / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim"
                  / "src" / "ARC3-Inference")


class ExecWMFatalDrift(RuntimeError):
    """An anchor was not present exactly once. Never swallowed."""


# ANCHOR S1 -- the solver's analyzer constructor. Verified count==1 in the
# anim-20260807 vehicle (the def line; call sites read "self._make_analyzer(").
ANCHOR_MAKE_ANALYZER = "    def _make_analyzer(\n"

WRAP_BLOCK = '''    def _make_analyzer(self, game, index, local_server=None):
        inner = self._make_analyzer_stock(game, index, local_server)
        try:
            from inference.agent.exec_wm import maybe_wrap_analyzer
            return maybe_wrap_analyzer(inner, game=game, index=index, solver=self)
        except Exception as exc:
            print(f"[execwm] wrap-failed {type(exc).__name__}: {exc} -- "
                  "stock analyzer in use", flush=True)
            return inner

    def _make_analyzer_stock(
'''


def exec_wm_source_text() -> str:
    return EXEC_WM_SOURCE.read_text(encoding="utf-8")


def exec_wm_sha() -> str:
    return hashlib.sha256(exec_wm_source_text().encode("utf-8")).hexdigest()[:16]


def apply_execwm_patch(bundle_root: Path, source_text: str | None = None) -> dict:
    """Patch a WORKING COPY of the ARC3-Inference tree in place.

    bundle_root: the copied .../ARC3-Inference directory.
    source_text: exec_wm.py module source (defaults to this repo's copy; the
                 notebook cell passes its own embedded copy).
    """
    bundle_root = Path(bundle_root)
    solver_path = bundle_root / "inference" / "framework" / "solver.py"
    agent_dir = bundle_root / "inference" / "agent"
    if not solver_path.is_file() or not agent_dir.is_dir():
        raise ExecWMFatalDrift(f"not an ARC3-Inference tree: {bundle_root}")

    src = source_text if source_text is not None else exec_wm_source_text()
    module_path = agent_dir / "exec_wm.py"
    module_path.write_text(src, encoding="utf-8")

    text = solver_path.read_text(encoding="utf-8")
    n = text.count(ANCHOR_MAKE_ANALYZER)
    if n != 1:
        raise ExecWMFatalDrift(
            f"anchor x{n} in solver.py: {ANCHOR_MAKE_ANALYZER!r}")
    if "_make_analyzer_stock" in text:
        raise ExecWMFatalDrift("solver.py already patched (double apply)")
    text = text.replace(ANCHOR_MAKE_ANALYZER, WRAP_BLOCK, 1)
    solver_path.write_text(text, encoding="utf-8")

    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    return {"module": str(module_path), "solver": str(solver_path),
            "exec_wm_sha": sha}
