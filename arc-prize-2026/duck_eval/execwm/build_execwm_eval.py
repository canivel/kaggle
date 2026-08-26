"""Builder: stage notebooks/execwm-eval/ from the certified field-floor vehicle.

Deterministic: vehicle bytes are NEVER edited -- the only change is ONE
inserted patch cell at position 6 (the P1/P2 patch-cell position) plus a
retargeted kernel-metadata.json. Cell 6 embeds duck_eval/execwm/exec_wm.py
verbatim (single source of truth; ewm_smoke S1 asserts byte-consistency by
sha), copies the mounted bundle to a working dir, writes the module, applies
the ONE anchored solver.py replacement (count==1 asserted, ExecWMFatalDrift
otherwise), shadows the path, and prints the boot marker.

Usage: uv run python duck_eval/execwm/build_execwm_eval.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ewm_patch  # noqa: E402

VEHICLE_NB = REPO / "notebooks" / "q38-field-eval" / "arc3-q38-field-eval.ipynb"
VEHICLE_META = REPO / "notebooks" / "q38-field-eval" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "execwm-eval"
OUT_NB = OUT_DIR / "arc3-execwm-eval.ipynb"
KERNEL_ID = "canivel/arc3-execwm-eval"
PATCH_CELL_INDEX = 6

CELL_TEMPLATE = '''# EXEC-WM PATCH CELL (prereg execwm_prereg_2026-08-25.md; arm: execwm).
# Executable world model: mine per-action object rules from recorded history,
# verify them prequentially, BFS-plan inside the verified program, and fall
# back PER LEVEL to the stock agent (the certified floor) everywhere else.
# Patch pattern: bundle copied, ONE new module written, ONE anchored solver.py
# replacement asserted count==1 -- drift dies LOUDLY, never a silent stock run.
import hashlib, shutil, sys

assert "inference" not in sys.modules, "EXECWM FATAL: inference imported before patch cell"

_EWM_SHA = {sha!r}
_EWM_SOURCE = r{source_literal}

assert hashlib.sha256(_EWM_SOURCE.encode("utf-8")).hexdigest()[:16] == _EWM_SHA, \\
    "EXECWM FATAL: embedded module bytes do not match the build-time sha"

_ewm_src_root = None
for _cand in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"):
    _c = _cand.parent / "src" / "ARC3-Inference"
    if _c.is_dir():
        _ewm_src_root = _c
        break
assert _ewm_src_root is not None, "EXECWM FATAL: bundle ARC3-Inference not found"

_ewm_dst = Path("/kaggle/working/execwm_patched/ARC3-Inference")
if _ewm_dst.exists():
    shutil.rmtree(_ewm_dst)
shutil.copytree(_ewm_src_root, _ewm_dst)

(_ewm_dst / "inference" / "agent" / "exec_wm.py").write_text(_EWM_SOURCE, encoding="utf-8")

_ewm_solver = _ewm_dst / "inference" / "framework" / "solver.py"
_ewm_anchor = {anchor!r}
_ewm_wrap = {wrap!r}
_ewm_text = _ewm_solver.read_text(encoding="utf-8")
_ewm_n = _ewm_text.count(_ewm_anchor)
assert _ewm_n == 1, f"EXECWM FATAL: solver anchor x{{_ewm_n}} (vehicle drifted)"
assert "_make_analyzer_stock" not in _ewm_text, "EXECWM FATAL: double apply"
_ewm_solver.write_text(_ewm_text.replace(_ewm_anchor, _ewm_wrap, 1), encoding="utf-8")

sys.path.insert(0, str(_ewm_dst))
import inference.framework.solver as _ewm_chk
assert str(_ewm_dst) in str(Path(_ewm_chk.__file__)), \\
    f"EXECWM FATAL: wrong module resolved: {{_ewm_chk.__file__}}"

os.environ["ARC3_EXECWM"] = "1"
os.environ.setdefault("ARC3_EXECWM_LLM", "1")
print(f"[execwm] patch applied sha={{_EWM_SHA}} shadowed at {{_ewm_dst}}", flush=True)
'''


def build() -> Path:
    src = ewm_patch.exec_wm_source_text()
    assert "'''" not in src, "exec_wm.py may not contain triple single quotes"
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cell_source = CELL_TEMPLATE.format(
        sha=sha,
        source_literal="'''" + src + "'''",
        anchor=ewm_patch.ANCHOR_MAKE_ANALYZER,
        wrap=ewm_patch.WRAP_BLOCK,
    )

    nb = json.loads(VEHICLE_NB.read_text(encoding="utf-8"))
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_source.splitlines(keepends=True),
    }
    nb = copy.deepcopy(nb)
    nb["cells"].insert(PATCH_CELL_INDEX, cell)

    meta = json.loads(VEHICLE_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = "arc3-execwm-eval"
    meta["code_file"] = OUT_NB.name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {OUT_NB} ({len(nb['cells'])} cells, exec_wm sha {sha})")
    return OUT_NB


if __name__ == "__main__":
    build()
