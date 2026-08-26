"""P1 builder: persistent tool namespace (`notes`) on the certified field floor.

PREREG: learnings/war_room/p1_notes_prereg_2026-08-22.md (sealed first).
Vehicle = FOYSAL bytes (sha 7227f3286cf60b25) + ONE inserted patch cell right after the
setup cell. The patch cell copies the bundle's read-only ARC3-Inference to a working dir,
applies 7 exact-anchor replacements (each asserted count==1 -> any drift is a loud cell
error = INFRA DEATH, never a silent stock run), and path-shadows the patched package
BEFORE anything imports `inference` (asserted).
"""
import hashlib
import json
from pathlib import Path

FOYSAL_NB = Path(r"C:/Users/dcani/AppData/Local/Temp/claude/f--kaggle/62c35e7c-0d05-4da2-99b0-f9b400a45a97/scratchpad/rethink0820/foysal_lb9/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb")
FOYSAL_META = Path(r"C:/Users/dcani/AppData/Local/Temp/claude/f--kaggle/62c35e7c-0d05-4da2-99b0-f9b400a45a97/scratchpad/rethink0820/foysal_lb9/kernel-metadata.json")
BASE_SHA = "7227f3286cf60b25"
OUT_ROOT = Path(r"F:/kaggle/arc-prize-2026/notebooks")
KERNEL_ID = "canivel/arc3-p1-notes-eval"

PATCH_CELL = r'''
# P1 PERSISTENT TOOL NAMESPACE (prereg p1_notes_prereg_2026-08-22.md). Patch the bundle's
# ARC3-Inference in a WORKING copy and path-shadow it. Every replacement asserts count==1:
# any upstream drift dies LOUDLY here (INFRA DEATH), never a silent stock run.
import shutil, sys
assert "inference" not in sys.modules, "P1 FATAL: inference imported before patch cell"
_p1_src = None
for _cand in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"):
    _c = _cand.parent / "src" / "ARC3-Inference"
    if _c.is_dir():
        _p1_src = _c
        break
assert _p1_src is not None, "P1 FATAL: bundle ARC3-Inference not found"
_p1_dst = Path("/kaggle/working/p1_patched/ARC3-Inference")
if _p1_dst.exists():
    shutil.rmtree(_p1_dst)
shutil.copytree(_p1_src, _p1_dst)

def _p1_patch(path, pairs):
    t = path.read_text(encoding="utf-8")
    for old, new in pairs:
        n = t.count(old)
        assert n == 1, f"P1 FATAL: anchor x{n} in {path.name}: {old[:60]!r}"
        t = t.replace(old, new)
    path.write_text(t, encoding="utf-8")

_sb = _p1_dst / "inference" / "agent" / "python_tool_sandbox.py"
_ta = _p1_dst / "inference" / "agent" / "tool_agent.py"

_p1_patch(_sb, [
    # S1 sandbox: inject notes into runtime_globals from the initial payload
    ('            "result": None,\n        }',
     '            "result": None,\n            "notes": initial.get("notes") if isinstance(initial.get("notes"), dict) else {},\n        }'),
    # S2 sandbox: return notes in the final payload
    ('                    "result": _json_safe(runtime_globals.get("result")),',
     '                    "result": _json_safe(runtime_globals.get("result")),\n                    "notes": _json_safe(runtime_globals.get("notes")),'),
    # S3 host: accept + forward notes
    ('    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,\n) -> dict[str, Any]:',
     '    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,\n    notes: dict[str, Any] | None = None,\n) -> dict[str, Any]:'),
    ('                "animation_enabled": animation_handler is not None,',
     '                "animation_enabled": animation_handler is not None,\n                "notes": notes if isinstance(notes, dict) else {},'),
])
_p1_patch(_ta, [
    # T5 armed marker (module import; the certification line)
    ('from inference.agent.python_tool_sandbox import run_sandboxed_python',
     'from inference.agent.python_tool_sandbox import run_sandboxed_python\nprint("[notes] persistent-namespace armed", flush=True)'),
    # T1 pass notes into the sandbox
    ('            animation_handler=_handle_animation if self._animation_awareness_enabled else None,\n        )',
     '            animation_handler=_handle_animation if self._animation_awareness_enabled else None,\n            notes=getattr(self, "_p1_notes", None),\n        )'),
    # T2 capture returned notes (8KB cap), write-marker, clear on run_complete
    ('        payload: dict[str, Any] = {"tool": "python"}',
     """        _p1r = sandbox_result.get("notes")
        if isinstance(_p1r, dict):
            try:
                import json as _p1j
                if len(_p1j.dumps(_p1r)) <= 8192:
                    if _p1r != getattr(self, "_p1_notes", None):
                        print("[notes] wrote", flush=True)
                    self._p1_notes = _p1r
                else:
                    print("[notes] oversize-dropped", flush=True)
            except Exception:
                pass
        if any(bool(i.get("run_complete")) for i in action_results):
            self._p1_notes = {}
        payload: dict[str, Any] = {"tool": "python"}"""),
    # T4 the 12-word prompt change: remove the obligation
    ('"Python code to run. The snippet is ephemeral and is not saved across tool calls."',
     '"Python code to run. `notes` is a dict that persists across tool calls within this game; everything else is ephemeral."'),
])

sys.path.insert(0, str(_p1_dst))
import importlib
import inference.agent.python_tool_sandbox as _p1chk
assert str(_p1_dst) in str(Path(_p1chk.__file__)), f"P1 FATAL: wrong module resolved: {_p1chk.__file__}"
print(f"[notes] P1 patch applied: 8 anchors, shadowed at {_p1_dst}", flush=True)
'''


def code_of(nb):
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def build():
    nb = json.loads(FOYSAL_NB.read_text(encoding="utf-8"))
    base = code_of(nb)
    assert hashlib.sha256(base.encode()).hexdigest()[:16] == BASE_SHA, "vehicle drifted"

    setup_idx = pkl_idx = None
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if '_run_shell_commands("setup_commands.json"' in src:
            setup_idx = i
        if 'deploy_target.pkl' in src and pkl_idx is None:
            pkl_idx = i
    assert setup_idx is not None and pkl_idx is not None and setup_idx < pkl_idx, \
        f"cell order: setup={setup_idx} pkl={pkl_idx}"

    cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": PATCH_CELL.strip().splitlines(keepends=True)}
    nb["cells"].insert(setup_idx + 1, cell)

    code = code_of(nb)
    for good in ('[notes] persistent-namespace armed', 'assert "inference" not in sys.modules',
                 "persists across tool calls within this game", "8192"):
        assert good in code, good
    for bad in ("taaf_grafts", "install(bm", "reasoning_effort", "EDGE1", "= 3960.0", "= 23760.0"):
        assert bad not in code, bad

    meta = json.loads(FOYSAL_META.read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta["id"] = KERNEL_ID
    meta["title"] = "arc3-p1-notes-eval"
    meta["code_file"] = "arc3-p1-notes-eval.ipynb"
    meta["is_private"] = True

    out = OUT_ROOT / "p1-notes-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "arc3-p1-notes-eval.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sha = hashlib.sha256(code.encode()).hexdigest()[:16]
    print(f"built {out}  cells={len(nb['cells'])} (base 11 + patch at {setup_idx + 1}) code_sha256={sha}")
    print("patch cell precedes pkl-load; fail-loud anchors x8; forbidden tokens absent")


if __name__ == "__main__":
    build()
