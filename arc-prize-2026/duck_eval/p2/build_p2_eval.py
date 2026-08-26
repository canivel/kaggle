"""P2 builder: reset-anchored episodic retry on the certified field floor.

PREREG: ``learnings/war_room/p2_reset_retry_prereg_2026-08-22.md`` (SEALED 2026-08-22).
FIREABILITY GATE: ``learnings/war_room/p2_trigger_fireability_2026-08-26.md`` --
the H=4 stuck trigger fires on 19/25 games on THIS vehicle against a sealed
D1 bar of >=15/25, measured before this build existed.

VEHICLE = the certified field floor, byte-untouched, + ONE inserted patch cell at
position 6 (after cell 5's setup_commands, before the first ``inference`` import).
Same pattern and same insertion point as P1 and exec-WM.

The patch cell embeds ``p2_patch.py`` VERBATIM -- the same module the two smokes
exercise (episode leg 18/18, trigger leg 50/50). There is exactly one source of
truth for the anchors; the notebook cannot drift from the tested code.

BOOT CHECK (`[p2] reset semantics OK`). ``attempt()`` is composed entirely from
the existing ``action()`` primitive and rests on ONE invariant of the shipped
bundle: RESET is a first-class action that is ALWAYS legal
(``taaf/game.py``: *"Legal action ids, with RESET (0) always present"*, and
``inference/agent/action_names.py`` maps it). If a bundle update ever removed
that, ``attempt()`` would silently stop resetting and every episode would
corrupt the level instead of probing it. So the cell asserts the invariant at
boot, in the bundle actually mounted, and dies LOUDLY if it has moved.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

VEHICLE_NB = REPO / "notebooks" / "q38-field-eval" / "arc3-q38-field-eval.ipynb"
VEHICLE_META = REPO / "notebooks" / "q38-field-eval" / "kernel-metadata.json"
OUT_DIR = REPO / "notebooks" / "p2-retry-eval"
OUT_NB = OUT_DIR / "arc3-p2-retry-eval.ipynb"
KERNEL_ID = "canivel/arc3-p2-retry-eval"
PATCH_CELL_INDEX = 6

CELL_TEMPLATE = '''# P2 RESET-ANCHORED EPISODIC RETRY (prereg p2_reset_retry_prereg_2026-08-22.md; arm: p2).
# attempt(seq) runs a candidate sequence from the CURRENT LEVEL START, reports what it
# reached, then RESETs back to that same start -- so one LLM turn can evaluate K candidate
# plans instead of committing to one. Actions are cheap (eps=0.17); turns are the binding
# constraint. A stuck trigger (H consecutive acting turns on one uncleared level) arms it.
#
# FIREABILITY MEASURED BEFORE THIS BUILD (p2_trigger_fireability_2026-08-26.md):
# 19/25 games on THIS vehicle vs a sealed D1 bar of >=15/25; 6/25 correctly REFUSE.
#
# Patch pattern: bundle copied, p2_patch.py embedded VERBATIM and applied. Every anchor is
# asserted count==1 -- drift dies LOUDLY here (INFRA DEATH), never a silent stock run.
import hashlib, shutil, sys

assert "inference" not in sys.modules, "P2 FATAL: inference imported before patch cell"

# The module carries BOTH quote styles (docstrings inside a triple-single-quoted
# block), so it is embedded base64 -- no literal can be broken by its own content,
# and the sha still binds the exact build-time bytes.
import base64 as _p2_b64
_P2_SHA = {sha!r}
_P2_MODULE = _p2_b64.b64decode({source_b64!r}).decode("utf-8")

assert hashlib.sha256(_P2_MODULE.encode("utf-8")).hexdigest()[:16] == _P2_SHA, \\
    "P2 FATAL: embedded patch-module bytes do not match the build-time sha"

_p2_src_root = None
for _cand in Path("/kaggle/input").rglob("taaf-kaggle-bundle.json"):
    _c = _cand.parent / "src" / "ARC3-Inference"
    if _c.is_dir():
        _p2_src_root = _c
        _p2_bundle_root = _cand.parent
        break
assert _p2_src_root is not None, "P2 FATAL: bundle ARC3-Inference not found"

# ---- BOOT CHECK: the invariant attempt() rests on, in the bundle actually mounted ----
# attempt() issues its own RESET to return to the level start. That is only sound while
# RESET is always-legal. Assert it here rather than discovering it from a corrupted run.
_p2_reset_ok = False
for _g in _p2_bundle_root.rglob("taaf/game.py"):
    _gt = _g.read_text(encoding="utf-8", errors="replace")
    if "RESET (0) always present" in _gt and "return [0, *raw]" in _gt:
        _p2_reset_ok = True
        break
assert _p2_reset_ok, "P2 FATAL: RESET-always-legal invariant not found in the mounted taaf"
_p2_names = (_p2_src_root / "inference" / "agent" / "action_names.py").read_text(encoding="utf-8")
assert '"RESET": "RESET"' in _p2_names, "P2 FATAL: RESET is not a first-class action name"
print("[p2] reset semantics OK", flush=True)

_p2_dst = Path("/kaggle/working/p2_patched/ARC3-Inference")
if _p2_dst.exists():
    shutil.rmtree(_p2_dst)
shutil.copytree(_p2_src_root, _p2_dst)

_p2_mod_path = Path("/kaggle/working/p2_patch_embedded.py")
_p2_mod_path.write_text(_P2_MODULE, encoding="utf-8")
sys.path.insert(0, str(_p2_mod_path.parent))
import p2_patch_embedded as _p2_patch

_p2_info = _p2_patch.apply_patch(_p2_dst)
assert _p2_info["anchors_applied"] == 6, f"P2 FATAL: anchors {{_p2_info['anchors_applied']}} != 6"
assert _p2_info["sandbox_is_vehicle_generation"], \\
    f"P2 FATAL: sandbox md5 {{_p2_info['sandbox_md5_before']}} is not the vehicle generation"

sys.path.insert(0, str(_p2_dst))
import inference.agent.tool_agent as _p2_chk
assert str(_p2_dst) in str(Path(_p2_chk.__file__)), \\
    f"P2 FATAL: wrong module resolved: {{_p2_chk.__file__}}"
for _m in ("_p2_note_acting_turn", "_p2_retry_armed", "_p2_count_attempt_calls", "_p2_flush"):
    assert any(_m in dir(_c) for _c in
               [getattr(_p2_chk, _n) for _n in dir(_p2_chk) if isinstance(getattr(_p2_chk, _n), type)]), \\
        f"P2 FATAL: {{_m}} not bound on any class in the patched tool_agent"

print(_p2_info["banner"], flush=True)
print(f"[p2] patch applied sha={{_P2_SHA}} shadowed at {{_p2_dst}}", flush=True)
'''


def build() -> Path:
    raw = (HERE / "p2_patch.py").read_bytes()
    src = raw.decode("utf-8")
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]

    cell_source = CELL_TEMPLATE.format(
        sha=sha, source_b64=base64.b64encode(raw).decode("ascii"))

    nb = copy.deepcopy(json.loads(VEHICLE_NB.read_text(encoding="utf-8")))
    n_before = len(nb["cells"])

    # The insertion point must sit AFTER setup_commands and BEFORE the first
    # `inference` import, exactly as P1 and exec-WM assert it.
    setup_idx = pkl_idx = None
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"])
        if '_run_shell_commands("setup_commands.json"' in s and setup_idx is None:
            setup_idx = i
        if "deploy_target.pkl" in s and pkl_idx is None:
            pkl_idx = i
    assert setup_idx is not None, "vehicle drifted: setup cell not found"
    assert setup_idx < PATCH_CELL_INDEX, f"patch cell {PATCH_CELL_INDEX} precedes setup {setup_idx}"
    if pkl_idx is not None:
        assert PATCH_CELL_INDEX <= pkl_idx, f"patch cell {PATCH_CELL_INDEX} follows pkl load {pkl_idx}"

    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_source.splitlines(keepends=True),
    }
    nb["cells"].insert(PATCH_CELL_INDEX, cell)

    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    # Required markers.
    for good in ("[p2] reset semantics OK", 'assert "inference" not in sys.modules',
                 "p2_patch_embedded", "anchors_applied"):
        assert good in code, "missing required marker in the notebook: %s" % good
    # These live in the embedded module, which the notebook carries base64 -- so
    # assert them against the module bytes that were just hashed into the cell.
    for good in ("H_STUCK_TURNS = 4", "episodes_available", "`attempt(actions)`",
                 "[p2] reset-retry armed", "_p2_flush"):
        assert good in src, "missing required marker in p2_patch.py: %s" % good

    # Sibling arms' markers are FORBIDDEN, so a compound can never be misread as
    # a single-variable arm (prereg S6 / exp 34's standing rule).
    for bad in ("[cadence]", "[notes]", "[execwm]", "reasoning_effort",
                "LOCAL_ANALYZER_MAX_OUTPUT", "taaf_grafts"):
        assert bad not in code, "forbidden sibling marker in the notebook: %s" % bad
        assert bad not in src, "forbidden sibling marker in p2_patch.py: %s" % bad

    meta = json.loads(VEHICLE_META.read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta["id"] = KERNEL_ID
    meta["title"] = "arc3-p2-retry-eval"
    meta["code_file"] = OUT_NB.name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                  encoding="utf-8")

    code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    print("built %s" % OUT_NB)
    print("  cells: %d (vehicle %d + patch at index %d)" % (len(nb["cells"]), n_before,
                                                            PATCH_CELL_INDEX))
    print("  patch-module sha: %s" % sha)
    print("  notebook code sha: %s" % code_sha)
    print("  markers present; sibling markers absent; insertion point asserted")
    return OUT_NB


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
