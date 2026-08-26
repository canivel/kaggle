"""ARM 1 builder: budget-elasticity variants of the certified field-floor vehicle.

PREREG: learnings/war_room/budget_elasticity_prereg_2026-08-22.md (sealed first).
ONE literal changes: bm.solver.max_runtime_s_per_game = 7920.0 -> the arm's value.
Everything else BYTE-IDENTICAL to the FOYSAL vehicle (code sha 7227f3286cf60b25).

Usage: python duck_eval/budget/build_budget_eval.py --arm t05|t3
"""
import argparse
import hashlib
import json
from pathlib import Path

FOYSAL_NB = Path(r"C:/Users/dcani/AppData/Local/Temp/claude/f--kaggle/62c35e7c-0d05-4da2-99b0-f9b400a45a97/scratchpad/rethink0820/foysal_lb9/lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb")
FOYSAL_META = Path(r"C:/Users/dcani/AppData/Local/Temp/claude/f--kaggle/62c35e7c-0d05-4da2-99b0-f9b400a45a97/scratchpad/rethink0820/foysal_lb9/kernel-metadata.json")
BASE_SHA = "7227f3286cf60b25"
ANCHOR = "bm.solver.max_runtime_s_per_game = 7920.0"
# OUT_ROOT is module-level so local_gate N4 can redirect it (OUT_* convention).
OUT_ROOT = Path(r"F:/kaggle/arc-prize-2026/notebooks")

ARMS = {
    "t05": {"value": "3960.0", "slug": "arc3-budget-t05-eval", "title": "arc3-budget-t05-eval"},
    "t3":  {"value": "23760.0", "slug": "arc3-budget-t3-eval",  "title": "arc3-budget-t3-eval"},
}


def code_of(nb):
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def build(arm: str):
    spec = ARMS[arm]
    nb = json.loads(FOYSAL_NB.read_text(encoding="utf-8"))
    base_code = code_of(nb)
    got = hashlib.sha256(base_code.encode()).hexdigest()[:16]
    assert got == BASE_SHA, f"FOYSAL vehicle drifted: {got}"
    assert base_code.count(ANCHOR) == 1, f"anchor count {base_code.count(ANCHOR)}"

    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if ANCHOR in src:
            src = src.replace(ANCHOR, f"bm.solver.max_runtime_s_per_game = {spec['value']}")
            c["source"] = src.splitlines(keepends=True)
            hits += 1
    assert hits == 1

    new_code = code_of(nb)
    assert f"max_runtime_s_per_game = {spec['value']}" in new_code
    assert "7920.0" not in new_code, "old constant survives"
    for other, o in ARMS.items():
        if other != arm:
            assert f"= {o['value']}" not in new_code, f"sibling constant {o['value']} present"
    for bad in ("taaf_grafts", "install(bm", "reasoning_effort", "EDGE1", "EDGE2"):
        assert bad not in new_code, bad

    meta = json.loads(FOYSAL_META.read_text(encoding="utf-8"))
    ref = json.loads(FOYSAL_META.read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta["id"] = f"canivel/{spec['slug']}"
    meta["title"] = spec["title"]
    meta["code_file"] = f"{spec['slug']}.ipynb"
    meta["is_private"] = True
    for key in ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
                "competition_sources", "dataset_sources", "kernel_sources", "model_sources",
                "language", "kernel_type", "keywords"):
        assert meta[key] == ref[key], key

    out = OUT_ROOT / f"budget-{arm}-eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{spec['slug']}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sha = hashlib.sha256(new_code.encode()).hexdigest()[:16]
    print(f"built {out}  arm={arm} max_runtime_s_per_game={spec['value']} code_sha256={sha}")
    print(f"base verified {BASE_SHA}; ONE literal changed; sibling constants absent; env byte-identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    build(ap.parse_args().arm)
