"""ARM 3 builder: Q38xGRAFT compound (`canivel/arc3-q38-graft-eval`).

PREREG: learnings/war_room/q38_graft_prereg_2026-08-21.md (sealed BEFORE this builder ran).
Vehicle: our layout-tolerant graft-floor v4 lineage + thtennant v21's deltas lifted VERBATIM
from the sha-pinned pull (engine block cell 6, setup patch cell 8, flags cell 12). v21's
4-game dup-gate cell 14 is EXCLUDED (we keep our 25-game path-resolved run cell).
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_graft_eval as bg  # our v4 lineage: MOUNTCHECK v2, cell4/14 path fixes

V21_NB = Path(r"C:/Users/dcani/AppData/Local/Temp/claude/f--kaggle/62c35e7c-0d05-4da2-99b0-f9b400a45a97/scratchpad/rethink0820/tennant_v21/arc3-duck-v21.ipynb")
V21_SHA = "71f0b1a8e1e5ab7b"
OUT_DIR = Path(r"F:/kaggle/arc-prize-2026/notebooks/q38-graft-eval")
KERNEL_ID = "canivel/arc3-q38-graft-eval"
MODEL_SOURCE = "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
FLAGS_LINE_V21 = ('    install(bm, flags={"efficiency": True, "retry_guard": True, '
                  '"shortcircuit": True, "goalkeep": True, "hudmask": True, "clickmap": True})')
FLAGS_LINE_OURS = ('    install(bm, flags={"efficiency": True, "retry_guard": True, '
                   '"shortcircuit": True, "goalkeep": True, "hudmask": True, "clickmap": True}, '
                   'expected_version=1)')


def src_of(c):
    return "".join(c["source"])


def build():
    # --- pin the v21 vehicle
    v21 = json.loads(V21_NB.read_text(encoding="utf-8"))
    v21_code = "".join(src_of(c) for c in v21["cells"] if c["cell_type"] == "code")
    got = hashlib.sha256(v21_code.encode()).hexdigest()[:16]
    assert got == V21_SHA, f"v21 pull drifted: {got} != {V21_SHA}"

    fork = json.loads(bg.SRC_NB.read_text(encoding="utf-8"))
    pristine = json.loads(bg.SRC_NB.read_text(encoding="utf-8"))["cells"]
    cells = fork["cells"]
    assert len(cells) == 17

    # --- cell 2: our v4 banner+MOUNTCHECK, re-identified for this arm
    c2 = src_of(cells[2]).replace(bg.CELL2_ANCHOR, bg.CELL2_NEW)
    reps = [
        ("mode=graft-floor-local25", "mode=q38graft-local25"),
        ('flags=efficiency+retry_guard+shortcircuit+goalkeep+hudmask ',
         "flags=efficiency+retry_guard+shortcircuit+goalkeep+hudmask+clickmap "),
        ("FORBIDDEN=banking+transfer ", "FORBIDDEN=banking+transfer+searchmap "),
        ("engine=driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot (UNCHANGED, vrfai/Qwen3.6-27B-FP8) ",
         "engine=foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 KaggleModel (Qwen/Qwen3.8-27B-FP8, xhigh default) "),
        ("baseline=duck-harness-kaggle m=3 lc 18/19/21 ",
         "comparator=q38-field-floor n=1 lc 28 score 6.173 "),
        ("primary=mean_dlc HARM<=-0.286320 SIGNAL>=+0.286320 ",
         "primary=lc_total HARM<=23 NULL 24-32 SIGNAL>=33 (1-sigma screen, diff-SD 5.011) "),
        ("audited_bundle_sha=df447f61caa181cca68049e28b139e02",
         "audited_bundle_sha=dde323ab4f8663c1523135301c596894"),
    ]
    for old, new in reps:
        assert old in c2, f"cell2 banner anchor missing: {old[:60]!r}"
        c2 = c2.replace(old, new, 1)
    cells[2]["source"] = c2.splitlines(keepends=True)

    # --- cell 4: our v4 wheels-path fix
    c4 = src_of(cells[4])
    assert c4.count(bg.CELL4_ANCHOR) == 1
    cells[4]["source"] = c4.replace(bg.CELL4_ANCHOR, bg.CELL4_NEW).splitlines(keepends=True)

    # --- cells 6 + 8: LIFT v21 VERBATIM (engine block + setup patch)
    cells[6]["source"] = list(v21["cells"][6]["source"])
    cells[8]["source"] = list(v21["cells"][8]["source"])

    # --- cell 12: v21 flags + our expected_version pin
    c12 = src_of(v21["cells"][12])
    assert c12.count(FLAGS_LINE_V21) == 1, "v21 flags line drifted"
    cells[12]["source"] = c12.replace(FLAGS_LINE_V21, FLAGS_LINE_OURS).splitlines(keepends=True)

    # --- cell 14: our v4 path-resolved 25-game run cell (NOT v21's dup-gate cell)
    c14 = src_of(cells[14])
    assert c14.count(bg.CELL14_ANCHOR) == 1
    cells[14]["source"] = c14.replace(bg.CELL14_ANCHOR, bg.CELL14_NEW).splitlines(keepends=True)

    # --- GATES
    changed = [i for i, (a, b) in enumerate(zip(pristine, cells)) if src_of(a) != src_of(b)]
    assert changed == [2, 4, 6, 8, 12, 14], f"diff cells {changed}"
    code = "".join(src_of(c) for c in cells if c["cell_type"] == "code")
    for bad in ('"banking": True', '"transfer": True', '"searchmap": True',
                "external_game_id", "bm.games[:3]"):
        assert bad not in code, f"forbidden token {bad!r}"
    for good in ('"clickmap": True', "expected_version=1", "GRAFT_COMP_ROOT",
                 "QWEN_SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B-FP8",
                 "taaf_grafts.composite import install"):
        assert good in code, f"required token {good!r} missing"
    assert "reasoning_effort" not in code

    ds_lines = [ln for ln in src_of(cells[6]).splitlines()
                if ln.strip().startswith("DATASET_SOURCES = ")]
    assert len(ds_lines) == 1
    assert ds_lines[0].strip().startswith('DATASET_SOURCES = ["thtennant/taaf-kaggle-source-share-fork"'), \
        "graft fork must be index 0 (BUNDLE_DIR mapping)"

    # --- metadata
    meta = json.loads(bg.SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = "arc3-q38-graft-eval"
    meta["code_file"] = "arc3-q38-graft-eval.ipynb"
    meta["dataset_sources"] = ["thtennant/taaf-kaggle-source-share-fork",
                               "driessmit1/arc3-vllm-h100-wheelhouse-v3"]
    meta["model_sources"] = [MODEL_SOURCE]
    ref = json.loads(bg.SRC_META.read_text(encoding="utf-8"))
    for key in ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
                "competition_sources", "kernel_sources", "language", "kernel_type",
                "is_private", "keywords"):
        assert meta[key] == ref[key], key

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "arc3-q38-graft-eval.ipynb").write_text(json.dumps(fork, indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sha = hashlib.sha256(code.encode()).hexdigest()[:16]
    print(f"built {OUT_DIR}  cells=17 code_sha256={sha}")
    print("flags ON        : efficiency retry_guard shortcircuit goalkeep hudmask clickmap")
    print("flags FORBIDDEN : banking transfer searchmap")
    print(f"engine          : {MODEL_SOURCE} (Kaggle Model, served Qwen/Qwen3.8-27B-FP8, xhigh default)")
    print(f"differing cells vs frozen fork: {changed}")
    print(f"v21 vehicle pinned: {V21_SHA}; dup-gate EXCLUDED; 25-game run cell = ours (path-resolved)")


if __name__ == "__main__":
    build()
