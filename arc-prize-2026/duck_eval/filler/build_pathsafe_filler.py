"""Builder for the FILLER-SAFE pathsafe fork (authorized 2026-08-20, coordinator ruling 2).

Frozen fork + COMP_ROOT mount-layout resolution ONLY. No grafts, no flags, no engine change,
no dataset change. Insurance against the rerun pool following the batch pool's 08-18 layout
migration (competitions/ intermediate dir removed). Diff vs frozen fork MUST be exactly
cells [2, 4, 14]. Staged unpushed; push timing is the coordinator's.
"""
import ast, hashlib, json
from pathlib import Path

SRC_NB = Path(r"F:/kaggle/arc-prize-2026/notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb")
SRC_META = Path(r"F:/kaggle/arc-prize-2026/notebooks/duckfork/kernel-metadata.json")
OUT_DIR = Path(r"F:/kaggle/arc-prize-2026/notebooks/duckfiller-pathsafe")
OUT_NB = OUT_DIR / "arc3-duck-repro-pathsafe.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"
KERNEL_ID = "canivel/arc3-duck-repro-pathsafe"

CELL2_ANCHOR = 'print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")'
NL = chr(10)
RESOLVER = NL.join([
    CELL2_ANCHOR,
    "",
    "# FILLER-PATHSAFE MOUNTCHECK (2026-08-20). On 08-18 Kaggle's BATCH env moved the",
    "# competition input from /kaggle/input/competitions/<comp> to /kaggle/input/<comp>",
    "# (proven by the graft lane's labelled v3 death; three eval kernels died on the old",
    "# hardcoded path). The RERUN env still runs the old layout (filler scored through it",
    "# nightly incl. 08-20), but pools migrate in waves: resolve the root once, fail LOUD",
    "# if neither candidate carries the wheels. Bootstrap path resolution only.",
    "import os as _os",
    "_COMP_CANDIDATES = [",
    '    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3",',
    '    "/kaggle/input/arc-prize-2026-arc-agi-3",',
    "]",
    "try:",
    '    _tree = sorted(_os.listdir("/kaggle/input"))',
    "except Exception as _exc:",
    '    _tree = ["<unlistable: " + repr(_exc)[:120] + ">"]',
    'print("FILLER-PATHSAFE MOUNTCHECK /kaggle/input =", _tree, flush=True)',
    "COMP_ROOT = next(",
    '    (c for c in _COMP_CANDIDATES if _os.path.isdir(c + "/arc_agi_3_wheels")), None)',
    "if COMP_ROOT is None:",
    "    raise RuntimeError(",
    '        "FILLER-PATHSAFE INFRA DEATH: competition wheels absent under BOTH mount layouts "',
    '        + repr(_COMP_CANDIDATES) + " (/kaggle/input = " + repr(_tree) + "). Kaggle "',
    '        "input-mounting failure, NOT an agent result."',
    "    )",
    'print("FILLER-PATHSAFE MOUNTCHECK OK competition root =", COMP_ROOT, flush=True)',
])
CELL4_ANCHOR = '        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",'
CELL4_NEW = '        COMP_ROOT + "/arc_agi_3_wheels",  # resolved in cell 2 (FILLER-PATHSAFE): layout moved 2026-08-18'
CELL14_ANCHOR = '    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")'
CELL14_NEW = '    competition_env_files = str(Path(COMP_ROOT) / "environment_files")  # resolved in cell 2 (FILLER-PATHSAFE)'


def src_of(c): return "".join(c["source"])


def replace_once(text, old, new, where):
    assert text.count(old) == 1, f"{where}: anchor count {text.count(old)} != 1"
    return text.replace(old, new)


nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
pristine = json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"]
cells = nb["cells"]
assert len(cells) == 17, f"fork drift: {len(cells)} cells"

cells[2]["source"] = replace_once(src_of(cells[2]), CELL2_ANCHOR, RESOLVER, "cell 2").splitlines(keepends=True)
cells[4]["source"] = replace_once(src_of(cells[4]), CELL4_ANCHOR, CELL4_NEW, "cell 4").splitlines(keepends=True)
cells[14]["source"] = replace_once(src_of(cells[14]), CELL14_ANCHOR, CELL14_NEW, "cell 14").splitlines(keepends=True)

# GATES
changed = [i for i, (a, b) in enumerate(zip(pristine, cells)) if src_of(a) != src_of(b)]
assert changed == [2, 4, 14], f"diff cells {changed}, expected [2, 4, 14]"
code = "".join(src_of(c) for c in cells if c["cell_type"] == "code")
for forbidden in ("taaf_grafts", "install(bm", "banking", "transfer_solver", "goalkeep",
                  "hudmask", "shortcircuit", "thtennant"):
    assert forbidden not in code, f"forbidden token {forbidden!r} in filler artifact"
assert code.count("COMP_ROOT") >= 4
assert 'os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")' in code, "gateway branch lost"
for i in (2, 4, 14):
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    compile(src_of(cells[i]), f"cell{i}", "exec", flags=flags)  # syntax gate

meta = json.loads(SRC_META.read_text(encoding="utf-8"))
ref = json.loads(SRC_META.read_text(encoding="utf-8"))
meta["id"] = KERNEL_ID
meta["title"] = "arc3-duck-repro-pathsafe"
meta["code_file"] = OUT_NB.name
for key in ("enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image",
            "competition_sources", "dataset_sources", "kernel_sources", "model_sources",
            "language", "kernel_type", "is_private", "keywords"):
    assert meta[key] == ref[key], f"env drift on {key}"

OUT_NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
sha = hashlib.sha256(code.encode()).hexdigest()[:16]
print(f"built {OUT_NB}")
print(f"cells=17 code_sha256={sha} diff=[2,4,14] datasets UNCHANGED (stock bundle) engine UNCHANGED")
print("gates: anchors unique, forbidden-token scan clean, gateway branch intact, syntax OK, env byte-identical")
