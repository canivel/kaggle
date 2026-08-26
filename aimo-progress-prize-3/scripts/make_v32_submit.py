"""
Build v32 — the ONE clean submission notebook.

Source: v27 (strategy diversity, plain 1/entropy voting)
 - Proven 10/10 on reference benchmark
 - Scored 40/50 in competition
 - NO EV voting (validated: hurts)
 - Fresh slug: aimo3-v32-submit

Run: python scripts/make_v32_submit.py
Then: cd notebooks/push_v32 && kaggle kernels push
"""

import json, io, shutil, pathlib

ROOT      = pathlib.Path(__file__).parent.parent
SRC_NB    = ROOT / "notebooks/submission_v27_diverse.ipynb"
OUT_DIR   = ROOT / "notebooks/push_v32"
OUT_NB    = OUT_DIR / "submission_v32.ipynb"
META      = OUT_DIR / "kernel-metadata.json"

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Copy v27 notebook verbatim — proven code, no changes
    shutil.copy(SRC_NB, OUT_NB)
    print(f"Copied: {SRC_NB.name} -> {OUT_NB}")

    # Fresh slug — never used before
    meta = {
        "id": "canivel/aimo3-v32-submit",
        "title": "AIMO3 v32 submit",
        "code_file": "submission_v32.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": ["ai-mathematical-olympiad-progress-prize-3"],
        "model_sources": ["danielhanchen/gpt-oss-120b/Transformers/default/1"],
        "dataset_sources": [],
        "kernel_sources": ["andreasbis/aimo-3-utils"],
        "keywords": [],
        "machine_shape": "NvidiaH100",
    }
    with io.open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Slug: aimo3-v32-submit")
    print(f"\nTo push:")
    print(f"  cd {OUT_DIR}")
    print(f"  kaggle kernels push")

if __name__ == "__main__":
    main()
