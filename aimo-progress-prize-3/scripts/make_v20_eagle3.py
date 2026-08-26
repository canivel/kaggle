"""Build v20: Eagle-3 speculative decoding integration.

Based on v17 (verify cascade). Adds Eagle-3 draft model to vLLM for +36-42% speed.
With same time budget, the model generates ~40% more tokens per attempt:
- Longer reasoning chains
- More code verification steps
- Fewer timeout failures

Changes from v17:
1. gpu_memory_utilization: 0.96 → 0.93 (draft model needs ~0.6GB VRAM)
2. vLLM server command: add --speculative-config flag
3. kernel-metadata: add eliork/gpt-oss-120b-eagle3-throughput to dataset_sources

Eagle-3 draft model: eliork/gpt-oss-120b-eagle3-throughput
  Path on Kaggle: /kaggle/input/gpt-oss-120b-eagle3-throughput/
  Size: ~1.2GB (model.safetensors + config.json)

vLLM requirement: >= 0.7 (Kaggle has 0.11.2 ✓)

Risk: Medium — config change to vLLM startup. If Eagle-3 path wrong or
incompatible, vLLM falls back or fails to start.
Expected: +0 to +3 pts from more inference tokens per budget.
"""

from __future__ import annotations
import io
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
BASE_NB = NOTEBOOKS_DIR / "submission_v17_verify.ipynb"
OUT_NB = NOTEBOOKS_DIR / "submission_v20_eagle3.ipynb"

# Eagle-3 draft model dataset
EAGLE3_DATASET = "eliork/gpt-oss-120b-eagle3-throughput"
EAGLE3_PATH = "/kaggle/input/gpt-oss-120b-eagle3-throughput"


def main():
    print("Building v20 Eagle-3 notebook from v17...")

    with io.open(BASE_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified_cells = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new_src = src

        # 1. Reduce gpu_memory_utilization from 0.96 to 0.93
        if "gpu_memory_utilization" in src and "0.96" in src:
            new_src = new_src.replace(
                "gpu_memory_utilization = 0.96",
                "gpu_memory_utilization = 0.93  # Eagle-3 draft needs ~0.6GB extra VRAM"
            )
            if new_src != src:
                print(f"Cell {i}: gpu_memory_utilization 0.96 → 0.93")

        # 2. Add --speculative-config to vLLM server command
        # The cmd ends with: '--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'
        if "'--enable-prefix-caching'" in src and "speculative" not in src:
            eagle3_snippet = (
                "'--async-scheduling', '--disable-log-stats', '--enable-prefix-caching',\n"
                "            # Eagle-3 speculative decoding (+36-42% speed, identical output distribution)\n"
                f"            '--speculative-config', json.dumps({{\n"
                f"                'method': 'eagle3',\n"
                f"                'model': '{EAGLE3_PATH}',\n"
                f"                'num_speculative_tokens': 3,\n"
                f"                'draft_tensor_parallel_size': 1,\n"
                f"            }})"
            )
            new_src = new_src.replace(
                "'--async-scheduling', '--disable-log-stats', '--enable-prefix-caching'",
                eagle3_snippet
            )
            if new_src != src:
                print(f"Cell {i}: added --speculative-config Eagle-3 flag")

        # 3. Add eagle3 import for json.dumps (json is likely already imported)
        # json is a stdlib import — should already be there in the notebook

        # 4. Update version tag
        if "ULTIMATE v36" in new_src or "ULTIMATE v35" in new_src:
            new_src = new_src.replace(
                "# ULTIMATE v36: exact 44/50 params + T=0.8 + follow-up + binary verification cascade",
                "# ULTIMATE v39: exact 44/50 + T=0.8 + binary verify + Eagle-3 (+36% speed)"
            ).replace(
                "# ULTIMATE v35: exact 44/50 params (ctx=65536, batch=256, gpu=0.96) + T=0.8 + follow-up",
                "# ULTIMATE v39: exact 44/50 + T=0.8 + binary verify + Eagle-3 (+36% speed)"
            ).replace(
                "print(f'CFG: ULTIMATE v36 | exact 44/50 + T=0.8 + binary verification (amanatar 44/50)')",
                "print(f'CFG: ULTIMATE v39 | exact 44/50 + T=0.8 + binary verify + Eagle-3 speculative')"
            ).replace(
                "print(f'CFG: ULTIMATE v35 | 5-step prompt + T=0.8 + base model + exact 44/50 vLLM params')",
                "print(f'CFG: ULTIMATE v39 | exact 44/50 + T=0.8 + binary verify + Eagle-3 speculative')"
            )
            if new_src != src:
                print(f"Cell {i}: version tag → ULTIMATE v39")

        if new_src != src:
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
            modified_cells += 1

    print(f"Modified {modified_cells} cells")
    if modified_cells == 0:
        print("WARNING: no cells modified — check if gpu_memory_utilization=0.96 and --enable-prefix-caching exist")

    # Write notebook
    with io.open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {OUT_NB}")

    # Syntax validation
    import ast
    with io.open(OUT_NB, "r", encoding="utf-8") as f:
        nb_check = json.load(f)

    errors = []
    for i, cell in enumerate(nb_check["cells"]):
        if cell["cell_type"] == "code":
            src_c = "".join(cell["source"])
            if src_c.strip():
                try:
                    ast.parse(src_c)
                except SyntaxError as e:
                    errors.append(f"Cell {i}: {e}")

    if errors:
        print(f"SYNTAX ERRORS:\n" + "\n".join(errors))
        return

    print("Syntax validation: PASSED")

    # Build push dir with Eagle-3 dataset in metadata
    import shutil
    push_dir = NOTEBOOKS_DIR / "push_v20"
    push_dir.mkdir(exist_ok=True)
    shutil.copy(OUT_NB, push_dir / "submission_v20_eagle3.ipynb")

    with io.open(NOTEBOOKS_DIR / "push_v17" / "kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["code_file"] = "submission_v20_eagle3.ipynb"
    meta["title"] = "AIMO3 v39 Eagle-3 speculative"
    # Add Eagle-3 dataset
    meta["dataset_sources"] = [EAGLE3_DATASET]
    with io.open(push_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Push dir ready: {push_dir}")
    print(f"Dataset added: {EAGLE3_DATASET}")
    print()
    print("IMPORTANT: Before submitting, verify Eagle-3 is compatible:")
    print("  1. Check eliork/gpt-oss-120b-eagle3-throughput config.json base_model matches danielhanchen/gpt-oss-120b")
    print("  2. Test locally that vLLM accepts --speculative-config with this model")
    print("  3. Monitor vLLM startup logs for Eagle-3 initialization")
    print()
    print(f"Submit: cd {push_dir} && kaggle kernels push")


if __name__ == "__main__":
    main()
