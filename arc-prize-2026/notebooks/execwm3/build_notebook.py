"""Build arc3-execwm-v3.ipynb from baseline's working notebook by ONLY
swapping the agent code (cell 1 = %%writefile my_agent.py).

CRITICAL: do NOT regenerate the rerun cell, the metadata.kaggle block,
the agents/__init__.py write, or the .env write. Those drifted in
v62-v65 and killed all 5 submissions. We start from the known-good
baseline notebook and surgically replace ONLY the agent file content.

Sims are added as a second dataSources entry pointing to canivel/exec-wm-sims.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "arc3-execwm-v3.ipynb"  # baseline-copied
OUT = TEMPLATE  # in-place


def main():
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    # Replace cell 1 (writefile my_agent.py) with v65 agent code
    agent_code = (ROOT / "notebooks/forge_agent/v65_agent.py").read_text(encoding="utf-8")
    new_cell1_source = f"%%writefile /kaggle/working/my_agent.py\n{agent_code}"
    nb["cells"][1]["source"] = new_cell1_source.splitlines(keepends=True)
    # Clear stale outputs
    if "outputs" in nb["cells"][1]:
        nb["cells"][1]["outputs"] = []
    nb["cells"][1]["execution_count"] = None

    # Add exec-wm-sims to dataSources (alongside the competition entry).
    # Kaggle CLI auto-syncs `dataset_sources` from kernel-metadata.json
    # into metadata.kaggle.dataSources on push, but we explicitly insert
    # so the structure matches baseline pattern at write time.
    meta = nb.setdefault("metadata", {}).setdefault("kaggle", {})
    ds = meta.setdefault("dataSources", [])
    if not any(d.get("sourceType") == "datasetVersion"
               and d.get("ownerName") == "canivel"
               and d.get("datasetName") == "exec-wm-sims" for d in ds):
        ds.append({
            "sourceType": "datasetVersion",
            "ownerName": "canivel",
            "datasetName": "exec-wm-sims",
            "type": "user",
        })

    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  cells: {len(nb['cells'])}")
    print(f"  nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
    print(f"  metadata.kaggle.dataSources: {len(nb['metadata']['kaggle']['dataSources'])} entries")


if __name__ == "__main__":
    main()
