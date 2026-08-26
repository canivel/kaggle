# ===== CELL 0 [code] =====
from pathlib import Path
from IPython.display import Image, display

cover_image_path = Path("/kaggle/input/datasets/pilkwang/pilkwang-public-dataset-for-notebooks-figures/biohub_cover.png")
if cover_image_path.exists():
    display(Image(filename=str(cover_image_path)))
# ===== CELL 1 [markdown] =====
## Full-score inference and post-submission visualization

The score-producing path and the teaching path are intentionally independent.

- A normal Kaggle **Batch** commit runs the complete test set exactly like the original 0.897 baseline.
- After `submission.csv` is finished, the notebook generates heatmaps, 3D viewers and animations from already-created predictions.
- Visualization never changes `BIOHUB_SLICE` during a Batch run.
- A SHA-256 integrity guard verifies that `submission.csv` is unchanged after all visual modules finish.
- An official competition rerun can still skip visualization to minimize timeout risk.

Useful overrides:

```text
BIOHUB_ENABLE_POST_SUBMISSION_VISUALS=1   # default for normal Batch/interactive runs
BIOHUB_DISABLE_VISUALIZATION=1            # disable all new visuals
BIOHUB_FORCE_SUBMISSION_MODE=1            # full test set
BIOHUB_VISUAL_DEMO_MODE=1                 # interactive small-slice teaching run
BIOHUB_VISUAL_SLICE=:3                    # teaching subset only
```

# ===== CELL 2 [code] =====
import os

def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

_KERNEL_RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip().lower()
_IS_BATCH_RUN = _KERNEL_RUN_TYPE == "batch"
_IS_COMPETITION_RERUN = _flag("KAGGLE_IS_COMPETITION_RERUN", False)
_FORCE_FULL_RUN = _flag("BIOHUB_FORCE_SUBMISSION_MODE", False)
_FORCE_VISUAL_DEMO = _flag("BIOHUB_VISUAL_DEMO_MODE", False)
_DISABLE_VISUALS = _flag("BIOHUB_DISABLE_VISUALIZATION", False)
_ENABLE_POST_VISUALS = _flag("BIOHUB_ENABLE_POST_SUBMISSION_VISUALS", True)

# Full-data selection and visualization are separate decisions.
# A normal Kaggle Batch run is full-data AND may still produce post-submission visuals.
IS_SUBMISSION_MODE = _FORCE_FULL_RUN or _IS_BATCH_RUN or _IS_COMPETITION_RERUN
VISUAL_DEMO_MODE = _FORCE_VISUAL_DEMO and not _IS_BATCH_RUN and not _IS_COMPETITION_RERUN

# Official competition reruns skip extra rendering by default.
# Normal Batch commits keep the visual output, but only after submission.csv exists.
RUN_PIPELINE_VISUALIZATION = (
    _ENABLE_POST_VISUALS
    and not _DISABLE_VISUALS
    and not _IS_COMPETITION_RERUN
)

# Only a deliberately requested interactive teaching run may apply a small slice.
# Normal Batch runs never touch BIOHUB_SLICE.
if VISUAL_DEMO_MODE and not os.environ.get("BIOHUB_SLICE", "").strip():
    os.environ["BIOHUB_SLICE"] = os.environ.get("BIOHUB_VISUAL_SLICE", ":3")

print("Data mode:", "FULL TEST SET" if IS_SUBMISSION_MODE else "INTERACTIVE CONFIGURATION")
print("Post-submission visualization:", "ON" if RUN_PIPELINE_VISUALIZATION else "OFF")
print("KAGGLE_KERNEL_RUN_TYPE:", _KERNEL_RUN_TYPE or "<unset>")
print("BIOHUB_SLICE:", os.environ.get("BIOHUB_SLICE", "<full>"))

# ===== CELL 3 [markdown] =====
# Biohub Cell Tracking: 0.897 Full-Score Pipeline + Interactive Visual Laboratory
Learned detections with motion relinking, conservative division recovery, learned-edge assignment bonus, and topology-preserving line-fit coordinate smoothing.

# ===== CELL 4 [markdown] =====
## Modeling Contract

For each time point, the volume is converted into a detection field.

$$
p_t(\mathbf r) = \sigma\left(h_\theta(V_{t:t+1})(\mathbf r)\right)
$$

Candidate nodes are local maxima above a fixed probability threshold.

$$
\mathcal D_t = \left\{\mathbf r \mid p_t(\mathbf r) > \tau
\quad\text{and}\quad
p_t(\mathbf r) = \max_{\mathbf u \in \mathcal N(\mathbf r)} p_t(\mathbf u)
\right\}
$$

For consecutive frames, candidate temporal edges are scored from image features,
node features, and relative geometry.

$$
s_{ij} = g_\phi\left(f_i, f_j, \mathbf r_i - \mathbf r_j\right),
\qquad i \in \mathcal D_t,\ j \in \mathcal D_{t+1}
$$


An optional motion relinker can ignore the learned edge set and rebuild a
one-to-one temporal graph from the detected nodes. If node $i$ has a predecessor,
its predicted next position is

$$
\hat{\mathbf r}_{i,t+1} = \mathbf r_{i,t} + \lambda(\mathbf r_{i,t} - \mathbf r_{i,t-1}),
\qquad \lambda=0.5.
$$

A first Hungarian pass uses a tight physical gate of $6.0\,\mu m$; a second pass
uses a relaxed gate of $10.0\,\mu m$ for unmatched nodes. The assignment cost is
measured from $\hat{\mathbf r}_{i,t+1}$, while the gate is measured from raw
frame-to-frame displacement.

The selected graph minimizes edge and event costs.

$$
\min_x \sum_{e \in E} w_e x_e
+ c_a A(x) + c_d D(x) + c_m M(x),
\qquad w_e = -\operatorname{edge\_prob}(e)
$$

The post-processor uses physical distance, not raw voxel distance:

$$
d_{\mu m}(i,j) =
\sqrt{(1.625\Delta z)^2 + (0.40625\Delta y)^2 + (0.40625\Delta x)^2}.
$$

A one-frame gap candidate connects an end node at frame $t$ to a start node at
frame $t+2$ only when

$$
d_{\mu m}(i,j) \le 2g,
\qquad g = 6.0\ \mu m.
$$

The geometric midpoint is

$$
\mathbf r_{t+1}^{new} = \frac{\mathbf r_t + \mathbf r_{t+2}}{2}.
$$

If an isolated detection already lies near this midpoint, that node is reused.
Otherwise, a synthetic node is inserted. For synthetic nodes, a local intensity
centroid is computed in the original frame:

$$
w(\mathbf r) = \max\left(I(\mathbf r) - P_{20}(I_W), 0\right),
\qquad
\hat{\mathbf r} =
\frac{\sum_{\mathbf r \in W} \mathbf r w(\mathbf r)}
{\sum_{\mathbf r \in W} w(\mathbf r)}.
$$

The refined coordinate is accepted only if it stays within a small physical
shift limit from the midpoint. This keeps gap recovery from turning into an
unbounded node generator.

Optional graph-repair passes are deliberately capped. Very short connected
components are removed when

$$
|C_k| < L_{\min},
\qquad L_{\min}=4,
$$

unless the component contains a division-like source and division components are
being preserved. A strict two-missing-frame bridge connects an end node at
frame $t$ to a start node at frame $t+3$ only when

$$
d_{\mu m}(i,j) \le 10.2,
\qquad \frac{d_{\mu m}(i,j)}{3} \le 4.4,
$$

and local predecessor/successor velocity context is not contradictory. This
creates exactly two interpolated nodes and three consecutive edges. Safe division
recovery is even more constrained: it adds only a second outgoing edge from a
node that already has one child, and only if the second child is close to the
parent and to the existing child:

$$
d_{\mu m}(p,c_2) \le 5.0,
\qquad d_{\mu m}(c_1,c_2) \le 8.0.
$$

Frame-level and graph-level caps keep both gap2 and division recovery from
becoming broad node or edge generators.


# ===== CELL 5 [code] =====
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import time
from pathlib import Path

import pandas as pd

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])
_test_dir_override = os.environ.get("BIOHUB_TEST_DIR", "").strip()
TEST_DIR = Path(_test_dir_override) if _test_dir_override else COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "candidate_11_motion_learned_linefit_smoothing"
TARGET_ARTIFACT_SLUG = os.environ.get("BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1")
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get(
    "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
))
ALLOW_ARTIFACT_FALLBACK = os.environ.get("BIOHUB_ALLOW_ARTIFACT_FALLBACK", "0") != "0"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.99"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))

# Empty for a real submission. Useful for local smoke tests, e.g. BIOHUB_SLICE=:1.
SLICE = os.environ.get("BIOHUB_SLICE", "").strip()

# If dependencies are not already installed and no offline wheels are attached,
# this controls whether the notebook attempts PyPI installation.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"
RUN_OUTPUT_DIAGNOSTICS = os.environ.get("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "1") != "0"
RUN_VISUAL_EDA = os.environ.get("BIOHUB_RUN_VISUAL_EDA", "1") != "0"

# Output-level graph post-processing.
OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0"
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0"
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0"
OUTPUT_PRUNE_ISOLATED = os.environ.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0"
OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"
MOTION_RELINK_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0"))
MOTION_RELINK_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0"))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600"))

OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_CLOSE_REUSE_EXISTING = os.environ.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0"
GAP_CLOSE_REUSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2"))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05"))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000"))
GAP_REFINE_SYNTHETIC = os.environ.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0"
GAP_REFINE_WIN_Z = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_Z", "1"))
GAP_REFINE_WIN_YX = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_YX", "3"))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2"))

OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6"))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"

OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))

OUTPUT_GAP2_RECOVERY = os.environ.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0"
GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2"))
GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4"))
GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045"))
GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180"))
GAP2_REQUIRE_CONTEXT = os.environ.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0"
GAP2_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006"))

OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))

CONFIG_DISPLAY = {
    "experiment_tag": EXPERIMENT_TAG,
    "method": METHOD,
    "weights": WEIGHTS_RELATIVE,
    "target_artifact_slug": TARGET_ARTIFACT_SLUG,
    "primary_artifact_manifest": str(PRIMARY_ARTIFACT_MANIFEST),
    "allow_artifact_fallback": ALLOW_ARTIFACT_FALLBACK,
    "det_threshold": DET_THRESHOLD,
    "unet_batch_size": UNET_BATCH_SIZE,
    "use_ilp": USE_ILP,
    "ilp_edge_weight": ILP_EDGE_WEIGHT,
    "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
    "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
    "ilp_division_weight": ILP_DIVISION_WEIGHT,
    "slice": SLICE,
    "allow_pip_install": ALLOW_PIP_INSTALL,
    "run_visual_eda": RUN_VISUAL_EDA,
    "output_edge_max_um": OUTPUT_EDGE_MAX_UM,
    "output_enforce_next_frame": OUTPUT_ENFORCE_NEXT_FRAME,
    "output_single_parent_repair": OUTPUT_SINGLE_PARENT_REPAIR,
    "output_single_child_repair": OUTPUT_SINGLE_CHILD_REPAIR,
    "output_prune_isolated": OUTPUT_PRUNE_ISOLATED,
    "output_motion_relink": OUTPUT_MOTION_RELINK,
    "motion_relink_tight_um": MOTION_RELINK_TIGHT_UM,
    "motion_relink_relaxed_um": MOTION_RELINK_RELAXED_UM,
    "motion_relink_velocity_weight": MOTION_RELINK_VELOCITY_WEIGHT,
    "motion_relink_learned_bonus": MOTION_RELINK_LEARNED_BONUS,
    "motion_relink_max_frame_nodes": MOTION_RELINK_MAX_FRAME_NODES,
    "output_division_geometry_filter": OUTPUT_DIVISION_GEOMETRY_FILTER,
    "div_parent_max_um": DIV_PARENT_MAX_UM,
    "div_sister_max_um": DIV_SISTER_MAX_UM,
    "div_drop_to_single_if_bad": DIV_DROP_TO_SINGLE_IF_BAD,
    "output_gap_close": OUTPUT_GAP_CLOSE,
    "gap_close_max_gap": GAP_CLOSE_MAX_GAP,
    "gap_close_effective_max_gap": min(GAP_CLOSE_MAX_GAP, 1),
    "gap_close_um": GAP_CLOSE_UM,
    "gap_close_reuse_existing": GAP_CLOSE_REUSE_EXISTING,
    "gap_close_reuse_um": GAP_CLOSE_REUSE_UM,
    "gap_close_max_added_frac": GAP_CLOSE_MAX_ADDED_FRAC,
    "gap_close_max_added_abs": GAP_CLOSE_MAX_ADDED_ABS,
    "gap_refine_synthetic": GAP_REFINE_SYNTHETIC,
    "gap_refine_win_z": GAP_REFINE_WIN_Z,
    "gap_refine_win_yx": GAP_REFINE_WIN_YX,
    "gap_refine_max_shift_um": GAP_REFINE_MAX_SHIFT_UM,
    "output_filter_short_tracks": OUTPUT_FILTER_SHORT_TRACKS,
    "output_min_track_len": OUTPUT_MIN_TRACK_LEN,
    "output_keep_division_components": OUTPUT_KEEP_DIVISION_COMPONENTS,
    "output_linefit_smooth": OUTPUT_LINEFIT_SMOOTH,
    "output_linefit_weight": OUTPUT_LINEFIT_WEIGHT,
    "output_linefit_window": OUTPUT_LINEFIT_WINDOW,
    "output_gap2_recovery": OUTPUT_GAP2_RECOVERY,
    "gap2_max_total_um": GAP2_MAX_TOTAL_UM,
    "gap2_max_step_um": GAP2_MAX_STEP_UM,
    "gap2_max_links_frac": GAP2_MAX_LINKS_FRAC,
    "gap2_max_links_abs": GAP2_MAX_LINKS_ABS,
    "gap2_require_context": GAP2_REQUIRE_CONTEXT,
    "gap2_frame_frac_cap": GAP2_FRAME_FRAC_CAP,
    "output_safe_divisions": OUTPUT_SAFE_DIVISIONS,
    "safe_div_max_um": SAFE_DIV_MAX_UM,
    "safe_div_sister_max_um": SAFE_DIV_SISTER_MAX_UM,
    "safe_div_existing_child_max_um": SAFE_DIV_EXISTING_CHILD_MAX_UM,
    "safe_div_frame_frac_cap": SAFE_DIV_FRAME_FRAC_CAP,
    "safe_div_global_frac_cap": SAFE_DIV_GLOBAL_FRAC_CAP,
}

print("Biohub learned UNet + node-transformer + ILP submission")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent=2, sort_keys=True))

# ===== CELL 6 [markdown] =====
## Data Geometry EDA

This section reads metadata only. It checks volume shapes, chunking, physical
scale, and available count estimates without loading the full hidden volumes.

The anisotropic voxel scale maps voxel coordinates into microns by

$$
\tilde{\mathbf r} = (1.625z,\;0.40625y,\;0.40625x).
$$

A physical radius $R$ is therefore an ellipsoid in voxel space:

$$
\left(\frac{\Delta z}{R/1.625}\right)^2 +
\left(\frac{\Delta y}{R/0.40625}\right)^2 +
\left(\frac{\Delta x}{R/0.40625}\right)^2 \le 1.
$$

This is the geometry used by both edge filtering and one-frame gap recovery.

# ===== CELL 7 [code] =====
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _scale_from_root_meta(root_meta: dict) -> tuple[float, float, float]:
    try:
        transforms = root_meta["attributes"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
        for transform in transforms:
            if transform.get("type") == "scale":
                scale = transform.get("scale", [])
                if len(scale) >= 4:
                    return tuple(float(v) for v in scale[-3:])
    except Exception:
        pass
    return (1.625, 0.40625, 0.40625)


def _quantiles_from_root_meta(root_meta: dict) -> dict:
    return root_meta.get("attributes", {}).get("image_statistics", {}).get("quantiles", {}) or {}


def _estimated_nodes_from_geff(geff_path: Path) -> int | None:
    meta = _read_json(geff_path / "zarr.json")
    value = (
        meta.get("attributes", {})
        .get("geff", {})
        .get("extra", {})
        .get("estimated_number_of_nodes")
    )
    return int(value) if value is not None else None


def _metadata_rows(split_name: str, split_dir: Path) -> list[dict[str, object]]:
    if not split_dir.exists():
        return []
    rows = []
    for zarr_path in sorted(split_dir.glob("*.zarr")):
        stem = zarr_path.name[:-5]
        arr_meta = _read_json(zarr_path / "0" / "zarr.json")
        root_meta = _read_json(zarr_path / "zarr.json")
        shape = arr_meta.get("shape", [None, None, None, None])
        chunks = arr_meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape", [])
        scale = _scale_from_root_meta(root_meta)
        quantiles = _quantiles_from_root_meta(root_meta)
        rows.append({
            "split": split_name,
            "dataset": stem,
            "embryo": stem.split("_")[0],
            "T": shape[0] if len(shape) > 0 else None,
            "Z": shape[1] if len(shape) > 1 else None,
            "Y": shape[2] if len(shape) > 2 else None,
            "X": shape[3] if len(shape) > 3 else None,
            "dtype": arr_meta.get("data_type"),
            "chunk_shape": tuple(chunks) if chunks else None,
            "scale_z_um": scale[0],
            "scale_y_um": scale[1],
            "scale_x_um": scale[2],
            "q001": quantiles.get("0.001"),
            "q999": quantiles.get("0.999"),
            "estimated_nodes": _estimated_nodes_from_geff(split_dir / f"{stem}.geff"),
        })
    return rows


metadata_rows = []
metadata_rows.extend(_metadata_rows("train", COMP_DIR / "train"))
metadata_rows.extend(_metadata_rows("test", TEST_DIR))
meta_df = pd.DataFrame(metadata_rows)

if len(meta_df):
    display(
        meta_df.groupby(["split", "T", "Z", "Y", "X", "dtype", "chunk_shape", "scale_z_um", "scale_y_um", "scale_x_um"], dropna=False)
        .size()
        .reset_index(name="videos")
    )

    split_summary = meta_df.groupby("split").agg(
        videos=("dataset", "count"),
        embryos=("embryo", "nunique"),
        estimated_nodes_median=("estimated_nodes", "median"),
        estimated_nodes_max=("estimated_nodes", "max"),
        q001_median=("q001", "median"),
        q999_median=("q999", "median"),
    ).reset_index()
    display(split_summary)

    gate_rows = []
    for radius_um in [2.65, 7.0, OUTPUT_EDGE_MAX_UM]:
        gate_rows.append({
            "radius_um": radius_um,
            "z_voxels": radius_um / 1.625,
            "y_voxels": radius_um / 0.40625,
            "x_voxels": radius_um / 0.40625,
        })
    display(pd.DataFrame(gate_rows).round(3))

    if meta_df["estimated_nodes"].notna().any():
        import matplotlib.pyplot as plt
        train_est = meta_df.loc[meta_df["estimated_nodes"].notna(), "estimated_nodes"].astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6), constrained_layout=True)
        axes[0].hist(train_est, bins=24)
        axes[0].set_title("Estimated cells per train video")
        axes[0].set_xlabel("estimated nodes")
        axes[0].set_ylabel("videos")
        axes[1].scatter(meta_df.index, meta_df["estimated_nodes"], s=20, alpha=0.65)
        axes[1].set_title("Estimated-node density by sample")
        axes[1].set_xlabel("sample index")
        axes[1].set_ylabel("estimated nodes")
        plt.show()
else:
    print("No Zarr metadata found. This is expected only outside the Kaggle data mount.")

# ===== CELL 8 [markdown] =====
## Artifact and Dependency Setup

The notebook expects a compact support artifact containing the inference source,
trained weights, and optionally offline dependency wheels. The primary Kaggle
attachment path is:

```text
/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json
```

The setup cell first uses already-installed modules, then attached wheels, and
only attempts an internet install when explicitly enabled for local development.
By default, the artifact resolver accepts only the 50-epoch package named by
`TARGET_ARTIFACT_SLUG`; set `BIOHUB_ALLOW_ARTIFACT_FALLBACK=1` only for local
debugging against an older package.

# ===== CELL 9 [code] =====
import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]

# The safe path for offline reruns is to use attached wheels.
# Set BIOHUB_ALLOW_PIP_INSTALL=1 only for an interactive internet-enabled run.
ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def candidate_roots_for_slug(slug: str) -> list[Path]:
    return [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
        Path(f"PublicNotebook/{slug}"),
    ]


def find_artifacts_root() -> Path:
    candidates: list[Path] = []
    for env_name in ["BIOHUB_MODEL_ARTIFACTS", "BIOHUB_ARTIFACTS"]:
        explicit = os.environ.get(env_name, "").strip()
        if explicit:
            candidates.append(Path(explicit))

    candidates.append(PRIMARY_ARTIFACT_MANIFEST.parent)
    candidates.extend(candidate_roots_for_slug(TARGET_ARTIFACT_SLUG))

    if ALLOW_ARTIFACT_FALLBACK:
        for slug in FALLBACK_ARTIFACT_SLUGS:
            candidates.extend(candidate_roots_for_slug(slug))

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            child_text = str(child)
            if TARGET_ARTIFACT_SLUG in child_text or ALLOW_ARTIFACT_FALLBACK:
                candidates.append(child)
                candidates.append(child / child.name)
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_model_artifact(candidate) and artifact_matches_target(candidate):
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(
        "Could not find the required model artifact. "
        f"Expected slug: {TARGET_ARTIFACT_SLUG}\n"
        "Attach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS.\n"
        "To debug with an older artifact, set BIOHUB_ALLOW_ARTIFACT_FALLBACK=1.\n"
        "Checked:\n" + checked
    )


def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates: list[Path] = [
        artifacts / "wheels",
        artifacts,
        Path("/kaggle/working"),
        Path("/kaggle/working/wheels"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if child.is_dir():
                candidates.extend([child / "wheels", child])
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.extend([grandchild / "wheels", grandchild])

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_package_file(candidate):
            out.append(candidate)
    return out


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)

# ===== CELL 10 [markdown] =====
## Visual EDA: One Frame Geometry

A single hidden-test frame is enough to verify orientation, anisotropy, dynamic
range, and chunk decoding. The plots use maximum projections only; they do not
load the full video into memory.

# ===== CELL 11 [code] =====
import numpy as np


def _read_one_frame_for_visual(zarr_path: Path, t: int) -> np.ndarray:
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        import blosc2 as _blosc2
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(_blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr
    return np.asarray(zarr.open(zarr_path / "0", mode="r")[t])


if RUN_VISUAL_EDA:
    try:
        import matplotlib.pyplot as plt

        visual_candidates = sorted(TEST_DIR.glob("*.zarr")) if TEST_DIR.exists() else []
        if not visual_candidates:
            print("Visual EDA skipped: no test volumes were found.")
        else:
            visual_path = visual_candidates[0]
            meta = json.loads((visual_path / "0" / "zarr.json").read_text())
            shape = tuple(int(v) for v in meta["shape"])
            t_vis = min(max(shape[0] // 2, 0), shape[0] - 1)
            frame = _read_one_frame_for_visual(visual_path, t_vis)
            clip_hi = float(np.percentile(frame, 99.7))
            clip_lo = float(np.percentile(frame, 1.0))
            z_proj = np.clip(frame.max(axis=0), clip_lo, clip_hi)
            y_proj = np.clip(frame.max(axis=1), clip_lo, clip_hi)
            x_proj = np.clip(frame.max(axis=2), clip_lo, clip_hi)

            fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
            axes[0, 0].imshow(z_proj, cmap="magma")
            axes[0, 0].set_title(f"Z projection: {visual_path.stem}, t={t_vis}")
            axes[0, 0].set_xlabel("x voxel")
            axes[0, 0].set_ylabel("y voxel")

            axes[0, 1].imshow(y_proj, cmap="magma", aspect="auto")
            axes[0, 1].set_title("Y projection")
            axes[0, 1].set_xlabel("x voxel")
            axes[0, 1].set_ylabel("z voxel")

            axes[1, 0].imshow(x_proj, cmap="magma", aspect="auto")
            axes[1, 0].set_title("X projection")
            axes[1, 0].set_xlabel("y voxel")
            axes[1, 0].set_ylabel("z voxel")

            sample = frame.ravel()[::max(1, frame.size // 250_000)]
            axes[1, 1].hist(sample, bins=80, log=True)
            axes[1, 1].axvline(clip_lo, color="black", linewidth=1, alpha=0.6)
            axes[1, 1].axvline(clip_hi, color="black", linewidth=1, alpha=0.6)
            axes[1, 1].set_title("Intensity distribution")
            axes[1, 1].set_xlabel("uint16 intensity")
            axes[1, 1].set_ylabel("voxels, log scale")
            plt.show()
    except Exception as exc:
        print(f"Visual EDA skipped safely: {type(exc).__name__}: {exc}")

# ===== CELL 12 [markdown] =====
## Predict Candidate Graphs

The inference step writes one `.geff` graph per test video. Keeping graph
prediction separate from CSV conversion makes the graph repair and diagnostics
transparent.

# ===== CELL 13 [code] =====
def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


test_stems = list_test_stems()
print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--weights",
    WEIGHTS_RELATIVE,
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])

start_time = time.time()
print(" ".join(predict_cmd))
subprocess.run(predict_cmd, cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
predict_seconds = time.time() - start_time
print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")

# ===== CELL 14 [markdown] =====
## Build `submission.csv`

Rows are streamed directly to disk with the required schema. This avoids holding
the full hidden-test submission table in memory.

The standard gap closer intentionally handles only one missing frame. Two-missing-frame repair is handled by the stricter `gap2` pass so that a loose environment override cannot introduce non-consecutive edges.

# ===== CELL 15 [code] =====
import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1



def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint


def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)
            stats["gap_candidates"] += int((d <= threshold_um).sum())
            if not np.isfinite(d).any():
                continue

            big = threshold_um * 1000.0 + 1.0
            cost = np.where(d <= threshold_um, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if d[r, c] > threshold_um:
                    continue
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            used_isolated.add(middle_id)
                            stats["gap_reused_existing"] += 1

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    keep: set[int] = set()
    for members in components.values():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    nodes_by_id, edges = close_single_frame_gaps(nodes_by_id, edges, stats, dataset=dataset)
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats)

    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id]

    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)

    return nodes_by_id, edges, stats


geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
print(f"Found {len(geffs)} prediction graphs")
if len(geffs) != len(test_stems):
    found = {path.stem for path in geffs}
    missing = sorted(set(test_stems) - found)
    raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

stats_rows: list[dict[str, object]] = []
seen_datasets: set[str] = set()
row_id = 0
total_nodes = 0
total_edges = 0

with SUBMISSION_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for geff_path in geffs:
        dataset = geff_path.stem
        seen_datasets.add(dataset)
        graph = graph_from_geff(geff_path)

        nodes_by_id: dict[int, dict[str, object]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }

        raw_edges: list[dict[str, object]] = []
        for row in graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })

        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        for node_id in sorted(nodes_by_id):
            node = nodes_by_id[node_id]
            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node["node_id"]),
                "t": int(node["t"]),
                "z": int(round(float(node["z"]))),
                "y": int(round(float(node["y"]))),
                "x": int(round(float(node["x"]))),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        division_sources: dict[int, int] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            target_id = int(edge["target_id"])
            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise AssertionError(f"{dataset}: dangling edge after filtering")
            writer.writerow({
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            })
            row_id += 1
            division_sources[source_id] = division_sources.get(source_id, 0) + 1

        node_count = len(nodes_by_id)
        edge_count = len(edges)
        total_nodes += node_count
        total_edges += edge_count
        stats_rows.append({
            "dataset": dataset,
            "raw_nodes": raw_node_count,
            "nodes": node_count,
            "raw_edges": filter_stats["raw_edges"],
            "edges": edge_count,
            "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
            "edge_to_node_ratio": edge_count / max(node_count, 1),
            "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
            **filter_stats,
        })

expected_datasets = set(test_stems)
missing_datasets = sorted(expected_datasets - seen_datasets)
extra_datasets = sorted(seen_datasets - expected_datasets)
if missing_datasets or extra_datasets:
    raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
assert total_nodes > 0, "No node rows produced"

header = SUBMISSION_PATH.open().readline().strip().split(",")
assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
stats["predict_minutes_total"] = predict_seconds / 60.0
stats["experiment_tag"] = EXPERIMENT_TAG
stats.to_csv(RUN_STATS_PATH, index=False)

print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
print(f"Wrote {RUN_STATS_PATH}")
display(stats.describe(include="all"))
display(pd.read_csv(SUBMISSION_PATH, nrows=8))

# ===== CELL 16 [markdown] =====
## Freeze the score-producing artifact before visualization

The next cell records the exact bytes, row count and dataset coverage of `submission.csv`.
Every teaching module runs only after this checkpoint.

# ===== CELL 17 [code] =====
import hashlib as _hashlib

_SUBMISSION_PATH = WORKING_DIR / "submission.csv"
if not _SUBMISSION_PATH.exists():
    raise FileNotFoundError(f"Expected submission file was not created: {_SUBMISSION_PATH}")

def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

SUBMISSION_SHA256_BEFORE_VISUALS = _sha256_file(_SUBMISSION_PATH)
SUBMISSION_BYTES_BEFORE_VISUALS = int(_SUBMISSION_PATH.stat().st_size)
_submission_guard_df = pd.read_csv(_SUBMISSION_PATH)
SUBMISSION_ROWS_BEFORE_VISUALS = int(len(_submission_guard_df))
SUBMISSION_DATASETS_BEFORE_VISUALS = tuple(
    sorted(_submission_guard_df["dataset"].astype(str).unique().tolist())
)

print("Submission frozen before visualization")
print("  sha256:", SUBMISSION_SHA256_BEFORE_VISUALS)
print("  bytes:", SUBMISSION_BYTES_BEFORE_VISUALS)
print("  rows:", SUBMISSION_ROWS_BEFORE_VISUALS)
print("  datasets:", len(SUBMISSION_DATASETS_BEFORE_VISUALS))

# ===== CELL 18 [markdown] =====
## Where the neural network is

The neural network is used **before** the graph-repair stages.

```text
4D microscopy volume [T, Z, Y, X]
              │
              ▼
3D UNet-style detector
  estimates likely cell-center locations
              │
              ▼
candidate nodes and local image features
              │
              ▼
Transformer / learned edge predictor
  scores possible links between nearby frames
              │
              ▼
ILP graph selection
  chooses a globally consistent candidate graph
              │
              ▼
non-neural graph post-processing
  motion relink, gap close, division recovery,
  pruning, short-track filtering and smoothing
```

The trained checkpoint used by this notebook is:

```text
weights/unet_transformer/split_0/edge_predictor_best.pth
```

There are three different kinds of changes a developer can make:

| Change | Retraining required? | Example |
|---|---:|---|
| Inference threshold | No | `BIOHUB_DET_THRESHOLD=0.98` |
| Graph/ILP or post-processing hyperparameter | No | `BIOHUB_OUTPUT_MIN_TRACK_LEN=5` |
| Neural architecture or learned weights | Usually yes | changing UNet channels, attention depth or feature dimensions |

The next cell inspects the actual attached repository and checkpoint at runtime. It does not guess a parameter count from the notebook text.

# ===== CELL 19 [code] =====
if RUN_PIPELINE_VISUALIZATION:
    import ast
    import html as _html
    from collections import defaultdict
    from IPython.display import HTML, display

    MODEL_REPORT_DIR = WORKING_DIR / "biohub_visual_walkthrough"
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _weight_path = REPO_DIR / WEIGHTS_RELATIVE
    _predict_script = REPO_DIR / "scripts" / "predict_unet_transformer.py"

    display(HTML("""
    <div style="border:1px solid #334155;border-radius:16px;padding:16px;background:#0b1220;color:#e2e8f0">
      <h3 style="margin-top:0;color:#7dd3fc">Neural-network inspection</h3>
      <p>This report is produced from the materialized source repository and checkpoint.
      It distinguishes exact checkpoint facts from conceptual explanations.</p>
    </div>
    """))

    def _checkpoint_tensor_candidates(obj, path="root", depth=0, max_depth=8):
        candidates = []
        if depth > max_depth:
            return candidates
        if isinstance(obj, dict):
            tensor_items = {
                str(k): v for k, v in obj.items()
                if hasattr(v, "numel") and hasattr(v, "shape")
            }
            if tensor_items:
                candidates.append((path, tensor_items))
            for key, value in obj.items():
                if isinstance(value, dict):
                    candidates.extend(
                        _checkpoint_tensor_candidates(value, f"{path}.{key}", depth + 1, max_depth)
                    )
        return candidates

    def _load_checkpoint_state_dict(path):
        import torch
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        candidates = _checkpoint_tensor_candidates(payload)
        if not candidates and isinstance(payload, dict):
            candidates = [("root", {
                str(k): v for k, v in payload.items()
                if hasattr(v, "numel") and hasattr(v, "shape")
            })]
        candidates = [(name, values) for name, values in candidates if values]
        if not candidates:
            return payload, None, {}
        selected_name, selected = max(
            candidates,
            key=lambda item: sum(int(v.numel()) for v in item[1].values())
        )
        return payload, selected_name, selected

    def _normalise_parameter_key(key):
        parts = str(key).split(".")
        while parts and parts[0] in {"module", "model", "network", "net", "state_dict"}:
            parts = parts[1:]
        return ".".join(parts)

    def _parameter_group(key):
        clean = _normalise_parameter_key(key)
        parts = clean.split(".")
        if not parts:
            return "unknown"
        # Two levels are usually enough to reveal UNet encoder/decoder,
        # transformer blocks and prediction heads without producing hundreds of rows.
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

    try:
        import torch

        _payload, _state_path, _state = _load_checkpoint_state_dict(_weight_path)
        if not _state:
            raise RuntimeError("No tensor state dictionary was found in the checkpoint.")

        _buffer_suffixes = (
            "running_mean", "running_var", "num_batches_tracked",
            "position_ids", "attn_mask"
        )
        _parameter_rows = []
        for key, tensor in _state.items():
            numel = int(tensor.numel())
            is_probable_buffer = str(key).endswith(_buffer_suffixes)
            _parameter_rows.append({
                "checkpoint_key": str(key),
                "module_group": _parameter_group(key),
                "shape": " × ".join(str(int(v)) for v in tensor.shape),
                "elements": numel,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "probable_buffer": bool(is_probable_buffer),
            })

        checkpoint_parameter_df = pd.DataFrame(_parameter_rows).sort_values(
            "elements", ascending=False
        )
        exact_stored_elements = int(checkpoint_parameter_df["elements"].sum())
        estimated_parameter_elements = int(
            checkpoint_parameter_df.loc[
                ~checkpoint_parameter_df["probable_buffer"], "elements"
            ].sum()
        )
        checkpoint_bytes = int(_weight_path.stat().st_size)

        module_parameter_df = (
            checkpoint_parameter_df.groupby("module_group", as_index=False)
            .agg(
                stored_elements=("elements", "sum"),
                tensor_count=("checkpoint_key", "count"),
                probable_buffer_elements=(
                    "elements",
                    lambda values: 0
                ),
            )
        )
        # Recompute the buffer count cleanly because groupby lambdas above do not
        # have access to the boolean column.
        _buffer_by_group = (
            checkpoint_parameter_df.loc[checkpoint_parameter_df["probable_buffer"]]
            .groupby("module_group")["elements"].sum()
        )
        module_parameter_df["probable_buffer_elements"] = (
            module_parameter_df["module_group"].map(_buffer_by_group).fillna(0).astype(int)
        )
        module_parameter_df["estimated_parameter_elements"] = (
            module_parameter_df["stored_elements"]
            - module_parameter_df["probable_buffer_elements"]
        )
        module_parameter_df = module_parameter_df.sort_values(
            "estimated_parameter_elements", ascending=False
        )

        display(HTML(f"""
        <div style="display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;margin:12px 0">
          <div style="background:#111827;border:1px solid #334155;border-radius:12px;padding:12px;color:#e2e8f0">
            <div style="font-size:11px;color:#94a3b8">Checkpoint file</div>
            <div style="font-size:18px;font-weight:700">{checkpoint_bytes / 1024**2:.2f} MB</div>
          </div>
          <div style="background:#111827;border:1px solid #334155;border-radius:12px;padding:12px;color:#e2e8f0">
            <div style="font-size:11px;color:#94a3b8">Stored tensors</div>
            <div style="font-size:18px;font-weight:700">{len(checkpoint_parameter_df):,}</div>
          </div>
          <div style="background:#111827;border:1px solid #334155;border-radius:12px;padding:12px;color:#e2e8f0">
            <div style="font-size:11px;color:#94a3b8">Stored tensor elements</div>
            <div style="font-size:18px;font-weight:700">{exact_stored_elements:,}</div>
          </div>
          <div style="background:#111827;border:1px solid #334155;border-radius:12px;padding:12px;color:#e2e8f0">
            <div style="font-size:11px;color:#94a3b8">Estimated parameters</div>
            <div style="font-size:18px;font-weight:700">{estimated_parameter_elements:,}</div>
          </div>
        </div>
        <p style="color:#cbd5e1;font-size:12px">
          <b>Stored tensor elements</b> is exact for the selected checkpoint state dictionary.
          <b>Estimated parameters</b> excludes common BatchNorm and mask buffers.
          An exact trainable-parameter count requires instantiating the original class,
          which may depend on repository-specific configuration.
        </p>
        """))

        display(module_parameter_df.head(30))
        display(checkpoint_parameter_df.head(40))

        _checkpoint_csv = MODEL_REPORT_DIR / "checkpoint_parameter_inventory.csv"
        _module_csv = MODEL_REPORT_DIR / "checkpoint_module_summary.csv"
        checkpoint_parameter_df.to_csv(_checkpoint_csv, index=False)
        module_parameter_df.to_csv(_module_csv, index=False)

        try:
            import plotly.express as px
            _plot_df = module_parameter_df.head(20).sort_values(
                "estimated_parameter_elements", ascending=True
            )
            _fig = px.bar(
                _plot_df,
                x="estimated_parameter_elements",
                y="module_group",
                orientation="h",
                title="Largest checkpoint module groups",
                labels={
                    "estimated_parameter_elements": "Estimated parameter elements",
                    "module_group": "Checkpoint module group",
                },
            )
            _fig.update_layout(template="plotly_dark", height=620, margin=dict(l=20, r=20, t=60, b=20))
            display(HTML(_fig.to_html(include_plotlyjs=True, full_html=False)))
            _fig.write_html(
                MODEL_REPORT_DIR / "checkpoint_parameter_groups.html",
                include_plotlyjs=True,
                full_html=True,
            )
        except Exception as _plot_exc:
            print("Parameter chart skipped:", type(_plot_exc).__name__, _plot_exc)

        print("Checkpoint tensor path selected:", _state_path)
        print("Saved:", _checkpoint_csv)
        print("Saved:", _module_csv)
    except Exception as _checkpoint_exc:
        print("Checkpoint inspection could not be completed:", type(_checkpoint_exc).__name__, _checkpoint_exc)
        checkpoint_parameter_df = pd.DataFrame()
        module_parameter_df = pd.DataFrame()

    def _base_name(base):
        try:
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return f"{_base_name(base.value)}.{base.attr}"
        except Exception:
            pass
        return "unknown"

    def _discover_model_source(repo_dir):
        rows = []
        keywords = ("unet", "transformer", "edge", "predictor", "attention", "model")
        for path in sorted((repo_dir / "src").rglob("*.py")) + sorted((repo_dir / "scripts").rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(text)
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = [_base_name(base) for base in node.bases]
                name_text = node.name.lower()
                source_segment = ast.get_source_segment(text, node) or ""
                looks_neural = (
                    any(word in name_text for word in keywords)
                    or any("module" in base.lower() for base in bases)
                    or "nn.Module" in source_segment
                    or "torch.nn" in source_segment
                )
                if looks_neural:
                    rows.append({
                        "class_name": node.name,
                        "base_classes": ", ".join(bases),
                        "source_file": str(path.relative_to(repo_dir)),
                        "line": int(getattr(node, "lineno", -1)),
                        "source_characters": len(source_segment),
                    })
        return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()

    architecture_source_df = _discover_model_source(REPO_DIR)
    display(HTML("""
    <h4>Source-discovered neural classes</h4>
    <p>The table below is generated by parsing the attached repository.
    It helps locate where the architecture can be modified.</p>
    """))
    if len(architecture_source_df):
        display(architecture_source_df.sort_values(["source_file", "line"]))
        architecture_source_df.to_csv(
            MODEL_REPORT_DIR / "neural_source_class_inventory.csv", index=False
        )
    else:
        print("No obvious nn.Module class was found by static source scanning.")

    if _predict_script.exists():
        _script_text = _predict_script.read_text(encoding="utf-8", errors="ignore")
        _interesting_lines = []
        for _line_no, _line in enumerate(_script_text.splitlines(), start=1):
            _low = _line.lower()
            if any(token in _low for token in (
                "model", "unet", "transformer", "edge_predict",
                "load_state", "checkpoint", "weights"
            )):
                _interesting_lines.append(f"{_line_no:04d}: {_line}")
        _excerpt = "\n".join(_interesting_lines[:180])
        display(HTML(
            "<details style='border:1px solid #334155;border-radius:12px;padding:12px;background:#0f172a;color:#e2e8f0'>"
            "<summary style='cursor:pointer;color:#7dd3fc;font-weight:700'>"
            "Open prediction-script model references</summary>"
            f"<pre style='white-space:pre-wrap;font-size:11px;line-height:1.4'>{_html.escape(_excerpt)}</pre>"
            "</details>"
        ))
        (MODEL_REPORT_DIR / "prediction_script_model_references.txt").write_text(
            _excerpt, encoding="utf-8"
        )

    display(HTML("""
    <div style="margin-top:14px;border-left:4px solid #fbbf24;padding:10px 14px;background:#1f2937;color:#e5e7eb">
      <b>Can the architecture be changed?</b><br>
      Yes, but architecture changes normally invalidate the current checkpoint.
      Changing detector channels, attention blocks, embeddings or heads requires
      retraining or a compatible new checkpoint. Thresholds and graph-repair
      hyperparameters can be changed without retraining.
    </div>
    """))
else:
    print("Neural-network teaching report is OFF in submission mode.")

# ===== CELL 20 [markdown] =====
### v030 fix: robust GEFF path resolver

The visualization replay no longer assumes that the prediction directory name equals the full dataset stem. It indexes the actual `.geff` files already discovered by the submission builder and falls back to a recursive search under `tracking_repo/predictions`. This fixes cases where the graph is saved as `predictions/<embryo>/unet_transformer/split_0/<dataset>.geff`.

# ===== CELL 21 [markdown] =====
## Full visual walkthrough: raw graph → repaired lineage → animation

This section runs only when `RUN_PIPELINE_VISUALIZATION=True`. It is automatically disabled in Kaggle submission reruns.

Unlike a single EDA image, this walkthrough reloads the predicted `.geff` graph and replays the same output-stage logic used for `submission.csv`:

1. raw detector / ILP graph,
2. physical edge filter,
3. motion relink,
4. single-parent repair,
5. one-frame gap close,
6. safe division recovery,
7. isolated-node pruning,
8. short-track filter with `min_track_len = 6`,
9. final line-fit smoothing.

The notebook then selects a small field of view. If a final division-like source exists, it centers the static plots and animation on that parent cell; otherwise it centers on the densest visible frame. The output includes several static figures and an embedded HTML animation showing the final repaired lineage graph over time.

# ===== CELL 22 [code] =====
if RUN_PIPELINE_VISUALIZATION:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display
    from collections import Counter, defaultdict
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    plt.rcParams["animation.embed_limit"] = int(os.environ.get("BIOHUB_VISUAL_ANIMATION_EMBED_LIMIT_MB", "80"))

    VISUAL_DIR = WORKING_DIR / "biohub_visual_walkthrough"
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_CROP_SIZE = int(os.environ.get("BIOHUB_VISUAL_CROP_SIZE", "192"))
    VISUAL_ANIMATION_RADIUS = int(os.environ.get("BIOHUB_VISUAL_ANIMATION_RADIUS", "7"))
    VISUAL_MAX_FRAMES = int(os.environ.get("BIOHUB_VISUAL_ANIMATION_MAX_FRAMES", "18"))
    VISUAL_MAX_NODES = int(os.environ.get("BIOHUB_VISUAL_MAX_NODES", "900"))
    VISUAL_MAX_EDGES = int(os.environ.get("BIOHUB_VISUAL_MAX_EDGES", "1200"))

    print("Visual mode is ON. These figures are skipped automatically during Kaggle submission reruns.")
    print(f"Visual outputs will also be saved under: {VISUAL_DIR}")

    def _copy_nodes(nodes):
        return {int(k): dict(v) for k, v in nodes.items()}

    def _copy_edges(edges):
        return [dict(edge) for edge in edges]

    def _empty_filter_stats(raw_edges_len=0):
        keys = [
            "raw_edges", "dropped_nonconsecutive_edges", "dropped_long_edges",
            "dropped_multi_parent_edges", "dropped_multi_child_edges", "dropped_division_edges",
            "gap_candidates", "gap_pairs_selected", "gap_reused_existing", "gap_inserted_synthetic",
            "gap_added_nodes", "gap_added_edges", "gap_skipped_node_cap", "gap_refined_synthetic",
            "gap_refine_failed", "gap_refine_rejected_shift", "gap_close_effective_max_gap",
            "pruned_isolated_nodes", "motion_relink_edges", "motion_relink_tight_edges",
            "motion_relink_relaxed_edges", "motion_relink_frames", "motion_relink_replaced_raw_edges",
            "motion_relink_fallback_raw", "motion_relink_skipped_large_frame", "gap2_candidates",
            "gap2_pairs_selected", "gap2_added_nodes", "gap2_added_edges", "gap2_skipped_cap",
            "safe_division_candidates", "safe_divisions_added", "safe_division_skipped_cap",
            "short_track_components_removed", "short_track_nodes_removed", "short_track_edges_removed",
            "short_track_filter_skipped_all", "linefit_smoothed_nodes", "linefit_skipped_nodes",
        ]
        stats_template = {key: 0 for key in keys}
        stats_template["raw_edges"] = int(raw_edges_len)
        return stats_template

    def _refresh_geff_index():
        """Build a robust lookup table for prediction graphs produced by the repo script.

        The baseline writer streams from the recursive `geffs` list. In some repo
        versions the prediction folder is grouped by embryo, while the .geff file
        is named by the full dataset stem. A hard-coded path such as
        `predictions/<dataset>/<method>/split_0/<dataset>.geff` can therefore be
        wrong even when the submission builder has already found the graph.
        """
        indexed = {}
        candidate_paths = []
        try:
            candidate_paths.extend(list(geffs))
        except NameError:
            pass
        prediction_root = REPO_DIR / "predictions"
        if prediction_root.exists():
            candidate_paths.extend(sorted(prediction_root.glob(f"**/{METHOD}/split_0/*.geff")))
            candidate_paths.extend(sorted(prediction_root.glob("**/*.geff")))
        for path in candidate_paths:
            path = Path(path)
            if path.exists() and path.suffix == ".geff":
                indexed.setdefault(path.stem, path)
        return indexed

    GEFF_BY_DATASET = _refresh_geff_index()
    print(f"Visual GEFF resolver indexed {len(GEFF_BY_DATASET)} graph(s).")

    def _resolve_geff_path(dataset):
        dataset = str(dataset)
        candidates = [
            REPO_DIR / "predictions" / dataset / METHOD / "split_0" / f"{dataset}.geff",
            REPO_DIR / "predictions" / dataset.split("_")[0] / METHOD / "split_0" / f"{dataset}.geff",
            GEFF_BY_DATASET.get(dataset),
        ]
        for candidate in candidates:
            if candidate is not None and Path(candidate).exists():
                return Path(candidate)

        # Last-resort recursive lookup. This handles future upstream changes in
        # prediction directory naming without changing the submission path.
        prediction_root = REPO_DIR / "predictions"
        if prediction_root.exists():
            exact = sorted(prediction_root.glob(f"**/{dataset}.geff"))
            if exact:
                GEFF_BY_DATASET[dataset] = exact[0]
                return exact[0]
            fuzzy = sorted(path for path in prediction_root.glob("**/*.geff") if path.stem == dataset)
            if fuzzy:
                GEFF_BY_DATASET[dataset] = fuzzy[0]
                return fuzzy[0]

        available = sorted(GEFF_BY_DATASET)[:12]
        raise FileNotFoundError(
            f"Could not locate a .geff graph for dataset={dataset!r}. "
            f"Indexed examples: {available}. Prediction root: {prediction_root}"
        )

    def _load_raw_graph_tables(dataset):
        geff_path = _resolve_geff_path(dataset)
        print(f"Loading raw graph for {dataset}: {geff_path}")
        graph = graph_from_geff(geff_path)
        nodes_by_id = {}
        for row in graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
        raw_edges = []
        for row in graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })
        return nodes_by_id, raw_edges

    def _physics_filter_edges(nodes_by_id, raw_edges, stats_v):
        edges = []
        for edge in raw_edges:
            source = nodes_by_id.get(int(edge["source_id"]))
            target = nodes_by_id.get(int(edge["target_id"]))
            if source is None or target is None:
                continue
            if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
                stats_v["dropped_nonconsecutive_edges"] += 1
                continue
            distance_um = edge_distance_um(source, target)
            edge = dict(edge)
            edge["distance_um"] = float(distance_um)
            if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
                stats_v["dropped_long_edges"] += 1
                continue
            edges.append(edge)
        return edges

    def _repair_single_parent(edges, stats_v):
        if not OUTPUT_SINGLE_PARENT_REPAIR or not edges:
            return edges
        best_by_target = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats_v["dropped_multi_parent_edges"] += sum(1 for edge in edges if id(edge) not in kept_ids)
        return [edge for edge in edges if id(edge) in kept_ids]

    def _repair_single_child(edges, stats_v):
        if not OUTPUT_SINGLE_CHILD_REPAIR or not edges:
            return edges
        best_by_source = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats_v["dropped_multi_child_edges"] += sum(1 for edge in edges if id(edge) not in kept_ids)
        return [edge for edge in edges if id(edge) in kept_ids]

    def _division_sources(edges):
        outgoing = Counter(int(edge["source_id"]) for edge in edges)
        return {source_id for source_id, count in outgoing.items() if count >= 2}

    def _frame_counts(nodes):
        counts = Counter(int(node["t"]) for node in nodes.values())
        return counts

    def _snapshot_summary(stage, nodes, edges):
        return {
            "stage": stage,
            "nodes": len(nodes),
            "edges": len(edges),
            "division_sources": len(_division_sources(edges)),
            "frames": len(_frame_counts(nodes)),
        }

    def _replay_output_stages(dataset):
        raw_nodes, raw_edges = _load_raw_graph_tables(dataset)
        stats_v = _empty_filter_stats(len(raw_edges))
        snapshots = []
        snapshots.append(("00 raw detector + ILP graph", _copy_nodes(raw_nodes), _copy_edges(raw_edges)))

        nodes_current = _copy_nodes(raw_nodes)
        edges_current = _physics_filter_edges(nodes_current, _copy_edges(raw_edges), stats_v)
        snapshots.append(("01 physical edge filter", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        if OUTPUT_MOTION_RELINK:
            learned_edge_probs = {}
            for edge in edges_current:
                prob = edge.get("edge_prob")
                if prob is None:
                    continue
                try:
                    prob = float(prob)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(prob):
                    key = (int(edge["source_id"]), int(edge["target_id"]))
                    learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
            motion_edges = motion_relink_edges(nodes_current, stats_v, learned_edge_probs)
            if motion_edges:
                stats_v["motion_relink_replaced_raw_edges"] = len(edges_current)
                edges_current = motion_edges
            else:
                stats_v["motion_relink_fallback_raw"] = 1
        snapshots.append(("02 motion relink", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        edges_current = _repair_single_parent(edges_current, stats_v)
        edges_current = _repair_single_child(edges_current, stats_v)
        snapshots.append(("03 parent repair", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        nodes_current, edges_current = close_single_frame_gaps(_copy_nodes(nodes_current), _copy_edges(edges_current), stats_v, dataset=dataset)
        snapshots.append(("04 one-frame gap close", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        nodes_current, edges_current = recover_strict_gap2(_copy_nodes(nodes_current), _copy_edges(edges_current), stats_v, dataset=dataset)
        if OUTPUT_GAP2_RECOVERY:
            snapshots.append(("05 strict gap2 recovery", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        edges_current = add_safe_divisions_postlink(nodes_current, _copy_edges(edges_current), stats_v)
        snapshots.append(("05 safe division recovery", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        if OUTPUT_DIVISION_GEOMETRY_FILTER and edges_current:
            # The 0.897 baseline keeps this disabled. The branch is included only for faithful replay under overrides.
            by_source = {}
            for edge in edges_current:
                by_source.setdefault(int(edge["source_id"]), []).append(edge)
            filtered = []
            for source_id, source_edges in by_source.items():
                if len(source_edges) <= 1:
                    filtered.extend(source_edges)
                    continue
                ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
                source = nodes_current[source_id]
                top1, top2 = ranked[0], ranked[1]
                d1, d2 = float(top1["distance_um"]), float(top2["distance_um"])
                sister = edge_distance_um(nodes_current[int(top1["target_id"])], nodes_current[int(top2["target_id"])])
                valid_division = (
                    max(d1, d2) <= DIV_PARENT_MAX_UM
                    and sister <= DIV_SISTER_MAX_UM
                    and int(nodes_current[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                    and int(nodes_current[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
                )
                if valid_division:
                    filtered.extend([top1, top2])
                    stats_v["dropped_division_edges"] += max(0, len(ranked) - 2)
                elif DIV_DROP_TO_SINGLE_IF_BAD:
                    filtered.append(top1)
                    stats_v["dropped_division_edges"] += len(ranked) - 1
                else:
                    filtered.extend(ranked)
            edges_current = filtered
            snapshots.append(("06 division geometry filter", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        if OUTPUT_PRUNE_ISOLATED:
            incident = {int(edge["source_id"]) for edge in edges_current} | {int(edge["target_id"]) for edge in edges_current}
            if incident:
                kept_nodes = {node_id: node for node_id, node in nodes_current.items() if node_id in incident}
                stats_v["pruned_isolated_nodes"] += len(nodes_current) - len(kept_nodes)
                nodes_current = kept_nodes
                edges_current = [edge for edge in edges_current if int(edge["source_id"]) in nodes_current and int(edge["target_id"]) in nodes_current]
        snapshots.append(("06 prune isolated nodes", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        nodes_current, edges_current = filter_short_track_components(_copy_nodes(nodes_current), _copy_edges(edges_current), stats_v)
        snapshots.append(("07 short-track filter min6", _copy_nodes(nodes_current), _copy_edges(edges_current)))

        nodes_current = linefit_smooth_output_graph(_copy_nodes(nodes_current), _copy_edges(edges_current), stats_v)
        snapshots.append(("08 final line-fit smoothing", _copy_nodes(nodes_current), _copy_edges(edges_current)))
        return snapshots, stats_v

    def _select_visual_dataset(stats_df):
        if len(stats_df) == 0:
            raise RuntimeError("No run stats are available for visualization.")
        candidates = stats_df.copy()
        candidates["_has_geff"] = candidates["dataset"].astype(str).map(lambda name: str(name) in GEFF_BY_DATASET or _quiet_path_exists(name))
        candidates = candidates[candidates["_has_geff"]].copy()
        if len(candidates) == 0:
            raise RuntimeError("Run stats exist, but none of those datasets have a resolvable .geff file for visualization.")
        if "division_like_sources" in candidates.columns and candidates["division_like_sources"].fillna(0).max() > 0:
            row = candidates.sort_values(["division_like_sources", "edges", "nodes"], ascending=False).iloc[0]
            reason = "highest final division-like source count"
        else:
            row = candidates.sort_values(["edges", "nodes"], ascending=False).iloc[0]
            reason = "no final divisions in the processed subset; selected the richest graph"
        return str(row["dataset"]), reason

    def _quiet_path_exists(dataset):
        try:
            _resolve_geff_path(dataset)
            return True
        except Exception:
            return False

    def _safe_percentile_image(image2d):
        arr = np.asarray(image2d, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo, hi = np.percentile(arr, [1.0, 99.7])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi <= lo:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0, 1)

    def _projection_for_frame(dataset, t, crop=None, frame_cache=None):
        frame_cache = frame_cache if frame_cache is not None else {}
        frame = read_test_frame(dataset, int(t), frame_cache)
        projection = np.max(frame, axis=0)
        if crop is not None:
            y0, y1, x0, x1 = crop
            projection = projection[y0:y1, x0:x1]
        return _safe_percentile_image(projection)

    def _node_in_crop(node, crop):
        if crop is None:
            return True
        y0, y1, x0, x1 = crop
        return y0 <= float(node["y"]) < y1 and x0 <= float(node["x"]) < x1

    def _resolve_test_zarr_path(dataset):
        dataset = str(dataset)
        candidates = [TEST_DIR / f"{dataset}.zarr"]
        if "_" in dataset:
            candidates.append(TEST_DIR / f"{dataset.split('_')[0]}.zarr")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted(TEST_DIR.glob(f"{dataset}*.zarr")) if TEST_DIR.exists() else []
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not locate test zarr for dataset={dataset!r} under {TEST_DIR}")

    def _crop_around(nodes, center_node, crop_size, dataset):
        zarr_path = _resolve_test_zarr_path(dataset)
        meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
        shape = tuple(int(v) for v in meta["shape"])
        _, _, height, width = shape
        cy = float(center_node["y"])
        cx = float(center_node["x"])
        half = max(16, crop_size // 2)
        y0 = int(max(0, round(cy) - half))
        y1 = int(min(height, y0 + crop_size))
        y0 = int(max(0, y1 - crop_size))
        x0 = int(max(0, round(cx) - half))
        x1 = int(min(width, x0 + crop_size))
        x0 = int(max(0, x1 - crop_size))
        return (y0, y1, x0, x1)

    def _select_focus(nodes, edges):
        div_sources = sorted(_division_sources(edges))
        if div_sources:
            source_id = div_sources[0]
            return source_id, int(nodes[source_id]["t"]), nodes[source_id], "final division source"
        counts = _frame_counts(nodes)
        focus_t = counts.most_common(1)[0][0] if counts else 0
        frame_nodes = [node for node in nodes.values() if int(node["t"]) == int(focus_t)]
        if frame_nodes:
            center_node = sorted(frame_nodes, key=lambda n: int(n["node_id"]))[len(frame_nodes) // 2]
        else:
            center_node = next(iter(nodes.values()))
        return int(center_node["node_id"]), int(focus_t), center_node, "densest frame center"

    def _sample_items(items, limit):
        items = list(items)
        if len(items) <= limit:
            return items
        idx = np.linspace(0, len(items) - 1, limit).astype(int)
        return [items[i] for i in idx]

    def _draw_overlay(ax, dataset, nodes, edges, frame_t, crop, title, edge_mode="touching", frame_cache=None):
        y0, y1, x0, x1 = crop
        projection = _projection_for_frame(dataset, frame_t, crop=crop, frame_cache=frame_cache)
        ax.imshow(projection, cmap="gray", extent=[x0, x1, y1, y0], interpolation="nearest")
        frame_nodes = [node for node in nodes.values() if int(node["t"]) == int(frame_t) and _node_in_crop(node, crop)]
        frame_nodes = _sample_items(sorted(frame_nodes, key=lambda n: int(n["node_id"])), VISUAL_MAX_NODES)
        if frame_nodes:
            ax.scatter([float(n["x"]) for n in frame_nodes], [float(n["y"]) for n in frame_nodes], s=13, c="#41d6ff", alpha=0.82, linewidths=0, label="nodes")
        div_sources = _division_sources(edges)
        div_nodes = [nodes[sid] for sid in div_sources if sid in nodes and int(nodes[sid]["t"]) == int(frame_t) and _node_in_crop(nodes[sid], crop)]
        if div_nodes:
            ax.scatter([float(n["x"]) for n in div_nodes], [float(n["y"]) for n in div_nodes], s=95, marker="*", c="#ffcc33", edgecolors="#111111", linewidths=0.5, label="division source")
        visible_edges = []
        for edge in edges:
            sid = int(edge["source_id"]); tid = int(edge["target_id"])
            if sid not in nodes or tid not in nodes:
                continue
            s = nodes[sid]; t = nodes[tid]
            touches_frame = int(s["t"]) == int(frame_t) or int(t["t"]) == int(frame_t)
            if edge_mode == "current_to_next":
                touches_frame = int(s["t"]) == int(frame_t)
            if not touches_frame:
                continue
            if not (_node_in_crop(s, crop) or _node_in_crop(t, crop)):
                continue
            visible_edges.append((s, t, sid in div_sources))
        visible_edges = _sample_items(visible_edges, VISUAL_MAX_EDGES)
        for s, t, is_div in visible_edges:
            color = "#ffcc33" if is_div else "#ff5ea8"
            alpha = 0.88 if is_div else 0.42
            lw = 1.7 if is_div else 0.8
            ax.plot([float(s["x"]), float(t["x"])], [float(s["y"]), float(t["y"])], color=color, alpha=alpha, linewidth=lw)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x voxel")
        ax.set_ylabel("y voxel")

    def _stage_accounting_figure(dataset, snapshots):
        rows = [_snapshot_summary(stage, nodes, edges) for stage, nodes, edges in snapshots]
        summary = pd.DataFrame(rows)
        display(summary)
        x = np.arange(len(summary))
        fig, ax1 = plt.subplots(figsize=(12, 4.8))
        ax1.plot(x, summary["nodes"], marker="o", label="nodes")
        ax1.plot(x, summary["edges"], marker="o", label="edges")
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary["stage"], rotation=35, ha="right")
        ax1.set_ylabel("count")
        ax1.set_title(f"{dataset}: graph size through output post-processing")
        ax1.grid(alpha=0.22)
        ax2 = ax1.twinx()
        ax2.bar(x, summary["division_sources"], alpha=0.24, label="division sources")
        ax2.set_ylabel("division-like source count")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        fig.tight_layout()
        path = VISUAL_DIR / f"{dataset}_stage_accounting.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(f"Saved: {path}")
        return summary

    def _raw_vs_final_figure(dataset, snapshots, focus_t, crop):
        raw_stage, raw_nodes, raw_edges = snapshots[0]
        final_stage, final_nodes, final_edges = snapshots[-1]
        frame_cache = {}
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
        _draw_overlay(axes[0], dataset, raw_nodes, raw_edges, focus_t, crop, f"Raw graph around t={focus_t}", frame_cache=frame_cache)
        _draw_overlay(axes[1], dataset, final_nodes, final_edges, focus_t, crop, f"Final repaired lineage around t={focus_t}", frame_cache=frame_cache)
        fig.suptitle("Raw learned/ILP graph vs final repaired graph", fontsize=13)
        fig.tight_layout()
        path = VISUAL_DIR / f"{dataset}_raw_vs_final_t{focus_t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(f"Saved: {path}")

    def _stage_grid_figure(dataset, snapshots, focus_t, crop):
        frame_cache = {}
        selected = []
        wanted = ["00 raw", "02 motion", "04 one-frame", "05 safe", "07 short", "08 final"]
        for prefix in wanted:
            for snap in snapshots:
                if snap[0].startswith(prefix):
                    selected.append(snap)
                    break
        selected = selected[:6]
        ncols = 3
        nrows = int(math.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, (stage, nodes, edges) in zip(axes, selected):
            _draw_overlay(ax, dataset, nodes, edges, focus_t, crop, stage, frame_cache=frame_cache)
        for ax in axes[len(selected):]:
            ax.axis("off")
        fig.suptitle("Same field of view after each major graph-repair stage", fontsize=13)
        fig.tight_layout()
        path = VISUAL_DIR / f"{dataset}_stage_grid_t{focus_t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(f"Saved: {path}")

    def _time_lineage_figure(dataset, nodes, edges, focus_t, crop, radius):
        t0 = max(min(int(n["t"]) for n in nodes.values()), int(focus_t) - radius)
        t1 = min(max(int(n["t"]) for n in nodes.values()), int(focus_t) + radius)
        y0, y1, x0, x1 = crop
        in_nodes = {nid: node for nid, node in nodes.items() if t0 <= int(node["t"]) <= t1 and _node_in_crop(node, crop)}
        in_edges = [edge for edge in edges if int(edge["source_id"]) in in_nodes and int(edge["target_id"]) in in_nodes]
        div_sources = _division_sources(in_edges)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for edge in _sample_items(in_edges, VISUAL_MAX_EDGES):
            s = in_nodes[int(edge["source_id"])]
            t = in_nodes[int(edge["target_id"])]
            is_div = int(edge["source_id"]) in div_sources
            ax.plot([int(s["t"]), int(t["t"])], [float(s["y"]), float(t["y"])], color="#ffcc33" if is_div else "#7dd3fc", alpha=0.8 if is_div else 0.35, linewidth=1.8 if is_div else 0.8)
        normal_nodes = [node for nid, node in in_nodes.items() if nid not in div_sources]
        branch_nodes = [in_nodes[nid] for nid in div_sources if nid in in_nodes]
        if normal_nodes:
            ax.scatter([int(n["t"]) for n in normal_nodes], [float(n["y"]) for n in normal_nodes], s=9, alpha=0.56, c="#41d6ff")
        if branch_nodes:
            ax.scatter([int(n["t"]) for n in branch_nodes], [float(n["y"]) for n in branch_nodes], s=120, marker="*", c="#ffcc33", edgecolors="#111111", linewidths=0.5, label="division source")
        ax.set_title(f"{dataset}: final lineage graph as time-vs-y tracks")
        ax.set_xlabel("time frame")
        ax.set_ylabel("y voxel within crop")
        ax.set_xlim(t0 - 0.2, t1 + 0.2)
        ax.set_ylim(y1, y0)
        ax.grid(alpha=0.22)
        if branch_nodes:
            ax.legend(loc="best")
        fig.tight_layout()
        path = VISUAL_DIR / f"{dataset}_lineage_time_plot_t{focus_t}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(f"Saved: {path}")

    def _make_final_lineage_animation(dataset, nodes, edges, focus_t, crop, radius):
        min_t = min(int(n["t"]) for n in nodes.values())
        max_t = max(int(n["t"]) for n in nodes.values())
        t0 = max(min_t, int(focus_t) - radius)
        t1 = min(max_t, int(focus_t) + radius)
        frames = list(range(t0, t1 + 1))
        if len(frames) > VISUAL_MAX_FRAMES:
            center_idx = len(frames) // 2
            half = VISUAL_MAX_FRAMES // 2
            frames = frames[max(0, center_idx - half): max(0, center_idx - half) + VISUAL_MAX_FRAMES]
        frame_cache = {}
        projections = {t: _projection_for_frame(dataset, t, crop=crop, frame_cache=frame_cache) for t in frames}
        y0, y1, x0, x1 = crop
        div_sources = _division_sources(edges)
        nodes_by_t = defaultdict(list)
        for node_id, node in nodes.items():
            if int(node["t"]) in frames and _node_in_crop(node, crop):
                nodes_by_t[int(node["t"])].append((node_id, node))
        trail = max(3, min(6, len(frames)))
        fig, ax = plt.subplots(figsize=(6.8, 6.5))

        def update(frame_t):
            ax.clear()
            ax.imshow(projections[frame_t], cmap="gray", extent=[x0, x1, y1, y0], interpolation="nearest")
            active_nodes = [item for item in nodes_by_t.get(frame_t, [])]
            active_nodes = _sample_items(sorted(active_nodes, key=lambda kv: int(kv[0])), VISUAL_MAX_NODES)
            if active_nodes:
                ax.scatter([float(n["x"]) for _, n in active_nodes], [float(n["y"]) for _, n in active_nodes], s=18, c="#41d6ff", alpha=0.9, linewidths=0)
            active_div = [(nid, n) for nid, n in active_nodes if nid in div_sources]
            if active_div:
                ax.scatter([float(n["x"]) for _, n in active_div], [float(n["y"]) for _, n in active_div], s=130, marker="*", c="#ffcc33", edgecolors="#111111", linewidths=0.6)
            visible_edges = []
            for edge in edges:
                sid = int(edge["source_id"]); tid = int(edge["target_id"])
                if sid not in nodes or tid not in nodes:
                    continue
                s = nodes[sid]; t = nodes[tid]
                if not (frame_t - trail <= int(s["t"]) <= frame_t and int(t["t"]) <= frame_t):
                    continue
                if not (_node_in_crop(s, crop) or _node_in_crop(t, crop)):
                    continue
                visible_edges.append((s, t, sid in div_sources))
            for s, t, is_div in _sample_items(visible_edges, VISUAL_MAX_EDGES):
                ax.plot([float(s["x"]), float(t["x"])], [float(s["y"]), float(t["y"])], color="#ffcc33" if is_div else "#ff5ea8", alpha=0.92 if is_div else 0.46, linewidth=2.0 if is_div else 0.9)
            ax.text(x0 + 4, y0 + 12, f"t = {frame_t}", color="white", fontsize=12, bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=4))
            if div_sources:
                ax.text(x0 + 4, y0 + 30, "★ final division source", color="#ffcc33", fontsize=10, bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=3))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_title(f"{dataset}: final lineage animation")
            ax.set_xlabel("x voxel")
            ax.set_ylabel("y voxel")
            return []

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=650, blit=False, repeat=True)
        html = anim.to_jshtml(fps=1.6)
        html_path = VISUAL_DIR / f"{dataset}_final_lineage_animation_t{focus_t}.html"
        html_path.write_text(html, encoding="utf-8")
        plt.close(fig)
        display(HTML(f"<h3>Final repaired lineage animation: {dataset}</h3>"))
        display(HTML(html))
        print(f"Saved animation HTML: {html_path}")
        if not div_sources:
            print("Note: no final division-like source was found in this processed subset. Increase BIOHUB_VISUAL_DEMO_SAMPLE_COUNT to search more samples.")

    visual_dataset, selection_reason = _select_visual_dataset(stats)
    print(f"Selected dataset for visual walkthrough: {visual_dataset} ({selection_reason})")
    snapshots, visual_stats = _replay_output_stages(visual_dataset)
    final_stage, final_nodes, final_edges = snapshots[-1]
    focus_node_id, focus_t, focus_node, focus_reason = _select_focus(final_nodes, final_edges)
    crop = _crop_around(final_nodes, focus_node, VISUAL_CROP_SIZE, visual_dataset)
    print(f"Focus node: {focus_node_id} at t={focus_t} ({focus_reason})")
    print(f"Crop window y/x: {crop}")
    print("Replay stats:")
    display(pd.DataFrame([visual_stats]).T.rename(columns={0: "value"}).query("value != 0"))

    _stage_accounting_figure(visual_dataset, snapshots)
    _raw_vs_final_figure(visual_dataset, snapshots, focus_t, crop)
    _stage_grid_figure(visual_dataset, snapshots, focus_t, crop)
    _time_lineage_figure(visual_dataset, final_nodes, final_edges, focus_t, crop, VISUAL_ANIMATION_RADIUS)
    _make_final_lineage_animation(visual_dataset, final_nodes, final_edges, focus_t, crop, VISUAL_ANIMATION_RADIUS)
else:
    print("Pipeline visualization is OFF. This is expected in Kaggle submission mode.")

# ===== CELL 23 [markdown] =====
## Detector localization laboratory: how the cell centers are found

A beginner often asks an earlier question than tracking:

> **Before linking cells across time, how does the model decide that a cell exists at all?**

For this baseline, the first learned stage is a **UNet-style detector**. Conceptually it behaves like this:

```text
3D microscopy frame [Z, Y, X]
          │
          ▼
UNet-style detector
          │
          ├─ dense center-likelihood volume (heatmap)
          │
          ├─ thresholding / candidate extraction
          │
          └─ final centroid nodes (t, z, y, x)
```

Important teaching note:

- In many biomedical pipelines, UNet predicts a **mask**.
- In this notebook, the most useful beginner view is usually a **cell-center likelihood heatmap**, not a full instance mask.
- If the repository does not materialize a dense detector score volume on disk, the notebook creates a **pedagogical surrogate heatmap** from the raw detected centroids. This still helps you understand the localization stage.

The next cell visualizes the detector step as:

1. raw microscopy projection,
2. detector heatmap,
3. thresholded detector mask,
4. extracted candidate centers.

It also shows sample node rows so that beginners can connect the picture to the node table used by later graph stages.

# ===== CELL 24 [code] =====
if RUN_PIPELINE_VISUALIZATION:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from IPython.display import HTML, display

    DETECTOR_LAB_DIR = VISUAL_DIR / "detector_localization_lab"
    DETECTOR_LAB_DIR.mkdir(parents=True, exist_ok=True)

    DETECTOR_SIGMA_Z = float(os.environ.get("BIOHUB_DETECTOR_SIGMA_Z", "1.2"))
    DETECTOR_SIGMA_XY = float(os.environ.get("BIOHUB_DETECTOR_SIGMA_XY", "2.8"))
    DETECTOR_THRESHOLD_VIEW = float(os.environ.get("BIOHUB_DETECTOR_THRESHOLD_VIEW", "0.35"))
    DETECTOR_MAX_TABLE_ROWS = int(os.environ.get("BIOHUB_DETECTOR_MAX_TABLE_ROWS", "12"))

    def _candidate_detector_score_paths(dataset):
        dataset = str(dataset)
        prediction_root = REPO_DIR / "predictions"
        patterns = [
            "**/*score*.npy", "**/*score*.npz",
            "**/*prob*.npy", "**/*prob*.npz",
            "**/*heat*.npy", "**/*heat*.npz",
            "**/*logit*.npy", "**/*logit*.npz",
            "**/*det*.npy", "**/*det*.npz",
        ]
        found = []
        if prediction_root.exists():
            for pattern in patterns:
                for path in prediction_root.glob(pattern):
                    low = str(path).lower()
                    if dataset.lower() in low:
                        found.append(path)
        # keep deterministic order
        uniq = []
        seen = set()
        for path in sorted(found):
            if path not in seen:
                uniq.append(path)
                seen.add(path)
        return uniq

    def _maybe_extract_score_volume(obj, frame_shape, t):
        arr = np.asarray(obj)
        if arr.ndim == 3 and tuple(arr.shape) == tuple(frame_shape):
            return arr.astype(np.float32)
        if arr.ndim == 4 and arr.shape[1:] == tuple(frame_shape) and 0 <= int(t) < arr.shape[0]:
            return np.asarray(arr[int(t)], dtype=np.float32)
        return None

    def _try_load_detector_score_volume(dataset, t, frame_shape):
        for path in _candidate_detector_score_paths(dataset):
            try:
                if path.suffix == ".npy":
                    arr = np.load(path, allow_pickle=True)
                    vol = _maybe_extract_score_volume(arr, frame_shape, t)
                    if vol is not None:
                        return vol, f"direct score volume from {path.name}"
                elif path.suffix == ".npz":
                    data = np.load(path, allow_pickle=True)
                    for key in data.files:
                        vol = _maybe_extract_score_volume(data[key], frame_shape, t)
                        if vol is not None:
                            return vol, f"direct score volume from {path.name}:{key}"
            except Exception:
                continue
        return None, None

    def _gaussian_surrogate_from_nodes(frame_shape, nodes, sigma_z, sigma_xy):
        zdim, ydim, xdim = [int(v) for v in frame_shape]
        heat = np.zeros((zdim, ydim, xdim), dtype=np.float32)
        rz = max(1, int(np.ceil(3.0 * sigma_z)))
        rxy = max(1, int(np.ceil(3.0 * sigma_xy)))
        for node in nodes:
            cz = float(node["z"])
            cy = float(node["y"])
            cx = float(node["x"])
            z0 = max(0, int(np.floor(cz)) - rz)
            z1 = min(zdim, int(np.floor(cz)) + rz + 1)
            y0 = max(0, int(np.floor(cy)) - rxy)
            y1 = min(ydim, int(np.floor(cy)) + rxy + 1)
            x0 = max(0, int(np.floor(cx)) - rxy)
            x1 = min(xdim, int(np.floor(cx)) + rxy + 1)
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            zz = np.arange(z0, z1, dtype=np.float32)[:, None, None]
            yy = np.arange(y0, y1, dtype=np.float32)[None, :, None]
            xx = np.arange(x0, x1, dtype=np.float32)[None, None, :]
            local = np.exp(-0.5 * (((zz - cz) / sigma_z) ** 2 + ((yy - cy) / sigma_xy) ** 2 + ((xx - cx) / sigma_xy) ** 2))
            heat[z0:z1, y0:y1, x0:x1] = np.maximum(heat[z0:z1, y0:y1, x0:x1], local.astype(np.float32))
        return heat

    def _norm01(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return arr
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0, 1)

    def _projection_triplet(volume, crop):
        y0, y1, x0, x1 = crop
        vol = np.asarray(volume)
        vol_crop = vol[:, y0:y1, x0:x1]
        proj_xy = np.max(vol_crop, axis=0)          # [Y, X]
        proj_xz = np.max(vol_crop, axis=1)          # [Z, X]
        proj_yz = np.max(vol_crop, axis=2)          # [Z, Y]
        return vol_crop, proj_xy, proj_xz, proj_yz

    def _nodes_for_frame(nodes_by_id, frame_t):
        return [node for node in nodes_by_id.values() if int(node["t"]) == int(frame_t)]

    def _nodes_for_frame_and_crop(nodes_by_id, frame_t, crop):
        return [node for node in nodes_by_id.values() if int(node["t"]) == int(frame_t) and _node_in_crop(node, crop)]

    def _plot_detector_projection_row(fig, axes_row, raw_proj, heat_proj, mask_proj, crop_nodes, plane_name, crop, zdim):
        y0, y1, x0, x1 = crop
        raw_im = _safe_percentile_image(raw_proj)
        heat_im = _norm01(heat_proj)
        mask_im = (heat_im >= DETECTOR_THRESHOLD_VIEW).astype(np.float32)
        cm_heat = "magma"

        if plane_name == "XY":
            extent = [x0, x1, y1, y0]
            xs = [float(n["x"]) for n in crop_nodes]
            ys = [float(n["y"]) for n in crop_nodes]
            xlabel, ylabel = "x", "y"
        elif plane_name == "XZ":
            extent = [x0, x1, zdim, 0]
            xs = [float(n["x"]) for n in crop_nodes]
            ys = [float(n["z"]) for n in crop_nodes]
            xlabel, ylabel = "x", "z"
        else:  # YZ
            extent = [y0, y1, zdim, 0]
            xs = [float(n["y"]) for n in crop_nodes]
            ys = [float(n["z"]) for n in crop_nodes]
            xlabel, ylabel = "y", "z"

        titles = [
            f"{plane_name} raw image",
            f"{plane_name} detector heatmap",
            f"{plane_name} thresholded mask",
            f"{plane_name} extracted candidates",
        ]
        images = [raw_im, heat_im, mask_im, raw_im]
        cmaps = ["gray", cm_heat, "viridis", "gray"]
        for ax, img, title, cmap in zip(axes_row, images, titles, cmaps):
            ax.imshow(img, cmap=cmap, extent=extent, interpolation="nearest")
            if title.endswith("extracted candidates"):
                if xs:
                    ax.scatter(xs, ys, s=18, c="#41d6ff", linewidths=0, alpha=0.88)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    display(HTML("""
    <div style='border:1px solid #334155;border-radius:16px;padding:16px;background:#0b1220;color:#e2e8f0'>
      <h3 style='margin-top:0;color:#7dd3fc'>Detector localization laboratory</h3>
      <p>
      This section focuses only on the first learned step: how a 3D microscopy frame becomes candidate cell centers.
      The notebook tries to load a direct detector score volume. If the repository does not save one, it builds
      a pedagogical 3D heatmap surrogate from the raw detector nodes so that beginners can still understand the localization logic.
      </p>
    </div>
    """))

    raw_stage_name, raw_nodes_by_id, raw_edges = snapshots[0]
    frame_cache = {}
    frame = read_test_frame(visual_dataset, int(focus_t), frame_cache)
    raw_frame_nodes = _nodes_for_frame(raw_nodes_by_id, focus_t)
    raw_crop_nodes = _nodes_for_frame_and_crop(raw_nodes_by_id, focus_t, crop)
    score_volume, score_source = _try_load_detector_score_volume(visual_dataset, int(focus_t), frame.shape)
    if score_volume is None:
        score_volume = _gaussian_surrogate_from_nodes(frame.shape, raw_frame_nodes, DETECTOR_SIGMA_Z, DETECTOR_SIGMA_XY)
        score_source = (
            "surrogate detector heatmap built from raw candidate centroids; "
            "useful for teaching when the repo does not save dense UNet logits"
        )
    score_volume = _norm01(score_volume)

    _frame_crop, raw_xy, raw_xz, raw_yz = _projection_triplet(frame, crop)
    _, heat_xy, heat_xz, heat_yz = _projection_triplet(score_volume, crop)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    _plot_detector_projection_row(fig, axes[0], raw_xy, heat_xy, heat_xy, raw_crop_nodes, "XY", crop, frame.shape[0])
    _plot_detector_projection_row(fig, axes[1], raw_xz, heat_xz, heat_xz, raw_crop_nodes, "XZ", crop, frame.shape[0])
    _plot_detector_projection_row(fig, axes[2], raw_yz, heat_yz, heat_yz, raw_crop_nodes, "YZ", crop, frame.shape[0])
    fig.suptitle(
        f"{visual_dataset} — detector localization for t={focus_t}\n"
        f"Heatmap source: {score_source}\n"
        f"View threshold = {DETECTOR_THRESHOLD_VIEW:.2f}",
        fontsize=14,
        y=1.02,
    )
    detector_png = DETECTOR_LAB_DIR / f"{visual_dataset}_t{int(focus_t):03d}_detector_localization_overview.png"
    fig.savefig(detector_png, dpi=180, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    print("Saved:", detector_png)

    # A simpler one-row beginner summary for the most intuitive XY view.
    fig2, ax2 = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    y0, y1, x0, x1 = crop
    extent_xy = [x0, x1, y1, y0]
    heat_mask_xy = (_norm01(heat_xy) >= DETECTOR_THRESHOLD_VIEW).astype(np.float32)
    for ax, img, title, cmap in zip(
        ax2,
        [_safe_percentile_image(raw_xy), _norm01(heat_xy), heat_mask_xy, _safe_percentile_image(raw_xy)],
        ["Raw XY image", "UNet heatmap / surrogate", "Thresholded localization mask", "Final candidate centers"],
        ["gray", "magma", "viridis", "gray"],
    ):
        ax.imshow(img, cmap=cmap, extent=extent_xy, interpolation="nearest")
        if title == "Final candidate centers" and raw_crop_nodes:
            ax.scatter([float(n["x"]) for n in raw_crop_nodes], [float(n["y"]) for n in raw_crop_nodes], s=20, c="#41d6ff", linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig2.suptitle(
        "Beginner summary: image → heatmap → mask → centroids",
        fontsize=14,
        y=1.04,
    )
    detector_png_xy = DETECTOR_LAB_DIR / f"{visual_dataset}_t{int(focus_t):03d}_detector_xy_summary.png"
    fig2.savefig(detector_png_xy, dpi=180, bbox_inches="tight")
    display(fig2)
    plt.close(fig2)
    print("Saved:", detector_png_xy)

    crop_node_df = pd.DataFrame([
        {
            "node_id": int(n["node_id"]),
            "t": int(n["t"]),
            "z": round(float(n["z"]), 2),
            "y": round(float(n["y"]), 2),
            "x": round(float(n["x"]), 2),
        }
        for n in sorted(raw_crop_nodes, key=lambda n: int(n["node_id"]))[:DETECTOR_MAX_TABLE_ROWS]
    ])

    detector_summary = pd.DataFrame([{
        "dataset": visual_dataset,
        "focus_t": int(focus_t),
        "heatmap_source": score_source,
        "frame_shape_z_y_x": tuple(int(v) for v in frame.shape),
        "crop_y0_y1_x0_x1": tuple(int(v) for v in crop),
        "raw_candidates_in_frame": int(len(raw_frame_nodes)),
        "raw_candidates_in_crop": int(len(raw_crop_nodes)),
        "detector_threshold_for_view": float(DETECTOR_THRESHOLD_VIEW),
        "baseline_det_threshold": float(DET_THRESHOLD),
    }])
    display(HTML("<h4>Detector step summary</h4>"))
    display(detector_summary)
    display(HTML("<h4>Example candidate nodes extracted from the localization stage</h4>"))
    if len(crop_node_df):
        display(crop_node_df)
    else:
        print("No raw candidate nodes fell inside the selected crop.")

    detector_summary.to_csv(DETECTOR_LAB_DIR / f"{visual_dataset}_t{int(focus_t):03d}_detector_summary.csv", index=False)
    crop_node_df.to_csv(DETECTOR_LAB_DIR / f"{visual_dataset}_t{int(focus_t):03d}_candidate_nodes.csv", index=False)

    display(HTML(f"""
    <div style='border-left:4px solid #fbbf24;padding:12px 14px;background:#1f2937;color:#e5e7eb;margin-top:12px'>
      <b>How to read this detector lab</b><br>
      1. <b>Raw image</b>: what the network sees for one 3D frame.<br>
      2. <b>Heatmap</b>: where the detector believes cell centers are likely to exist.<br>
      3. <b>Thresholded mask</b>: a binary view of the confident localization region.<br>
      4. <b>Candidate centers</b>: the point coordinates that become graph nodes.<br><br>
      The competition submission does not require full segmentation masks. It only needs node coordinates and temporal edges.
      That is why the centroid view is the most important output of the detector stage.
    </div>
    """))
else:
    print("Detector localization laboratory is OFF because visualization is disabled.")

# ===== CELL 25 [markdown] =====
## Interactive 4D viewer and stage switcher

The earlier animation is useful, but it still feels like a 2D projection. This section adds two beginner-friendly tools:

1. **Interactive 3D + time viewer** — cells are shown inside a 3D physical space. A time slider plays the lineage graph through time. You can rotate, zoom, and inspect parent-to-daughter splits.
2. **Stage-by-stage transformation report** — every post-processing stage is summarized as data: how many nodes and edges remain, how many were added or removed, how many division-like sources exist, and why the stage is needed.

These outputs run only in visual mode. They are automatically skipped during Kaggle competition reruns, so the submission notebook still behaves like the original 0.897 baseline.

# ===== CELL 26 [code] =====
if RUN_PIPELINE_VISUALIZATION:
    from IPython.display import HTML, display
    import json as _json
    from collections import defaultdict, Counter

    PLOTLY_MAX_STAGE_NODES = int(os.environ.get("BIOHUB_PLOTLY_MAX_STAGE_NODES", "1600"))
    PLOTLY_MAX_STAGE_EDGES = int(os.environ.get("BIOHUB_PLOTLY_MAX_STAGE_EDGES", "2200"))
    PLOTLY_TIME_TRAIL = int(os.environ.get("BIOHUB_PLOTLY_TIME_TRAIL", "4"))
    PLOTLY_WINDOW_RADIUS = int(os.environ.get("BIOHUB_PLOTLY_WINDOW_RADIUS", str(max(6, VISUAL_ANIMATION_RADIUS))))
    PLOTLY_DISPLAY_INLINE = os.environ.get("BIOHUB_PLOTLY_DISPLAY_INLINE", "1") != "0"
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except Exception as _plotly_exc:
        go = None
        PLOTLY_AVAILABLE = False
        print(f"Plotly is not available, so interactive 3D viewers will be skipped: {_plotly_exc}")

    _STAGE_RATIONALE = {
        "00 raw detector + ILP graph": "Raw model output: high-threshold detections plus learned/ILP links. It is informative but still contains long jumps, duplicate parents, short noise tracks, and occasional missing-frame breaks.",
        "01 physical edge filter": "Removes edges that violate the physical motion gate or do not connect consecutive frames. This prevents impossible jumps from becoming false tracking edges.",
        "02 motion relink": "Rebuilds the main one-to-one motion chain with Hungarian matching. This is needed because real cells usually move smoothly across adjacent frames.",
        "03 parent repair": "Keeps one parent per target node. A cell at t+1 normally cannot have two different parents unless the graph has an error.",
        "04 one-frame gap close": "Repairs short missing detections by inserting or reusing a node between a track end and a nearby track start.",
        "05 strict gap2 recovery": "Optional conservative two-frame recovery. In the 0.897 baseline this is normally disabled, so it should usually change nothing.",
        "05 safe division recovery": "Adds only very local and capped parent-to-two-daughter splits. This recovers division recall without flooding the graph with false mitosis events.",
        "06 division geometry filter": "Optional geometry cleanup for implausible divisions. In the 0.897 baseline this is normally disabled.",
        "06 prune isolated nodes": "Deletes single-frame isolated detections that do not help edge Jaccard and may increase the node over-prediction penalty.",
        "07 short-track filter min6": "Deletes tiny non-division components shorter than six nodes. This is one of the key precision guards in the 0.897 baseline.",
        "08 final line-fit smoothing": "Moves internal track nodes toward a local straight-line fit. It keeps topology unchanged but can improve centroid matching.",
    }

    def _edge_key(edge):
        return (int(edge["source_id"]), int(edge["target_id"]))

    def _node_key(node_id):
        return int(node_id)

    def _incoming_outgoing_counts(edges):
        incoming = Counter()
        outgoing = Counter()
        for edge in edges:
            incoming[int(edge["target_id"])] += 1
            outgoing[int(edge["source_id"])] += 1
        return incoming, outgoing

    def _graph_component_count(nodes, edges):
        parent = {int(k): int(k) for k in nodes.keys()}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        for edge in edges:
            if int(edge["source_id"]) in nodes and int(edge["target_id"]) in nodes:
                union(edge["source_id"], edge["target_id"])
        return len({find(k) for k in parent.keys()}) if parent else 0

    def _stage_metrics(nodes, edges):
        incoming, outgoing = _incoming_outgoing_counts(edges)
        node_ids = set(int(k) for k in nodes.keys())
        isolated = sum(1 for nid in node_ids if incoming[nid] == 0 and outgoing[nid] == 0)
        starts = sum(1 for nid in node_ids if incoming[nid] == 0 and outgoing[nid] > 0)
        ends = sum(1 for nid in node_ids if incoming[nid] > 0 and outgoing[nid] == 0)
        division_sources = sum(1 for nid, c in outgoing.items() if c >= 2)
        multi_parent_targets = sum(1 for nid, c in incoming.items() if c >= 2)
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "components": _graph_component_count(nodes, edges),
            "isolated_nodes": isolated,
            "track_starts": starts,
            "track_ends": ends,
            "division_like_sources": division_sources,
            "multi_parent_targets": multi_parent_targets,
        }

    def _build_stage_transform_table(snapshots):
        rows = []
        prev_nodes = set()
        prev_edges = set()
        for idx, (stage, nodes, edges) in enumerate(snapshots):
            node_set = set(_node_key(k) for k in nodes.keys())
            edge_set = set(_edge_key(e) for e in edges)
            metrics = _stage_metrics(nodes, edges)
            row = {
                "step": idx,
                "stage": stage,
                **metrics,
                "nodes_added_vs_prev": len(node_set - prev_nodes) if idx else len(node_set),
                "nodes_removed_vs_prev": len(prev_nodes - node_set) if idx else 0,
                "edges_added_vs_prev": len(edge_set - prev_edges) if idx else len(edge_set),
                "edges_removed_vs_prev": len(prev_edges - edge_set) if idx else 0,
                "why_this_step_exists": _STAGE_RATIONALE.get(stage, "Post-processing stage used by the baseline output builder."),
            }
            rows.append(row)
            prev_nodes, prev_edges = node_set, edge_set
        return pd.DataFrame(rows)

    def _format_stage_cards(df):
        cards = []
        for _, row in df.iterrows():
            cards.append(f"""
            <div style='border:1px solid #263244;border-radius:14px;padding:14px;margin:10px 0;background:linear-gradient(135deg,#0b1020,#111827);color:#d9f7ff;font-family:Inter,Arial,sans-serif;'>
              <div style='font-size:15px;font-weight:700;color:#7dd3fc;'>Step {int(row['step'])}: {row['stage']}</div>
              <div style='display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 6px 0;font-size:12px;'>
                <span>nodes <b style='color:#ffffff'>{int(row['nodes'])}</b></span>
                <span>edges <b style='color:#ffffff'>{int(row['edges'])}</b></span>
                <span>components <b style='color:#ffffff'>{int(row['components'])}</b></span>
                <span>divisions <b style='color:#ffd166'>{int(row['division_like_sources'])}</b></span>
                <span>isolated <b style='color:#fb7185'>{int(row['isolated_nodes'])}</b></span>
              </div>
              <div style='font-size:12px;color:#a7f3d0;'>Δ nodes +{int(row['nodes_added_vs_prev'])} / -{int(row['nodes_removed_vs_prev'])}, Δ edges +{int(row['edges_added_vs_prev'])} / -{int(row['edges_removed_vs_prev'])}</div>
              <p style='font-size:12px;line-height:1.45;color:#cbd5e1;margin:8px 0 0 0;'>{row['why_this_step_exists']}</p>
            </div>
            """)
        return "<h3>Stage-by-stage data transformation</h3>" + "\n".join(cards)

    def _phys_xyz(node):
        return (
            float(node["x"]) * VOXEL_SCALE_UM[2],
            float(node["y"]) * VOXEL_SCALE_UM[1],
            float(node["z"]) * VOXEL_SCALE_UM[0],
        )

    def _node_inside_visual_window(node, frames, crop):
        return int(node["t"]) in frames and _node_in_crop(node, crop)

    def _subset_graph_for_4d(nodes, edges, focus_t, crop, radius, max_nodes, max_edges):
        min_t = min(int(n["t"]) for n in nodes.values()) if nodes else 0
        max_t = max(int(n["t"]) for n in nodes.values()) if nodes else 0
        t0 = max(min_t, int(focus_t) - radius)
        t1 = min(max_t, int(focus_t) + radius)
        frames = list(range(t0, t1 + 1))
        node_items = [(int(nid), node) for nid, node in nodes.items() if _node_inside_visual_window(node, set(frames), crop)]
        node_items = _sample_items(sorted(node_items, key=lambda kv: (int(kv[1]["t"]), int(kv[0]))), max_nodes)
        keep_ids = {int(nid) for nid, _ in node_items}
        subset_nodes = {int(nid): dict(node) for nid, node in node_items}
        subset_edges = []
        for edge in edges:
            sid = int(edge["source_id"]); tid = int(edge["target_id"])
            if sid in keep_ids and tid in keep_ids:
                subset_edges.append(dict(edge))
        subset_edges = _sample_items(subset_edges, max_edges)
        return frames, subset_nodes, subset_edges

    def _line_trace_for_edges(nodes, edges, current_t=None, trail=4, div_only=False):
        div_sources = _division_sources(edges)
        xs, ys, zs = [], [], []
        for edge in edges:
            sid = int(edge["source_id"]); tid = int(edge["target_id"])
            if sid not in nodes or tid not in nodes:
                continue
            s = nodes[sid]; t = nodes[tid]
            if current_t is not None:
                tt = int(t["t"])
                if not (int(current_t) - trail <= tt <= int(current_t)):
                    continue
            is_div = sid in div_sources
            if bool(div_only) != bool(is_div):
                continue
            sx, sy, sz = _phys_xyz(s)
            tx, ty, tz = _phys_xyz(t)
            xs += [sx, tx, None]
            ys += [sy, ty, None]
            zs += [sz, tz, None]
        return xs, ys, zs

    def _box_trace_from_nodes(nodes, name="visual crop box"):
        if not nodes:
            return None
        xs = [float(n["x"]) * VOXEL_SCALE_UM[2] for n in nodes.values()]
        ys = [float(n["y"]) * VOXEL_SCALE_UM[1] for n in nodes.values()]
        zs = [float(n["z"]) * VOXEL_SCALE_UM[0] for n in nodes.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        # Add padding so a nearly flat crop is still visible as a box.
        pad_x = max(2.0, 0.04 * max(1.0, xmax - xmin)); pad_y = max(2.0, 0.04 * max(1.0, ymax - ymin)); pad_z = max(2.0, 0.06 * max(1.0, zmax - zmin))
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
        zmin, zmax = zmin - pad_z, zmax + pad_z
        corners = [
            (xmin,ymin,zmin),(xmax,ymin,zmin),(xmax,ymax,zmin),(xmin,ymax,zmin),(xmin,ymin,zmin),
            (xmin,ymin,zmax),(xmax,ymin,zmax),(xmax,ymax,zmax),(xmin,ymax,zmax),(xmin,ymin,zmax),
            (xmax,ymin,zmax),(xmax,ymin,zmin),(xmax,ymax,zmin),(xmax,ymax,zmax),(xmin,ymax,zmax),(xmin,ymax,zmin)
        ]
        return go.Scatter3d(
            x=[c[0] for c in corners], y=[c[1] for c in corners], z=[c[2] for c in corners],
            mode="lines", name=name, line=dict(color="rgba(148,163,184,0.50)", width=3), hoverinfo="skip"
        )

    def _make_frame_traces(nodes, edges, frame_t, trail):
        div_sources = _division_sources(edges)
        current = [(nid, n) for nid, n in nodes.items() if int(n["t"]) == int(frame_t)]
        history = [(nid, n) for nid, n in nodes.items() if int(frame_t) - trail <= int(n["t"]) < int(frame_t)]
        normal_current = [(nid, n) for nid, n in current if nid not in div_sources]
        div_current = [(nid, n) for nid, n in current if nid in div_sources]
        def scatter_from(items, name, size, symbol="circle", opacity=0.92, color=None):
            xs, ys, zs, text = [], [], [], []
            for nid, node in items:
                x, y, z = _phys_xyz(node)
                xs.append(x); ys.append(y); zs.append(z)
                text.append(f"node={nid}<br>t={int(node['t'])}<br>z={float(node['z']):.1f}, y={float(node['y']):.1f}, x={float(node['x']):.1f}")
            marker = dict(size=size, opacity=opacity, symbol=symbol)
            if color is not None:
                marker["color"] = color
            return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name=name, marker=marker, text=text, hovertemplate="%{text}<extra></extra>")
        hx, hy, hz = [], [], []
        for _, node in history:
            x, y, z = _phys_xyz(node); hx.append(x); hy.append(y); hz.append(z)
        history_trace = go.Scatter3d(
            x=hx, y=hy, z=hz, mode="markers", name="recent past cells",
            marker=dict(size=3, opacity=0.22, color="rgba(148,163,184,0.35)"), hoverinfo="skip"
        )
        normal_edges = _line_trace_for_edges(nodes, edges, current_t=frame_t, trail=trail, div_only=False)
        div_edges = _line_trace_for_edges(nodes, edges, current_t=frame_t, trail=trail, div_only=True)
        normal_edge_trace = go.Scatter3d(x=normal_edges[0], y=normal_edges[1], z=normal_edges[2], mode="lines", name="recent tracking links", line=dict(color="rgba(255,94,168,0.42)", width=3), hoverinfo="skip")
        div_edge_trace = go.Scatter3d(x=div_edges[0], y=div_edges[1], z=div_edges[2], mode="lines", name="division links", line=dict(color="rgba(255,209,102,0.95)", width=7), hoverinfo="skip")
        return [history_trace, normal_edge_trace, div_edge_trace, scatter_from(normal_current, "current cells", 5, color="#41d6ff"), scatter_from(div_current, "division source at current t", 9, symbol="diamond", color="#ffd166")]

    def _make_interactive_4d_viewer(dataset, nodes, edges, focus_t, crop, radius):
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available, skipping 3D interactive viewer.")
            return None
        frames, subset_nodes, subset_edges = _subset_graph_for_4d(nodes, edges, focus_t, crop, radius, PLOTLY_MAX_STAGE_NODES, PLOTLY_MAX_STAGE_EDGES)
        if not frames or not subset_nodes:
            print("No nodes available for the 3D+time viewer in the selected crop.")
            return None
        first_t = frames[0]
        data = _make_frame_traces(subset_nodes, subset_edges, first_t, PLOTLY_TIME_TRAIL)
        box = _box_trace_from_nodes(subset_nodes)
        if box is not None:
            data.append(box)
        plotly_frames = []
        for t in frames:
            traces = _make_frame_traces(subset_nodes, subset_edges, t, PLOTLY_TIME_TRAIL)
            if box is not None:
                traces.append(box)
            plotly_frames.append(go.Frame(name=str(t), data=traces))
        fig = go.Figure(data=data, frames=plotly_frames)
        sliders = [{
            "active": 0,
            "currentvalue": {"prefix": "time frame t = ", "font": {"size": 14}},
            "pad": {"t": 46},
            "steps": [
                {"label": str(t), "method": "animate", "args": [[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
                for t in frames
            ],
        }]
        fig.update_layout(
            title=f"{dataset} — interactive 3D space + time lineage viewer",
            template="plotly_dark",
            height=760,
            margin=dict(l=0, r=0, t=52, b=0),
            scene=dict(
                xaxis_title="x physical position (µm)",
                yaxis_title="y physical position (µm)",
                zaxis_title="z physical position (µm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.65, y=1.65, z=1.15)),
            ),
            legend=dict(orientation="h", y=1.03, x=0),
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {"label": "▶ Play time", "method": "animate", "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True, "transition": {"duration": 100}}]},
                    {"label": "❚❚ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }],
            sliders=sliders,
            annotations=[dict(
                text="Rotate with mouse. The slider is time. Pink lines are recent tracking links; yellow lines are parent-to-daughter division links.",
                x=0.01, y=0.965, xref="paper", yref="paper", showarrow=False, align="left", font=dict(size=12, color="#cbd5e1")
            )],
        )
        html_path = VISUAL_DIR / f"{dataset}_interactive_3d_time_lineage.html"
        html_path.write_text(fig.to_html(include_plotlyjs=True, full_html=True), encoding="utf-8")
        display(HTML("<h3>Interactive 3D + time lineage viewer</h3><p>This is the closest visual analogy to the real 4D task: x/y/z are space, and the slider is time.</p>"))
        if PLOTLY_DISPLAY_INLINE:
            display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))
        print(f"Saved interactive 3D+time viewer: {html_path}")
        return html_path

    def _make_stage_switcher_3d(dataset, snapshots, focus_t, crop):
        if not PLOTLY_AVAILABLE:
            print("Plotly is not available, skipping stage switcher.")
            return None
        all_stage_traces = []
        stage_trace_counts = []
        all_subset_nodes = {}
        for stage, nodes, edges in snapshots:
            _, subset_nodes, subset_edges = _subset_graph_for_4d(nodes, edges, focus_t, crop, radius=1, max_nodes=PLOTLY_MAX_STAGE_NODES, max_edges=PLOTLY_MAX_STAGE_EDGES)
            all_subset_nodes.update(subset_nodes)
            div_sources = _division_sources(subset_edges)
            cur_nodes = [(nid, n) for nid, n in subset_nodes.items() if abs(int(n["t"]) - int(focus_t)) <= 1]
            normal_nodes = [(nid, n) for nid, n in cur_nodes if nid not in div_sources]
            div_nodes = [(nid, n) for nid, n in cur_nodes if nid in div_sources]
            def scatter(items, name, color, size, symbol="circle"):
                xs, ys, zs, text = [], [], [], []
                for nid, node in items:
                    x, y, z = _phys_xyz(node); xs.append(x); ys.append(y); zs.append(z)
                    text.append(f"{stage}<br>node={nid}<br>t={int(node['t'])}")
                return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name=name, marker=dict(size=size, color=color, opacity=0.88, symbol=symbol), text=text, hovertemplate="%{text}<extra></extra>", visible=False)
            normal_edges = _line_trace_for_edges(subset_nodes, subset_edges, current_t=focus_t+1, trail=2, div_only=False)
            div_edges = _line_trace_for_edges(subset_nodes, subset_edges, current_t=focus_t+1, trail=2, div_only=True)
            traces = [
                go.Scatter3d(x=normal_edges[0], y=normal_edges[1], z=normal_edges[2], mode="lines", name="normal links", line=dict(color="rgba(255,94,168,0.45)", width=4), hoverinfo="skip", visible=False),
                go.Scatter3d(x=div_edges[0], y=div_edges[1], z=div_edges[2], mode="lines", name="division links", line=dict(color="rgba(255,209,102,0.95)", width=8), hoverinfo="skip", visible=False),
                scatter(normal_nodes, "cells", "#41d6ff", 5),
                scatter(div_nodes, "division source cells", "#ffd166", 9, symbol="diamond"),
            ]
            all_stage_traces.extend(traces)
            stage_trace_counts.append(len(traces))
        if not all_stage_traces:
            print("No traces available for stage switcher.")
            return None
        for i in range(stage_trace_counts[0]):
            all_stage_traces[i].visible = True
        fig = go.Figure(data=all_stage_traces)
        buttons = []
        offset = 0
        for idx, ((stage, _, _), n_traces) in enumerate(zip(snapshots, stage_trace_counts)):
            visible = [False] * len(all_stage_traces)
            for j in range(offset, offset + n_traces):
                visible[j] = True
            buttons.append({
                "label": stage,
                "method": "update",
                "args": [{"visible": visible}, {"title": f"{dataset} — stage view: {stage}"}],
            })
            offset += n_traces
        box = _box_trace_from_nodes(all_subset_nodes)
        if box is not None:
            fig.add_trace(box)
            # Keep the box visible for all buttons.
            for button in buttons:
                button["args"][0]["visible"].append(True)
        fig.update_layout(
            title=f"{dataset} — stage view: {snapshots[0][0]}",
            template="plotly_dark",
            height=720,
            margin=dict(l=0, r=0, t=78, b=0),
            scene=dict(
                xaxis_title="x physical position (µm)", yaxis_title="y physical position (µm)", zaxis_title="z physical position (µm)", aspectmode="data",
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.1)),
            ),
            updatemenus=[{"buttons": buttons, "direction": "down", "x": 0.01, "y": 1.10, "showactive": True}],
            annotations=[dict(text="Use the dropdown to see what the graph looks like after each operation. Same crop, same focus time.", x=0.01, y=1.02, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="#cbd5e1"))],
            legend=dict(orientation="h", y=1.0, x=0.35),
        )
        html_path = VISUAL_DIR / f"{dataset}_stage_by_stage_3d_switcher.html"
        html_path.write_text(fig.to_html(include_plotlyjs=True, full_html=True), encoding="utf-8")
        display(HTML("<h3>Interactive stage-by-stage 3D switcher</h3><p>Change the dropdown to observe how each post-processing operation transforms the graph.</p>"))
        if PLOTLY_DISPLAY_INLINE:
            display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))
        print(f"Saved stage-by-stage 3D switcher: {html_path}")
        return html_path

    display(HTML("<h2>v032 interactive 4D understanding layer</h2>"))
    transform_df = _build_stage_transform_table(snapshots)
    transform_csv = VISUAL_DIR / f"{visual_dataset}_stage_transform_report.csv"
    transform_df.to_csv(transform_csv, index=False)
    display(transform_df[[
        "step", "stage", "nodes", "edges", "nodes_added_vs_prev", "nodes_removed_vs_prev",
        "edges_added_vs_prev", "edges_removed_vs_prev", "components", "isolated_nodes",
        "division_like_sources", "multi_parent_targets"
    ]])
    display(HTML(_format_stage_cards(transform_df)))
    print(f"Saved stage transform report: {transform_csv}")
    _make_interactive_4d_viewer(visual_dataset, final_nodes, final_edges, focus_t, crop, PLOTLY_WINDOW_RADIUS)
    _make_stage_switcher_3d(visual_dataset, snapshots, focus_t, crop)
else:
    print("v032 interactive 4D viewer is OFF because pipeline visualization is OFF.")

# ===== CELL 27 [markdown] =====
## v035 — Beginner Pipeline Laboratory + Detector Heatmaps

A beginner should not have to infer what each operation does from one final animation.

This laboratory treats every stage as a small transformation:

```text
input graph
    │
    ├─ operation and decision rule
    │
    ├─ tunable hyperparameters
    │
    └─ added / removed / modified nodes and edges
    ▼
output graph
```

For each stage the notebook reports:

1. **Input:** what data enters the stage.
2. **Operation:** what rule is applied.
3. **Output:** what changes.
4. **Baseline status:** enabled or disabled in the original 0.897 configuration.
5. **Necessity:** whether it is mathematically required, required to reproduce this baseline, or only an optional experiment.
6. **Hyperparameters:** the exact environment variables that can be changed.
7. **Failure mode:** what can go wrong when the step is too weak or too aggressive.
8. **Visualization:** unchanged objects, removed objects and added objects in the same 3D crop.

The word **required** is used carefully. Most post-processing stages are not mathematically mandatory for cell tracking. They are enabled because they improved the behavior of this specific 0.897 solution.

# ===== CELL 28 [code] =====
if RUN_PIPELINE_VISUALIZATION:
    from IPython.display import HTML, display
    import html as _html
    from collections import defaultdict, Counter

    BEGINNER_LAB_DIR = VISUAL_DIR / "beginner_pipeline_lab"
    BEGINNER_LAB_DIR.mkdir(parents=True, exist_ok=True)

    _stage_catalog = {
        "00 raw detector + ILP graph": {
            "input": "4D microscopy voxels plus the trained checkpoint",
            "operation": "The UNet-style detector proposes cell centers. The learned edge predictor scores links. ILP selects a candidate graph.",
            "output": "Raw nodes with t/z/y/x coordinates and candidate temporal edges.",
            "baseline_status": "Enabled",
            "necessity": "Core learned stage. A tracker needs detections and candidate links, although a different model could replace it.",
            "toggle": "BIOHUB_USE_ILP",
            "hyperparameters": [
                ("BIOHUB_DET_THRESHOLD", DET_THRESHOLD, "Higher values reduce candidate nodes but may miss cells."),
                ("BIOHUB_UNET_BATCH_SIZE", UNET_BATCH_SIZE, "Memory/runtime control; it should not intentionally change predictions."),
                ("BIOHUB_ILP_EDGE_WEIGHT", ILP_EDGE_WEIGHT, "Relative cost of selecting temporal links."),
                ("BIOHUB_ILP_APPEARANCE_WEIGHT", ILP_APPEARANCE_WEIGHT, "Cost for starting tracks."),
                ("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", ILP_DISAPPEARANCE_WEIGHT, "Cost for ending tracks."),
                ("BIOHUB_ILP_DIVISION_WEIGHT", ILP_DIVISION_WEIGHT, "Relative division-event cost."),
            ],
            "too_weak": "Too many false candidate nodes or fragmented links.",
            "too_strong": "Real cells and real divisions may be removed before graph repair can recover them.",
        },
        "01 physical edge filter": {
            "input": "Raw candidate nodes and learned/ILP edges.",
            "operation": "Reject non-consecutive links and links longer than the physical motion gate.",
            "output": "A graph without impossible temporal jumps.",
            "baseline_status": "Enabled",
            "necessity": "Required to reproduce the 0.897 baseline; conceptually replaceable by an equally strong motion prior.",
            "toggle": "BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_EDGE_MAX_UM", OUTPUT_EDGE_MAX_UM, "Maximum allowed physical displacement."),
                ("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", OUTPUT_ENFORCE_NEXT_FRAME, "Require t → t+1 edges."),
            ],
            "too_weak": "Long false links survive and become edge false positives.",
            "too_strong": "Fast-moving real cells are disconnected.",
        },
        "02 motion relink": {
            "input": "Filtered nodes, learned edge probabilities and recent motion history.",
            "operation": "Hungarian matching reconstructs the main one-to-one chain using predicted motion, raw distance and learned-edge bonus.",
            "output": "A smoother adjacent-frame motion graph.",
            "baseline_status": "Enabled",
            "necessity": "A major baseline stage, but not mathematically mandatory. Another association algorithm could replace it.",
            "toggle": "BIOHUB_OUTPUT_MOTION_RELINK",
            "hyperparameters": [
                ("BIOHUB_MOTION_RELINK_TIGHT_UM", MOTION_RELINK_TIGHT_UM, "First-pass distance gate."),
                ("BIOHUB_MOTION_RELINK_RELAXED_UM", MOTION_RELINK_RELAXED_UM, "Second-pass fallback gate."),
                ("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", MOTION_RELINK_VELOCITY_WEIGHT, "Strength of constant-velocity prediction."),
                ("BIOHUB_MOTION_RELINK_LEARNED_BONUS", MOTION_RELINK_LEARNED_BONUS, "Reward for a high neural edge score."),
                ("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", MOTION_RELINK_MAX_FRAME_NODES, "Safety cap for dense frames."),
            ],
            "too_weak": "Tracks remain noisy or fragmented.",
            "too_strong": "Nearby cells can be swapped when motion prediction dominates appearance evidence.",
        },
        "03 parent repair": {
            "input": "Motion-linked graph that may contain multiple parents for one target.",
            "operation": "Keep the best incoming edge for each ordinary target node.",
            "output": "At most one parent per target.",
            "baseline_status": "Enabled",
            "necessity": "Required to reproduce the baseline and biologically natural for ordinary cell continuation.",
            "toggle": "BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", OUTPUT_SINGLE_PARENT_REPAIR, "Enable one-parent repair."),
                ("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", OUTPUT_SINGLE_CHILD_REPAIR, "Normally disabled because divisions need two children."),
            ],
            "too_weak": "Impossible merges inflate edge false positives.",
            "too_strong": "A generic one-child repair would erase true divisions.",
        },
        "04 one-frame gap close": {
            "input": "Track ends at t and track starts at t+2, with a possible missed cell at t+1.",
            "operation": "Reuse a nearby isolated node or insert a synthetic midpoint and optionally refine it from image intensity.",
            "output": "A repaired t → t+1 → t+2 segment.",
            "baseline_status": "Enabled",
            "necessity": "Optional in principle, enabled in the 0.897 baseline because one-frame misses are common.",
            "toggle": "BIOHUB_OUTPUT_GAP_CLOSE",
            "hyperparameters": [
                ("BIOHUB_GAP_CLOSE_MAX_GAP", GAP_CLOSE_MAX_GAP, "Number of missing frames; baseline uses one."),
                ("BIOHUB_GAP_CLOSE_UM", GAP_CLOSE_UM, "Maximum end-to-start distance."),
                ("BIOHUB_GAP_CLOSE_REUSE_UM", GAP_CLOSE_REUSE_UM, "Radius for reusing an existing node."),
                ("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", GAP_CLOSE_MAX_ADDED_FRAC, "Fractional cap on inserted nodes."),
                ("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", GAP_CLOSE_MAX_ADDED_ABS, "Absolute cap on inserted nodes."),
                ("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", GAP_REFINE_MAX_SHIFT_UM, "Maximum image-refinement shift."),
            ],
            "too_weak": "Real tracks remain broken by one missed frame.",
            "too_strong": "Synthetic nodes and false bridges increase node and edge false positives.",
        },
        "05 strict gap2 recovery": {
            "input": "Track ends and starts separated by two missing frames.",
            "operation": "Attempt an additional conservative two-frame bridge.",
            "output": "Potential t → t+1 → t+2 → t+3 recovery.",
            "baseline_status": "Disabled",
            "necessity": "Not required for the original 0.897 baseline; included only as an experiment hook.",
            "toggle": "BIOHUB_OUTPUT_GAP2_RECOVERY",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_GAP2_RECOVERY", OUTPUT_GAP2_RECOVERY, "Master switch."),
                ("BIOHUB_GAP2_MAX_TOTAL_UM", GAP2_MAX_TOTAL_UM, "Maximum total displacement."),
                ("BIOHUB_GAP2_MAX_STEP_UM", GAP2_MAX_STEP_UM, "Maximum interpolated step."),
                ("BIOHUB_GAP2_MAX_LINKS_FRAC", GAP2_MAX_LINKS_FRAC, "Fractional recovery cap."),
                ("BIOHUB_GAP2_MAX_LINKS_ABS", GAP2_MAX_LINKS_ABS, "Absolute recovery cap."),
            ],
            "too_weak": "Longer missing segments remain fragmented.",
            "too_strong": "Ambiguous cells are bridged over too much time.",
        },
        "05 safe division recovery": {
            "input": "A parent with one linked child plus a nearby unmatched candidate child.",
            "operation": "Add a second outgoing edge only when parent/child and sister geometry pass strict gates and caps.",
            "output": "A parent → two daughters division event.",
            "baseline_status": "Enabled",
            "necessity": "Needed to recover division recall after one-to-one motion matching; replaceable by a stronger learned division model.",
            "toggle": "BIOHUB_OUTPUT_SAFE_DIVISIONS",
            "hyperparameters": [
                ("BIOHUB_SAFE_DIV_MAX_UM", SAFE_DIV_MAX_UM, "Parent-to-new-daughter gate."),
                ("BIOHUB_SAFE_DIV_SISTER_MAX_UM", SAFE_DIV_SISTER_MAX_UM, "Maximum daughter-to-daughter distance."),
                ("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", SAFE_DIV_EXISTING_CHILD_MAX_UM, "Parent-to-existing-child gate."),
                ("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", SAFE_DIV_FRAME_FRAC_CAP, "Per-frame division cap."),
                ("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", SAFE_DIV_GLOBAL_FRAC_CAP, "Global division cap."),
            ],
            "too_weak": "True mitoses are missed.",
            "too_strong": "Ordinary nearby cells are incorrectly labeled as daughters.",
        },
        "06 division geometry filter": {
            "input": "Predicted division branches.",
            "operation": "Optionally reject branches with implausible parent/daughter or sister geometry.",
            "output": "A more conservative division graph.",
            "baseline_status": "Disabled",
            "necessity": "Not required for the original 0.897 baseline.",
            "toggle": "BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", OUTPUT_DIVISION_GEOMETRY_FILTER, "Master switch."),
                ("BIOHUB_DIV_PARENT_MAX_UM", DIV_PARENT_MAX_UM, "Parent-to-daughter gate."),
                ("BIOHUB_DIV_SISTER_MAX_UM", DIV_SISTER_MAX_UM, "Daughter separation gate."),
                ("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", DIV_DROP_TO_SINGLE_IF_BAD, "Keep one child if a split is rejected."),
            ],
            "too_weak": "Implausible divisions survive.",
            "too_strong": "Real asymmetric divisions are deleted.",
        },
        "06 prune isolated nodes": {
            "input": "Graph containing detections with no incoming or outgoing edge.",
            "operation": "Delete isolated single-frame nodes.",
            "output": "Only nodes participating in a trajectory or lineage remain.",
            "baseline_status": "Enabled",
            "necessity": "A precision guard in this baseline; not mathematically required.",
            "toggle": "BIOHUB_OUTPUT_PRUNE_ISOLATED",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_PRUNE_ISOLATED", OUTPUT_PRUNE_ISOLATED, "Master switch."),
            ],
            "too_weak": "Unhelpful detections increase the node over-prediction penalty.",
            "too_strong": "A valid one-frame labeled cell could be removed, although such nodes do not contribute an edge.",
        },
        "07 short-track filter min6": {
            "input": "Connected components representing tracks and small lineages.",
            "operation": "Delete non-division components shorter than six nodes.",
            "output": "A cleaner graph dominated by persistent trajectories.",
            "baseline_status": "Enabled",
            "necessity": "A key precision stage in the 0.897 baseline, but the threshold is empirical rather than universal.",
            "toggle": "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_MIN_TRACK_LEN", OUTPUT_MIN_TRACK_LEN, "Minimum component length."),
                ("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", OUTPUT_KEEP_DIVISION_COMPONENTS, "Preserve short components containing a division."),
            ],
            "too_weak": "Short false tracks survive.",
            "too_strong": "Short but real trajectories are removed.",
        },
        "08 final line-fit smoothing": {
            "input": "Final graph topology and slightly noisy centroid coordinates.",
            "operation": "Fit a local line through ordinary track nodes and blend coordinates toward the fit.",
            "output": "The same graph topology with smoother centroids.",
            "baseline_status": "Enabled",
            "necessity": "Optional coordinate refinement; it does not create or delete lineage edges.",
            "toggle": "BIOHUB_OUTPUT_LINEFIT_SMOOTH",
            "hyperparameters": [
                ("BIOHUB_OUTPUT_LINEFIT_WEIGHT", OUTPUT_LINEFIT_WEIGHT, "Blend weight toward the fitted line."),
                ("BIOHUB_OUTPUT_LINEFIT_WINDOW", OUTPUT_LINEFIT_WINDOW, "Number of neighboring time points."),
            ],
            "too_weak": "Centroid jitter remains.",
            "too_strong": "Curved or accelerating trajectories can be over-smoothed.",
        },
    }

    def _edge_tuple_set(edges):
        return {(int(e["source_id"]), int(e["target_id"])) for e in edges}

    def _sample_node_rows(nodes, ids, limit=6):
        rows = []
        for node_id in sorted(ids)[:limit]:
            node = nodes.get(int(node_id))
            if node is None:
                continue
            rows.append({
                "node_id": int(node_id),
                "t": int(node["t"]),
                "z": round(float(node["z"]), 2),
                "y": round(float(node["y"]), 2),
                "x": round(float(node["x"]), 2),
            })
        return rows

    def _sample_edge_rows(edges, edge_keys, limit=8):
        by_key = {
            (int(e["source_id"]), int(e["target_id"])): e
            for e in edges
        }
        rows = []
        for key in sorted(edge_keys)[:limit]:
            edge = by_key.get(key, {})
            rows.append({
                "source_id": int(key[0]),
                "target_id": int(key[1]),
                "distance_um": round(float(edge.get("distance_um", float("nan"))), 3)
                if edge.get("distance_um") is not None else None,
                "edge_prob": round(float(edge.get("edge_prob")), 4)
                if edge.get("edge_prob") is not None else None,
            })
        return rows

    def _params_html(items):
        if not items:
            return "<i>No exposed hyperparameters.</i>"
        body = []
        for name, value, meaning in items:
            body.append(
                f"<tr><td><code>{_html.escape(str(name))}</code></td>"
                f"<td>{_html.escape(str(value))}</td>"
                f"<td>{_html.escape(str(meaning))}</td></tr>"
            )
        return (
            "<table style='width:100%;border-collapse:collapse;font-size:11px'>"
            "<tr><th style='text-align:left'>Environment variable</th>"
            "<th style='text-align:left'>Current value</th>"
            "<th style='text-align:left'>Meaning</th></tr>"
            + "".join(body) + "</table>"
        )

    def _small_table_html(rows, empty_text):
        if not rows:
            return f"<div style='color:#94a3b8;font-size:11px'>{_html.escape(empty_text)}</div>"
        return pd.DataFrame(rows).to_html(index=False, border=0, classes="dataframe")

    beginner_stage_rows = []
    beginner_stage_details = []
    for stage_idx, (stage_name, output_nodes, output_edges) in enumerate(snapshots):
        guide = _stage_catalog.get(stage_name, {
            "input": "Previous graph snapshot",
            "operation": "Repository-specific graph transformation.",
            "output": "Updated graph snapshot.",
            "baseline_status": "Unknown",
            "necessity": "Inspect source code before changing this stage.",
            "toggle": "",
            "hyperparameters": [],
            "too_weak": "Unknown.",
            "too_strong": "Unknown.",
        })
        if stage_idx == 0:
            input_nodes, input_edges = output_nodes, output_edges
        else:
            _, input_nodes, input_edges = snapshots[stage_idx - 1]

        input_node_ids = set(map(int, input_nodes))
        output_node_ids = set(map(int, output_nodes))
        input_edge_ids = _edge_tuple_set(input_edges)
        output_edge_ids = _edge_tuple_set(output_edges)

        added_nodes = output_node_ids - input_node_ids
        removed_nodes = input_node_ids - output_node_ids
        added_edges = output_edge_ids - input_edge_ids
        removed_edges = input_edge_ids - output_edge_ids

        input_metrics = _stage_metrics(input_nodes, input_edges)
        output_metrics = _stage_metrics(output_nodes, output_edges)

        row = {
            "step": stage_idx,
            "stage": stage_name,
            "baseline_status": guide["baseline_status"],
            "necessity": guide["necessity"],
            "input_nodes": input_metrics["nodes"],
            "output_nodes": output_metrics["nodes"],
            "nodes_added": len(added_nodes),
            "nodes_removed": len(removed_nodes),
            "input_edges": input_metrics["edges"],
            "output_edges": output_metrics["edges"],
            "edges_added": len(added_edges),
            "edges_removed": len(removed_edges),
            "input_divisions": input_metrics["division_like_sources"],
            "output_divisions": output_metrics["division_like_sources"],
            "toggle": guide["toggle"],
        }
        beginner_stage_rows.append(row)

        detail = {
            **row,
            "input_description": guide["input"],
            "operation": guide["operation"],
            "output_description": guide["output"],
            "too_weak": guide["too_weak"],
            "too_strong": guide["too_strong"],
            "hyperparameters": [
                {"name": name, "current_value": str(value), "meaning": meaning}
                for name, value, meaning in guide["hyperparameters"]
            ],
            "sample_added_nodes": _sample_node_rows(output_nodes, added_nodes),
            "sample_removed_nodes": _sample_node_rows(input_nodes, removed_nodes),
            "sample_added_edges": _sample_edge_rows(output_edges, added_edges),
            "sample_removed_edges": _sample_edge_rows(input_edges, removed_edges),
        }
        beginner_stage_details.append(detail)

        card = f"""
        <div style='border:1px solid #334155;border-radius:18px;padding:16px;margin:16px 0;background:linear-gradient(145deg,#07111f,#111827);color:#e2e8f0;font-family:Inter,Arial,sans-serif'>
          <div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap'>
            <div>
              <div style='font-size:11px;color:#94a3b8'>STEP {stage_idx}</div>
              <div style='font-size:19px;font-weight:800;color:#7dd3fc'>{_html.escape(stage_name)}</div>
            </div>
            <div style='font-size:12px;border:1px solid #475569;border-radius:999px;padding:6px 10px;height:max-content'>
              Baseline: <b>{_html.escape(guide["baseline_status"])}</b>
            </div>
          </div>

          <div style='display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:stretch;margin-top:14px'>
            <div style='background:#0f172a;border-radius:12px;padding:12px'>
              <div style='color:#a5b4fc;font-weight:700'>INPUT</div>
              <p style='font-size:12px;line-height:1.5'>{_html.escape(guide["input"])}</p>
              <div style='font-size:12px'>nodes <b>{input_metrics["nodes"]}</b> · edges <b>{input_metrics["edges"]}</b> · divisions <b>{input_metrics["division_like_sources"]}</b></div>
            </div>
            <div style='display:flex;align-items:center;font-size:28px;color:#fbbf24'>→</div>
            <div style='background:#0f172a;border-radius:12px;padding:12px'>
              <div style='color:#86efac;font-weight:700'>OUTPUT</div>
              <p style='font-size:12px;line-height:1.5'>{_html.escape(guide["output"])}</p>
              <div style='font-size:12px'>nodes <b>{output_metrics["nodes"]}</b> · edges <b>{output_metrics["edges"]}</b> · divisions <b>{output_metrics["division_like_sources"]}</b></div>
            </div>
          </div>

          <div style='background:#172033;border-radius:12px;padding:12px;margin-top:12px'>
            <div style='color:#fcd34d;font-weight:700'>OPERATION</div>
            <p style='font-size:12px;line-height:1.5;margin-bottom:5px'>{_html.escape(guide["operation"])}</p>
            <div style='font-size:12px;color:#a7f3d0'>Δ nodes +{len(added_nodes)} / -{len(removed_nodes)} · Δ edges +{len(added_edges)} / -{len(removed_edges)}</div>
          </div>

          <details style='margin-top:12px'>
            <summary style='cursor:pointer;color:#c4b5fd;font-weight:700'>Is this step necessary?</summary>
            <p style='font-size:12px;line-height:1.5'>{_html.escape(guide["necessity"])}</p>
            <p style='font-size:12px'><b>If too weak:</b> {_html.escape(guide["too_weak"])}</p>
            <p style='font-size:12px'><b>If too aggressive:</b> {_html.escape(guide["too_strong"])}</p>
          </details>

          <details style='margin-top:10px'>
            <summary style='cursor:pointer;color:#93c5fd;font-weight:700'>Tunable hyperparameters</summary>
            <div style='margin-top:8px'>{_params_html(guide["hyperparameters"])}</div>
          </details>

          <details style='margin-top:10px'>
            <summary style='cursor:pointer;color:#f9a8d4;font-weight:700'>Example rows added or removed</summary>
            <div style='display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:12px;margin-top:10px'>
              <div><b style='color:#86efac'>Added nodes</b>{_small_table_html(detail["sample_added_nodes"], "No nodes added.")}</div>
              <div><b style='color:#fb7185'>Removed nodes</b>{_small_table_html(detail["sample_removed_nodes"], "No nodes removed.")}</div>
              <div><b style='color:#86efac'>Added edges</b>{_small_table_html(detail["sample_added_edges"], "No edges added.")}</div>
              <div><b style='color:#fb7185'>Removed edges</b>{_small_table_html(detail["sample_removed_edges"], "No edges removed.")}</div>
            </div>
          </details>
        </div>
        """
        display(HTML(card))

    beginner_stage_df = pd.DataFrame(beginner_stage_rows)
    display(HTML("<h3>Compact stage ledger</h3>"))
    display(beginner_stage_df)

    beginner_stage_df.to_csv(
        BEGINNER_LAB_DIR / f"{visual_dataset}_beginner_stage_ledger.csv",
        index=False,
    )
    (BEGINNER_LAB_DIR / f"{visual_dataset}_beginner_stage_details.json").write_text(
        json.dumps(beginner_stage_details, indent=2),
        encoding="utf-8",
    )

    def _edge_xyz_for_keys(nodes, keys):
        xs, ys, zs = [], [], []
        for source_id, target_id in sorted(keys):
            source = nodes.get(int(source_id))
            target = nodes.get(int(target_id))
            if source is None or target is None:
                continue
            sx, sy, sz = _phys_xyz(source)
            tx, ty, tz = _phys_xyz(target)
            xs.extend([sx, tx, None])
            ys.extend([sy, ty, None])
            zs.extend([sz, tz, None])
        return xs, ys, zs

    def _node_scatter_for_ids(nodes, ids, name, color, symbol, size, visible=False):
        xs, ys, zs, texts = [], [], [], []
        for node_id in sorted(ids):
            node = nodes.get(int(node_id))
            if node is None:
                continue
            x, y, z = _phys_xyz(node)
            xs.append(x); ys.append(y); zs.append(z)
            texts.append(
                f"{name}<br>node={node_id}<br>t={int(node['t'])}"
                f"<br>z={float(node['z']):.2f}, y={float(node['y']):.2f}, x={float(node['x']):.2f}"
            )
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            name=name,
            marker=dict(size=size, color=color, symbol=symbol, opacity=0.90),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            visible=visible,
        )

    def _make_transformation_delta_lab(dataset, snapshots, focus_t, crop):
        if not PLOTLY_AVAILABLE or len(snapshots) < 2:
            print("Transformation delta lab skipped because Plotly or snapshots are unavailable.")
            return None

        all_traces = []
        trace_ranges = []
        buttons = []
        shared_nodes_for_box = {}

        for transition_idx in range(1, len(snapshots)):
            before_name, before_nodes_all, before_edges_all = snapshots[transition_idx - 1]
            after_name, after_nodes_all, after_edges_all = snapshots[transition_idx]

            _, before_nodes, before_edges = _subset_graph_for_4d(
                before_nodes_all, before_edges_all, focus_t, crop,
                radius=1,
                max_nodes=PLOTLY_MAX_STAGE_NODES,
                max_edges=PLOTLY_MAX_STAGE_EDGES,
            )
            _, after_nodes, after_edges = _subset_graph_for_4d(
                after_nodes_all, after_edges_all, focus_t, crop,
                radius=1,
                max_nodes=PLOTLY_MAX_STAGE_NODES,
                max_edges=PLOTLY_MAX_STAGE_EDGES,
            )
            shared_nodes_for_box.update(before_nodes)
            shared_nodes_for_box.update(after_nodes)

            before_node_ids = set(before_nodes)
            after_node_ids = set(after_nodes)
            before_edge_ids = _edge_tuple_set(before_edges)
            after_edge_ids = _edge_tuple_set(after_edges)

            unchanged_nodes = before_node_ids & after_node_ids
            removed_nodes = before_node_ids - after_node_ids
            added_nodes = after_node_ids - before_node_ids
            unchanged_edges = before_edge_ids & after_edge_ids
            removed_edges = before_edge_ids - after_edge_ids
            added_edges = after_edge_ids - before_edge_ids

            ux, uy, uz = _edge_xyz_for_keys(after_nodes, unchanged_edges)
            rx, ry, rz = _edge_xyz_for_keys(before_nodes, removed_edges)
            ax, ay, az = _edge_xyz_for_keys(after_nodes, added_edges)

            start = len(all_traces)
            all_traces.extend([
                go.Scatter3d(
                    x=ux, y=uy, z=uz, mode="lines",
                    name="unchanged edges",
                    line=dict(color="rgba(148,163,184,0.34)", width=3),
                    hoverinfo="skip", visible=False,
                ),
                go.Scatter3d(
                    x=rx, y=ry, z=rz, mode="lines",
                    name="removed edges",
                    line=dict(color="rgba(251,113,133,0.96)", width=7),
                    hoverinfo="skip", visible=False,
                ),
                go.Scatter3d(
                    x=ax, y=ay, z=az, mode="lines",
                    name="added edges",
                    line=dict(color="rgba(134,239,172,0.96)", width=7),
                    hoverinfo="skip", visible=False,
                ),
                _node_scatter_for_ids(
                    after_nodes, unchanged_nodes, "unchanged nodes",
                    "#67e8f9", "circle", 4, visible=False
                ),
                _node_scatter_for_ids(
                    before_nodes, removed_nodes, "removed nodes",
                    "#fb7185", "x", 8, visible=False
                ),
                _node_scatter_for_ids(
                    after_nodes, added_nodes, "added nodes",
                    "#86efac", "diamond", 8, visible=False
                ),
            ])
            end = len(all_traces)
            trace_ranges.append((start, end))

            guide = _stage_catalog.get(after_name, {})
            buttons.append({
                "label": f"{transition_idx:02d} · {after_name}",
                "method": "update",
                "args": [
                    {"visible": []},  # filled after optional box trace is known
                    {
                        "title": (
                            f"{dataset} — transformation into {after_name}"
                            f"<br><sup>green = added, red = removed, cyan/gray = unchanged</sup>"
                        ),
                        "annotations": [{
                            "text": (
                                f"<b>Input:</b> {_html.escape(str(guide.get('input', before_name)))}"
                                f"<br><b>Operation:</b> {_html.escape(str(guide.get('operation', after_name)))}"
                                f"<br><b>Output:</b> {_html.escape(str(guide.get('output', after_name)))}"
                            ),
                            "x": 0.01, "y": 1.08, "xref": "paper", "yref": "paper",
                            "showarrow": False, "align": "left",
                            "font": {"size": 11, "color": "#cbd5e1"},
                        }],
                    },
                ],
            })

        if not all_traces:
            return None

        for trace_idx in range(*trace_ranges[0]):
            all_traces[trace_idx].visible = True

        fig = go.Figure(data=all_traces)
        box = _box_trace_from_nodes(shared_nodes_for_box)
        has_box = box is not None
        if has_box:
            fig.add_trace(box)

        total_trace_count = len(fig.data)
        for button_idx, (start, end) in enumerate(trace_ranges):
            visible = [False] * total_trace_count
            for trace_idx in range(start, end):
                visible[trace_idx] = True
            if has_box:
                visible[-1] = True
            buttons[button_idx]["args"][0]["visible"] = visible

        fig.update_layout(
            title=f"{dataset} — transformation into {snapshots[1][0]}",
            template="plotly_dark",
            height=780,
            margin=dict(l=0, r=0, t=145, b=0),
            scene=dict(
                xaxis_title="x physical position (µm)",
                yaxis_title="y physical position (µm)",
                zaxis_title="z physical position (µm)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.08)),
            ),
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.18,
                "showactive": True,
            }],
            legend=dict(orientation="h", y=1.01, x=0.30),
            annotations=[{
                "text": "Choose a stage. The view shows exactly what the operation added or removed in the same 3D crop.",
                "x": 0.01, "y": 1.08, "xref": "paper", "yref": "paper",
                "showarrow": False, "align": "left",
                "font": dict(size=11, color="#cbd5e1"),
            }],
        )

        html_path = BEGINNER_LAB_DIR / f"{dataset}_input_delta_output_3d_lab.html"
        html_path.write_text(
            fig.to_html(include_plotlyjs=True, full_html=True),
            encoding="utf-8",
        )
        display(HTML("""
        <h3>Interactive input → delta → output 3D laboratory</h3>
        <p>
        Select a stage from the dropdown.
        <b style="color:#86efac">Green</b> objects were added,
        <b style="color:#fb7185">red</b> objects were removed,
        and cyan/gray objects survived unchanged.
        </p>
        """))
        if PLOTLY_DISPLAY_INLINE:
            display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))
        print("Saved transformation lab:", html_path)
        return html_path

    _make_transformation_delta_lab(visual_dataset, snapshots, focus_t, crop)

    ablation_rows = []
    for stage_name, guide in _stage_catalog.items():
        ablation_rows.append({
            "stage": stage_name,
            "baseline_status": guide["baseline_status"],
            "master_switch_or_main_control": guide["toggle"],
            "what_a_beginner_should_test": (
                "Run the same visual subset twice, changing only this stage. "
                "Compare the stage ledger, the 3D delta view and final graph statistics."
            ),
            "main_risk_when_more_aggressive": guide["too_strong"],
        })
    ablation_df = pd.DataFrame(ablation_rows)
    display(HTML("""
    <h3>Safe ablation protocol</h3>
    <p>
    Do not change five parameters at once. Use the same small dataset subset,
    change one switch or threshold, rerun, and compare the final graph with the
    original 0.897 configuration.
    </p>
    """))
    display(ablation_df)
    ablation_df.to_csv(BEGINNER_LAB_DIR / "ablation_experiment_guide.csv", index=False)

    display(HTML(f"""
    <div style='border:1px solid #334155;border-radius:16px;padding:14px;background:#0f172a;color:#e2e8f0;margin-top:14px'>
      <b>Files created for secondary development</b>
      <ul>
        <li><code>{visual_dataset}_beginner_stage_ledger.csv</code></li>
        <li><code>{visual_dataset}_beginner_stage_details.json</code></li>
        <li><code>{visual_dataset}_input_delta_output_3d_lab.html</code></li>
        <li><code>ablation_experiment_guide.csv</code></li>
      </ul>
    </div>
    """))
else:
    print("Beginner Pipeline Laboratory is OFF because visualization is disabled.")

# ===== CELL 29 [markdown] =====
## How to extend this notebook

For secondary development, change one stage at a time and compare the visual accounting:

- **Detector sensitivity:** adjust `BIOHUB_DET_THRESHOLD` and inspect raw candidate density.
- **Main tracking edges:** tune motion-relink gates and compare raw versus final edge overlays.
- **Broken tracks:** tune one-frame gap close and watch `Gap nodes added`.
- **Division recall/precision:** tune safe-division gates and inspect star markers in the lineage figure.
- **False positives:** tune `OUTPUT_MIN_TRACK_LEN` and compare removed short-track nodes.

When preparing a real submission, leave submission mode enabled so the notebook processes every dataset and skips figures.

# ===== CELL 30 [markdown] =====
## Verify that visualization did not change the submission

This is the final separation guarantee between the 0.897 scoring pipeline and the visual teaching layer.

# ===== CELL 31 [code] =====
SUBMISSION_SHA256_AFTER_VISUALS = _sha256_file(_SUBMISSION_PATH)
SUBMISSION_BYTES_AFTER_VISUALS = int(_SUBMISSION_PATH.stat().st_size)
_submission_after_df = pd.read_csv(_SUBMISSION_PATH)
SUBMISSION_ROWS_AFTER_VISUALS = int(len(_submission_after_df))
SUBMISSION_DATASETS_AFTER_VISUALS = tuple(
    sorted(_submission_after_df["dataset"].astype(str).unique().tolist())
)

_integrity_checks = {
    "sha256_unchanged": SUBMISSION_SHA256_AFTER_VISUALS == SUBMISSION_SHA256_BEFORE_VISUALS,
    "byte_size_unchanged": SUBMISSION_BYTES_AFTER_VISUALS == SUBMISSION_BYTES_BEFORE_VISUALS,
    "row_count_unchanged": SUBMISSION_ROWS_AFTER_VISUALS == SUBMISSION_ROWS_BEFORE_VISUALS,
    "dataset_coverage_unchanged": SUBMISSION_DATASETS_AFTER_VISUALS == SUBMISSION_DATASETS_BEFORE_VISUALS,
}
display(pd.DataFrame([_integrity_checks]))

if not all(_integrity_checks.values()):
    raise RuntimeError(
        "Visualization changed submission.csv. "
        f"Before={SUBMISSION_SHA256_BEFORE_VISUALS}, "
        f"after={SUBMISSION_SHA256_AFTER_VISUALS}"
    )

print("PASS: submission.csv is byte-for-byte unchanged after all visualization modules.")
print("Final submission sha256:", SUBMISSION_SHA256_AFTER_VISUALS)

# ===== CELL 32 [markdown] =====
## Output Diagnostics

These diagnostics are not part of the submission file. They summarize graph
size, filtering, gap recovery, and division-like branching so the score behavior
can be interpreted after a run.

# ===== CELL 33 [code] =====
if RUN_OUTPUT_DIAGNOSTICS and len(stats):
    import matplotlib.pyplot as plt

    stats_view = stats.copy()
    stats_view["dropped_edges_total"] = 0
    for col in [
        "dropped_nonconsecutive_edges",
        "dropped_long_edges",
        "dropped_multi_parent_edges",
        "dropped_multi_child_edges",
        "dropped_division_edges",
    ]:
        if col in stats_view:
            stats_view["dropped_edges_total"] += stats_view[col].fillna(0)
    stats_view["kept_edge_fraction"] = stats_view["edges"] / stats_view["raw_edges"].clip(lower=1)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)
    axes = axes.ravel()

    axes[0].hist(stats_view["nodes"], bins=min(24, max(5, len(stats_view))))
    axes[0].set_title("Nodes per video")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("videos")

    axes[1].hist(stats_view["edges"], bins=min(24, max(5, len(stats_view))))
    axes[1].set_title("Edges per video")
    axes[1].set_xlabel("edges")

    axes[2].scatter(stats_view["nodes"], stats_view["edges"], s=30, alpha=0.75)
    axes[2].set_title("Edge count vs node count")
    axes[2].set_xlabel("nodes")
    axes[2].set_ylabel("edges")

    axes[3].hist(stats_view["edge_to_node_ratio"], bins=min(24, max(5, len(stats_view))))
    axes[3].set_title("Edges per node")
    axes[3].set_xlabel("edge_to_node_ratio")
    axes[3].set_ylabel("videos")

    axes[4].scatter(stats_view["raw_edges"], stats_view["edges"], s=30, alpha=0.75)
    axes[4].plot(
        [0, max(stats_view["raw_edges"].max(), 1)],
        [0, max(stats_view["raw_edges"].max(), 1)],
        color="black", linewidth=1, alpha=0.35,
    )
    axes[4].set_title("Raw vs kept edges")
    axes[4].set_xlabel("raw edges")
    axes[4].set_ylabel("kept edges")

    axes[5].hist(stats_view["division_like_sources"], bins=min(20, max(5, len(stats_view))))
    axes[5].set_title("Division-like sources")
    axes[5].set_xlabel("sources with out-degree >= 2")

    plt.show()

    drop_cols = [c for c in stats_view.columns if c.startswith("dropped_")]
    if drop_cols:
        drop_summary = stats_view[["dataset", "raw_edges", "edges", "kept_edge_fraction", *drop_cols]].copy()
        display(drop_summary.sort_values("dropped_edges_total", ascending=False).head(12))

    gap_cols = [c for c in stats_view.columns if c.startswith("gap_")]
    if gap_cols:
        gap_summary = stats_view[["dataset", "nodes", "edges", *gap_cols]].copy()
        display(gap_summary.sort_values("gap_added_edges", ascending=False).head(12))
        display(gap_summary[gap_cols].sum().to_frame("total").T)

    display(stats_view.sort_values("nodes", ascending=False).head(10))

    repair_cols = [
        "motion_relink_edges", "motion_relink_tight_edges", "motion_relink_relaxed_edges",
        "motion_relink_replaced_raw_edges", "motion_relink_fallback_raw",
        "gap_added_nodes", "gap_added_edges", "gap2_candidates",
        "gap2_pairs_selected", "gap2_added_nodes", "gap2_added_edges",
        "safe_division_candidates", "safe_divisions_added",
        "short_track_nodes_removed", "short_track_edges_removed",
    ]
    repair_cols = [col for col in repair_cols if col in stats_view.columns]
    if repair_cols:
        repair_summary = stats_view[repair_cols].sum().rename("total").to_frame()
        display(repair_summary)

