"""A17 v6 full-window bench smoke — validates the sealed-window DATASET-weights composition.

Derived from a17_v5_smoke.py (same sections); only S2/S5 expectations change,
because v6 = v5 with the window restored to the sealed bench config
(A17_WINDOW_S=7920, original budget-derived soft_end) and banner
mode=throughput-canary-v6-dataset-weights. Checks:

  S1  notebook JSON loads; cells 2/8/12/14 compile (top-level await allowed)
  S2  cell 2 carries the v6 banner (v4/v5 banners gone) + BOTH pins
  S3  cell 8 serve-config graft UNTOUCHED (all 10 anchors + vetoes + finder)
  S4  cell 12 still carries continuation + fenced-recovery grafts (v4 comp.)
  S5  cell 14: window 7920 restored, budget-derived soft_end restored,
      v5 short-window strings gone, heartbeat/report/zero-action-abort
      machinery intact
  S6  metadata: NO model_sources; weights dataset attached; rest byte-equal
      to the duckwar family fields (env-match discipline)
  S7  model-finder simulation against the REAL downloaded weights dir:
      exactly one hit (the VL-AWQ dir), 27B-decoy refused

Run:  uv run python duck_eval/a17/a17_v6_smoke.py [--weights <dir>]
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "notebooks" / "a17-canary" / "arc3-a17-72b-canary.ipynb"
META_PATH = REPO / "notebooks" / "a17-canary" / "kernel-metadata.json"
DUCKWAR_META = REPO / "notebooks" / "duckwar" / "kernel-metadata.json"
WEIGHTS_DIR = REPO / "_weights" / "qwen25-vl-72b-awq"
WEIGHTS_DATASET = "canivel/qwen25-vl-72b-awq"

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not ok else ""))


def cell_src(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def main() -> int:
    weights = WEIGHTS_DIR
    if "--weights" in sys.argv:
        weights = Path(sys.argv[sys.argv.index("--weights") + 1])

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    print("S1 compile")
    for idx in (2, 8, 12, 14):
        try:
            compile(cell_src(nb, idx), f"cell{idx}", "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            check(f"cell {idx} compiles", True)
        except SyntaxError as exc:
            check(f"cell {idx} compiles", False, repr(exc))

    print("S2 cell 2")
    c2 = cell_src(nb, 2)
    check("v6 banner", "mode=throughput-canary-v6-dataset-weights" in c2)
    check("v5 banner gone", "mode=boot-canary-v5-dataset-weights" not in c2)
    check("v4 banner gone", "throughput-canary-v4" not in c2)
    check("dataset-weights provenance kept", "weights=DATASET canivel/qwen25-vl-72b-awq" in c2)
    check("seed comment restored (rho_action denominator / full 7920s)",
          "scope v2 sec3 rho_action denominator; sec5 full 7920s window" in c2)
    check("ONLY_RESET_LEVELS pin", 'os.environ["ONLY_RESET_LEVELS"] = "true"' in c2)
    check("offline-bench force pin", 'os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"' in c2)
    check("MEASUREMENT ONLY notice", "MEASUREMENT ONLY" in c2)
    check("no GO/NO-GO wording beyond the no-reading notice", "GO/NO-GO" not in c2)

    print("S3 cell 8 serve-config graft untouched")
    c8 = cell_src(nb, 8)
    for token in ("A17_SETUP_REWRITES", "_a17_patch_setup_commands",
                  "Qwen2.5-VL-72B-Instruct-AWQ", "hermes", "awq_marlin",
                  "VLLM_MAX_MODEL_LEN = 32768", "_a17_serve_asserts()",
                  "A17-CANARY gpu=", "a17_vllm_cmd.json",
                  "Qwen2_5_VLForConditionalGeneration", "_a17_find_72b_model"):
        check(f"cell 8 token {token!r}", token in c8)
    check("cell 8 graft not doubled", c8.count("_a17_patch_setup_commands(") == 2)

    print("S4 cell 12 v4 composition")
    c12 = cell_src(nb, 12)
    check("continuation graft", "import continuation_patch" in c12)
    check("fenced-recovery graft", "import fenced_recovery_patch" in c12)
    check("fenced fail-loud", "A17-CANARY FATAL: fenced_recovery_patch.apply() returned False" in c12)

    print("S5 cell 14 sealed full window")
    c14 = cell_src(nb, 14)
    check("window 7920 restored", "    A17_WINDOW_S = 7920.0" in c14)
    check("1500 window gone", "A17_WINDOW_S = 1500.0" not in c14)
    check("v5 short-window comment gone", "BOOT CANARY" not in c14)
    check("budget-derived soft_end restored",
          'budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)' in c14)
    check("budget soft_end anchored at NOTEBOOK_START_EPOCH",
          "soft_end = min(soft_end, datetime.fromtimestamp(NOTEBOOK_START_EPOCH)"
          " + timedelta(seconds=budget - min(600.0, budget / 2)))" in c14)
    check("v5 now()-anchored soft_end gone",
          "datetime.now() + timedelta(seconds=A17_WINDOW_S)" not in c14)
    check("kill-disarm derived from window", "A17_KILL_DISARM_S = A17_WINDOW_S - 600.0" in c14)
    check("heartbeat", "_a17_start_heartbeat()" in c14)
    check("post-run report", "_a17_post_run_report()" in c14)
    check("zero-action abort retained", "ZERO-ACTION-ABORT" in c14)
    check("liveness exit-70 retained", "os._exit(70)" in c14)
    check("screen games filter", 'A17_SCREEN_GAMES = ["ft09-0d8bbf25", "sb26-7fbdac44", "lp85-305b61c3", "vc33-5430563c"]' in c14)

    print("S6 metadata")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    duck = json.loads(DUCKWAR_META.read_text(encoding="utf-8"))
    check("no model_sources", "model_sources" not in meta)
    check("weights dataset attached", WEIGHTS_DATASET in meta.get("dataset_sources", []))
    check("27B dataset still attached (env-match)",
          "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot" in meta["dataset_sources"])
    check("dataset_sources = duckwar + weights",
          meta["dataset_sources"][:-1] == duck["dataset_sources"]
          and meta["dataset_sources"][-1] == WEIGHTS_DATASET)
    for field in ("docker_image", "machine_shape", "enable_gpu", "enable_internet",
                  "competition_sources", "kernel_sources", "is_private"):
        check(f"metadata field {field} matches duckwar", meta.get(field) == duck.get(field))

    print("S7 model-finder simulation (real dataset layout)")
    cfg = weights / "config.json"
    if not cfg.is_file():
        check("weights dir has config.json", False, f"{cfg} missing (download incomplete?)")
    else:
        text = cfg.read_text(encoding="utf-8")
        check("config arch = Qwen2_5_VLForConditionalGeneration",
              "Qwen2_5_VLForConditionalGeneration" in text)
        check("config has quantization_config", "quantization_config" in text)
        shards = sorted(weights.glob("*.safetensors"))
        total = sum(p.stat().st_size for p in shards)
        check("safetensors present", bool(shards))
        print(f"        shards={len(shards)} total_bytes={total:,} ({total/1e9:.1f} GB)")
        # Replay the notebook's finder against a fake /kaggle/input:
        # the weights dataset + a 27B decoy; must return EXACTLY the VL dir.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decoy = root / "vrfai-qwen3-6-27b-fp8-hf-snapshot" / "m"
            decoy.mkdir(parents=True)
            (decoy / "config.json").write_text(
                '{"architectures": ["Qwen3_5ForConditionalGeneration"], "quantization_config": {}}',
                encoding="utf-8")
            (decoy / "a.safetensors").write_text("x", encoding="utf-8")
            hits = []
            for c in sorted(list(root.rglob("config.json")) + [cfg]):
                t = c.read_text(encoding="utf-8", errors="ignore")
                if "Qwen2_5_VLForConditionalGeneration" in t and "quantization_config" in t:
                    if any(c.parent.glob("*.safetensors")):
                        hits.append(c.parent)
            check("finder: exactly 1 hit", len(hits) == 1, repr(hits))
            check("finder: hit is the VL dir", bool(hits) and hits[0] == weights)

    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
