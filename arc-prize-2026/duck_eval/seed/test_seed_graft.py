"""Smoke + negative controls for the SEED arm graft.

Runs the notebook's real _patch_qwen38_setup_commands against the REAL pinned
bundle setup_commands.json. Every guard gets a negative control: a guard that
has never refused may be one that CANNOT (feedback_guard_never_fired).
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks" / "q38-seed-eval" / "arc3-q38-seed-eval.ipynb"
BUNDLE = (
    REPO / "runs" / "harness_diff_0813" / "ds"
    / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim" / "setup_commands.json"
)

PASS, FAIL = 0, 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label} {detail}")


def load_patcher() -> types.ModuleType:
    """Exec only the setup-patching machinery out of the notebook cell."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code"
        and "def _patch_qwen38_setup_commands" in "".join(c["source"])
    )
    # Take from the seed block through the end of the patch function.
    start = src.index("# --- SEED ARM (2026-08-28)")
    end = src.index("def _run_shell_commands")
    body = src[start:end]

    mod = types.ModuleType("seed_patch")
    mod.__dict__.update(
        {
            "re": re,
            "QWEN_MODEL_OWNER": "foysalemonshanto",
            "QWEN_MODEL_SLUG": "qwen3-8-27b-fp8-repacked-v1",
            "QWEN_SERVED_MODEL_NAME": "qwen3-8-27b-fp8",
        }
    )
    # _replace_python_assignment lives above the seed block; pull it in too.
    ra_start = src.index("def _replace_python_assignment(")
    ra_end = src.index("def _patch_qwen38_setup_commands")
    exec(compile(src[ra_start:ra_end], "<nb:_replace>", "exec"), mod.__dict__)
    exec(compile(body, "<nb:seed>", "exec"), mod.__dict__)
    return mod


def main() -> int:
    mod = load_patcher()
    commands = json.loads(BUNDLE.read_text(encoding="utf-8"))
    seed = mod.ANALYZER_SEED

    print("POSITIVE: patch the real pinned bundle")
    out = mod._patch_qwen38_setup_commands(list(commands))
    joined = "\n".join(out)
    check("seed key injected exactly once", joined.count("LOCAL_ANALYZER_SEED") == 1)
    check(
        "seed injected with the configured value",
        f"'LOCAL_ANALYZER_SEED': '{seed}'," in joined,
    )
    check(
        "seed sits inside the setup_env dict, next to temperature",
        f"'LOCAL_ANALYZER_TEMPERATURE': '0.6',\n    'LOCAL_ANALYZER_SEED': '{seed}',"
        in joined,
    )
    check("seed is a non-negative int (>=0 reaches the vLLM wire)", int(seed) >= 0)
    check("model identity replacement still lands", "qwen3-8-27b-fp8" in joined)
    for name, value in mod._TEETH_INVARIANTS.items():
        check(f"invariant preserved: {name}={value}", f"'{name}': '{value}'," in joined)

    print("\nNEGATIVE CONTROLS: every guard must be able to refuse")

    def refuses(label: str, mutate) -> None:
        bad = [mutate(str(c)) for c in commands]
        try:
            mod._patch_qwen38_setup_commands(bad)
        except RuntimeError as exc:
            check(label, True)
            print(f"       -> {str(exc)[:90]}")
        except Exception as exc:  # noqa: BLE001
            check(label, False, f"raised {type(exc).__name__}, not RuntimeError: {exc}")
        else:
            check(label, False, "did NOT refuse")

    refuses(
        "refuses when the anchor is gone (silent no-op)",
        lambda c: c.replace("    'LOCAL_ANALYZER_TEMPERATURE': '0.6',\n", ""),
    )
    refuses(
        "refuses when the bundle already sets a seed",
        lambda c: c.replace(
            "    'LOCAL_ANALYZER_TEMPERATURE': '0.6',\n",
            "    'LOCAL_ANALYZER_SEED': '999',\n"
            "    'LOCAL_ANALYZER_TEMPERATURE': '0.6',\n",
        ),
    )
    refuses(
        "refuses when an untested variable drifted (YIELD_SECONDS 60->180)",
        lambda c: c.replace(
            "'LOCAL_ANALYZER_YIELD_SECONDS': '60',",
            "'LOCAL_ANALYZER_YIELD_SECONDS': '180',",
        ),
    )
    refuses(
        "refuses when TOOL_STEPS drifted (crosses the B47 inert-cap finding)",
        lambda c: c.replace(
            "'LOCAL_ANALYZER_TOOL_STEPS': '0',",
            "'LOCAL_ANALYZER_TOOL_STEPS': '12',",
        ),
    )
    refuses(
        "refuses when the temperature value itself drifted",
        lambda c: c.replace(
            "    'LOCAL_ANALYZER_TEMPERATURE': '0.6',\n",
            "    'LOCAL_ANALYZER_TEMPERATURE': '1.0',\n",
        ),
    )
    refuses(
        "refuses when the model-identity anchor is gone (pre-existing guard)",
        lambda c: re.sub(r"(?m)^MODEL_SLUG\s*=.*$", "", c),
    )

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
