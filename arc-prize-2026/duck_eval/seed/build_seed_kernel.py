"""Derive the SEED arm notebook from the certified field-floor notebook.

The arm injects LOCAL_ANALYZER_SEED into the TAAF setup env. The key is ABSENT
from the pinned bundle (taaf-kaggle-source-anim-20260807-anim), so this is an
INJECTION, not a replacement -- verified against that bundle's
setup_commands.json, where the anchor occurs exactly once.

Treatment-can-fire chain, verified by direct read before the build (2026-08-28):
  LOCAL_ANALYZER_SEED (env)
    -> tool_agent.py:159  _LOCAL_ANALYZER_SEED = _get_env_int(..., -1)
    -> tool_agent.py:1536 build_chat_payload(seed=_LOCAL_ANALYZER_SEED)
    -> openai_compat.py   if provider == 'vllm' and seed >= 0: payload['seed'] = seed
Our provider is 'vllm' and the -1 default means NO seed reaches the wire today.

Everything else is byte-identical to the floor notebook. Derivation, not a copy:
the script fails closed if any anchor it expects has moved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "notebooks" / "q38-field-eval"
SRC_NB = SRC_DIR / "arc3-q38-field-eval.ipynb"
# Named OUT* so scripts/local_gate.py's determinism driver can redirect them.
OUT_NB = REPO / "notebooks" / "q38-seed-eval" / "arc3-q38-seed-eval.ipynb"
OUT_META = REPO / "notebooks" / "q38-seed-eval" / "kernel-metadata.json"

KERNEL_ID = "canivel/arc3-q38-seed-eval"
ANALYZER_SEED = "20260828"

# The cell that patches the bundled TAAF setup commands.
PATCH_FN_ANCHOR = "def _patch_qwen38_setup_commands(commands: list[str]) -> list[str]:"
COUNTS_ANCHOR = """    replacement_counts = {
        "MODEL_OWNER": 0,
        "MODEL_SLUG": 0,
        "SERVED_MODEL_NAME": 0,
    }
"""
APPEND_ANCHOR = "        patched.append(command)\n"
REPORT_ANCHOR = '    print("taaf.kaggle: Qwen3.8 setup patch =", replacement_counts, flush=True)\n'

SEED_BLOCK = '''
# --- SEED ARM (2026-08-28) -------------------------------------------------
# LOCAL_ANALYZER_SEED is ABSENT from the pinned bundle, so this is an
# INJECTION beside the temperature anchor, not a replacement.
ANALYZER_SEED = "%s"

_SEED_ANCHOR = "    'LOCAL_ANALYZER_TEMPERATURE': '0.6',\\n"
_SEED_LINE = "    'LOCAL_ANALYZER_SEED': '" + ANALYZER_SEED + "',\\n"

# TEETH: variables this arm must NOT move. A silent no-op would score normally
# while measuring nothing, so each of these is asserted present at its pinned
# value in the same command the seed is injected into.
_TEETH_INVARIANTS = {
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
}
''' % ANALYZER_SEED

SEED_INJECT = '''
        # --- SEED ARM injection + TEETH ---
        anchor_hits = command.count(_SEED_ANCHOR)
        if anchor_hits:
            if anchor_hits != 1:
                raise RuntimeError(
                    "SEED ARM: temperature anchor occurs %d times; two "
                    "injections would race." % anchor_hits
                )
            if "LOCAL_ANALYZER_SEED" in command:
                raise RuntimeError(
                    "SEED ARM: the bundle already sets LOCAL_ANALYZER_SEED. "
                    "The ABSENT premise this arm was built on is stale -- "
                    "re-verify against setup_commands.json before running."
                )
            for _name, _value in _TEETH_INVARIANTS.items():
                _needle = "'%s': '%s'," % (_name, _value)
                if _needle not in command:
                    raise RuntimeError(
                        "SEED ARM TEETH: expected %s at %r and did not find it. "
                        "An untested variable moved, or the bundle changed; "
                        "this arm must not run." % (_name, _value)
                    )
            command = command.replace(
                _SEED_ANCHOR, _SEED_ANCHOR + _SEED_LINE, 1
            )
            seed_injections += 1

'''

REPORT_LINE = '''    if seed_injections != 1:
        raise RuntimeError(
            "SEED ARM: expected exactly 1 seed injection, made %d. The setup "
            "anchor moved and the replace was a silent no-op." % seed_injections
        )
    print("taaf.kaggle: SEED ARM injected LOCAL_ANALYZER_SEED =", ANALYZER_SEED,
          "| teeth ok:", sorted(_TEETH_INVARIANTS), flush=True)
'''


def _require(haystack: str, needle: str, label: str) -> None:
    n = haystack.count(needle)
    if n != 1:
        raise SystemExit(
            f"BUILD FAILED: anchor {label!r} occurs {n} times, expected 1. "
            "The floor notebook changed; re-derive by hand."
        )


def build() -> int:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    targets = [
        i for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and PATCH_FN_ANCHOR in "".join(c["source"])
    ]
    if len(targets) != 1:
        raise SystemExit(
            f"BUILD FAILED: found {len(targets)} cells defining the setup "
            "patcher, expected exactly 1."
        )
    idx = targets[0]
    src = "".join(nb["cells"][idx]["source"])

    for needle, label in (
        (COUNTS_ANCHOR, "replacement_counts"),
        (APPEND_ANCHOR, "patched.append"),
        (REPORT_ANCHOR, "patch report print"),
        (PATCH_FN_ANCHOR, "patch fn def"),
    ):
        _require(src, needle, label)

    # 1. module-level seed constants, immediately before the patch function
    src = src.replace(PATCH_FN_ANCHOR, SEED_BLOCK.lstrip("\n") + "\n\n" + PATCH_FN_ANCHOR, 1)
    # 2. injection counter alongside the existing replacement counters
    src = src.replace(COUNTS_ANCHOR, COUNTS_ANCHOR + "    seed_injections = 0\n", 1)
    # 3. the injection itself, just before the command is appended
    src = src.replace(APPEND_ANCHOR, SEED_INJECT.lstrip("\n") + APPEND_ANCHOR, 1)
    # 4. fail-closed report after the loop
    src = src.replace(REPORT_ANCHOR, REPORT_LINE + REPORT_ANCHOR, 1)

    if "LOCAL_ANALYZER_SEED" not in src or "seed_injections" not in src:
        raise SystemExit("BUILD FAILED: graft did not land in the cell source.")

    nb["cells"][idx]["source"] = src.splitlines(keepends=True)

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # metadata: byte-identical to the floor except id/title/code_file
    meta = json.loads((SRC_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    floor_meta = dict(meta)
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/", 1)[1]
    meta["code_file"] = OUT_NB.name
    for key, value in floor_meta.items():
        if key in {"id", "title", "code_file"}:
            continue
        if meta[key] != value:
            raise SystemExit(f"BUILD FAILED: metadata drifted on {key!r}.")
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"built {OUT_NB}")
    print(f"  kernel id : {meta['id']}")
    print(f"  seed      : {ANALYZER_SEED}")
    print(f"  cell      : {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
