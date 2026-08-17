"""ONE-VARIABLE PROOF: the low arm differs from the completed engine arm by the effort value
and nothing else.

The engine arm (`arc3-q38-engine-eval`, effort=medium) is COMPLETE and its result is sealed
(prereg section 12: REFUTE-2x, 21 levels, 2,857 actions). The token-cost arm is read AGAINST it
(PRIMARY-B), so any second difference between the two notebooks silently converts a mechanism
measurement into a confounded one. Prose cannot enforce that. This does.

Allowed differences, exhaustively:
  * the effort literal inside --default-chat-template-kwargs
  * the cell-2 provenance banner's `arm=` tag
  * the decode-rate probe, which is REPORT-ONLY instrumentation present in BOTH arms
    (it is built from the same source, so it must be byte-identical between them)

Anything else fails.

    python duck_eval/q38/q38_arm_diff.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MED = REPO / "notebooks" / "q38-eval" / "arc3-q38-engine-eval.ipynb"
LOW = REPO / "notebooks" / "q38-low-eval" / "arc3-q38-low-eval.ipynb"

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name + (f" - {detail}" if detail else ""))


def build_both() -> None:
    env_med = {"Q38_ARM": "medium"}
    env_low = {"Q38_ARM": "low"}
    import os
    for env in (env_med, env_low):
        e = os.environ.copy()
        e.update(env)
        subprocess.run([sys.executable, str(REPO / "duck_eval" / "q38" / "build_q38_eval.py")],
                       check=True, env=e, capture_output=True)


def cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def main() -> int:
    build_both()
    check("both arm notebooks exist", MED.is_file() and LOW.is_file())
    a, b = cells(MED), cells(LOW)
    check("same cell count", len(a) == len(b), f"{len(a)} vs {len(b)}")

    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    check("exactly cells [2, 8] differ between the arms (banner + effort literal)",
          differing == [2, 8], str(differing))

    # cell 8: the ONLY textual differences may be the DECLARED arm-dependent tokens: the
    # chat-template-kwargs literal and the serve-defs Q38_EFFORT constant. The verification
    # LOGIC must be identical; its EXPECTED VALUES track the arm (the low arm's v1 died
    # because Q38_EFFORT was frozen at 'medium' by the previous version of this very check —
    # a one-variable proof that pinned the instrument to the old arm's semantics).
    norm = (b[8]
            .replace('"reasoning_effort": "low"', '"reasoning_effort": "medium"')
            .replace("Q38_EFFORT = " + chr(92) + "'low" + chr(92) + "'",
                     "Q38_EFFORT = " + chr(92) + "'medium" + chr(92) + "'"))
    check("** cell 8 is byte-identical once the effort literal is normalised - the serve config, "
          "all 18 invariants, both probes and the decode-rate instrument are the same bytes",
          norm == a[8],
          f"{sum(1 for x, y in zip(norm, a[8]) if x != y)} residual char diffs, "
          f"len {len(norm)} vs {len(a[8])}")

    # cell 2: only the arm tag and the effort word
    n2 = b[2].replace("arm=low", "arm=medium").replace(
        "reasoning_effort=PINNED-low", "reasoning_effort=PINNED-medium")
    check("cell 2 is byte-identical once the arm tag is normalised", n2 == a[2])

    for i in (6, 12, 14):
        check(f"cell {i} byte-identical across arms", a[i] == b[i])

    for name, src in (("medium", a[8]), ("low", b[8])):
        check(f"{name} arm carries the decode-rate probe", "_q38_decode_rate" in src)
        check(f"{name} arm cell 8 is pure ASCII",
              not [c for c in src if ord(c) > 127])

    check("the serve-defs Q38_EFFORT constant tracks each arm (the low-arm v1 killer)",
          "Q38_EFFORT = " + chr(92) + "'medium" + chr(92) + "'" in a[8]
          and "Q38_EFFORT = " + chr(92) + "'low" + chr(92) + "'" in b[8])
    check("the effort literals are actually different (this is not a no-op build)",
          '"reasoning_effort": "medium"' in a[8] and '"reasoning_effort": "low"' in b[8])
    check("neither arm contains the other's effort value",
          '"reasoning_effort": "low"' not in a[8]
          and '"reasoning_effort": "medium"' not in b[8])

    # metadata: only id/title/code_file may differ
    ma = json.loads((MED.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    mb = json.loads((LOW.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    diff_keys = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
    check("metadata differs only in id/title/code_file",
          diff_keys == ["code_file", "id", "title"], str(diff_keys))
    check("both arms attach the identical dataset_sources incl. the engine",
          ma["dataset_sources"] == mb["dataset_sources"]
          and "saltb0x/qwen3-8-27b-fp8" in mb["dataset_sources"])

    print("\n" + "=" * 78)
    print(f"ARM DIFF: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAIL " + f)
        return 1
    print("ONE VARIABLE CONFIRMED: reasoning_effort medium -> low, and nothing else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
