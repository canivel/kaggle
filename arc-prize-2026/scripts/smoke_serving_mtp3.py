"""CPU smoke test for the A24 serving arm.  Runs BEFORE any Kaggle push.

feedback_test_before_submit / feedback_audit_the_instrument: the arm is a text patch
against a bundled here-doc, so the failure mode is a silent no-op or a syntax error in
the child script -- both of which would waste the night.  This test executes the real
patch functions against the REAL bundled setup_commands.json and compiles the resulting
child script.
"""
import ast
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NB = Path("notebooks/serving-mtp3/arc3-serving-mtp3.ipynb")
BASE_NB = Path("notebooks/q38-field-eval/arc3-q38-field-eval.ipynb")
BUNDLE = Path(
    "runs/harness_diff_0813/ds/jakobbrggen_taaf-kaggle-source-anim-20260807-anim/"
    "setup_commands.json"
)
START = "# ============================ A24 SERVING ARM (2026-08-31) ="
END = "# ========================== end A24 SERVING ARM injection ="

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def heredoc_body(command: str) -> str:
    """Extract the python body of the `"$PYTHON" - <<'PYSETUP' ... PYSETUP` here-doc."""
    marker = "<<'PYSETUP'\n"
    start = command.index(marker) + len(marker)
    end = command.index("\nPYSETUP", start)
    return command[start:end]


def main() -> int:
    nb = json.load(io.open(NB, encoding="utf-8"))
    base = json.load(io.open(BASE_NB, encoding="utf-8"))
    cell5 = "".join(nb["cells"][5]["source"])

    # --- S1: the whole modified cell is syntactically valid Python -------------------
    try:
        ast.parse(cell5)
        check("S1 cell 5 parses", True)
    except SyntaxError as exc:
        check("S1 cell 5 parses", False, repr(exc))
        return 1

    # --- S2: only cell 5 differs from the certified floor ---------------------------
    changed = [
        i
        for i in range(len(nb["cells"]))
        if "".join(nb["cells"][i]["source"]) != "".join(base["cells"][i]["source"])
    ]
    check("S2 exactly one changed cell (5)", changed == [5], f"changed={changed}")
    check("S2b cell count unchanged", len(nb["cells"]) == len(base["cells"]))

    # --- S3: exec the injected block against a stub environment ----------------------
    block = cell5[cell5.index(START) : cell5.index(END)]
    tmp = Path(tempfile.mkdtemp())
    ns = {
        "WORKING_DIR": tmp,
        "json": json,
        "os": __import__("os"),
        "time": time,
        "subprocess": subprocess,
        "Path": Path,
        "print": print,
    }
    try:
        exec(compile(block, "<a24-block>", "exec"), ns)
        check("S3 injected block executes", True)
    except Exception as exc:
        check("S3 injected block executes", False, repr(exc))
        return 1

    # --- S4: patch the REAL bundled setup command ------------------------------------
    commands = [str(c) for c in json.load(io.open(BUNDLE, encoding="utf-8"))]
    armed_commands, armed = ns["_serving_arm_commands"](commands)
    check("S4 arm fires on the real bundle", armed is True)
    check("S4b armed list same length", len(armed_commands) == len(commands))

    if not armed:
        return 1

    armed_cmd = armed_commands[0]

    # --- S5: all four flags present, exactly once, in the argv -----------------------
    for flag, value in (
        ("'--kv-cache-dtype',\n        'fp8',", None),
        ("'--speculative-config',\n        '{\"method\":\"mtp\",\"num_speculative_tokens\":3}',", None),
        ("'--async-scheduling',", None),
    ):
        check(f"S5 argv contains {flag.splitlines()[0].strip()}", armed_cmd.count(flag) == 1,
              f"count={armed_cmd.count(flag)}")
    check("S5d VLLM_MAX_MODEL_LEN = 262144",
          "\nVLLM_MAX_MODEL_LEN = 262144\n" in armed_cmd
          and "\nVLLM_MAX_MODEL_LEN = 65536\n" not in armed_cmd)
    check("S5e launch manifest hook injected",
          armed_cmd.count("arc3_vllm_launch.json") == 1)

    # --- S6: the patched child script still compiles ---------------------------------
    body = heredoc_body(armed_cmd)
    try:
        ast.parse(body)
        check("S6 patched child script parses", True)
    except SyntaxError as exc:
        check("S6 patched child script parses", False, repr(exc))

    # --- S7: the ONLY differences vs the floor command are the intended ones ----------
    floor_cmd = commands[0]
    import difflib

    added = [
        line[1:].strip()
        for line in difflib.unified_diff(
            floor_cmd.splitlines(), armed_cmd.splitlines(), n=0, lineterm=""
        )
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:].strip()
        for line in difflib.unified_diff(
            floor_cmd.splitlines(), armed_cmd.splitlines(), n=0, lineterm=""
        )
        if line.startswith("-") and not line.startswith("---")
    ]
    expected_added = {
        "VLLM_MAX_MODEL_LEN = 262144",
        "'--kv-cache-dtype',",
        "'fp8',",
        "'--speculative-config',",
        "'{\"method\":\"mtp\",\"num_speculative_tokens\":3}',",
        "'--async-scheduling',",
        "(WORKING_DIR / 'arc3_vllm_launch.json').write_text(json.dumps(cmd), encoding='utf-8')",
    }
    expected_removed = {"VLLM_MAX_MODEL_LEN = 65536"}
    check("S7 no unintended additions", set(added) == expected_added,
          f"unexpected={sorted(set(added) - expected_added)}")
    check("S7b no unintended removals", set(removed) == expected_removed,
          f"unexpected={sorted(set(removed) - expected_removed)}")

    # --- S8: partial-arm refusal -- a drifted bundle must NOT half-arm ----------------
    drifted = commands[0].replace("\nVLLM_MAX_MODEL_LEN = 65536\n", "\nVLLM_MAX_MODEL_LEN = 40000\n")
    _, armed_drift = ns["_serving_arm_commands"]([drifted])
    check("S8 refuses a partially-matching bundle", armed_drift is False)

    no_manifest = commands[0].replace(
        "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)", "    pass"
    )
    _, armed_nm = ns["_serving_arm_commands"]([no_manifest])
    check("S8b refuses when the arm would be unobservable", armed_nm is False)

    # --- S9: the record writer works and reports the flags ----------------------------
    (tmp / "arc3_vllm_launch.json").write_text(
        json.dumps(
            ["python", "-m", "vllm.entrypoints.openai.api_server",
             "--max-model-len", "262144", "--kv-cache-dtype", "fp8",
             "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
             "--async-scheduling"]
        ),
        encoding="utf-8",
    )
    ns["_record_serving_arm"]("ARMED", "smoke")
    rec = json.loads((tmp / "arc3_serving_arm.json").read_text(encoding="utf-8"))
    check("S9 record captures all three flags",
          rec["flags_present"] == ["--async-scheduling", "--kv-cache-dtype", "--speculative-config"],
          str(rec["flags_present"]))
    check("S9b record captures max_model_len", rec.get("max_model_len") == "262144")

    print(f"\n{len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
