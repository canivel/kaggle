"""Build the A24 SERVING arm: the certified field-floor notebook, changed ONLY in the
four vLLM serving flags that the 2.66 public-ceiling artifact says it changed.

Base      : notebooks/q38-field-eval/arc3-q38-field-eval.ipynb  (config mean 1.541, n=8)
Delta     : cell 5 only -- the setup-command rewrite hook already used for MODEL_*.
Reference : romantamrazov/arc-real-agi-solution cell 8 (`v22_block` / `_v31_spawn`),
            same wheelhouse (driessmit1/arc3-vllm-h100-wheelhouse-v3, vllm==0.19.0)
            and the same model pin we already serve.

feedback_vllm_params is binding: ALL FOUR flags or NONE.  262144 without fp8 KV is an
OOM, not a weaker arm.  If the armed launch fails, the ladder reruns the byte-identical
floor command, so this arm can never cost more than a floor draw.
"""
import json
import io
import hashlib
import sys
from pathlib import Path

BASE = Path("notebooks/q38-field-eval/arc3-q38-field-eval.ipynb")
OUT_DIR = Path("notebooks/serving-mtp3")
OUT_NB = OUT_DIR / "arc3-serving-mtp3.ipynb"
SLUG = "canivel/arc3-serving-mtp3"

INJECT = r'''

# ============================ A24 SERVING ARM (2026-08-31) ============================
# The four vLLM serving flags that the public-ceiling artifact
# (romantamrazov/arc-real-agi-solution, The AGI Boys #22 @ 2.66) states are the ONLY
# thing it changed relative to the stack we already run.  Its own comment, verbatim:
#     "Fallback = exact MTP3+async serving that produced 2.66 LB."
#     "V31 keeps the V22 gameplay/prompt/tool policy unchanged."
# Same wheelhouse (driessmit1/arc3-vllm-h100-wheelhouse-v3, vllm==0.19.0), same model
# pin (foysalemonshanto/qwen3-8-27b-fp8-repacked-v1), same machine shape.
#
# feedback_vllm_params: ALL FOUR OR NONE.  Raising --max-model-len to 262144 WITHOUT
# --kv-cache-dtype fp8 quadruples KV demand -- that is an OOM, not a weaker arm.
#
# This changes SERVING ONLY.  ANALYZER_CONTEXT_WINDOW stays 32768 and the gameplay /
# prompt / tool policy is untouched, so the arm is architecture-neutral and generalises
# to the private set (feedback_arc_generalization_first).

SERVING_ARM_NAME = "a24-mtp3-async-fp8kv-262144"
VLLM_LAUNCH_MANIFEST = WORKING_DIR / "arc3_vllm_launch.json"
SERVING_ARM_RECORD = WORKING_DIR / "arc3_serving_arm.json"

_SERVING_ARGV_SRC = """        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
    ]"""

_SERVING_ARGV_DST = """        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
        '--kv-cache-dtype',
        'fp8',
        '--speculative-config',
        '{"method":"mtp","num_speculative_tokens":3}',
        '--async-scheduling',
    ]"""

_MAX_LEN_SRC = "\nVLLM_MAX_MODEL_LEN = 65536\n"
_MAX_LEN_DST = "\nVLLM_MAX_MODEL_LEN = 262144\n"

# Persist the argv the child process is ACTUALLY launched with.  feedback_guard_never_fired
# / feedback_audit_the_instrument: an arm not OBSERVED to have fired is not an arm.
_MANIFEST_SRC = "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)"
_MANIFEST_DST = (
    "    (WORKING_DIR / 'arc3_vllm_launch.json').write_text(json.dumps(cmd), encoding='utf-8')\n"
    "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)"
)

SERVING_REQUIRED_FLAGS = (
    "--kv-cache-dtype",
    "--speculative-config",
    "--async-scheduling",
)


def _patch_serving_flags(command: str) -> "tuple[str, dict[str, int]]":
    """Add the four serving flags atomically.  Returns (command, counts)."""
    counts = {"SERVING_ARGV": 0, "MAX_MODEL_LEN": 0, "LAUNCH_MANIFEST": 0}
    if "vllm.entrypoints.openai.api_server" not in command:
        return command, counts

    if _SERVING_ARGV_SRC in command:
        command = command.replace(_SERVING_ARGV_SRC, _SERVING_ARGV_DST, 1)
        counts["SERVING_ARGV"] = 1
    if _MAX_LEN_SRC in command:
        command = command.replace(_MAX_LEN_SRC, _MAX_LEN_DST, 1)
        counts["MAX_MODEL_LEN"] = 1
    if _MANIFEST_SRC in command:
        command = command.replace(_MANIFEST_SRC, _MANIFEST_DST, 1)
        counts["LAUNCH_MANIFEST"] = 1
    return command, counts


def _serving_arm_commands(commands: "list[str]") -> "tuple[list[str], bool]":
    """Arm every setup command.  Refuses a PARTIAL arm outright."""
    armed = []
    total = {"SERVING_ARGV": 0, "MAX_MODEL_LEN": 0, "LAUNCH_MANIFEST": 0}
    for command in commands:
        command, counts = _patch_serving_flags(str(command))
        for key, value in counts.items():
            total[key] += value
        armed.append(command)

    print("taaf.kaggle: serving-arm patch =", total, flush=True)
    if total["SERVING_ARGV"] == 1 and total["MAX_MODEL_LEN"] == 1:
        if total["LAUNCH_MANIFEST"] != 1:
            print(
                "serving-arm: launch manifest hook did not apply; the flags would be "
                "unobservable, so the arm REFUSES to fire.",
                flush=True,
            )
            return list(commands), False
        return armed, True

    print(
        "serving-arm: bundled setup no longer matches the known serving block "
        f"({total}); running the certified floor unchanged.",
        flush=True,
    )
    return list(commands), False


def _kill_partial_vllm() -> None:
    """Reap a half-started server before the ladder retries (tamrazov _v22_cleanup)."""
    import signal

    pid_path = WORKING_DIR / "vllm-openai-server.pid"
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
        except OSError:
            pass
        try:
            os.kill(pid, 0)
        except OSError:
            pass
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except Exception as exc:
        print(f"serving-arm: partial-server cleanup warning: {exc!r}", flush=True)
    finally:
        pid_path.unlink(missing_ok=True)


def _record_serving_arm(state: str, detail: str) -> None:
    payload = {"arm": SERVING_ARM_NAME, "state": state, "detail": detail}
    manifest = []
    if VLLM_LAUNCH_MANIFEST.is_file():
        try:
            manifest = json.loads(VLLM_LAUNCH_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            manifest = []
    payload["argv"] = manifest
    payload["flags_present"] = sorted(
        flag for flag in SERVING_REQUIRED_FLAGS if flag in manifest
    )
    if "--max-model-len" in manifest:
        payload["max_model_len"] = manifest[manifest.index("--max-model-len") + 1]
    SERVING_ARM_RECORD.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"serving-arm: STATE={state} ({detail}) flags={payload['flags_present']} "
        f"max_model_len={payload.get('max_model_len')}",
        flush=True,
    )


def _run_setup_with_ladder(floor_commands, armed_commands, armed: bool) -> None:
    """Primary = the four-flag arm.  Fallback = the byte-identical certified floor.

    The arm can therefore never cost more than a floor draw: a startup failure spends
    setup minutes, not the night.
    """
    env = _command_env()

    def _execute(commands):
        nonlocal env
        for command in commands:
            print(f"taaf.kaggle: setup command: {command}", flush=True)
            subprocess.run(str(command), shell=True, check=True, cwd=WORKING_DIR, env=env)
            env.update(_load_setup_env())
            os.environ.update(env)

    if not armed:
        _execute(floor_commands)
        _record_serving_arm("FLOOR", "patch anchors absent; arm never fired")
        return

    try:
        print("serving-arm: launching MTP3 + async-scheduling + fp8 KV + 262144.", flush=True)
        _execute(armed_commands)
    except subprocess.CalledProcessError as exc:
        print(
            f"serving-arm: armed startup FAILED ({exc!r}); falling back to the "
            "certified floor serving config.",
            flush=True,
        )
        _kill_partial_vllm()
        VLLM_LAUNCH_MANIFEST.unlink(missing_ok=True)
        env = _command_env()
        _execute(floor_commands)
        _record_serving_arm("FALLBACK", f"armed startup failed: {exc!r}")
        return

    _record_serving_arm("ARMED", "armed startup succeeded")
# ========================== end A24 SERVING ARM injection ============================
'''

OLD_DISPATCH = '''    if filename == "setup_commands.json":
        commands = _patch_qwen38_setup_commands(commands)

    env = _command_env()'''

NEW_DISPATCH = '''    if filename == "setup_commands.json":
        floor_commands = _patch_qwen38_setup_commands(commands)
        armed_commands, armed = _serving_arm_commands(floor_commands)
        _run_setup_with_ladder(floor_commands, armed_commands, armed)
        return

    env = _command_env()'''

OLD_TAIL = "# Fail early if the analyzer is still exposing an old model identity."

NEW_TAIL = '''# Report which serving arm actually fired.  Never raise: if the ladder fell back we
# still want the certified floor to produce a submission (feedback_verify_treatment_can_fire
# -- fireability must be OBSERVED, but a refusal must not cost the draw).
if SERVING_ARM_RECORD.is_file():
    _arm_state = json.loads(SERVING_ARM_RECORD.read_text(encoding="utf-8"))
    print("\\n=== A24 serving arm ===")
    print("state:         ", _arm_state.get("state"))
    print("flags present: ", _arm_state.get("flags_present"))
    print("max_model_len: ", _arm_state.get("max_model_len"))
    print("detail:        ", _arm_state.get("detail"))
else:
    print("\\n=== A24 serving arm === NO RECORD WRITTEN (unexpected)")

# Fail early if the analyzer is still exposing an old model identity.'''


def main() -> int:
    base_nb = json.load(io.open(BASE, encoding="utf-8"))
    nb = json.load(io.open(BASE, encoding="utf-8"))
    cell5 = "".join(nb["cells"][5]["source"])

    for name, needle in (("OLD_DISPATCH", OLD_DISPATCH), ("OLD_TAIL", OLD_TAIL)):
        if cell5.count(needle) != 1:
            print(f"FATAL: {name} anchor found {cell5.count(needle)}x in base cell 5")
            return 1

    anchor = "import re\n"
    if cell5.count(anchor) != 1:
        print("FATAL: 'import re' anchor not unique")
        return 1

    new5 = cell5.replace(anchor, anchor + INJECT, 1)
    new5 = new5.replace(OLD_DISPATCH, NEW_DISPATCH, 1)
    new5 = new5.replace(OLD_TAIL, NEW_TAIL, 1)
    nb["cells"][5]["source"] = new5.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    meta = json.load(io.open("notebooks/q38-field-eval/kernel-metadata.json", encoding="utf-8"))
    meta["id"] = SLUG
    meta["title"] = "arc3-serving-mtp3"
    meta["code_file"] = OUT_NB.name
    with io.open(OUT_DIR / "kernel-metadata.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    changed = [
        i
        for i in range(len(nb["cells"]))
        if "".join(nb["cells"][i]["source"]) != "".join(base_nb["cells"][i]["source"])
    ]
    print(f"wrote {OUT_NB}  sha256={hashlib.sha256(OUT_NB.read_bytes()).hexdigest()}")
    print(f"cells: {len(nb['cells'])} (base {len(base_nb['cells'])})  changed cells: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
