"""Builder for `canivel/arc3-q38-private-eval` — the PRIVATE-CAPABILITY arm.

Composition (mission order 2026-08-21, principal: "if you know the private kernels are
winning let's build one based on it"):

  BASE          field-floor recipe (our staged `notebooks/q38-field-eval/` — byte-audited
                FOYSAL rebase, 1.59 public / lc 28 local) re-based onto Jakob Bruggen's
                NEWEST bundle: `jakobbrggen/taaf-kaggle-source` (benchmark_label
                `model-20260815-q38-p1`, pulled + sha-pinned 2026-08-21). With both flags
                OFF this notebook IS the pure-base Arm A vehicle (harness-generation
                isolation; the graft-lane session seals Arm A's own prereg).
  PRIVATE EDGE1 context-ceiling raise, flag `PRIVATE_EDGE1_CTX_RAISE`:
                ANALYZER_CONTEXT_WINDOW 32768 -> 65536 (effective agent budget
                31744 -> 64512) and VLLM_MAX_MODEL_LEN 65536 -> 131072, applied as
                fail-loud int-assignment patches to the bundle's own setup heredoc.
                KV arithmetic sealed in learnings/war_room/private_arm_prereg_2026-08-21.md.
  PRIVATE EDGE2 visible-updates capture contract, flag `PRIVATE_EDGE2_VISIBLE_CONTRACT`:
                Q38-strengthened ("yelling", forum 734843) port of the byte-audited obirdy
                prior art (notebooks/obirdy-rebase-candidate/, STAGING_NOTE_2026-08-18.md).
                Flag OFF => zero-touch (nothing imported, nothing patched).

NEVER re-authored from scratch (feedback_arc_kernel_structural_drift): every cell is the
staged field-floor notebook's bytes except the three declared deltas + one inserted cell.

SHA PINS (verified at build; build REFUSES on drift):
  base notebook code sha   7227f3286cf60b25   (notebooks/q38-field-eval, 11 cells / 10 code)
  j0815 setup_commands     7ca43b0b700cc4c3   (10,104 B, 1 command; VLLM_MAX_MODEL_LEN=65536,
                                               ANALYZER_CONTEXT_WINDOW=32768, no reasoning_effort)
  j0815 tool_agent.py      c53df973c3378337   (107,875 B; visible-only capture defect present;
                                               `If you include assistant text` sentence at L1458)
  j0815 manifest md5       798246d79122856ca1806c9445a7e57b  (75 files, __pycache__ excluded)

Usage:
    python duck_eval/private/build_private_eval.py                # base arm (flags OFF)
    python duck_eval/private/build_private_eval.py --edge1        # base + ctx raise
    python duck_eval/private/build_private_eval.py --edge1 --edge2
    python duck_eval/private/build_private_eval.py --edge2        # contract alone (if ever sealed)

NO PUSH HAPPENS HERE. Output is staged to notebooks/q38-private-eval/ only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_NB_PATH = REPO / "notebooks" / "q38-field-eval" / "arc3-q38-field-eval.ipynb"
OUT_DIR = REPO / "notebooks" / "q38-private-eval"
OUT_NB_PATH = OUT_DIR / "arc3-q38-private-eval.ipynb"
OUT_META_PATH = OUT_DIR / "kernel-metadata.json"

BASE_CODE_SHA = "7227f3286cf60b25"

OLD_BUNDLE_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
NEW_BUNDLE_REF = "jakobbrggen/taaf-kaggle-source"
EXPECTED_BUNDLE_LABEL = "model-20260815-q38-p1"

EDGE1_ANALYZER_CONTEXT_WINDOW = 65536
EDGE1_VLLM_MAX_MODEL_LEN = 131072

FORBIDDEN_TOKENS = (
    "taaf_grafts",
    "install(bm",
    "reasoning_effort",
    "banking",
    "searchmap",
    "clickmap",
    "litellm",
)

# ---------------------------------------------------------------------------
# Delta payloads
# ---------------------------------------------------------------------------

MARKDOWN_CELL0 = """\
# ARC-AGI-3 Solver — PRIVATE ARM — Qwen3.8-27B-FP8 × 08-15 harness — 25-Game P1 Eval

Field-floor recipe re-based onto Jakob's 08-15 bundle (`model-20260815-q38-p1`), with two
independently flag-gated private edges (context-ceiling raise; visible-updates capture
contract). Flags OFF = pure Arm A base. Flag states are printed in the PRIVATE-ARM BANNER
and certified per-arm. Public/offline eval = 25 public games × 1 pass; competition reruns
still use the live private game list from the Kaggle gateway.
"""

FLAGS_BLOCK = """\

# ---- PRIVATE-ARM COMPOSITION (arc3-q38-private-eval) -----------------------
# Base = field-floor recipe on the 08-15 bundle. Both flags False == pure Arm A
# base (harness-generation isolation vehicle). Each edge is independently gated;
# the flag states are printed in the PRIVATE-ARM BANNER after setup and certified
# per-arm by duck_eval/private/private_score.py. Arms are pushed as separate
# kernel VERSIONS with these constants set by duck_eval/private/build_private_eval.py
# (never hand-edited).
PRIVATE_EDGE1_CTX_RAISE = {edge1}  # 2x agent-visible context; KV arithmetic in prereg
PRIVATE_EDGE2_VISIBLE_CONTRACT = {edge2}  # Q38-strengthened visible-updates contract

BASE_ANALYZER_CONTEXT_WINDOW = 32768
BASE_VLLM_MAX_MODEL_LEN = 65536
EDGE1_ANALYZER_CONTEXT_WINDOW = 65536  # effective agent budget 64512 (= window - 1024)
EDGE1_VLLM_MAX_MODEL_LEN = 131072  # generation headroom scales with the window

EXPECTED_BUNDLE_LABEL = "model-20260815-q38-p1"
# ---------------------------------------------------------------------------
"""

BUNDLE_LABEL_ASSERT = """\

# PRIVATE ARM: refuse to run on a drifted bundle generation. Kaggle attaches the
# dataset's LATEST version; if Jakob republishes, the label changes and this run
# must die loudly as INFRA (never silently measure a different harness).
_bundle_meta = json.loads((BUNDLE_DIR / DATASET_BUNDLE_MARKER).read_text(encoding="utf-8"))
_bundle_label = str(_bundle_meta.get("benchmark_label", ""))
if _bundle_label != EXPECTED_BUNDLE_LABEL:
    raise RuntimeError(
        "PRIVATE-ARM INFRA DEATH: attached TAAF bundle generation drifted: "
        f"benchmark_label={_bundle_label!r}, expected {EXPECTED_BUNDLE_LABEL!r}. "
        "Re-audit the bundle before running (private_arm_prereg_2026-08-21.md)."
    )
print(f"TAAF bundle generation: {_bundle_label}")
"""

INT_PATCHER_FUNC = """\
def _replace_python_int_assignment(
    command: str,
    variable_name: str,
    value: int,
) -> tuple[str, int]:
    \"\"\"Replace a top-level Python int assignment inside the setup here-doc.\"\"\"
    pattern = rf"(?m)^{re.escape(variable_name)}\\s*=\\s*\\d+\\s*$"
    replacement = f"{variable_name} = {int(value)}"
    return re.subn(pattern, replacement, command, count=1)


"""

EDGE1_PATCH_BLOCK = """\

        # PRIVATE EDGE 1 — context-ceiling raise (flag-gated; fail-loud counts).
        if PRIVATE_EDGE1_CTX_RAISE:
            for variable_name, value in (
                ("VLLM_MAX_MODEL_LEN", EDGE1_VLLM_MAX_MODEL_LEN),
                ("ANALYZER_CONTEXT_WINDOW", EDGE1_ANALYZER_CONTEXT_WINDOW),
            ):
                command, count = _replace_python_int_assignment(
                    command,
                    variable_name,
                    value,
                )
                replacement_counts.setdefault(variable_name, 0)
                replacement_counts[variable_name] += count
"""

BANNER_BLOCK = """\

# ---- PRIVATE-ARM BANNER (runtime certification anchor) ---------------------
# Asserts served model + agent-visible context + contract-armed state so the
# scorer certifies WHAT RAN, per arm, from the log (feedback_audit_the_instrument).
_expected_ctx = str(
    EDGE1_ANALYZER_CONTEXT_WINDOW if PRIVATE_EDGE1_CTX_RAISE else BASE_ANALYZER_CONTEXT_WINDOW
)
_expected_mml = (
    EDGE1_VLLM_MAX_MODEL_LEN if PRIVATE_EDGE1_CTX_RAISE else BASE_VLLM_MAX_MODEL_LEN
)
_actual_ctx = os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW", "")
if _actual_ctx != _expected_ctx:
    raise RuntimeError(
        "PRIVATE-ARM INFRA DEATH: analyzer context window mismatch: "
        f"env={_actual_ctx!r}, expected {_expected_ctx!r} "
        f"(PRIVATE_EDGE1_CTX_RAISE={PRIVATE_EDGE1_CTX_RAISE})."
    )
print(
    "PRIVATE-ARM BANNER: "
    f"bundle={_bundle_label} "
    f"served={_actual_model_id} "
    f"edge1_ctx_raise={PRIVATE_EDGE1_CTX_RAISE} "
    f"analyzer_ctx={_actual_ctx} "
    f"effective_ctx_budget={int(_actual_ctx) - 1024} "
    f"vllm_max_model_len={_expected_mml} "
    f"edge2_contract={PRIVATE_EDGE2_VISIBLE_CONTRACT}"
)
"""

EDGE2_CELL = '''\
# PRIVATE EDGE 2 — Q38-strengthened visible-updates capture contract (exp 17).
#
# The harness builds world-model state ONLY from VISIBLE assistant text
# (`_update_summarized_knowledge_from_assistant`); 97.64% of model content routes to
# the hidden reasoning channel under the tool-calling grammar (exp 17, ACCEPT).
# Prior art: obirdy visible-memory candidate, byte-audited clean 2026-08-18
# (notebooks/obirdy-rebase-candidate/). Strengthened for Qwen3.8 per forum 734843
# (Jason Feng: "You might need to yell at Qwen3.8-27b ... about putting updates in
# visible output"). Prompt-only: no transport, no capture-path, no scoring change.
#
# Flag OFF => ZERO-TOUCH: nothing imported, nothing patched, one skip line printed.
if not PRIVATE_EDGE2_VISIBLE_CONTRACT:
    print("PRIVATE EDGE 2 (visible-updates contract): OFF - zero-touch")
else:
    from inference.agent import tool_agent as _pv_tool_agent

    _PV_SENTINEL = "PRIVATE VISIBLE WORLD-MODEL UPDATE CONTRACT"
    _PV_CONTRACT = """
PRIVATE VISIBLE WORLD-MODEL UPDATE CONTRACT (Q38-STRENGTHENED, MANDATORY):
READ THIS TWICE. YOUR HIDDEN REASONING IS DISCARDED BETWEEN STEPS. Anything you work
out but do not WRITE AS VISIBLE ASSISTANT TEXT is PERMANENTLY LOST to your future self.
The system that carries your memory forward reads ONLY visible assistant response text.

BEFORE EVERY `python` TOOL CALL you MUST first write the complete revised world model
as visible assistant response text in the same assistant message, THEN make the tool
call. Always begin with a non-empty `World model:` line, and include `Goal model:`,
`Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and
`Cross-level notes:` whenever they carry information. These visible lines are the ONLY
persistent memory you have.

DO NOT leave the update in hidden thinking or reasoning. Hidden reasoning is for
private deliberation ONLY and is NEVER carried forward. The earlier statement that
begins `If you include assistant text before a tool call` describes an optional
courtesy; THIS CONTRACT OVERRIDES IT: visible world-model text is REQUIRED before
every `python` tool call, every time, without exception. A tool call without a
preceding visible `World model:` update is a contract violation and wastes the step.
""".strip()

    if not getattr(_pv_tool_agent.ToolAgent, "_pv_contract_installed", False):
        _pv_original_init = _pv_tool_agent.ToolAgent.__init__

        def _pv_init(self, *args, **kwargs):
            _pv_original_init(self, *args, **kwargs)
            if _PV_SENTINEL not in self._system_prompt:
                self._system_prompt = (
                    self._system_prompt.rstrip() + "\\n\\n" + _PV_CONTRACT
                )

        _pv_tool_agent.ToolAgent.__init__ = _pv_init
        _pv_tool_agent.ToolAgent._pv_contract_installed = True

    # Preflight against the REAL agent class (fail-loud; any miss is INFRA DEATH).
    _pv_probe = _pv_tool_agent.ToolAgent(model=getattr(bm.solver, "model", "local"))
    _pv_prompt = _pv_probe._system_prompt
    _pv_required = (
        _PV_SENTINEL,
        "visible assistant response text",
        "NEVER carried forward",
        "REQUIRED before\\nevery `python` tool call",
        # The base-prompt sentence the contract overrides MUST still exist upstream;
        # if the harness generation drops it, this contract text needs re-audit.
        "If you include assistant text before a tool call",
    )
    _pv_missing = [marker for marker in _pv_required if marker not in _pv_prompt]
    if _pv_missing:
        raise RuntimeError(
            "PRIVATE-ARM INFRA DEATH: visible-updates contract failed preflight: "
            + repr(_pv_missing)
        )

    _pv_parser_sample = _pv_tool_agent._extract_scientist_note(
        "World model: visible test update\\nPlan: visible test plan"
    )
    if not any(str(value or "").strip() for value in _pv_parser_sample.values()):
        raise RuntimeError(
            "PRIVATE-ARM INFRA DEATH: harness parser rejected the visible-update sample."
        )

    _pv_audit = {
        "contract": "private visible world-model updates (Q38-strengthened)",
        "installation": "runtime ToolAgent initialization wrapper",
        "system_prompt_appended": True,
        "visible_assistant_text_required_before_python": True,
        "hidden_reasoning_used_as_persistent_memory": False,
        "hidden_reasoning_copied_to_visible_content": False,
        "request_or_response_transport_modified": False,
        "harness_parser_accepts_visible_sample": True,
        "prior_art": "obirdy/arc3-duck-qwen-3-8-visible-memory-candidate (audited 2026-08-18)",
        "strengthened_per": "forum topic 734843 (Q38 needs yelling)",
    }
    (WORKING_DIR / "private_visible_updates_contract.json").write_text(
        json.dumps(_pv_audit, indent=2) + "\\n",
        encoding="utf-8",
    )
    print("PRIVATE EDGE 2 (visible-updates contract): ACTIVE (Q38-strengthened)", flush=True)
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_src(cell: dict) -> str:
    return "".join(cell["source"])


def _set_src(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _code_sha(nb: dict) -> str:
    code = "".join(_cell_src(c) for c in nb["cells"] if c["cell_type"] == "code")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _surgical_replace(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(
            f"GATE FAIL [{label}]: anchor occurs {text.count(old)}x (need exactly 1). "
            "Base notebook drifted; re-audit before building."
        )
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(edge1: bool, edge2: bool) -> None:
    base_raw = BASE_NB_PATH.read_bytes()
    nb = json.loads(base_raw.decode("utf-8"))
    if len(nb["cells"]) != 11:
        raise SystemExit(f"GATE FAIL: base notebook has {len(nb['cells'])} cells, expected 11.")
    actual_sha = _code_sha(nb)
    if actual_sha != BASE_CODE_SHA:
        raise SystemExit(
            f"GATE FAIL: base code sha {actual_sha} != pinned {BASE_CODE_SHA}. "
            "The staged field-eval notebook drifted; re-audit before building."
        )

    out = copy.deepcopy(nb)

    # ---- Delta 0: markdown title ------------------------------------------
    _set_src(out["cells"][0], MARKDOWN_CELL0)

    # ---- Delta 3: bundle retarget + flags + label assert -------------------
    c3 = _cell_src(out["cells"][3])
    c3 = _surgical_replace(
        c3,
        f'    "{OLD_BUNDLE_REF}",\n',
        f'    "{NEW_BUNDLE_REF}",  # 08-15 bundle: {EXPECTED_BUNDLE_LABEL}\n',
        "cell3-bundle-ref",
    )
    c3 = _surgical_replace(
        c3,
        "KERNEL_SOURCES: list[str] = []\n",
        "KERNEL_SOURCES: list[str] = []\n"
        + FLAGS_BLOCK.format(edge1=repr(edge1), edge2=repr(edge2)),
        "cell3-flags",
    )
    c3 = _surgical_replace(
        c3,
        'print(f"TAAF source bundle: {BUNDLE_DIR}")\n',
        'print(f"TAAF source bundle: {BUNDLE_DIR}")\n' + BUNDLE_LABEL_ASSERT,
        "cell3-bundle-label-assert",
    )
    _set_src(out["cells"][3], c3)

    # ---- Delta 5: int patcher + edge1 patch + banner -----------------------
    c5 = _cell_src(out["cells"][5])
    c5 = _surgical_replace(
        c5,
        "def _patch_qwen38_setup_commands(",
        INT_PATCHER_FUNC + "def _patch_qwen38_setup_commands(",
        "cell5-int-patcher",
    )
    c5 = _surgical_replace(
        c5,
        "\n        # Make offline behavior explicit in the child process as well.",
        EDGE1_PATCH_BLOCK
        + "\n        # Make offline behavior explicit in the child process as well.",
        "cell5-edge1-patch",
    )
    c5 = _surgical_replace(
        c5,
        'print("Analyzer endpoint:", os.environ.get("LOCAL_ANALYZER_BASE_URL"))',
        'print("Analyzer endpoint:", os.environ.get("LOCAL_ANALYZER_BASE_URL"))\n'
        + BANNER_BLOCK,
        "cell5-banner",
    )
    _set_src(out["cells"][5], c5)

    # ---- Delta: insert edge-2 cell after the unpickle cell (index 7) -------
    edge2_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [],
    }
    _set_src(edge2_cell, EDGE2_CELL)
    out["cells"].insert(8, edge2_cell)

    # ---- Gates -------------------------------------------------------------
    # G1: diff confinement — changed [0,3,5], inserted [8], others byte-identical.
    base_cells = nb["cells"]
    out_cells = out["cells"]
    if len(out_cells) != 12:
        raise SystemExit("GATE FAIL: output must have 12 cells.")
    mapping = list(range(8)) + [None] + list(range(8, 11))
    changed = []
    for out_idx, base_idx in enumerate(mapping):
        if base_idx is None:
            continue
        if _cell_src(out_cells[out_idx]) != _cell_src(base_cells[base_idx]):
            changed.append(out_idx)
    if changed != [0, 3, 5]:
        raise SystemExit(f"GATE FAIL: changed cells {changed}, expected [0, 3, 5].")

    # G2: forbidden tokens (case-insensitive) in CODE cells.
    all_code = "".join(_cell_src(c) for c in out_cells if c["cell_type"] == "code").lower()
    hits = [t for t in FORBIDDEN_TOKENS if t.lower() in all_code]
    if hits:
        raise SystemExit(f"GATE FAIL: forbidden tokens present: {hits}")

    # G3: required markers.
    required = [
        "PRIVATE-ARM BANNER",
        "EXPECTED_BUNDLE_LABEL",
        "PRIVATE_EDGE1_CTX_RAISE",
        "PRIVATE_EDGE2_VISIBLE_CONTRACT",
        "_replace_python_int_assignment",
        "PRIVATE EDGE 2 (visible-updates contract)",
    ]
    missing = [t for t in required if t not in "".join(
        _cell_src(c) for c in out_cells if c["cell_type"] == "code")]
    if missing:
        raise SystemExit(f"GATE FAIL: required markers missing: {missing}")

    # G4: every code cell compiles.
    for i, c in enumerate(out_cells):
        if c["cell_type"] != "code":
            continue
        try:
            ast.parse(_cell_src(c))
        except SyntaxError as exc:
            raise SystemExit(f"GATE FAIL: cell {i} does not compile: {exc}")

    # G5: token parity with base on submission-critical tokens.
    for token in ("KAGGLE_IS_COMPETITION_RERUN", "TAAF_RUN_AS_SUBMISSION", "submission.parquet"):
        b = sum(_cell_src(c).count(token) for c in base_cells if c["cell_type"] == "code")
        o = sum(_cell_src(c).count(token) for c in out_cells if c["cell_type"] == "code")
        if b != o:
            raise SystemExit(f"GATE FAIL: token parity broken for {token!r}: base {b} != out {o}")

    # G6: flag constants stamped exactly as requested.
    c3_out = _cell_src(out_cells[3])
    for name, val in (
        ("PRIVATE_EDGE1_CTX_RAISE", edge1),
        ("PRIVATE_EDGE2_VISIBLE_CONTRACT", edge2),
    ):
        stamp = f"{name} = {val!r}"
        if stamp not in c3_out:
            raise SystemExit(f"GATE FAIL: flag stamp missing: {stamp!r}")

    # ---- Write -------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB_PATH.write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    meta = json.loads((BASE_NB_PATH.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "canivel/arc3-q38-private-eval"
    meta["title"] = "arc3-q38-private-eval"
    meta["code_file"] = "arc3-q38-private-eval.ipynb"
    meta["dataset_sources"] = [
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        NEW_BUNDLE_REF,
    ]
    # model_sources / docker_image / machine_shape inherited byte-identical from the
    # field-floor metadata (feedback_kaggle_env_match): FOYSAL repacked Kaggle Model,
    # docker sha 57e612b..., NvidiaRtxPro6000, internet off, competition attached.
    OUT_META_PATH.write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    arm = "base" if not (edge1 or edge2) else (
        "edge1" if edge1 and not edge2 else ("edge12" if edge1 and edge2 else "edge2")
    )
    print("BUILD OK")
    print(f"  arm            : {arm}  (edge1={edge1} edge2={edge2})")
    print(f"  output         : {OUT_NB_PATH}")
    print(f"  cells          : {len(out_cells)} (base 11 + inserted edge-2 cell at index 8)")
    print(f"  changed cells  : [0, 3, 5] + inserted [8]")
    print(f"  base code sha  : {BASE_CODE_SHA} (verified)")
    print(f"  output code sha: {_code_sha(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge1", action="store_true", help="arm the context-ceiling raise")
    parser.add_argument("--edge2", action="store_true", help="arm the visible-updates contract")
    args = parser.parse_args()
    build(edge1=args.edge1, edge2=args.edge2)
