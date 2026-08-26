"""Runtime smoke for the LORA SERVE CANARY (2026-08-14).

`feedback_test_before_submit`: v38 scored 0.00 from a missing import. Nothing goes to
Kaggle until the injected code has actually RUN here.
`feedback_kaggle_model_attach` / `feedback_kaggle_dataset_code_sync`: Kaggle drops an
unattachable dataset SILENTLY. The canary's whole measurement lives inside the adapter
dataset, so the MOUNT-PATH contract gets its own section, exercised against a fake mount
that reproduces the PUBLISHED layout byte-for-byte (subdirectories `lora-noop/`,
`lora-probe/` — NOT the dataset root).

No GPU, no network, no Kaggle call. It:
  * rebuilds the notebook from the frozen duckfork and checks determinism,
  * runs the structural-diff gate against the frozen fork (`scripts/preflight.py`
    --family duck-harness is PULL-based, i.e. POST-push only; this is the local
    equivalent — see learnings/lora_canary_readiness_2026-08-14.md),
  * extracts the REWRITTEN setup command from the built notebook and EXECUTES its
    definition section against a fake /kaggle/input mount so `_lora_find_adapters()`
    really runs, including every FAIL-LOUD post-condition, positive AND negative,
  * renders the actual `--lora-modules` argv the server will receive and asserts it
    points INTO the subdirectories,
  * checks the shipped adapter artifacts against their own manifest and against the
    shas the builder baked into the notebook,
  * exercises `_lora_png_b64` and the projection arithmetic,
  * runs the sealed scorer's selftest (`lora_serve_score.py --selftest`).

Usage:  python duck_eval/lora/lora_canary_smoke.py
"""

from __future__ import annotations

import ast
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILDER = HERE / "build_lora_serve_canary.py"
SCORER = HERE / "lora_serve_score.py"
OUT_DIR = REPO / "notebooks" / "lora-serve-canary"
OUT_NB = OUT_DIR / "arc3-lora-serve-canary.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"
SRC_NB = REPO / "notebooks" / "duckfork" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
SRC_META = SRC_NB.parent / "kernel-metadata.json"
ADAPTER_DIR = REPO / "runs" / "lora_lane" / "probe_adapters"
ADAPTER_DS = "canivel/arc3-lora-probe-adapters"
ADAPTER_SLUG = ADAPTER_DS.split("/", 1)[1]

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' - ' + detail) if detail else ''}")
    print(("  PASS " if cond else "  FAIL ") + name + ((" - " + detail) if detail else ""))


def expect_raises(name: str, fn, needle: str) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        check(name, needle in str(exc), f"raised {type(exc).__name__}: {str(exc)[:140]}")
        return
    check(name, False, "did NOT raise")


# ---------------------------------------------------------------------------
# B - build determinism
# ---------------------------------------------------------------------------
def section_build() -> None:
    print("\n[B] builder (rebuild from the frozen duckfork; never hand-built)")
    r1 = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True)
    check("B1 builder exits 0", r1.returncode == 0, r1.stderr.strip()[-300:])
    first = OUT_NB.read_bytes()
    r2 = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True)
    check("B2 builder is idempotent (rebuild byte-identical)",
          r2.returncode == 0 and OUT_NB.read_bytes() == first)


# ---------------------------------------------------------------------------
# S - structural diff vs the frozen fork (the gate preflight cannot be pre-push)
# ---------------------------------------------------------------------------
def section_structure() -> None:
    print("\n[S] structural diff vs the frozen duckfork")
    base = json.loads(SRC_NB.read_text(encoding="utf-8"))
    new = json.loads(OUT_NB.read_text(encoding="utf-8"))
    check("S1 cell count 17 vs 17", len(base["cells"]) == 17 == len(new["cells"]),
          f"{len(base['cells'])} vs {len(new['cells'])}")
    diff = [i for i in range(len(base["cells"]))
            if "".join(base["cells"][i]["source"]) != "".join(new["cells"][i]["source"])]
    check("S2 exactly cells 2, 6, 8, 14 differ", diff == [2, 6, 8, 14], f"differ={diff}")
    check("S3 cell types unchanged",
          [c["cell_type"] for c in base["cells"]] == [c["cell_type"] for c in new["cells"]])
    bad = []
    for i, cell in enumerate(new["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            compile("".join(cell["source"]), f"cell{i}", "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            bad.append(f"cell {i}: {exc}")
    check("S4 every code cell compiles", not bad, "; ".join(bad)[:200])

    bmeta = json.loads(SRC_META.read_text(encoding="utf-8"))
    nmeta = json.loads(OUT_META.read_text(encoding="utf-8"))
    changed = {k for k in set(bmeta) | set(nmeta) if bmeta.get(k) != nmeta.get(k)}
    check("S5 metadata delta is only {id,title,code_file,dataset_sources}",
          changed == {"id", "title", "code_file", "dataset_sources"},
          f"changed={sorted(changed)}")
    check("S6 env matched to the family (feedback_kaggle_env_match)",
          nmeta["docker_image"] == bmeta["docker_image"]
          and nmeta["machine_shape"] == bmeta["machine_shape"]
          and nmeta["enable_gpu"] is True and nmeta["enable_internet"] is False,
          f"gpu={nmeta.get('enable_gpu')} shape={nmeta.get('machine_shape')}")
    check("S7 fresh kernel slug (feedback_fresh_kernel_slug)",
          nmeta["id"] == "canivel/arc3-lora-serve-canary")
    check("S8 datasets = the scored TRIPLE, UNCHANGED, + the adapter dataset and nothing else",
          nmeta["dataset_sources"] == ["jeroencottaar/taaf-kaggle-source-share",
                                       "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                       "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
                                       ADAPTER_DS],
          repr(nmeta["dataset_sources"]))
    cell6 = "".join(new["cells"][6]["source"])
    check("S9 cell-6 DATASET_SOURCES matches the metadata exactly "
          "(feedback_kaggle_dataset_code_sync)",
          all(d in cell6 for d in nmeta["dataset_sources"])
          and cell6.count(ADAPTER_DS) == 1)
    cell14 = "".join(new["cells"][14]["source"])
    check("S10 boot-only short-circuit present and rerun-safe",
          "LORA_BOOT_ONLY" in cell14 and "not TRUE_SUBMISSION" in cell14
          and "await bm.run(" in cell14)
    check("S11 no graft in cell 12 (the solver surface is untouched; ONE variable)",
          "".join(new["cells"][12]["source"]) == "".join(base["cells"][12]["source"]))
    check("S12 competition source still attached (gateway spin-up)",
          nmeta.get("competition_sources") == ["arc-prize-2026-arc-agi-3"])


# ---------------------------------------------------------------------------
# R - the rewritten setup command
# ---------------------------------------------------------------------------
def _rewritten_setup_text() -> str:
    """Reproduce, from the BUILT notebook, exactly what will run on Kaggle."""
    nb = json.loads(OUT_NB.read_text(encoding="utf-8"))
    cell8 = "".join(nb["cells"][8]["source"])
    ns: dict = {}
    block = cell8.split("# --- LORA-SERVE-CANARY BEGIN serve-config rewrite", 1)[1]
    block = block.split("# --- LORA-SERVE-CANARY END serve-config rewrite", 1)[0]
    exec("LORA_SETUP_REWRITES" + block.split("LORA_SETUP_REWRITES", 1)[1], ns)  # noqa: S102
    original = json.loads((REPO / "duck_eval" / "taaf_bundle" / "setup_commands.json")
                          .read_text(encoding="utf-8"))
    return ns["_lora_patch_setup_commands"](original)[0]


def _python_body(text: str) -> str:
    lines = text.splitlines()
    if lines[0] != '"$PYTHON" - <<\'PYSETUP\'' or lines[-1] != "PYSETUP":
        raise AssertionError(f"unexpected heredoc wrapper: {lines[0]!r} ... {lines[-1]!r}")
    return "\n".join(lines[1:-1])


def _compiles(text: str) -> bool:
    try:
        compile(text, "setup", "exec")
        return True
    except SyntaxError as exc:
        print("   syntax error:", exc)
        return False


def section_rewrite(text: str) -> None:
    print("\n[R] rewritten setup command")
    body = _python_body(text)
    check("R1 the heredoc body compiles as Python", _compiles(body))
    check("R1b nothing we inject breaks the heredoc (no bare PYSETUP sentinel)",
          "PYSETUP" not in body and text.count("PYSETUP") == 2)
    for token, want in [
        ("--enable-lora", True), ("--max-lora-rank", True), ("--max-loras", True),
        ("--lora-dtype", True), ("--lora-modules", True),
        ("arc3-noop=", True), ("arc3-probe=", True),
        # the scored config must survive UNCHANGED - this canary moves ONE variable
        ("qwen3_coder", True), ("--reasoning-parser", True),
        ("--enable-prefix-caching", True), ("--enable-auto-tool-choice", True),
        ('{"preserve_thinking": true}', True),
        ("--quantization", False),      # not in the scored duck serve line
        ("hermes", False),
    ]:
        check(f"R2 {'has' if want else 'lacks'} {token}", (token in text) is want)
    check("R3 served brain UNCHANGED (the scored 27B)",
          "SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'" in text
          and "MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'" in text)
    check("R4 wheelhouse UNCHANGED (vLLM 0.19.0 - the result is scoped to 0.19.0)",
          "WHEELHOUSE_SLUG = 'arc3-vllm-h100-wheelhouse-v3'" in text
          and "vllm==0.19.0" in text)
    check("R5 max-model-len UNCHANGED at 65536", "VLLM_MAX_MODEL_LEN = 65536" in text)
    # Count the ARGV forms, not prose: '--enable-lora' also appears inside two FATAL
    # message strings ("the vision path does not survive --enable-lora").
    check("R6 exactly one --enable-lora / --max-lora-rank / --lora-modules in the argv",
          text.count("'--enable-lora',") == 1 and text.count("'--max-lora-rank',") == 1
          and text.count("'--lora-modules',") == 1,
          f"enable={text.count(chr(39) + '--enable-lora' + chr(39) + ',')} "
          f"rank={text.count(chr(39) + '--max-lora-rank' + chr(39) + ',')}")
    check("R7 --max-loras 2 (both adapters resident at once)",
          "'--max-loras',\n        '2'," in text)
    check("R8 rank on the serve line == r in BOTH shipped adapter_config.json",
          all(json.loads((ADAPTER_DIR / sub / "adapter_config.json").read_text())["r"] == 16
              for sub in ("lora-noop", "lora-probe"))
          and "'--max-lora-rank',\n        '16'," in text)
    # Anchor on the CALL SITES in the imperative tail, not the `def` lines above them.
    check("R9 ordering: adapters resolved BEFORE the server starts, asserts AFTER",
          body.index("\n_lora_find_adapters()\nprint(f'vLLM wheelhouse path:")
          < body.index("\nstart_vllm_server()\n_lora_boot_s")
          < body.index("\n_lora_serve_asserts()\n")
          < body.index("\n_lora_throughput(_lora_boot_s)\n"))
    check("R9b the boot clock starts before install+load and is handed to the probe",
          "_lora_t0 = time.monotonic()\nstart_vllm_server()" in body
          and "_lora_boot_s = time.monotonic() - _lora_t0" in body)
    check("R10 the differential is FAIL-LOUD in both directions",
          "a ZERO-delta adapter changed the output" in body
          and "SILENTLY IGNORED" in body)
    check("R11 the never-called Tufa guard is staged into the SERVER process",
          "_lora_install_guard()" in body and "vllm_runtime_lora_guard.py" in body
          and "sitecustomize.py" in body)
    check("R12 sha + rank of each shipped adapter re-verified at runtime",
          "the dataset push did not ship what we built" in body
          and "LORA-CANARY FATAL: ' + sub + ' rank" in body)
    check("R13 tool-call and MM round-trips are addressed to the ADAPTER, not the base",
          body.count("'model': 'arc3-probe'") >= 2)


# ---------------------------------------------------------------------------
# D - THE MOUNT-PATH CONTRACT (highest-value check; a mismatch wastes the slot)
# ---------------------------------------------------------------------------
def _fake_mount(root: Path, *, layout: str = "published") -> Path:
    """Reproduce the PUBLISHED dataset layout. `kaggle datasets files
    canivel/arc3-lora-probe-adapters` (read-only, 2026-08-14) returns:
        PROBE_ADAPTERS.json                        399
        README.md                                 1104
        lora-noop/adapter_config.json              709
        lora-noop/adapter_model.safetensors   41962184
        lora-probe/adapter_config.json             709
        lora-probe/adapter_model.safetensors  41962184
    i.e. the adapters are in SUBDIRECTORIES, never at the dataset root.
    The real 41 MB weights are hardlinked/copied so the runtime sha check is real."""
    mount = root / "input" / ADAPTER_SLUG
    mount.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ADAPTER_DIR / "PROBE_ADAPTERS.json", mount / "PROBE_ADAPTERS.json")
    for sub in ("lora-noop", "lora-probe"):
        dest = mount if layout == "flat" else mount / sub
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ADAPTER_DIR / sub / "adapter_config.json", dest / "adapter_config.json")
        shutil.copy2(ADAPTER_DIR / sub / "adapter_model.safetensors",
                     dest / "adapter_model.safetensors")
        if layout == "flat":
            break  # a flat layout can only hold ONE adapter - that IS the failure
    return mount


def _defs_namespace(text: str, root: Path, *, attach: bool = True) -> dict:
    """Exec the setup command's DEFINITION section (everything before the first
    imperative statement, which is `_lora_find_adapters()`)."""
    head = _python_body(text).split("_lora_find_adapters()\nprint(f'vLLM wheelhouse path:", 1)[0]
    working = root / "working"
    working.mkdir(parents=True, exist_ok=True)
    # Exactly what cell 6 exports: {ref: mount_path} for every ATTACHED dataset.
    mapping = {"driessmit1/arc3-vllm-h100-wheelhouse-v3": str(root / "input" / "wheels"),
               "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot": str(root / "input" / "model")}
    if attach:
        mapping[ADAPTER_DS] = str(root / "input" / ADAPTER_SLUG)
    os.environ["TAAF_KAGGLE_INPUT_PATHS"] = json.dumps(mapping, sort_keys=True)
    os.environ["TAAF_KAGGLE_WORKING_DIR"] = str(working)
    ns: dict = {"__name__": "lora_setup"}
    exec(compile(head, "setup_defs", "exec"), ns)  # noqa: S102
    return ns


def _rendered_lora_modules(ns: dict) -> list[str]:
    """The exact argv fragment the vLLM server will receive."""
    return ["arc3-noop=" + str(ns["_LORA_ADAPTERS"]["arc3-noop"]),
            "arc3-probe=" + str(ns["_LORA_ADAPTERS"]["arc3-probe"])]


def section_mount(text: str) -> None:
    print("\n[D] mount-path contract vs the PUBLISHED dataset layout")
    root = Path(tempfile.mkdtemp(prefix="lorasmoke_"))
    try:
        mount = _fake_mount(root)
        ns = _defs_namespace(text, root)
        ns["_lora_find_adapters"]()
        noop = ns["_LORA_ADAPTERS"]["arc3-noop"]
        probe = ns["_LORA_ADAPTERS"]["arc3-probe"]
        check("D1 both adapters resolved", noop is not None and probe is not None)
        check("D2 resolved paths point INTO the subdirectories, not at the dataset root",
              noop == mount / "lora-noop" and probe == mount / "lora-probe",
              f"noop={noop} probe={probe}")
        argv = _rendered_lora_modules(ns)
        check("D3 rendered --lora-modules argv is name=<mount>/<subdir>",
              argv == [f"arc3-noop={mount / 'lora-noop'}",
                       f"arc3-probe={mount / 'lora-probe'}"],
              repr(argv))
        check("D4 each rendered path holds adapter_config.json AND adapter_model.safetensors",
              all((Path(a.split("=", 1)[1]) / f).is_file()
                  for a in argv for f in ("adapter_config.json", "adapter_model.safetensors")))
        check("D5 resolver maps the ref through TAAF_KAGGLE_INPUT_PATHS (cell 6's export)",
              ns["resolve_kaggle_dataset_path"]("canivel", ADAPTER_SLUG) == mount)
        check("D6 and falls back to /kaggle/input/<slug> when the map is absent",
              str(ns["resolve_kaggle_dataset_path"].__doc__ or "") is not None
              and _fallback_is_kaggle_input(ns))

        print("  -- negative paths (each MUST kill the kernel loudly) --")
        # THE 08-13 NEAR-MISS: Kaggle silently drops an unattachable dataset.
        root2 = Path(tempfile.mkdtemp(prefix="lorasmoke_unattached_"))
        try:
            (root2 / "input").mkdir(parents=True)
            ns2 = _defs_namespace(text, root2, attach=False)
            expect_raises("D7 dataset SILENTLY DROPPED by Kaggle => FATAL before any GPU-hour "
                          "(feedback_kaggle_model_attach)",
                          ns2["_lora_find_adapters"], "adapter dataset not mounted at")
        finally:
            shutil.rmtree(root2, ignore_errors=True)

        # A flattened dataset (the exact path mismatch this section exists to catch).
        root3 = Path(tempfile.mkdtemp(prefix="lorasmoke_flat_"))
        try:
            _fake_mount(root3, layout="flat")
            ns3 = _defs_namespace(text, root3)
            expect_raises("D8 FLAT layout (adapters at the dataset root) => FATAL, "
                          "never a silent serve of the wrong path",
                          ns3["_lora_find_adapters"], "no lora-noop/adapter_config.json")
        finally:
            shutil.rmtree(root3, ignore_errors=True)

        # The dataset shipped something other than what we built.
        root4 = Path(tempfile.mkdtemp(prefix="lorasmoke_sha_"))
        try:
            m4 = _fake_mount(root4)
            w = m4 / "lora-probe" / "adapter_model.safetensors"
            raw = bytearray(w.read_bytes())
            raw[-1] ^= 0xFF
            w.write_bytes(bytes(raw))
            ns4 = _defs_namespace(text, root4)
            expect_raises("D9 tampered/stale adapter weights => FATAL on the sha",
                          ns4["_lora_find_adapters"],
                          "the dataset push did not ship what we built")
        finally:
            shutil.rmtree(root4, ignore_errors=True)

        # A rank the serve line cannot honour.
        root5 = Path(tempfile.mkdtemp(prefix="lorasmoke_rank_"))
        try:
            m5 = _fake_mount(root5)
            cfg = json.loads((m5 / "lora-noop" / "adapter_config.json").read_text())
            cfg["r"] = 32
            (m5 / "lora-noop" / "adapter_config.json").write_text(json.dumps(cfg))
            ns5 = _defs_namespace(text, root5)
            expect_raises("D10 adapter rank != --max-lora-rank => FATAL",
                          ns5["_lora_find_adapters"], "LORA-CANARY FATAL: lora-noop rank")
        finally:
            shutil.rmtree(root5, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop("TAAF_KAGGLE_INPUT_PATHS", None)
        os.environ.pop("TAAF_KAGGLE_WORKING_DIR", None)


def _fallback_is_kaggle_input(ns: dict) -> bool:
    saved = os.environ.pop("TAAF_KAGGLE_INPUT_PATHS", None)
    try:
        return ns["resolve_kaggle_dataset_path"]("canivel", ADAPTER_SLUG) == Path(
            "/kaggle/input") / ADAPTER_SLUG
    finally:
        if saved is not None:
            os.environ["TAAF_KAGGLE_INPUT_PATHS"] = saved


# ---------------------------------------------------------------------------
# A - the shipped adapter artifacts
# ---------------------------------------------------------------------------
def section_adapters(text: str) -> None:
    print("\n[A] adapter artifacts (local == manifest == what the notebook pins)")
    import hashlib
    manifest = json.loads((ADAPTER_DIR / "PROBE_ADAPTERS.json").read_text(encoding="utf-8"))
    for sub, key in (("lora-noop", "noop"), ("lora-probe", "probe")):
        w = ADAPTER_DIR / sub / "adapter_model.safetensors"
        raw = w.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        check(f"A1[{sub}] size matches the manifest AND the published dataset (41,962,184 B)",
              len(raw) == manifest[key]["bytes"] == 41962184, f"{len(raw)} B")
        check(f"A2[{sub}] sha16 matches the manifest", digest == manifest[key]["sha256_16"],
              digest)
        check(f"A3[{sub}] the notebook pins that same sha", digest in text)
    check("A4 noop has ZERO non-zero B modules (its delta is exactly zero by construction)",
          manifest["noop"]["nonzero_lora_B_modules"] == 0)
    check("A5 probe has 64 non-zero B modules (its delta CANNOT be zero)",
          manifest["probe"]["nonzero_lora_B_modules"] == 64)
    check("A6 the two adapters are NOT the same bytes (a copy-paste would fake a PASS)",
          manifest["noop"]["sha256_16"] != manifest["probe"]["sha256_16"])
    check("A7 same tensor count / param count, so the ONLY difference is B",
          manifest["noop"]["tensors"] == manifest["probe"]["tensors"] == 128
          and manifest["noop"]["params"] == manifest["probe"]["params"] == 10485760)


# ---------------------------------------------------------------------------
# M - arithmetic + the MM probe
# ---------------------------------------------------------------------------
def section_math(text: str) -> None:
    print("\n[M] projection arithmetic and the MM probe")
    check("M1 tokens-per-action derived from the frozen 27B anchor (192*7920/480)",
          "TOKENS_PER_ACTION = 3168.0" in text)
    check("M2 bar 100 actions/window present", "ACTION_BAR = 100.0" in text)
    check("M3 harness concurrency 28 used for the throughput measurement",
          "CONCURRENCY = 28" in text)
    check("M4 bar is 40.0 agg tok/s", abs(100.0 * 3168.0 / 7920.0 - 40.0) < 1e-9)
    check("M5 results persisted as an artifact", "lora_canary.json" in text)
    root = Path(tempfile.mkdtemp(prefix="lorasmoke_png_"))
    try:
        _fake_mount(root)
        ns = _defs_namespace(text, root)
        png = base64.b64decode(ns["_lora_png_b64"]())
        check("M6 MM probe emits a real PNG (magic + IHDR 64x64 + valid CRC)",
              png[:8] == b"\x89PNG\r\n\x1a\n"
              and struct.unpack(">II", png[16:24]) == (64, 64)
              and zlib.crc32(png[12:16] + png[16:16 + 13]) & 0xFFFFFFFF
              == struct.unpack(">I", png[29:33])[0])
    finally:
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop("TAAF_KAGGLE_INPUT_PATHS", None)
        os.environ.pop("TAAF_KAGGLE_WORKING_DIR", None)


# ---------------------------------------------------------------------------
# V - the sealed scorer must exist and must pass its own selftest
# ---------------------------------------------------------------------------
def section_scorer() -> None:
    print("\n[V] sealed verdict scorer")
    check("V1 a sealed scorer exists for this canary", SCORER.is_file(), str(SCORER))
    if not SCORER.is_file():
        return
    r = subprocess.run([sys.executable, str(SCORER), "--selftest"],
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    check("V2 scorer selftest exits 0", r.returncode == 0, tail)


# ---------------------------------------------------------------------------
# P - preflight's host gates, run as a library (preflight itself is post-push)
# ---------------------------------------------------------------------------
def section_preflight() -> None:
    """`scripts/preflight.py --family duck-harness` PULLS the kernel (K1), so it cannot
    run before the push; the structural gate it would apply is section [S] above. Its
    host gates H1-H4, however, are pure functions and DO run here. They were silently
    inapplicable to this slug until 2026-08-14 (`_family_of` tags by the substring
    'duck'), which made H1/H4 report a vacuous OK — checked here so it stays fixed."""
    print("\n[P] preflight host gates (pure functions; preflight itself is POST-push)")
    sys.path.insert(0, str(REPO / "scripts"))
    import preflight as P  # noqa: PLC0415

    kernel = "canivel/arc3-lora-serve-canary"
    check("P1 this slug is tagged as the duck lineage (H1/H4 actually apply)",
          P._family_of(kernel) == "duck", P._family_of(kernel))
    nb = json.loads(OUT_NB.read_text(encoding="utf-8"))
    kmeta = json.loads(OUT_META.read_text(encoding="utf-8"))
    gates = {c["check"]: c for c in P.host_gates(kernel, nb, kmeta, strict=True)}
    for code in ("H1", "H2", "H3", "H4"):
        msg = gates[code]["message"]
        check(f"P2[{code}] OK and NOT a vacuous n/a",
              gates[code]["status"] == "OK" and "n/a" not in msg, msg[:90])


def main() -> int:
    print("LORA SERVE CANARY SMOKE")
    section_build()
    section_structure()
    text = _rewritten_setup_text()
    section_rewrite(text)
    section_mount(text)
    section_adapters(text)
    section_math(text)
    section_preflight()
    section_scorer()
    print(f"\n{len(PASS)} passed / {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
