
# ===== CELL 0 [markdown] =====
# Tufa Labs ARC3 submission

![Tufa Labs](attachment:tufa_labs.png)

**Note**: this notebook is a more readable version of the notebook that scored our milestone-winning 1.21; unfortunately, we haven't had the same lucky result with this one. The original one is also shared here, but using it is not recommended: https://www.kaggle.com/code/jeroencottaar/taaf-duck-harness-kaggle

**Note**: if you make a copy of this notebook, you will have to manually select the proper GPU (RTX Pro 6000).

Link to writeup explaining what's going on here: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133

Link to Machine Learning Street Talk interview by Tim Scarfe about this duck harness: https://x.com/MLStreetTalk/status/2072326433922297975?s=20

This notebook executes the ARC-AGI-3 solver written by the Tufa Labs team; in alphabetical order: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar, and Stefano Viel.

You will only find infrastructure and diagnostics in this notebook; the actual solver code is in an attached dataset. See our writeup on the competition forum to learn more about that the solver actually does.

It installs the ARC runtime from the competition wheelhouse, makes the bundled source
snapshot importable, runs any solver setup commands, loads the pickled benchmark, plays the
competition games, and writes results to `/kaggle/working`. Diagnostics are minimised during
a real competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) and kept full otherwise.

# ===== CELL 1 [markdown] =====
## 1. Environment and submission mode

Detect whether this is a real competition rerun (which minimises diagnostics), set the
framework's environment flags, and put the CUDA libraries on the linker path.

# ===== CELL 2 [code] =====
import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# In submission, disable the periodic JSON/HTML diagnostics writes and per-frame logging.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# Prepend the CUDA toolkit to the linker path (it is off it on Kaggle GPU images) so the
# solver's GPU libraries (e.g. vllm / torch) can link against libcuda.
cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")

# --- Warpack fast-submit gate (fork-band adoption R1a) -----------------------
# Heavy setup (wheel install, source import, vLLM boot, benchmark run) only
# happens in a real competition rerun, or when explicitly forced for a
# build-time offline eval (WARPACK_FORCE_OFFLINE_BENCH=1). An interactive
# Save Version therefore finishes in seconds with a dummy submission.parquet
# (see the run cell) -> daily resubmission costs ~no GPU quota; the official
# score is the max over stochastic reruns.
FORCE_OFFLINE_BENCH = os.environ.get("WARPACK_FORCE_OFFLINE_BENCH", "").strip().lower() in {"1", "true"}
RUN_HEAVY = TRUE_SUBMISSION or FORCE_OFFLINE_BENCH
print(f"taaf.kaggle: RUN_HEAVY={RUN_HEAVY} (fast-submit gate {'off' if RUN_HEAVY else 'ON'})")


# ===== CELL 3 [markdown] =====
## 2. Install the ARC runtime

Install `arc-agi` from the offline competition wheelhouse (the Kaggle submission environment
has no internet).

# ===== CELL 4 [code] =====
# Warpack fast-submit gate: heavy setup only in a real rerun (or forced
# offline bench). Interactive Save Version skips this cell in ~0s.
if RUN_HEAVY:
    # Install the ARC runtime from the bundled competition wheels.
    # Quiet: stdout is discarded; stderr (and a non-zero exit) still surface real failures.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-warn-conflicts",
            "--disable-pip-version-check",
            "--find-links",
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
            "arc-agi",
        ],
        stdout=subprocess.DEVNULL,
    )

# ===== CELL 5 [markdown] =====
## 3. Locate the source bundle

Find the uploaded TAAF source dataset by its marker file, and record where Kaggle mounted
every attached input so setup commands and the solver can find them.

# ===== CELL 6 [code] =====
# Warpack fast-submit gate: heavy setup only in a real rerun (or forced
# offline bench). Interactive Save Version skips this cell in ~0s.
if RUN_HEAVY:
    # Kaggle inputs attached to this notebook, plus bookkeeping paths used below.
    DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share", "driessmit1/arc3-vllm-h100-wheelhouse-v3", "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]
    KERNEL_SOURCES = []
    DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
    SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"


    # Locate the source dataset by its marker file rather than a fixed mount path.
    def _find_bundle_dir() -> Path:
        for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
            return marker.parent
        raise RuntimeError("TAAF source bundle not found under /kaggle/input.")


    # Kaggle mounts a dataset at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>
    # (depending on owner / slug collisions), so probe both and use whichever exists. Utility
    # scripts mount under /kaggle/usr/lib/notebooks/<owner>/<slug>.
    def _dataset_mount_candidates(ref: str) -> list[Path]:
        owner, slug = ref.split("/", 1)
        return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


    def _kernel_mount_candidates(ref: str) -> list[Path]:
        owner, slug = ref.split("/", 1)
        return [Path("/kaggle/usr/lib/notebooks") / owner / slug]


    def _first_existing(candidates: list[Path]) -> Path | None:
        return next((c for c in candidates if c.exists()), None)


    BUNDLE_DIR = _find_bundle_dir()
    print(f"taaf.kaggle: source bundle = {BUNDLE_DIR}")

    # Map each attached input to where Kaggle actually mounted it (the source bundle is index 0).
    kaggle_input_paths: dict[str, str] = {}
    for i, ref in enumerate(DATASET_SOURCES):
        candidates = _dataset_mount_candidates(ref)
        resolved = BUNDLE_DIR if i == 0 else _first_existing(candidates)
        kaggle_input_paths[ref] = str(resolved or candidates[0])
    for ref in KERNEL_SOURCES:
        candidates = _kernel_mount_candidates(ref)
        kaggle_input_paths[ref] = str(_first_existing(candidates) or candidates[0])

    # Published to setup commands and the solver via the environment:
    setup_env = {
        # JSON {ref: mount_path} so they can locate every attached dataset / utility script.
        "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
        # The attached dataset refs in order (index 0 is this source bundle).
        "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
        # The attached utility-script / kernel refs.
        "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
    }
    os.environ.update(setup_env)
    SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\n")
    print(f"taaf.kaggle: input paths = {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")

# ===== CELL 7 [markdown] =====
## 4. Import the bundled source and run solver setup

Put the snapshotted repositories on the path (this process and any child processes), then run
the solver's setup commands — installing wheels, fetching model weights, and so on.

# ===== CELL 8 [code] =====
# Warpack fast-submit gate: heavy setup only in a real rerun (or forced
# offline bench). Interactive Save Version skips this cell in ~0s.
if RUN_HEAVY:
    # Each bundled repo exposes its importable tree at <repo>/src or <repo>.
    def _source_path_entries(bundle_dir: Path) -> list:
        entries = []
        for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
            for candidate in (repo / "src", repo):
                if candidate.is_dir():
                    entries.append(candidate)
        return entries


    # Environment handed to each setup command (paths + any keys it has persisted).
    def _command_env() -> dict:
        env = os.environ.copy()
        # "$PYTHON" in a command resolves to this notebook's interpreter.
        env["PYTHON"] = sys.executable
        # Absolute path to the mounted source bundle.
        env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
        # The writable /kaggle/working directory.
        env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
        # A command writes a JSON object here to persist env keys to later commands + the run.
        env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
        env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
        return env


    # Make the bundled repos importable here (sys.path) and in child processes (.pth).
    source_entries = _source_path_entries(BUNDLE_DIR)
    for entry in source_entries:
        sys.path.insert(0, str(entry))
    pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
    pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
    print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")

    # Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
    env = _command_env()
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
        # Re-read in case the command persisted new env keys.
        env = _command_env()
        os.environ.update(env)

    # Honour any PYTHONPATH a setup command exported.
    for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
        if entry not in sys.path:
            sys.path.insert(0, entry)

# ===== CELL 9 [markdown] =====
## 5. Load the benchmark

Unpickle the deployment target and the benchmark, stamping the real submission state onto the
target and pointing the benchmark's outputs at the Kaggle working directory.

# ===== CELL 10 [code] =====
# Warpack fast-submit gate: heavy setup only in a real rerun (or forced
# offline bench). Interactive Save Version skips this cell in ~0s.
if RUN_HEAVY:
    # Restore the deployment target and record the real submission state on it.
    with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
        target = pickle.load(file)
    target.actual_run_as_submission = TRUE_SUBMISSION
    target.is_competition_rerun = TRUE_SUBMISSION

    # Restore the benchmark and point its outputs at the Kaggle working dir.
    with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
        bm = pickle.load(file)
    bm.job_dir = WORKING_DIR

# ===== CELL 11 [markdown] =====
## 6. Customization hook

Optional: tweak `bm`, `bm.games`, or `bm.solver` here before the run starts — the safe place
for one-off experiments once the deployed bundle has loaded.

# ===== CELL 12 [code] =====
# ============================================================================
# Cell 12 - Customization hook: warpack fork-band adoption grafts
# Paste this whole block into the duckfork notebook's customization-hook cell.
# Runs AFTER `bm` is unpickled (cell 10) and the bundled sources are
# importable (cell 8). With the fast-submit gate, this cell is inert during
# an interactive Save Version (RUN_HEAVY False -> bm is None).
#
# Sourcing order for the warpack module:
#   1. an attached Kaggle dataset containing warpack_patch.py (marker file,
#      mount-path agnostic)
#   2. a warpack/ directory next to this notebook (local interactive runs)
#
# Failure policy: unless WARPACK_STRICT=1, any patch failure prints a
# traceback and the run continues as VANILLA duck (score = baseline, never 0).
# ============================================================================
import os
import sys
import traceback
from pathlib import Path

WARPACK_MARKER = "warpack_patch.py"
WARPACK_STRICT = os.environ.get("WARPACK_STRICT", "").strip() in {"1", "true"}

# --- warpack arm configuration (env-driven; defaults = all grafts ON)
os.environ.setdefault("WARPACK_ENABLE", "1")         # master kill switch
os.environ.setdefault("WARPACK_BANKING", "1")        # max-over-plays win replay
os.environ.setdefault("WARPACK_RECOVERY", "1")       # GAME_OVER-loop / lock-in refresh
os.environ.setdefault("WARPACK_SHORTCIRCUIT", "1")   # stop homogeneous no-op batches
os.environ.setdefault("WARPACK_RETRY_GUARD", "1")    # report-only counters
os.environ.setdefault("WARPACK_BANK_MIN_TIME", "120")


def _find_warpack_dir() -> Path | None:
    candidates: list[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(marker.parent for marker in kaggle_input.rglob(WARPACK_MARKER))
    here = Path.cwd()
    for probe in (here / "warpack", here, here.parent / "warpack"):
        if (probe / WARPACK_MARKER).is_file():
            candidates.append(probe)
    return candidates[0] if candidates else None


if not RUN_HEAVY:  # noqa: F821 - defined in cell 2 (fast-submit gate)
    print("warpack: fast-submit save (RUN_HEAVY=False) - customization skipped")
else:
    try:
        warpack_dir = _find_warpack_dir()
        if warpack_dir is None:
            raise FileNotFoundError(
                f"{WARPACK_MARKER} not found under /kaggle/input or ./warpack - "
                "attach the warpack dataset to the notebook."
            )
        if str(warpack_dir) not in sys.path:
            sys.path.insert(0, str(warpack_dir))
        import warpack_patch

        cfg = warpack_patch.apply(bm)  # noqa: F821 - bm from cell 10
        version = getattr(warpack_patch, "VERSION", "v1")
        bm.label = f"{bm.label}-warpack-{version}"  # noqa: F821
        print(f"warpack {version}: patches applied from {warpack_dir}")
        print(
            "warpack: banking={0.enable_banking} recovery={0.enable_recovery} "
            "shortcircuit={0.enable_shortcircuit} retry_guard={0.enable_retry_guard} "
            "bank_min_time={0.bank_min_time_s} bank_strict={0.bank_strict_frames} "
            "recovery_repeats={0.recovery_repeat_threshold} "
            "recovery_gameovers={0.recovery_gameover_threshold}".format(cfg)
        )
    except Exception:
        if WARPACK_STRICT:
            raise
        print("warpack: PATCH FAILED - continuing with VANILLA duck harness")
        traceback.print_exc()


# ===== CELL 13 [markdown] =====
## 7. Run the benchmark

In a real competition rerun (`KAGGLE_IS_COMPETITION_RERUN`), wait for the Kaggle gateway and
play the **live competition Arcade**. Otherwise — an interactive "Save & Run" — play the
competition's **bundled environment files offline**, with no gateway required, so the notebook
runs end-to-end without a submission. Teardown commands run afterward even if the run raises.

# ===== CELL 14 [code] =====
# Fast-submit path: during an interactive Save Version (gate false) write a
# dummy submission.parquet in seconds and skip the benchmark entirely.
def _write_dummy_submission(working_dir):
    # Same schema as the vanilla offline path: Kaggle only requires that a
    # submission.parquet exists after Save Version; the rerun rewrites it.
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet(working_dir / "submission.parquet", index=False)


if not RUN_HEAVY:
    _write_dummy_submission(WORKING_DIR)
    print(
        "taaf.kaggle: FAST-SUBMIT - dummy submission.parquet written "
        f"{time.time() - NOTEBOOK_START_EPOCH:.1f}s after start; benchmark skipped"
    )
else:
    # Build the live competition game list from the gateway's available environments.
    def _competition_games():
        import arc_agi

        import taaf.game_api

        spec = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=os.environ["ARC_BASE_URL"],
            environments_dir="",
        )
        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=spec.arc_base_url,
            environments_dir="",
        )
        game_ids = [env_info.game_id for env_info in arcade.available_environments]
        if not game_ids:
            raise RuntimeError("Competition Arcade exposed zero environments.")
        return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]

    # Build the offline game list from the competition's bundled environment files.
    def _offline_games(env_dir: str):
        import arc_agi

        import taaf.game_api

        spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
        game_ids = [env_info.game_id for env_info in arcade.available_environments]
        if not game_ids:
            raise RuntimeError(f"No offline environments found under {env_dir}.")
        return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]

    # The gateway can take a while to come up; poll until it answers.
    def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{base_url}api/games", timeout=10) as response:
                    if response.status < 500:
                        return
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(5)
        raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")

    # Print the run preamble and persist the launcher's git status for diagnostics.
    print((BUNDLE_DIR / "preamble.txt").read_text())
    (WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())

    # arc_agi reads RECORDINGS_DIR and ARC_API_KEY from env (ArcadeSpec carries neither); operation
    # mode, environments dir, and base url are all passed explicitly via the spec, so no env is needed.
    os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

    if TRUE_SUBMISSION:
        # Real submission: play the live competition Arcade served by the Kaggle gateway.
        os.environ.setdefault("ARC_API_KEY", "test-key-123")
        os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
        # The gateway boots asynchronously; wait before swapping in its game list.
        _wait_for_gateway(os.environ["ARC_BASE_URL"])
        bm.games = _competition_games()
    else:
        # Forced offline bench: play the bundled competition environments (no gateway).
        competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
        bm.games = _offline_games(competition_env_files)

    bm.n_passes = 1
    bm.game_weights = None

    # Warpack safety pack (fork-band adoption R1b): a real rerun gets
    # soft_end = start + 11h20m (vanilla had None) so the solver drains and
    # the shared scorecard closes before Kaggle's 12h hard kill; a hard kill
    # would zero the whole run. Offline bench keeps the vanilla budget rule.
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(hours=11, minutes=20)
    if not TRUE_SUBMISSION:
        budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
        if budget > 0:
            soft_end = min(soft_end, datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2)))

    # Play the benchmark; teardown commands run even if the run raises.
    try:
        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
        if not TRUE_SUBMISSION:
            # An offline run isn't scored, but Kaggle still expects a submission.parquet output.
            _write_dummy_submission(WORKING_DIR)
    finally:
        for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
            print(f"taaf.kaggle: teardown command: {command}", flush=True)
            subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())


# ===== CELL 15 [markdown] =====
## 8. Show the diagnostics

A non-submission run writes `diagnostics.html` to `/kaggle/working`; it is rendered inline below
(and downloadable from the working directory). You should be able to click around through the links.

# ===== CELL 16 [code] =====
from html import escape

from IPython.display import HTML, display

diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    # Isolate the full document in an iframe so its styles don't leak into the notebook.
    display(
        HTML(
            f'<iframe srcdoc="{escape(diagnostics_html.read_text(), quote=True)}" '
            'width="100%" height="900" style="border:0"></iframe>'
        )
    )
else:
    print("No diagnostics.html — minimal diagnostics (real submission) suppresses it.")
