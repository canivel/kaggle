#!/usr/bin/env bash
# One seeded local eval sweep of the duck harness against TAAF's
# competition_arcade simulator. Produces runs/duck_eval/<tag>_seed<N>.json
#
# Usage (on the pod, vLLM already serving on :8000):
#   bash run_local_eval.sh baseline 1
#   bash run_local_eval.sh baseline 2 ...
set -euxo pipefail

TAG=${1:?tag}
SEED=${2:?seed}
WORK=${WORK:-/workspace}
OUT_DIR="$WORK/runs/duck_eval"
mkdir -p "$OUT_DIR"

cd "$WORK/duck_eval/taaf_bundle/src/ARC3-Inference"

# The bundle ships a pickled Benchmark (deploy_target.pkl + benchmark_initial.pkl)
# whose run logic lives in inference/framework/run.py. The
# --simulate-competition-arcade flag runs TAAF's localhost Arcade simulator
# (inline deployment target, n-passes 1) — the same interface the Kaggle
# gateway exposes, backed by the local engine files.
export PYTHONHASHSEED=$SEED
export TAAF_SEED=$SEED
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local"

python -m inference.framework.run \
  --deployment-target inline \
  --simulate-competition-arcade \
  --n-passes 1 \
  --seed "$SEED" \
  --output "$OUT_DIR/${TAG}_seed${SEED}.json" \
  2>&1 | tee "$OUT_DIR/${TAG}_seed${SEED}.log"

echo "wrote $OUT_DIR/${TAG}_seed${SEED}.json"
# NOTE: exact flag names may differ; run `python -m inference.framework.run --help`
# on the pod and adjust. The simulator + inline-target + single-pass invariants
# come from _enter_competition_arcade() in run.py (lines ~669-694).
