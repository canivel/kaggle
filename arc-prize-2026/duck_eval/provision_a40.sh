#!/usr/bin/env bash
# Provision a RunPod A40 pod for local duck-harness evaluation.
# Run ON THE POD after syncing duck_eval/ + kaggle-data/environment_files.
set -euxo pipefail

WORK=${WORK:-/workspace}
cd "$WORK"

# --- deps ---------------------------------------------------------------
pip install -q uv
uv pip install --system -e "$WORK/duck_eval/taaf_bundle/src/tufa-arc-agi-framework"
uv pip install --system -e "$WORK/duck_eval/taaf_bundle/src/ARC3-Inference"
uv pip install --system "vllm>=0.8" huggingface_hub

# --- model --------------------------------------------------------------
# Qwen3 27B FP8 snapshot (same one the winning kernel uses, mirrored on HF).
# A40 = Ampere: no FP8 hardware -> vLLM will dequant to BF16 (fits in 48GB
# for 27B at ~54GB BF16? NO — needs quantization). Use the AWQ-int8 fallback
# per the plan when BF16 doesn't fit:
MODEL_DIR="$WORK/models/qwen3-27b"
mkdir -p "$MODEL_DIR"
python - <<'EOF'
from huggingface_hub import snapshot_download
import os
# Preferred: the same FP8 snapshot family the winner used; vLLM handles
# marlin-dequant on Ampere. Fallback: AWQ int4/int8 build of the same model.
for repo in ["vrfai/qwen3-6-27b-fp8", "Qwen/Qwen3-27B-AWQ"]:
    try:
        snapshot_download(repo, local_dir=os.environ.get("MODEL_DIR", "/workspace/models/qwen3-27b"))
        print("downloaded:", repo)
        break
    except Exception as e:
        print("miss:", repo, repr(e))
EOF

# --- vLLM server --------------------------------------------------------
# Marlin/dequant path on Ampere; --max-model-len matches the kernel's 32k.
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --port 8000 \
  > "$WORK/vllm.log" 2>&1 &

echo "waiting for vLLM..."
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 10; done
echo "vLLM ready"

# --- tokens/s parity probe ----------------------------------------------
python - <<'EOF'
import json, time, urllib.request
body = json.dumps({
    "model": "/workspace/models/qwen3-27b",
    "prompt": "Write a python function that rotates a 64x64 grid 90 degrees.",
    "max_tokens": 512, "temperature": 0.0,
}).encode()
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
dt = time.time() - t0
n = resp["usage"]["completion_tokens"]
print(f"TOKENS_PER_SECOND={n/dt:.1f}  ({n} tokens in {dt:.1f}s)")
EOF
