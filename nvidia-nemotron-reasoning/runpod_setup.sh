#!/bin/bash
# RunPod setup script — run once after pod starts
# Template: RunPod PyTorch 2.5+ / CUDA 12.x
set -e

echo "=== RunPod Setup for Nemotron Reasoning ==="

# Install dependencies
pip install -q transformers>=4.47.0 peft>=0.14.0 trl>=0.13.0 \
    datasets>=3.2.0 accelerate>=1.2.0 bitsandbytes>=0.45.0 \
    mamba_ssm causal_conv1d \
    pandas numpy scikit-learn

# Download the model (cached for future runs)
python -c "
from huggingface_hub import snapshot_download
print('Downloading Nemotron-3-Nano-30B-A3B-BF16...')
snapshot_download('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', local_dir='/workspace/model')
print('Done.')
"

# Create workspace
mkdir -p /workspace/data /workspace/adapters /workspace/results

echo ""
echo "=== Setup complete ==="
echo "Upload train.csv to /workspace/data/"
echo "Upload experiment configs to /workspace/data/"
echo "Then run: python /workspace/gpu_worker.py --config /workspace/data/config.json"
