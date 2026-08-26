"""EAGLE-3 speculative decoding integration for AIMO3.

Adds the --speculative-config flag to vLLM server startup.
This gives +36-42% inference speedup with IDENTICAL output distribution.

Requirements:
- vLLM >= 0.7 (Kaggle has 0.11.2 ✓)
- Draft model at /kaggle/input/gpt-oss-120b-eagle3-aimo3/ (uploaded as dataset)
- gpu_memory_utilization reduced to 0.93 (draft model needs ~0.6GB)

The only change to _start_server():
    cmd.extend(['--speculative-config', json.dumps({
        "method": "eagle3",
        "model": EAGLE3_DRAFT_PATH,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    })])

And gpu_memory_utilization from 0.99 to 0.93.
"""

import json
import glob

# Find the draft model path
EAGLE3_DRAFT_PATH = None
for candidate in glob.glob('/kaggle/input/**/config.json', recursive=True):
    d = candidate.replace('/config.json', '')
    if 'eagle3' in d.lower():
        EAGLE3_DRAFT_PATH = d
        break

EAGLE3_SPEC_CONFIG = {
    "method": "eagle3",
    "model": EAGLE3_DRAFT_PATH or "/kaggle/input/gpt-oss-120b-eagle3-aimo3",
    "num_speculative_tokens": 3,
    "draft_tensor_parallel_size": 1,
}


def get_eagle3_server_args():
    """Return additional vLLM server args for EAGLE-3."""
    if EAGLE3_DRAFT_PATH is None:
        print("WARNING: EAGLE-3 draft model not found. Running without speculative decoding.")
        return []
    return ['--speculative-config', json.dumps(EAGLE3_SPEC_CONFIG)]
