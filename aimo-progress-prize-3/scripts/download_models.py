"""Download and prepare models for Kaggle upload.

Models must be uploaded as Kaggle datasets since the competition
has no internet access during inference.

Usage:
    uv run python scripts/download_models.py --model nemotron-14b
    uv run python scripts/download_models.py --model numinamath-7b
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


MODELS = {
    "nemotron-14b": {
        "repo": "nvidia/OpenMath-Nemotron-14B-Kaggle",
        "local_dir": "models/openmath-nemotron-14b-kaggle",
        "description": "AIMO2 winner model, 14B params, TIR-trained",
    },
    "nemotron-32b": {
        "repo": "nvidia/OpenMath-Nemotron-32B",
        "local_dir": "models/openmath-nemotron-32b",
        "description": "Full-size Nemotron, 32B params",
    },
    "numinamath-7b": {
        "repo": "AI-MO/NuminaMath-7B-TIR",
        "local_dir": "models/numinamath-7b-tir",
        "description": "AIMO1 winner, 7B params, good for local dev",
    },
    "deepseek-r1-32b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "local_dir": "models/deepseek-r1-distill-qwen-32b",
        "description": "DeepSeek R1 distilled, strong reasoning",
    },
}


def download_model(model_key: str):
    """Download a model from HuggingFace."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        return

    info = MODELS[model_key]
    local_dir = Path(info["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {info['repo']} to {local_dir}/")
    print(f"Description: {info['description']}")

    snapshot_download(
        repo_id=info["repo"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nDone! Model saved to {local_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Create a Kaggle dataset from {local_dir}/")
    print(f"     kaggle datasets create -p {local_dir}/ -r zip")
    print(f"  2. Or upload via Kaggle web UI")
    print(f"  3. Add as input to your submission notebook")
    print(f"  4. Reference as: /kaggle/input/<dataset-slug>/")


def main():
    parser = argparse.ArgumentParser(description="Download models for AIMO3")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="Model to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list:
        for key, info in MODELS.items():
            print(f"  {key:20s} {info['repo']:50s} {info['description']}")
        return

    download_model(args.model)


if __name__ == "__main__":
    main()
