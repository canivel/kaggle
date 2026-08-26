# Learning: Docker Image Controls EVERYTHING

## The Problem
Without the custom docker image, kernel_sources don't mount and model paths differ.

## Evidence
- v18 (no docker, kernel_sources in metadata): kernel_sources mounted, model at /models/danielhanchen/...
- v19 (no docker, no kernel_sources): openai_harmony ImportError
- v20 (no docker, kernel_sources): worked! model at /models/danielhanchen/...
- v21 (no docker, kernel_sources, 44/50 model path): FAILED — /gpt-oss-120b/ doesn't exist
- v22 (WITH docker, kernel_sources): should work — docker mounts model at /gpt-oss-120b/

## The Rule
The custom docker image `gcr.io/kaggle-private-byod/python@sha256:536e3d...`:
1. Mounts kernel_sources at /kaggle/input/<slug>/
2. Mounts models at /kaggle/input/<model-name>/ (not /kaggle/input/models/<owner>/)
3. Pre-installs compatible CUDA/torch versions
4. Both 44/50 notebooks use this exact docker image

WITHOUT docker image:
- kernel_sources may or may not mount (inconsistent)
- Models mount at /kaggle/input/models/<owner>/<model>/<framework>/<variant>/<version>
- Pre-installed packages may differ

## Impact
Must use the docker image for reproducible behavior. All top notebooks use it.
