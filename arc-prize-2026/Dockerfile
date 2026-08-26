FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy kaggle-agent framework first (for caching)
COPY kaggle-agent/ /app/kaggle-agent/

# Copy competition files
COPY arc-prize-2026/pyproject.toml /app/
COPY arc-prize-2026/agent/ /app/agent/
COPY arc-prize-2026/run_baseline.py /app/
COPY arc-prize-2026/run_iter1_cnn.py /app/
COPY arc-prize-2026/config.yaml /app/
COPY arc-prize-2026/data/ /app/data/

# Install dependencies
RUN uv sync

# Default: run CNN training
CMD ["uv", "run", "python", "-u", "run_iter1_cnn.py"]
