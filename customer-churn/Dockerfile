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
COPY customer-churn/pyproject.toml /app/
COPY customer-churn/run_iter5.py /app/
COPY customer-churn/run_iter5_submit.py /app/
COPY customer-churn/data/ /app/data/

# Install dependencies
RUN uv sync

# Default: run the full iter5
CMD ["uv", "run", "python", "-u", "run_iter5.py"]
