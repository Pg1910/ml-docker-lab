# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Prevent MLflow/temp staging from ever targeting non-writable locations like /mlflow
    TMPDIR=/tmp \
    MLFLOW_TMP_DIR=/tmp \
    # Silence GitPython warning (install git if you want SHA instead)
    GIT_PYTHON_REFRESH=quiet

# System deps
# - ca-certificates: TLS
# - git: optional, enables Git SHA logging without warnings
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for caching
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src/ /app/src/

# Non-root user (good practice)
RUN useradd -m -u 10001 appuser
USER appuser

# Default artifacts location (mount a volume here)
ENV ARTIFACTS_DIR=/artifacts

ENTRYPOINT ["python", "-m", "src.train"]
