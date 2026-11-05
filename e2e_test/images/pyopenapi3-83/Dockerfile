FROM --platform=linux/amd64 python:3.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

ARG TASK_REPO_COMMIT=2ef34c3213eb292703e0e5e6f2185b1e4725bbde

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    procps \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /testbed

RUN git clone https://github.com/Algebra8/pyopenapi3.git . \
    && git checkout --detach "${TASK_REPO_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${TASK_REPO_COMMIT}"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt -e ".[dev]" "pytest==6.2.2"

LABEL org.opencontainers.image.source="https://github.com/Algebra8/pyopenapi3" \
      org.opencontainers.image.revision="2ef34c3213eb292703e0e5e6f2185b1e4725bbde" \
      org.opencontainers.image.description="SWE-ReBench task environment for Algebra8/pyopenapi3-83"

CMD ["sleep", "infinity"]
