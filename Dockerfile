# syntax=docker/dockerfile:1
# Copyright 2026 Rahul Reddy
# Licensed under the Apache License, Version 2.0
# Unified AI-Assisted DevOps Platform - Production Multi-Stage Dockerfile

# ==============================================================================
# Stage 1: Build the distribution wheel
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Ensure build tooling is installed
RUN pip install --no-cache-dir --upgrade pip build hatchling

# Copy project metadata and source code required to build the distribution wheel
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src/ ./src/

# Build a clean pure-Python wheel without isolation (using pre-installed hatchling)
RUN python -m build --wheel --no-isolation

# ==============================================================================
# Stage 2: Hardened, minimal non-root runtime image
# ==============================================================================
FROM python:3.11-slim AS runtime

# Set security and operational environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    METRICS_HOST="0.0.0.0" \
    METRICS_PORT="9090" \
    INCIDENT_STORAGE_DIR="/app/.devops_ai/incidents" \
    RAG_VECTOR_STORE_PATH="/app/data/vector_store" \
    PATH="/home/devopsai/.local/bin:${PATH}"

# Create dedicated non-root user and group with UID/GID 10001
RUN groupadd -g 10001 devopsai && \
    useradd -u 10001 -g devopsai -m -s /bin/bash devopsai

WORKDIR /app

# Copy built wheel from the builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the wheel and its production dependencies; clean up wheel artifact
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Ensure persistent directories exist and are owned by the non-root user
RUN mkdir -p /app/data /app/.devops_ai/incidents && \
    chown -R devopsai:devopsai /app

# Switch to non-root user (UID:GID 10001:10001)
USER 10001:10001

# Expose default Prometheus metrics server port
EXPOSE 9090

# Periodic health check using the verified CLI health probe command
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["devops-ai", "health", "check"]

# Entrypoint delegates to the devops-ai CLI executable
ENTRYPOINT ["devops-ai"]

# Default command displays CLI usage information
CMD ["--help"]
