# 1. Use the official OpenEnv base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# 2. Set the working directory to /app
WORKDIR /app

# 3. Install git (required for some python dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 4. Install uv (as used in your pyproject.toml)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# 5. Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml .
# Note: Uncomment the line below if you have a uv.lock file in your root
# COPY uv.lock . 

# 6. Install dependencies defined in pyproject.toml
RUN uv sync --no-install-project --no-editable

# 7. Copy the ENTIRE project into the container
# This includes /data, /server, models.py, and openenv.yaml
COPY . .

# 8. Install the project itself
RUN uv sync --no-editable

# --- Final Runtime Stage ---
FROM ${BASE_IMAGE}

WORKDIR /app

# 9. Copy the virtual environment and all project files from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# 10. Set environment variables so the app and imports work
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# 11. Health check for the FastAPI server
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Start the FastAPI server using the entry point from openenv.yaml
# We run from /app so the 'data/' folder is reachable at './data/'
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]