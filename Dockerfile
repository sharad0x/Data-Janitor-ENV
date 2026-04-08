# 1. Use the official OpenEnv base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (git/curl are needed for uv/pip)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# 4. Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# 5. Copy dependency files from the ROOT
# Since the Dockerfile is now in the root, pyproject.toml is at the same level
COPY pyproject.toml .

# 6. Install dependencies defined in pyproject.toml
RUN uv sync --no-install-project --no-editable

# 7. Copy the entire project into the container
# This picks up /data, models.py, openenv.yaml, and the /server folder
COPY . .

# 8. Finalize the uv installation to include the project itself
RUN uv sync --no-editable

# --- Final Runtime Stage ---
FROM ${BASE_IMAGE}
WORKDIR /app

# 9. Copy the virtual environment and project from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# 10. Set paths so your code and models.py are correctly importable
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# 11. Start the server
# (Custom HEALTHCHECK removed because Hugging Face handles port monitoring natively)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]