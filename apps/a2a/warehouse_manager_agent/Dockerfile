FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy workspace root files for dependency resolution
COPY pyproject.toml uv.lock ./

# Copy package files and source
COPY apps/a2a/warehouse_manager_agent ./apps/a2a/warehouse_manager_agent

ENV UV_COMPILE_BYTECODE=1

# Install dependencies for the warehouse-manager-agent workspace member
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package warehouse-manager-agent

ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose the A2A server port
EXPOSE 10001

WORKDIR /app/apps/a2a/warehouse_manager_agent

# Command to run the A2A server
CMD ["python", "app.py"]
