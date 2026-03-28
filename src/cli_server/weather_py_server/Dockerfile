# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for faster startups
ENV UV_COMPILE_BYTECODE=1
# Copy only the dependency files first for better caching
COPY pyproject.toml uv.lock ./ 

# Install dependencies into a virtualenv at /app/.venv
# --frozen ensures we use the exact versions in the lockfile
RUN uv sync --frozen --no-install-project --no-dev

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy your actual code
COPY src/cli_server/weather_py_server/*.py ./

# Set the path so the system finds the virtualenv's python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Run the CLI
EXPOSE 8000
CMD ["python", "weather_cli.py", "8000"]