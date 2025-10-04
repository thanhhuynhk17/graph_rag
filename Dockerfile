# Base image
FROM python:3.11-slim

WORKDIR /app

# Install uv (instead of pip)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

# Copy project metadata first (better caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies via uv
RUN uv sync --frozen --no-dev

# Copy source code
COPY ./src ./src

# Expose port
EXPOSE 8000

# Default command to run Uvicorn
CMD ["uv", "run", "uvicorn", "src.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
