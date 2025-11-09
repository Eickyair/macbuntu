FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir --prefix=/install -r requirements.txt; \
    else \
        pip install --no-cache-dir --prefix=/install "fastapi[standard]" uvicorn; \
    fi && \
    find /install -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete && \
    find /install -name "*.dist-info" -type d -exec rm -rf {}/RECORD {}/INSTALLER {} + 2>/dev/null || true && \
    find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]