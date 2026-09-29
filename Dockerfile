# One image for the API and both worker types; the command decides the role.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_CACHE_DIR=/cache \
    HF_HOME=/cache/huggingface \
    BLOB_ROOT=/data/blobs

# Runtime libraries Docling's PDF/image stack needs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ingest,api,s3,azure]"
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /cache /data/blobs \
 && chown -R app /cache /data
USER app

EXPOSE 8000
# One process per container; scale with replicas (docker compose --scale / Kubernetes HPA).
CMD ["python", "-m", "rag.serve", "--host", "0.0.0.0", "--port", "8000"]
