# Backend image for the TrialMatch-RAG live demo API (src/api/app.py).
#
# data/trials.db, indexes/bm25-critfields/, indexes/dense/qwen3.base.npz are
# multi-GB, gitignored, and built per machine (see README's Setup) — they are
# NOT baked into this image. Mount them as volumes and pass GEMINI_API_KEY_*
# via --env-file at `docker run`. CPU-only by default; src/api/state.py
# auto-detects a GPU if one is passed through (`docker run --gpus all ...`).
#
#     docker build -t trialmatch-api .
#     docker run -p 8000:8000 --env-file .env \
#         -v "$(pwd)/data:/app/data" -v "$(pwd)/indexes:/app/indexes" \
#         trialmatch-api
#
# The frontend (frontend/) is a separate static site, not part of this image
# — see README's "Live demo" section for how to point it at this backend.

FROM python:3.12-slim

# pyserini's BM25 search runs on Lucene, which needs a JVM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir uv

# Dependencies first, so this (slow: torch + transformers) layer is cached
# across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --inexact

COPY src/ ./src/
COPY prompts/ ./prompts/

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
