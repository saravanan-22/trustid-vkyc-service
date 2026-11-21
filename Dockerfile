FROM python:3.12-slim

# libgl1/libglib2.0-0: OpenCV runtime. tesseract-ocr: server-side document OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fetch and checksum-verify the face models at build time.
COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py --dest /app/models

COPY app/ ./app/
COPY static/ ./static/
COPY client_sdk/ ./client_sdk/

RUN useradd --create-home --uid 10001 trustid \
    && mkdir -p /app/data \
    && chown -R trustid:trustid /app/data
USER trustid

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    WEB_CONCURRENCY=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# More than one worker requires REDIS_URL (see docker-compose.yml).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
