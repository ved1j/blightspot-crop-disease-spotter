# CPU-only image: ~1 GB smaller than the default CUDA wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

RUN useradd --create-home blightspot && chown -R blightspot /app
USER blightspot

EXPOSE 8000
WORKDIR /app/backend
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
