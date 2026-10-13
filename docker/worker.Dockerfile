FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/worker.requirements.txt /app/requirements.txt
RUN python -m pip install --only-binary=:all: -r requirements.txt && python -m pip check
RUN useradd --uid 10001 --create-home serving
COPY src/citypulse_ai /app/src/citypulse_ai
USER 10001:10001
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)" || exit 1
CMD ["python", "-m", "uvicorn", "citypulse_ai.api.lstm_worker:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
