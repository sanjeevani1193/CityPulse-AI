FROM python:3.14.7-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/gateway.requirements.txt /app/requirements.txt
RUN python -m pip install --only-binary=:all: -r requirements.txt && python -m pip check
RUN useradd --uid 10001 --create-home serving
COPY src/citypulse_ai /app/src/citypulse_ai
USER 10001:10001
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["python", "-m", "uvicorn", "citypulse_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
