FROM python:3.14.7-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CITYPULSE_STATION_ARTIFACT_DIR=/app/artifacts
WORKDIR /app
RUN python -c "import platform; assert platform.machine() == 'x86_64', 'Build with --platform=linux/amd64'" \
    && apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/cloudrun-xgboost.requirements.txt /app/requirements.txt
RUN python -m pip install --only-binary=:all: -r requirements.txt && python -m pip check \
    && useradd --uid 10001 --create-home serving
COPY src/citypulse_ai /app/src/citypulse_ai
COPY artifacts /app/artifacts
RUN python -c "import sys; from citypulse_ai.api.station_runtime import StationRuntime; StationRuntime().load('xgboost'); assert not any(n in sys.modules for n in ('torch','tensorflow','sklearn'))" \
    && chmod -R a-w /app/artifacts
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/health',timeout=2)"
CMD ["python", "-m", "citypulse_ai.api.cloudrun_xgboost"]
