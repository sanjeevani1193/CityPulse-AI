FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SQLPILOT_MCP_PYTHON=/opt/sqlpilot-core/bin/python \
    SQLPILOT_DATABASE=/data/sample.duckdb
WORKDIR /app
COPY requirements/sqlpilot-app.txt requirements/sqlpilot-app.lock.txt /app/requirements/
RUN python -m pip install --only-binary=:all: -r requirements/sqlpilot-app.txt \
    && python -m pip check \
    && python -m venv /opt/sqlpilot-core \
    && /opt/sqlpilot-core/bin/python -m pip install --only-binary=:all: duckdb==1.5.6 pytz==2026.5 \
    && useradd --uid 10001 --create-home sqlpilot
COPY src/citypulse_ai/__init__.py /app/src/citypulse_ai/__init__.py
COPY src/citypulse_ai/sqlpilot /app/src/citypulse_ai/sqlpilot
COPY .streamlit/config.toml /app/.streamlit/config.toml
USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "src/citypulse_ai/sqlpilot/ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
