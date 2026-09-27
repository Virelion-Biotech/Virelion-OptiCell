FROM python:3.12-slim

ARG OPTICELL_EXTRAS=ui
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPTICELL_WORKSPACE=/data/opticell \
    OPTICELL_DB=/data/opticell/workbench.sqlite3 \
    OPTICELL_DATA_ROOT=/data/input \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[${OPTICELL_EXTRAS}]" && \
    useradd --create-home --uid 10001 opticell && \
    mkdir -p /data/opticell /data/input && chown -R opticell:opticell /data /app

USER opticell
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"
CMD ["streamlit", "run", "app_streamlit.py"]
