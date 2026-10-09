FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/huggingface
WORKDIR /app
COPY pyproject.toml ./
COPY flux_engine ./flux_engine
RUN python -m pip install --no-cache-dir '.[model]'
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready', timeout=4)"
CMD ["uvicorn", "flux_engine.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
