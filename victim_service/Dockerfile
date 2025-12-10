<<<<<<< HEAD
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY model_backend.py covert_channel.py server.py ./
RUN mkdir -p /app/logs

# Default environment
ENV MODEL_NAME=fake_a
ENV USE_REAL_MODELS=0
ENV COVERT_ENABLED=0
ENV COVERT_SLOT_MS=2.0
ENV OBFUSCATION_STRATEGY=none
ENV OBFUSCATION_PARAM=0
ENV LOG_DIR=/app/logs
ENV LOG_REQUESTS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
=======
# victim_service/Dockerfile

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py model_backend.py covert_channel.py ./

EXPOSE 8000

ENV MODEL_NAME=fake
ENV COVERT_ENABLED=0
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
