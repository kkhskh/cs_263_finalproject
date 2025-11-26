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

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
