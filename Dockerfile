FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

COPY deapsleep/ ./deapsleep
COPY app/*.py ./  

RUN apt-get update && \
    apt-get install -y python3-tk && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "start.py"]

