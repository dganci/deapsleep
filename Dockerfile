FROM python:3.13-slim

WORKDIR /deapsleep

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        tk-dev \
        git \
        wget \
        unzip \
        meson \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/deapsleep

CMD ["python", "app/start.py"]
