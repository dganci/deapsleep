# Usa una base leggera di Python
FROM python:3.13-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file di requirements
COPY requirements.txt .

# Copia il pacchetto deapsleep e gli script GUI
COPY deapsleep/ ./deapsleep
COPY app/*.py ./  

# Installa dipendenze
RUN apt-get update && \
    apt-get install -y python3-tk && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Imposta il comando di default
CMD ["python", "start.py"]
