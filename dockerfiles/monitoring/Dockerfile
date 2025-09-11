FROM python:3.11-slim

WORKDIR /app

# 1) System-Dependencies for DVC
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2) Install python packages
COPY dockerfiles/monitoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dvc[s3]

# 3) Copy app code
COPY monitor/monitor.py ./monitor.py

# 4) Copy entrypoint and make executable
COPY dockerfiles/monitoring/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# 5) Copy dvc metafiles
COPY .dvc/ .dvc/
COPY *.dvc ./

# 6) Starting command
ENTRYPOINT ["./entrypoint.sh"]