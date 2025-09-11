# Base imaged
FROM apache/airflow:2.8.0-python3.11

USER root

# System packages & Docker support 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      docker.io \
      git \
      curl \
      ca-certificates \
      gnupg \
      lsb-release \
      sudo \
    && rm -rf /var/lib/apt/lists/*

# Docker Compose V2 
RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

# Fix Docker permissions
RUN groupadd -f -g 999 docker \
    && usermod -aG docker airflow \
    && echo "airflow ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/local/bin/docker-compose" >> /etc/sudoers \
    && touch /var/run/docker.sock \
    && chmod 666 /var/run/docker.sock

# Git safe.directory 
RUN git config --system --add safe.directory '*'

# Prepare project directories
RUN mkdir -p /opt/airflow/shared_volume /opt/airflow/project \
    && chown -R airflow:root /opt/airflow/shared_volume /opt/airflow/project \
    && chmod -R 775 /opt/airflow/shared_volume /opt/airflow/project

USER airflow

# Python dependencies 
# Install Docker provider + helper libs + DVC
RUN pip install --no-cache-dir \
      apache-airflow-providers-docker==3.8.0 \
      docker \
      python-dotenv \
      dvc[s3]

# Set PYTHONPATH 
ENV PYTHONPATH=/opt/airflow/project:$PYTHONPATH

# DVC-Cache as Docker-Volume 
VOLUME ["/opt/airflow/project/.dvc/cache"]