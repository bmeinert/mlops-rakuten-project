FROM python:3.11-slim

WORKDIR /workspace

# install git and curl
RUN apt-get update && apt-get install -y git curl gosu && rm -rf /var/lib/apt/lists/*

# Git safe directory configuration 
RUN git config --global --add safe.directory /app
RUN git config --global --add safe.directory '*'

# upgrade pip
RUN pip install --upgrade pip

# install dvc with s3 support
RUN pip install dvc[s3]

# Set working directory to /app
WORKDIR /app

COPY ./ /app/
COPY dockerfiles/dvc/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script - user permissions
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# pull the data from s3
CMD ["dvc", "pull"]