FROM python:3.11-slim

WORKDIR /app

# Install system dependencies, including Git and gosu (for permission control)
RUN apt-get update \
  && apt-get install -y --no-install-recommends git gosu \
  && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory '*'

COPY dockerfiles/model_training/requirements-model.txt ./requirements-model.txt
RUN pip install --no-cache-dir -r requirements-model.txt
RUN pip install --no-cache-dir "dvc[s3]"

COPY dockerfiles/model_training/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY . .

ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "plugins/cd4ml/model_training/run_model_training.py"]
