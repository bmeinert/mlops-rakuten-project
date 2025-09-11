# Use a slim base image
FROM python:3.11-slim

# Set working directory for the build context initially
WORKDIR /app

# Install system dependencies, including Git and gosu
RUN apt-get update \
  && apt-get install -y --no-install-recommends git gosu \
  && rm -rf /var/lib/apt/lists/*

# Git safe directory configuration
RUN git config --global --add safe.directory /app && \
    git config --global --add safe.directory '*'

# Copy requirements file and install Python dependencies
COPY dockerfiles/model_validation/requirements-model.txt ./requirements-model.txt
RUN pip install --no-cache-dir -r requirements-model.txt
RUN pip install --no-cache-dir "dvc[s3]"

# Copy the entrypoint script and make it executable
COPY dockerfiles/model_validation/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of your application code
COPY . .

# Set PYTHONPATH for possibility to use other scripts
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Set environment variables
ENV MODEL_OUTPUT_DIR=/app/models
ENV DATA_INPUT_DIR=/app/data/processed

# Set the entrypoint to our script, and the default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "plugins/cd4ml/model_validation/run_model_validation.py"]