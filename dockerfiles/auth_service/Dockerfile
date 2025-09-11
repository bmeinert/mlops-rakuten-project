# Use a slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Git for DVC)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# copy requirements.txt
COPY dockerfiles/auth_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8001

# Run the FastAPI app
CMD ["uvicorn", "plugins.cd4ml.inference.auth_service:auth_app", "--host", "0.0.0.0", "--port", "8001"]

# Health-Check for Auth-Service
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl --silent --fail http://localhost:8001/health || exit 1