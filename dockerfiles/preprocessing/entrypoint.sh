#!/bin/sh
set -e

CONTAINER_UID=${AIRFLOW_UID:-50000}
CONTAINER_GID=${AIRFLOW_GID:-0}

if [ -z "$(ls -A /app/shared_volume)" ]; then
    echo "Initializing /app/shared_volume permissions for user ${CONTAINER_UID}:${CONTAINER_GID}"
    mkdir -p /app/shared_volume/data/feedback /app/shared_volume/data/processed /app/shared_volume/data/raw /app/shared_volume/models
    chown -R ${CONTAINER_UID}:${CONTAINER_GID} /app/shared_volume
fi

exec "$@"