#!/usr/bin/env sh
set -e

echo "[entrypoint] pulling data via DVC..."
dvc pull

echo "[entrypoint] starting monitor.py..."
exec python monitor.py

