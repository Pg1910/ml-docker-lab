#!/bin/sh
set -eu

# Ensure mount points exist (parent /mlflow already exists from image build)
mkdir -p /mlflow/backend /mlflow/artifacts

# If running as root, fix ownership of mounted volumes, then drop privileges
if [ "$(id -u)" -eq 0 ]; then
  chown -R 10001:10001 /mlflow || true
  exec su -s /bin/sh -c "mlflow server $*" mlflowuser
else
  exec mlflow server "$@"
fi
