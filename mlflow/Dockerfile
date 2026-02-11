FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow==2.14.3

# Create a dedicated user
RUN useradd -m -u 10001 mlflowuser

# Pre-create parent dir so non-root can work even before volumes mount
RUN mkdir -p /mlflow/backend /mlflow/artifacts && chown -R 10001:10001 /mlflow

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
