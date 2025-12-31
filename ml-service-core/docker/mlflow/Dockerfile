FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.18.0

WORKDIR /app
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
