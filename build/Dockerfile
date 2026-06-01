FROM python:3.13-slim

COPY ./basic_auth.ini .

RUN pip install mlflow[auth] psycopg2-binary boto3 bulkboto3

EXPOSE 5000